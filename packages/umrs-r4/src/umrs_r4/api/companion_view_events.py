"""Console view events of a remote session (UI plan §3.4, lane UX-V2, ADR-119).

Worker rail (lease-token principal, like every ``/rail/`` verb):

    POST /umrs/companion/sessions/{sid}/rail/view-ops
        body  {"epoch": "<emitter run id>", "ops": [<view op>, ...]}
        200   {"accepted": n, "ignored": n, "reset": bool}

Owner route (gateway-verified identity; another user's session is 404):

    GET  /umrs/companion/sessions/{sid}/events[?follow=false]
        text/event-stream; resume with ``Last-Event-ID`` (or ``?last_event_id=``)

The venue worker posts the view-contract ops it emits; they go into this
process's :class:`ViewRing` and out to the other UMRS processes on
``NOTIFY`` (``_view_fanout``). The stream sends each op as one SSE message:
``id: <epoch>:<seq>`` and ``data: <op JSON>``. A reconnect whose
``Last-Event-ID`` the ring still covers gets the ops after it; any other
start gets a ``session.snapshot`` first: the emitter's own when the ring
holds it with every op after it, else one built from the journal row
(``detail: limited`` until the session emits natively). When a stream that
already had ops starts over (gap, emitter restart, a resume the ring cannot
serve), an ``event: reset`` comes before the snapshot: the client calls
``expectSnapshot`` (packages/r4-console-core) so the snapshot applies whatever
its seq. A comment line every 15 s keeps proxies from closing an idle stream.
``?follow=false`` sends what is there and ends (polling clients, tests).

Any ``Last-Event-ID`` that is not a resume the ring can serve (malformed,
another epoch, past the ring) still means the client holds a view, so the
snapshot comes after ``event: reset``: without it a client whose last seq is
not below the snapshot's would ignore it (the reducer's replay rule). The
journal snapshot of a session with no ops yet stands at no seq, so its id
(``<epoch>:-``) never resumes. One user holds at most
``MAX_STREAMS_PER_USER`` open streams in a process (429 beyond).

Epoch: the emitter names its run (``epoch``, outside the op so the frozen
contract stays as it is); a batch under another epoch starts the session's
ring over. A bare-integer ``Last-Event-ID`` means the current epoch.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from umrs.api.typed_errors import PlatformError, raise_typed

from umrs_r4.api._rail_identity import RailPrincipal
from umrs_r4.api._view_fanout import StreamSlot, ViewHub
from umrs_r4.api._view_ring import MAX_SAFE_INT, Entry, Plan, safe_seq, valid_epoch
from umrs_r4.api._view_snapshot import snapshot_op
from umrs_r4.api.companion_sessions import _pool, _uid, session_owned, snapshot_detail_row
from umrs_r4.api.companion_venue import rail_principal

router = APIRouter(prefix="/umrs/companion/sessions", tags=["companion-sessions"])

HEARTBEAT_SECONDS = 15.0
MAX_BATCH = 256
MAX_OP_BYTES = 256 * 1024
MAX_STREAMS_PER_USER = 16
_OP_NAME_RE = re.compile(r"[a-z][a-z0-9_.]{0,63}")
_LAST_ID_RE = re.compile(r"(?:([A-Za-z0-9._-]{0,64}):)?([0-9]{1,16})")
RESET_FRAME = 'event: reset\ndata: {"reason":"resync"}\n\n'
_END_FRAME = 'event: end\ndata: {"reason":"not_found"}\n\n'


def get_view_hub(request: Request) -> ViewHub:
    """Process-cached hub (tests inject ``app.state.companion_view_hub``)."""
    hub = getattr(request.app.state, "companion_view_hub", None)
    if hub is None:
        hub = ViewHub(pool=getattr(request.app.state, "umrs_db_pool", None))
        request.app.state.companion_view_hub = hub
    hub.ensure_listener()
    return hub


# ── rail verb ──────────────────────────────────────────────────────────────


class ViewOpsBody(BaseModel):
    """One batch of view ops from the session's emitter."""

    epoch: str = Field(default="", max_length=64)
    ops: list[dict[str, Any]] = Field(min_length=1, max_length=MAX_BATCH)


def _bad(message: str) -> None:
    raise_typed(PlatformError.companionvenue("bad_request", message))


def _is_safe_int(value: Any) -> bool:
    """An integer the client reads exactly (``Number.isSafeInteger``)."""
    return type(value) is int and -MAX_SAFE_INT <= value <= MAX_SAFE_INT


def check_op(op: dict[str, Any], session_id: str) -> None:
    """The envelope of the contract (``{v: 1, sid, seq, at, op}``); the body
    is the emitter's to validate against the schema."""
    if op.get("v") != 1 or op.get("sid") != session_id:
        _bad("view op: v must be 1 and sid this session")
    if not safe_seq(op.get("seq")) or not _is_safe_int(op.get("at")):
        _bad("view op: seq must be a positive safe integer and at a safe integer")
    if not isinstance(op.get("op"), str) or not _OP_NAME_RE.fullmatch(op["op"]):
        _bad("view op: op must name an op")
    if len(json.dumps(op, separators=(",", ":"))) > MAX_OP_BYTES:
        _bad(f"view op {op['seq']}: larger than {MAX_OP_BYTES} bytes")


@router.post("/{session_id}/rail/view-ops", response_model=dict[str, Any])
async def rail_post_view_ops(
    session_id: str,
    body: ViewOpsBody,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """Take one batch of the session's view ops into the ring (and fan it out).
    # auth: lease-token
    """
    if not valid_epoch(body.epoch):
        _bad("epoch: up to 64 of A-Z a-z 0-9 . _ -")
    for op in body.ops:
        check_op(op, session_id)
    hub = get_view_hub(request)
    outcome = await asyncio.to_thread(hub.post, principal.owner, session_id, body.epoch, body.ops)
    return {"accepted": outcome.accepted, "ignored": outcome.ignored, "reset": outcome.reset}


# ── owner stream ───────────────────────────────────────────────────────────


def parse_last_event_id(raw: str | None) -> tuple[str | None, int] | None:
    """``<epoch>:<seq>`` or a bare ``<seq>`` (epoch None); else None.

    ASCII digits only and at most 2**53 - 1: ``int()`` takes other Unicode
    digits and refuses over 4300 of them, and the client's seq is a safe
    integer, so anything else is no resume point (never a 500).
    """
    found = _LAST_ID_RE.fullmatch((raw or "").strip())
    if found is None or int(found.group(2)) > MAX_SAFE_INT:
        return None
    return found.group(1), int(found.group(2))


def op_frame(epoch: str, seq: int, data: str) -> str:
    return f"id: {epoch}:{seq}\ndata: {data}\n\n"


def _now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class EventStream:
    """One reader's cursor over the ring of ``key``."""

    hub: ViewHub
    key: tuple[str, str]
    load_detail: Callable[[], dict[str, Any] | None]
    follow: bool = True
    gen: int = 0
    seq: int = 0
    announced: bool = False
    ended: bool = False

    async def frames(self, plan: Plan) -> list[str]:
        """The SSE frames for ``plan``; moves the cursor past them."""
        frames: list[str] = []
        if plan.resync:
            if self.announced:
                frames.append(RESET_FRAME)
            snapshot = await self._snapshot_frame(plan)
            if snapshot is None:
                self.ended = True
                return [_END_FRAME]
            frames.append(snapshot)
        frames.extend(op_frame(plan.epoch, e.seq, e.data) for e in plan.ops)
        self.gen, self.seq = plan.gen, plan.head
        self.announced = self.announced or bool(frames)
        return frames

    async def _snapshot_frame(self, plan: Plan) -> str | None:
        if plan.pinned is not None:
            return _entry_frame(plan.epoch, plan.pinned)
        detail = await asyncio.to_thread(self.load_detail)
        if detail is None:
            return None
        op = snapshot_op(detail, seq=plan.head, at=_now_ms(), native=plan.native)
        data = json.dumps(op, separators=(",", ":"))
        if plan.head == 0:
            # No op yet: the snapshot's seq 1 is not the emitter's op 1, so
            # this id must never resume (the emitter's op 1 would be lost).
            return f"id: {plan.epoch}:-\ndata: {data}\n\n"
        return op_frame(plan.epoch, op["seq"], data)

    def first_plan(self, last_id: tuple[str | None, int] | None) -> Plan:
        if last_id is None:
            return self.hub.ring.plan(self.key, None, 0)
        epoch, seq = last_id
        return self.hub.ring.resume_plan(self.key, epoch, seq)

    async def next_frames(self) -> list[str]:
        await self.hub.ring.wait(self.key, self.gen, self.seq, HEARTBEAT_SECONDS)
        return await self.frames(self.hub.ring.plan(self.key, self.gen, self.seq))


def _entry_frame(epoch: str, entry: Entry) -> str:
    return op_frame(epoch, entry.seq, entry.data)


async def event_stream(
    stream: EventStream,
    last_id: tuple[str | None, int] | None,
    disconnected: Callable[[], Any],
    slot: StreamSlot | None = None,
) -> AsyncIterator[str]:
    """The SSE body: what the cursor gets now, then (``follow``) what comes."""
    try:
        yield "retry: 3000\n\n"
        for frame in await stream.frames(stream.first_plan(last_id)):
            yield frame
        while stream.follow and not stream.ended:
            if await disconnected():
                return
            frames = await stream.next_frames()
            for frame in frames or [": hb\n\n"]:
                yield frame
    finally:
        if slot is not None:
            slot.release()


def _detail_loader(pool: Any, uid: str, session_id: str) -> Callable[[], dict[str, Any] | None]:
    def load() -> dict[str, Any] | None:
        conn = pool.getconn()
        try:
            return snapshot_detail_row(conn, uid, session_id)
        finally:
            pool.putconn(conn)

    return load


def _require_owned(pool: Any, uid: str, session_id: str) -> None:
    conn = pool.getconn()
    try:
        owned = session_owned(conn, uid, session_id)
    finally:
        pool.putconn(conn)
    if not owned:
        raise_typed(PlatformError.companionsessions("not_found", "snapshot not found"))


@router.get("/{session_id}/events")
async def session_view_events(
    session_id: str,
    request: Request,
    follow: bool = True,
    last_event_id: str = "",
) -> StreamingResponse:
    """The session's view ops as server-sent events (see the module docstring).
    # auth: verified
    """
    uid = _uid(request)
    pool = _pool(request)
    await asyncio.to_thread(_require_owned, pool, uid, session_id)
    raw_last_id = (request.headers.get("last-event-id") or last_event_id).strip()
    hub = get_view_hub(request)
    stream = EventStream(
        hub=hub,
        key=(uid, session_id),
        load_detail=_detail_loader(pool, uid, session_id),
        follow=follow,
        # A client that names any last id holds a view: reset before a snapshot.
        announced=bool(raw_last_id),
    )
    last_id = parse_last_event_id(raw_last_id)
    slot = hub.open_stream(uid, MAX_STREAMS_PER_USER)
    if slot is None:
        raise_typed(
            PlatformError.quota(
                "concurrent_sessions_exceeded",
                f"at most {MAX_STREAMS_PER_USER} open event streams per user",
                limit=MAX_STREAMS_PER_USER,
            )
        )
    return StreamingResponse(
        event_stream(stream, last_id, request.is_disconnected, slot),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
