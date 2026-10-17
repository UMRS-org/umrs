"""SSE endpoint for real-time KnowledgeEvent broadcast.

No module-level imports from ``umrs.api`` or ``umrs.server`` — only stdlib,
events/adapter, and knowledge modules. The caller-scope resolver is a lazy
composition import inside ``_caller_scope`` (ADR-024 tolerates it).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from umrs.jobs.knowledge_adapters import KnowledgeEvent
from umrs.knowledge.access_scope import CallerScope, project_readable

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

sse_timeout_sec: int = int(os.environ.get("SSE_TIMEOUT_SEC", "300"))
sse_max_clients: int = int(os.environ.get("SSE_MAX_CLIENTS", "50"))

# ── Client registry ──────────────────────────────────────────────────────────

_client_counter: int = 0
_client_lock = asyncio.Lock()

# project_id -> list of client_ids
_projects: dict[str, list[str]] = defaultdict(list)

# client_id -> {"project": str, "queue": asyncio.Queue, "connected_at": float}
_clients: dict[str, dict[str, Any]] = {}


def _evict_dead() -> None:
    """Remove disconnected clients (not yet tracked by disconnect hook —
    safety net for ungraceful disconnects)."""
    now = time.monotonic()
    dead: list[str] = []
    for cid, info in _clients.items():
        if now - info["connected_at"] > sse_timeout_sec:
            dead.append(cid)
    for cid in dead:
        _remove_client(cid)
    if dead:
        logger.info("Evicted %d dead SSE clients", len(dead))


def _project_client_count(project_id: str) -> int:
    """Return number of active clients for a project (approx)."""
    return len([c for c in _projects[project_id] if c in _clients])


def _remove_client(client_id: str) -> None:
    """Remove a client from all registries."""
    info = _clients.pop(client_id, None)
    if info is None:
        return
    project_id = info["project"]
    clients = _projects.get(project_id, [])
    if client_id in clients:
        clients.remove(client_id)
    if not clients:
        _projects.pop(project_id, None)


def _is_last_event_id_match(client_id: str, last_event_id: str) -> bool:
    """Check if the client's stored last_event_id matches."""
    info = _clients.get(client_id)
    return info is not None and info.get("last_event_id") == last_event_id


# ── Broadcast engine ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BroadcastEvent:
    """Internal event pushed into client queues."""

    event_type: str
    data: dict[str, Any]
    event_id: str
    project_id: str = ""


_broadcast_queue: asyncio.Queue[BroadcastEvent] = asyncio.Queue(maxsize=5000)
_broadcast_handler: asyncio.Task | None = None

# Project attribution for events emitted deep inside job helpers: backends
# set this contextvar at submit time so every KnowledgeEvent within the
# attempt inherits the correct project (no signature threading).
_project_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "sse_project_ctx", default=""
)


def set_project_ctx(project_id: str) -> contextvars.Token:
    """Bind the current execution context to a project for SSE attribution."""
    return _project_ctx.set(project_id)


def reset_project_ctx(token: contextvars.Token) -> None:
    """Undo set_project_ctx (required in finally blocks)."""
    _project_ctx.reset(token)


def _target_clients(bevent: BroadcastEvent) -> list[tuple[str, dict[str, Any]]]:
    """Clients that may receive this event (project-scoped fan-out).

    Events carry a project; clients only see events for THEIR project
    (cross-project delivery was the latent leak once the producer was
    wired). Unattributed events go to nobody — they are dropped at emit().
    Caller must hold _client_lock.
    """
    return [
        (cid, info)
        for cid, info in _clients.items()
        if info.get("project") == bevent.project_id
    ]


async def _broadcast_loop() -> None:
    """Drain the broadcast queue and fan out to same-project client queues."""
    logger.info("SSE broadcast loop started")
    try:
        while True:
            bevent: BroadcastEvent = await _broadcast_queue.get()
            delivered: int = 0
            dropped: int = 0
            async with _client_lock:
                for cid, info in _target_clients(bevent):
                    try:
                        qsize = info["queue"].qsize()
                        if qsize < 100:  # drop if queue is full
                            await info["queue"].put(bevent)
                            delivered += 1
                        else:
                            dropped += 1
                    except Exception:
                        pass  # client disconnected mid-broadcast
            if bevent.event_type == "error":
                logger.warning(
                    "SSE broadcast: %s delivered=%d dropped=%d",
                    bevent.event_type,
                    delivered,
                    dropped,
                )
    except asyncio.CancelledError:
        logger.info("SSE broadcast loop stopped")


def _start_broadcast_loop() -> None:
    global _broadcast_handler
    if _broadcast_handler is None or _broadcast_handler.done():
        _broadcast_handler = asyncio.create_task(_broadcast_loop())


def emit(event: KnowledgeEvent, project_id: str = "") -> None:
    """Public API: broadcast a KnowledgeEvent to same-project SSE clients.

    Project attribution order: explicit argument > event.project_id (when
    present) > the backend's project contextvar. Unattributed events are
    dropped — fail closed rather than fan out across projects. When nobody
    is listening the queue never fills up.
    """
    project = project_id or getattr(event, "project_id", "") or _project_ctx.get()
    if not project:
        return
    if not _clients:
        return  # no listeners — nothing to do, queue never grows
    data = {**asdict(event), "project_id": project}
    event_id = str(uuid.uuid4())[:8]
    bevent = BroadcastEvent(
        event_type=event.event_type,
        data=data,
        event_id=event_id,
        project_id=project,
    )
    try:
        _broadcast_queue.put_nowait(bevent)
    except asyncio.QueueFull:
        logger.warning("SSE broadcast queue full, dropping event %s", event_id)


# ── SSE endpoint ─────────────────────────────────────────────────────────────

router = APIRouter(
    prefix="/umrs/knowledge",
    tags=["knowledge-sse"],
)


def _caller_scope(request: Request) -> CallerScope:
    """The HMAC-verified caller (401 in strict mode without identity)."""
    from umrs.api._scope_identity import request_caller_scope

    return request_caller_scope(request)


def _authorize_project(request: Request, project_id: str) -> CallerScope:
    """The caller when it may read ``project_id``'s events; 403 otherwise.

    The stream and its health took no identity: any caller subscribed to any
    project's knowledge events (hardening plan H-1 item 5, 2026-10-17).
    """
    caller = _caller_scope(request)
    if not project_readable(project_id, caller):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_denied",
                "message": "Project is outside the caller's access tier.",
            },
        )
    return caller


@router.get("/projects/{project_id}/events")
async def sse_events(
    project_id: str,
    request: Request,
    last_event_id: str | None = Query(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Stream the project's KnowledgeEvents as SSE (callers who may read it)."""
    _authorize_project(request, project_id)
    _evict_dead()

    async with _client_lock:
        count = _project_client_count(project_id)
        if count >= sse_max_clients:
            raise HTTPException(
                status_code=429,
                detail=f"Project {project_id} has reached the maximum "
                f"of {sse_max_clients} SSE connections",
            )

    client_id, info = await _register_client(project_id, last_event_id)

    logger.info(
        "SSE client %s connected to project %s (total: %d)",
        client_id,
        project_id,
        _project_client_count(project_id),
    )

    _start_broadcast_loop()

    return StreamingResponse(
        _event_stream_iter(client_id, info, last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _event_stream_iter(
    client_id: str,
    info: dict[str, Any],
    last_event_id: str | None,
) -> AsyncIterator[str]:
    """Top-level generator for SSE event streaming.

    Kept separate from sse_events() to keep the route function under CCN 10.
    """
    try:
        yield ": heartbeat\n\n"

        if last_event_id:
            async with _client_lock:
                for ev in info["buffer"]:
                    if _is_last_event_id_match(client_id, ev.event_id):
                        yield _format_sse_event(ev.event_id, ev.event_type, ev.data)

        while True:
            try:
                ev = await asyncio.wait_for(info["queue"].get(), timeout=10.0)
                if _is_last_event_id_match(client_id, ev.event_id):
                    continue
                if len(info["buffer"]) >= 10:
                    info["buffer"].pop(0)
                info["buffer"].append(ev)
                yield _format_sse_event(ev.event_id, ev.event_type, ev.data)
            except asyncio.TimeoutError:
                async with _client_lock:
                    if client_id not in _clients:
                        break
                yield ": heartbeat\n\n"
                continue
    except GeneratorExit:
        pass
    finally:
        _remove_client(client_id)


async def _register_client(
    project_id: str, last_event_id: str | None
) -> tuple[str, dict[str, Any]]:
    """Register a new SSE client and return (client_id, info)."""
    global _client_counter
    async with _client_lock:
        _client_counter += 1
        client_id = f"sse-{_client_counter}-{uuid.uuid4().hex[:6]}"

    client_queue: asyncio.Queue[BroadcastEvent] = asyncio.Queue(maxsize=200)
    info: dict[str, Any] = {
        "project": project_id,
        "queue": client_queue,
        "connected_at": time.monotonic(),
        "last_event_id": last_event_id,
        "buffer": [],  # last 10 events for reconnection
    }

    async with _client_lock:
        _clients[client_id] = info
        _projects[project_id].append(client_id)

    return client_id, info


def _format_sse_event(event_id: str, event_type: str, data: dict[str, Any]) -> str:
    """Format a single SSE event."""
    return f"id: {event_id}\nevent: {event_type}\ndata: {json.dumps(data)}\n\n"


# ── Lifecycle management ─────────────────────────────────────────────────────

_sse_started = False


def start_sse() -> None:
    """Start the SSE broadcast loop. Call during UMRS server startup."""
    global _sse_started
    if _sse_started:
        return
    _start_broadcast_loop()
    _sse_started = True
    logger.info(
        "SSE broadcast engine started (timeout=%ds, max_clients=%d)",
        sse_timeout_sec,
        sse_max_clients,
    )


def stop_sse() -> None:
    """Stop the SSE broadcast loop. Call during UMRS server shutdown."""
    global _broadcast_handler, _sse_started
    if _broadcast_handler is not None:
        _broadcast_handler.cancel()
        _sse_started = False
        logger.info("SSE broadcast engine stopped")


# ── Health check ─────────────────────────────────────────────────────────────


@router.get("/projects/{project_id}/events/health")
async def sse_health(project_id: str, request: Request) -> dict[str, Any]:
    """Return SSE client counts for monitoring.

    Platform-wide figures (all projects' clients, the shared broadcast
    queue) are for admins only; other callers get ``None`` there.
    """
    caller = _authorize_project(request, project_id)
    unrestricted = caller.is_admin or caller.anonymous
    _evict_dead()
    async with _client_lock:
        total = len(_clients)
        project_count = _project_client_count(project_id)
    return {
        "total_clients": total if unrestricted else None,
        "project_clients": project_count,
        "max_clients": sse_max_clients,
        "timeout_sec": sse_timeout_sec,
        "broadcast_queue_size": _broadcast_queue.qsize() if unrestricted else None,
    }


# ── Import guard ─────────────────────────────────────────────────────────────
import os
