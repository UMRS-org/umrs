"""Console view events of a remote session (UI plan §3.4, lane UX-V2).

The rail verb ``POST …/{sid}/rail/view-ops`` (lease token), the bounded
in-process ring, and the owner's ``GET …/{sid}/events`` SSE stream with
``Last-Event-ID`` resume and a snapshot when the reader is past the ring.
Neither route existed before this lane (404). The rail harness is the venue
route suite's (real lease token from a started venue, fake podman and pool).
The ``_pg`` test runs the ``NOTIFY`` fan-out between two hubs on a
CI-shaped Postgres (env-gated like ``test_companion_snapshot_lww_pg``).
"""

from __future__ import annotations

import asyncio
import json
import threading

import pytest
from test_companion_snapshot_lww_pg import _db_url
from test_companion_venue_routes import (  # noqa: F401 — fixtures used by name
    H,
    _site_declares_rikyu,
    harness,
    rail,
)
from umrs_r4.api import companion_view_events as events
from umrs_r4.api._view_fanout import MAX_PAYLOAD, ViewHub, apply_payload, chunk_payloads
from umrs_r4.api._view_ring import ViewRing
from umrs_r4.api._view_snapshot import snapshot_op

SID = "sess-1"
BASE = f"/umrs/companion/sessions/{SID}"


def op(seq: int, name: str = "block.patch", sid: str = SID, **extra) -> dict:
    return {"v": 1, "sid": sid, "seq": seq, "at": 1_795_000_000_000 + seq, "op": name, **extra}


def snap(seq: int) -> dict:
    session = {"id": SID, "project": "", "title": "emitted", "venue": "frontend",
               "model": "m", "mode": "chat", "state": "idle", "ctxPct": 0,
               "detail": "full"}  # fmt: skip
    return op(seq, "session.snapshot", session=session, turns=[], tasks=[], asks=[], artifacts=[])


def frames(text: str) -> list[dict]:
    """SSE text → frames ``{"id"?, "event"?, "data"?}``; comments dropped."""
    out = []
    for block in text.split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if line and not line.startswith(":")
        )
        if "data" in fields or "event" in fields:
            out.append(fields)
    return out


def seqs(fs: list[dict]) -> list[int]:
    return [json.loads(f["data"])["seq"] for f in fs if "event" not in f]


@pytest.fixture
def view(rail):  # noqa: F811 — the imported fixture, by name
    rail.client.app.include_router(events.router)
    rail.hub = ViewHub(ring=ViewRing(max_ops=4))
    rail.client.app.state.companion_view_hub = rail.hub
    return rail


def post(h, ops: list[dict], epoch: str = "run-1"):
    return h.client.post(
        f"{BASE}/rail/view-ops", json={"epoch": epoch, "ops": ops}, headers=h.rail_headers
    )


def read(h, last_id: str | None = None, headers=H) -> list[dict]:
    extra = {"last-event-id": last_id} if last_id is not None else {}
    r = h.client.get(f"{BASE}/events?follow=false", headers={**headers, **extra})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    return frames(r.text)


# ── the rail verb ──────────────────────────────────────────────────────────


class TestRailVerb:
    def test_accepts_a_batch_and_ignores_replays(self, view):
        r = post(view, [snap(1), op(2), op(3)])
        assert r.status_code == 200 and r.json() == {"accepted": 3, "ignored": 0, "reset": False}
        assert post(view, [op(3), op(4)]).json() == {"accepted": 1, "ignored": 1, "reset": False}

    def test_needs_the_lease_token(self, view):
        r = view.client.post(f"{BASE}/rail/view-ops", json={"ops": [op(1)]}, headers=H)
        assert r.status_code == 401

    def test_refuses_another_sessions_ops(self, view):
        assert post(view, [op(1, sid="other")]).status_code == 400

    @pytest.mark.parametrize(
        "bad",
        [
            {**op(1), "v": 2},
            {**op(1), "seq": 0},
            {**op(1), "seq": True},
            {**op(1), "at": "now"},
            {**op(1), "op": ""},
            {k: v for k, v in op(1).items() if k != "op"},
        ],
    )
    def test_refuses_a_bad_envelope(self, view, bad):
        assert post(view, [bad]).status_code == 400
        assert view.hub.ring.plan(("alice", SID), 0, 0).head == 0

    def test_refuses_a_bad_epoch_and_an_empty_batch(self, view):
        assert post(view, [op(1)], epoch="no spaces").status_code == 400
        assert post(view, []).status_code == 422


# ── the owner stream ───────────────────────────────────────────────────────


class TestStream:
    def test_new_reader_gets_the_emitter_snapshot_then_ops(self, view):
        post(view, [snap(1), op(2), op(3)])
        fs = read(view)
        assert seqs(fs) == [1, 2, 3]
        assert json.loads(fs[0]["data"])["session"]["title"] == "emitted"
        assert [f["id"] for f in fs] == ["run-1:1", "run-1:2", "run-1:3"]

    def test_reconnect_inside_the_ring_gets_only_what_it_missed(self, view):
        post(view, [snap(1), op(2), op(3), op(4)])
        fs = read(view, "run-1:2")
        assert seqs(fs) == [3, 4] and not [f for f in fs if f.get("event") == "reset"]

    def test_bare_seq_resumes_in_the_current_epoch(self, view):
        post(view, [snap(1), op(2), op(3)])
        assert seqs(read(view, "2")) == [3]

    def test_reconnect_past_the_ring_gets_reset_and_a_journal_snapshot(self, view):
        post(view, [snap(1)] + [op(n) for n in range(2, 10)])  # max_ops=4: 6..9 held
        fs = read(view, "run-1:2")
        assert fs[0]["event"] == "reset"
        first = json.loads(fs[1]["data"])
        assert first["op"] == "session.snapshot" and first["seq"] == 9
        assert first["session"]["detail"] == "full" and first["session"]["venue"] == "frontend"
        assert seqs(fs[2:]) == []

    def test_journal_snapshot_carries_the_rows_turns(self, view):
        messages = [{"role": "user", "content": "hi", "metadata": {"turn_id": "t1"}},
                    {"role": "assistant", "content": "hello"}]  # fmt: skip
        view.pool.seed("alice", SID, messages=messages)
        fs = read(view)
        snapshot = json.loads(fs[0]["data"])
        assert snapshot["session"]["detail"] == "limited"
        assert [t["id"] for t in snapshot["turns"]] == ["t1"]
        assert snapshot["turns"][0]["blocks"][0]["body"] == "hello"

    def test_new_epoch_starts_the_ring_over(self, view):
        post(view, [snap(1), op(2), op(3)])
        assert post(view, [snap(1), op(2)], epoch="run-2").json()["reset"] is True
        fs = read(view, "run-1:3")
        assert fs[0]["event"] == "reset"
        assert [f["id"] for f in fs[1:]] == ["run-2:1", "run-2:2"]

    def test_another_users_session_is_404(self, view):
        post(view, [snap(1)])
        r = view.client.get(f"{BASE}/events?follow=false", headers={"x-user-id": "bob"})
        assert r.status_code == 404

    def test_needs_identity(self, view):
        assert view.client.get(f"{BASE}/events?follow=false").status_code == 401

    def test_ring_is_keyed_by_owner(self, view):
        post(view, [snap(1), op(2)])
        assert view.hub.ring.plan(("bob", SID), None, 0).head == 0


# ── live follow (the generator, not the transport) ─────────────────────────


async def _never() -> bool:
    return False


def test_live_stream_wakes_on_a_post_from_another_thread(monkeypatch):
    monkeypatch.setattr(events, "HEARTBEAT_SECONDS", 5.0)
    hub = ViewHub()
    key = ("alice", SID)
    hub.ring.append(key, "e", [snap(1)])
    stream = events.EventStream(hub=hub, key=key, load_detail=lambda: None)

    async def scenario() -> list[str]:
        gen = events.event_stream(stream, None, _never)
        got = [await gen.__anext__(), await gen.__anext__()]  # retry, snapshot
        threading.Timer(0.05, hub.ring.append, (key, "e", [op(2)])).start()
        got.append(await asyncio.wait_for(gen.__anext__(), 2.0))
        await gen.aclose()
        return got

    got = asyncio.run(scenario())
    assert got[0].startswith("retry:") and seqs(frames("".join(got))) == [1, 2]


def test_idle_stream_sends_a_heartbeat_comment(monkeypatch):
    monkeypatch.setattr(events, "HEARTBEAT_SECONDS", 0.05)
    hub = ViewHub()
    hub.ring.append(("alice", SID), "e", [snap(1)])
    stream = events.EventStream(hub=hub, key=("alice", SID), load_detail=lambda: None)

    async def scenario() -> str:
        gen = events.event_stream(stream, None, _never)
        for _ in range(2):
            await gen.__anext__()
        frame = await asyncio.wait_for(gen.__anext__(), 2.0)
        await gen.aclose()
        return frame

    assert asyncio.run(scenario()).startswith(":")


def test_stream_ends_when_the_row_is_gone():
    stream = events.EventStream(hub=ViewHub(), key=("alice", SID), load_detail=lambda: None)

    async def scenario() -> list[str]:
        return [frame async for frame in events.event_stream(stream, None, _never)]

    assert frames("".join(asyncio.run(scenario()))) == [
        {"event": "end", "data": '{"reason":"not_found"}'}
    ]


# ── the ring, the snapshot, the fan-out payloads ───────────────────────────


def test_ring_bounds_bytes_and_sessions():
    ring = ViewRing(max_ops=100, max_bytes=400, max_sessions=2)
    ring.append(("a", "s1"), "", [op(n, sid="s1") for n in range(1, 10)])
    plan = ring.plan(("a", "s1"), None, 0)
    assert plan.resync and plan.head == 9
    assert ring.resume_plan(("a", "s1"), "", 1).resync  # the oldest went out
    ring.append(("a", "s2"), "", [op(1, sid="s2")])
    ring.append(("a", "s3"), "", [op(1, sid="s3")])
    assert ring.plan(("a", "s1"), None, 0).head == 0  # least recently written out


def test_a_hole_in_seq_drops_what_came_before_it():
    ring = ViewRing()
    ring.append(("a", SID), "", [snap(1), op(2), op(5)])
    assert ring.resume_plan(("a", SID), "", 2).resync
    assert not ring.resume_plan(("a", SID), "", 4).resync


def test_journal_snapshot_names_only_contract_keys():
    detail = {"session_id": SID, "name": "n", "venue": "frontend", "state": "error",
              "payload": json.dumps({"model": "m", "metadata": json.dumps(
                  {"interaction_mode": "nonsense", "project_id": "p"}),
                  "messages": json.dumps([{"role": "assistant", "content": "a\nb"}])})}  # fmt: skip
    snapshot = snapshot_op(detail, seq=0, at=5, native=False)
    assert snapshot["seq"] == 1 and set(snapshot) == {
        "v", "sid", "seq", "at", "op", "session", "turns", "tasks", "asks", "artifacts"}  # fmt: skip
    session = snapshot["session"]
    assert (session["mode"], session["state"], session["project"]) == ("chat", "error", "p")
    turn = snapshot["turns"][0]
    assert turn["user"] == "" and turn["blocks"][0]["summary"] == "a"


def test_fanout_chunks_stay_under_the_notify_limit():
    head = {"o": "x", "u": "alice", "s": SID, "e": ""}
    ops = [op(n, body="y" * 3000) for n in range(1, 6)] + [op(6, body="z" * 9000)]
    payloads = chunk_payloads(head, ops)
    assert all(len(p) < MAX_PAYLOAD for p in payloads)
    ring = ViewRing()
    for payload in payloads:
        assert apply_payload(ring, "another-process", payload)
    plan = ring.plan(("alice", SID), None, 0)
    assert plan.head == 6 and plan.ops == ()  # 6 went out as a gap marker
    assert not apply_payload(ring, "x", payloads[0])  # our own notification


@pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")
def test_pg_notify_fans_a_batch_out_to_another_process():
    from psycopg2.pool import ThreadedConnectionPool

    pool = ThreadedConnectionPool(1, 2, _db_url())
    here, there = ViewHub(pool=pool), ViewHub(pool=pool)
    try:
        there.ensure_listener()
        assert there.wait_listening(5.0)
        here.post("alice", SID, "run-1", [snap(1), op(2)])
        deadline = 50
        while there.ring.plan(("alice", SID), None, 0).head < 2 and deadline:
            threading.Event().wait(0.1)
            deadline -= 1
        plan = there.ring.plan(("alice", SID), None, 0)
        assert plan.head == 2 and plan.pinned is not None and plan.epoch == "run-1"
    finally:
        there.stop()
        pool.closeall()
