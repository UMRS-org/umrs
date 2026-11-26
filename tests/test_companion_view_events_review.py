"""Adversarial review of lane UX-V2 (view-op rail, ring, SSE stream).

Each test here failed on the lane's first commit (449ef1fd):

* a ``Last-Event-ID`` of non-ASCII digits or over 4300 digits crashed the
  route (``int()`` ValueError, 500);
* the emitter's ``session.snapshot`` was kept aside (``pinned``) and never
  sent to a live or resuming reader, so the client saw a seq gap and went
  stale until another snapshot;
* the epoch and op-name checks used ``$``, which matches before a trailing
  newline, so an epoch ``"x\\n"`` split the SSE ``id:`` line;
* ``seq``/``at`` beyond 2**53 - 1 went in, which the client drops
  (``Number.isSafeInteger``) while the ring head jumped past every later op;
* a ``Last-Event-ID`` the ring cannot resume (malformed, or the journal
  snapshot's head-0 id) started without ``event: reset``, so a client whose
  ``lastSeq`` was not below the new snapshot's seq ignored it (replay rule);
* the ring had no byte budget across sessions (512 x 4 MiB);
* one user could hold any number of open streams.
"""

from __future__ import annotations

import asyncio
import gc
import json

import pytest
from test_companion_venue_routes import (  # noqa: F401 — fixtures used by name
    H,
    _site_declares_rikyu,
    harness,
    rail,
)
from test_companion_view_events import BASE, SID, frames, op, post, read, seqs, snap
from umrs_r4.api import companion_view_events as events
from umrs_r4.api._view_fanout import ViewHub, apply_payload
from umrs_r4.api._view_ring import ViewRing

SAFE_MAX = 2**53 - 1
BOB = {"x-user-id": "bob"}


@pytest.fixture
def view(rail):  # noqa: F811 — the imported fixture, by name
    rail.client.app.include_router(events.router)
    rail.hub = ViewHub(ring=ViewRing(max_ops=4))
    rail.client.app.state.companion_view_hub = rail.hub
    return rail


# ── Last-Event-ID manipulation ─────────────────────────────────────────────


HOSTILE_IDS = {
    "superscript-two": "%C2%B2",
    "arabic-indic-three": "%D9%A3",
    "5000-digits": "9" * 5000,
    "epoch-and-5000-digits": "run-1:" + "9" * 5000,
    "negative": "-1",
    "negative-in-epoch": "run-1:-3",
    "colons": "::",
    "nul": "%00",
    "line-break": "run-1%0A:1",
}


@pytest.mark.parametrize("raw", list(HOSTILE_IDS.values()), ids=list(HOSTILE_IDS))
def test_hostile_last_event_id_gets_a_snapshot_not_a_500(view, raw):
    post(view, [snap(1), op(2)])
    r = view.client.get(f"{BASE}/events?follow=false&last_event_id={raw}", headers=H)
    assert r.status_code == 200
    fs = frames(r.text)
    assert fs[0]["event"] == "reset"
    assert json.loads(fs[1]["data"])["op"] == "session.snapshot"


def test_huge_last_event_id_in_range_is_not_a_resume(view):
    post(view, [snap(1), op(2)])
    fs = read(view, f"run-1:{SAFE_MAX}")
    assert fs[0]["event"] == "reset" and seqs(fs[1:]) == [1, 2]


def test_another_sessions_last_event_id_does_not_resume(view):
    post(view, [snap(1), op(2), op(3)])
    fs = read(view, "other-run:1")
    assert fs[0]["event"] == "reset" and seqs(fs[1:]) == [1, 2, 3]


# ── the emitter's snapshot reaches every reader ────────────────────────────


def test_a_resuming_reader_gets_the_emitters_new_snapshot(view):
    post(view, [snap(1), op(2)])
    post(view, [snap(3), op(4)])
    assert seqs(read(view, "run-1:2")) == [3, 4]


def test_a_live_reader_gets_the_emitters_new_snapshot():
    hub = ViewHub()
    key = ("alice", SID)
    hub.ring.append(key, "e", [snap(1), op(2)])
    stream = events.EventStream(hub=hub, key=key, load_detail=lambda: None)

    async def scenario() -> list[str]:
        got = await stream.frames(stream.first_plan(None))
        hub.ring.append(key, "e", [snap(3), op(4)])
        return got + await stream.frames(hub.ring.plan(key, stream.gen, stream.seq))

    assert seqs(frames("".join(asyncio.run(scenario())))) == [1, 2, 3, 4]


def test_new_reader_starts_at_the_newest_emitter_snapshot(view):
    post(view, [snap(1), op(2), snap(3), op(4)])
    assert seqs(read(view)) == [3, 4]


# ── envelope checks ────────────────────────────────────────────────────────


@pytest.mark.parametrize("epoch", ["run\n", "run\r", "a\nb"])
def test_epoch_with_a_line_break_is_refused(view, epoch):
    assert post(view, [op(1)], epoch=epoch).status_code in (400, 422)
    assert view.hub.ring.plan(("alice", SID), None, 0).head == 0


def test_op_name_with_a_trailing_newline_is_refused(view):
    assert post(view, [op(1, name="block.patch\n")]).status_code == 400


@pytest.mark.parametrize("field,value", [("seq", SAFE_MAX + 1), ("at", 2**63), ("at", -(2**63))])
def test_unsafe_integers_are_refused(view, field, value):
    assert post(view, [{**op(1), field: value}]).status_code == 400
    assert view.hub.ring.plan(("alice", SID), None, 0).head == 0


def test_fanout_refuses_a_line_break_in_the_epoch():
    ring = ViewRing()
    payload = json.dumps({"o": "peer", "u": "alice", "s": SID, "e": "x\nid: 9", "ops": [op(1)]})
    assert not apply_payload(ring, "here", payload)
    assert ring.plan(("alice", SID), None, 0).head == 0


def test_fanout_skips_an_op_without_a_safe_seq():
    ring = ViewRing()
    ops = [op(1), {**op(2), "seq": "2"}, {**op(2), "seq": SAFE_MAX + 1}]
    payload = json.dumps({"o": "peer", "u": "alice", "s": SID, "e": "", "ops": ops})
    assert apply_payload(ring, "here", payload)
    assert ring.plan(("alice", SID), None, 0).head == 1


# ── the journal snapshot of an empty ring is not a resume point ───────────


def test_journal_snapshot_of_an_empty_ring_does_not_resume(view):
    fs = read(view)
    first_id = fs[0]["id"]
    post(view, [snap(1), op(2)], epoch="")
    again = read(view, first_id)
    assert again[0]["event"] == "reset" and seqs(again[1:]) == [1, 2]


# ── memory bounds ──────────────────────────────────────────────────────────


def test_ring_keeps_a_byte_budget_across_sessions():
    ring = ViewRing(max_ops=1000, max_bytes=10_000, max_sessions=100, max_total_bytes=25_000)
    for n in range(10):
        ring.append(("a", f"s{n}"), "", [op(k, sid=f"s{n}", body="x" * 900) for k in range(1, 10)])
    assert ring.total_bytes() <= 25_000
    assert ring.plan(("a", "s9"), None, 0).head == 9  # the newest stays
    assert ring.plan(("a", "s0"), None, 0).head == 0  # the oldest went out


def test_emitter_snapshot_counts_in_the_session_budget():
    ring = ViewRing(max_ops=1000, max_bytes=2_000)
    ring.append(("a", SID), "", [snap(1)] + [op(n, body="y" * 500) for n in range(2, 10)])
    assert ring.total_bytes() <= 2_000 + len(json.dumps(snap(1)))


# ── many connections per user ──────────────────────────────────────────────


def test_open_streams_per_user_are_capped(view, monkeypatch):
    monkeypatch.setattr(events, "MAX_STREAMS_PER_USER", 2)
    post(view, [snap(1)])
    hub = view.hub
    held = [hub.open_stream("alice", 2), hub.open_stream("alice", 2)]
    assert all(held)
    r = view.client.get(f"{BASE}/events?follow=false", headers=H)
    assert r.status_code == 429
    assert "Traceback" not in r.text and "psycopg" not in r.text
    assert view.client.get(f"{BASE}/events?follow=false", headers=BOB).status_code == 404
    held.pop().release()
    assert view.client.get(f"{BASE}/events?follow=false", headers=H).status_code == 200


def test_a_finished_stream_gives_its_slot_back(view):
    post(view, [snap(1)])
    for _ in range(events.MAX_STREAMS_PER_USER + 2):
        assert view.client.get(f"{BASE}/events?follow=false", headers=H).status_code == 200
    assert view.hub.open_streams("alice") == 0


def test_a_stream_that_never_started_gives_its_slot_back():
    hub = ViewHub()
    stream = events.EventStream(hub=hub, key=("alice", SID), load_detail=lambda: None)
    body = events.event_stream(stream, None, _never, hub.open_stream("alice", 1))
    assert hub.open_streams("alice") == 1 and hub.open_stream("alice", 1) is None
    del body  # the client went away before the first frame: never iterated
    gc.collect()
    assert hub.open_streams("alice") == 0


# ── process restart ─────────────────────────────────────────────────────────


def test_a_restarted_process_answers_a_resume_with_reset_and_journal(view):
    post(view, [snap(1), op(2), op(3)])
    view.client.app.state.companion_view_hub = ViewHub()  # the ring is gone
    fs = read(view, "run-1:3")
    assert fs[0]["event"] == "reset"
    snapshot = json.loads(fs[1]["data"])
    assert snapshot["op"] == "session.snapshot" and snapshot["session"]["detail"] == "limited"
    assert fs[1]["id"].endswith(":-") and len(fs) == 2


async def _never() -> bool:
    return False
