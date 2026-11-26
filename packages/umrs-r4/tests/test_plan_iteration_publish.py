"""D-32: stop / ask correctness of the server-hosted plan loop's round, on
the venue fake (``FakeHost``) of ``test_plan_loop_server``.

The round's plan run is a ``cost`` action: an ask-mode loop and a profile
naming ``cost`` ask before anything starts; a profile naming another class
runs it unasked. A spent ADR-067 wall clock starts no run, including a
clock spent while the question waited on a human (the run gets what is left
after the answer, never the figure read before the ask). A run that never
starts never publishes: nothing is proposed. No live venue, no model.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from test_plan_loop_server import (  # noqa: F401 — ``inbox`` is an autouse fixture
    PICKED,
    FakeHost,
    _decisions,
    _drive,
    _loop,
    inbox,
)
from umrs.knowledge import tool_approvals
from umrs.knowledge.loop import LoopStatus
from umrs_r4.knowledge import plan_iteration

WALL_CLOCK = "budget_exhausted:wall_clock"


def _hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _answer_plan_run(
    decision: str, stop: threading.Event, decided: list[str], before: Any = None
) -> None:
    """A human answering this test's plan_run intent (and only that one);
    ``before()`` runs just before the answer (time passing while asked)."""
    while not stop.is_set():
        mine = [
            p
            for p in tool_approvals.pending(user_id="alice")
            if p["action"].get("kind") == plan_iteration.PLAN_RUN
        ]
        if mine:
            if before is not None:
                before()
            tool_approvals.decide(mine[0]["id"], decision, decided_by="alice")
            decided.append(str(mine[0]["action"].get("remaining_wallclock_sec")))
            return
        time.sleep(0.02)


def _drive_answered(loop, host: FakeHost, decision: str, before: Any = None) -> list[str]:
    decided: list[str] = []
    stop = threading.Event()
    thread = threading.Thread(
        target=_answer_plan_run, args=(decision, stop, decided, before), daemon=True
    )
    thread.start()
    try:
        _drive(loop, host)
    finally:
        stop.set()
        thread.join(timeout=5)
    return decided


# ── ask: the plan run is a cost action ─────────────────────────────────────


def test_ask_a_profile_naming_cost_asks_before_the_run_starts():
    host = FakeHost([{"review": PICKED}])
    loop = _loop(mode="profile", max_iterations=3, pause_profile={"triggers": ["cost"]})
    decided = _drive_answered(loop, host, "once")
    last = loop.iterations[-1]
    asked = [it.status for it in loop.iterations[:-1]]
    assert len(decided) == 1 and len(host.starts) == 1
    assert set(asked) <= {"awaiting_decision"}  # a deferred round started nothing
    assert last.results["loop_decision"]["decision"] == "publish"


def test_ask_a_profile_naming_another_class_runs_unasked():
    host = FakeHost([{"review": PICKED}])
    loop = _loop(mode="profile", max_iterations=1, pause_profile={"triggers": ["physical_action"]})
    _drive(loop, host)
    assert len(host.starts) == 1 and _decisions(loop) == ["publish"]
    assert tool_approvals.pending(user_id="alice") == []


def test_ask_a_human_no_publishes_nothing():
    host = FakeHost([{"review": PICKED}])
    loop = _loop(mode="ask", max_iterations=2)
    _drive_answered(loop, host, "no")
    assert (host.starts, host.proposals) == ([], [])
    assert "finding" not in loop.iterations[0].results
    assert loop.metadata["state_reason"] == "approval_rejected:plan_run"


# ── stop: a spent wall clock starts no run ──────────────────────────────────


def test_stop_a_spent_wall_clock_starts_no_run_and_publishes_nothing():
    host = FakeHost([{"review": PICKED}])
    loop = _loop(max_iterations=2, max_wall_clock_hours=1.0)
    loop.started_at = _hours_ago(2.0)
    _drive(loop, host)
    assert (host.starts, host.proposals) == ([], [])
    assert loop.status == LoopStatus.STOPPED
    assert WALL_CLOCK in str(loop.metadata.get("state_reason"))


def test_stop_the_clock_runs_while_a_human_decides():
    """Fail-before: the round got the wall clock read before the ask, so a
    run started after the loop's cap was spent waiting on the human."""
    host = FakeHost([{"review": PICKED}])
    loop = _loop(mode="ask", max_iterations=1, max_wall_clock_hours=1.0)

    def _an_hour_and_more_passes() -> None:
        loop.started_at = _hours_ago(1.5)

    decided = _drive_answered(loop, host, "once", before=_an_hour_and_more_passes)
    assert float(decided[0]) > 3000  # asked with most of the hour left
    assert (host.starts, host.proposals) == ([], [])
    assert _decisions(loop) == ["stop"]
    assert loop.metadata["state_reason"] == WALL_CLOCK


def test_stop_the_run_gets_the_clock_left_after_the_answer():
    host = FakeHost([{"review": PICKED}])
    loop = _loop(mode="ask", max_iterations=1, max_wall_clock_hours=1.0)

    def _half_an_hour_passes() -> None:
        loop.started_at = _hours_ago(0.5)

    _drive_answered(loop, host, "once", before=_half_an_hour_passes)
    (start,) = host.starts
    assert 1700 < start["caps"]["max_wall_clock_sec"] <= 1800
    assert _decisions(loop) == ["publish"]
