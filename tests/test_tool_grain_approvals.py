"""Tool-grain approval gate pins (ADR-028).

- Ask-mode engine gate: requests one intent, parks, proceeds on approve.
- Rejection flips the gate to False (engine raises upstream).
- Registry terminal-decision rules (409-on-redecision, 404-on-unknown).
- `always` unblocks + (route-side) flips loop metadata to auto.
"""

from __future__ import annotations

import threading
import time

import pytest
from umrs.knowledge import tool_approvals
from umrs.knowledge.engine_loop_runner import EngineLoopRunner
from umrs.knowledge.loop import LoopPhase, LoopStatus, ResearchLoop


@pytest.fixture(autouse=True)
def _clean_approvals():
    tool_approvals.clear_all()
    yield
    tool_approvals.clear_all()


def _loop(mode: str) -> ResearchLoop:
    return ResearchLoop(
        loop_id="loop-gate-1",
        project_id="proj-a",
        research_goal="goal",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.EXPERIMENT_EXECUTION,
        max_iterations=3,
        current_iteration=1,
        metadata={"mode": mode},
    )


class _MemStore:
    def __init__(self, loop: ResearchLoop) -> None:
        self._loop = loop
        self.saves = 0

    def get(self, loop_id: str):  # noqa: ARG002
        return self._loop

    def save(self, loop: ResearchLoop) -> None:  # noqa: ARG002
        self.saves += 1


def _gate_for(loop: ResearchLoop, store: _MemStore):
    runner = EngineLoopRunner(loop, engine_factory=lambda: None, store=store, poll_seconds=0.01)
    return runner._make_approval_gate(loop)  # noqa: SLF001


ACTION = {
    "kind": "workflow_trigger",
    "workflow_name": "wf-x",
    "iteration": 1,
    "params": {},
}


def test_auto_mode_never_gates():
    loop = _loop("auto")
    gate = _gate_for(loop, _MemStore(loop))
    assert gate(ACTION) is True
    assert tool_approvals.pending() == []


def test_ask_mode_parks_then_approves():
    loop = _loop("ask")
    store = _MemStore(loop)
    gate = _gate_for(loop, store)
    outcome: list[bool] = []

    thread = threading.Thread(target=lambda: outcome.append(gate(ACTION)), daemon=True)
    thread.start()
    # The gate parks immediately after registering the intent.
    time.sleep(0.05)
    pending = tool_approvals.pending()
    assert len(pending) == 1
    assert loop.metadata.get("awaiting_approval") == pending[0]["id"]

    tool_approvals.decide(pending[0]["id"], "once", "operator")
    thread.join(timeout=2.0)
    assert outcome == [True]
    assert "awaiting_approval" not in loop.metadata
    assert store.saves >= 2


def test_rejection_returns_false():
    loop = _loop("ask")
    gate = _gate_for(loop, _MemStore(loop))
    outcome: list[bool] = []

    thread = threading.Thread(target=lambda: outcome.append(gate(ACTION)), daemon=True)
    thread.start()
    time.sleep(0.05)
    record = tool_approvals.pending()[0]
    tool_approvals.decide(record["id"], "no", "operator")
    thread.join(timeout=2.0)
    assert outcome == [False]


def test_regregistry_terminal_rules():
    record = tool_approvals.request_approval("loop-x", 1, ACTION)
    tool_approvals.decide(record["id"], "once", "op")
    with pytest.raises(ValueError, match="already decided"):
        tool_approvals.decide(record["id"], "once", "op")
    with pytest.raises(KeyError):
        tool_approvals.decide("appr-nonexistent", "once", "op")
    with pytest.raises(ValueError, match="decision"):
        tool_approvals.decide(record["id"], "bogus", "op")


def test_intent_recreation_is_idempotent_under_same_id():
    first = tool_approvals.request_approval("loop-y", 2, ACTION, approval_id="appr-fixed")
    second = tool_approvals.request_approval("loop-y", 2, ACTION, approval_id="appr-fixed")
    assert first["id"] == second["id"]
    assert len(tool_approvals.pending()) == 1
