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


# ── WP-05: durable inbox semantics at the gate (ADR-036 (a)) ─────────────────


@pytest.fixture()
def _durable_inbox(tmp_path):
    from umrs.companion_trust.approval_inbox import ApprovalInboxStore

    path = str(tmp_path / "inbox.sqlite")
    tool_approvals.configure(ApprovalInboxStore(path=path))
    yield path
    tool_approvals.configure(None)


def test_gate_requests_carry_owner_job_and_attempt():
    loop = _loop("ask")
    loop.metadata.update(owner_user_id="alice", durable_job_id="job-7", attempt_id="att-7")
    gate = _gate_for(loop, _MemStore(loop))
    thread = threading.Thread(target=lambda: gate(ACTION), daemon=True)
    thread.start()
    time.sleep(0.05)
    record = tool_approvals.pending(user_id="alice")[0]
    assert (record["user_id"], record["job_id"], record["attempt_id"]) == (
        "alice",
        "job-7",
        "att-7",
    )
    assert tool_approvals.pending(user_id="bob") == []
    tool_approvals.decide(record["id"], "once", "alice")
    thread.join(timeout=2.0)


def test_restart_reads_recorded_decision_without_reasking(_durable_inbox):
    """Loop restart e2e (G-P8-1): the question AND the answer survive the process.

    1. the gate parks (intent row + metadata.awaiting_approval marker)
    2. the process dies mid-park (driver abort here)
    3. another client decides against the durable inbox
    4. the restarted driver re-enters the gate under the marker → continues
       immediately; nothing is re-asked.
    """
    from umrs.companion_trust.approval_inbox import ApprovalInboxStore

    loop = _loop("ask")
    loop.metadata["owner_user_id"] = "alice"
    store = _MemStore(loop)
    runner = EngineLoopRunner(loop, engine_factory=lambda: None, store=store, poll_seconds=0.01)
    gate = runner._make_approval_gate(loop)  # noqa: SLF001
    outcome: list[bool] = []
    thread = threading.Thread(target=lambda: outcome.append(gate(ACTION)), daemon=True)
    thread.start()
    time.sleep(0.05)
    approval_id = loop.metadata["awaiting_approval"]
    assert tool_approvals.get(approval_id)["status"] == "pending"

    runner.abort()  # "process dies" — marker stays on the loop record
    thread.join(timeout=2.0)
    assert outcome == [False]
    assert loop.metadata["awaiting_approval"] == approval_id

    ApprovalInboxStore(path=_durable_inbox).decide(approval_id, "once", "alice")

    tool_approvals.configure(ApprovalInboxStore(path=_durable_inbox))  # "restart"
    restarted = EngineLoopRunner(loop, engine_factory=lambda: None, store=store, poll_seconds=0.01)
    resumed_gate = restarted._make_approval_gate(loop)  # noqa: SLF001
    assert resumed_gate(ACTION) is True  # no park, no thread needed
    assert "awaiting_approval" not in loop.metadata
    assert tool_approvals.pending() == []


# ── WP-07 live finding: a stop while parked is a cancellation, not an error ──


def _real_engine_loop():
    from umrs.knowledge.execution_port import FakeWorkflowExecutionPort
    from umrs.knowledge.loop import AutonomousResearchLoop
    from umrs.knowledge.planner import ResearchExperiment

    engine = AutonomousResearchLoop(simulated=False, workflow_port=FakeWorkflowExecutionPort())
    loop = ResearchLoop(
        loop_id="loop-park",
        project_id="p",
        research_goal="g",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.EXPERIMENT_EXECUTION,
        max_iterations=1,
        current_iteration=1,
        metadata={"mode": "ask", "workflow_name": "wf-a"},
    )
    experiment = ResearchExperiment(experiment_id="e1")
    return engine, loop, experiment


def test_stop_while_parked_cancels_instead_of_erroring():
    from umrs.knowledge.loop import LoopIteration

    engine, loop, experiment = _real_engine_loop()
    engine._approval_gate = lambda _action: False  # noqa: SLF001 — driver abort/stop path
    engine._cancel_check = lambda: True  # noqa: SLF001 — the store says STOPPED
    iteration = LoopIteration(iteration_id="it", loop_id=loop.loop_id, iteration_number=1)
    engine._execute_experiment(loop, iteration, experiment)  # noqa: SLF001
    assert loop.status == LoopStatus.STOPPED
    assert loop.phase == LoopPhase.CANCELLED
    assert iteration.status == "cancelled"
    assert "error" not in loop.metadata


def test_rejection_without_stop_is_an_error():
    from umrs.knowledge.loop import LoopIteration

    engine, loop, experiment = _real_engine_loop()
    engine._approval_gate = lambda _action: False  # noqa: SLF001 — decided "no"
    iteration = LoopIteration(iteration_id="it", loop_id=loop.loop_id, iteration_number=1)
    with pytest.raises(RuntimeError, match="operator rejected"):
        engine._execute_experiment(loop, iteration, experiment)  # noqa: SLF001
