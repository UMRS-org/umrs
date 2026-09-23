"""D-34 first leg (ADR-066 amendment 2026-09-24): deferred continuation.

A profile-mode loop with a PENDING named-class trigger no longer
hard-parks: the iteration finalises as ``awaiting_decision``, the loop
continues on ungated work, the durable question stays in the inbox
(idempotent re-request under the same approval id) — and the runner's
deferral ledger bounds the spin (3 strikes under one approval id →
hard park as before). Ask mode keeps hard parking.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from umrs.knowledge.agentic_engine import AgenticEngine
from umrs.knowledge.engine_loop_runner import EngineLoopRunner
from umrs.knowledge.loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop


def _store(records):
    class _S:
        def get(self, loop_id):
            return records.get(loop_id)

        def save(self, loop):
            records[loop.loop_id] = loop

    return _S()


def _loop(**meta) -> ResearchLoop:
    loop = ResearchLoop(
        loop_id="L-d",
        project_id="proj-d",
        research_goal="g",
        max_iterations=5,
        status=LoopStatus.RUNNING,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    loop.metadata.update(meta)
    return loop


def _profile_loop():
    return _loop(
        mode="profile", pause_profile={"triggers": ["cost", "competing_hypotheses"]}
    )


class _FakeApprovals:
    """Scripted durable inbox: request mints ids, `get` answers decisions."""

    def __init__(self, decisions=None, decide_after=0):
        self._decisions = decisions or {}
        self._decide_after = decide_after
        self.requests = 0
        self.gets = 0

    def request_approval(self, loop_id, iteration, action, approval_id=None, **kw):
        self.requests += 1
        return {"id": approval_id or f"appr-{self.requests}"}

    def get(self, approval_id):
        self.gets += 1
        status = self._decisions.get(approval_id)
        if status is None and self._decide_after and self.gets > self._decide_after:
            status = "approved"
        return {"id": approval_id, "status": status} if status else None


def _gate(monkeypatch, approvals, loop):
    monkeypatch.setattr(
        "umrs.knowledge.tool_approvals.request_approval", approvals.request_approval
    )
    monkeypatch.setattr("umrs.knowledge.tool_approvals.get", approvals.get)
    runner = EngineLoopRunner(
        loop, engine_factory=lambda: None, store=_store({loop.loop_id: loop}),
        poll_seconds=0.01,
    )
    return runner._make_approval_gate(loop)


class TestProfileDeferral:
    def test_pending_trigger_defers(self, monkeypatch):
        approvals = _FakeApprovals()
        gate = _gate(monkeypatch, approvals, _profile_loop())
        assert gate({"kind": "workflow_trigger", "iteration": 1}) == "deferred"
        loop = _profile_loop()
        gate = _gate(monkeypatch, _FakeApprovals(), loop)
        gate({"kind": "workflow_trigger", "iteration": 1})
        assert loop.metadata["awaiting_approval"] == "appr-1"
        assert loop.metadata["deferred_gate"] == {"approval_id": "appr-1", "count": 1}

    def test_decided_approved_proceeds(self, monkeypatch):
        approvals = _FakeApprovals(decisions={"appr-1": "approved"})
        loop = _profile_loop()
        gate = _gate(monkeypatch, approvals, loop)
        assert gate({"kind": "workflow_trigger", "iteration": 1}) is True
        assert "awaiting_approval" not in loop.metadata  # cleared on decision

    def test_decided_rejected_denies(self, monkeypatch):
        approvals = _FakeApprovals(decisions={"appr-1": "rejected"})
        gate = _gate(monkeypatch, approvals, _profile_loop())
        assert gate({"kind": "workflow_trigger", "iteration": 1}) is False

    def test_non_trigger_class_ungated(self, monkeypatch):
        gate = _gate(monkeypatch, _FakeApprovals(), _profile_loop())
        assert gate({"kind": "mystery"}) is True

    def test_three_strikes_hard_park(self, monkeypatch):
        approvals = _FakeApprovals(decide_after=6)
        loop = _profile_loop()
        gate = _gate(monkeypatch, approvals, loop)
        assert gate({"kind": "workflow_trigger", "iteration": 1}) == "deferred"
        assert gate({"kind": "workflow_trigger", "iteration": 2}) == "deferred"
        # strike 3 under the same approval id: hard park (decided → True)
        assert gate({"kind": "workflow_trigger", "iteration": 3}) is True
        assert approvals.gets > 6  # it polled — the park path ran

    def test_ask_mode_never_defers(self, monkeypatch):
        approvals = _FakeApprovals(decide_after=1)
        loop = _loop(mode="ask")
        gate = _gate(monkeypatch, approvals, loop)
        assert gate({"kind": "workflow_trigger", "iteration": 1}) is True
        assert "deferred_gate" not in loop.metadata


class TestEngineDeferredFinalisation:
    def _engine(self):
        from umrs.knowledge.loop import AutonomousResearchLoop

        return AutonomousResearchLoop(simulated=True)

    def _iteration(self, loop):
        return LoopIteration(
            iteration_id="it-1",
            loop_id=loop.loop_id,
            iteration_number=1,
            phase=LoopPhase.HYPOTHESIS_GENERATION,
            status="running",
        )

    def test_hypothesis_gate_deferred_finalises_iteration(self):
        from umrs.knowledge.planner import ResearchHypothesis, ResearchPlan

        loop = _profile_loop()
        loop.plan = ResearchPlan(
            hypotheses=[
                ResearchHypothesis(statement="a"),
                ResearchHypothesis(statement="b"),
            ]
        )
        engine = self._engine()
        engine._approval_gate = lambda action: "deferred"
        iteration = self._iteration(loop)
        assert engine._select_and_gate_hypothesis(loop, iteration, []) is False
        assert iteration.status == "awaiting_decision"
        assert iteration.results["deferred_gate"] == "competing-hypotheses"
        assert loop.iterations[-1] is iteration  # appended, nothing fabricated

    def test_hypothesis_gate_decided_no_still_denies(self):
        from umrs.knowledge.planner import ResearchHypothesis, ResearchPlan

        loop = _profile_loop()
        loop.plan = ResearchPlan(
            hypotheses=[
                ResearchHypothesis(statement="a"),
                ResearchHypothesis(statement="b"),
            ]
        )
        engine = self._engine()
        engine._approval_gate = lambda action: False
        iteration = self._iteration(loop)
        with pytest.raises(RuntimeError):
            engine._select_and_gate_hypothesis(loop, iteration, [])
        assert iteration.status == "running"  # denied path never finalises


class TestAgenticDeferredIteration:
    def test_deferred_verdict_yields_awaiting_iteration(self):
        calls = []

        def executor(loop_id, n, goal):
            calls.append(n)
            return {"hypothesis": "h"}

        engine = AgenticEngine(executor=executor)
        engine._approval_gate = lambda action: "deferred"
        loop = _profile_loop()
        iteration = engine._advance_one(loop)
        assert iteration.status == "awaiting_decision"
        assert iteration.results["deferred_gate"] == "agent_iteration"
        assert calls == []  # the gated executor never ran — nothing fabricated
        assert loop.iterations[-1] is iteration
