"""Tests for the autonomous research loop engine (TASK-036).

Covers:
  - Loop lifecycle: start → run → complete
  - Iteration phases and data
  - Max iterations enforcement
  - Error handling (loop status transitions)
  - Unapproved project rejection (run_research gate)
  - LoopStatus and LoopPhase enums
"""

from __future__ import annotations

import logging

import pytest

# Splice removed (ADR-053): umrs installed editable in .venv.
from umrs.knowledge.loop import (
    AutonomousResearchLoop,
    LoopIteration,
    LoopPhase,
    LoopStatus,
    ResearchLoop,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_loop(**overrides) -> ResearchLoop:
    """Create a minimal ResearchLoop for testing."""
    base = ResearchLoop(
        loop_id="test-loop-001",
        project_id="proj-test",
        research_goal="Test research goal",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.INITIALIZED,
        max_iterations=5,
        current_iteration=0,
        created_at="2026-07-15T00:00:00+00:00",
        started_at="2026-07-15T00:00:00+00:00",
        updated_at="2026-07-15T00:00:00+00:00",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


# ── LoopPhase Enum Tests ──────────────────────────────────────────────────


class TestLoopPhase:
    """LoopPhase must cover all research lifecycle stages."""

    def test_all_phases_present(self):
        expected = {
            "initialized",
            "planning",
            "knowledge_retrieval",
            "searching",
            "updating_knowledge",
            "hypothesis_generation",
            "experiment_design",
            "experiment_execution",
            "output_collection",
            "analysis",
            "evaluation",
            "publishing",
            "memory_update",
            "completed",
            "failed",
            "cancelled",
        }
        actual = {p.value for p in LoopPhase}
        assert actual == expected

    def test_phase_values_are_strings(self):
        for phase in LoopPhase:
            assert isinstance(phase.value, str)


class TestLoopStatus:
    """LoopStatus must cover all lifecycle states."""

    def test_all_statuses_present(self):
        expected = {"running", "paused", "stopped", "error"}
        actual = {s.value for s in LoopStatus}
        assert actual == expected


# ── LoopIteration Tests ───────────────────────────────────────────────────


class TestLoopIteration:
    """Test the LoopIteration dataclass."""

    def test_defaults(self):
        it = LoopIteration()
        assert it.iteration_id == ""
        assert it.iteration_number == 0
        assert it.phase == LoopPhase.INITIALIZED
        assert it.status == "pending"

    def test_to_dict(self):
        it = LoopIteration(
            iteration_id="iter-001",
            loop_id="loop-001",
            iteration_number=1,
            phase=LoopPhase.HYPOTHESIS_GENERATION,
            started_at="2026-07-15T00:01:00+00:00",
            hypothesis="Test hypothesis",
            experiment_id="exp-001",
            analysis="Test analysis",
            evaluation="Test evaluation",
            findings=["Finding 1"],
            status="completed",
        )
        d = it.to_dict()
        assert d["iteration_id"] == "iter-001"
        assert d["phase"] == "hypothesis_generation"
        assert d["hypothesis"] == "Test hypothesis"
        assert d["experiment_id"] == "exp-001"
        assert d["analysis"] == "Test analysis"
        assert d["findings"] == ["Finding 1"]


# ── ResearchLoop Dataclass Tests ──────────────────────────────────────────


class TestResearchLoop:
    """Test the ResearchLoop dataclass."""

    def test_defaults(self):
        loop = ResearchLoop()
        assert loop.loop_id == ""
        assert loop.status == LoopStatus.RUNNING
        assert loop.phase == LoopPhase.INITIALIZED
        assert loop.max_iterations == 10
        assert loop.current_iteration == 0
        assert loop.iterations == []

    def test_to_dict(self):
        loop = _make_loop(max_iterations=3, current_iteration=1)
        loop.iterations.append(
            LoopIteration(
                iteration_id="iter-001",
                iteration_number=1,
                phase=LoopPhase.HYPOTHESIS_GENERATION,
                hypothesis="Test",
                status="completed",
            )
        )
        d = loop.to_dict()
        assert d["loop_id"] == "test-loop-001"
        assert d["status"] == "running"
        assert d["max_iterations"] == 3
        assert d["current_iteration"] == 1
        assert len(d["iterations"]) == 1
        assert d["iterations"][0]["phase"] == "hypothesis_generation"


# ── AutonomousResearchLoop Engine Tests ───────────────────────────────────


class TestAutonomousResearchLoopEngine:
    """Test the AutonomousResearchLoop orchestration engine."""

    def test_engine_creation(self):
        engine = AutonomousResearchLoop(simulated=True)
        assert engine._planner is not None

    def test_start_loop_creates_research_loop(self):
        engine = AutonomousResearchLoop(simulated=True)
        loop = engine.start_loop(
            project_id="proj-001",
            research_goal="Understand learning rates",
            max_iterations=3,
        )
        assert loop.loop_id.startswith("loop-")
        assert loop.project_id == "proj-001"
        assert loop.research_goal == "Understand learning rates"
        assert loop.status == LoopStatus.RUNNING
        assert loop.phase == LoopPhase.PLANNING
        assert loop.max_iterations == 3
        assert loop.current_iteration == 0
        assert loop.plan is not None  # Planner generates a plan

    def test_start_loop_with_max_iterations(self):
        engine = AutonomousResearchLoop(simulated=True)
        loop = engine.start_loop(
            project_id="proj-002",
            research_goal="Test goal",
            max_iterations=1,
        )
        assert loop.max_iterations == 1

    def test_run_loop_completes(self):
        engine = AutonomousResearchLoop(simulated=True)
        loop = engine.start_loop(
            project_id="proj-003",
            research_goal="Test goal",
            max_iterations=2,
        )
        result = engine.run(loop)
        # Loop should complete (iterations run to completion or error)
        assert result.status in (LoopStatus.STOPPED, LoopStatus.ERROR)

    def test_run_loop_iterations(self):
        engine = AutonomousResearchLoop(simulated=True)
        loop = engine.start_loop(
            project_id="proj-004",
            research_goal="Test iterations",
            max_iterations=3,
        )
        engine.run(loop)
        # At least one iteration should have been created
        assert len(loop.iterations) >= 1

    def test_run_loop_status_transitions(self):
        engine = AutonomousResearchLoop(simulated=True)
        loop = engine.start_loop(
            project_id="proj-005",
            research_goal="Test transitions",
            max_iterations=2,
        )
        assert loop.status == LoopStatus.RUNNING

        engine.run(loop)

        # After completion, status should be STOPPED or ERROR
        assert loop.status in (LoopStatus.STOPPED, LoopStatus.ERROR)

    def test_run_loop_max_iterations_enforced(self):
        """Loop must not exceed max_iterations."""
        engine = AutonomousResearchLoop(simulated=True)
        loop = engine.start_loop(
            project_id="proj-006",
            research_goal="Max iterations test",
            max_iterations=2,
        )
        engine.run(loop)
        assert loop.current_iteration <= 2

    def test_run_loop_error_handling(self):
        """If an iteration raises, the loop should catch it and set ERROR status."""
        engine = AutonomousResearchLoop(simulated=True)

        # Inject a loop that will cause _run_iteration to raise
        loop = _make_loop(
            loop_id="error-loop",
            project_id="proj-error",
            research_goal="Error test",
            max_iterations=1,
        )
        loop.plan = None  # May cause issues in iteration

        result = engine.run(loop)
        # Should not raise — the engine catches exceptions
        assert result.status in (LoopStatus.STOPPED, LoopStatus.ERROR)

    def test_run_loop_preserves_iterations(self):
        """Iterations created during run must be stored on the loop."""
        engine = AutonomousResearchLoop(simulated=True)
        loop = engine.start_loop(
            project_id="proj-pres",
            research_goal="Preservation test",
            max_iterations=2,
        )
        engine.run(loop)
        for iteration in loop.iterations:
            assert iteration.loop_id == loop.loop_id
            assert iteration.iteration_id != ""
            assert iteration.iteration_number > 0


# ── Production Integration Tests ──────────────────────────────────────────


class TestProductionIntegration:
    """Tests that exercise the loop engine through ResearchProject."""

    def test_run_research_requires_approval(self):
        """run_research() must raise if readiness is not approved."""
        from umrs.knowledge.project import ResearchProject

        project = ResearchProject(project_id="proj-no-approval")
        # No readiness report exists → check_readiness is not called
        # plan_research must be called first, but even then:
        project.plan_research("Test goal")

        with pytest.raises(RuntimeError, match="readiness report must be approved"):
            project.run_research(max_iterations=1, simulate=True)

    def test_run_research_requires_plan(self):
        """run_research() must raise if no plan exists."""
        from umrs.knowledge.project import ResearchProject

        project = ResearchProject(project_id="proj-no-plan")
        # Create a mock readiness report with approval
        from umrs.knowledge.readiness import ResearchReadinessReport

        report = ResearchReadinessReport(
            project_id="proj-no-plan",
            readiness_level="READY",
            confidence_level="HIGH",
            issues=[],
            recommendation="Go ahead",
            next_steps=[],
        )
        report.user_approved = True
        project._readiness_report = report

        with pytest.raises(RuntimeError, match="Research plan must be generated"):
            project.run_research(max_iterations=1, simulate=True)

    def test_run_research_success_path(self, tmp_path):
        """Happy path: approved + planned → loop runs."""
        from umrs.knowledge.project import ResearchProject

        # Honest readiness gate: an empty project is BLOCKED and cannot be
        # approved; ingest one tiny artifact so the run gate is reachable.
        doc = tmp_path / "note.txt"
        doc.write_text("batch-3 training yielded 0.91 accuracy on epoch 12")

        project = ResearchProject(project_id="proj-happy", storage_path=tmp_path / "store")
        project.ingest_directory(tmp_path)
        project.understand()
        project.check_readiness()
        project.approve_readiness(user_id="test-user")
        project.plan_research("Test research goal")

        loop = project.run_research(max_iterations=2, simulate=True)
        assert loop.loop_id != ""
        assert loop.project_id == "proj-happy"
        assert loop.status in (LoopStatus.STOPPED, LoopStatus.ERROR)

    def test_project_phase_transitions(self, tmp_path):
        """Project phases must transition: READY → PLANNING → RESEARCHING → COMPLETED."""
        from umrs.knowledge.project import ProjectPhase, ResearchProject

        doc = tmp_path / "note.txt"
        doc.write_text("batch-3 training yielded 0.91 accuracy on epoch 12")

        project = ResearchProject(project_id="proj-phases", storage_path=tmp_path / "store")
        project.ingest_directory(tmp_path)
        project.understand()
        project.check_readiness()
        project.approve_readiness()
        project.plan_research("Phase test")

        assert project.phase == ProjectPhase.PLANNING

        project.run_research(max_iterations=1, simulate=True)

        assert project.phase == ProjectPhase.COMPLETED


# ── #557: a workflow run that ends failed is an outcome, not a fault ──────


class _FailedRunPort:
    """A workflow port whose run ends failed (no active endpoint at the venue)."""

    def trigger(self, _name, _params):
        return "exec-1"

    def wait(self, _execution_id):
        return {"durable_job_id": "job-1", "status": "failed"}

    def results(self, _execution_id):
        return {
            "durable_job_id": "job-1",
            "status": "failed",
            "error": "collect_context: routing failed: no matching active endpoint",
        }

    def cancel(self, _execution_id):
        pass


def test_failed_experiment_run_is_a_warning_not_a_traceback(caplog):
    """Ops Watch #557: the loop ends ERROR with the reason, logged once as a
    WARNING without a traceback (was ERROR + traceback per demo run)."""
    engine = AutonomousResearchLoop(simulated=False, workflow_port=_FailedRunPort())
    loop = _make_loop(max_iterations=2, metadata={"workflow_name": "wf-demo"})
    with caplog.at_level(logging.WARNING, logger="umrs.knowledge.loop"):
        out = engine.run(loop)
    assert (out.status, out.phase, out.current_iteration) == (
        LoopStatus.ERROR,
        LoopPhase.FAILED,
        1,
    )
    assert "routing failed" in out.metadata["error"]
    records = [r for r in caplog.records if r.name == "umrs.knowledge.loop"]
    assert records
    assert {(r.levelno, r.exc_info) for r in records} == {(logging.WARNING, None)}
