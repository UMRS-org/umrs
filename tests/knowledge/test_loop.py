"""Tests for the autonomous research loop module (Phase 12)."""

from __future__ import annotations

import pytest

from umrs.knowledge.loop import (
    AutonomousResearchLoop,
    LoopIteration,
    LoopPhase,
    LoopStatus,
    ResearchLoop,
    run_research_loop,
)


# ── Enum Tests ─────────────────────────────────────────────────────────────


def test_loop_phase_enum():
    """Test LoopPhase enum values."""
    assert LoopPhase.INITIALIZED == "initialized"
    assert LoopPhase.PLANNING == "planning"
    assert LoopPhase.KNOWLEDGE_RETRIEVAL == "knowledge_retrieval"
    assert LoopPhase.SEARCHING == "searching"
    assert LoopPhase.UPDATING_KNOWLEDGE == "updating_knowledge"
    assert LoopPhase.HYPOTHESIS_GENERATION == "hypothesis_generation"
    assert LoopPhase.EXPERIMENT_DESIGN == "experiment_design"
    assert LoopPhase.EXPERIMENT_EXECUTION == "experiment_execution"
    assert LoopPhase.OUTPUT_COLLECTION == "output_collection"
    assert LoopPhase.ANALYSIS == "analysis"
    assert LoopPhase.EVALUATION == "evaluation"
    assert LoopPhase.PUBLISHING == "publishing"
    assert LoopPhase.MEMORY_UPDATE == "memory_update"
    assert LoopPhase.COMPLETED == "completed"
    assert LoopPhase.FAILED == "failed"
    assert LoopPhase.CANCELLED == "cancelled"


def test_loop_status_enum():
    """Test LoopStatus enum values."""
    assert LoopStatus.RUNNING == "running"
    assert LoopStatus.PAUSED == "paused"
    assert LoopStatus.STOPPED == "stopped"
    assert LoopStatus.ERROR == "error"


# ── Model Tests ────────────────────────────────────────────────────────────


def test_loop_iteration_defaults():
    """Test LoopIteration defaults."""
    iteration = LoopIteration()

    assert iteration.iteration_id == ""
    assert iteration.loop_id == ""
    assert iteration.iteration_number == 0
    assert iteration.phase == LoopPhase.INITIALIZED
    assert iteration.status == "pending"
    assert iteration.results == {}
    assert iteration.findings == []


def test_loop_iteration_to_dict():
    """Test LoopIteration to_dict."""
    iteration = LoopIteration(
        iteration_id="iter-001",
        loop_id="loop-001",
        iteration_number=1,
        phase=LoopPhase.ANALYSIS,
        status="completed",
    )

    data = iteration.to_dict()
    assert data["iteration_id"] == "iter-001"
    assert data["loop_id"] == "loop-001"
    assert data["iteration_number"] == 1
    assert data["phase"] == "analysis"
    assert data["status"] == "completed"


def test_research_loop_defaults():
    """Test ResearchLoop defaults."""
    loop = ResearchLoop()

    assert loop.loop_id == ""
    assert loop.project_id == ""
    assert loop.research_goal == ""
    assert loop.status == LoopStatus.RUNNING
    assert loop.phase == LoopPhase.INITIALIZED
    assert loop.max_iterations == 10
    assert loop.current_iteration == 0
    assert loop.iterations == []
    assert loop.total_hypotheses_tested == 0
    assert loop.total_experiments_run == 0
    assert loop.total_findings == 0
    assert loop.success_rate == 0.0


def test_research_loop_to_dict():
    """Test ResearchLoop to_dict."""
    loop = ResearchLoop(
        loop_id="loop-001",
        project_id="proj-001",
        research_goal="Understand X",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.PLANNING,
    )

    data = loop.to_dict()
    assert data["loop_id"] == "loop-001"
    assert data["project_id"] == "proj-001"
    assert data["research_goal"] == "Understand X"
    assert data["status"] == "running"
    assert data["phase"] == "planning"


# ── AutonomousResearchLoop Tests ──────────────────────────────────────────


def test_loop_engine_init():
    """Test AutonomousResearchLoop initialization."""
    loop_engine = AutonomousResearchLoop(simulated=True)
    assert loop_engine is not None


def test_loop_engine_start_loop():
    """Test starting a research loop."""
    loop_engine = AutonomousResearchLoop(simulated=True)
    loop = loop_engine.start_loop(
        project_id="proj-001",
        research_goal="Understand the effect of learning rate on convergence",
        max_iterations=3,
    )

    assert loop.project_id == "proj-001"
    assert loop.status == LoopStatus.RUNNING
    assert loop.phase == LoopPhase.PLANNING
    assert loop.max_iterations == 3
    assert loop.plan is not None
    assert len(loop.plan.objectives) >= 1
    assert len(loop.plan.hypotheses) >= 1
    assert len(loop.plan.experiments) >= 1


def test_loop_engine_run_loop():
    """Test running a research loop to completion."""
    loop_engine = AutonomousResearchLoop(simulated=True)
    loop = loop_engine.start_loop(
        project_id="proj-001",
        research_goal="Understand X",
        max_iterations=3,
    )

    loop = loop_engine.run(loop)

    assert loop.current_iteration == 3
    assert loop.phase == LoopPhase.COMPLETED
    assert loop.status == LoopStatus.STOPPED
    assert len(loop.iterations) == 3
    assert loop.total_experiments_run == 3
    assert loop.total_findings == 3
    assert loop.total_hypotheses_tested == 3
    assert loop.success_rate == 1.0  # 3 findings / 3 experiments


def test_loop_engine_run_loop_with_zero_iterations():
    """Test running a loop with max_iterations=0."""
    loop_engine = AutonomousResearchLoop(simulated=True)
    loop = loop_engine.start_loop(
        project_id="proj-001",
        research_goal="Understand X",
        max_iterations=0,
    )

    loop = loop_engine.run(loop)

    assert loop.current_iteration == 0
    assert loop.phase == LoopPhase.COMPLETED
    assert len(loop.iterations) == 0


def test_loop_engine_iterations_have_unique_ids():
    """Test that iterations have unique IDs."""
    loop_engine = AutonomousResearchLoop(simulated=True)
    loop = loop_engine.start_loop(
        project_id="proj-001",
        research_goal="Understand X",
        max_iterations=5,
    )

    loop = loop_engine.run(loop)

    iter_ids = [it.iteration_id for it in loop.iterations]
    assert len(iter_ids) == len(set(iter_ids))


def test_loop_engine_iterations_have_correct_numbers():
    """Test that iterations have correct sequential numbers."""
    loop_engine = AutonomousResearchLoop(simulated=True)
    loop = loop_engine.start_loop(
        project_id="proj-001",
        research_goal="Understand X",
        max_iterations=3,
    )

    loop = loop_engine.run(loop)

    for i, iteration in enumerate(loop.iterations):
        assert iteration.iteration_number == i + 1


def test_loop_engine_run_loop_phases():
    """Test that each iteration goes through the correct phases."""
    loop_engine = AutonomousResearchLoop(simulated=True)
    loop = loop_engine.start_loop(
        project_id="proj-001",
        research_goal="Understand X",
        max_iterations=1,
    )

    loop = loop_engine.run(loop)

    iteration = loop.iterations[0]
    # Final phase should be MEMORY_UPDATE
    assert iteration.phase == LoopPhase.MEMORY_UPDATE
    assert iteration.status == "completed"


def test_memory_update_phase_registers_findings():
    """MEMORY_UPDATE must write analysis + findings when a manager is injected."""
    from umrs.knowledge.memory import MemoryEntryType, ScientificMemoryManager

    manager = ScientificMemoryManager()
    engine = AutonomousResearchLoop(simulated=True, memory_manager=manager)
    loop = engine.start_loop("proj-mem", "goal", max_iterations=1)
    engine.run(loop)

    findings = manager.list_by_type(MemoryEntryType.ACCEPTED_FINDING, project_id="proj-mem")
    analyses = manager.list_by_type(MemoryEntryType.ANALYSIS_RESULT, project_id="proj-mem")
    assert len(findings) == 1
    assert len(analyses) == 1
    assert "Finding from iteration" in findings[0].content


def test_resolve_workflow_name_requires_explicit_metadata():
    from umrs.knowledge.planner import ResearchExperiment

    with pytest.raises(RuntimeError, match="no experiment selected"):
        AutonomousResearchLoop._resolve_workflow_name(None)

    bare = ResearchExperiment(experiment_id="exp-1")
    with pytest.raises(RuntimeError, match="workflow_name"):
        AutonomousResearchLoop._resolve_workflow_name(bare)

    stamped = ResearchExperiment(
        experiment_id="exp-2",
        metadata={"workflow_name": "demo08_yield_eval"},
    )
    assert AutonomousResearchLoop._resolve_workflow_name(stamped) == "demo08_yield_eval"


def test_extract_evidence_figures_does_not_invent_batch3():
    figures = AutonomousResearchLoop._extract_evidence_figures(
        ["accuracy was 92% on the holdout set"],
        {"metrics": {}},
    )
    assert "yield_pct" not in figures
    assert "batch_id" not in figures


# ── Convenience Function Tests ─────────────────────────────────────────────


def test_run_research_loop_convenience_function():
    """Test the run_research_loop convenience function."""
    loop = run_research_loop(
        project_id="proj-001",
        research_goal="Understand X",
        max_iterations=2,
        simulate=True,
    )

    assert isinstance(loop, ResearchLoop)
    assert loop.project_id == "proj-001"
    assert loop.current_iteration == 2
    assert len(loop.iterations) == 2
    assert loop.phase == LoopPhase.COMPLETED
