"""RC1 M2 Integration Tests — Pipeline Integration.

Tests the `run_project_workflow()` function that exercises the full
scientific workflow end-to-end:

  1. CREATE — Initialize ResearchProject
  2. INGEST — Ingest all files from directories
  3. UNDERSTAND — Extract knowledge, build graph, calculate coverage
  4. VERIFY — Answer verification questions with evidence
  5. CHECK_READINESS — Generate Research Readiness Report
  6. APPROVE — User approves (or auto_approve for testing)
  7. PLAN — Generate research plan
  8. RESEARCH — Execute autonomous research loop
  9. MEMORY — Update scientific memory
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from umrs.knowledge.project import ProjectPhase, ResearchProject
from umrs.knowledge.workflow import (
    WorkflowOptions,
    WorkflowResult,
    create_and_ingest,
    full_research_cycle,
    run_project_workflow,
    understand_and_verify,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a realistic project directory."""
    (tmp_path / "README.md").write_text("# Test Project\n\nThis is a test.\n")

    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "paper1.pdf").write_bytes(
        b"%PDF-1.4\nThis paper presents a novel method for molecular property prediction. "
        b"The objective is to demonstrate transformers can predict molecular properties. "
        b"Keywords: machine learning, molecular properties, transformers"
    )

    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    nb1 = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"kernelspec": {"name": "python3", "language": "python"}},
        "cells": [
            {"cell_type": "markdown", "source": "# Experiment\n", "metadata": {}},
            {"cell_type": "code", "source": "import torch\nlr = 0.001", "execution_count": 1, "outputs": [], "metadata": {}},
        ],
    }
    (notebooks_dir / "experiment.ipynb").write_text(json.dumps(nb1))

    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "model.py").write_text(
        '"""Transformer model."""\nimport torch.nn as nn\n'
        "class TransformerModel(nn.Module):\n    pass\n"
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "qm9.csv").write_text("smiles,prop\nCCO,1.5\n")

    return tmp_path


# ── M2: run_project_workflow() Tests ─────────────────────────────────────


def test_run_project_workflow_basic(project_dir: Path):
    """M2: run_project_workflow() runs the full pipeline end-to-end."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        name="Test Project",
        research_goal="What research should we perform next?",
        max_iterations=2,
        auto_approve=True,
    )

    assert isinstance(result, WorkflowResult)
    assert result.success is True
    assert result.error is None
    assert result.project.phase == ProjectPhase.COMPLETED
    assert result.artifacts_ingested > 0
    assert result.total_duration_seconds > 0


def test_run_project_workflow_ingests_files(project_dir: Path):
    """M2: run_project_workflow() ingests all files from the directory."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        auto_approve=True,
        max_iterations=1,
    )

    assert result.artifacts_ingested >= 4  # README, paper, notebook, code, data


def test_run_project_workflow_extracts_knowledge(project_dir: Path):
    """M2: run_project_workflow() extracts knowledge from artifacts."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        auto_approve=True,
        max_iterations=1,
    )

    assert result.knowledge_extracted > 0
    assert result.graph_nodes > 0
    assert result.coverage_score >= 0.0


def test_run_project_workflow_checks_readiness(project_dir: Path):
    """M2: run_project_workflow() generates a Research Readiness Report."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        auto_approve=True,
        max_iterations=1,
    )

    assert result.readiness_level != "unknown"
    assert result.confidence_level != "unknown"


def test_run_project_workflow_plans_research(project_dir: Path):
    """M2: run_project_workflow() generates a research plan."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        research_goal="What research should we perform next?",
        auto_approve=True,
        max_iterations=1,
    )

    assert result.research_objectives > 0
    assert result.research_hypotheses > 0
    assert result.research_experiments > 0


def test_run_project_workflow_runs_research(project_dir: Path):
    """M2: run_project_workflow() executes the research loop."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        auto_approve=True,
        max_iterations=3,
    )

    assert result.research_iterations == 3
    assert result.research_findings > 0
    assert result.project.phase == ProjectPhase.COMPLETED


def test_run_project_workflow_updates_memory(project_dir: Path):
    """M2: run_project_workflow() updates scientific memory."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        auto_approve=True,
        max_iterations=2,
    )

    assert result.memory_entries > 0


def test_run_project_workflow_with_verification_questions(project_dir: Path):
    """M2: run_project_workflow() can answer verification questions."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        auto_approve=True,
        max_iterations=1,
        verify_questions=["Explain my project", "What algorithms are implemented?"],
    )

    assert result.success is True
    # The verification questions should have been asked
    assert "verify" in result.phase_durations


def test_run_project_workflow_without_auto_approve(project_dir: Path):
    """M2: run_project_workflow() without auto_approve stops at readiness check."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        auto_approve=False,
        max_iterations=3,
    )

    # Should not have run research
    assert result.research_iterations == 0
    # Project should be in READY phase (not COMPLETED)
    assert result.project.phase == ProjectPhase.READY


def test_run_project_workflow_phase_durations(project_dir: Path):
    """M2: run_project_workflow() records per-phase durations."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        auto_approve=True,
        max_iterations=1,
    )

    assert "ingest" in result.phase_durations
    assert "understand" in result.phase_durations
    assert "check_readiness" in result.phase_durations
    assert "plan" in result.phase_durations
    assert "research" in result.phase_durations
    # All durations should be non-negative
    for phase, duration in result.phase_durations.items():
        assert duration >= 0, f"Phase {phase} has negative duration: {duration}"


def test_run_project_workflow_to_dict(project_dir: Path):
    """M2: WorkflowResult.to_dict() returns complete summary."""
    result = run_project_workflow(
        "proj-001",
        project_dir,
        auto_approve=True,
        max_iterations=1,
    )

    data = result.to_dict()

    assert data["success"] is True
    assert data["artifacts_ingested"] > 0
    assert data["knowledge_extracted"] > 0
    assert data["graph_nodes"] > 0
    assert "project_summary" in data
    assert data["project_summary"]["project_id"] == "proj-001"


def test_run_project_workflow_handles_errors(project_dir: Path):
    """M2: run_project_workflow() captures errors gracefully."""
    result = run_project_workflow(
        "proj-001",
        "/nonexistent/directory",
        auto_approve=True,
        max_iterations=1,
    )

    # Should not crash, but may have 0 artifacts
    assert isinstance(result, WorkflowResult)


# ── M2: Convenience Function Tests ────────────────────────────────────────


def test_create_and_ingest(project_dir: Path):
    """M2: create_and_ingest() creates project and ingests files."""
    project = create_and_ingest(
        "proj-001",
        project_dir,
        name="Test Project",
    )

    assert isinstance(project, ResearchProject)
    assert project.project_id == "proj-001"
    assert project.name == "Test Project"
    assert len(project.knowledge_store.list_artifacts()) > 0


def test_understand_and_verify(project_dir: Path):
    """M2: understand_and_verify() understands project and verifies."""
    project = create_and_ingest("proj-001", project_dir)
    result = understand_and_verify(
        project,
        questions=["Explain my project", "What algorithms are implemented?"],
    )

    assert "project_summary" in result
    assert "verification_answers" in result
    assert len(result["verification_answers"]) == 2


def test_full_research_cycle(project_dir: Path):
    """M2: full_research_cycle() runs the full research cycle."""
    result = full_research_cycle(
        "proj-001",
        project_dir,
        research_goal="What research should we perform next?",
        max_iterations=2,
        auto_approve=True,
    )

    assert isinstance(result, WorkflowResult)
    assert result.success is True
    assert result.project.phase == ProjectPhase.COMPLETED
    assert result.research_iterations == 2


def test_workflow_options_defaults():
    """M2: WorkflowOptions has sensible defaults."""
    opts = WorkflowOptions()

    assert opts.research_goal == "What research should we perform next?"
    assert opts.max_iterations == 3
    assert opts.verify_questions == []
    assert opts.auto_approve is False
    assert opts.user_id == "user"


def test_workflow_options_custom():
    """M2: WorkflowOptions can be customized."""
    opts = WorkflowOptions(
        research_goal="Custom goal",
        max_iterations=5,
        verify_questions=["Question 1", "Question 2"],
        auto_approve=True,
        user_id="test-user",
    )

    assert opts.research_goal == "Custom goal"
    assert opts.max_iterations == 5
    assert len(opts.verify_questions) == 2
    assert opts.auto_approve is True
    assert opts.user_id == "test-user"


# ── M2: End-to-End Integration Tests ──────────────────────────────────────


def test_full_pipeline_end_to_end(project_dir: Path):
    """M2: The full pipeline runs end-to-end without manual intervention.

    This is the Blank Project Test in miniature:
      1. Create project
      2. Ingest files
      3. Understand
      4. Verify
      5. Check readiness
      6. Approve
      7. Plan
      8. Research
      9. Memory updated

    If any step is manual, that becomes the next engineering target.
    """
    # Run the full pipeline
    result = run_project_workflow(
        "e2e-001",
        project_dir,
        name="End-to-End Test",
        description="Full pipeline test",
        research_goal="What research should we perform next?",
        max_iterations=3,
        auto_approve=True,
        verify_questions=[
            "Explain my project",
            "What algorithms are implemented?",
            "Which notebook produced Figure 1?",
        ],
    )

    # ── Verify every stage completed ─────────────────────────────────
    assert result.success is True, f"Workflow failed: {result.error}"
    assert result.error is None

    # 1. Project created
    assert result.project.project_id == "e2e-001"
    assert result.project.name == "End-to-End Test"

    # 2. Ingest
    assert result.artifacts_ingested >= 4
    assert "ingest" in result.phase_durations

    # 3. Understand
    assert result.knowledge_extracted > 0
    assert result.graph_nodes > 0
    assert "understand" in result.phase_durations

    # 4. Verify
    assert "verify" in result.phase_durations

    # 5. Check readiness
    assert result.readiness_level != "unknown"
    assert "check_readiness" in result.phase_durations

    # 6. Approve (auto-approved)
    assert result.project.readiness_report is not None
    assert result.project.readiness_report.user_approved is True

    # 7. Plan
    assert result.research_objectives > 0
    assert result.research_hypotheses > 0
    assert result.research_experiments > 0
    assert "plan" in result.phase_durations

    # 8. Research
    assert result.research_iterations == 3
    assert result.research_findings > 0
    assert result.project.phase == ProjectPhase.COMPLETED
    assert "research" in result.phase_durations

    # 9. Memory updated
    assert result.memory_entries > 0

    # ── Verify workflow result to_dict ────────────────────────────────
    data = result.to_dict()
    assert data["success"] is True
    assert data["artifacts_ingested"] >= 4
    assert data["research_iterations"] == 3
    assert data["memory_entries"] > 0


def test_pipeline_is_reproducible(project_dir: Path):
    """M2: The pipeline produces consistent results across runs."""
    result1 = run_project_workflow(
        "repro-001",
        project_dir,
        auto_approve=True,
        max_iterations=2,
    )
    result2 = run_project_workflow(
        "repro-002",
        project_dir,
        auto_approve=True,
        max_iterations=2,
    )

    # Both should succeed
    assert result1.success is True
    assert result2.success is True
    # Both should ingest the same number of artifacts
    assert result1.artifacts_ingested == result2.artifacts_ingested
    # Both should extract the same amount of knowledge
    assert result1.knowledge_extracted == result2.knowledge_extracted
    # Both should have the same number of research iterations
    assert result1.research_iterations == result2.research_iterations


def test_pipeline_handles_empty_directory(tmp_path: Path):
    """M2: The pipeline handles empty directories gracefully."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = run_project_workflow(
        "empty-001",
        empty_dir,
        auto_approve=True,
        max_iterations=1,
    )

    # Should not crash
    assert isinstance(result, WorkflowResult)
    # Should have 0 artifacts
    assert result.artifacts_ingested == 0


def test_pipeline_handles_multiple_directories(project_dir: Path):
    """M2: The pipeline handles multiple input directories."""
    # Pass individual subdirectories
    result = run_project_workflow(
        "multi-001",
        project_dir / "papers",
        project_dir / "notebooks",
        project_dir / "code",
        project_dir / "data",
        auto_approve=True,
        max_iterations=1,
    )

    assert result.success is True
    assert result.artifacts_ingested >= 3  # paper, notebook, code
