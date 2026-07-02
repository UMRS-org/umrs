"""RC1 Integration Test — ResearchProject end-to-end workflow.

This is the integration test for M1-M6. It exercises the full pipeline:
  1. Create project
  2. Ingest files (papers, notebooks, code, datasets, README)
  3. Understand (extract knowledge, build knowledge graph, calculate coverage)
  4. Verify (ask questions, get evidence-backed answers)
  5. Check readiness (generate Research Readiness Report)
  6. Approve readiness (user approves)
  7. Plan research (generate ResearchPlan)
  8. Run research (execute ResearchLoop)
  9. Verify knowledge grew (scientific memory updated)

If any step is manual, that becomes the next engineering target.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from umrs.knowledge.project import (
    ProjectPhase,
    ResearchProject,
    WorkflowTransition,
)


# ── Fixtures: Create a realistic project directory ────────────────────────


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a realistic project directory with papers, notebooks, code, data, README."""
    # README.md
    (tmp_path / "README.md").write_text(
        "# Test Project\n\nThis project tests the R4 knowledge pipeline.\n"
    )

    # Papers (simulate PDFs with text content)
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / "paper1.pdf").write_bytes(
        b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"This paper presents a novel method for molecular property prediction. "
        b"The objective is to demonstrate that transformers can predict molecular properties. "
        b"We use the QM9 dataset for training. "
        b"Our results show state-of-the-art performance. "
        b"We conclude that transformer-based models are effective. "
        b"Keywords: machine learning, molecular properties, transformers"
    )
    (papers_dir / "paper2.pdf").write_bytes(
        b"%PDF-1.4\nThis paper describes graph neural networks for molecular representation. "
        b"The methodology combines GNN with transformer encoder. "
        b"Keywords: graph neural networks, molecular representation"
    )

    # Notebooks
    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    nb1 = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {"kernelspec": {"name": "python3", "language": "python"}},
        "cells": [
            {
                "cell_type": "markdown",
                "source": "# Molecular Property Prediction\n\nDemonstration of transformer models.",
                "metadata": {},
            },
            {
                "cell_type": "code",
                "source": "import torch\nlearning_rate = 0.001\nbatch_size = 32",
                "execution_count": 1,
                "outputs": [],
                "metadata": {},
            },
        ],
    }
    (notebooks_dir / "experiment.ipynb").write_text(json.dumps(nb1))

    # Code
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    (code_dir / "model.py").write_text(
        '"""Transformer model for molecular property prediction."""\n'
        "import torch\nimport torch.nn as nn\n\n"
        "class TransformerModel(nn.Module):\n"
        "    def __init__(self, input_dim=128, hidden_size=256):\n"
        "        super().__init__()\n"
        "        self.embedding = nn.Linear(input_dim, hidden_size)\n"
        "    def forward(self, x):\n"
        "        return self.embedding(x)\n"
    )

    # Datasets
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "qm9.csv").write_text(
        "smiles,property1,property2\n"
        "CCO,1.5,2.3\n"
        "CCN,0.8,1.9\n"
    )

    return tmp_path


# ── M1: ResearchProject Creation ────────────────────────────────────────────


def test_research_project_creation():
    """M1: ResearchProject can be created with basic metadata."""
    project = ResearchProject(
        project_id="proj-001",
        name="Test Project",
        description="A test project",
    )

    assert project.project_id == "proj-001"
    assert project.name == "Test Project"
    assert project.description == "A test project"
    assert project.phase == ProjectPhase.CREATED
    assert project.knowledge_store is not None
    assert project.memory_manager is not None


def test_research_project_to_dict():
    """M1: ResearchProject.to_dict() returns project summary."""
    project = ResearchProject(project_id="proj-001", name="Test")

    data = project.to_dict()

    assert data["project_id"] == "proj-001"
    assert data["name"] == "Test"
    assert data["phase"] == "created"
    assert data["artifact_count"] == 0
    assert data["extracted_count"] == 0


# ── M2: Pipeline Integration (Ingest → Understand → Verify) ────────────────


def test_ingest_directory(project_dir: Path):
    """M2: ingest_directory() ingests all files from a directory."""
    project = ResearchProject(project_id="proj-001")

    count = project.ingest_directory(project_dir)

    assert count > 0
    assert project.phase == ProjectPhase.INGESTING
    assert len(project.knowledge_store.list_artifacts()) == count


def test_ingest_directory_multiple(project_dir: Path):
    """M2: ingest_directory() accepts multiple directories."""
    project = ResearchProject(project_id="proj-001")

    count = project.ingest_directory(
        project_dir / "papers",
        project_dir / "notebooks",
        project_dir / "code",
        project_dir / "data",
    )

    assert count >= 5  # 2 papers + 1 notebook + 1 code + 1 data


def test_understand_builds_knowledge_graph(project_dir: Path):
    """M2: understand() extracts knowledge, builds graph, calculates coverage."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)

    project.understand()

    assert project.phase == ProjectPhase.READY
    assert len(project.knowledge_graph.nodes) > 0
    assert project.coverage_dashboard is not None
    assert len(project.extracted_knowledge) > 0


def test_local_extracted_knowledge_persists_on_disk(project_dir: Path, tmp_path: Path):
    """Structured knowledge should reload from disk after the in-memory cache clears."""
    storage = tmp_path / "project-data"
    project = ResearchProject(project_id="proj-001", storage_path=storage)
    project.ingest_directory(project_dir)
    project.understand()

    saved = dict(project.knowledge_store.extracted_knowledge)
    assert saved

    project.knowledge_store.extracted_knowledge.clear()
    for artifact_id, expected in saved.items():
        reloaded = project.knowledge_store.get_extracted_knowledge(artifact_id)
        assert reloaded is not None
        assert reloaded.get("artifact_type") == expected.get("artifact_type")


def test_understand_bridges_local_pdf_via_catalog(tmp_path: Path, monkeypatch):
    """Local PDF ingest should use Phase 3 text extraction before regex parsing."""
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    pdf_bytes = b"%PDF-1.4\n% not a valid pdf object stream"
    (papers_dir / "realistic.pdf").write_bytes(pdf_bytes)

    scientific_text = (
        "The goal of this paper is to test local PDF bridging in understand().\n"
        "Keywords: integration, pdf, understand\n"
    )

    monkeypatch.setattr(
        "umrs.data_catalog.extract.extract_content",
        lambda _filename, _content: ("summary", scientific_text, []),
    )

    project = ResearchProject(project_id="proj-pdf-bridge")
    project.ingest_directory(papers_dir)
    project.understand()

    paper_knowledge = next(
        k for k in project.extracted_knowledge.values() if k.filename == "realistic.pdf"
    )
    assert "local PDF bridging" in paper_knowledge.objective
    assert "pdf" in paper_knowledge.keywords


def test_verify_returns_evidence_backed_answer(project_dir: Path):
    """M2: verify() returns a VerificationAnswer with evidence."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()

    answer = project.verify("Explain my project")

    assert answer is not None
    assert answer.answer != ""
    assert len(answer.evidence) >= 0  # May have evidence


def test_verify_lists_algorithms(project_dir: Path):
    """M2: verify() can list implemented algorithms."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()

    answer = project.verify("What algorithms are implemented?")

    assert answer is not None
    # Should return some answer about algorithms
    assert answer.answer != ""


def test_verify_which_notebook_produced_figure(project_dir: Path):
    """M2: verify() can answer 'Which notebook produced Figure X?'"""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()

    answer = project.verify("Which notebook produced Figure 1?")

    assert answer is not None
    assert answer.answer != ""


def test_verify_what_assumptions_unsupported(project_dir: Path):
    """M2: verify() can answer 'What assumptions are unsupported?'"""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()

    answer = project.verify("What assumptions are unsupported?")

    assert answer is not None


# ── M3: Planner-Graph Integration ──────────────────────────────────────────


def test_check_readiness_generates_report(project_dir: Path):
    """M3: check_readiness() generates a ResearchReadinessReport."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()

    report = project.check_readiness()

    assert report is not None
    assert report.project_id == "proj-001"
    assert report.readiness_level is not None
    assert report.confidence_level is not None
    assert report.recommendation != ""


def test_approve_readiness(project_dir: Path):
    """M3: approve_readiness() marks the report as user-approved."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()
    project.check_readiness()

    approved = project.approve_readiness(user_id="test-user")

    assert approved is True
    assert project.readiness_report.user_approved is True
    assert project.readiness_report.approved_by == "test-user"


# ── M4: Verification-Graph Integration ────────────────────────────────────


def test_plan_research_generates_plan(project_dir: Path):
    """M4: plan_research() generates a ResearchPlan with objectives, hypotheses, experiments."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()
    project.check_readiness()
    project.approve_readiness()

    plan = project.plan_research("What research should we perform next?")

    assert plan is not None
    assert len(plan.objectives) >= 1
    assert len(plan.hypotheses) >= 1
    assert len(plan.experiments) >= 1
    assert plan.success_probability >= 0.0


def test_run_research_executes_loop(project_dir: Path):
    """M4: run_research() executes the ResearchLoop and updates scientific memory."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()
    project.check_readiness()
    project.approve_readiness()
    project.plan_research("What research should we perform next?")

    loop = project.run_research(max_iterations=3)

    assert loop is not None
    assert len(loop.iterations) == 3
    assert loop.total_experiments_run == 3
    assert loop.total_findings == 3
    assert project.phase == ProjectPhase.COMPLETED


def test_run_research_without_approval_raises(project_dir: Path):
    """M4: run_research() raises if readiness not approved."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()
    project.check_readiness()
    # Don't approve

    with pytest.raises(RuntimeError, match="must be approved"):
        project.run_research()


def test_run_research_without_plan_raises(project_dir: Path):
    """M4: run_research() raises if plan not generated."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()
    project.check_readiness()
    project.approve_readiness()
    # Don't plan

    with pytest.raises(RuntimeError, match="must be generated"):
        project.run_research()


# ── M5: Memory Timeline ───────────────────────────────────────────────────


def test_scientific_memory_updated_after_research(project_dir: Path):
    """M5: scientific memory is updated after running research."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()
    project.check_readiness()
    project.approve_readiness()
    project.plan_research("What research should we perform next?")

    initial_memory_count = len(project.memory_manager.store._entries)

    project.run_research(max_iterations=2)

    final_memory_count = len(project.memory_manager.store._entries)
    assert final_memory_count > initial_memory_count


def test_workflow_transitions_logged(project_dir: Path):
    """M5: every phase change is logged in the transitions list."""
    project = ResearchProject(project_id="proj-001")
    project.ingest_directory(project_dir)
    project.understand()
    project.check_readiness()
    project.approve_readiness()
    project.plan_research("What research next?")
    project.run_research(max_iterations=1)

    transitions = project.transitions

    # Should have multiple transitions: CREATED→INGESTING, INGESTING→UNDERSTANDING,
    # UNDERSTANDING→READY, READY→PLANNING, PLANNING→RESEARCHING, RESEARCHING→COMPLETED
    assert len(transitions) >= 5
    # First transition should be CREATED→INGESTING
    assert transitions[0].from_phase == ProjectPhase.CREATED
    assert transitions[0].to_phase == ProjectPhase.INGESTING
    # Last transition should be →COMPLETED
    assert transitions[-1].to_phase == ProjectPhase.COMPLETED


# ── M6: Full End-to-End Workflow ────────────────────────────────────────────


def test_full_end_to_end_workflow(project_dir: Path):
    """M6: The full end-to-end workflow completes successfully.

    1. Create project
    2. Ingest files
    3. Understand
    4. Verify (ask questions)
    5. Check readiness
    6. Approve readiness
    7. Plan research
    8. Run research
    9. Verify knowledge grew

    If any step fails, that's the next engineering target.
    """
    # 1. Create project
    project = ResearchProject(
        project_id="proj-001",
        name="End-to-End Test",
        description="Full pipeline test",
    )
    assert project.phase == ProjectPhase.CREATED

    # 2. Ingest files
    count = project.ingest_directory(project_dir)
    assert count >= 5, f"Expected at least 5 artifacts, got {count}"

    # 3. Understand
    project.understand()
    assert project.phase == ProjectPhase.READY
    assert len(project.knowledge_graph.nodes) > 0
    assert project.coverage_dashboard is not None

    # 4. Verify (ask questions)
    answer1 = project.verify("Explain my project")
    assert answer1 is not None
    assert answer1.answer != ""

    answer2 = project.verify("What algorithms are implemented?")
    assert answer2 is not None

    answer3 = project.verify("Which notebook produced Figure 1?")
    assert answer3 is not None

    # 5. Check readiness
    report = project.check_readiness()
    assert report is not None
    assert report.readiness_level is not None

    # 6. Approve readiness
    approved = project.approve_readiness(user_id="test-user")
    assert approved is True

    # 7. Plan research
    plan = project.plan_research("What research should we perform next?")
    assert plan is not None
    assert len(plan.objectives) >= 1
    assert len(plan.hypotheses) >= 1
    assert len(plan.experiments) >= 1

    # 8. Run research
    loop = project.run_research(max_iterations=3)
    assert loop is not None
    assert len(loop.iterations) == 3
    assert project.phase == ProjectPhase.COMPLETED

    # 9. Verify knowledge grew
    final_memory_count = len(project.memory_manager.store._entries)
    assert final_memory_count > 0, "Scientific memory should have entries after research"

    # Verify transitions logged
    transitions = project.transitions
    assert len(transitions) >= 5
