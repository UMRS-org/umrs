"""RC1 Blank Project Test — local E2E without gateway/S3."""

from __future__ import annotations

from pathlib import Path

import importlib.util

from umrs.knowledge.project import ProjectPhase
from umrs.knowledge.workflow import run_project_workflow
from umrs.knowledge_model import KnowledgeObjectType


def _load_materialize_blank_project():
    path = Path(__file__).resolve().with_name("blank_project_fixture.py")
    spec = importlib.util.spec_from_file_location("blank_project_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.materialize_blank_project


materialize_blank_project = _load_materialize_blank_project()


def test_understand_persists_knowledge_model_objects(tmp_path: Path):
    """Structured extraction should be stored as UMRS knowledge objects."""
    from umrs.knowledge.project import ResearchProject

    root = materialize_blank_project(tmp_path / "km-fixture")
    project = ResearchProject(project_id="proj-km")
    project.ingest_directory(root)
    project.understand()

    objects = project.umrs_knowledge_model.list_objects(project_id="proj-km")
    assert len(objects) == len(project.extracted_knowledge)
    assert any(obj.object_type == KnowledgeObjectType.PAPER for obj in objects)


def test_blank_project_workflow_e2e(tmp_path: Path):
    """Blank Project mix: ingest → understand → verify → plan → run."""
    root = materialize_blank_project(tmp_path / "blank-project")

    result = run_project_workflow(
        "blank-rc1",
        root,
        name="Blank Project RC1",
        research_goal="What GCMC MOF research should we perform next?",
        max_iterations=2,
        auto_approve=True, simulate=True,
        verify_questions=[
            "Explain my project",
            "What algorithms are implemented?",
        ],
    )

    assert result.success is True, result.error
    assert result.artifacts_ingested >= 18
    assert result.knowledge_extracted >= 15
    assert result.graph_nodes >= 15
    assert result.research_objectives > 0
    assert result.research_iterations == 2
    assert result.project.phase == ProjectPhase.COMPLETED

    knowledge_objects = result.project.umrs_knowledge_model.list_objects(
        project_id="blank-rc1",
    )
    assert len(knowledge_objects) >= 15
    assert len(result.project.research_plan.source_knowledge_ids) > 0
