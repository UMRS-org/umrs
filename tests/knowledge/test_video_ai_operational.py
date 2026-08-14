"""Operational test: research experiment → explicit Video AI workflow_name."""

from __future__ import annotations

from pathlib import Path

from umrs.knowledge.execution_port import FakeWorkflowExecutionPort
from umrs.knowledge.project import ProjectPhase, ResearchProject


def _video_project(tmp_path: Path) -> ResearchProject:
    project = ResearchProject(
        project_id="video-ai",
        name="Video AI",
        storage_path=tmp_path / "video-ai",
    )
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "note.md").write_text(
        "# Mouse tracking\n\nYOLO detections on sample video.\n",
        encoding="utf-8",
    )
    project.ingest_directory(papers)
    project.understand()
    project.check_readiness()
    project.approve_readiness(user_id="test")
    project.plan_research("Run YOLO detection on staged videos")
    assert project.research_plan is not None
    for exp in project.research_plan.experiments:
        exp.metadata = dict(exp.metadata or {})
        exp.metadata["workflow_name"] = "video-ai-yolo"
        exp.parameters = dict(exp.parameters or {})
        exp.parameters.update(
            {
                "inventory_key": "data/video_inventory.csv",
                "analysis_plan_key": "data/analysis_plan.csv",
                "video_keys": ["videos/sample.mp4"],
                "runtime": {"mode": "mock"},
            }
        )
    return project


def test_video_ai_explicit_workflow_name_and_durable_job(tmp_path: Path) -> None:
    project = _video_project(tmp_path)
    port = FakeWorkflowExecutionPort(durable_job_id="job-video-ai")
    project.set_workflow_port(port)

    loop = project.run_research(max_iterations=1, simulate=False)
    assert loop.iterations
    iteration = loop.iterations[0]
    assert iteration.durable_job_id == "job-video-ai"
    assert iteration.workflow_run_id
    trigger = next(c for c in port.calls if c[0] == "trigger")
    assert trigger[1][0] == "video-ai-yolo"
    params = trigger[1][1]
    assert params["inventory_key"] == "data/video_inventory.csv"
    assert params["analysis_plan_key"] == "data/analysis_plan.csv"
    assert project.phase == ProjectPhase.COMPLETED
