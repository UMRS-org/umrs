"""Phase 0/1 simulation gate: stamp + real-mode provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from umrs.knowledge.execution_port import FakeWorkflowExecutionPort
from umrs.knowledge.loop import AutonomousResearchLoop, LoopPhase
from umrs.knowledge.project import ProjectPhase, ResearchProject


def _approved_project(tmp_path: Path) -> ResearchProject:
    project = ResearchProject(
        project_id="sim-gate",
        name="Simulation Gate",
        storage_path=tmp_path / "sim-gate",
    )
    papers = tmp_path / "papers"
    papers.mkdir()
    (papers / "note.md").write_text(
        "# Yield study\n\nBatch-3 yield was 85% at 4.21.\n",
        encoding="utf-8",
    )
    project.ingest_directory(papers)
    project.understand()
    project.check_readiness()
    project.approve_readiness(user_id="test")
    project.plan_research("Link yield model to lab data")
    return project


def test_run_research_real_mode_requires_workflow_port(tmp_path: Path) -> None:
    """Real mode without a port fails with a clear error (not the old refusal)."""
    project = _approved_project(tmp_path)
    with pytest.raises(RuntimeError, match="workflow execution port required"):
        project.run_research(max_iterations=1)


def test_run_research_simulate_stamps_iterations_and_findings(tmp_path: Path) -> None:
    """Required: --simulate / simulate=True stamps every iteration and finding."""
    project = _approved_project(tmp_path)
    loop = project.run_research(max_iterations=2, simulate=True)

    assert loop.metadata.get("simulated") is True
    assert len(loop.iterations) == 2
    for iteration in loop.iterations:
        assert iteration.simulated is True
        data = iteration.to_dict()
        assert data["simulated"] is True
        assert data["findings"]
        for finding in data["findings"]:
            assert isinstance(finding, dict)
            assert finding.get("simulated") is True
            assert "statement" in finding
            assert finding.get("finding_id")


def test_loop_engine_simulated_flag_propagates() -> None:
    engine = AutonomousResearchLoop(simulated=True)
    loop = engine.start_loop("proj", "goal", max_iterations=1)
    assert loop.metadata.get("simulated") is True
    engine.run(loop)
    assert loop.iterations[0].simulated is True
    assert loop.iterations[0].phase == LoopPhase.MEMORY_UPDATE


def test_project_run_findings_trace_to_real_records(tmp_path: Path) -> None:
    """Real mode with FakeWorkflowExecutionPort emits provenance + durable_job_id."""
    project = _approved_project(tmp_path)
    port = FakeWorkflowExecutionPort(durable_job_id="job-demo08-durable")
    project.set_workflow_port(port)

    loop = project.run_research(max_iterations=1, simulate=False)
    assert loop.iterations, "expected at least one iteration"

    iteration = loop.iterations[0]
    assert iteration.simulated is False
    assert iteration.workflow_run_id
    assert iteration.durable_job_id == "job-demo08-durable"

    findings = iteration.to_dict()["findings"]
    assert findings, "expected findings"

    for finding in findings:
        assert isinstance(finding, dict)
        assert finding.get("finding_id")
        assert finding.get("hypothesis_id")
        assert finding.get("evidence_refs")
        assert finding.get("workflow_run_id")
        assert finding.get("durable_job_id") == "job-demo08-durable"
        assert finding.get("simulated") is False

    assert any(c[0] == "trigger" for c in port.calls)
    assert project.phase == ProjectPhase.COMPLETED


def test_run_research_failure_maps_to_failed_phase(tmp_path: Path) -> None:
    """Workflow failures must not report project COMPLETED."""
    project = _approved_project(tmp_path)

    class BoomPort(FakeWorkflowExecutionPort):
        def trigger(self, workflow_name: str, params: dict) -> str:
            raise RuntimeError("workflow boom")

    project.set_workflow_port(BoomPort())
    loop = project.run_research(max_iterations=1, simulate=False)
    assert loop.status.value == "error"
    assert project.phase == ProjectPhase.FAILED


def test_run_research_failed_workflow_status_maps_to_failed_phase(
    tmp_path: Path,
) -> None:
    """Terminal workflow status=failed must not publish findings as COMPLETED."""
    project = _approved_project(tmp_path)
    port = FakeWorkflowExecutionPort(
        wait_payload={
            "execution_id": "exec-failed",
            "status": "failed",
            "durable_job_id": "job-failed",
            "error": "SlurmStepExecutor requires a container image",
        },
        results_payload={
            "execution_id": "exec-failed",
            "status": "failed",
            "durable_job_id": "job-failed",
            "error": "SlurmStepExecutor requires a container image",
            "outputs": {},
            "metrics": {},
            "artifacts": [],
        },
    )
    project.set_workflow_port(port)
    loop = project.run_research(max_iterations=1, simulate=False)
    assert loop.status.value == "error"
    assert project.phase == ProjectPhase.FAILED
    assert loop.iterations == [] or all(
        not (it.findings if hasattr(it, "findings") else []) for it in loop.iterations
    )


def test_execute_experiment_keeps_loop_project_id(tmp_path: Path) -> None:
    """experiment.parameters must not overwrite the loop project_id."""
    project = _approved_project(tmp_path)
    # Poison plan experiment parameters with JWT/profile default.
    assert project.research_plan is not None
    for exp in project.research_plan.experiments:
        exp.parameters = dict(exp.parameters or {})
        exp.parameters["project_id"] = "default"

    seen: list[dict] = []

    class CapturePort(FakeWorkflowExecutionPort):
        def trigger(self, workflow_name: str, params: dict) -> str:
            seen.append(dict(params))
            return super().trigger(workflow_name, params)

    project.set_workflow_port(CapturePort())
    project.run_research(max_iterations=1, simulate=False)
    assert seen
    assert seen[0].get("project_id") == "sim-gate"


def test_retrieval_failure_fails_run_instead_of_publishing(tmp_path: Path) -> None:
    """A retrieval error must fail the run, not yield an unevidenced finding."""
    project = _approved_project(tmp_path)
    project.set_workflow_port(FakeWorkflowExecutionPort())

    class BrokenStore:
        def search_artifacts(self, query):
            raise RuntimeError("catalog unreachable")

        def list_artifacts(self):
            raise RuntimeError("catalog unreachable")

        def get_artifact(self, artifact_id):
            raise RuntimeError("catalog unreachable")

    project._knowledge_store = BrokenStore()

    loop = project.run_research(max_iterations=1, simulate=False)
    assert loop.status.value == "error"
    assert "catalog unreachable" in loop.metadata.get("error", "")
    assert project.phase == ProjectPhase.FAILED
    assert all(not it.findings for it in loop.iterations)


def test_real_mode_refuses_unevidenced_finding(tmp_path: Path) -> None:
    """Empty retrieval is not an error, but it must not produce a finding."""
    project = _approved_project(tmp_path)
    project.set_workflow_port(FakeWorkflowExecutionPort())

    class EmptyStore:
        def search_artifacts(self, query):
            return []

        def list_artifacts(self):
            return []

        def get_artifact(self, artifact_id):
            return None

    project._knowledge_store = EmptyStore()
    project.research_plan.source_knowledge_ids = []

    loop = project.run_research(max_iterations=1, simulate=False)
    assert loop.status.value == "error"
    assert "no evidence" in loop.metadata.get("error", "")
    assert project.phase == ProjectPhase.FAILED
