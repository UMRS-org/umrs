"""Demo 08 operational exit criteria for RC1 Phase 1.

Proves one real iteration where:
  - hypothesis / finding cite batch-3 lab numbers (85% / 4.21)
  - experiment runs through a durable workflow lifecycle (non-null durable_job_id)
  - restart reconciliation can resume against an alive workflow
  - Finding nodes + SUPPORTS edges land in the knowledge graph
"""

from __future__ import annotations

from pathlib import Path

from r4_contracts.research import ResearchRunStatus
from umrs.knowledge.execution_port import FakeWorkflowExecutionPort
from umrs.knowledge.project import ProjectPhase, ResearchProject
from umrs.knowledge.run_store import ResearchRunStore
from umrs.knowledge.graph_types import RelationType


import pytest



@pytest.fixture(autouse=True)
def _graphs_always_in_memory(monkeypatch):
    """Premise tests exercise pipeline semantics, never PG durability.

    The adapter/ResearchProject constructors call default_view() internally,
    which resolves a live pool when a knowledge DSN exists — then sync-context
    mutations (`asyncio.run` per write) crash cross-loop against the
    loop-bound pool (CI self-hosted class 2026-09-13). Pin the resolution to
    in-memory; the pg-backed path has its own coverage in
    tests/test_research_pg_store_factories.py.
    """
    from umrs.knowledge import canonical_graph_view as _cgv

    monkeypatch.setattr(
        _cgv,
        "default_view",
        lambda *args, **kwargs: _cgv.in_memory_view(*args, **kwargs),
    )



from blank_project_fixture import materialize_blank_project  # noqa: E402


def _demo08_project(tmp_path: Path) -> ResearchProject:
    fixture = materialize_blank_project(tmp_path / "blank")
    project = ResearchProject(
        project_id="demo08",
        name="Demo 08",
        storage_path=tmp_path / "demo08-store",
    )
    project.ingest_directory(fixture)
    project.understand()
    project.check_readiness()
    project.approve_readiness(user_id="test")
    project.plan_research("Link YieldPredictor to batch-3 lab yields")
    assert project.research_plan is not None
    for exp in project.research_plan.experiments:
        exp.metadata = dict(exp.metadata or {})
        exp.metadata["workflow_name"] = "demo08_yield_eval"
    return project


def test_demo08_real_iteration_cites_batch3_and_durable_job(tmp_path: Path) -> None:
    project = _demo08_project(tmp_path)
    port = FakeWorkflowExecutionPort(durable_job_id="job-demo08-durable")
    project.set_workflow_port(port)

    loop = project.run_research(max_iterations=1, simulate=False)
    assert loop.iterations
    iteration = loop.iterations[0]
    assert iteration.simulated is False
    assert iteration.durable_job_id == "job-demo08-durable"
    assert iteration.workflow_run_id
    assert iteration.analysis
    assert "85" in iteration.analysis or "85.0" in iteration.analysis
    assert "4.21" in iteration.analysis
    assert "supports" in iteration.evaluation.lower()

    findings = iteration.findings
    assert findings
    finding = findings[0]
    statement = finding.statement
    assert "85" in statement
    assert "4.21" in statement
    assert finding.durable_job_id == "job-demo08-durable"
    assert finding.evidence_refs
    assert finding.simulated is False

    # Per-iteration persistence (not only post-run)
    assert project.run_store is not None
    runs = project.run_store.list_runs(project.project_id)
    assert runs
    assert runs[-1].status == ResearchRunStatus.COMPLETED
    latest = project.run_store._latest_iteration(runs[-1].run_id)
    assert latest is not None
    assert latest["durable_job_id"] == "job-demo08-durable"
    assert latest["status"] == "completed"

    # Graph: Finding node + SUPPORTS → hypothesis
    graph = project.umrs_knowledge_graph.graph
    node = graph.get_node(finding.finding_id)
    assert node is not None
    assert node.node_type == "finding"
    related = graph.get_related(finding.finding_id, RelationType.SUPPORTS)
    assert finding.hypothesis_id in related

    assert project.phase == ProjectPhase.COMPLETED
    assert any(c[0] == "trigger" for c in port.calls)
    assert any(c[0] == "wait" for c in port.calls)


def test_demo08_restart_reconcile_keeps_alive_then_completes(tmp_path: Path) -> None:
    """Crash mid-run: reconcile sees alive workflow, then completes on success."""
    store_path = tmp_path / "restart-store"
    store = ResearchRunStore(store_path)
    handle = store.create_run("demo08-restart", simulated=False)
    store.update_status(handle.run_id, ResearchRunStatus.RUNNING, loop_id="loop-x")
    store.record_iteration(
        handle.run_id,
        iteration_number=1,
        workflow_execution_id="exec-alive-1",
        durable_job_id="job-demo08-durable",
        status="running",
        payload={"phase": "experiment_execution"},
        iteration_id="iter-checkpoint-1",
    )

    # Process restart #1 — workflow still running.
    reopened = ResearchRunStore(store_path)
    alive = reopened.reconcile_on_startup(
        status_fn=lambda _eid: {
            "status": "running",
            "durable_job_id": "job-demo08-durable",
        }
    )
    assert len(alive) == 1
    assert alive[0].status == ResearchRunStatus.RUNNING

    # Process restart #2 — durable job finished.
    done = reopened.reconcile_on_startup(
        status_fn=lambda _eid: {
            "status": "succeeded",
            "durable_job_id": "job-demo08-durable",
        }
    )
    assert len(done) == 1
    assert done[0].status == ResearchRunStatus.COMPLETED

    # ResearchProject no longer reconciles on construct — call explicitly at process start.
    project = ResearchProject(
        project_id="demo08-restart",
        storage_path=store_path,
        workflow_port=FakeWorkflowExecutionPort(durable_job_id="job-demo08-durable"),
    )
    assert project.run_store is not None
    assert project.run_store.list_running() == []
    assert project.reconcile_runs() == []


def test_demo08_resume_pending_run(tmp_path: Path) -> None:
    store = ResearchRunStore(tmp_path / "resume")
    handle = store.create_run("demo08", simulated=False)
    assert handle.status == ResearchRunStatus.PENDING
    resumed = store.resume(handle.run_id)
    assert resumed.status == ResearchRunStatus.RUNNING
