"""Tests for ResearchRunStore persistence and lifecycle."""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.knowledge.run_store import ResearchRunStore

from r4_contracts.research import ResearchRunStatus


def test_create_get_and_status_transitions(tmp_path: Path) -> None:
    store = ResearchRunStore(tmp_path / "runs")
    handle = store.create_run("proj-1", simulated=False)
    assert handle.status == ResearchRunStatus.PENDING
    assert store.get_run(handle.run_id) is not None

    running = store.update_status(handle.run_id, ResearchRunStatus.RUNNING, loop_id="loop-1")
    assert running.status == ResearchRunStatus.RUNNING
    assert running.loop_id == "loop-1"

    done = store.update_status(handle.run_id, ResearchRunStatus.COMPLETED)
    assert done.status == ResearchRunStatus.COMPLETED

    with pytest.raises(ValueError, match="illegal status transition"):
        store.update_status(handle.run_id, ResearchRunStatus.RUNNING)


def test_cancel_is_idempotent(tmp_path: Path) -> None:
    store = ResearchRunStore(tmp_path / "runs")
    handle = store.create_run("proj-1")
    store.update_status(handle.run_id, ResearchRunStatus.RUNNING)

    first = store.cancel(handle.run_id)
    assert first.status == ResearchRunStatus.CANCELLED
    second = store.cancel(handle.run_id)
    assert second.status == ResearchRunStatus.CANCELLED
    assert second.run_id == first.run_id


def test_resume_completed_is_noop(tmp_path: Path) -> None:
    store = ResearchRunStore(tmp_path / "runs")
    handle = store.create_run("proj-1")
    store.update_status(handle.run_id, ResearchRunStatus.RUNNING)
    store.update_status(handle.run_id, ResearchRunStatus.COMPLETED)

    resumed = store.resume(handle.run_id)
    assert resumed.status == ResearchRunStatus.COMPLETED


def test_record_iteration_and_list_running(tmp_path: Path) -> None:
    store = ResearchRunStore(tmp_path / "runs")
    a = store.create_run("proj-a")
    b = store.create_run("proj-b")
    store.update_status(a.run_id, ResearchRunStatus.RUNNING)
    store.update_status(b.run_id, ResearchRunStatus.RUNNING)

    iid = store.record_iteration(
        a.run_id,
        iteration_number=1,
        workflow_execution_id="exec-1",
        durable_job_id="job-1",
        status="running",
        payload={"phase": "experiment_execution"},
    )
    assert iid

    running = store.list_running()
    assert {h.run_id for h in running} == {a.run_id, b.run_id}
    assert len(store.list_running(project_id="proj-a")) == 1

    rows = store.list_iterations(a.run_id)
    assert len(rows) == 1
    assert rows[0]["workflow_execution_id"] == "exec-1"
    assert rows[0]["payload"]["phase"] == "experiment_execution"
    assert store.list_iterations(b.run_id) == []


def test_record_iteration_upserts_same_iteration_id(tmp_path: Path) -> None:
    store = ResearchRunStore(tmp_path / "runs")
    handle = store.create_run("proj-1")
    store.update_status(handle.run_id, ResearchRunStatus.RUNNING)

    store.record_iteration(
        handle.run_id,
        iteration_number=1,
        workflow_execution_id="exec-1",
        durable_job_id="job-1",
        status="running",
        payload={"phase": "experiment_execution"},
        iteration_id="iter-fixed",
    )
    store.record_iteration(
        handle.run_id,
        iteration_number=1,
        workflow_execution_id="exec-1",
        durable_job_id="job-1",
        status="completed",
        payload={"phase": "memory_update", "findings": []},
        iteration_id="iter-fixed",
    )
    rows = store._conn.execute(
        "SELECT status, payload_json FROM iterations WHERE iteration_id = ?",
        ("iter-fixed",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert "memory_update" in rows[0]["payload_json"]


def test_reconcile_marks_failed_without_status_fn(tmp_path: Path) -> None:
    store = ResearchRunStore(tmp_path / "runs")
    handle = store.create_run("proj-1")
    store.update_status(handle.run_id, ResearchRunStatus.RUNNING)
    store.record_iteration(
        handle.run_id,
        iteration_number=1,
        workflow_execution_id="exec-dead",
        durable_job_id="job-dead",
    )

    reconciled = store.reconcile_on_startup()
    assert len(reconciled) == 1
    assert reconciled[0].status == ResearchRunStatus.FAILED
    assert "died" in (reconciled[0].error or "")


def test_reconcile_keeps_alive_workflows(tmp_path: Path) -> None:
    store = ResearchRunStore(tmp_path / "runs")
    handle = store.create_run("proj-1")
    store.update_status(handle.run_id, ResearchRunStatus.RUNNING)
    store.record_iteration(
        handle.run_id,
        iteration_number=1,
        workflow_execution_id="exec-alive",
        durable_job_id="job-alive",
    )

    def status_fn(execution_id: str) -> dict:
        assert execution_id == "exec-alive"
        return {"status": "running", "durable_job_id": "job-alive"}

    reconciled = store.reconcile_on_startup(status_fn=status_fn)
    assert reconciled[0].status == ResearchRunStatus.RUNNING
