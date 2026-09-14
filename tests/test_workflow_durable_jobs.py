"""Durable job bridge tests for workflow executions."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from datetime import timezone as _tz
from unittest.mock import patch

from umrs.core.TaskCmdStatus import TaskCmd
from umrs.jobs.models import Job, JobAttempt, JobStatus, JobType, RestartPolicy
from umrs.jobs.workflow_run import WorkflowRunBackend
from umrs.server.job_tracking_handler import DurableJobCommandBackend
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_registry_handler import handle_workflow_command, register_workflow


class FakeJobStore:
    def __init__(self):
        self.jobs = {}
        self.attempts = {}
        self.enqueued = []

    def create_job(
        self,
        project_id,
        user_id,
        job_type,
        spec,
        **options,
    ):
        job_id = f"job-{len(self.jobs) + 1:03d}"
        job = Job(
            job_id=job_id,
            project_id=project_id,
            user_id=user_id,
            job_type=job_type,
            spec=spec,
            idempotency_key=options.get("idempotency_key") or f"idem-{job_id}",
            status=JobStatus.PLANNED,
            priority=options.get("priority", 0),
            desired_max_attempts=options.get("max_attempts", 3),
            restart_policy=options.get("restart_policy", RestartPolicy.ON_CRASH),
            created_at=datetime.now(_tz.utc),
            updated_at=datetime.now(_tz.utc),
        )
        self.jobs[job_id] = job
        self.attempts[job_id] = []
        return job_id

    def enqueue_job(self, job_id):
        self.enqueued.append(job_id)
        self.jobs[job_id].status = JobStatus.QUEUED

    def get_job(self, job_id):
        return self.jobs[job_id]

    def get_attempts_for_job(self, job_id):
        return self.attempts[job_id]

    def list_jobs(self, project_id=None, status=None, job_type=None, limit=1000, offset=0):
        jobs = list(self.jobs.values())
        if project_id is not None:
            jobs = [job for job in jobs if job.project_id == project_id]
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
        if job_type is not None:
            jobs = [job for job in jobs if job.job_type == job_type]
        return jobs[offset : offset + limit]

    def cancel_job(self, job_id, reason="user_cancelled"):
        self.jobs[job_id].status = JobStatus.CANCELLED


def test_durable_backend_accepts_workflow_run_job_type() -> None:
    store = FakeJobStore()
    backend = DurableJobCommandBackend(store)

    result = json.loads(
        backend.submit(
            "alice",
            json.dumps(
                {
                    "project_id": "project-a",
                    "job_type": "workflow_run",
                    "spec": {
                        "workflow_name": "ct-analysis",
                        "execution_id": "wf-123",
                        "params": {"bucket": "demo"},
                    },
                }
            ),
        )
    )

    assert result["status"] == "ok"
    assert result["job"]["job_type"] == "workflow_run"
    assert result["job"]["status"] == "queued"
    assert store.enqueued == [result["job"]["job_id"]]


def test_workflow_run_spec_requires_execution_id() -> None:
    backend = DurableJobCommandBackend(FakeJobStore())

    result = json.loads(
        backend.submit(
            "alice",
            json.dumps(
                {
                    "project_id": "project-a",
                    "job_type": "workflow_run",
                    "spec": {"workflow_name": "ct-analysis"},
                }
            ),
        )
    )

    assert result["status"] == "error"
    assert "workflow_run spec.execution_id is required" in result["message"]


def test_workflow_run_spec_requires_params_object() -> None:
    backend = DurableJobCommandBackend(FakeJobStore())

    result = json.loads(
        backend.submit(
            "alice",
            json.dumps(
                {
                    "project_id": "project-a",
                    "job_type": "workflow_run",
                    "spec": {
                        "workflow_name": "ct-analysis",
                        "execution_id": "wf-123",
                        "params": ["not", "object"],
                    },
                }
            ),
        )
    )

    assert result["status"] == "error"
    assert "workflow_run spec.params must be an object" in result["message"]


def test_workflow_run_backend_launches_registry_record(tmp_path) -> None:
    reset_execution_registry_for_tests()
    workflow_dir = tmp_path / "ct-analysis"
    workflow_dir.mkdir()
    (workflow_dir / "workflow.json").write_text(
        json.dumps({"name": "ct-analysis", "steps": []}),
        encoding="utf-8",
    )

    backend = WorkflowRunBackend()
    job = Job(
        job_id="job-001",
        project_id="project-a",
        user_id="alice",
        job_type=JobType.WORKFLOW_RUN,
        spec={
            "execution_id": "wf-123",
            "workflow_name": "ct-analysis",
            "params": {"project_id": "project-a"},
            "source_path": str(workflow_dir),
        },
        idempotency_key="idem",
    )
    attempt = JobAttempt(
        attempt_id="att-001",
        job_id="job-001",
        attempt_no=1,
        backend="workflow-run",
    )

    result = _run(backend.submit_job(job, attempt))

    assert result == "workflow_run|execution_id=wf-123"
    record = get_execution_registry().get("wf-123")
    assert record is not None
    assert record.durable_job_id == "job-001"


def test_run_workflow_uses_durable_job_backend_when_available(tmp_path) -> None:
    reset_execution_registry_for_tests()
    workflow_dir = tmp_path / "ct-analysis"
    workflow_dir.mkdir()
    workflow_json = {"name": "ct-analysis", "version": "1.0", "steps": []}
    (workflow_dir / "workflow.json").write_text(json.dumps(workflow_json), encoding="utf-8")
    with patch(
        "umrs.server.workflow_registry_handler._catalog_path",
        return_value=str(tmp_path / "workflow_catalog.json"),
    ):
        register_workflow(
            workflow_name="ct-analysis",
            version="1.0",
            user_id="alice",
            source_path=str(workflow_dir),
            workflow_json=workflow_json,
        )

        store = FakeJobStore()
        backend = DurableJobCommandBackend(store)
        result = json.loads(
            handle_workflow_command(
                TaskCmd.RUN_WORKFLOW,
                "alice",
                json.dumps(
                    {
                        "workflow_name": "ct-analysis",
                        "params": {"project_id": "project-a", "bucket": "demo"},
                    }
                ),
                job_backend=backend,
            )
        )

    assert result["status"] == "ok"
    assert result["durable_job_id"] == "job-001"
    assert result["execution_id"].startswith("wf-")
    assert get_execution_registry().get(result["execution_id"]) is None
    assert store.enqueued == ["job-001"]
    assert store.jobs["job-001"].spec["workflow_name"] == "ct-analysis"


def _run(awaitable):
    return asyncio.run(awaitable)
