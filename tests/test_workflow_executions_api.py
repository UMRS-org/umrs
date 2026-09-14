"""Tests for the new workflow execution status / cancel HTTP routes."""

from __future__ import annotations

import importlib
from datetime import datetime
from datetime import timezone as _tz

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.jobs.models import Job, JobStatus, JobType, RestartPolicy
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)

jobs_api = importlib.import_module("umrs.api.jobs")


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _registry():
    reset_execution_registry_for_tests()
    yield
    reset_execution_registry_for_tests()


class FakeJobStore:
    def __init__(self, job):
        self.job = job
        self.cancelled = False

    def get_job(self, job_id):
        if job_id != self.job.job_id:
            raise KeyError(job_id)
        return self.job

    def list_jobs(self, project_id=None, status=None, job_type=None, limit=1000, offset=0):
        jobs = [self.job]
        if project_id is not None:
            jobs = [job for job in jobs if job.project_id == project_id]
        if status is not None:
            jobs = [job for job in jobs if job.status == status]
        if job_type is not None:
            jobs = [job for job in jobs if job.job_type == job_type]
        return jobs[offset : offset + limit]

    def cancel_job(self, job_id, reason="user_cancelled"):
        self.cancelled = True
        self.job.status = JobStatus.CANCELLED


def _workflow_job(status=JobStatus.QUEUED):
    return Job(
        job_id="job-001",
        project_id="project-a",
        user_id="alice",
        job_type=JobType.WORKFLOW_RUN,
        spec={
            "workflow_name": "wf",
            "execution_id": "wf-durable",
            "params": {"project_id": "project-a"},
        },
        idempotency_key="idem",
        status=status,
        restart_policy=RestartPolicy.ON_CRASH,
        created_at=datetime.now(_tz.utc),
        updated_at=datetime.now(_tz.utc),
        last_state_change_at=datetime.now(_tz.utc),
    )


def _auth_headers(user: str = "alice", project: str = "project-a") -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
    }


def test_executions_status_404_for_unknown(client):
    response = client.get(
        "/umrs/workflows/executions/does-not-exist",
        headers=_auth_headers(),
    )
    assert response.status_code == 404


def test_executions_status_returns_record(client):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )

    response = client.get(
        f"/umrs/workflows/executions/{record.execution_id}",
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["execution"]["execution_id"] == record.execution_id
    assert body["execution"]["status"] == "submitted"


def test_executions_results_includes_artifacts(client):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )
    # Pretend a step produced an artifact.
    record.results.setdefault("blobs", {})["wf-runner/foo.json"] = b"{}"
    record.add_artifact("wf-runner/foo.json")
    record.update(status="succeeded", progress=1.0)

    response = client.get(
        f"/umrs/workflows/executions/{record.execution_id}/results",
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["execution_status"] == "succeeded"
    assert body["artifacts"][0]["key"] == "wf-runner/foo.json"
    assert body["artifacts"][0]["size_bytes"] == 2


def test_cancel_marks_submitted_record_as_cancelled(client):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )
    response = client.post(
        f"/umrs/workflows/executions/{record.execution_id}/cancel",
        headers=_auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["execution"]["status"] == "cancelled"


def test_cancel_unknown_returns_404(client):
    response = client.post(
        "/umrs/workflows/executions/does-not-exist/cancel",
        headers=_auth_headers(),
    )
    assert response.status_code == 404


def test_status_falls_back_to_durable_workflow_job(client, monkeypatch):
    store = FakeJobStore(_workflow_job())
    monkeypatch.setattr(jobs_api, "_job_store", store)

    response = client.get(
        "/umrs/workflows/executions/wf-durable",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    execution = response.json()["execution"]
    assert execution["execution_id"] == "wf-durable"
    assert execution["status"] == "submitted"
    assert execution["durable_job_id"] == "job-001"


def test_cancel_falls_back_to_durable_workflow_job(client, monkeypatch):
    store = FakeJobStore(_workflow_job())
    monkeypatch.setattr(jobs_api, "_job_store", store)

    response = client.post(
        "/umrs/workflows/executions/wf-durable/cancel",
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    assert store.cancelled is True
    assert response.json()["execution"]["status"] == "cancelled"
