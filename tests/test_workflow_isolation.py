"""Cross-tenant workflow execution isolation tests (HTTP + ZMQ)."""

from __future__ import annotations

import importlib
import json
from datetime import datetime
from datetime import timezone as _tz
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.jobs.models import Job, JobStatus, JobType, RestartPolicy
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_registry_handler import (
    _resolve_workflow_catalog_entry,
    handle_workflow_command,
    register_workflow,
)

jobs_api = importlib.import_module("umrs.api.jobs")


def _auth_headers(
    user: str = "alice", project: str = "project-a", admin: bool = False
) -> dict[str, str]:
    """Gateway-equivalent signed identity headers.

    Wave-6 F1 sealed workflow routes behind the HMAC trust gate under strict
    mode; these tests exercise the strict paths, so emit the same headers the
    real gateway sidecar would.
    """
    import os

    from r4_contracts.handshake_hmac import sign_claims

    secret = os.environ.get("UMRS_HANDSHAKE_HMAC_SECRET", "")
    headers = {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
    }
    if secret:
        headers["X-Verified-Is-Admin"] = "true" if admin else "false"
        headers["X-Verified-HMAC"] = sign_claims(secret, user, project, admin)
    return headers


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


@pytest.fixture(autouse=True)
def _handshake_secret(monkeypatch):
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
    yield


@pytest.fixture
def tmp_catalog(tmp_path):
    catalog_file = str(tmp_path / "workflow_catalog.json")
    with patch(
        "umrs.server.workflow_registry_handler._catalog_path",
        return_value=catalog_file,
    ):
        yield catalog_file


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


def _workflow_job(*, user="alice", project="project-a", execution_id="wf-durable"):
    return Job(
        job_id="job-001",
        project_id=project,
        user_id=user,
        job_type=JobType.WORKFLOW_RUN,
        spec={
            "workflow_name": "wf",
            "execution_id": execution_id,
            "params": {"project_id": project},
        },
        idempotency_key="idem",
        status=JobStatus.QUEUED,
        restart_policy=RestartPolicy.ON_CRASH,
        created_at=datetime.now(_tz.utc),
        updated_at=datetime.now(_tz.utc),
        last_state_change_at=datetime.now(_tz.utc),
    )


def test_http_status_denies_cross_tenant(client):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )

    response = client.get(
        f"/umrs/workflows/executions/{record.execution_id}",
        headers=_auth_headers(user="bob", project="project-b"),
    )
    assert response.status_code == 403
    body = response.json()["detail"]
    assert body["code"] == "workflow_access_denied"


def test_http_status_allows_owner(client):
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


def test_http_results_denies_unverified_identity(client):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="unverified",
    )

    response = client.get(
        f"/umrs/workflows/executions/{record.execution_id}/results",
        headers=_auth_headers(),
    )
    assert response.status_code == 403


def test_http_cancel_denies_cross_project(client):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )

    response = client.post(
        f"/umrs/workflows/executions/{record.execution_id}/cancel",
        headers=_auth_headers(user="alice", project="project-b"),
    )
    assert response.status_code == 403


def test_http_durable_job_cross_tenant_denied(client, monkeypatch):
    store = FakeJobStore(_workflow_job())
    monkeypatch.setattr(jobs_api, "_job_store", store)

    response = client.get(
        "/umrs/workflows/executions/wf-durable",
        headers=_auth_headers(user="bob", project="project-b"),
    )
    assert response.status_code == 403


def test_zmq_status_denies_cross_tenant():
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )
    raw = handle_workflow_command(
        TaskCmd.WORKFLOW_STATUS,
        "bob",
        json.dumps(
            {
                "execution_id": record.execution_id,
                "project_id": "project-b",
                "_identity_verified": True,
            }
        ),
    )
    data = json.loads(raw)
    assert data["status"] == "error"
    assert data["code"] == "workflow_access_denied"


def test_zmq_results_allows_owner():
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )
    raw = handle_workflow_command(
        TaskCmd.WORKFLOW_RESULTS,
        "alice",
        json.dumps(
            {
                "execution_id": record.execution_id,
                "project_id": "project-a",
                "_identity_verified": True,
            }
        ),
    )
    data = json.loads(raw)
    assert data["status"] == "ok"


def test_zmq_logs_denies_cross_tenant():
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )
    raw = handle_workflow_command(
        TaskCmd.WORKFLOW_LOGS,
        "bob",
        json.dumps(
            {
                "execution_id": record.execution_id,
                "project_id": "project-b",
                "_identity_verified": True,
            }
        ),
    )
    data = json.loads(raw)
    assert data["status"] == "error"
    assert data["code"] == "workflow_access_denied"


def test_strict_mode_denies_mvp_claim_provenance(client, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="mvp_claim",
    )
    response = client.get(
        f"/umrs/workflows/executions/{record.execution_id}",
        headers=_auth_headers(),
    )
    assert response.status_code == 403


def test_zmq_cancel_durable_job(client, monkeypatch):
    store = FakeJobStore(_workflow_job())
    monkeypatch.setattr(jobs_api, "_job_store", store)

    raw = handle_workflow_command(
        TaskCmd.WORKFLOW_CANCEL,
        "alice",
        json.dumps(
            {
                "execution_id": "wf-durable",
                "project_id": "project-a",
                "_identity_verified": True,
            }
        ),
    )
    data = json.loads(raw)
    assert data["status"] == "ok"
    assert store.cancelled is True
    assert data["execution"]["status"] == "cancelled"


@pytest.mark.parametrize(
    "cmd",
    [TaskCmd.WORKFLOW_STATUS, TaskCmd.WORKFLOW_RESULTS, TaskCmd.WORKFLOW_CANCEL],
)
def test_http_zmq_strict_parity_cross_tenant_denied(client, monkeypatch, cmd):
    """HTTP and ZMQ return the same denial code under strict auth."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )

    if cmd == TaskCmd.WORKFLOW_STATUS:
        http_resp = client.get(
            f"/umrs/workflows/executions/{record.execution_id}",
            headers=_auth_headers(user="bob", project="project-b"),
        )
    elif cmd == TaskCmd.WORKFLOW_RESULTS:
        http_resp = client.get(
            f"/umrs/workflows/executions/{record.execution_id}/results",
            headers=_auth_headers(user="bob", project="project-b"),
        )
    else:
        http_resp = client.post(
            f"/umrs/workflows/executions/{record.execution_id}/cancel",
            headers=_auth_headers(user="bob", project="project-b"),
        )
    assert http_resp.status_code == 403
    assert http_resp.json()["detail"]["code"] == "workflow_access_denied"

    zmq_raw = handle_workflow_command(
        cmd,
        "bob",
        json.dumps(
            {
                "execution_id": record.execution_id,
                "project_id": "project-b",
                "_identity_verified": True,
            }
        ),
    )
    zmq_data = json.loads(zmq_raw)
    assert zmq_data["status"] == "error"
    assert zmq_data["code"] == "workflow_access_denied"


def test_zmq_strict_denies_unverified_identity(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="wf",
        user_id="alice",
        params={"project_id": "project-a"},
        identity_provenance="gateway_verified",
    )
    raw = handle_workflow_command(
        TaskCmd.WORKFLOW_STATUS,
        "alice",
        json.dumps(
            {
                "execution_id": record.execution_id,
                "project_id": "project-a",
                "_identity_verified": False,
            }
        ),
    )
    data = json.loads(raw)
    assert data["status"] == "error"
    assert data["code"] in (
        "workflow_access_denied",
        "workflow_identity_unverified",
        "unauthorized",
    )


def test_workflow_name_resolution_prefers_owner_then_system(tmp_catalog, tmp_path):
    alice_dir = tmp_path / "alice-wf"
    alice_dir.mkdir()
    (alice_dir / "workflow.json").write_text(
        json.dumps(
            {
                "name": "shared-name",
                "version": "1.0",
                "steps": [{"name": "s1", "entrypoint": "m:S"}],
            }
        )
    )
    bob_dir = tmp_path / "bob-wf"
    bob_dir.mkdir()
    (bob_dir / "workflow.json").write_text(
        json.dumps(
            {
                "name": "shared-name",
                "version": "1.0",
                "steps": [{"name": "s1", "entrypoint": "m:S"}],
            }
        )
    )
    system_dir = tmp_path / "system-wf"
    system_dir.mkdir()
    (system_dir / "workflow.json").write_text(
        json.dumps(
            {
                "name": "shared-name",
                "version": "1.0",
                "steps": [{"name": "s1", "entrypoint": "m:S"}],
            }
        )
    )

    register_workflow(
        workflow_name="shared-name",
        version="1.0",
        user_id="bob",
        source_path=str(bob_dir),
        workflow_json=json.loads((bob_dir / "workflow.json").read_text()),
        scope="user",
    )
    register_workflow(
        workflow_name="shared-name",
        version="1.0",
        user_id="system",
        source_path=str(system_dir),
        workflow_json=json.loads((system_dir / "workflow.json").read_text()),
        scope="system",
    )

    entry = _resolve_workflow_catalog_entry("shared-name", "alice")
    assert entry is not None
    assert entry["scope"] == "system"

    register_workflow(
        workflow_name="shared-name",
        version="1.0",
        user_id="alice",
        source_path=str(alice_dir),
        workflow_json=json.loads((alice_dir / "workflow.json").read_text()),
        scope="user",
    )
    entry = _resolve_workflow_catalog_entry("shared-name", "alice")
    assert entry is not None
    assert entry["user_id"] == "alice"
