"""Caller-facing workflow resolution is owner-scoped (ADR-074 D-8).

A ``user``-scope workflow resolves on trigger and describe only for its owner
(admins any); ``system`` stays shared. The cross-owner name fallback let a
normal user run another owner's code against their own buckets. The internal
durable path keeps the ``source_path`` it recorded at trigger time.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.workflows import router
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.server.commands.workflow_commands import RunWorkflowHandler
from umrs.server.workflow_registry_handler import (
    _workflow_run_context,
    handle_workflow_command,
    register_workflow,
    resolve_workflow_for_caller,
)


@pytest.fixture(autouse=True)
def tmp_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "test-secret")
    catalog_file = str(tmp_path / "workflow_catalog.json")
    with patch(
        "umrs.server.workflow_registry_handler._catalog_path",
        return_value=catalog_file,
    ):
        yield catalog_file


def _register(tmp_path, name: str, owner: str, scope: str = "user") -> str:
    source = tmp_path / f"{owner}-{name}-{scope}"
    source.mkdir()
    workflow = {
        "name": name,
        "version": "1.0",
        "steps": [{"name": "s1", "entrypoint": "mod:Step"}],
    }
    (source / "workflow.json").write_text(json.dumps(workflow))
    register_workflow(
        workflow_name=name,
        version="1.0",
        user_id=owner,
        source_path=str(source),
        workflow_json=workflow,
        scope=scope,
    )
    return str(source)


# ── resolution rules ───────────────────────────────────────────────────────


def test_another_owners_user_workflow_does_not_resolve(tmp_path):
    _register(tmp_path, "wf", "bob")
    assert resolve_workflow_for_caller("wf", "alice") is None


def test_admins_resolve_any_owner(tmp_path):
    bob = _register(tmp_path, "wf", "bob")
    entry = resolve_workflow_for_caller("wf", "root", is_admin=True)
    assert entry is not None and entry["source_path"] == bob


def test_system_scope_stays_shared_and_the_owners_own_wins(tmp_path):
    _register(tmp_path, "wf", "bob")
    system = _register(tmp_path, "wf", "platform", scope="system")
    assert resolve_workflow_for_caller("wf", "alice")["source_path"] == system
    alice = _register(tmp_path, "wf", "alice")
    assert resolve_workflow_for_caller("wf", "alice")["source_path"] == alice


# ── RUN_WORKFLOW (HTTP trigger + ZMQ) ──────────────────────────────────────


class _Jobs:
    def __init__(self) -> None:
        self.specs: list[dict] = []

    def submit_workflow_run(self, _client_id, payload):
        self.specs.append(json.loads(payload)["spec"])
        return json.dumps({"status": "ok", "job": {"job_id": "j1"}})


def _run(client_id: str, jobs: _Jobs, **kw) -> dict:
    raw = handle_workflow_command(
        TaskCmd.RUN_WORKFLOW,
        client_id,
        json.dumps({"workflow_name": "wf", "params": {}}),
        job_backend=jobs,
        **kw,
    )
    return json.loads(raw)


def test_run_refuses_another_owners_workflow(tmp_path):
    _register(tmp_path, "wf", "bob")
    jobs = _Jobs()
    out = _run("alice", jobs)
    assert out["status"] == "error" and "not found" in out["message"]
    assert jobs.specs == []


def test_admin_run_records_the_resolved_source_path(tmp_path):
    bob = _register(tmp_path, "wf", "bob")
    jobs = _Jobs()
    out = _run("root", jobs, caller_is_admin=True)
    assert out["status"] == "ok", out
    assert jobs.specs[0]["source_path"] == bob  # the durable job carries it


def test_the_durable_path_keeps_its_source_path_and_resolves_owner_scoped(tmp_path):
    """A durable job's recorded source_path (the server resolved it for the
    caller at trigger time) is used as-is; a job without one resolves like a
    trigger — for its owner only, never another owner's same-named workflow."""
    bob = _register(tmp_path, "wf", "bob")
    path, _, steps = _workflow_run_context("wf", "scheduler", source_path=bob)
    assert path == bob and steps == 1
    with pytest.raises(ValueError, match="not found in registry"):
        _workflow_run_context("wf", "alice")
    assert _workflow_run_context("wf", "bob")[0] == bob


# ── durable workflow_run jobs come only from the server's trigger path ─────


def _foreign_spec(source_path: str | None = None) -> dict:
    spec = {"workflow_name": "wf", "execution_id": "wf-0123456789ab", "params": {}}
    if source_path:
        spec["source_path"] = source_path
    return spec


class _Store:
    def __init__(self) -> None:
        self.created: list[dict] = []
        self.db_pool = None

    def create_job(self, **kw):
        self.created.append(kw)
        return "job-1"

    create_fan_out_graph = create_job

    def enqueue_job(self, _job_id):
        pass


@pytest.mark.parametrize("by_path", [False, True])
def test_zmq_job_submit_refuses_a_workflow_run_job(tmp_path, by_path):
    from umrs.server.job_tracking_handler import (
        JOB_SUBMIT,
        DurableJobCommandBackend,
        handle_job_command,
    )

    bob = _register(tmp_path, "wf", "bob")
    store = _Store()
    payload = {"job_type": "workflow_run", "spec": _foreign_spec(bob if by_path else None)}
    out = json.loads(
        handle_job_command(
            JOB_SUBMIT, "alice", json.dumps(payload), backend=DurableJobCommandBackend(store)
        )
    )
    assert out["status"] == "error" and "never submitted directly" in out["message"]
    assert store.created == []


@pytest.mark.parametrize("by_path", [False, True])
@pytest.mark.parametrize("fan_out", [False, True])
def test_http_job_create_refuses_a_workflow_run_job(tmp_path, by_path, fan_out):
    from umrs.api.jobs import get_job_store
    from umrs.api.jobs import router as jobs_router

    bob = _register(tmp_path, "wf", "bob")
    spec = _foreign_spec(bob if by_path else None)
    body = {"project_id": "project-a", "user_id": "alice", "job_type": "workflow_run",
            "spec": spec}  # fmt: skip
    if fan_out:
        body = {**body, "job_type": "index_build", "spec": {"database": "d", "table": "t"},
                "fan_out_jobs": [{"name": "c", "job_type": "workflow_run", "spec": spec}]}  # fmt: skip
    app = FastAPI()
    app.include_router(jobs_router)
    store = _Store()
    app.dependency_overrides[get_job_store] = lambda: store
    response = TestClient(app).post("/umrs/jobs", json=body, headers=_headers("alice"))
    assert response.status_code == 400, response.text
    assert "never submitted directly" in response.text
    assert store.created == []


def test_a_durable_job_without_a_source_path_never_runs_another_owners_workflow(tmp_path):
    import asyncio

    from umrs.jobs.models import Job, JobAttempt, JobType
    from umrs.jobs.workflow_run import WorkflowRunBackend

    _register(tmp_path, "wf", "bob")
    job = Job(job_id="j", project_id="p", user_id="alice", job_type=JobType.WORKFLOW_RUN,
              spec=_foreign_spec(), idempotency_key="k")  # fmt: skip
    attempt = JobAttempt(attempt_id="a", job_id="j", attempt_no=1, backend="workflow-run")
    with pytest.raises(ValueError, match="not found in registry"):
        asyncio.run(WorkflowRunBackend().submit_job(job, attempt))


@pytest.mark.parametrize(("role_admin", "expected"), [(True, "ok"), (False, "error")])
def test_zmq_admin_comes_from_the_auth_session(tmp_path, role_admin, expected):
    _register(tmp_path, "wf", "bob")
    jobs = _Jobs()
    sent: list[str] = []
    ctx = SimpleNamespace(
        job_command_backend=jobs,
        workflow_scheduler_active=True,
        control_mq=SimpleNamespace(
            get_auth_context=lambda _cid: SimpleNamespace(is_admin=role_admin),
            send_string=lambda msg, _sender: sent.append(msg),
        ),
    )
    jobs_status_updates: list[tuple[int, str]] = []

    class _TasksStatus:
        def create_task(self, cmd, payload):
            return 1

        def update_status(self, task_id, status):
            jobs_status_updates.append((task_id, status))

    ctx.task_manager = SimpleNamespace(tasks_status=_TasksStatus())
    payload = json.dumps({"workflow_name": "wf", "params": {}, "is_admin": True})
    # Handler contract since the ejeannot batch sync (upstream 211790ab):
    # execute(payload, ctx); the handler creates its own task via _make_task.
    RunWorkflowHandler().execute(("root", payload, b"sender"), ctx)
    assert json.loads(sent[0])["status"] == expected


# ── HTTP describe / trigger ────────────────────────────────────────────────


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _headers(user: str, admin: bool = False) -> dict[str, str]:
    from r4_contracts.handshake_hmac import sign_claims

    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": "project-a",
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims("test-secret", user, "project-a", admin),
    }


def test_http_describe_is_owner_scoped(client, tmp_path):
    _register(tmp_path, "wf", "bob")
    assert client.get("/umrs/workflows/wf", headers=_headers("alice")).status_code == 404
    assert client.get("/umrs/workflows/wf", headers=_headers("bob")).status_code == 200
    admin = client.get("/umrs/workflows/wf", headers=_headers("root", admin=True))
    assert admin.status_code == 200


def test_http_trigger_refuses_another_owners_workflow(client, tmp_path):
    _register(tmp_path, "wf", "bob")
    response = client.post(
        "/umrs/workflows/wf/trigger",
        headers=_headers("alice"),
        json={"params": {"project_id": "project-a"}},
    )
    assert response.status_code == 404


def test_http_triggers_list_is_owner_scoped(client, tmp_path):
    _register(tmp_path, "wf", "bob")
    response = client.get("/umrs/workflows/wf/triggers", headers=_headers("alice"))
    assert response.status_code == 404
