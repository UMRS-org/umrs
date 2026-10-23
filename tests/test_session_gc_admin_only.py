"""session_gc is an operator job: admin-only on both submit seams (review 2026-10-23).

A session_gc job archives every non-archived research session older than
``spec.max_days`` in the shared umrs_db — every project's, every user's
(``SessionStore.archive_aged_sessions`` has no project or user filter), and
all fronts share that database. Once the lane was wired (d0608ae2), any
authenticated non-admin could POST /gw/jobs {"job_type": "session_gc",
"spec": {"max_days": 1}} (or JOB_SUBMIT it over ZMQ) and archive every
tenant's sessions. Both seams now refuse it unless the verified caller is an
admin — the parent job and every fan-out child.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.jobs.models import JobType

from r4_contracts.handshake_hmac import sign_claims

SECRET = "session-gc-admin-only-secret"
GC_SPEC = {"max_days": 1}


def _signed(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _signed("alice", "proj-a")
ADMIN = _signed("ops", "proj-a", admin=True)


class _JobStore:
    _get_conn = None  # no catalog autopin

    def __init__(self):
        self.created: list[dict] = []

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return "job-1"

    create_fan_out_graph = create_job

    def enqueue_job(self, _job_id):
        pass


@pytest.fixture
def jobs(monkeypatch):
    from umrs.api.jobs import get_job_store
    from umrs.api.jobs import router as jobs_router

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    store = _JobStore()
    app = FastAPI()
    app.include_router(jobs_router)
    app.dependency_overrides[get_job_store] = lambda: store
    return TestClient(app), store


def _body(fan_out: bool) -> dict:
    body = {
        "project_id": "proj-a",
        "user_id": "alice",
        "job_type": "session_gc",
        "spec": GC_SPEC,
        "enqueue_immediately": False,
    }
    if fan_out:  # a harmless parent; the child is the operator job
        body["job_type"] = "cleanup"
        body["spec"] = {}
        body["fan_out_jobs"] = [{"name": "gc", "job_type": "session_gc", "spec": GC_SPEC}]
    return body


@pytest.mark.parametrize("fan_out", [False, True], ids=["parent", "fan-out-child"])
def test_rest_refuses_session_gc_from_a_non_admin(jobs, fan_out):
    client, store = jobs
    r = client.post("/umrs/jobs", json=_body(fan_out), headers=ALICE)
    assert r.status_code == 403, r.text
    assert "session_gc" in r.text and "admin" in r.text
    assert store.created == []


@pytest.mark.parametrize("fan_out", [False, True], ids=["parent", "fan-out-child"])
def test_rest_accepts_session_gc_from_an_admin(jobs, fan_out):
    client, store = jobs
    r = client.post("/umrs/jobs", json=_body(fan_out), headers=ADMIN)
    assert r.status_code == 201, r.text
    assert len(store.created) == 1


def _zmq_backend():
    from umrs.server.job_tracking_handler import DurableJobCommandBackend

    store = MagicMock()
    store.create_job.return_value = "job-1"
    store.get_job.return_value = SimpleNamespace(
        job_id="job-1",
        user_id="ops",
        job_type=JobType.SESSION_GC,
        to_dict=lambda: {"job_id": "job-1"},
    )
    store.get_attempts_for_job.return_value = []
    return DurableJobCommandBackend(store), store


def test_the_shared_refusal_makes_session_gc_admin_only():
    """The rule both seams call (HTTP: _enforce_spec_resources per target;
    ZMQ: _raise_for_spec_resources)."""
    from umrs.server.job_tracking_handler import spec_resource_refusal

    refusal = spec_resource_refusal(JobType.SESSION_GC, GC_SPEC, "alice", False)
    assert refusal and "session_gc" in refusal and "admin" in refusal
    assert spec_resource_refusal(JobType.SESSION_GC, GC_SPEC, "ops", True) is None


@pytest.mark.parametrize("admin", [False, True], ids=["user", "admin"])
def test_zmq_never_creates_a_session_gc_job(admin):
    """JOB_SUBMIT admits only VALID_JOB_TYPES (session_gc is not one) and
    never creates fan-out children; the shared refusal above stands behind
    that allowlist should it ever grow."""
    backend, store = _zmq_backend()
    payload = json.dumps({"project_id": "p", "job_type": "session_gc", "spec": GC_SPEC})
    out = json.loads(backend.submit("alice", payload, is_admin=admin))
    assert out["status"] == "error" and "session_gc" in out["message"]
    store.create_job.assert_not_called()


def test_zmq_applies_the_shared_refusal_to_an_admitted_type(monkeypatch):
    from umrs.server import job_tracking_handler as jth

    monkeypatch.setattr(jth, "VALID_JOB_TYPES", {*jth.VALID_JOB_TYPES, "session_gc"})
    backend, store = _zmq_backend()
    payload = json.dumps({"project_id": "p", "job_type": "session_gc", "spec": GC_SPEC})
    out = json.loads(backend.submit("alice", payload))
    assert out["status"] == "error" and "admin" in out["message"]
    store.create_job.assert_not_called()
    assert json.loads(backend.submit("ops", payload, is_admin=True))["status"] == "ok"
