"""Route tests: job manifests (P7a) + executor-backed reproduce (P7b)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import jobs as jobs_api
from umrs.api.jobs import router
from umrs.jobs.store import JobNotFoundError


class _FakeCaller:
    """CallerIdentity stand-in (owner, admin, or outsider per test)."""

    def __init__(self, user_id="alice", project_id="proj-x", is_admin=False):
        self.user_id = user_id
        self.project_id = project_id
        self.is_admin = is_admin

    def can_access_job(self, job):
        return self.is_admin or job.project_id == self.project_id


class _FakeJob:
    def __init__(self, project_id="proj-x", user_id="alice"):
        self.job_id = "job-1"
        self.project_id = project_id
        self.user_id = user_id
        self.spec = {"workflow_name": "demo08"}
        self.commit_attempt_id = "att-2"


class _FakeAttempt:
    def __init__(self, attempt_id, no=1):
        self.attempt_id = attempt_id
        self.attempt_no = no


class _FakeStore:
    """Store stub: job row + per-attempt manifests + retry counter."""

    def __init__(self):
        self.job = _FakeJob()
        self.manifests = {
            "att-1": {"code_versions": [{"commit_sha": "old"}], "environment": {}},
            "att-2": {"code_versions": [{"commit_sha": "cur"}], "environment": {}},
        }
        self.retried = []

    def get_job(self, job_id):
        if job_id != self.job.job_id:
            raise JobNotFoundError(job_id)
        return self.job

    def get_attempts_for_job(self, job_id):
        return [_FakeAttempt("att-1"), _FakeAttempt("att-2", no=2)]

    def get_attempt_manifest(self, attempt_id):
        return self.manifests.get(attempt_id)

    def get_latest_attempt_manifest(self, job_id):
        # commit attempt att-2 wins over att-1
        return self.manifests.get("att-2")

    def retry_job(self, job_id):
        if job_id != self.job.job_id:
            raise JobNotFoundError(job_id)
        self.retried.append(job_id)


def _app(store, caller=None):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[jobs_api.get_job_store] = lambda: store
    if caller is not None:
        app.dependency_overrides[jobs_api.get_caller_identity] = lambda: caller
    return app


@pytest.fixture()
def store():
    return _FakeStore()


@pytest.fixture()
def client(store):
    return TestClient(_app(store, caller=_FakeCaller()))


class TestManifestRoutes:
    def test_latest_manifest_prefers_commit_attempt(self, client):
        data = client.get("/umrs/jobs/job-1/manifest").json()
        assert data["manifest"]["code_versions"][0]["commit_sha"] == "cur"

    def test_manifests_lists_persisted_attempts(self, client):
        data = client.get("/umrs/jobs/job-1/manifests").json()
        assert data["count"] == 2
        assert {m["attempt_id"] for m in data["manifests"]} == {"att-1", "att-2"}

    def test_manifest_missing_job_404(self, client):
        assert client.get("/umrs/jobs/nope/manifest").status_code == 404

    def test_outsider_denied(self, store):
        client = TestClient(
            _app(store, caller=_FakeCaller(project_id="proj-y"))
        )
        assert client.get("/umrs/jobs/job-1/manifest").status_code == 403


class TestReproduceRoute:
    def test_reproduce_requeues_and_reports_drift(self, client, store):
        data = client.post("/umrs/jobs/job-1/reproduce").json()
        assert data["success"] is True
        assert store.retried == ["job-1"]
        drift = data["drift"]
        assert drift["pinned"] is True
        assert isinstance(drift["pinned"], bool)
        assert "current_code_sha" in drift

    def test_reproduce_without_recorded_code_declares_window(self, store):
        store.manifests = {}
        data = TestClient(
            _app(store, caller=_FakeCaller())
        ).post("/umrs/jobs/job-1/reproduce").json()
        assert data["drift"]["pinned"] is False
        assert data["drift"]["same_code"] is None
        assert "undeclared" in data["drift"]["note"]

    def test_reproduce_missing_job_404(self, client):
        assert client.post("/umrs/jobs/nope/reproduce").status_code == 404

    def test_reproduce_outsider_denied(self, store):
        client = TestClient(
            _app(store, caller=_FakeCaller(project_id="proj-y"))
        )
        assert client.post("/umrs/jobs/job-1/reproduce").status_code == 403
