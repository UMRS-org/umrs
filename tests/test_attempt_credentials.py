"""Attempt-scoped short-lived credentials (WP-11, G-P5-3)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import jobs as jobs_api
from umrs.api.jobs import router
from umrs.jobs.attempt_credentials import (
    AttemptCredentials,
    key_in_scope,
    mint_attempt_credentials,
    scope_prefix_for,
)
from umrs.jobs.store import JobNotFoundError

NOW = datetime(2026, 9, 7, tzinfo=timezone.utc)


class TestMintRecord:
    def test_scope_uses_durable_attempt_layout(self):
        record = mint_attempt_credentials("job-1", "att-1", 2, now=NOW)
        assert record.scope_prefixes == ("jobs/job-1/attempt_2/",)
        assert record.ttl_sec > 0
        assert "secret" not in record.to_dict()
        assert "access_key" not in record.to_dict()

    def test_expiry_is_now_plus_ttl(self):
        record = mint_attempt_credentials("job-1", "att-1", 1, ttl_sec=900, now=NOW)
        expires = datetime.fromisoformat(record.expires_at)
        assert expires == NOW + timedelta(seconds=900)

    def test_ttl_must_be_positive(self):
        with pytest.raises(ValueError):
            mint_attempt_credentials("job-1", "att-1", 1, ttl_sec=0)

    def test_scope_prefix_layout_matches_convention(self):
        assert scope_prefix_for("job-x", 7) == "jobs/job-x/attempt_7/"


class TestKeyInScope:
    def _record(self, ttl_sec=900, prefixes=()):
        return mint_attempt_credentials(
            "job-1", "att-1", 3, ttl_sec=ttl_sec, extra_prefixes=prefixes, now=NOW
        )

    def test_inside_attempt_prefix(self):
        record = self._record()
        assert key_in_scope("jobs/job-1/attempt_3/out/result.csv", record, now=NOW)

    def test_outside_attempt_prefix_denied(self):
        record = self._record()
        assert not key_in_scope("jobs/job-1/attempt_2/out/result.csv", record, now=NOW)
        assert not key_in_scope("other/path.csv", record, now=NOW)

    def test_sibling_attempt_denied_per_attempt_binding(self):
        record = self._record()
        assert not key_in_scope("jobs/job-1/attempt_4/file.bin", record, now=NOW)

    def test_extra_prefixes_allow_durable_inputs(self):
        record = self._record(prefixes=("jobs/inputs/models/",))
        assert key_in_scope("jobs/inputs/models/qwen.gguf", record, now=NOW)

    def test_expired_record_denies(self):
        record = self._record(ttl_sec=10)
        later = NOW + timedelta(seconds=11)
        assert not key_in_scope("jobs/job-1/attempt_3/out/x", record, now=later)

    def test_leading_slash_tolerated(self):
        record = self._record()
        assert key_in_scope("/jobs/job-1/attempt_3/x", record, now=NOW)

    def test_round_trips_through_metrics_dict(self):
        record = self._record(prefixes=("models/",))
        restored = AttemptCredentials.from_dict(record.to_dict())
        assert restored.scope_prefixes == record.scope_prefixes
        assert restored.expires_at == record.expires_at


class _FakeCaller:
    def __init__(self, user_id="alice", project_id="proj-x", is_admin=False):
        self.user_id = user_id
        self.project_id = project_id
        self.is_admin = is_admin

    def can_access_job(self, job):
        return self.is_admin or job.project_id == self.project_id


class _FakeJob:
    project_id = "proj-x"
    user_id = "alice"
    job_id = "job-1"


class _FakeAttempt:
    def __init__(self, attempt_id, attempt_no):
        self.attempt_id = attempt_id
        self.attempt_no = attempt_no


class _FakeStore:
    def __init__(self):
        self.job = _FakeJob()
        self.attempts = [_FakeAttempt("att-1", 1), _FakeAttempt("att-2", 2)]
        self.creds: dict = {}

    def get_job(self, job_id):
        if job_id != "job-1":
            raise JobNotFoundError(job_id)
        return self.job

    def get_attempts_for_job(self, job_id):
        return self.attempts

    def get_attempt_credentials(self, attempt_id):
        return self.creds.get(attempt_id)

    def record_attempt_credentials(self, attempt_id, credentials):
        self.creds[attempt_id] = credentials


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


class TestCredentialRoutes:
    def test_mint_persists_and_returns_record(self, client, store):
        data = client.post("/umrs/jobs/job-1/attempts/att-1/credentials").json()
        record = data["credentials"]
        assert record["scope_prefixes"] == ["jobs/job-1/attempt_1/"]
        assert record["ttl_sec"] > 0
        assert store.creds["att-1"] is record or store.creds["att-1"] == record

    def test_mint_is_idempotent_while_live(self, client, store):
        first = client.post("/umrs/jobs/job-1/attempts/att-1/credentials").json()
        second = client.post("/umrs/jobs/job-1/attempts/att-1/credentials").json()
        assert first["credentials"]["minted_at"] == second["credentials"]["minted_at"]

    def test_mint_unknown_attempt_404(self, client):
        assert client.post("/umrs/jobs/job-1/attempts/att-nope/credentials").status_code == 404

    def test_mint_outsider_denied(self, store):
        outsider = TestClient(_app(store, caller=_FakeCaller(project_id="proj-y")))
        assert outsider.post("/umrs/jobs/job-1/attempts/att-1/credentials").status_code == 404

    def test_presign_requires_mint_first(self, client):
        resp = client.post(
            "/umrs/jobs/job-1/attempts/att-2/presign",
            json={"key": "jobs/job-1/attempt_2/out/x.csv", "method": "get"},
        )
        assert resp.status_code == 409

    def test_presign_inside_scope(self, client, store, monkeypatch):
        monkeypatch.setattr(
            "umrs.shared.storage.s3_presign.generate_presigned_get",
            lambda client_, bucket, key, ttl=None: f"https://s3/x?k={key}&t={ttl}",
        )
        client.post("/umrs/jobs/job-1/attempts/att-2/credentials")
        resp = client.post(
            "/umrs/jobs/job-1/attempts/att-2/presign",
            json={"key": "jobs/job-1/attempt_2/out/x.csv", "method": "get"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "jobs/job-1/attempt_2/out/x.csv" in body["url"]
        assert body["expires_in_sec"] <= store.creds["att-2"]["ttl_sec"]

    def test_presign_outside_scope_403(self, client):
        client.post("/umrs/jobs/job-1/attempts/att-1/credentials")
        resp = client.post(
            "/umrs/jobs/job-1/attempts/att-1/presign",
            json={"key": "jobs/job-1/attempt_2/out/x.csv", "method": "get"},
        )
        assert resp.status_code == 403

    def test_presign_put_routes_put_generator(self, client, monkeypatch):
        monkeypatch.setattr(
            "umrs.shared.storage.s3_presign.generate_presigned_put",
            lambda client_, bucket, key, ttl=None: f"https://s3/x?put={key}&t={ttl}",
        )
        client.post("/umrs/jobs/job-1/attempts/att-1/credentials")
        resp = client.post(
            "/umrs/jobs/job-1/attempts/att-1/presign",
            json={"key": "jobs/job-1/attempt_1/out/log.txt", "method": "put"},
        )
        assert resp.status_code == 200
        assert "put=" in resp.json()["url"]
