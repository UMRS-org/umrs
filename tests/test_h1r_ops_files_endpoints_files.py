"""H-1 remaining gaps (hardening plan Appendix A.2, 2026-10-18), files router.

``GET /umrs/files/download-status`` answered any job's status and readiness
by ``?job_id`` with no identity at all (the blocking form polled a foreign
job until its timeout). It now resolves the caller's project
(``get_project_id``: HMAC-verified in strict mode) and a job of another
project reads exactly like a missing one: 404 ``files.job_not_found``
(signed admins excepted). ``POST /umrs/files/presign`` with a ``job_id``
gated a download of the caller's own key on another project's job (409 vs
200 showed its state); a foreign job is 404 there too.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1r-ops-files-endpoints-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _forged(user: str, project: str) -> dict[str, str]:
    """Bogus HMAC, admin flag on, a project the caller is not in."""
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true",
        "X-Verified-HMAC": "0" * 64,
    }


def _status(value: str) -> SimpleNamespace:
    return SimpleNamespace(value=value)


class _FakeJobStore:
    def __init__(self) -> None:
        self.jobs = {
            "job-a": SimpleNamespace(
                job_id="job-a",
                project_id="proj-a",
                status=_status("completed"),
                active_attempt_id=None,
            ),
            "job-b": SimpleNamespace(
                job_id="job-b",
                project_id="proj-b",
                status=_status("completed"),
                active_attempt_id=None,
            ),
            "job-b-running": SimpleNamespace(
                job_id="job-b-running",
                project_id="proj-b",
                status=_status("running"),
                active_attempt_id=None,
            ),
            "job-a-running": SimpleNamespace(
                job_id="job-a-running",
                project_id="proj-a",
                status=_status("running"),
                active_attempt_id=None,
            ),
        }
        self.reads: list[str] = []

    def get_job(self, job_id: str):
        from umrs.jobs.store import JobNotFoundError

        self.reads.append(job_id)
        if job_id not in self.jobs:
            raise JobNotFoundError(job_id)
        return self.jobs[job_id]

    def get_attempt(self, _attempt_id: str):  # pragma: no cover - no attempts here
        raise AssertionError("no attempts in this fake")


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


@pytest.fixture
def store(monkeypatch):
    import umrs.api.jobs as jobs_api

    fake = _FakeJobStore()
    monkeypatch.setattr(jobs_api, "get_optional_job_store", lambda: fake)
    return fake


@pytest.fixture
def no_sleep(monkeypatch):
    import umrs.api.files as files_api

    async def _sleep(_seconds):
        raise AssertionError("a foreign or missing job must not be polled")

    monkeypatch.setattr(files_api.asyncio, "sleep", _sleep)


@pytest.fixture
def client():
    from umrs.api.error_responses import structured_warning_exception_handler
    from umrs.api.files import get_project_s3_config, router
    from umrs.shared.warnings import UMRSWarningError

    app = FastAPI()
    app.include_router(router)
    app.add_exception_handler(UMRSWarningError, structured_warning_exception_handler)
    app.dependency_overrides[get_project_s3_config] = lambda: SimpleNamespace(
        bucket="umrs-default", default_ttl=3600, max_ttl=86400
    )
    return TestClient(app, raise_server_exceptions=False)


# ── download-status ─────────────────────────────────────────────────────────


def test_anonymous_download_status_is_401_in_strict(strict, store, client):
    resp = client.get("/umrs/files/download-status", params={"job_id": "job-a"})
    assert resp.status_code == 401
    assert store.reads == []


def test_forged_tuple_is_refused(strict, store, client):
    resp = client.get(
        "/umrs/files/download-status",
        params={"job_id": "job-b"},
        headers=_forged("alice", "proj-b"),
    )
    assert resp.status_code == 401
    assert store.reads == []


def test_own_job_status_is_readable(strict, store, client):
    resp = client.get(
        "/umrs/files/download-status",
        params={"job_id": "job-a"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ready"] is True


def test_foreign_job_reads_as_not_found(strict, store, client):
    resp = client.get(
        "/umrs/files/download-status",
        params={"job_id": "job-b"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 404
    assert "completed" not in resp.text


def test_missing_job_reads_the_same_as_a_foreign_one(strict, store, client):
    missing = client.get(
        "/umrs/files/download-status",
        params={"job_id": "nope"},
        headers=_headers("alice", "proj-a"),
    )
    foreign = client.get(
        "/umrs/files/download-status",
        params={"job_id": "job-b"},
        headers=_headers("alice", "proj-a"),
    )
    assert missing.status_code == foreign.status_code == 404
    code = "x-platform-error-code"
    assert missing.headers[code] == foreign.headers[code] == "files.job_not_found"


def test_blocking_poll_of_a_foreign_job_is_404_at_once(strict, store, client, no_sleep):
    resp = client.get(
        "/umrs/files/download-status",
        params={"job_id": "job-b-running", "blocking": "true", "timeout": 30},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 404


def test_signed_admin_reads_any_project_job(strict, store, client):
    resp = client.get(
        "/umrs/files/download-status",
        params={"job_id": "job-b"},
        headers=_headers("root", "ops", admin=True),
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["ready"] is True


# ── presign download gated on a job ─────────────────────────────────────────


def _presign(client, job_id: str, headers: dict[str, str]):
    return client.post(
        "/umrs/files/presign",
        json={"operation": "download", "key": "proj-a/out.txt", "job_id": job_id},
        headers=headers,
    )


def test_presign_gated_on_a_foreign_job_is_404(strict, store, client):
    resp = _presign(client, "job-b-running", _headers("alice", "proj-a"))
    assert resp.status_code == 404
    assert resp.headers["x-platform-error-code"] == "files.job_not_found"


def test_presign_gated_on_own_running_job_is_still_409(strict, store, client):
    resp = _presign(client, "job-a-running", _headers("alice", "proj-a"))
    assert resp.status_code == 409
