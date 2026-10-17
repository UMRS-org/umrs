"""H-1 remaining gaps (hardening plan Appendix A, 2026-10-18), ops router.

- ``GET /umrs/ops/timeline``: the ``@router.get("/timeline")`` decorator sat
  above the helper ``_enum_value``, so the path answered 422/500 with no
  authentication and ``ops_timeline`` (admin-only) was never mounted.
- ``GET /umrs/ops/connect-info``: with ``UMRS_EMIT_S3CFG=1`` a bare
  ``X-User-ID`` returned (and minted) that user's s3cfg. Strict mode now
  emits it only for the user of a gateway-signed identity tuple.
"""

from __future__ import annotations

from datetime import datetime
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


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


@pytest.fixture
def client():
    from umrs.api.ops import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


# ── timeline ────────────────────────────────────────────────────────────────


class _TimelineStore:
    def __init__(self) -> None:
        now = datetime.utcnow()
        self.job = SimpleNamespace(
            job_id="job-1",
            job_type=SimpleNamespace(value="endpoint_provision"),
            status=SimpleNamespace(value="running"),
            created_at=now,
            updated_at=now,
        )

    def list_jobs(self, **_kw):
        return [self.job]

    def get_attempts_for_job(self, _job_id: str):
        return []


@pytest.fixture
def timeline_store(monkeypatch):
    import umrs.api.ops as ops

    store = _TimelineStore()
    monkeypatch.setattr(ops, "_lazy_get_optional_job_store", lambda: store)
    return store


def test_timeline_route_is_bound_to_ops_timeline():
    from umrs.api.ops import ops_timeline, router

    routes = [r for r in router.routes if getattr(r, "path", "") == "/umrs/ops/timeline"]
    assert [r.endpoint for r in routes] == [ops_timeline]


def test_signed_admin_reads_the_timeline(strict, client, timeline_store):
    resp = client.get("/umrs/ops/timeline", headers=_headers("root", "ops", admin=True))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["window_minutes"] == 60
    assert [e["job_id"] for e in body["events"]] == ["job-1"]


def test_signed_non_admin_is_refused_the_timeline(strict, client, timeline_store):
    resp = client.get("/umrs/ops/timeline", headers=_headers("alice", "proj-a"))
    assert resp.status_code == 403


def test_forged_admin_tuple_is_refused_the_timeline(strict, client, timeline_store):
    resp = client.get("/umrs/ops/timeline", headers=_forged("alice", "proj-b"))
    assert resp.status_code == 401


def test_anonymous_timeline_is_401_in_strict(strict, client, timeline_store):
    resp = client.get("/umrs/ops/timeline")
    assert resp.status_code == 401


# ── connect-info s3cfg emission ─────────────────────────────────────────────


@pytest.fixture
def minted(monkeypatch):
    import umrs.api.ops as ops

    calls: list[str] = []

    def _mint(username: str) -> str:
        calls.append(username)
        return f"s3cfg-for-{username}"

    monkeypatch.setenv("UMRS_EMIT_S3CFG", "1")
    monkeypatch.setattr(ops, "_get_or_create_user_s3cfg", _mint)
    return calls


def test_bare_user_header_gets_no_s3cfg_in_strict(strict, client, minted):
    resp = client.get("/umrs/ops/connect-info", headers={"X-User-ID": "alice"})
    assert resp.status_code == 200, resp.text
    assert resp.json().get("s3cfg_content") is None
    assert minted == []


def test_forged_tuple_gets_no_s3cfg(strict, client, minted):
    resp = client.get("/umrs/ops/connect-info", headers=_forged("alice", "proj-b"))
    assert resp.status_code == 401
    assert minted == []


def test_signed_caller_gets_only_its_own_s3cfg(strict, client, minted):
    resp = client.get("/umrs/ops/connect-info", headers=_headers("alice", "proj-a"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["s3cfg_content"] == "s3cfg-for-alice"
    assert minted == ["alice"]


def test_anonymous_discovery_stays_open(strict, client, minted):
    resp = client.get("/umrs/ops/connect-info")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "control_port" in body["zmq"]
    assert body.get("s3cfg_content") is None
    assert minted == []


def test_mvp_keeps_the_legacy_header_trust(monkeypatch, client, minted):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    resp = client.get("/umrs/ops/connect-info", headers={"X-User-ID": "alice"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["s3cfg_content"] == "s3cfg-for-alice"
