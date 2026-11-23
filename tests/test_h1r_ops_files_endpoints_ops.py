"""H-1 remaining gaps (hardening plan Appendix A, 2026-10-18), ops router.

- ``GET /umrs/ops/timeline``: the ``@router.get("/timeline")`` decorator sat
  above the helper ``_enum_value``, so the path answered 422/500 with no
  authentication and ``ops_timeline`` (admin-only) was never mounted.
- ``GET /umrs/ops/connect-info``: with ``UMRS_EMIT_S3CFG=1`` a bare
  ``X-User-ID`` returned (and minted) that user's s3cfg. Strict mode then
  emitted it only for a gateway-signed identity tuple; DEC-016 (2026-11-24)
  removed the emission, a forged tuple is still 401.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs_protocol.handshake_hmac import sign_claims

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


# ── connect-info identity tuple ─────────────────────────────────────────────
# connect-info returns nothing per user since DEC-016 (2026-11-24; the s3cfg
# emission pins live in test_connect_info_s3cfg_gate.py). A forged tuple is
# still refused and anonymous discovery stays open.


def test_forged_tuple_is_refused(strict, client):
    resp = client.get("/umrs/ops/connect-info", headers=_forged("alice", "proj-b"))
    assert resp.status_code == 401


def test_anonymous_discovery_stays_open(strict, client):
    resp = client.get("/umrs/ops/connect-info")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "control_port" in body["zmq"]
    assert body.get("s3cfg_content") is None
