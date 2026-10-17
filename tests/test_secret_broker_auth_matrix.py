"""G-6 route-auth matrix for /umrs/secrets/*.

Every route: no identity → 401 (strict); forged HMAC → 401; valid user HMAC
→ passes the envelope; admin-only verbs → 403 for non-admin users;
owner-scoped leases stay invisible across users (404, no disclosure).
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

os.environ.setdefault("UMRS_AUTH_MODE", "strict")

from umrs.api.secret_broker import get_broker, router

SECRET = "matrix-secret"

METHOD_ROUTES = [
    ("post", "/umrs/secrets/lease"),
    ("post", "/umrs/secrets/validate"),
    ("get", "/umrs/secrets/lease"),
    ("get", "/umrs/secrets/lease/l-x"),
    ("post", "/umrs/secrets/lease/l-x/revoke"),
    ("post", "/umrs/secrets/lease/l-x/renew"),
    ("post", "/umrs/secrets/job/j-1/revoke"),
    ("get", "/umrs/secrets/audit"),
    ("post", "/umrs/secrets/cleanup"),
]


def _headers(secret: str, user: str, project: str, admin: bool) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(secret, user, project, admin),
    }


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    # Hermetic: in-memory broker, rebuilt fresh per test. Otherwise the
    # env-driven factory (R0-2) picks the real shared pool and this matrix
    # reads production lease rows.
    monkeypatch.setenv("UMRS_SECRET_BROKER_BACKEND", "memory")
    from umrs.api import secret_broker as api_mod

    api_mod._broker = None
    api_mod.get_broker()
    app = FastAPI()
    app.include_router(router)
    return app


def _hit(client: TestClient, method: str, path: str, headers: dict[str, str] | None = None):
    if method == "post":
        return client.post(path, json={}, headers=headers)
    return client.get(path, headers=headers)


@pytest.mark.parametrize("method, path", METHOD_ROUTES)
def test_no_identity_401(app, method, path):
    with TestClient(app) as client:
        response = _hit(client, method, path)
    assert response.status_code == 401


@pytest.mark.parametrize("method, path", METHOD_ROUTES)
def test_forged_hmac_401(app, method, path):
    headers = _headers("wrong-secret", "alice", "alice-proj", False)
    with TestClient(app) as client:
        response = _hit(client, method, path, headers=headers)
    assert response.status_code == 401


def test_valid_user_envelope_passes_for_read(app):
    headers = _headers(SECRET, "alice", "alice-proj", False)
    with TestClient(app) as client:
        response = client.get("/umrs/secrets/lease", headers=headers)
    assert response.status_code == 200
    assert response.json() == []


def test_admin_only_verbs_reject_non_admin(app):
    headers = _headers(SECRET, "alice", "alice-proj", False)
    with TestClient(app) as client:
        assert client.post("/umrs/secrets/cleanup", headers=headers).status_code == 403
        assert client.post("/umrs/secrets/job/j-1/revoke", headers=headers).status_code == 403


def test_admin_verbs_pass_for_admin(app):
    headers = _headers(SECRET, "root", "system", True)
    with TestClient(app) as client:
        assert client.post("/umrs/secrets/cleanup", headers=headers).status_code == 200


def test_owner_pinning_on_issue(app):
    user_headers = _headers(SECRET, "alice", "alice-proj", False)
    with TestClient(app) as client:
        response = client.post(
            "/umrs/secrets/lease",
            headers=user_headers,
            json={"job_id": "j-1", "secret_name": "s", "scope": "general", "owner_id": "bob"},
        )
        assert response.status_code == 201
        assert response.json()["owner_id"] == "alice"


def test_cross_user_lease_invisible(app):
    alice = _headers(SECRET, "alice", "alice-proj", False)
    bob = _headers(SECRET, "bob", "bob-proj", False)
    with TestClient(app) as client:
        issued = client.post(
            "/umrs/secrets/lease",
            headers=alice,
            json={"job_id": "j-1", "secret_name": "s", "scope": "general"},
        ).json()
        lease_id = issued["lease_id"]
        # Bob cannot see or mutate Alice's lease (404, no disclosure).
        assert client.get(f"/umrs/secrets/lease/{lease_id}", headers=bob).status_code == 404
        assert client.post(f"/umrs/secrets/lease/{lease_id}/revoke", headers=bob).status_code == 404
        listing = client.get("/umrs/secrets/lease", headers=bob).json()
        assert all(row["lease_id"] != lease_id for row in listing)
        audit = client.get("/umrs/secrets/audit", headers=bob).json()
        assert all(row["lease_id"] != lease_id for row in audit)
        # Alice sees her own.
        assert client.get(f"/umrs/secrets/lease/{lease_id}", headers=alice).status_code == 200
        # Reset shared broker state for following tests in this process.
        broker = get_broker()
        broker._leases.clear()
        broker._secret_index.clear()
        broker._audit.clear()


def test_inference_bindings_flow_through_issue(app):
    """The issue route passes inference bindings to the broker (live-leg lesson:
    the R0 HTTP route dated R0-2 by not passing them; this pins the guard)."""
    headers = _headers(SECRET, "alice", "alice-proj", False)
    with TestClient(app) as client:
        r = client.post(
            "/umrs/secrets/lease",
            headers=headers,
            json={
                "job_id": "sess-1",
                "secret_name": "t",
                "scope": "inference",
                "session_id": "sess-1",
                "source_ids": ["11111111-1111-4111-8111-111111111111"],
                "default_source_id": "11111111-1111-4111-8111-111111111111",
                "egress_ip": "10.1.1.1",
                # The caller's own project: a lease for another project is
                # refused (H-1, test_h1_identity_minting_scope.py).
                "project": "alice-proj",
                "max_lifetime_seconds": 86_400,
            },
        )
        assert r.status_code == 201
        lease_id = r.json()["lease_id"]
        from umrs.api.secret_broker import get_broker

        lease = get_broker().get_lease(lease_id)
        assert lease.session_id == "sess-1"
        assert lease.source_ids == ("11111111-1111-4111-8111-111111111111",)
        assert lease.default_source_id == "11111111-1111-4111-8111-111111111111"
        assert lease.egress_ip == "10.1.1.1"
        assert lease.project == "alice-proj"
        assert lease.max_lifetime_at is not None
