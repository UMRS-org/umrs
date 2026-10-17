"""H-1 (hardening plan, 2026-10-17), item 7: spoofable admin and identity
headers on the ops and usage routers.

- ``ops.require_ops_admin`` honoured ``X-Verified-Is-Admin: true`` without
  checking the gateway HMAC, so anyone reaching the sidecar port was an ops
  admin.
- ``ops.require_authenticated`` checked header *presence* only, and the
  governance ownership/quota reads took any ``project_id``: every project's
  ownership rows and quotas were readable by anyone.
- The governance share routes trusted a bare ``X-User-ID`` for the
  owner-or-admin check, so naming the owner was enough to share or revoke.
- ``usage.py`` decided admin from an unverified ``X-Verified-Is-Admin`` or an
  ``X-User-Role: admin`` header (summary/records/billing) and took the caller
  from a bare ``X-User-ID``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-admin-identity-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _forged_admin(user: str, project: str) -> dict[str, str]:
    """A non-admin's real signature with the admin flag flipped."""
    headers = _headers(user, project, admin=False)
    headers["X-Verified-Is-Admin"] = "true"
    return headers


def _unsigned(user: str, project: str, admin: bool = False) -> dict[str, str]:
    headers = _headers(user, project, admin)
    del headers["X-Verified-HMAC"]
    return headers


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


# ── ops: require_ops_admin ───────────────────────────────────────────────────


@pytest.fixture
def ops_client():
    from umrs.api.ops import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_forged_admin_flag_is_not_an_ops_admin(ops_client):
    resp = ops_client.get("/umrs/ops/health", headers=_forged_admin("alice", "proj-a"))
    assert resp.status_code == 401


def test_unsigned_admin_headers_are_not_an_ops_admin(ops_client):
    resp = ops_client.get("/umrs/ops/health", headers=_unsigned("root", "ops", admin=True))
    assert resp.status_code == 401


def test_signed_admin_reads_ops_health(ops_client):
    resp = ops_client.get("/umrs/ops/health", headers=_headers("root", "ops", admin=True))
    assert resp.status_code == 200, resp.text


def test_signed_non_admin_is_refused_ops_health(ops_client):
    resp = ops_client.get("/umrs/ops/health", headers=_headers("alice", "proj-a"))
    assert resp.status_code == 403


# ── ops: governance reads (require_authenticated + project scope) ────────────


class _GovStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.owners = {("endpoint", "ep-1"): {"owner_id": "bob"}}

    def list_quotas(self, project_id: str) -> list[dict[str, Any]]:
        self.calls.append(("list_quotas", (project_id,), {}))
        return [{"project_id": project_id, "resource_type": "endpoints"}]

    def list_by_project(self, project_id: str, resource_type: str = "") -> list:
        self.calls.append(("list_by_project", (project_id,), {"resource_type": resource_type}))
        return [{"project_id": project_id}]

    def list_owned(self, owner_id: str, resource_type: str = "", project_id: str = "") -> list:
        self.calls.append(
            ("list_owned", (owner_id,), {"resource_type": resource_type, "project_id": project_id})
        )
        return [{"owner_id": owner_id, "project_id": project_id}]

    def get_owner(self, resource_type: str, resource_id: str):
        return self.owners.get((resource_type, resource_id))

    def create_share(self, resource_type, resource_id, granted_to, **kwargs):
        self.calls.append(("create_share", (resource_type, resource_id, granted_to), kwargs))
        return {"resource_id": resource_id, "granted_to": granted_to}

    def revoke_share(self, resource_type, resource_id, granted_to, **kwargs):
        self.calls.append(("revoke_share", (resource_type, resource_id, granted_to), kwargs))
        return True

    def list_shares_for_user(self, user_id: str) -> list:
        self.calls.append(("list_shares_for_user", (user_id,), {}))
        return []


@pytest.fixture
def gov(monkeypatch, ops_client):
    import umrs.api.audit_events as audit_events
    import umrs.api.governance as governance

    store = _GovStore()
    monkeypatch.setattr(governance, "get_governance_store", lambda: store)
    monkeypatch.setattr(audit_events, "emit_audit", lambda *_a, **_k: None)
    return ops_client, store


def test_quotas_of_another_project_are_refused(gov):
    client, store = gov
    resp = client.get(
        "/umrs/ops/governance/quotas",
        params={"project_id": "proj-b"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 403
    assert not store.calls


def test_quotas_with_unsigned_identity_are_refused(gov):
    client, store = gov
    resp = client.get(
        "/umrs/ops/governance/quotas",
        params={"project_id": "proj-b"},
        headers=_unsigned("alice", "proj-a"),
    )
    assert resp.status_code == 401
    assert not store.calls


def test_own_project_quotas_are_readable(gov):
    client, store = gov
    resp = client.get(
        "/umrs/ops/governance/quotas",
        params={"project_id": "proj-a"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 200, resp.text
    assert store.calls == [("list_quotas", ("proj-a",), {})]


def test_admin_reads_any_project_quotas(gov):
    client, store = gov
    resp = client.get(
        "/umrs/ops/governance/quotas",
        params={"project_id": "proj-b"},
        headers=_headers("root", "ops", admin=True),
    )
    assert resp.status_code == 200, resp.text
    assert store.calls == [("list_quotas", ("proj-b",), {})]


def test_ownership_of_another_project_is_refused(gov):
    client, store = gov
    resp = client.get(
        "/umrs/ops/governance/ownership",
        params={"project_id": "proj-b"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 403
    assert not store.calls


def test_ownership_by_owner_stays_in_the_callers_project(gov):
    client, store = gov
    resp = client.get(
        "/umrs/ops/governance/ownership",
        params={"owner_id": "bob"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 200, resp.text
    assert store.calls == [("list_owned", ("bob",), {"resource_type": "", "project_id": "proj-a"})]


# ── ops: governance shares (owner check on a verified user) ──────────────────


def test_bare_owner_header_cannot_create_a_share(gov):
    client, store = gov
    resp = client.post(
        "/umrs/ops/governance/shares",
        params={"resource_type": "endpoint", "resource_id": "ep-1", "granted_to": "mallory"},
        headers={"X-User-ID": "bob"},
    )
    assert resp.status_code in (401, 403)
    assert not [c for c in store.calls if c[0] == "create_share"]


def test_owner_impersonation_with_a_foreign_signature_is_refused(gov):
    client, store = gov
    headers = _headers("mallory", "proj-a")
    headers["X-User-ID"] = "bob"
    resp = client.delete(
        "/umrs/ops/governance/shares",
        params={"resource_type": "endpoint", "resource_id": "ep-1", "granted_to": "carol"},
        headers=headers,
    )
    assert resp.status_code in (401, 403)
    assert not [c for c in store.calls if c[0] == "revoke_share"]


def test_signed_owner_creates_a_share(gov):
    client, store = gov
    resp = client.post(
        "/umrs/ops/governance/shares",
        params={"resource_type": "endpoint", "resource_id": "ep-1", "granted_to": "carol"},
        headers=_headers("bob", "proj-a"),
    )
    assert resp.status_code == 200, resp.text
    assert [c[0] for c in store.calls] == ["create_share"]


def test_admin_token_still_creates_a_share_without_identity(gov, monkeypatch):
    client, store = gov
    monkeypatch.setenv("UMRS_ADMIN_TOKEN", "tok")
    resp = client.post(
        "/umrs/ops/governance/shares",
        params={"resource_type": "endpoint", "resource_id": "ep-1", "granted_to": "carol"},
        headers={"x-admin-token": "tok"},
    )
    assert resp.status_code == 200, resp.text


# ── usage ───────────────────────────────────────────────────────────────────


class _Cursor:
    def __init__(self, log: list) -> None:
        self._log = log
        self.description = [("user_id",), ("request_count",)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._log.append((sql, list(params or [])))

    def fetchall(self):
        return []


class _Conn:
    def __init__(self, log: list) -> None:
        self._log = log

    def cursor(self):
        return _Cursor(self._log)


class _Pool:
    def __init__(self) -> None:
        self.log: list = []

    def getconn(self):
        return _Conn(self.log)

    def putconn(self, conn):
        del conn


@pytest.fixture
def usage():
    from umrs.api.usage import router

    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = _Pool()
    return TestClient(app), app.state.umrs_db_pool


def test_forged_admin_flag_cannot_read_another_users_usage(usage):
    client, pool = usage
    resp = client.get(
        "/umrs/usage/summary", params={"user_id": "bob"}, headers=_forged_admin("alice", "proj-a")
    )
    assert resp.status_code == 401
    assert not pool.log


def test_user_role_header_is_not_admin(usage):
    client, pool = usage
    headers = {**_headers("alice", "proj-a"), "X-User-Role": "admin"}
    resp = client.get("/umrs/usage/records", params={"user_id": "bob"}, headers=headers)
    assert resp.status_code == 200, resp.text
    ((_sql, params),) = pool.log
    assert "alice" in params and "bob" not in params


def test_user_role_header_does_not_open_billing(usage):
    client, pool = usage
    headers = {**_headers("alice", "proj-a"), "X-User-Role": "admin"}
    resp = client.get("/umrs/usage/billing", headers=headers)
    assert resp.status_code == 403
    assert not pool.log


def test_unsigned_user_cannot_read_own_usage(usage):
    client, pool = usage
    resp = client.get("/umrs/usage/me", headers={"X-User-ID": "bob"})
    assert resp.status_code == 401
    assert not pool.log


def test_signed_admin_reads_billing_and_any_user(usage):
    client, pool = usage
    admin = _headers("root", "ops", admin=True)
    assert client.get("/umrs/usage/billing", headers=admin).status_code == 200
    resp = client.get("/umrs/usage/summary", params={"user_id": "bob"}, headers=admin)
    assert resp.status_code == 200, resp.text
    assert "bob" in pool.log[-1][1]


def test_signed_user_reads_own_usage(usage):
    client, pool = usage
    resp = client.get("/umrs/usage/me", headers=_headers("alice", "proj-a"))
    assert resp.status_code == 200, resp.text
    assert resp.json()["user_id"] == "alice"
    assert "alice" in pool.log[-1][1]
