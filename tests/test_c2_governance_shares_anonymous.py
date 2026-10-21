"""Cycle-2 umrs-api lane (2026-10-21): the governance share routes need a
caller — a verified identity or the admin token.

Found live on the UMRS port (the H-1 threat model: a co-tenant reaching the
sidecar directly): ``GET /umrs/ops/governance/shares`` with no identity at
all answered 200. ``governance_user_id`` resolves no identity to user ``""``,
and the owner check compares that with the resource's ``owner_id``: a
resource recorded without an owner (``record_resource_created(...,
requesting_user or "")`` for endpoints) matched the anonymous caller, who
could then share or revoke it.
"""

from __future__ import annotations

import pytest
from test_h1_admin_identity_ops_usage import _headers

SHARES = "/umrs/ops/governance/shares"
OWNERLESS = {"resource_type": "endpoint", "resource_id": "ep-ownerless"}


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "h1-admin-identity-secret")
    monkeypatch.delenv("UMRS_ADMIN_TOKEN", raising=False)


@pytest.fixture
def gov(monkeypatch):
    import umrs.api.audit_events as audit_events
    import umrs.api.governance as governance
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from test_h1_admin_identity_ops_usage import _GovStore
    from umrs.api.ops import router

    store = _GovStore()
    store.owners[("endpoint", "ep-ownerless")] = {"owner_id": ""}
    monkeypatch.setattr(governance, "get_governance_store", lambda: store)
    monkeypatch.setattr(audit_events, "emit_audit", lambda *_a, **_k: None)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), store


def test_anonymous_list_is_401(gov):
    client, store = gov
    assert client.get(SHARES).status_code == 401
    assert not store.calls


@pytest.mark.parametrize("method", ["post", "delete"])
def test_anonymous_cannot_touch_an_ownerless_resource(gov, method):
    client, store = gov
    resp = client.request(method.upper(), SHARES, params={**OWNERLESS, "granted_to": "mallory"})
    assert resp.status_code == 401
    assert not store.calls


def test_signed_caller_still_lists_own_shares(gov):
    client, store = gov
    assert client.get(SHARES, headers=_headers("bob", "proj-a")).status_code == 200
    assert store.calls == [("list_shares_for_user", ("bob",), {})]
