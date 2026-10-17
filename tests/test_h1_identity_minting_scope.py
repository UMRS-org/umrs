"""H-1 (hardening plan, 2026-10-17): identity-minting paths honour only a
project the caller may write.

A secret-broker lease (scope ``inference``) and a frontend-venue start both
named their project in the body, unchecked. The venue rail then minted
HMAC-signed identity for (lease owner, lease project), so any authenticated
user could act as a member of another project; the rail also fell back to the
session snapshot's project, which the client writes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


# ── secret-broker lease ─────────────────────────────────────────────────────


@pytest.fixture
def lease_client(monkeypatch):
    monkeypatch.setenv("UMRS_SECRET_BROKER_BACKEND", "memory")
    from umrs.api import secret_broker as api_mod

    api_mod._broker = None
    api_mod.get_broker()
    app = FastAPI()
    app.include_router(api_mod.router)
    return TestClient(app), api_mod


def _lease(client, headers, project=None):
    body = {"job_id": "session-s1", "secret_name": "inference-token", "scope": "inference"}
    if project is not None:
        body["project"] = project
    return client.post("/umrs/secrets/lease", json=body, headers=headers)


def _stored_project(api_mod, lease_id: str) -> str:
    return api_mod.get_broker().get_lease(lease_id).project


def test_lease_for_another_project_is_refused(lease_client):
    client, _ = lease_client
    resp = _lease(client, _headers("alice", "proj-a"), project="proj-b")
    assert resp.status_code == 403


def test_lease_for_own_project_is_issued(lease_client):
    client, api_mod = lease_client
    resp = _lease(client, _headers("alice", "proj-a"), project="proj-a")
    assert resp.status_code == 201
    assert _stored_project(api_mod, resp.json()["lease_id"]) == "proj-a"


def test_lease_without_project_binds_the_callers_project(lease_client):
    client, api_mod = lease_client
    resp = _lease(client, _headers("alice", "proj-a"))
    assert resp.status_code == 201
    assert _stored_project(api_mod, resp.json()["lease_id"]) == "proj-a"


def test_personal_project_is_writable(lease_client):
    client, _ = lease_client
    assert _lease(client, _headers("alice", "proj-a"), project="user:alice").status_code == 201
    assert _lease(client, _headers("alice", "proj-a"), project="user:bob").status_code == 403


def test_admin_may_name_any_project(lease_client):
    client, api_mod = lease_client
    resp = _lease(client, _headers("root", "ops", admin=True), project="proj-b")
    assert resp.status_code == 201
    assert _stored_project(api_mod, resp.json()["lease_id"]) == "proj-b"


# ── frontend-venue start ────────────────────────────────────────────────────


@pytest.fixture
def venue_client():
    from umrs.api import companion_venue

    service = MagicMock()
    service.start.return_value = {"state": "provisioning"}
    app = FastAPI()
    app.include_router(companion_venue.router)
    app.state.umrs_db_pool = MagicMock()  # the cpu-venue probe swallows its errors
    app.state.companion_venue_service = service
    return TestClient(app), service


def _start(client, headers, project=None):
    body = {"source_refs": []}
    if project is not None:
        body["project"] = project
    return client.post("/umrs/companion/sessions/s1/venue/start", json=body, headers=headers)


def test_venue_start_for_another_project_is_refused(venue_client):
    client, service = venue_client
    resp = _start(client, _headers("alice", "proj-a"), project="proj-b")
    assert resp.status_code == 403
    service.start.assert_not_called()


def test_venue_start_binds_own_or_named_writable_project(venue_client):
    client, service = venue_client
    assert _start(client, _headers("alice", "proj-a")).status_code == 200
    assert service.start.call_args.kwargs["project"] == "proj-a"
    assert _start(client, _headers("alice", "proj-a"), project="proj-a").status_code == 200
    assert service.start.call_args.kwargs["project"] == "proj-a"


# ── rail identity ───────────────────────────────────────────────────────────


def test_rail_never_takes_the_project_from_the_client_written_snapshot(monkeypatch):
    from umrs.api import companion_venue

    monkeypatch.setenv("JWT_DEFAULT_PROJECT", "default")
    legacy = SimpleNamespace(lease=SimpleNamespace(project=""), owner="alice", session_id="s1")
    request = MagicMock()  # a snapshot naming proj-b would have been read from here
    assert companion_venue.rail_project_for(legacy, request) == "default"
    checked = SimpleNamespace(lease=SimpleNamespace(project="proj-a"), owner="alice", session_id="s1")
    assert companion_venue.rail_project_for(checked, request) == "proj-a"
