"""H-1 remaining gaps (hardening plan Appendix A, 2026-10-18), endpoint mounts.

``POST /umrs/endpoints/provision`` checked ``mounts.mcp_servers[*]``,
``mounts.project_data`` and ``mounts.rag_index`` sources only against the
site-wide prefixes (``UMRS_ENDPOINT_ALLOWED_MOUNT_PREFIXES``: /home,
/projects, /shared, ...), so any caller could bind-mount another user's home
or another project's data into an endpoint it controls. A non-admin's
sources must now lie under the caller's own per-user / per-project roots
(``UMRS_ENDPOINT_CALLER_MOUNT_ROOTS`` templates); anything else is 403.
"""

from __future__ import annotations

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


class _StubService:
    def __init__(self) -> None:
        self.requests: list = []

    async def provision(self, request):
        from umrs.orchestrator.service import ProvisionResult

        self.requests.append(request)
        return ProvisionResult(success=True, job_id="job-123")


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    monkeypatch.delenv("UMRS_ENDPOINT_ALLOWED_MOUNT_PREFIXES", raising=False)
    monkeypatch.delenv("UMRS_ENDPOINT_CALLER_MOUNT_ROOTS", raising=False)
    monkeypatch.setenv("EGRESS_GUARD_IMAGE", "localhost/r4-mcp:test")


@pytest.fixture
def service(monkeypatch):
    from umrs.api import endpoints as endpoints_api

    stub = _StubService()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda _name: {"storage_ref": "/shared/models/qwen3-8b", "model_type": "llm"},
    )
    monkeypatch.setattr(endpoints_api, "_check_requesting_user_gate", lambda _user: None)
    monkeypatch.setattr("umrs.orchestrator.service.get_orchestration_service", lambda: stub)
    return stub


@pytest.fixture
def client():
    from umrs.api.endpoints import router
    from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

    app = FastAPI()
    app.include_router(router)
    registry = EndpointRegistry()
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app, raise_server_exceptions=False)


def _provision(client, headers: dict[str, str], mounts: dict):
    return client.post(
        "/umrs/endpoints/provision",
        json={"model": "qwen3-8b", "tier": "project", "mounts": mounts},
        headers=headers,
    )


def _data(source: str) -> dict:
    return {"project_data": {"source": source, "target": "/project"}}


def test_foreign_project_data_source_is_403(strict, service, client):
    resp = _provision(client, _headers("alice", "proj-a"), _data("/projects/proj-b/data"))
    assert resp.status_code == 403, resp.text
    assert service.requests == []


def test_another_users_home_as_mcp_source_is_403(strict, service, client):
    mounts = {"mcp_servers": [{"source": "/home/bob/tools", "target": "/mcp-servers/tools"}]}
    resp = _provision(client, _headers("alice", "proj-a"), mounts)
    assert resp.status_code == 403, resp.text
    assert service.requests == []


def test_foreign_rag_index_is_403(strict, service, client):
    mounts = {"rag_index": {"source": "/shared/projects/proj-b/rag", "target": "/rag-index"}}
    resp = _provision(client, _headers("alice", "proj-a"), mounts)
    assert resp.status_code == 403, resp.text
    assert service.requests == []


def test_site_wide_prefix_alone_is_not_enough(strict, service, client):
    resp = _provision(client, _headers("alice", "proj-a"), _data("/project/data"))
    assert resp.status_code == 403, resp.text
    assert service.requests == []


def test_traversal_out_of_own_home_is_403(strict, service, client):
    resp = _provision(client, _headers("alice", "proj-a"), _data("/home/alice/../bob/data"))
    assert resp.status_code == 403, resp.text
    assert service.requests == []


def test_dot_dot_project_id_does_not_widen_the_root(strict, service, client):
    resp = _provision(client, _headers("alice", ".."), _data("/projects/proj-b/data"))
    assert resp.status_code == 403, resp.text
    assert service.requests == []


def test_own_roots_are_forwarded(strict, service, client):
    mounts = {
        "mcp_servers": [{"source": "/home/alice/tools", "target": "/mcp-servers/tools"}],
        "project_data": {"source": "/projects/proj-a/data", "target": "/project"},
        "rag_index": {"source": "/shared/projects/proj-a/rag", "target": "/rag-index"},
    }
    resp = _provision(client, _headers("alice", "proj-a"), mounts)
    assert resp.status_code == 202, resp.text
    forwarded = service.requests[0].mounts
    assert forwarded["project_data"]["source"].endswith("/projects/proj-a/data")
    assert forwarded["mcp_servers"][0]["source"].endswith("/home/alice/tools")


def test_signed_admin_may_mount_any_allowed_source(strict, service, client):
    resp = _provision(client, _headers("root", "ops", admin=True), _data("/projects/proj-b/data"))
    assert resp.status_code == 202, resp.text
    assert len(service.requests) == 1


def test_forged_admin_tuple_is_refused(strict, service, client):
    resp = _provision(client, _forged("alice", "proj-b"), _data("/projects/proj-b/data"))
    assert resp.status_code == 401, resp.text
    assert service.requests == []


def test_site_root_templates_override_the_defaults(strict, service, client, monkeypatch):
    monkeypatch.setenv(
        "UMRS_ENDPOINT_CALLER_MOUNT_ROOTS", "/lvs0/{project_id},/lvs0/users/{user_id}"
    )
    ok = _provision(client, _headers("alice", "proj-a"), _data("/lvs0/proj-a/data"))
    assert ok.status_code == 202, ok.text
    refused = _provision(client, _headers("alice", "proj-a"), _data("/home/alice/data"))
    assert refused.status_code == 403, refused.text
    assert len(service.requests) == 1


def test_no_mounts_is_unaffected(strict, service, client):
    resp = client.post(
        "/umrs/endpoints/provision",
        json={"model": "qwen3-8b", "tier": "project"},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 202, resp.text
