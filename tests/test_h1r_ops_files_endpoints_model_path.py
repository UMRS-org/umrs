"""H-1 remaining gaps (hardening plan Appendix A, 2026-10-18), endpoint model path.

The mount-scope fix bound ``mounts.*.source`` to the caller's own roots, but
the model weights are a second host path bound into the endpoint container:
``POST /umrs/endpoints/provision`` with a ``model_path`` override, or with an
absolute ``model`` that is not in the catalog, was checked only against the
site-wide prefixes. A signed non-admin could serve another user's or another
project's weights from an endpoint it controls. A host model path the caller
names itself (anything but the authorised catalog entry's ``storage_ref``)
must now lie under the caller's own roots; anything else is 403.
"""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1r-ops-files-endpoints-secret"
CATALOG_REF = "/shared/models/qwen3-8b"


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


def _catalog(name: str):
    if name == "qwen3-8b":
        return {"storage_ref": CATALOG_REF, "model_type": "llm"}
    return None


@pytest.fixture
def service(monkeypatch):
    from umrs.api import endpoints as endpoints_api

    stub = _StubService()
    monkeypatch.setattr(endpoints_api, "_resolve_registered_model_entry", _catalog)
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


def _provision(client, headers: dict[str, str], **body):
    payload = {"model": "qwen3-8b", "tier": "project", **body}
    return client.post("/umrs/endpoints/provision", json=payload, headers=headers)


def test_foreign_model_path_override_is_403(strict, service, client):
    resp = _provision(client, _headers("alice", "proj-a"), model_path="/home/bob/private-weights")
    assert resp.status_code == 403, resp.text
    assert service.requests == []


def test_foreign_absolute_model_name_is_403(strict, service, client):
    resp = _provision(client, _headers("alice", "proj-a"), model="/projects/proj-b/weights")
    assert resp.status_code == 403, resp.text
    assert service.requests == []


def test_own_model_path_override_is_forwarded(strict, service, client):
    resp = _provision(client, _headers("alice", "proj-a"), model_path="/home/alice/weights")
    assert resp.status_code == 202, resp.text
    assert service.requests[0].model_path.endswith("/home/alice/weights")


def test_own_absolute_model_name_is_forwarded(strict, service, client):
    resp = _provision(client, _headers("alice", "proj-a"), model="/projects/proj-a/weights")
    assert resp.status_code == 202, resp.text
    assert service.requests[0].model_path.endswith("/projects/proj-a/weights")


def test_catalog_storage_ref_stays_usable(strict, service, client):
    """The authorised catalog entry's own path is not a caller-named path."""
    default = _provision(client, _headers("alice", "proj-a"))
    explicit = _provision(client, _headers("alice", "proj-a"), model_path=CATALOG_REF)
    assert default.status_code == 202, default.text
    assert explicit.status_code == 202, explicit.text
    assert [r.model_path for r in service.requests] == [CATALOG_REF, CATALOG_REF]


def test_signed_admin_may_name_any_allowed_model_path(strict, service, client):
    headers = _headers("root", "ops", admin=True)
    resp = _provision(client, headers, model_path="/home/bob/private-weights")
    assert resp.status_code == 202, resp.text
    assert len(service.requests) == 1


def test_forged_admin_tuple_is_refused(strict, service, client):
    resp = _provision(client, _forged("alice", "proj-b"), model_path="/projects/proj-b/weights")
    assert resp.status_code == 401, resp.text
    assert service.requests == []


def test_symlink_out_of_own_root_is_403(strict, service, client, monkeypatch, tmp_path):
    """A link the server can see is resolved before the root check.

    (A link the server cannot see, or one swapped after this check, is the
    residual recorded in the report: the compute-node launcher must re-check.)
    """
    (tmp_path / "bob" / "weights").mkdir(parents=True)
    (tmp_path / "alice").mkdir()
    os.symlink(tmp_path / "bob" / "weights", tmp_path / "alice" / "link")
    monkeypatch.setenv("UMRS_ENDPOINT_ALLOWED_MOUNT_PREFIXES", str(tmp_path))
    monkeypatch.setenv("UMRS_ENDPOINT_CALLER_MOUNT_ROOTS", f"{tmp_path}/{{user_id}}")
    resp = _provision(client, _headers("alice", "proj-a"), model_path=f"{tmp_path}/alice/link")
    assert resp.status_code == 403, resp.text
    assert service.requests == []
