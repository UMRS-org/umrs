"""RS-0: the admin JWT ``umrs-server admin endpoint`` sends reaches every project.

An admin JWT always carries a ``project_id`` claim, which scoped the
registry list, ``GET /{job_id}`` and ``POST /{job_id}/refresh`` to that one
project: the admin CLI could not see another project's endpoint. An admin
now asks for the cross-project list (``all_projects=true``) and reads any
project's endpoint; a non-admin stays in their own project.
"""

from __future__ import annotations

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt as jose_jwt
from umrs.api import endpoints as endpoints_api
from umrs.api.endpoints import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

SECRET = "test-jwt-secret-not-real"


def _bearer(sub: str, roles: list[str], project: str) -> dict[str, str]:
    claims = {"sub": sub, "roles": roles, "project_id": project, "exp": int(time.time()) + 600}
    return {"Authorization": "Bearer " + jose_jwt.encode(claims, SECRET, algorithm="HS256")}


ADMIN = _bearer("root", ["admin"], "ops")
USER = _bearer("alice", ["user"], "proj-a")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    registry = EndpointRegistry()
    for project, job in (("proj-a", "job-a"), ("proj-b", "job-b")):
        registry.register(
            project_id=project, model="m", job_id=job, host="n", port=1, backend="slurm"
        )

    async def all_live(_registry, endpoints):
        return endpoints

    async def unchanged(_registry, endpoint):
        return endpoint

    monkeypatch.setattr(endpoints_api, "_filter_live_ready_endpoints", all_live)
    monkeypatch.setattr(endpoints_api, "_refresh_endpoint_readiness", unchanged)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def _jobs(resp) -> list[str]:
    assert resp.status_code == 200, resp.text
    return sorted(e["job_id"] for e in resp.json()["endpoints"])


def test_admin_lists_every_project_when_asked(client):
    assert _jobs(client.get("/umrs/endpoints?all_projects=true", headers=ADMIN)) == [
        "job-a",
        "job-b",
    ]


def test_admin_without_the_flag_keeps_the_claim_project(client):
    assert _jobs(client.get("/umrs/endpoints", headers=ADMIN)) == []


def test_non_admin_flag_is_ignored(client):
    assert _jobs(client.get("/umrs/endpoints?all_projects=true", headers=USER)) == ["job-a"]


def test_admin_reads_another_projects_endpoint(client):
    assert client.get("/umrs/endpoints/job-b", headers=ADMIN).status_code == 200
    assert client.get("/umrs/endpoints/job-b", headers=USER).status_code == 404
    assert client.get("/umrs/endpoints/job-a", headers=USER).status_code == 200


def test_a_disabled_admins_jwt_is_not_an_admin(client, tmp_path, monkeypatch):
    """ADR-097 Amendment D-8 (2026-10-23): the registry's admin check
    (``_caller_is_admin``) honoured an admin JWT for its whole lifetime,
    after its owner was disabled."""
    from umrs.admin.user_store import UserStore

    monkeypatch.setenv("UMRS_USER_FILES_DIR", str(tmp_path))
    users = UserStore(tmp_path)
    users.create("root", role="admin")
    assert _jobs(client.get("/umrs/endpoints?all_projects=true", headers=ADMIN)) == [
        "job-a",
        "job-b",
    ]
    users.set_disabled("root", True)
    # no longer an admin: the claim project alone, and nothing runs there
    assert _jobs(client.get("/umrs/endpoints?all_projects=true", headers=ADMIN)) == []
