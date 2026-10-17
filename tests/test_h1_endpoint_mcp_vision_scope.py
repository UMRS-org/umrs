"""H-1 (hardening plan, 2026-10-17), items 2–4: endpoint location rewrite,
MCP tool lookup/call, and vision status/cancel by job id.

- PUT /umrs/endpoints/{job_id}/location was documented admin-only but any
  caller could rewrite any endpoint's host and port.
- GET /umrs/mcp/tools/lookup and POST /umrs/mcp/call|invoke took no identity:
  any caller found any project's tool and the call went out with that
  server's stored upstream credentials.
- /umrs/vision/infer/status|cancel acted on a bare job_id: any durable job of
  any type in any project could be cancelled or marked finished.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-items-secret"


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


# ── item 2: endpoint location ────────────────────────────────────────────────


@pytest.fixture
def endpoints_client():
    from umrs.api.endpoints import router
    from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

    registry = EndpointRegistry()
    registry.register(
        project_id="proj-a", model="m", job_id="job-1", host="node-01", port=8000, backend="slurm"
    )
    registry.mark_ready("job-1")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app), registry


def test_non_admin_cannot_rewrite_an_endpoint_location(endpoints_client):
    client, registry = endpoints_client
    resp = client.put(
        "/umrs/endpoints/job-1/location",
        json={"host": "evil", "port": 1},
        headers=_headers("alice", "proj-a"),
    )
    assert resp.status_code == 403
    assert registry.get_by_job_id("job-1").host == "node-01"


def test_signed_admin_may_rewrite_an_endpoint_location(endpoints_client):
    client, registry = endpoints_client
    resp = client.put(
        "/umrs/endpoints/job-1/location",
        json={"host": "node-02", "port": 8001},
        headers=_headers("root", "ops", admin=True),
    )
    assert resp.status_code == 200
    assert registry.get_by_job_id("job-1").host == "node-02"


# ── item 3: MCP lookup / call / invoke ───────────────────────────────────────


@pytest.fixture
def mcp_client(monkeypatch):
    from umrs.api import mcp

    server = SimpleNamespace(
        server_id="srv-b",
        name="b-tools",
        url="http://b.local",
        version="1",
        transport="http",
        status="healthy",
        description="",
        auth_type="bearer",
        auth_config={"token": "upstream-secret"},
        metadata={},
        registered_by="bob",
        project_id="proj-b",
        tools=[],
        created_at=None,
        updated_at=None,
        trust_class="",
    )
    store = MagicMock()
    store.get_server_for_tool.return_value = server
    monkeypatch.setattr(mcp, "_get_store", lambda: store)
    proxied = []

    async def _fake_proxy(*args, **kwargs):
        proxied.append(args)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(mcp, "_proxy_tool_call", _fake_proxy)
    app = FastAPI()
    app.include_router(mcp.router)
    return TestClient(app), proxied


@pytest.mark.parametrize("path", ["/umrs/mcp/call", "/umrs/mcp/invoke"])
def test_another_projects_tool_cannot_be_called(mcp_client, path):
    client, proxied = mcp_client
    resp = client.post(path, json={"tool": "b_tool", "arguments": {}}, headers=_headers("alice", "proj-a"))
    assert resp.status_code == 404
    assert proxied == []


def test_another_projects_tool_cannot_be_looked_up(mcp_client):
    client, _ = mcp_client
    resp = client.get("/umrs/mcp/tools/lookup", params={"tool": "b_tool"}, headers=_headers("alice", "proj-a"))
    assert resp.status_code == 404


def test_tool_calls_need_identity_in_strict_mode(mcp_client):
    client, proxied = mcp_client
    assert client.post("/umrs/mcp/call", json={"tool": "b_tool", "arguments": {}}).status_code == 401
    assert proxied == []


def test_own_projects_tool_is_callable(mcp_client):
    client, proxied = mcp_client
    resp = client.post("/umrs/mcp/call", json={"tool": "b_tool", "arguments": {}}, headers=_headers("bob", "proj-b"))
    assert resp.status_code == 200
    assert len(proxied) == 1


# ── item 4: vision job ownership ─────────────────────────────────────────────


def _job_store(project: str, job_type):
    store = MagicMock()
    store.get_job.return_value = SimpleNamespace(project_id=project, job_type=job_type, active_attempt_id="att-1")
    return store


def test_owned_job_id_only_for_own_vision_jobs():
    from umrs.api.vision_durable import owned_job_id
    from umrs.jobs.models import JobType

    own = _job_store("proj-a", JobType.VISION_INFER)
    assert owned_job_id(own, "job-1", "proj-a") == "job-1"
    assert owned_job_id(own, "job-1", "proj-b") is None
    other_type = _job_store("proj-a", next(t for t in JobType if t != JobType.VISION_INFER))
    assert owned_job_id(other_type, "job-1", "proj-a") is None
    missing = MagicMock()
    missing.get_job.side_effect = KeyError("job-x")
    assert owned_job_id(missing, "job-x", "proj-a") is None


def test_status_poll_never_transitions_a_foreign_job(monkeypatch):
    from umrs.api import vision, vision_durable
    from umrs.jobs.models import JobType

    store = _job_store("proj-b", JobType.VISION_INFER)
    monkeypatch.setattr(vision_durable, "store_for", lambda _req: store)
    transitions = []
    monkeypatch.setattr(vision_durable, "terminal", lambda *a, **_k: transitions.append(a))
    monkeypatch.setattr(vision_durable, "cancel", lambda *a, **_k: transitions.append(a))
    response = SimpleNamespace(json=lambda: {"status": "cancelled"})
    vision._record_status_transition(MagicMock(), "job-1", response, "proj-a")
    assert transitions == []
    vision._record_status_transition(MagicMock(), "job-1", response, "proj-b")
    assert transitions
