"""GET /umrs/knowledge/readiness is project-scoped (TUI audit 2026-10-17).

It returned the module-global explorer report (project "default", counted
from an in-memory model nothing fills) whatever project was asked for, so the
TUI readiness gate and the MCP knowledge_readiness tool always saw not_ready.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import knowledge as kn


def _report(confidence: float, degraded: bool = False):
    return SimpleNamespace(confidence=confidence, degraded=degraded)


@pytest.mark.parametrize(
    ("artifacts", "latest", "level", "can_run"),
    [
        (0, None, "not_ready", False),
        (3, None, "not_ready", False),
        (3, _report(0.9, degraded=True), "not_ready", False),
        (3, _report(0.9), "ready", True),
        (3, _report(0.6), "partially_ready", True),
        (3, _report(0.3), "not_ready", False),
    ],
)
def test_payload_levels(artifacts, latest, level, can_run):
    payload = kn.project_readiness_payload("lab1", artifacts, latest)
    assert payload["project_id"] == "lab1"
    assert (payload["readiness_level"], payload["can_run"]) == (level, can_run)
    assert bool(payload["blockers"]) is not can_run
    assert [c["check"] for c in payload["readiness_checks"]] == ["artifacts", "verification"]


@pytest.fixture
def client(monkeypatch):
    seen: dict[str, list[str]] = {"artifacts": [], "reports": []}

    class _Store:
        def __init__(self, _pool):
            pass

        async def list_artifacts(self, *, project_id):
            seen["artifacts"].append(project_id)
            return [object(), object()], 2

    class _Verification:
        def __init__(self, _pool):
            pass

        async def list_reports(self, *, project_id, limit):
            assert limit == 1
            seen["reports"].append(project_id)
            return [_report(0.85)]

    async def _pool(_request):
        return object()

    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", _Store)
    monkeypatch.setattr(kn, "VerificationStore", _Verification)
    monkeypatch.setattr(kn, "knowledge_asyncpg_pool", _pool)
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    app = FastAPI()
    app.include_router(kn.router)
    test_client = TestClient(app)
    test_client.seen = seen
    return test_client


def test_reports_the_requested_project(client):
    resp = client.get(
        "/umrs/knowledge/readiness",
        params={"project_id": "lab1"},
        headers={"x-user-id": "u1", "x-verified-project-id": "lab1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert (body["project_id"], body["readiness_level"], body["can_run"]) == ("lab1", "ready", True)
    assert client.seen == {"artifacts": ["lab1"], "reports": ["lab1"]}


def test_defaults_to_the_caller_project(client):
    resp = client.get(
        "/umrs/knowledge/readiness",
        headers={"x-user-id": "u1", "x-verified-project-id": "lab1"},
    )
    assert resp.status_code == 200
    assert resp.json()["project_id"] == "lab1"


def test_without_a_project_fails_closed(client):
    resp = client.get("/umrs/knowledge/readiness")
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "scope_required"
    assert client.seen == {"artifacts": [], "reports": []}


def test_another_project_is_denied_in_strict_mode(client, monkeypatch):
    from r4_contracts.handshake_hmac import sign_claims

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "sec")
    headers = {
        "X-User-ID": "alice",
        "X-Verified-Project-ID": "proj-a",
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims("sec", "alice", "proj-a", False),
    }
    denied = client.get(
        "/umrs/knowledge/readiness", params={"project_id": "proj-b"}, headers=headers
    )
    assert denied.status_code == 403
    assert denied.json()["detail"]["error"] == "scope_denied"
    own = client.get("/umrs/knowledge/readiness", headers=headers)
    assert own.status_code == 200
    assert own.json()["project_id"] == "proj-a"


@pytest.mark.parametrize("route", ["/status", "/findings", "/readiness/graph"])
def test_project_query_routes_deny_another_project(client, monkeypatch, route):
    """These routes took ?project_id= and never compared it with the caller."""
    from r4_contracts.handshake_hmac import sign_claims

    findings_calls: list[dict] = []

    class _Store:
        def __init__(self, _pool):
            pass

        async def list_artifacts(self, **_kw):
            return [], 0

    class _Verification:
        def __init__(self, _pool):
            pass

        async def list_reports(self, **_kw):
            return []

        async def list_findings(self, **kw):
            findings_calls.append(kw)
            return []

    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", _Store)
    monkeypatch.setattr(kn, "VerificationStore", _Verification)
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "sec")
    headers = {
        "X-User-ID": "alice",
        "X-Verified-Project-ID": "proj-a",
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims("sec", "alice", "proj-a", False),
    }
    other = client.get(f"/umrs/knowledge{route}", params={"project_id": "proj-b"}, headers=headers)
    assert other.status_code == 403
    own = client.get(f"/umrs/knowledge{route}", params={"project_id": "proj-a"}, headers=headers)
    assert own.status_code == 200
    if route == "/findings":
        # Findings were listed across every project; the store now filters.
        assert [c["project_id"] for c in findings_calls] == ["proj-a"]


def test_list_findings_filters_by_project_through_the_report():
    import asyncio

    from umrs.knowledge.verification_store import VerificationStore

    captured: dict = {}

    class _Conn:
        async def fetch(self, sql, *params):
            captured["sql"], captured["params"] = sql, params
            return []

    class _Acquire:
        async def __aenter__(self):
            return _Conn()

        async def __aexit__(self, *_exc):
            return False

    pool = SimpleNamespace(acquire=lambda: _Acquire())
    store = VerificationStore(pool)
    asyncio.run(store.list_findings(project_id="proj-a", status="open"))
    assert "SELECT report_id FROM verification_reports WHERE project_id = $1" in captured["sql"]
    assert "status = $2" in captured["sql"]
    assert captured["params"] == ("proj-a", "open")
