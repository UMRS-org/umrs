"""Readiness approval is per project (TUI audit 2026-10-17, round 2).

``POST /umrs/knowledge/approval`` held ONE approval for the whole server in
the module-global explorer controller (client-asserted ``approved_by``, no
project); ``KnowledgeClient.readiness_approve`` posted to
``/readiness/approve``, which UMRS did not serve; nothing wrote the
migration-0055 ``readiness_approvals`` table.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import knowledge as kn
from umrs.knowledge.readiness_approvals import ReadinessApprovalStore

_T0 = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)


class _Conn:
    """In-memory stand-in for the readiness_approvals table (asyncpg shape)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.sql: list[str] = []

    async def fetchrow(self, sql, *params):
        self.sql.append(sql)
        if sql.startswith("INSERT"):
            keys = ("approval_id", "project_id", "threshold", "approved_by")
            row = dict(zip(keys, params[:4], strict=True))
            row.update(approved_at=params[4], expires_at=params[5], reason=params[6])
            row["threshold"] = Decimal(str(row["threshold"]))
            self.rows.append(row)
            return row
        project_id, at = params
        live = [
            r
            for r in self.rows
            if r["project_id"] == project_id and (r["expires_at"] is None or r["expires_at"] > at)
        ]
        live.sort(key=lambda r: r["approved_at"], reverse=True)
        return live[0] if live else None


class _Pool:
    def __init__(self):
        self.conn = _Conn([])

    def acquire(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *_exc):
                return False

        return _Ctx()


def test_store_records_and_reads_the_latest_unexpired_row():
    pool = _Pool()
    store = ReadinessApprovalStore(pool)
    asyncio.run(store.record(project_id="p1", confidence=0.876, approved_by="alice", now=_T0))
    later = asyncio.run(
        store.record(
            project_id="p1",
            confidence=1.7,
            approved_by="bob",
            reason="ok",
            now=_T0 + timedelta(hours=1),
        )
    )
    assert pool.conn.rows[0]["threshold"] == Decimal("0.88")
    assert later.threshold == 1.0  # clamped to the NUMERIC(3, 2) range
    assert later.expires_at == _T0 + timedelta(hours=1, days=7)
    got = asyncio.run(store.latest_active("p1", now=_T0 + timedelta(hours=2)))
    assert (got.approved_by, got.reason) == ("bob", "ok")
    assert asyncio.run(store.latest_active("p2", now=_T0)) is None
    expired = asyncio.run(store.latest_active("p1", now=_T0 + timedelta(days=8)))
    assert expired is None


@pytest.fixture
def client(monkeypatch):
    pool = _Pool()
    reports = [SimpleNamespace(confidence=0.62, degraded=False)]

    class _Store:
        def __init__(self, _pool):
            pass

        async def list_artifacts(self, **_kw):
            return [object()], 1

    class _Verification:
        def __init__(self, _pool):
            pass

        async def list_reports(self, **_kw):
            return reports[:1]

    async def _get_pool(_request):
        return pool

    monkeypatch.setattr("umrs.knowledge.store.KnowledgeStore", _Store)
    monkeypatch.setattr(kn, "VerificationStore", _Verification)
    monkeypatch.setattr(kn, "knowledge_asyncpg_pool", _get_pool)
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    app = FastAPI()
    app.include_router(kn.router)
    test_client = TestClient(app)
    test_client.pool = pool
    test_client.reports = reports
    return test_client


def _strict_headers(monkeypatch, user: str, project: str) -> dict[str, str]:
    from r4_contracts.handshake_hmac import sign_claims

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "sec")
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims("sec", user, project, False),
    }


def test_approve_writes_a_row_and_readiness_reports_it(client, monkeypatch):
    headers = _strict_headers(monkeypatch, "alice", "proj-a")
    before = client.get("/umrs/knowledge/readiness", headers=headers).json()
    assert before["user_approved"] is False

    resp = client.post(
        "/umrs/knowledge/readiness/approve",
        json={"project_id": "proj-a", "reason": "reviewed"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["project_id"], body["approved_by"], body["threshold"]) == ("proj-a", "alice", 0.62)
    assert body["user_approved"] is True
    row = client.pool.conn.rows[0]
    assert (row["project_id"], row["approved_by"], row["reason"]) == ("proj-a", "alice", "reviewed")

    after = client.get("/umrs/knowledge/readiness", headers=headers).json()
    assert after["user_approved"] is True
    assert after["approved_at"] == body["approved_at"]
    assert after["approved_by"] == "alice"


def test_approval_does_not_leak_to_another_project(client, monkeypatch):
    a = _strict_headers(monkeypatch, "alice", "proj-a")
    assert client.post("/umrs/knowledge/readiness/approve", json={}, headers=a).status_code == 200
    b = _strict_headers(monkeypatch, "bob", "proj-b")
    assert client.get("/umrs/knowledge/readiness", headers=b).json()["user_approved"] is False


def test_approving_an_unwritable_project_fails_closed(client, monkeypatch):
    headers = _strict_headers(monkeypatch, "alice", "proj-a")
    for project in ("proj-b", "system", "user:bob"):
        resp = client.post(
            "/umrs/knowledge/readiness/approve", json={"project_id": project}, headers=headers
        )
        assert resp.status_code == 403, project
        assert resp.json()["detail"]["error"] == "scope_denied"
    assert client.pool.conn.rows == []


def test_approve_without_a_project_fails_closed(client):
    resp = client.post("/umrs/knowledge/readiness/approve", json={})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "scope_required"


def test_unauthenticated_approve_is_rejected_in_strict_mode(client, monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    resp = client.post("/umrs/knowledge/readiness/approve", json={"project_id": "proj-a"})
    assert resp.status_code == 401
    assert client.pool.conn.rows == []


def test_legacy_global_approval_is_gone(client, monkeypatch):
    headers = _strict_headers(monkeypatch, "alice", "proj-a")
    resp = client.post(
        "/umrs/knowledge/approval",
        json={"approved": True, "approved_by": "mallory"},
        headers=headers,
    )
    assert resp.status_code == 410
    assert "/umrs/knowledge/readiness/approve" in resp.json()["detail"]["message"]
    assert client.pool.conn.rows == []
    assert not hasattr(kn, "_readiness_controller")


def test_readiness_still_answers_when_the_approval_lookup_fails(client, monkeypatch):
    class _Broken:
        def __init__(self, _pool):
            pass

        async def latest_active(self, _project_id):
            raise RuntimeError("relation readiness_approvals does not exist")

    monkeypatch.setattr(kn, "ReadinessApprovalStore", _Broken)
    headers = _strict_headers(monkeypatch, "alice", "proj-a")
    resp = client.get("/umrs/knowledge/readiness", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["user_approved"] is False


def test_an_agent_actor_cannot_approve(client, monkeypatch):
    headers = _strict_headers(monkeypatch, "alice", "proj-a")
    resp = client.post(
        "/umrs/knowledge/readiness/approve",
        json={"project_id": "proj-a"},
        headers={**headers, "x-r4-actor": "agent"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == "approval_forbidden_for_agent"
    assert client.pool.conn.rows == []


@pytest.mark.parametrize(
    "report",
    [
        SimpleNamespace(confidence=0.30, degraded=False),
        SimpleNamespace(confidence=0.95, degraded=True),
    ],
    ids=["confidence-dropped", "degraded"],
)
def test_an_approval_lapses_when_readiness_falls_below_it(client, monkeypatch, report):
    headers = _strict_headers(monkeypatch, "alice", "proj-a")
    client.reports[0] = SimpleNamespace(confidence=0.876, degraded=False)
    assert (
        client.post("/umrs/knowledge/readiness/approve", json={}, headers=headers).status_code
        == 200
    )
    held = client.get("/umrs/knowledge/readiness", headers=headers).json()
    assert held["user_approved"] is True  # same confidence, at stored precision

    client.reports[0] = report
    after = client.get("/umrs/knowledge/readiness", headers=headers).json()
    assert after["readiness_level"] == "not_ready"
    assert after["user_approved"] is False
    assert after["approved_at"] is None
    assert (after["approval_stale"], after["approval_threshold"]) == (True, 0.88)


def test_an_approval_still_holds_when_confidence_rises(client, monkeypatch):
    headers = _strict_headers(monkeypatch, "alice", "proj-a")
    assert (
        client.post("/umrs/knowledge/readiness/approve", json={}, headers=headers).status_code
        == 200
    )
    client.reports[0] = SimpleNamespace(confidence=0.9, degraded=False)
    body = client.get("/umrs/knowledge/readiness", headers=headers).json()
    assert body["user_approved"] is True
    assert "approval_stale" not in body


def test_the_approve_route_is_annotated_for_the_route_audit():
    # tools/route_inventory.py fails a new unannotated mutating route.
    assert "# auth: verified" in (kn.approve_project_readiness.__doc__ or "")
