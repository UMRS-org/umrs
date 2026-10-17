"""H-1 Appendix A.2 (hardening plan, 2026-10-17): knowledge objects acted on
by id without a project check, and one write side effect before the scope
check.

- ``POST /umrs/knowledge/approve`` read any project's report and findings.
- ``PATCH /umrs/knowledge/findings/{finding_key}`` wrote any project's finding.
- ``POST /umrs/knowledge/graphs/{graph_id}/nodes|edges/{id}/establish``
  ignored ``graph_id`` and promoted any node/edge.
- ``GET /umrs/knowledge/graphs?project_id=`` materialized the named
  project's default graph before any scope check.
- ``GET /umrs/knowledge/proposals/stats`` without ``graph_id`` counted the
  proposals of every graph.

A foreign object reads as not found (no disclosure).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.knowledge.graph_registry import KnowledgeGraphMeta, default_graph_id

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-knowledge-objects-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _headers("alice", "proj-a")
BOB = _headers("bob", "proj-b")


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


# ── /knowledge/approve and /knowledge/findings/{key} ─────────────────────────


class _FakeVerificationStore:
    """VerificationStore double: two reports, one finding each."""

    reports = {
        "rep-a": SimpleNamespace(report_id="rep-a", project_id="proj-a", confidence=0.9),
        "rep-b": SimpleNamespace(report_id="rep-b", project_id="proj-b", confidence=0.9),
    }
    findings = {
        "f-a": "rep-a",
        "f-b": "rep-b",
    }
    transitions: list[str] = []

    def __init__(self, _pool) -> None:
        pass

    async def get_report(self, report_id):
        return self.reports.get(report_id)

    async def list_blocking_findings(self, _report_id):
        return []

    async def get_finding(self, finding_key):
        if finding_key not in self.findings:
            raise ValueError(f"Finding '{finding_key}' not found")
        return self._finding(finding_key, "open")

    async def transition_finding(self, *, finding_key, to_status, **_kwargs):
        self.transitions.append(finding_key)
        return self._finding(finding_key, to_status)

    async def waive_finding(self, *, finding_key, **_kwargs):
        self.transitions.append(finding_key)
        return self._finding(finding_key, "waived")

    def _finding(self, key, status):
        return SimpleNamespace(
            finding_key=key, report_id=self.findings[key], status=status, updated_at=None
        )


@pytest.fixture
def knowledge(monkeypatch):
    from umrs.api import knowledge as mod

    _FakeVerificationStore.transitions = []

    async def _pool(_request):
        return object()

    monkeypatch.setattr(mod, "knowledge_asyncpg_pool", _pool)
    monkeypatch.setattr(mod, "VerificationStore", _FakeVerificationStore)
    app = FastAPI()
    app.include_router(mod.router)
    return TestClient(app)


def test_approve_reads_only_the_callers_report(knowledge):
    assert (
        knowledge.post(
            "/umrs/knowledge/approve", json={"report_id": "rep-a"}, headers=ALICE
        ).status_code
        == 200
    )
    foreign = knowledge.post("/umrs/knowledge/approve", json={"report_id": "rep-b"}, headers=ALICE)
    assert foreign.status_code == 404


def test_approve_needs_identity(knowledge):
    assert knowledge.post("/umrs/knowledge/approve", json={"report_id": "rep-a"}).status_code == 401


def test_admin_approves_any_report(knowledge):
    admin = _headers("root", "ops", admin=True)
    assert (
        knowledge.post(
            "/umrs/knowledge/approve", json={"report_id": "rep-b"}, headers=admin
        ).status_code
        == 200
    )


def _transition(client, key, headers):
    body = {"finding_key": key, "from_status": "open", "new_status": "acknowledged"}
    return client.patch(f"/umrs/knowledge/findings/{key}", json=body, headers=headers)


def test_finding_of_another_project_is_not_found(knowledge):
    assert _transition(knowledge, "f-b", ALICE).status_code == 404
    assert _FakeVerificationStore.transitions == []


def test_finding_of_own_project_transitions(knowledge):
    resp = _transition(knowledge, "f-a", ALICE)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "acknowledged"
    assert _FakeVerificationStore.transitions == ["f-a"]


def test_unknown_finding_is_not_found(knowledge):
    assert _transition(knowledge, "f-zzz", ALICE).status_code == 404


# ── establish nodes / edges ──────────────────────────────────────────────────

G_A = "g-aaaa"
G_B = "g-bbbb"


def _meta(graph_id: str, project: str, kind: str = "research") -> KnowledgeGraphMeta:
    return KnowledgeGraphMeta(
        graph_id=graph_id, name=graph_id, owner_id="", project_id=project, kind=kind
    )


class _FakeRegistry:
    def __init__(self, graphs: dict[str, KnowledgeGraphMeta]) -> None:
        self.graphs = graphs
        self.ensured: list[str] = []

    def get_graph(self, graph_id):
        return self.graphs.get(graph_id)

    def ensure_default_graph(self, project_id, **_kwargs):
        self.ensured.append(project_id)
        return _meta(default_graph_id(project_id), project_id, kind="project")

    def list_graphs(self, *, project_id=None, **_kwargs):
        return [g for g in self.graphs.values() if not project_id or g.project_id == project_id]

    def search_graphs(self, _q, *, project_id=None, **_kwargs):
        return self.list_graphs(project_id=project_id)

    def list_shared_graphs(self, **_kwargs):
        return []


class _AsyncConn:
    """asyncpg double over node/edge rows {id: {status, project_id, graph_id}}."""

    def __init__(self, rows):
        self.rows = rows
        self.updates: list[str] = []

    async def fetchrow(self, _sql, *params):
        return self.rows.get(params[0])

    async def fetchval(self, _sql, *_params):
        return {}

    async def execute(self, _sql, *params):
        self.updates.append(params[0])

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _AsyncPool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return self.conn


@pytest.fixture
def graphs(monkeypatch):
    from umrs.api import _asyncpg_pool
    from umrs.api import knowledge_graphs as mod

    rows = {
        "n-a": {"status": "proposed", "project_id": "proj-a", "graph_id": G_A},
        "n-b": {"status": "proposed", "project_id": "proj-b", "graph_id": G_B},
        "n-legacy": {"status": "proposed", "project_id": "proj-a", "graph_id": ""},
        "e-a": {"status": "proposed", "project_id": "proj-a", "graph_id": G_A},
        "e-b": {"status": "proposed", "project_id": "proj-b", "graph_id": G_B},
    }
    conn = _AsyncConn(rows)

    async def _pool(_request):
        return _AsyncPool(conn)

    registry = _FakeRegistry({G_A: _meta(G_A, "proj-a"), G_B: _meta(G_B, "proj-b")})
    monkeypatch.setattr(_asyncpg_pool, "knowledge_asyncpg_pool", _pool)
    monkeypatch.setattr(mod, "_registry", lambda _request: registry)
    app = FastAPI()
    app.include_router(mod.router)
    return TestClient(app), conn, registry


def _establish(client, graph_id, kind, row_id, headers):
    return client.post(
        f"/umrs/knowledge/graphs/{graph_id}/{kind}/{row_id}/establish", headers=headers
    )


def test_establish_node_of_another_project_is_not_found(graphs):
    client, conn, _ = graphs
    assert _establish(client, G_B, "nodes", "n-b", ALICE).status_code == 404
    assert conn.updates == []


def test_establish_checks_the_row_belongs_to_the_graph(graphs):
    """A foreign node named under the caller's own graph is still not found."""
    client, conn, _ = graphs
    assert _establish(client, G_A, "nodes", "n-b", ALICE).status_code == 404
    assert _establish(client, G_A, "edges", "e-b", ALICE).status_code == 404
    assert conn.updates == []


def test_establish_edge_of_another_project_is_not_found(graphs):
    client, conn, _ = graphs
    assert _establish(client, G_B, "edges", "e-b", ALICE).status_code == 404
    assert conn.updates == []


def test_establish_own_node_and_edge(graphs):
    client, conn, _ = graphs
    assert _establish(client, G_A, "nodes", "n-a", ALICE).status_code == 200
    assert _establish(client, G_A, "edges", "e-a", ALICE).status_code == 200
    assert conn.updates == ["n-a", "e-a"]


def test_establish_legacy_row_through_the_default_graph(graphs):
    """Legacy rows (graph_id '') belong to their project's default graph,
    which need not be materialized in the registry yet."""
    client, conn, _ = graphs
    default_a = default_graph_id("proj-a")
    assert _establish(client, default_a, "nodes", "n-legacy", ALICE).status_code == 200
    assert _establish(client, default_a, "nodes", "n-legacy", BOB).status_code == 404
    assert conn.updates == ["n-legacy"]


def test_establish_needs_identity(graphs):
    client, conn, _ = graphs
    assert _establish(client, G_A, "nodes", "n-a", {}).status_code == 401
    assert conn.updates == []


# ── GET /umrs/knowledge/graphs default-graph side effect ─────────────────────


def test_listing_another_project_does_not_materialize_its_default_graph(graphs):
    client, _, registry = graphs
    resp = client.get("/umrs/knowledge/graphs", params={"project_id": "proj-b"}, headers=ALICE)
    assert resp.status_code == 200
    assert registry.ensured == []
    assert resp.json()["graphs"] == []


def test_listing_without_identity_writes_nothing(graphs):
    client, _, registry = graphs
    assert client.get("/umrs/knowledge/graphs", params={"project_id": "proj-b"}).status_code == 401
    assert registry.ensured == []


def test_listing_own_project_materializes_its_default_graph(graphs):
    client, _, registry = graphs
    resp = client.get("/umrs/knowledge/graphs", params={"project_id": "proj-a"}, headers=ALICE)
    assert resp.status_code == 200
    assert registry.ensured == ["proj-a"]


# ── GET /umrs/knowledge/proposals/stats ──────────────────────────────────────


def _stats_client(script):
    from umrs.api import knowledge_proposals as mod

    app = FastAPI()
    app.include_router(mod.router)
    app.state.umrs_db_pool = FakePool(script)
    return TestClient(app), app.state.umrs_db_pool


def _counted(graph_id, project, accepted, declined, pending, access=None):
    """One per-graph row: proposal counts joined with the graph's access fields."""
    return {
        "target_graph_id": graph_id,
        "accepted": accepted,
        "declined": declined,
        "pending": pending,
        "kind": "research",
        "project_id": project,
        "owner_id": "",
        "access": access or [],
    }


def _per_graph_rows():
    # proj-b first: an unscoped single-row read would report proj-b's counts.
    return [_counted(G_B, "proj-b", 5, 3, 1), _counted(G_A, "proj-a", 1, 0, 2)]


def test_stats_without_graph_count_only_readable_graphs():
    client, pool = _stats_client([_per_graph_rows()])
    resp = client.get("/umrs/knowledge/proposals/stats", headers=ALICE)
    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "accepted": 1,
        "declined": 0,
        "pending": 2,
        "decided": 1,
        "acceptance_rate": 1.0,
    }
    # Review (2026-10-18): one query carries counts and graph access fields,
    # never one registry lookup per graph across all projects.
    assert len(pool.record) == 1
    assert "JOIN knowledge_graphs" in pool.record[0][0]


def test_stats_count_a_graph_shared_with_the_caller():
    rows = [_counted(G_B, "proj-b", 5, 3, 1, access=["user:alice"])]
    client, _pool = _stats_client([rows])
    resp = client.get("/umrs/knowledge/proposals/stats", headers=ALICE)
    assert resp.status_code == 200, resp.text
    assert resp.json()["accepted"] == 5
    assert resp.json()["decided"] == 8


def test_stats_without_any_readable_graph_is_empty():
    client, _pool = _stats_client([_per_graph_rows()[:1]])
    resp = client.get("/umrs/knowledge/proposals/stats", headers=ALICE)
    assert resp.status_code == 200, resp.text
    assert resp.json()["decided"] == 0
    assert resp.json()["acceptance_rate"] is None


def test_admin_stats_stay_global():
    client, _pool = _stats_client([{"accepted": 6, "declined": 3, "pending": 3}])
    resp = client.get(
        "/umrs/knowledge/proposals/stats", headers=_headers("root", "ops", admin=True)
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["accepted"] == 6
