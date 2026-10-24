"""K1.5 minimal review (knowledge plan 2026-10-24): the proposed rows of a
graph, read through ``GET /umrs/knowledge/graphs/{graph_id}/proposed``.

No route exposed the D-35 tier before: a proposed loop finding could be
promoted by id but never listed, so nobody could find it to review.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.knowledge.graph_registry import KnowledgeGraphMeta, default_graph_id
from umrs.knowledge.review_queue import ReviewQueueStore, row_origin, row_run_id

from r4_contracts.handshake_hmac import sign_claims

SECRET = "k15-review-queue-secret"
G_A = "g-review-a"
G_B = "g-review-b"


def _headers(user: str, project: str) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, False),
    }


ALICE = _headers("alice", "proj-a")


def _meta(graph_id: str, project: str) -> KnowledgeGraphMeta:
    return KnowledgeGraphMeta(
        graph_id=graph_id, name=graph_id, owner_id="", project_id=project, kind="research"
    )


# ── row helpers ─────────────────────────────────────────────────────────────


def test_a_loop_row_names_its_loop_as_run_and_origin():
    content = {"statement": "x", "source_loop": "loop-7"}
    assert row_run_id(content) == "loop-7"
    assert row_origin(content) == "loop"


def test_a_recorded_actor_wins_and_a_bare_row_names_nothing():
    assert row_origin({"origin_actor": "agent", "source_loop": "l"}) == "agent"
    assert row_origin({}) == "" and row_run_id({}) == ""
    assert row_run_id({"job_id": "job-1", "artifact_id": "art-1"}) == "job-1"


# ── the store's SQL (cursor double) ─────────────────────────────────────────


class _Cursor:
    def __init__(self, results: list[list[dict[str, Any]]]) -> None:
        self.results = results
        self.calls: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple) -> None:
        self.calls.append((sql, params))

    def fetchall(self) -> list[dict[str, Any]]:
        return self.results.pop(0)


class _CursorStore(ReviewQueueStore):
    def __init__(self, cursor: _Cursor) -> None:
        super().__init__(None)
        self.cursor = cursor

    @contextmanager
    def _cursor(self):
        yield self.cursor


def test_the_store_reads_only_proposed_rows_of_the_graph():
    cursor = _Cursor([[], []])
    _CursorStore(cursor).proposed_rows(_meta(G_A, "proj-a"), limit=5)
    (nodes_sql, params), (edges_sql, _) = cursor.calls
    assert "status = 'proposed'" in nodes_sql and "status = 'proposed'" in edges_sql
    assert params == (G_A, False, "proj-a", None, None, 5)


def test_the_default_graph_keeps_the_legacy_rows():
    cursor = _Cursor([[], []])
    default_a = default_graph_id("proj-a")
    _CursorStore(cursor).proposed_rows(_meta(default_a, "proj-a"))
    assert cursor.calls[0][1] == (default_a, True, "proj-a", None, None, 200)


def test_rows_carry_kind_summary_origin_and_run():
    node = {
        "node_id": "art-f1",
        "label": "finding:art-f1",
        "node_type": "concept",
        "confidence": 0.0,
        "content": {"statement": "MOF-5 binds CO2", "source_loop": "loop-7"},
        "created_at": None,
        "graph_id": G_A,
    }
    edge = {
        "edge_id": "e-1",
        "source_id": "art-f1",
        "target_id": "loop:loop-7",
        "relation_type": "produced_by",
        "confidence": None,
        "content": {"iteration": 1},
        "created_at": None,
        "graph_id": G_A,
    }
    out = _CursorStore(_Cursor([[node], [edge]])).proposed_rows(_meta(G_A, "proj-a"))
    first = out["nodes"][0]
    assert (first["kind"], first["id"], first["summary"]) == ("node", "art-f1", "MOF-5 binds CO2")
    assert (first["origin"], first["run_id"]) == ("loop", "loop-7")
    assert out["edges"][0]["kind"] == "edge" and out["edges"][0]["source_id"] == "art-f1"
    assert out["truncated"] is False


# ── the route ───────────────────────────────────────────────────────────────


class _FakeRegistry:
    def __init__(self) -> None:
        self.graphs = {G_A: _meta(G_A, "proj-a"), G_B: _meta(G_B, "proj-b")}

    def resolve_graph(self, graph_id: str):
        return self.graphs.get(graph_id)


class _FakeQueue:
    def __init__(self) -> None:
        self.read: list[tuple[str, int]] = []

    def proposed_rows(self, graph, *, limit, run_id=None):
        self.read.append((graph.graph_id, limit) if run_id is None else (graph.graph_id, limit, run_id))
        node = {"kind": "node", "id": "art-f1", "run_id": "loop-7", "origin": "loop"}
        return {"nodes": [node], "edges": [], "truncated": False}


@pytest.fixture
def client(monkeypatch):
    from umrs.api import knowledge_graphs as mod

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    queue = _FakeQueue()
    monkeypatch.setattr(mod, "_registry", lambda _request: _FakeRegistry())
    monkeypatch.setattr(mod, "_review_queue", lambda _request: queue)
    app = FastAPI()
    app.include_router(mod.router)
    return TestClient(app), queue


def test_a_member_lists_the_proposed_rows_of_their_graph(client):
    http, queue = client
    resp = http.get(f"/umrs/knowledge/graphs/{G_A}/proposed", headers=ALICE)
    assert http.get(f"/umrs/knowledge/graphs/{G_A}/proposed?limit=5", headers=ALICE).is_success
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["graph_id"] == G_A and body["project_id"] == "proj-a"
    assert body["nodes"][0]["run_id"] == "loop-7"
    assert queue.read == [(G_A, 200), (G_A, 5)]


def test_the_route_narrows_to_one_run(client):
    """Review of K1 part A: a run past the newest-``limit`` window was
    unreachable, so ``establish --run`` promoted part of it or nothing."""
    http, queue = client
    resp = http.get(f"/umrs/knowledge/graphs/{G_A}/proposed?run_id=loop-7", headers=ALICE)
    assert resp.status_code == 200, resp.text
    assert queue.read == [(G_A, 200, "loop-7")]


def test_the_store_filters_by_the_run_key_row_run_id_reads():
    cursor = _Cursor([[], []])
    _CursorStore(cursor).proposed_rows(_meta(G_A, "proj-a"), run_id="loop-7")
    sql, params = cursor.calls[0]
    assert params == (G_A, False, "proj-a", "loop-7", "loop-7", 200)
    for key in ("source_loop", "loop_id", "job_id", "artifact_id"):
        assert f"content->>'{key}'" in sql


def test_a_model_extraction_names_its_origin():
    """K-1 rows (entities a model extracted from a person's upload) said
    nothing about where they came from: the inbox printed ``origin=-``."""
    assert row_origin({"semantic": True, "artifact_id": "art-1"}) == "extracted"
    assert row_origin({"semantic_type": "entity", "artifact_id": "art-1"}) == "extracted"
    assert row_origin({"origin_actor": "extracted", "artifact_id": "art-1"}) == "extracted"


def test_the_inbox_needs_a_verified_caller(client):
    http, queue = client
    assert http.get(f"/umrs/knowledge/graphs/{G_A}/proposed").status_code == 401
    assert queue.read == []


def test_another_projects_graph_reads_as_not_found(client):
    http, queue = client
    assert http.get(f"/umrs/knowledge/graphs/{G_B}/proposed", headers=ALICE).status_code == 404
    assert http.get("/umrs/knowledge/graphs/g-none/proposed", headers=ALICE).status_code == 404
    assert queue.read == []


# ── a real Postgres (skipped without a database URL) ─────────────────────────
#
# Not marked ``integration``: the conftest skips that marker unless
# UMRS_TEST_DB_URL is set, which the CI job does not set; the fixture reads
# the CI-shape POSTGRES_* variables instead (review of K1 part A).


@pytest.fixture()
def pg_conn(request: pytest.FixtureRequest) -> Any:
    psycopg2 = pytest.importorskip("psycopg2")
    url = (
        request.config.getoption("--db-url")
        or os.environ.get("UMRS_TEST_DB_URL")
        or _ci_postgres_url()
    )
    if not url:
        pytest.skip("no database URL (--db-url / UMRS_TEST_DB_URL)")
    conn = psycopg2.connect(url)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def _ci_postgres_url() -> str:
    """The CI-shape Postgres (``POSTGRES_HOST`` …), which the CI job sets;
    ``UMRS_TEST_DB_URL`` alone left this test skipped there."""
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return ""
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


class _OneConn:
    """The test's own transaction: the store's ``with conn`` never commits."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def getconn(self) -> Any:
        return _NoCommit(self.conn)

    def putconn(self, _conn: Any) -> None:
        return None


class _NoCommit:
    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def __enter__(self) -> Any:
        return self.conn

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def cursor(self, **kwargs: Any) -> Any:
        return self.conn.cursor(**kwargs)


def _insert_node(cur: Any, node_id: str, graph_id: str, status: str, content: dict) -> None:
    cur.execute(
        "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, content, "
        "content_hash, graph_id, status) VALUES (%s, 'rq-it', %s, 'concept', %s::jsonb, "
        "'h', %s, %s)",
        (node_id, node_id, json.dumps(content), graph_id, status),
    )


def test_the_sql_returns_only_proposed_rows_of_the_graph(pg_conn: Any) -> None:
    default_graph = default_graph_id("rq-it")
    with pg_conn.cursor() as cur:
        _insert_node(cur, "rq-prop", default_graph, "proposed", {"source_loop": "loop-1"})
        _insert_node(cur, "rq-est", default_graph, "established", {})
        _insert_node(cur, "rq-legacy", "", "proposed", {})
        _insert_node(cur, "rq-other", "g-rq-other", "proposed", {})
        cur.execute(
            "INSERT INTO knowledge_edges (edge_id, project_id, source_id, target_id, "
            "relation_type, graph_id, status) VALUES ('rq-e', 'rq-it', 'rq-prop', "
            "'rq-est', 'supports', %s, 'proposed')",
            (default_graph,),
        )
    store = ReviewQueueStore(_OneConn(pg_conn))
    out = store.proposed_rows(_meta(default_graph, "rq-it"))
    assert sorted(n["id"] for n in out["nodes"]) == ["rq-legacy", "rq-prop"]
    assert [e["id"] for e in out["edges"]] == ["rq-e"]
    assert next(n for n in out["nodes"] if n["id"] == "rq-prop")["run_id"] == "loop-1"


def test_a_run_is_reachable_past_the_newest_rows(pg_conn: Any) -> None:
    default_graph = default_graph_id("rq-it")
    with pg_conn.cursor() as cur:
        _insert_node(cur, "rq-finding", default_graph, "proposed", {"source_loop": "loop-9"})
        for n in range(5):
            _insert_node(
                cur, f"rq-sem-{n}", default_graph, "proposed",
                {"semantic_type": "entity", "artifact_id": "art-paper"},
            )
        cur.execute(
            "UPDATE knowledge_nodes SET created_at = now() - interval '1 day' "
            "WHERE node_id = 'rq-finding'"
        )
    store = ReviewQueueStore(_OneConn(pg_conn))
    window = store.proposed_rows(_meta(default_graph, "rq-it"), limit=3)
    assert "rq-finding" not in {n["id"] for n in window["nodes"]} and window["truncated"]
    run = store.proposed_rows(_meta(default_graph, "rq-it"), limit=3, run_id="loop-9")
    assert [n["id"] for n in run["nodes"]] == ["rq-finding"]
    paper = store.proposed_rows(_meta(default_graph, "rq-it"), run_id="art-paper")
    assert {n["origin"] for n in paper["nodes"]} == {"extracted"}
