"""K3 facade — graph backend wiring, selection, and provenance (0069)."""

from __future__ import annotations

import pytest
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.knowledge_query import router
from umrs.knowledge.graph_registry import default_graph_id


class _FakeMetaStore:
    def search(self, project_id: str, query: str, *, limit: int = 20, **_f):
        del project_id, query, limit  # metadata backend not exercised in these tests
        return []


@pytest.fixture(autouse=True)
def _catalog(monkeypatch):
    monkeypatch.setattr(
        "umrs.api.knowledge_query.get_optional_catalog_meta_store",
        lambda: _FakeMetaStore(),
    )


def _client(script) -> tuple[TestClient, FakePool]:
    app = FastAPI()
    app.include_router(router)
    pool = FakePool(script)
    app.state.umrs_db_pool = pool
    return TestClient(app), pool


def _graph_row(**overrides):
    row = {
        "graph_id": "g-exp",
        "name": "experiment-history",
        "description": "Historical experiments",
        "owner_id": "alice",
        "project_id": "proj-a",
        "kind": "research",
        "status": "active",
        "ontology_id": None,
        "ontology_version": None,
        "provenance": {},
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


_NODE = ("node:ir-1", "proj-a", "FWI inversion solver", "class", {"filename": "solver.py"}, "g-exp")


def _query(client: TestClient, **params) -> object:
    params.setdefault("q", "FWI")
    return client.get(
        "/umrs/knowledge/query",
        params=params,
        headers={"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"},
    )


class TestGraphBackendWiring:
    def test_default_project_graph_wires_backend(self):
        default_id = default_graph_id("proj-a")
        client, pool = _client(
            [
                None,  # ensure_default insert
                _graph_row(graph_id=default_id, name="proj-a"),  # ensure re-read
                [("node:ir-9", "proj-a", "FWI notes", "concept", {}, "g-legacy")],
            ]
        )
        resp = _query(client, project_id="proj-a")
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert "knowledge_graph" in payload["wired_backends"]
        assert payload["selected_graphs"] == [default_id]
        assert payload["unavailable_backends"] == {
            "embeddings": "query-embedding generation requires the embedding service",
            "rag": "RAG search requires a deployed mini-rivault instance",
        }

    def test_explicit_graph_selection_carries_provenance(self):
        client, pool = _client(
            [
                None,  # resolve 'experiment-history': by id
                _graph_row(),  # resolve: by name
                [_NODE],  # graph search rows
            ]
        )
        resp = _query(
            client, project_id="proj-a", graphs="experiment-history", scope="knowledge_graph"
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert payload["selected_graphs"] == ["g-exp"]
        hit = payload["results"][0]
        assert hit["metadata"]["graph_id"] == "g-exp"
        assert hit["metadata"]["graph_name"] == "experiment-history"
        assert hit["metadata"]["node_type"] == "class"
        assert hit["source_type"] == "knowledge_graph_node"
        # SQL scoped to the resolved graph id
        search_sql, params = pool.record[2]
        assert "graph_id = ANY" in search_sql
        assert params[0] == ["g-exp"]

    def test_legacy_rows_visible_through_default_graph(self):
        default_id = default_graph_id("proj-a")
        client, pool = _client(
            [
                None,
                _graph_row(graph_id=default_id, name="proj-a"),
                [("node:ir-2", "proj-a", "FWI paper", "concept", {}, "")],
            ]
        )
        resp = _query(client, project_id="proj-a", scope="knowledge_graph")
        assert resp.status_code == 200, resp.text
        hit = resp.json()["query"]["results"][0]
        # Legacy '' row attributed to the default graph
        assert hit["metadata"]["graph_id"] == default_id
        search_sql, params = pool.record[2]
        assert "graph_id = '' AND project_id = ANY" in search_sql
        assert params[1] == ["proj-a"]

    def test_exclusion_wins_and_causes_honest_unavailability(self):
        client, _ = _client([_graph_row()])
        resp = _query(
            client,
            project_id="proj-a",
            graphs="experiment-history",
            exclude_graphs="experiment-history",
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert payload["selected_graphs"] == []
        assert payload["excluded_graphs"] == ["experiment-history"]
        assert "knowledge_graph" in payload["unavailable_backends"]

    def test_unknown_include_is_skipped(self):
        client, _ = _client([None, None])
        resp = _query(client, project_id="proj-a", graphs="g-missing")
        assert resp.status_code == 200, resp.text
        assert resp.json()["query"]["selected_graphs"] == []

    def test_no_pool_keeps_honest_reason(self):
        app = FastAPI()
        app.include_router(router)
        resp = _query(TestClient(app), project_id="proj-a")
        assert resp.status_code == 200
        payload = resp.json()["query"]
        assert payload["wired_backends"] == ["catalog"]
        assert "requires the database" in payload["unavailable_backends"]["knowledge_graph"]
