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
                None,  # 0075 regclass probe (embeddings unwired here)
                None,  # WP-15: knowledge_nodes regclass probe (rag unwired here)
                [("node:ir-9", "proj-a", "FWI notes", "concept", {}, "g-legacy")],
            ]
        )
        resp = _query(client, project_id="proj-a")
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert "knowledge_graph" in payload["wired_backends"]
        assert payload["selected_graphs"] == [default_id]
        assert payload["unavailable_backends"] == {
            # P2 wave 2: 0075 probe drives two honest states for the
            # embeddings backend (missing migration here via FakePool).
            "embeddings": "embeddings search unavailable (migration 0075 not applied)",
            # WP-15: the rag scope is wired over the central corpus; absent
            # knowledge_nodes it names the corpus honestly (never the
            # endpoint-coupled corpus as the default truth).
            "rag": "central knowledge corpus unavailable (knowledge nodes not present)",
        }

    def test_explicit_graph_selection_carries_provenance(self):
        client, pool = _client(
            [
                None,  # resolve 'experiment-history': by id
                _graph_row(),  # resolve: by name
                None,  # 0075 regclass probe (embeddings unwired here)
                None,  # WP-15: knowledge_nodes regclass probe (rag unwired here)
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
        search_sql, params = pool.record[4]
        assert "graph_id = ANY" in search_sql
        assert params[0] == ["g-exp"]

    def test_legacy_rows_visible_through_default_graph(self):
        default_id = default_graph_id("proj-a")
        client, pool = _client(
            [
                None,
                _graph_row(graph_id=default_id, name="proj-a"),
                None,  # 0075 regclass probe (embeddings unwired here)
                None,  # WP-15: knowledge_nodes regclass probe (rag unwired here)
                [("node:ir-2", "proj-a", "FWI paper", "concept", {}, "")],
            ]
        )
        resp = _query(client, project_id="proj-a", scope="knowledge_graph")
        assert resp.status_code == 200, resp.text
        hit = resp.json()["query"]["results"][0]
        # Legacy '' row attributed to the default graph
        assert hit["metadata"]["graph_id"] == default_id
        search_sql, params = pool.record[4]
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


class TestGraphReadAccess:
    """KG-05 (2026-10-09): `graphs=` resolves an id, or a name across every
    project (default graphs are named after their project), and the selection
    was never access-checked: any caller could read another project's nodes
    by naming its graph. A graph the caller may not read is now skipped like
    an unknown one, so its existence is not disclosed either."""

    def test_another_projects_graph_is_skipped_like_an_unknown_one(self):
        foreign = _graph_row(graph_id="g-b", name="proj-b", project_id="proj-b", owner_id="bob")
        client, _ = _client([foreign])
        resp = _query(client, project_id="proj-a", graphs="g-b")
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert payload["selected_graphs"] == [] and payload["ontologies"] == []
        assert "g-b" not in resp.text

    def test_a_graph_shared_with_the_callers_project_is_selected(self):
        shared = _graph_row(
            graph_id="g-b",
            name="proj-b",
            project_id="proj-b",
            owner_id="bob",
            access=["project:proj-a"],
        )
        client, _ = _client([shared])
        resp = _query(client, project_id="proj-a", graphs="g-b")
        assert resp.status_code == 200, resp.text
        assert resp.json()["query"]["selected_graphs"] == ["g-b"]

    def test_a_shared_name_resolves_to_the_callers_own_graph(self):
        """Names are not unique across projects; another project's older graph
        of the same name must neither leak nor hide the caller's own."""
        foreign = _graph_row(graph_id="g-b", name="batteries", project_id="proj-b", owner_id="bob")
        own = _graph_row(graph_id="g-a", name="batteries")
        client, _ = _client([None, [foreign, own]])  # no id match, then every name match
        resp = _query(client, project_id="proj-a", graphs="batteries")
        assert resp.status_code == 200, resp.text
        assert resp.json()["query"]["selected_graphs"] == ["g-a"]

    def test_a_shared_graph_exposes_none_of_its_owners_legacy_rows(self):
        """Legacy rows (graph_id '') belong to the owner's DEFAULT graph only.
        Sharing a non-default graph exposed every one of them (review, 2026-10-09)."""
        shared = _graph_row(graph_id="g-shared", name="shared", access=["project:proj-b"])
        client, pool = _client([shared])
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "secret", "graphs": "g-shared", "scope": "knowledge_graph"},
            headers={"X-User-ID": "bob", "X-Verified-Project-ID": "proj-b"},
        )
        assert resp.status_code == 200, resp.text
        node_query = next(p for sql, p in pool.record if "graph_id = ANY" in sql)
        assert node_query[0] == ["g-shared"] and node_query[1] == []

    def test_an_excluded_name_stays_excluded_whatever_project_holds_it(self):
        foreign = _graph_row(graph_id="g-b", name="batteries", project_id="proj-b", owner_id="bob")
        own = _graph_row(graph_id="g-a", name="batteries")
        # exclusions resolve first (id miss, every name match), then includes
        client, _ = _client([None, [foreign, own], None, [foreign, own]])
        resp = _query(client, project_id="proj-a", graphs="batteries", exclude_graphs="batteries")
        assert resp.status_code == 200, resp.text
        assert resp.json()["query"]["selected_graphs"] == []
