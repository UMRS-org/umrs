"""Route-level tests for the K3 unified knowledge query facade."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.knowledge_query import router
from umrs.data_catalog.schema import CatalogObjectMeta
from umrs.data_catalog.scoring import CatalogHit


@pytest.fixture
def app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


class _FakeMetaStore:
    """Minimal CatalogMetaStore stand-in for facade tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def search(self, project_id: str, query: str, *, limit: int = 20, **_filters):
        self.calls.append((project_id, query, limit))
        if project_id != "proj-a":
            return []
        meta = CatalogObjectMeta.pending(
            s3_key="proj-a/docs/ct.tif",
            object_etag="etag-1",
            owner="alice",
            acl_scope="proj-a",
            filename="ct.tif",
        )
        meta.detected_modality = "image"
        meta.summary = "CT scan of sample S12"
        return [CatalogHit(meta=meta, score=0.92)]


def _headers(user: str = "alice", project: str = "proj-a") -> dict[str, str]:
    return {"X-User-ID": user, "X-Verified-Project-ID": project}


@pytest.fixture
def meta_store(monkeypatch) -> _FakeMetaStore:
    store = _FakeMetaStore()
    monkeypatch.setattr(
        "umrs.api.knowledge_query.get_optional_catalog_meta_store",
        lambda: store,
    )
    return store


@pytest.mark.usefixtures("meta_store")
class TestKnowledgeQueryFacade:
    def test_503_when_catalog_store_unavailable(self, client, monkeypatch):
        monkeypatch.setattr(
            "umrs.api.knowledge_query.get_optional_catalog_meta_store",
            lambda: None,
        )
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct", "project_id": "proj-a"},
            headers=_headers(),
        )
        assert resp.status_code == 503

    def test_happy_path_returns_normalized_results(self, client):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct scan", "project_id": "proj-a"},
            headers=_headers(),
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()["query"]
        assert payload["query"] == "ct scan"
        assert payload["total_results"] == 1

        _assert_hit_shape(payload["results"][0])

        # Coverage truth: catalog wired, others honestly unavailable.
        assert payload["wired_backends"] == ["catalog"]
        assert set(payload["unavailable_backends"].keys()) == {
            "knowledge_graph",
            "embeddings",
            "rag",
        }


def _assert_hit_shape(hit: dict) -> None:
    """Normalized search_api shape for a catalog hit."""
    assert hit["title"] == "ct.tif"
    assert hit["source_id"] == "proj-a/docs/ct.tif"
    assert hit["result_type"] == "catalog_hit"
    assert hit["score"] == 0.92


@pytest.mark.usefixtures("meta_store")
class TestKnowledgeQueryFacadeEdge:
    def test_unknown_scope_is_400(self, client):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct", "project_id": "proj-a", "scope": "nope"},
            headers=_headers(),
        )
        assert resp.status_code == 400

    def test_missing_project_is_401_via_identity_gate(self, client):
        # Identity gate (get_caller_identity) fires before the route's own
        # project check: with no verified project header and no legacy query
        # params the caller identity itself is incomplete → 401, never 400.
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct"},
            headers={"X-User-ID": "alice"},
        )
        assert resp.status_code == 401

    def test_verified_project_header_overrides_param(self, client, meta_store):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct", "project_id": "other-project"},
            headers=_headers(project="proj-a"),
        )
        assert resp.status_code == 200
        assert meta_store.calls[-1][0] == "proj-a"

    def test_catalog_scope_matches(self, client):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct", "project_id": "proj-a", "scope": "catalog"},
            headers=_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["query"]["scope"] == "catalog"


class TestOntologyBlocks:
    def test_ontologies_blocks_per_selected_graph(self, client, meta_store, monkeypatch):
        """ADR-049 Leg B: response carries the bound ontology per selected
        graph as its own block; anunknown reference is reported via
        ontology_error, never smudged."""
        from umrs.knowledge.graph_registry import KnowledgeGraphMeta

        metas = [
            KnowledgeGraphMeta(
                graph_id="g-default",
                name="default",
                project_id="proj-a",
                ontology_id="r4-default",
            ),
            KnowledgeGraphMeta(
                graph_id="g-bogus",
                name="bogus",
                project_id="proj-a",
                ontology_id="never-registered",
            ),
        ]

        class _FakeRegistry:
            def ensure_default_graph(self, project_id):
                return metas[0]

        monkeypatch.setattr(
            "umrs.api.knowledge_query._resolve_selected_graphs",
            lambda *_args, **_kwargs: (metas, []),
        )
        # Pool slot occupied so the graph backend wires — psycopg shape
        # (getconn/cursor) returning zero rows; registry resolution is
        # monkeypatched above.
        from unittest.mock import MagicMock

        pool = MagicMock()
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        cursor.fetchone.return_value = None  # ontology registry: no live row
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        pool.getconn.return_value = conn
        client.app.state.umrs_db_pool = pool
        try:
            resp = client.get(
                "/umrs/knowledge/query",
                params={"q": "ct", "project": "proj-a"},
                headers={"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"},
            )
        finally:
            client.app.state.umrs_db_pool = None

        assert resp.status_code == 200
        blocks = resp.json()["query"]["ontologies"]
        by_id = {b["graph_id"]: b for b in blocks}
        assert set(by_id) == {"g-default", "g-bogus"}
        assert by_id["g-default"]["ontology"]["ontology_id"] == "r4-default"
        assert "transitive_relations" in by_id["g-default"]["ontology"]
        assert "ontology_error" in by_id["g-bogus"]
