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

    def test_missing_project_is_400(self, client):
        resp = client.get(
            "/umrs/knowledge/query",
            params={"q": "ct"},
            headers={"X-User-ID": "alice"},
        )
        assert resp.status_code == 400

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
