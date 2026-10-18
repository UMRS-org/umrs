"""
Pipeline and RAG Profile API endpoints (X-001, X-002).

Pipeline definitions were retired 2026-10-18 (hardening plan H-10.1
follow-up): an in-memory module-global store that nothing at ingest or
search time read. They answered 410, then the routes were deleted the same
day with their clients; ``tests/test_h10b_followups.py`` pins that. The
rag-profiles CRUD (retired 2026-08-22, 410 from then) was deleted the same
day once its last client (``r4 rag retriever`` CRUD) stopped calling — the
H-10.1 rag-profiles wave. The component registry stays.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.pipelines import router

_DEV_HEADERS = {"X-User-Id": "user1", "X-Project-Id": "project1"}


@pytest.fixture
def app():
    """Create a test FastAPI app with the pipeline router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


class TestPipelineAPIDeleted:
    """The /umrs/pipelines CRUD is deleted (X-001; 410 from umrs d4a8ad85,
    removed with its clients on 2026-10-18, H-10.1 follow-up)."""

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("POST", "/umrs/pipelines", {"name": "test-pipeline"}),
            ("GET", "/umrs/pipelines", None),
            ("GET", "/umrs/pipelines/pipe-1", None),
            ("PATCH", "/umrs/pipelines/pipe-1", {"name": "renamed"}),
            ("DELETE", "/umrs/pipelines/pipe-1", None),
        ],
    )
    def test_pipeline_verbs_are_not_mounted(self, client, method, path, body):
        response = client.request(method, path, json=body, headers=_DEV_HEADERS)
        assert response.status_code == 404


class TestRAGProfileAPIDeleted:
    """The /umrs/rag-profiles CRUD is deleted (X-002). It answered 410 from
    the owner decision 2026-08-22 until the last client (``r4 rag
    retriever`` CRUD) was retired; the routes, the lane family and the
    ``/gw/rag-profiles`` edge locations went together on 2026-10-18
    (H-10.1 rag-profiles wave). On the old sources each of these answered
    410 instead of 404."""

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("POST", "/umrs/rag-profiles", {"name": "test-profile"}),
            ("GET", "/umrs/rag-profiles", None),
            ("GET", "/umrs/rag-profiles/rag-1", None),
            ("PATCH", "/umrs/rag-profiles/rag-1", {"temperature": 0.8}),
            ("DELETE", "/umrs/rag-profiles/rag-1", None),
        ],
    )
    def test_rag_profile_verbs_are_not_mounted(self, client, method, path, body):
        response = client.request(method, path, json=body, headers=_DEV_HEADERS)
        assert response.status_code == 404


class TestComponentRegistryAPI:
    """Tests for component registry endpoints (X-003, X-004, X-005)."""

    def test_list_chunkers(self, client):
        """List available chunkers."""
        response = client.get("/umrs/components/chunkers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should have default chunkers
        assert len(data) > 0

    def test_list_embedders(self, client):
        """List available embedders."""
        response = client.get("/umrs/components/embedders")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_list_embedders_with_filter(self, client):
        """List embedders with language filter."""
        response = client.get("/umrs/components/embedders?language=en")
        assert response.status_code == 200

    def test_list_rerankers(self, client):
        """List available rerankers."""
        response = client.get("/umrs/components/rerankers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
