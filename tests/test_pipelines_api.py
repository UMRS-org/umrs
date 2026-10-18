"""
Pipeline and RAG Profile API endpoints (X-001, X-002): both retired (410).

Pipeline definitions were retired 2026-10-18 (hardening plan H-10.1
follow-up): an in-memory module-global store that nothing at ingest or
search time read. The CRUD and validation tests of the old store are gone
with it; the retirement and its identity rule are pinned in
``tests/test_h10b_followups.py``. The component registry stays.
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


class TestPipelineAPIRetired:
    """Every /umrs/pipelines verb answers 410 with a pointer (X-001 retired)."""

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
    def test_pipeline_verbs_are_gone(self, client, method, path, body):
        response = client.request(method, path, json=body, headers=_DEV_HEADERS)
        assert response.status_code == 410
        assert "retired" in response.json()["detail"]

    def test_invalid_bodies_are_gone_too(self, client):
        """No body validation runs before the 410 (was 422 / 400)."""
        response = client.post(
            "/umrs/pipelines",
            json={"name": "", "chunker": {"chunker_type": "invalid_type"}},
            headers=_DEV_HEADERS,
        )
        assert response.status_code == 410


class TestRAGProfileAPI:
    """RAG profile routes are retired (R3, drift remediation 2026-08-22):
    every /umrs/rag-profiles verb now returns 410 with descriptive detail.
    """

    def test_create_rag_profile_retired(self, client):
        response = client.post(
            "/umrs/rag-profiles",
            json={"name": "test-profile"},
            headers=_DEV_HEADERS,
        )
        assert response.status_code == 410
        assert "retired" in response.json()["detail"].lower()

    def test_create_rag_profile_full_also_retired(self, client):
        response = client.post(
            "/umrs/rag-profiles",
            json={"name": "full-profile", "search_type": "hybrid"},
            headers=_DEV_HEADERS,
        )
        assert response.status_code == 410

    def test_list_rag_profiles_retired(self, client):
        response = client.get(
            "/umrs/rag-profiles",
            headers=_DEV_HEADERS,
        )
        assert response.status_code == 410

    def test_get_rag_profile_retired(self, client):
        response = client.get(
            "/umrs/rag-profiles/rag-1",
            headers=_DEV_HEADERS,
        )
        assert response.status_code == 410

    def test_update_rag_profile_retired(self, client):
        response = client.patch(
            "/umrs/rag-profiles/rag-1",
            json={"temperature": 0.8},
            headers=_DEV_HEADERS,
        )
        assert response.status_code == 410

    def test_delete_rag_profile_retired(self, client):
        response = client.delete(
            "/umrs/rag-profiles/rag-1",
            headers=_DEV_HEADERS,
        )
        assert response.status_code == 410


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
