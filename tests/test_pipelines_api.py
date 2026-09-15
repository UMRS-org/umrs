"""
Unit tests for Pipeline and RAG Profile API endpoints (X-001, X-002).
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from umrs.api.pipelines import router, PipelineStore, set_pipeline_store


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


@pytest.fixture(autouse=True)
def reset_store():
    """Reset the pipeline store before each test."""
    set_pipeline_store(PipelineStore())
    yield
    set_pipeline_store(PipelineStore())


class TestPipelineAPI:
    """Tests for Pipeline CRUD API (X-001)."""

    def test_create_pipeline_minimal(self, client):
        """Create pipeline with minimal config."""
        response = client.post(
            "/umrs/pipelines",
            json={"name": "test-pipeline"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test-pipeline"
        assert data["project_id"] == "project1"
        assert data["owner_id"] == "user1"
        assert data["pipeline_id"].startswith("pipe-")

    def test_create_pipeline_full(self, client):
        """Create pipeline with full configuration."""
        response = client.post(
            "/umrs/pipelines",
            json={
                "name": "full-pipeline",
                "description": "Test description",
                "chunker": {
                    "chunker_type": "sentence",
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                },
                "embedder": {
                    "model_name": "BAAI/bge-base-en-v1.5",
                    "dimension": 768,
                },
                "supported_extensions": [".pdf", ".txt"],
                "is_default": True,
            },
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "full-pipeline"
        assert data["description"] == "Test description"
        assert data["chunker"]["chunker_type"] == "sentence"
        assert data["embedder"]["model_name"] == "BAAI/bge-base-en-v1.5"
        assert data["is_default"] is True

    def test_list_pipelines(self, client):
        """List pipelines for a project."""
        # Create pipelines
        client.post(
            "/umrs/pipelines",
            json={"name": "pipe-1"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        client.post(
            "/umrs/pipelines",
            json={"name": "pipe-2"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        client.post(
            "/umrs/pipelines",
            json={"name": "pipe-other"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project2"},
        )

        # List for project1
        response = client.get(
            "/umrs/pipelines",
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_pipeline(self, client):
        """Get single pipeline by ID."""
        # Create pipeline
        create_response = client.post(
            "/umrs/pipelines",
            json={"name": "get-test"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        pipeline_id = create_response.json()["pipeline_id"]

        # Get pipeline
        response = client.get(
            f"/umrs/pipelines/{pipeline_id}",
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "get-test"

    def test_get_pipeline_not_found(self, client):
        """Get non-existent pipeline."""
        response = client.get(
            "/umrs/pipelines/pipe-notexist",
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 404

    def test_get_pipeline_wrong_project(self, client):
        """Cannot get pipeline from different project."""
        # Create in project1
        create_response = client.post(
            "/umrs/pipelines",
            json={"name": "private-pipe"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        pipeline_id = create_response.json()["pipeline_id"]

        # Try to get from project2
        response = client.get(
            f"/umrs/pipelines/{pipeline_id}",
            headers={"X-User-Id": "user1", "X-Project-Id": "project2"},
        )
        assert response.status_code == 403

    def test_update_pipeline(self, client):
        """Update pipeline fields."""
        # Create pipeline
        create_response = client.post(
            "/umrs/pipelines",
            json={"name": "update-test"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        pipeline_id = create_response.json()["pipeline_id"]

        # Update
        response = client.patch(
            f"/umrs/pipelines/{pipeline_id}",
            json={"name": "updated-name", "description": "New description"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "updated-name"
        assert response.json()["description"] == "New description"

    def test_delete_pipeline(self, client):
        """Delete pipeline."""
        # Create pipeline
        create_response = client.post(
            "/umrs/pipelines",
            json={"name": "delete-test"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        pipeline_id = create_response.json()["pipeline_id"]

        # Delete
        response = client.delete(
            f"/umrs/pipelines/{pipeline_id}",
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 204

        # Verify deleted
        response = client.get(
            f"/umrs/pipelines/{pipeline_id}",
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 404

    def test_duplicate_name_rejected(self, client):
        """Cannot create duplicate pipeline name in same project."""
        client.post(
            "/umrs/pipelines",
            json={"name": "dup-name"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        response = client.post(
            "/umrs/pipelines",
            json={"name": "dup-name"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 409


class TestRAGProfileAPI:
    """Tests for RAG Profile CRUD API (X-002)."""

    def test_create_rag_profile_minimal(self, client):
        """Create RAG profile with minimal config."""
        response = client.post(
            "/umrs/rag-profiles",
            json={"name": "test-profile"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test-profile"
        assert data["profile_id"].startswith("rag-")

    def test_create_rag_profile_full(self, client):
        """Create RAG profile with full configuration."""
        response = client.post(
            "/umrs/rag-profiles",
            json={
                "name": "full-profile",
                "description": "Test RAG profile",
                "search_type": "hybrid",
                "vector_weight": 0.8,
                "keyword_weight": 0.2,
                "top_k_retrieval": 30,
                "reranker": {
                    "model_name": "BAAI/bge-reranker-base",
                    "top_k": 10,
                },
                "max_context_tokens": 8192,
                "temperature": 0.7,
            },
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["search_type"] == "hybrid"
        assert data["vector_weight"] == 0.8
        assert data["reranker"]["model_name"] == "BAAI/bge-reranker-base"

    def test_list_rag_profiles(self, client):
        """List RAG profiles for a project."""
        client.post(
            "/umrs/rag-profiles",
            json={"name": "profile-1"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        client.post(
            "/umrs/rag-profiles",
            json={"name": "profile-2"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )

        response = client.get(
            "/umrs/rag-profiles",
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 200
        assert len(response.json()) == 2

    def test_get_rag_profile(self, client):
        """Get single RAG profile."""
        create_response = client.post(
            "/umrs/rag-profiles",
            json={"name": "get-test"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        profile_id = create_response.json()["profile_id"]

        response = client.get(
            f"/umrs/rag-profiles/{profile_id}",
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "get-test"

    def test_update_rag_profile(self, client):
        """Update RAG profile."""
        create_response = client.post(
            "/umrs/rag-profiles",
            json={"name": "update-test", "temperature": 0.5},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        profile_id = create_response.json()["profile_id"]

        response = client.patch(
            f"/umrs/rag-profiles/{profile_id}",
            json={"temperature": 0.8, "vector_weight": 0.9},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 200
        assert response.json()["temperature"] == 0.8
        assert response.json()["vector_weight"] == 0.9

    def test_delete_rag_profile(self, client):
        """Delete RAG profile."""
        create_response = client.post(
            "/umrs/rag-profiles",
            json={"name": "delete-test"},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        profile_id = create_response.json()["profile_id"]

        response = client.delete(
            f"/umrs/rag-profiles/{profile_id}",
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 204


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


class TestPipelineValidation:
    """Test input validation for pipelines."""

    def test_empty_name_rejected(self, client):
        """Empty name is rejected."""
        response = client.post(
            "/umrs/pipelines",
            json={"name": ""},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 422

    def test_invalid_chunker_type(self, client):
        """Invalid chunker type is rejected."""
        response = client.post(
            "/umrs/pipelines",
            json={
                "name": "test",
                "chunker": {"chunker_type": "invalid_type"},
            },
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 400

    def test_chunk_size_constraints(self, client):
        """Chunk size must be within constraints."""
        response = client.post(
            "/umrs/pipelines",
            json={
                "name": "test",
                "chunker": {"chunk_size": 5},  # Below minimum
            },
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 422


class TestRAGProfileValidation:
    """Test input validation for RAG profiles."""

    def test_invalid_search_weights(self, client):
        """Search weights must be 0-1."""
        response = client.post(
            "/umrs/rag-profiles",
            json={"name": "test", "vector_weight": 1.5},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 422

    def test_temperature_constraints(self, client):
        """Temperature must be 0-2."""
        response = client.post(
            "/umrs/rag-profiles",
            json={"name": "test", "temperature": 3.0},
            headers={"X-User-Id": "user1", "X-Project-Id": "project1"},
        )
        assert response.status_code == 422
