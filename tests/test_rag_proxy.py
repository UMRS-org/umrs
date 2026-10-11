"""Tests for the UMRS RAG proxy router."""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.rag import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry


def _make_registry_with_endpoint(
    *,
    project_id: str = "test-project",
    model: str = "qwen3-8b",
    job_id: str = "job-1",
    host: str = "compute-01",
    port: int = 11434,
) -> EndpointRegistry:
    registry = EndpointRegistry()
    registry.register(
        project_id=project_id,
        model=model,
        job_id=job_id,
        host=host,
        port=port,
        backend="slurm",
    )
    registry.mark_ready(job_id)
    return registry


def _create_client(registry: EndpointRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def _mock_rag_response(content: dict, status_code: int = 200):
    import httpx

    return httpx.Response(
        status_code=status_code,
        json=content,
        headers={"content-type": "application/json"},
    )


def test_query_rag_discovers_and_proxies(monkeypatch):
    """Query should discover endpoint and proxy to RAG service."""
    # The spine answers by default (2026-10-09); this is the endpoint facade.
    monkeypatch.setenv("UMRS_RAG_QUERY_FACADE", "endpoint")
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    rag_response = {
        "answer": "The answer is 42.",
        "sources": [{"document_id": "doc1", "content": "snippet", "score": 0.95, "metadata": {}}],
    }

    async def fake_post(self, url, **kwargs):
        assert "compute-01:8003" in str(url)
        assert "rag/query" in str(url)
        return _mock_rag_response(rag_response)

    import httpx

    with patch.object(httpx.AsyncClient, "post", fake_post):
        response = client.post(
            "/umrs/rag/query",
            json={
                "query": "what is the answer?",
                "project_id": "test-project",
            },
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "The answer is 42."


def test_query_rag_no_endpoint_returns_503(monkeypatch):
    """If no endpoint is available, return 503."""
    monkeypatch.setenv("UMRS_RAG_QUERY_FACADE", "endpoint")
    registry = EndpointRegistry()
    client = _create_client(registry)

    response = client.post(
        "/umrs/rag/query",
        json={"query": "test", "project_id": "test-project"},
        headers={"x-user-id": "user-1", "x-project-id": "test-project"},
    )

    assert response.status_code == 503


def test_query_rag_project_mismatch_returns_403():
    """If project_id in body doesn't match header, return 403."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    response = client.post(
        "/umrs/rag/query",
        json={"query": "test", "project_id": "other-project"},
        headers={"x-user-id": "user-1", "x-project-id": "test-project"},
    )

    assert response.status_code == 403


def test_query_rag_missing_project_returns_401():
    """If X-Project-ID header is missing, return 401."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    response = client.post(
        "/umrs/rag/query",
        json={"query": "test", "project_id": "test-project"},
        headers={"x-user-id": "user-1"},
    )

    assert response.status_code == 401


def test_query_rag_requires_verified_project_in_strict_mode(monkeypatch):
    """Strict mode rejects advisory X-Project-ID without Gateway verification."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    response = client.post(
        "/umrs/rag/query",
        json={"query": "test", "project_id": "test-project"},
        headers={"x-user-id": "user-1", "x-project-id": "test-project"},
    )

    assert response.status_code == 401


def test_query_rag_prefers_verified_project(monkeypatch):
    """Verified Gateway project identity wins over advisory X-Project-ID."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    registry = _make_registry_with_endpoint(project_id="verified-project")
    client = _create_client(registry)

    async def fake_post(self, url, **kwargs):
        assert "compute-01:8003" in str(url)
        return _mock_rag_response({"answer": "ok", "sources": []})

    import httpx

    with patch.object(httpx.AsyncClient, "post", fake_post):
        response = client.post(
            "/umrs/rag/query",
            json={"query": "test", "project_id": "verified-project"},
            headers={
                "x-user-id": "user-1",
                "x-project-id": "attacker-project",
                "x-verified-project-id": "verified-project",
            },
        )

    assert response.status_code == 200


def test_list_documents_endpoint():
    """List documents endpoint should work."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    async def fake_get(self, url, **kwargs):
        return _mock_rag_response({"documents": []})

    import httpx

    with patch.object(httpx.AsyncClient, "get", fake_get):
        response = client.get(
            "/umrs/rag/documents",
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
    assert response.json() == []


def test_delete_document_never_proxies_and_records_nothing_unresolved(monkeypatch):
    """ADR-093 MD-16 (decided 2026-10-11): the delete runs in UMRS over the
    knowledge tables, never through the rag service's old placeholder (which
    answered "deleted" and deleted nothing). Without a knowledge database the
    id cannot resolve: the answer says so, nothing is called or recorded.
    (The resolved delete: test_knowledge_artifact_delete.py.)"""
    for var in ("UMRS_DATA_CATALOG_DB", "DATABASE_URL", "DB_HOST", "DB_PORT", "UMRS_JOBS_DB"):
        monkeypatch.delenv(var, raising=False)
    for var in ("POSTGRES_PORT", "POSTGRES_DB", "DB_SUPERUSER", "DB_SUPERUSER_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    for var in ("POSTGRES_USER", "POSTGRES_PASSWORD"):
        monkeypatch.delenv(var, raising=False)
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)
    calls: list[str] = []

    async def fake_delete(self, url, **kwargs):
        calls.append(str(url))
        return _mock_rag_response({"status": "deleted"})

    import httpx

    from umrs.api import governance

    monkeypatch.setattr(
        governance, "record_resource_deleted", lambda *_a, **_k: calls.append("recorded")
    )
    with patch.object(httpx.AsyncClient, "delete", fake_delete):
        response = client.delete(
            "/umrs/rag/documents/doc-123",
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 503
    assert calls == []


def test_get_document_endpoint():
    """Get document metadata should work."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    async def fake_get(self, url, **kwargs):
        return _mock_rag_response(
            {
                "document_id": "doc-123",
                "filename": "paper.pdf",
                "project_id": "test-project",
                "chunks": 5,
                "metadata": {},
            }
        )

    import httpx

    with patch.object(httpx.AsyncClient, "get", fake_get):
        response = client.get(
            "/umrs/rag/documents/doc-123",
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
    assert response.json()["filename"] == "paper.pdf"


def test_rag_falls_back_to_system_endpoints():
    """If no project endpoint, should try system tier endpoints."""
    registry = _make_registry_with_endpoint(project_id="system")
    client = _create_client(registry)

    async def fake_get(self, url, **kwargs):
        return _mock_rag_response({"documents": []})

    import httpx

    with patch.object(httpx.AsyncClient, "get", fake_get):
        response = client.get(
            "/umrs/rag/documents",
            headers={"x-user-id": "user-1", "x-project-id": "other-project"},
        )

    assert response.status_code == 200
