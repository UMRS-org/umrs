"""Three-tier scope enforcement on the UMRS RAG proxy (ADR-027 P0).

Predicates come from umrs.knowledge.access_scope; these tests pin the
route-level wiring: admin bypass, system-corpus read-shared/write-admin,
and user:<uid> owner-only namespaces.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.rag import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry


def _create_client(project_id: str = "test-project") -> TestClient:
    registry = EndpointRegistry()
    registry.register(
        project_id=project_id,
        model="m",
        job_id="job-1",
        host="compute-01",
        port=11434,
        backend="slurm",
    )
    registry.mark_ready("job-1")
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def _mock_rag_response(payload: dict, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=payload,
        headers={"content-type": "application/json"},
    )


def _mock_post(payload: dict):
    async def fake_post(self, url, **kwargs):
        return _mock_rag_response(payload)

    return fake_post


def test_query_project_mismatch_403_message():
    client = _create_client()
    response = client.post(
        "/umrs/rag/query",
        json={"query": "q", "project_id": "other-project"},
        headers={"x-user-id": "u1", "x-project-id": "test-project"},
    )
    assert response.status_code == 403
    assert "read scope" in response.json()["detail"]
    assert "other-project" in response.json()["detail"]


def test_query_system_corpus_allowed_for_member():
    """project_readable('system') is True for any verified caller."""
    client = _create_client()
    payload = {"answer": "ok", "sources": []}
    with patch.object(httpx.AsyncClient, "post", _mock_post(payload)):
        response = client.post(
            "/umrs/rag/query",
            json={"query": "q", "project_id": "system"},
            headers={"x-user-id": "u1", "x-project-id": "test-project"},
        )
    assert response.status_code == 200


def test_query_admin_reads_any_project():
    client = _create_client()
    payload = {"answer": "ok", "sources": []}
    with patch.object(httpx.AsyncClient, "post", _mock_post(payload)):
        response = client.post(
            "/umrs/rag/query",
            json={"query": "q", "project_id": "someone-else"},
            headers={
                "x-user-id": "admin",
                "x-project-id": "test-project",
                "x-verified-is-admin": "true",
            },
        )
    assert response.status_code == 200


def test_upload_to_system_denied_for_member():
    """project_writable('system') is False for non-admin callers."""
    client = _create_client()
    response = client.post(
        "/umrs/rag/upload",
        files={"file": ("a.txt", b"x", "text/plain")},
        headers={"x-user-id": "u1", "x-project-id": "system"},
    )
    assert response.status_code == 403
    assert "write scope" in response.json()["detail"]


def test_upload_user_namespace_owner_only():
    """Writing into another user's personal namespace is denied."""
    client = _create_client()
    response = client.post(
        "/umrs/rag/upload",
        files={"file": ("a.txt", b"x", "text/plain")},
        headers={"x-user-id": "u2", "x-project-id": "user:u1"},
    )
    assert response.status_code == 403


def test_delete_foreign_project_denied():
    """Verified project is user:u1 but caller is u2 → delete denied."""
    client = _create_client()
    response = client.delete(
        "/umrs/rag/documents/doc-1",
        headers={"x-user-id": "u2", "x-project-id": "user:u1"},
    )
    assert response.status_code == 403


def _mock_get(payload, seen: list):
    async def fake_get(self, url, **kwargs):
        seen.append(kwargs.get("params"))
        return _mock_rag_response(payload)

    return fake_get


# TUI audit 2026-10-17: project_id was not a parameter of GET /documents, so
# `r4 rag docs <project>` listed the caller's own corpus under that name.


def test_documents_lists_the_named_corpus():
    client = _create_client()
    seen: list = []
    docs = {"documents": [{"document_id": "d1", "filename": "a.pdf"}]}
    with patch.object(httpx.AsyncClient, "get", _mock_get(docs, seen)):
        response = client.get(
            "/umrs/rag/documents",
            params={"project_id": "system"},
            headers={"x-user-id": "u1", "x-project-id": "test-project"},
        )
    assert response.status_code == 200
    assert seen == [{"project_id": "system"}]
    assert response.json()[0]["project_id"] == "system"


def test_documents_defaults_to_the_caller_corpus():
    client = _create_client()
    seen: list = []
    with patch.object(httpx.AsyncClient, "get", _mock_get({"documents": []}, seen)):
        response = client.get(
            "/umrs/rag/documents",
            headers={"x-user-id": "u1", "x-project-id": "test-project"},
        )
    assert response.status_code == 200
    assert seen == [{"project_id": "test-project"}]


def test_documents_of_a_foreign_project_denied():
    client = _create_client()
    response = client.get(
        "/umrs/rag/documents",
        params={"project_id": "other-project"},
        headers={"x-user-id": "u1", "x-project-id": "test-project"},
    )
    assert response.status_code == 403
    assert "other-project" in response.json()["detail"]
