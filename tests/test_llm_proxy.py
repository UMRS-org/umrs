"""Tests for the UMRS LLM proxy router."""

from __future__ import annotations

import json
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import llm as llm_api
from umrs.api.llm import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry


def _make_registry_with_endpoint(
    *,
    project_id: str = "test-project",
    model: str = "qwen3-8b",
    job_id: str = "job-1",
    host: str = "compute-01",
    port: int = 11434,
    served_model_name: str | None = "Qwen/Qwen3-8B",
) -> EndpointRegistry:
    registry = EndpointRegistry()
    metadata = {}
    if served_model_name:
        metadata["served_model_name"] = served_model_name
    registry.register(
        project_id=project_id,
        model=model,
        job_id=job_id,
        host=host,
        port=port,
        backend="slurm",
        metadata=metadata,
    )
    registry.mark_ready(job_id)
    return registry


def _create_client(registry: EndpointRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def _mock_litellm_response(content: dict, status_code: int = 200):
    """Create a mock httpx.Response for LiteLLM."""
    import httpx

    return httpx.Response(
        status_code=status_code,
        json=content,
        headers={"content-type": "application/json"},
    )


def test_chat_completions_discovers_and_proxies():
    """Chat completions should discover endpoint and proxy to LiteLLM."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    llm_response = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "choices": [{"message": {"content": "Hello!"}}],
    }

    async def fake_request(self, method, url, **kwargs):
        assert "compute-01:11434" in str(url)
        assert "v1/chat/completions" in str(url)
        return _mock_litellm_response(llm_response)

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        response = client.post(
            "/umrs/llm/v1/chat/completions",
            json={"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-abc"


def test_chat_completions_marks_used():
    """After proxying, the endpoint's last_used_at should be updated."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    llm_response = {"id": "chatcmpl-abc", "object": "chat.completion", "choices": []}

    async def fake_request(self, method, url, **kwargs):
        return _mock_litellm_response(llm_response)

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        client.post(
            "/umrs/llm/v1/chat/completions",
            json={"model": "qwen3-8b", "messages": []},
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    ep = registry.get_by_job_id("job-1")
    assert ep is not None
    assert ep.last_used_at is not None


def test_chat_completions_no_endpoint_returns_503():
    """If no endpoint is available, return 503."""
    registry = EndpointRegistry()  # empty
    client = _create_client(registry)

    response = client.post(
        "/umrs/llm/v1/chat/completions",
        json={"model": "qwen3-8b", "messages": []},
        headers={"x-user-id": "user-1", "x-project-id": "test-project"},
    )

    assert response.status_code == 503


def test_chat_completions_missing_model_returns_400():
    """If model field is missing, return 400."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    response = client.post(
        "/umrs/llm/v1/chat/completions",
        json={"messages": []},
        headers={"x-user-id": "user-1", "x-project-id": "test-project"},
    )

    assert response.status_code == 400


def test_chat_completions_missing_project_returns_401():
    """If X-Project-ID header is missing, return 401."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    response = client.post(
        "/umrs/llm/v1/chat/completions",
        json={"model": "qwen3-8b", "messages": []},
        headers={"x-user-id": "user-1"},
    )

    assert response.status_code == 401


def test_chat_completions_requires_verified_project_in_strict_mode(monkeypatch):
    """Strict mode rejects legacy X-Project-ID without Gateway verification."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    response = client.post(
        "/umrs/llm/v1/chat/completions",
        json={"model": "qwen3-8b", "messages": []},
        headers={"x-user-id": "user-1", "x-project-id": "test-project"},
    )

    assert response.status_code == 401


def test_chat_completions_prefers_verified_project(monkeypatch):
    """Verified Gateway project identity wins over advisory X-Project-ID."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    registry = _make_registry_with_endpoint(project_id="verified-project")
    client = _create_client(registry)

    async def fake_request(self, method, url, **kwargs):
        assert "compute-01:11434" in str(url)
        return _mock_litellm_response({"id": "x", "choices": []})

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        response = client.post(
            "/umrs/llm/v1/chat/completions",
            json={"model": "qwen3-8b", "messages": []},
            headers={
                "x-user-id": "user-1",
                "x-project-id": "attacker-project",
                "x-verified-project-id": "verified-project",
            },
        )

    assert response.status_code == 200


def test_model_alias_rewriting_catalog_to_served():
    """When client sends catalog name, it should be rewritten to served name."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    captured_body = {}

    async def fake_request(self, method, url, **kwargs):
        body = kwargs.get("content", b"")
        if body:
            captured_body.update(json.loads(body))
        return _mock_litellm_response({"id": "x", "choices": []})

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        client.post(
            "/umrs/llm/v1/chat/completions",
            json={"model": "qwen3-8b", "messages": []},
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert captured_body.get("model") == "Qwen/Qwen3-8B"


def test_model_alias_rewriting_keeps_explicit_served_model():
    """When endpoint metadata names the upstream model, keep that served name."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    captured_body = {}

    async def fake_request(self, method, url, **kwargs):
        body = kwargs.get("content", b"")
        if body:
            captured_body.update(json.loads(body))
        return _mock_litellm_response({"id": "x", "choices": []})

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        client.post(
            "/umrs/llm/v1/chat/completions",
            json={"model": "Qwen/Qwen3-8B", "messages": []},
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert captured_body.get("model") == "Qwen/Qwen3-8B"


def test_model_alias_rewriting_heuristic_served_to_catalog_without_metadata():
    """Heuristic aliases are recognized, but upstream defaults to registry model."""
    registry = _make_registry_with_endpoint(served_model_name=None)
    client = _create_client(registry)

    captured_body = {}

    async def fake_request(self, method, url, **kwargs):
        body = kwargs.get("content", b"")
        if body:
            captured_body.update(json.loads(body))
        return _mock_litellm_response({"id": "x", "choices": []})

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        client.post(
            "/umrs/llm/v1/chat/completions",
            json={"model": "Qwen/Qwen3-8B", "messages": []},
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert captured_body.get("model") == "qwen3-8b"


def test_model_alias_rewriting_keeps_catalog_without_metadata():
    """Live RCCS endpoints without served metadata should keep catalog names upstream."""
    registry = _make_registry_with_endpoint(served_model_name=None)
    client = _create_client(registry)

    captured_body = {}

    async def fake_request(self, method, url, **kwargs):
        body = kwargs.get("content", b"")
        if body:
            captured_body.update(json.loads(body))
        return _mock_litellm_response({"id": "x", "choices": []})

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        client.post(
            "/umrs/llm/v1/chat/completions",
            json={"model": "qwen3-8b", "messages": []},
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert captured_body.get("model") == "qwen3-8b"


def test_list_models_empty():
    """With no endpoints, /v1/models should return empty list."""
    registry = EndpointRegistry()
    client = _create_client(registry)

    response = client.get(
        "/umrs/llm/v1/models",
        headers={"x-user-id": "user-1", "x-project-id": "test-project"},
    )

    assert response.status_code == 200
    assert response.json() == {"object": "list", "data": []}


def _mock_models_get(url: str, content: dict, status_code: int = 200):
    """Create a mock httpx.Response for a GET /v1/models call.

    A bound request is required so ``raise_for_status`` does not raise a
    RuntimeError on the mock.
    """
    import httpx

    return httpx.Response(
        status_code=status_code,
        json=content,
        headers={"content-type": "application/json"},
        request=httpx.Request("GET", url),
    )


def test_list_models_proxies_to_litellm():
    """With an active endpoint, /v1/models should list that endpoint's models."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    models_response = {
        "object": "list",
        "data": [{"id": "Qwen/Qwen3-8B", "object": "model"}],
    }

    async def fake_get(self, url, **kwargs):
        assert "v1/models" in str(url)
        return _mock_models_get(str(url), models_response)

    import httpx

    with patch.object(httpx.AsyncClient, "get", fake_get):
        response = client.get(
            "/umrs/llm/v1/models",
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    assert [m["id"] for m in data] == ["Qwen/Qwen3-8B"]


def test_list_models_aggregates_across_all_ready_endpoints():
    """/v1/models should merge models from every ready endpoint, deduped by id."""
    registry = _make_registry_with_endpoint(
        model="qwen3-8b", job_id="job-1", host="compute-01",
        served_model_name="Qwen/Qwen3-8B",
    )
    registry.register(
        project_id="test-project", model="llama3-70b", job_id="job-2",
        host="compute-02", port=11434, backend="slurm",
        metadata={"served_model_name": "meta/Llama-3-70B"},
    )
    registry.mark_ready("job-2")
    client = _create_client(registry)

    by_host = {
        "compute-01": {"object": "list", "data": [{"id": "Qwen/Qwen3-8B", "object": "model"}]},
        "compute-02": {"object": "list", "data": [{"id": "meta/Llama-3-70B", "object": "model"}]},
    }

    async def fake_get(self, url, **kwargs):
        host = next(h for h in by_host if h in str(url))
        return _mock_models_get(str(url), by_host[host])

    import httpx

    with patch.object(httpx.AsyncClient, "get", fake_get):
        response = client.get(
            "/umrs/llm/v1/models",
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
    ids = sorted(m["id"] for m in response.json()["data"])
    assert ids == ["Qwen/Qwen3-8B", "meta/Llama-3-70B"]


def test_list_models_synthesizes_entry_when_endpoint_listing_fails():
    """A model still appears when its endpoint's /v1/models fetch fails."""
    registry = _make_registry_with_endpoint(served_model_name="Qwen/Qwen3-8B")
    client = _create_client(registry)

    import httpx

    async def fake_get(self, url, **kwargs):
        raise httpx.ConnectError("boom", request=httpx.Request("GET", str(url)))

    with patch.object(httpx.AsyncClient, "get", fake_get):
        response = client.get(
            "/umrs/llm/v1/models",
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
    assert [m["id"] for m in response.json()["data"]] == ["Qwen/Qwen3-8B"]


def test_completions_endpoint():
    """Legacy completions endpoint should work."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    async def fake_request(self, method, url, **kwargs):
        assert "v1/completions" in str(url)
        return _mock_litellm_response({"id": "cmpl-x", "choices": []})

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        response = client.post(
            "/umrs/llm/v1/completions",
            json={"model": "qwen3-8b", "prompt": "Hello"},
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200


def test_embeddings_endpoint():
    """Embeddings endpoint should work."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    async def fake_request(self, method, url, **kwargs):
        assert "v1/embeddings" in str(url)
        return _mock_litellm_response({"object": "list", "data": [{"embedding": [0.1]}]})

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        response = client.post(
            "/umrs/llm/v1/embeddings",
            json={"model": "qwen3-8b", "input": "Hello"},
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200


def test_service_key_injected():
    """GATEWAY_LITELLM_SERVICE_KEY should be injected as Authorization header."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    captured_headers = {}

    async def fake_request(self, method, url, **kwargs):
        # Capture the headers from the build_request or request call
        headers = kwargs.get("headers", {})
        captured_headers.update(headers)
        return _mock_litellm_response({"id": "x", "choices": []})

    import httpx

    with patch("umrs.api.llm._LITELLM_SERVICE_KEY", "sk-test-key"):
        with patch.object(httpx.AsyncClient, "request", fake_request):
            client.post(
                "/umrs/llm/v1/chat/completions",
                json={"model": "qwen3-8b", "messages": []},
                headers={"x-user-id": "user-1", "x-project-id": "test-project"},
            )

    assert captured_headers.get("Authorization") == "Bearer sk-test-key"


def test_service_key_resolution_falls_back_to_litellm_master_key(monkeypatch):
    """RCCS server env provides LITELLM_MASTER_KEY rather than Gateway-specific key."""
    monkeypatch.delenv("GATEWAY_LITELLM_SERVICE_KEY", raising=False)
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master-key")

    assert llm_api._resolve_litellm_service_key() == "sk-master-key"


def test_discovery_by_served_name():
    """Endpoint should be discoverable by served_model_name."""
    registry = _make_registry_with_endpoint()
    client = _create_client(registry)

    async def fake_request(self, method, url, **kwargs):
        return _mock_litellm_response({"id": "x", "choices": []})

    import httpx

    with patch.object(httpx.AsyncClient, "request", fake_request):
        response = client.post(
            "/umrs/llm/v1/chat/completions",
            json={"model": "Qwen/Qwen3-8B", "messages": []},
            headers={"x-user-id": "user-1", "x-project-id": "test-project"},
        )

    assert response.status_code == 200
