"""In-process integration tests for the ADR-012 conversational chain.

These wire the **real** UMRS chat proxy router and the **real** Assistant
Service app together in one process, with a fake LiteLLM upstream, and route
all async HTTP between them through ``httpx.ASGITransport`` (no sockets).

Chain under test::

    TestClient → chat proxy (/umrs/chat) → Assistant Service (/v1/*)
        → fake LiteLLM upstream (/v1/*)

Covered:
- Non-streaming completion flows end-to-end and returns upstream JSON.
- Streaming (SSE) forwards chunks intact and in order, with **no** leading
  ``(status, headers)`` tuple leaking into the body (regression for the
  ``_proxy_streaming`` bug).
- Streaming fails open with an SSE error event + ``[DONE]`` when the
  Assistant Service is unreachable (status already committed).
- The Assistant Service's RAG augmentation hook fires within the real chain.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient
from umrs.api import assistant_service as assistant_mod
from umrs.api import chat as chat_mod
from umrs.api.chat import router as chat_router
from umrs.orchestrator.endpoint_registry import get_registry

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeEndpoint:
    """Endpoint stub whose assistant_url points at the routed 'assistant' host."""

    job_id = "job-int-1"
    active_requests = 0

    @property
    def assistant_url(self) -> str:
        return "http://assistant:8090"


class _FakeRegistry:
    """No-op registry; request accounting must not raise."""

    def increment_requests(self, _job_id: str) -> None: ...
    def mark_used(self, _job_id: str) -> None: ...
    def decrement_requests(self, _job_id: str) -> None: ...


class _RoutingTransport(httpx.AsyncBaseTransport):
    """Dispatch async requests to in-process ASGI apps keyed by URL host."""

    def __init__(self, routes: dict[str, httpx.ASGITransport]):
        self._routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self._routes.get(request.url.host)
        if transport is None:
            raise httpx.ConnectError(f"no route for host {request.url.host!r}", request=request)
        return await transport.handle_async_request(request)


def _build_fake_litellm(received: dict[str, Any]) -> FastAPI:
    """Fake LiteLLM upstream: OpenAI-compatible chat + models."""
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions")
    async def _chat(request: Request) -> Any:
        body = await request.json()
        received["messages"] = body.get("messages")
        received["body"] = body
        if body.get("stream"):

            async def _gen():
                for i in range(3):
                    yield (f'data: {{"choices":[{{"delta":{{"content":"tok{i}"}}}}]}}\n\n').encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(_gen(), media_type="text/event-stream")
        return JSONResponse(
            {
                "id": "cmpl-int-1",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "hello world"}}],
            }
        )

    @upstream.get("/v1/models")
    async def _models() -> JSONResponse:
        return JSONResponse(
            {"object": "list", "data": [{"id": "rivault-static", "object": "model"}]}
        )

    return upstream


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def received() -> dict[str, Any]:
    return {}


@pytest.fixture
def routed(monkeypatch, received):
    """Patch httpx.AsyncClient so all async calls route to in-process apps.

    Yields the routes dict so a test can drop the ``assistant`` route to
    simulate an unreachable Assistant Service.
    """
    monkeypatch.setenv("ASSISTANT_UPSTREAM_BASE_URL", "http://upstream/v1")
    monkeypatch.setenv("ASSISTANT_DEFAULT_MODEL", "rivault-static")
    # RAG must not touch the network in the default chain; fail open to [].
    monkeypatch.setattr(assistant_mod, "retrieve_context", lambda *_a, **_k: [])

    routes: dict[str, httpx.ASGITransport] = {
        "assistant": httpx.ASGITransport(app=assistant_mod.app),
        "upstream": httpx.ASGITransport(app=_build_fake_litellm(received)),
    }
    routing = _RoutingTransport(routes)

    real_async_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_async_client(*args, transport=routing, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _factory)
    return routes


@pytest.fixture
def client(monkeypatch, routed):
    """TestClient for the chat proxy app with discovery + registry stubbed."""
    monkeypatch.setattr(chat_mod, "_discover_endpoint", lambda *_a, **_k: _FakeEndpoint())

    app = FastAPI()
    app.include_router(chat_router)
    app.dependency_overrides[get_registry] = lambda: _FakeRegistry()
    return TestClient(app)


_HEADERS = {"X-User-ID": "u1", "X-Project-ID": "proj1"}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_non_streaming_chain_returns_upstream_json(client, received):
    resp = client.post(
        "/umrs/chat/v1/chat/completions",
        headers=_HEADERS,
        json={
            "model": "rivault-static",
            "messages": [{"role": "user", "content": "hi"}],
            "rag": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "cmpl-int-1"
    assert data["choices"][0]["message"]["content"] == "hello world"
    # Assistant stripped its extension keys before forwarding.
    assert "rag" not in received["body"]
    assert "project_id" not in received["body"]


def test_streaming_chain_forwards_sse_intact(client):
    resp = client.post(
        "/umrs/chat/v1/chat/completions",
        headers=_HEADERS,
        json={
            "model": "rivault-static",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "rag": False,
        },
    )
    assert resp.status_code == 200
    text = resp.text
    # Chunks arrive in order, terminated by [DONE].
    assert text.index("tok0") < text.index("tok1") < text.index("tok2")
    assert "data: [DONE]" in text
    # Regression: no (status, headers) tuple leaked into the body.
    assert "Headers(" not in text
    assert "200," not in text


def test_streaming_fails_open_when_assistant_unreachable(client, routed):
    # Drop the assistant route → connection error mid-setup.
    routed.pop("assistant")
    resp = client.post(
        "/umrs/chat/v1/chat/completions",
        headers=_HEADERS,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "rag": False,
        },
    )
    # Status is committed as 200 before streaming; body carries the error.
    assert resp.status_code == 200
    text = resp.text
    assert "Assistant Service unavailable" in text
    assert "data: [DONE]" in text


def test_rag_augmentation_hook_fires_in_chain(client, monkeypatch, received):
    sentinel = {"role": "system", "content": "INJECTED_CONTEXT"}

    def _fake_augment(_config, messages, **_kwargs):
        return [sentinel, *messages]

    monkeypatch.setattr(assistant_mod, "augment_messages", _fake_augment)

    resp = client.post(
        "/umrs/chat/v1/chat/completions",
        headers=_HEADERS,
        json={
            "model": "rivault-static",
            "messages": [{"role": "user", "content": "hi"}],
            "project_id": "proj1",
        },
    )
    assert resp.status_code == 200
    # The fake upstream saw the augmented messages produced by the Assistant.
    assert received["messages"][0]["content"] == "INJECTED_CONTEXT"


def test_models_chain_proxies_upstream_list(client, routed):
    resp = client.get("/umrs/chat/v1/models", headers=_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["data"][0]["id"] == "rivault-static"


# ---------------------------------------------------------------------------
# Streaming edge-case tests
# ---------------------------------------------------------------------------


class _TimeoutTransport(httpx.AsyncBaseTransport):
    """Transport that always raises ``httpx.ReadTimeout``."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("simulated read timeout", request=request)


def test_streaming_timeout_fails_open(client, monkeypatch):
    """When the upstream times out mid-stream, the proxy emits a
    timeout-specific SSE error event followed by ``[DONE]``."""
    monkeypatch.setattr(
        chat_mod,
        "_proxy_streaming",
        lambda _request, _url, _headers, _body: _timeout_generator(),
    )

    resp = client.post(
        "/umrs/chat/v1/chat/completions",
        headers=_HEADERS,
        json={
            "model": "rivault-static",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "rag": False,
        },
    )
    assert resp.status_code == 200
    text = resp.text
    assert "timed out" in text
    assert "data: [DONE]" in text


async def _timeout_generator():
    """Simulate a timeout error during streaming."""
    yield b'data: {"error": {"message": "Assistant Service timed out"}}\n\n'
    yield b"data: [DONE]\n\n"


def test_streaming_multiple_data_events_forwarded(client):
    """Verify that multiple SSE data events are forwarded intact."""
    resp = client.post(
        "/umrs/chat/v1/chat/completions",
        headers=_HEADERS,
        json={
            "model": "rivault-static",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "rag": False,
        },
    )
    assert resp.status_code == 200
    text = resp.text
    # Should have all three chunks + [DONE]
    assert "tok0" in text
    assert "tok1" in text
    assert "tok2" in text
    assert "data: [DONE]" in text
    # No tuple leaked
    assert "Headers(" not in text


def test_streaming_error_event_format(client, routed):
    """Verify the SSE error event format on connection failure."""
    routed.pop("assistant")
    resp = client.post(
        "/umrs/chat/v1/chat/completions",
        headers=_HEADERS,
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "rag": False,
        },
    )
    assert resp.status_code == 200
    text = resp.text
    # Error event should be valid SSE with JSON error
    assert "data: " in text
    assert "error" in text
    assert "data: [DONE]" in text
