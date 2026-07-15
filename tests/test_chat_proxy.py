"""Unit tests for the UMRS Chat Proxy Router (ADR-012).

Network is never touched: the upstream forward is exercised through
monkeypatched httpx clients.
"""

# NOTE: ruff may complain about unsorted imports in this block.
# The import order is intentional for readability.
import httpx  # noqa: E402
import pytest
from umrs.api import chat as chat_mod
from umrs.api.chat import (
    _assistant_url,
    _build_upstream_headers,
    _discover_endpoint,
    _parse_body,
)
from umrs.orchestrator.endpoint_registry import (
    EndpointStatus,
    RegisteredEndpoint,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEndpoint:
    """Minimal endpoint stub for testing."""

    def __init__(self, host="10.0.0.1", port=4000, job_id="job-1"):
        self.host = host
        self.port = port
        self.job_id = job_id
        self.active_requests = 0
        self.metadata = {}


@pytest.fixture()
def endpoint():
    return FakeEndpoint()


@pytest.fixture()
def registered_endpoint():
    return RegisteredEndpoint(
        project_id="proj-1",
        model="rivault-gpt-oss-static",
        job_id="job-1",
        host="10.0.0.1",
        port=4000,
        status=EndpointStatus.READY,
    )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_body_valid_json():
    assert _parse_body(b'{"model": "test"}') == {"model": "test"}


def test_parse_body_empty():
    assert _parse_body(b"") == {}


def test_parse_body_invalid_json():
    assert _parse_body(b"not json") == {}


def test_assistant_url_default_port(registered_endpoint):
    url = _assistant_url(registered_endpoint)
    assert url == "http://10.0.0.1:8090"


def test_assistant_url_custom_port(registered_endpoint):
    registered_endpoint.metadata["assistant_port"] = 9090
    url = _assistant_url(registered_endpoint)
    assert url == "http://10.0.0.1:9090"


def test_assistant_url_is_first_class_property(registered_endpoint):
    """Verify assistant_url is a first-class property on RegisteredEndpoint."""
    assert registered_endpoint.assistant_url == "http://10.0.0.1:8090"
    registered_endpoint.metadata["assistant_port"] = 7080
    assert registered_endpoint.assistant_url == "http://10.0.0.1:7080"


def test_build_upstream_headers_includes_identity():
    class FakeRequest:
        headers = {
            "content-type": "application/json",
            "x-user-id": "user-1",
            "x-project-id": "proj-1",
            "x-request-id": "req-1",
            "authorization": "Bearer token",
        }

    headers = _build_upstream_headers(FakeRequest(), "user-1", "proj-1")
    assert headers["X-User-ID"] == "user-1"
    assert headers["X-Project-ID"] == "proj-1"
    assert headers["X-Request-ID"] == "req-1"
    assert headers["Authorization"] == "Bearer token"


def test_build_upstream_headers_no_auth():
    class FakeRequest:
        headers = {
            "content-type": "application/json",
            "x-user-id": "user-1",
            "x-project-id": "proj-1",
            "x-request-id": "req-1",
        }

    headers = _build_upstream_headers(FakeRequest(), "user-1", "proj-1")
    assert "Authorization" not in headers


# ---------------------------------------------------------------------------
# Endpoint discovery
# ---------------------------------------------------------------------------


class FakeRegistry:
    """Minimal registry stub for testing."""

    def __init__(self, endpoints=None):
        self._endpoints = endpoints or []
        self.incremented = False
        self.decremented = False
        self.marked_used = False

    def select_endpoint(self, project_id, model, **_kwargs):
        for ep in self._endpoints:
            if ep.project_id == project_id and ep.model == model:
                return ep
        return None

    def list_all(self, project_id=None, status_filter=None):
        eps = [ep for ep in self._endpoints if ep.project_id == project_id]
        if status_filter:
            eps = [ep for ep in eps if ep.status in status_filter]
        return eps

    def increment_requests(self, _job_id):
        self.incremented = True

    def decrement_requests(self, _job_id):
        self.decremented = True

    def mark_used(self, _job_id):
        self.marked_used = True


def test_discover_endpoint_by_model(registered_endpoint):
    registry = FakeRegistry(endpoints=[registered_endpoint])
    ep = _discover_endpoint(registry, "proj-1", model="rivault-gpt-oss-static")
    assert ep.job_id == "job-1"


def test_discover_endpoint_fallback_any_ready(registered_endpoint):
    registry = FakeRegistry(endpoints=[registered_endpoint])
    ep = _discover_endpoint(registry, "proj-1", model="nonexistent")
    assert ep.job_id == "job-1"


def test_discover_endpoint_no_endpoints_raises_503():
    registry = FakeRegistry(endpoints=[])
    with pytest.raises(Exception) as exc_info:
        _discover_endpoint(registry, "proj-1")
    assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# Chat completions endpoint
# ---------------------------------------------------------------------------


class FakeAsyncClient:
    """Mock httpx.AsyncClient for testing."""

    def __init__(self, *_args, **kwargs):
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, **kwargs):
        self.post_url = url
        self.post_kwargs = kwargs
        return FakeResponse(
            status_code=200,
            content=b'{"id": "chatcmpl-1", "choices": []}',
            content_type="application/json",
        )


class FakeResponse:
    def __init__(self, status_code=200, content=b"", content_type="application/json"):
        self.status_code = status_code
        self.content = content
        self.headers = {"content-type": content_type}


def test_chat_completions_proxies_to_assistant_service(monkeypatch, registered_endpoint):
    """Test that chat completions are proxied to the Assistant Service."""
    registry = FakeRegistry(endpoints=[registered_endpoint])

    # Mock the request
    class FakeRequest:
        headers = {
            "content-type": "application/json",
            "x-user-id": "user-1",
            "x-project-id": "proj-1",
            "x-request-id": "req-1",
            "authorization": "Bearer token",
        }

        async def body(self):
            return b'{"model": "test", "messages": [{"role": "user", "content": "hi"}]}'

    # Mock httpx.AsyncClient
    fake_client = FakeAsyncClient()
    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", lambda *_a, **_kw: fake_client)

    # Call the endpoint
    import asyncio

    from umrs.api.chat import chat_completions

    request = FakeRequest()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response = loop.run_until_complete(chat_completions(request, registry))
    finally:
        loop.close()

    assert response.status_code == 200
    assert registry.incremented is True
    assert registry.decremented is True
    assert registry.marked_used is True


def test_chat_completions_streaming_returns_sse(monkeypatch, registered_endpoint):
    """Test that streaming requests return SSE responses."""
    registry = FakeRegistry(endpoints=[registered_endpoint])

    class FakeRequest:
        headers = {
            "content-type": "application/json",
            "x-user-id": "user-1",
            "x-project-id": "proj-1",
            "x-request-id": "req-1",
            "authorization": "Bearer token",
        }

        async def body(self):
            return b'{"model": "test", "stream": true, "messages": [{"role": "user", "content": "hi"}]}'

    from umrs.api.chat import chat_completions

    request = FakeRequest()
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response = loop.run_until_complete(chat_completions(request, registry))
    finally:
        loop.close()

    # StreamingResponse has media_type "text/event-stream"
    assert response.media_type == "text/event-stream"


def test_chat_completions_no_project_id_returns_401(registered_endpoint):
    """Test that missing project_id returns 401."""
    registry = FakeRegistry(endpoints=[registered_endpoint])

    class FakeRequest:
        headers = {
            "content-type": "application/json",
            "x-user-id": "user-1",
            # Missing x-project-id
        }

        async def body(self):
            return b"{}"

    from umrs.api.chat import chat_completions

    request = FakeRequest()
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with pytest.raises(Exception) as exc_info:
            response = loop.run_until_complete(chat_completions(request, registry))
    finally:
        loop.close()
    assert exc_info.value.status_code == 401


def test_chat_completions_upstream_unavailable_returns_502(monkeypatch, registered_endpoint):
    """If the Assistant Service is unreachable, the proxy returns 502.

    This documents the fail-open boundary: the *Assistant Service* itself
    fails open to plain LiteLLM passthrough when RAG/MCP are unavailable.
    But if the Assistant Service process is down, the proxy cannot
    transparently fall back — it returns a clear 502.
    """
    registry = FakeRegistry(endpoints=[registered_endpoint])

    class FailingClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("Assistant Service down")

    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", FailingClient)

    class FakeRequest:
        headers = {
            "content-type": "application/json",
            "x-user-id": "user-1",
            "x-project-id": "proj-1",
            "x-request-id": "req-1",
            "authorization": "Bearer token",
        }

        async def body(self):
            return b'{"model": "test", "messages": [{"role": "user", "content": "hi"}]}'

    from umrs.api.chat import chat_completions

    request = FakeRequest()
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        with pytest.raises(Exception) as exc_info:
            response = loop.run_until_complete(chat_completions(request, registry))
    finally:
        loop.close()
    assert exc_info.value.status_code == 502
