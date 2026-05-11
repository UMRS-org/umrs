import httpx
import pytest

from umrs.orchestrator.readiness import ReadinessConfig, ReadinessPoller


def _make_poller(monkeypatch, handler):
    poller = ReadinessPoller(ReadinessConfig(auth_token="test-token"))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def _fake_get_client():
        return client

    monkeypatch.setattr(poller, "_get_client", _fake_get_client)
    return poller, client


@pytest.mark.asyncio
async def test_check_ready_detail_with_chat_probe_waits_for_inference(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3-8b"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(500, json={"error": {"message": "still loading"}})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    poller, client = _make_poller(monkeypatch, handler)
    try:
        is_ready, models, error = await poller.check_ready_detail(
            "ready.test",
            8001,
            probe_model="qwen3-8b",
            probe_kind="chat",
        )
    finally:
        await client.aclose()

    assert not is_ready
    assert models == ["qwen3-8b"]
    assert "Chat probe HTTP 500" in error


@pytest.mark.asyncio
async def test_check_ready_detail_with_chat_probe_succeeds_after_models(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3-8b"}]})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-ready",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    poller, client = _make_poller(monkeypatch, handler)
    try:
        is_ready, models, error = await poller.check_ready_detail(
            "ready.test",
            8001,
            probe_model="qwen3-8b",
            probe_kind="chat",
        )
    finally:
        await client.aclose()

    assert is_ready
    assert models == ["qwen3-8b"]
    assert error == ""


@pytest.mark.asyncio
async def test_check_ready_detail_with_http_probe_uses_custom_path(monkeypatch):
    seen_paths = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/readyz":
            return httpx.Response(200, json={"status": "ok"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    poller, client = _make_poller(monkeypatch, handler)
    try:
        is_ready, models, error = await poller.check_ready_detail(
            "ready.test",
            3001,
            probe_kind="http",
            probe_path="readyz",
        )
    finally:
        await client.aclose()

    assert is_ready
    assert models == []
    assert error == ""
    assert seen_paths == ["/readyz"]