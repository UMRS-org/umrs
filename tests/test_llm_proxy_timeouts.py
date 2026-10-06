"""W5/U1 (2026-10-05): the LLM proxy's upstream limits are split.

connect (unchanged), idle-between-chunks for streaming (600 s) and a total
deadline for non-streaming requests (7200 s). Long reasoning turns — measured
up to 93 min for one non-streaming glm-5.3 request — must not be cut by the
old single 120 s read timeout. The old env name stays a deprecated alias of
the idle value.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import llm as llm_api
from umrs.api._proxy_timeouts import (
    DEFAULT_CONNECT_TIMEOUT_SEC,
    DEFAULT_IDLE_TIMEOUT_SEC,
    DEFAULT_TOTAL_TIMEOUT_SEC,
    ProxyTimeouts,
    resolve_proxy_timeouts,
)
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

_HEADERS = {"x-user-id": "user-1", "x-project-id": "test-project"}
_CHAT = {"model": "qwen3-8b", "messages": [{"role": "user", "content": "Hi"}]}


def _registry() -> EndpointRegistry:
    registry = EndpointRegistry()
    registry.register(
        project_id="test-project",
        model="qwen3-8b",
        job_id="job-1",
        host="compute-01",
        port=11434,
        backend="slurm",
        metadata={"served_model_name": "Qwen/Qwen3-8B"},
    )
    registry.mark_ready("job-1")
    return registry


def _client(registry: EndpointRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(llm_api.router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def _json_response(body: dict) -> httpx.Response:
    return httpx.Response(status_code=200, json=body, headers={"content-type": "application/json"})


# ---------------------------------------------------------------------------
# Resolution of the three limits from the environment
# ---------------------------------------------------------------------------


def test_defaults_are_connect_10_idle_600_total_7200():
    assert resolve_proxy_timeouts("LLM_PROXY", env={}) == ProxyTimeouts(
        connect=10.0, idle=600.0, total=7200.0
    )
    assert DEFAULT_CONNECT_TIMEOUT_SEC == 10.0
    assert DEFAULT_IDLE_TIMEOUT_SEC == 600.0
    assert DEFAULT_TOTAL_TIMEOUT_SEC == 7200.0


def test_new_env_names_set_each_limit():
    env = {
        "LLM_PROXY_CONNECT_TIMEOUT": "5",
        "LLM_PROXY_IDLE_TIMEOUT": "900",
        "LLM_PROXY_TOTAL_TIMEOUT": "3600",
    }
    assert resolve_proxy_timeouts("LLM_PROXY", env=env) == ProxyTimeouts(
        connect=5.0, idle=900.0, total=3600.0
    )


def test_deprecated_read_timeout_is_an_alias_of_idle_only(caplog):
    """The pre-split name keeps working — for the idle limit, never the total."""
    with caplog.at_level(logging.WARNING, logger="umrs.api._proxy_timeouts"):
        timeouts = resolve_proxy_timeouts("LLM_PROXY", env={"LLM_PROXY_READ_TIMEOUT": "120"})
    assert timeouts.idle == 120.0
    assert timeouts.total == DEFAULT_TOTAL_TIMEOUT_SEC
    assert "LLM_PROXY_READ_TIMEOUT is deprecated" in caplog.text
    assert "LLM_PROXY_IDLE_TIMEOUT" in caplog.text


def test_new_idle_name_wins_over_deprecated_alias():
    env = {"LLM_PROXY_IDLE_TIMEOUT": "900", "LLM_PROXY_READ_TIMEOUT": "120"}
    assert resolve_proxy_timeouts("LLM_PROXY", env=env).idle == 900.0


def test_chat_prefix_has_its_own_names_and_alias():
    env = {
        "CHAT_PROXY_READ_TIMEOUT": "240",
        "LLM_PROXY_IDLE_TIMEOUT": "900",
        "LLM_PROXY_TOTAL_TIMEOUT": "60",
    }
    chat = resolve_proxy_timeouts("CHAT_PROXY", env=env)
    assert chat.idle == 240.0
    assert chat.total == DEFAULT_TOTAL_TIMEOUT_SEC


@pytest.mark.parametrize("raw", ["abc", "0", "-5"])
def test_invalid_values_fail_loudly_with_the_variable_name(raw):
    with pytest.raises(ValueError, match="LLM_PROXY_TOTAL_TIMEOUT"):
        resolve_proxy_timeouts("LLM_PROXY", env={"LLM_PROXY_TOTAL_TIMEOUT": raw})


def test_module_limits_come_from_the_resolver():
    assert resolve_proxy_timeouts("LLM_PROXY") == llm_api._TIMEOUTS


# ---------------------------------------------------------------------------
# Non-streaming: total deadline, reads wait for the total limit
# ---------------------------------------------------------------------------


def test_oneshot_reads_wait_for_the_total_limit_not_the_idle_one(monkeypatch):
    """A non-streaming reply arrives only when the model is done: its read
    timeout is the total limit; connect is unchanged."""
    monkeypatch.setattr(llm_api, "_TIMEOUTS", ProxyTimeouts(connect=7.0, idle=11.0, total=13.0))
    seen: list[httpx.Timeout] = []

    async def fake_request(self, method, url, **kwargs):
        seen.append(self.timeout)
        return _json_response({"id": "chatcmpl-1", "choices": []})

    with patch.object(httpx.AsyncClient, "request", fake_request):
        response = _client(_registry()).post(
            "/umrs/llm/v1/chat/completions", json=_CHAT, headers=_HEADERS
        )

    assert response.status_code == 200
    assert seen[0].connect == 7.0
    assert seen[0].read == 13.0
    assert seen[0].write == 11.0


def test_oneshot_longer_than_idle_but_inside_total_succeeds(monkeypatch):
    monkeypatch.setattr(llm_api, "_TIMEOUTS", ProxyTimeouts(connect=10.0, idle=0.01, total=5.0))

    async def slow_request(self, method, url, **kwargs):
        await asyncio.sleep(0.1)
        return _json_response({"id": "chatcmpl-slow", "choices": []})

    with patch.object(httpx.AsyncClient, "request", slow_request):
        response = _client(_registry()).post(
            "/umrs/llm/v1/chat/completions", json=_CHAT, headers=_HEADERS
        )

    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-slow"


def test_oneshot_total_deadline_maps_to_504_and_releases_the_endpoint(monkeypatch):
    """Exceeding the total deadline is the same typed 504 as an httpx timeout."""
    monkeypatch.setattr(llm_api, "_TIMEOUTS", ProxyTimeouts(connect=10.0, idle=600.0, total=0.05))
    registry = _registry()

    async def hung_request(self, method, url, **kwargs):
        await asyncio.sleep(5)
        return _json_response({"id": "never"})

    with patch.object(httpx.AsyncClient, "request", hung_request):
        response = _client(registry).post(
            "/umrs/llm/v1/chat/completions", json=_CHAT, headers=_HEADERS
        )

    assert response.status_code == 504
    assert response.headers["x-platform-error-code"] == "llm.gateway_timeout"
    assert response.json()["detail"] == "Request to upstream service timed out"
    assert registry.get_by_job_id("job-1").active_requests == 0


def test_oneshot_httpx_timeout_mapping_is_unchanged():
    async def timing_out(self, method, url, **kwargs):
        raise httpx.ReadTimeout("slow", request=httpx.Request(method, url))

    with patch.object(httpx.AsyncClient, "request", timing_out):
        response = _client(_registry()).post(
            "/umrs/llm/v1/chat/completions", json=_CHAT, headers=_HEADERS
        )

    assert response.status_code == 504
    assert response.headers["x-platform-error-code"] == "llm.gateway_timeout"
    assert response.json()["detail"] == "Request to upstream service timed out"


def test_oneshot_connect_error_mapping_is_unchanged():
    async def refused(self, method, url, **kwargs):
        raise httpx.ConnectError("refused", request=httpx.Request(method, url))

    with patch.object(httpx.AsyncClient, "request", refused):
        response = _client(_registry()).post(
            "/umrs/llm/v1/chat/completions", json=_CHAT, headers=_HEADERS
        )

    assert response.status_code == 502
    assert response.headers["x-platform-error-code"] == "llm.bad_gateway"
    assert response.json()["detail"] == "Failed to connect to upstream service"


# ---------------------------------------------------------------------------
# Streaming: idle limit per read, no total limit while chunks flow
# ---------------------------------------------------------------------------


def test_streaming_reads_use_the_idle_limit(monkeypatch):
    monkeypatch.setattr(llm_api, "_TIMEOUTS", ProxyTimeouts(connect=7.0, idle=11.0, total=13.0))
    seen: list[httpx.Timeout] = []

    async def fake_send(self, request, **kwargs):
        seen.append(self.timeout)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: [DONE]\n\n",
            request=request,
        )

    with patch.object(httpx.AsyncClient, "send", fake_send):
        response = _client(_registry()).post(
            "/umrs/llm/v1/chat/completions",
            json={**_CHAT, "stream": True},
            headers=_HEADERS,
        )

    assert response.status_code == 200
    assert seen[0].connect == 7.0
    assert seen[0].read == 11.0


def test_streaming_has_no_total_limit_while_chunks_flow(monkeypatch):
    """A stream that outlasts the total deadline completes when chunks keep
    arriving inside the idle limit."""
    monkeypatch.setattr(llm_api, "_TIMEOUTS", ProxyTimeouts(connect=10.0, idle=600.0, total=0.05))
    chunks = [b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'] * 5 + [b"data: [DONE]\n\n"]

    class _SlowStream:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_bytes(self):
            for chunk in chunks:
                await asyncio.sleep(0.03)
                yield chunk

        async def aclose(self):
            return None

    class _Client:
        async def aclose(self):
            return None

    async def fake_open_stream(request, upstream_url, headers, body):
        return _Client(), _SlowStream()

    with patch.object(llm_api, "_open_stream", fake_open_stream):
        response = _client(_registry()).post(
            "/umrs/llm/v1/chat/completions",
            json={**_CHAT, "stream": True},
            headers=_HEADERS,
        )

    assert response.status_code == 200
    assert response.content == b"".join(chunks)


# ---------------------------------------------------------------------------
# Model listing keeps its own short read limit
# ---------------------------------------------------------------------------


def test_model_listing_fan_out_keeps_the_short_read_limit(monkeypatch):
    """One hung endpoint must not stall GET /v1/models for the idle/total
    limits; the listing keeps the pre-split 120 s read limit."""
    monkeypatch.setattr(llm_api, "_TIMEOUTS", ProxyTimeouts(connect=7.0, idle=600.0, total=7200.0))
    monkeypatch.setattr(llm_api, "_LIST_READ_TIMEOUT", 3.0)
    seen: list[httpx.Timeout] = []

    async def fake_get(self, url, **kwargs):
        seen.append(self.timeout)
        return httpx.Response(
            200,
            json={"object": "list", "data": [{"id": "Qwen/Qwen3-8B"}]},
            request=httpx.Request("GET", str(url)),
        )

    with patch.object(httpx.AsyncClient, "get", fake_get):
        response = _client(_registry()).get("/umrs/llm/v1/models", headers=_HEADERS)

    assert response.status_code == 200
    assert seen[0].connect == 7.0
    assert seen[0].read == 3.0
