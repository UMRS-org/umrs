"""W5/U1 (2026-10-05): the assistant chat proxy's upstream limits are split.

Same shape as the LLM proxy (``test_llm_proxy_timeouts.py``): streaming reads
use the idle limit (``CHAT_PROXY_IDLE_TIMEOUT``), a non-streaming request
waits up to the total deadline (``CHAT_PROXY_TOTAL_TIMEOUT``). The chat
proxy's error mapping is unchanged: every upstream failure, timeouts
included, is ``chatapi.bad_gateway`` (502) — the chatapi taxonomy has no 504.
"""

from __future__ import annotations

import asyncio
import contextlib

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import chat as chat_mod
from umrs.api._proxy_timeouts import ProxyTimeouts, resolve_proxy_timeouts
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

_HEADERS = {"x-user-id": "user-1", "x-project-id": "proj-1"}
_CHAT = {"model": "assistant-model", "messages": [{"role": "user", "content": "hi"}]}


def _registry() -> EndpointRegistry:
    registry = EndpointRegistry()
    registry.register(
        project_id="proj-1",
        model="assistant-model",
        job_id="job-1",
        host="10.0.0.1",
        port=4000,
        backend="slurm",
    )
    registry.mark_ready("job-1")
    return registry


def _client(registry: EndpointRegistry) -> TestClient:
    app = FastAPI()
    app.include_router(chat_mod.router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


class _Resp:
    status_code = 200
    content = b'{"id": "chatcmpl-1", "choices": []}'
    headers = {"content-type": "application/json"}

    async def aiter_raw(self):
        yield b"data: [DONE]\n\n"


def _fake_client_class(post_delay: float = 0.0):
    """httpx.AsyncClient stand-in that records the timeout it was built with."""

    class _FakeClient:
        timeouts: list[httpx.Timeout] = []

        def __init__(self, *_args, **kwargs):
            _FakeClient.timeouts.append(kwargs.get("timeout"))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, *_args, **_kwargs):
            await asyncio.sleep(post_delay)
            return _Resp()

        @contextlib.asynccontextmanager
        async def stream(self, *_args, **_kwargs):
            yield _Resp()

    return _FakeClient


def test_module_limits_come_from_the_resolver():
    assert resolve_proxy_timeouts("CHAT_PROXY") == chat_mod._TIMEOUTS
    assert chat_mod._TIMEOUTS.connect == chat_mod._CONNECT_TIMEOUT


def test_oneshot_reads_wait_for_the_total_limit(monkeypatch):
    monkeypatch.setattr(chat_mod, "_TIMEOUTS", ProxyTimeouts(connect=7.0, idle=11.0, total=13.0))
    fake = _fake_client_class()
    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", fake)

    response = _client(_registry()).post(
        "/umrs/chat/v1/chat/completions", json=_CHAT, headers=_HEADERS
    )

    assert response.status_code == 200
    assert fake.timeouts[0].connect == 7.0
    assert fake.timeouts[0].read == 13.0
    assert fake.timeouts[0].write == 7.0  # write stays the connect limit


def test_oneshot_total_deadline_is_a_typed_502_and_releases_the_endpoint(
    monkeypatch,
):
    monkeypatch.setattr(chat_mod, "_TIMEOUTS", ProxyTimeouts(connect=10.0, idle=600.0, total=0.05))
    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", _fake_client_class(5.0))
    registry = _registry()

    response = _client(registry).post(
        "/umrs/chat/v1/chat/completions", json=_CHAT, headers=_HEADERS
    )

    assert response.status_code == 502
    assert response.headers["x-platform-error-code"] == "chatapi.bad_gateway"
    assert response.json()["detail"] == "Assistant Service timed out"
    assert registry.get_by_job_id("job-1").active_requests == 0


def test_oneshot_longer_than_idle_but_inside_total_succeeds(monkeypatch):
    monkeypatch.setattr(chat_mod, "_TIMEOUTS", ProxyTimeouts(connect=10.0, idle=0.01, total=5.0))
    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", _fake_client_class(0.1))

    response = _client(_registry()).post(
        "/umrs/chat/v1/chat/completions", json=_CHAT, headers=_HEADERS
    )

    assert response.status_code == 200


def test_oneshot_connect_error_mapping_is_unchanged(monkeypatch):
    class _Refusing(_fake_client_class()):
        async def post(self, *_args, **_kwargs):
            raise httpx.ConnectError("down")

    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", _Refusing)

    response = _client(_registry()).post(
        "/umrs/chat/v1/chat/completions", json=_CHAT, headers=_HEADERS
    )

    assert response.status_code == 502
    assert response.headers["x-platform-error-code"] == "chatapi.bad_gateway"
    assert response.json()["detail"] == "Failed to connect to Assistant Service"


def test_streaming_reads_use_the_idle_limit(monkeypatch):
    monkeypatch.setattr(chat_mod, "_TIMEOUTS", ProxyTimeouts(connect=7.0, idle=11.0, total=13.0))
    fake = _fake_client_class()
    monkeypatch.setattr(chat_mod.httpx, "AsyncClient", fake)

    response = _client(_registry()).post(
        "/umrs/chat/v1/chat/completions",
        json={**_CHAT, "stream": True},
        headers=_HEADERS,
    )

    assert response.status_code == 200
    assert response.content == b"data: [DONE]\n\n"
    assert fake.timeouts[0].connect == 7.0
    assert fake.timeouts[0].read == 11.0
