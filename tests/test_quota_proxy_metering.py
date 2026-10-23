"""ADR-099 D-4 (RS-1 fix Q2): a remote session cannot dodge the token cap.

- a streamed OpenAI-style completion is asked for its usage chunk
  (``stream_options.include_usage = true``), on a user's source and on the
  site source; before, a stream sent without it recorded zero tokens;
- a client that disconnects mid-stream still leaves its usage row; before,
  the row was written by the background task Starlette skips then.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time

import httpx
import pytest
from test_infer_proxy import _FakeBroker, _FakeStore, _lease, _stream_response
from test_infer_proxy_site import _registry
from test_quota_proxy import CHAT, USAGE_BODY, _app, _Pool
from umrs.infer_proxy.app import create_app
from umrs.shared.site_source import SITE_SOURCE_ID

AUTH = {"authorization": "Bearer tok-session"}


def _bodies(monkeypatch, calls_app):
    """Wrap the MockTransport handler's client to keep each upstream body."""
    app, pool, broker, _calls = calls_app
    bodies: list[bytes] = []
    client = app.state.http_client
    original = client.send

    async def send(request, **kw):
        bodies.append(request.content if request.content else await request.aread())
        return await original(request, **kw)

    monkeypatch.setattr(client, "send", send)
    return app, pool, bodies


def _post(app, body: dict, url: str = CHAT):
    from test_infer_proxy import _request

    return _request(app, "POST", url, json=body, headers=AUTH)


class TestIncludeUsage:
    def test_a_stream_is_asked_for_its_usage(self, monkeypatch):
        app, _pool, bodies = _bodies(monkeypatch, _app(monkeypatch))
        assert _post(app, {"model": "gpt-x", "stream": True, "messages": []}).status_code == 200
        sent = json.loads(bodies[0])
        assert sent["stream_options"] == {"include_usage": True}
        assert sent["model"] == "gpt-x" and sent["stream"] is True

    def test_other_stream_options_are_kept(self, monkeypatch):
        app, _pool, bodies = _bodies(monkeypatch, _app(monkeypatch))
        body = {"model": "m", "stream": True, "stream_options": {"x": 1, "include_usage": False}}
        _post(app, body)
        assert json.loads(bodies[0])["stream_options"] == {"x": 1, "include_usage": True}

    def test_a_one_shot_body_is_relayed_byte_for_byte(self, monkeypatch):
        app, _pool, bodies = _bodies(monkeypatch, _app(monkeypatch))
        raw = b'{"model":"m",  "messages":[]}'
        from test_infer_proxy import _request

        _request(
            app, "POST", CHAT, content=raw, headers={**AUTH, "content-type": "application/json"}
        )
        assert bodies[0] == raw

    def test_the_site_source_stream_is_asked_too(self, monkeypatch):
        lease = _lease(
            source_ids=("src-a", SITE_SOURCE_ID), project="proj-a", owner_id="alice", venue="rccs"
        )
        monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _p, _t: lease)
        monkeypatch.setenv("GATEWAY_LITELLM_SERVICE_KEY", "site-service-key")
        seen: list[dict] = []

        def handler(request):
            seen.append(json.loads(request.read()))
            return _stream_response(body=USAGE_BODY, headers={"content-type": "application/json"})

        app = create_app(
            umrs_db_pool=_Pool(),
            unseal_privkey="unused",
            sealed_store=_FakeStore(None),
            broker=_FakeBroker(),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            site_registry=_registry(),
        )
        res = _post(
            app, {"model": "qwen3-8b", "stream": True}, url="/s/sess-1/site/v1/chat/completions"
        )
        assert res.status_code == 200
        assert seen[0]["stream_options"] == {"include_usage": True}

    @pytest.mark.parametrize(
        ("payload", "changed"),
        [
            ({"stream": True}, True),
            ({"stream": False}, False),
            ({}, False),
            ({"stream": True, "stream_options": {"include_usage": True}}, False),
            ([], False),
        ],
    )
    def test_with_stream_usage(self, payload, changed):
        from umrs.infer_proxy.app import with_stream_usage

        assert with_stream_usage(payload) is changed


class TestDisconnect:
    def test_a_client_gone_mid_stream_still_leaves_its_usage_row(self, monkeypatch):
        sse = (
            b'data: {"model": "gpt-x", "choices": [{"delta": {"content": "hi"}}]}\n\n'
            b'data: {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 9}}\n\n'
        )
        app, pool, _broker, _calls = _app(monkeypatch, body=sse, content_type="text/event-stream")
        asyncio.run(_disconnecting_call(app))
        deadline = time.monotonic() + 5
        while not pool.inserts() and time.monotonic() < deadline:
            time.sleep(0.01)
        (row,) = pool.inserts()
        assert row[9] == 29 and row[-1] == "infer_proxy"


async def _disconnecting_call(app) -> None:
    """One ASGI 2.4 call whose client is gone at the first body chunk."""
    body = json.dumps({"model": "gpt-x", "stream": True}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": CHAT,
        "raw_path": CHAT.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"authorization", b"Bearer tok-session"), (b"host", b"testserver")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    sent = iter([{"type": "http.request", "body": body, "more_body": False}])

    async def receive():
        try:
            return next(sent)
        except StopIteration:
            await asyncio.sleep(3600)

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("client went away")

    with contextlib.suppress(Exception):  # the disconnect surfaces as an error
        await app(scope, receive, send)
    await asyncio.sleep(0.2)  # let the executor thread write the row
