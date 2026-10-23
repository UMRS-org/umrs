"""ADR-099 D-4, Amendment 2 (RS-1 round-2 review, finding 1): a capped
user's own code cannot avoid the daily token cap.

- a ``stream`` that is not a JSON boolean (``1``, ``"true"``), which vLLM
  reads as true, is sent upstream as ``true`` with ``include_usage``;
  before, it went unchanged and the stream carried no usage;
- a client that leaves after the content but before the final usage chunk
  leaves an estimated row, not a zero one; before, the row said 0 tokens;
- a stream that ends without usage records an estimate; a whole answer with
  usage, a one-shot body without usage and a GET keep the old rows.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
import test_quota_proxy as tqp
from test_quota_proxy import _app
from test_quota_proxy_metering import _bodies, _disconnecting_call, _post
from umrs.infer_proxy.app import with_stream_usage
from umrs.quotas.usage_tap import UsageTap

CONTENT = b'data: {"model": "gpt-x", "choices": [{"delta": {"content": "%s"}}]}\n\n'
USAGE = b'data: {"choices": [], "usage": {"prompt_tokens": 20000, "completion_tokens": 9000}}\n\n'

PROMPT, COMPLETION, TOTAL, EXTRA = 7, 8, 9, 19


def _two_chunks(*chunks: bytes):
    """An upstream that sends ``chunks`` one by one (the last after a pause)."""

    def response(status=200, body=b"", headers=None):
        async def _yield():
            for i, chunk in enumerate(chunks):
                if i:
                    await asyncio.sleep(0.05)
                yield chunk

        from httpx._content import AsyncIteratorByteStream

        return httpx.Response(
            status,
            headers=headers or {"content-type": "text/event-stream"},
            stream=AsyncIteratorByteStream(_yield()),
        )

    return response


def _one_row(pool):
    deadline = time.monotonic() + 5
    while not pool.inserts() and time.monotonic() < deadline:
        time.sleep(0.01)
    (row,) = pool.inserts()
    return row


class TestStreamFlag:
    @pytest.mark.parametrize("flag", [1, "true", "True", "yes", 2.0])
    def test_a_non_boolean_true_stream_is_sent_as_true_with_usage(self, flag):
        payload = {"stream": flag}
        assert with_stream_usage(payload) is True
        assert payload == {"stream": True, "stream_options": {"include_usage": True}}

    @pytest.mark.parametrize("flag", [0, "false", "0", None, ""])
    def test_a_non_boolean_false_stream_is_sent_as_false(self, flag):
        payload = {"stream": flag}
        assert with_stream_usage(payload) is True
        assert payload == {"stream": False}

    def test_stream_one_reaches_upstream_as_true_with_usage(self, monkeypatch):
        app, _pool, bodies = _bodies(monkeypatch, _app(monkeypatch))
        assert _post(app, {"model": "gpt-x", "stream": 1}).status_code == 200
        sent = json.loads(bodies[0])
        assert sent["stream"] is True
        assert sent["stream_options"] == {"include_usage": True}


class TestEstimatedRows:
    def test_a_client_gone_before_the_usage_chunk_leaves_an_estimate(self, monkeypatch):
        content = CONTENT % (b"x" * 400)
        monkeypatch.setattr(tqp, "_stream_response", _two_chunks(content, USAGE))
        app, pool, _broker, _calls = _app(monkeypatch, content_type="text/event-stream")
        asyncio.run(_disconnecting_call(app))
        row = _one_row(pool)
        assert row[COMPLETION] >= 100 and row[PROMPT] > 0
        assert row[TOTAL] == row[PROMPT] + row[COMPLETION]
        assert json.loads(row[EXTRA])["usage"] == "estimated"

    def test_a_stream_that_ends_without_usage_records_an_estimate(self, monkeypatch):
        body = CONTENT % b"hi" * 30 + b"data: [DONE]\n\n"
        app, pool, _broker, _calls = _app(monkeypatch, body=body, content_type="text/event-stream")
        assert _post(app, {"model": "gpt-x", "stream": True}).status_code == 200
        row = _one_row(pool)
        assert row[COMPLETION] == 30  # one token per content event at least
        assert json.loads(row[EXTRA])["usage"] == "estimated"

    def test_a_whole_stream_keeps_its_reported_usage(self, monkeypatch):
        body = CONTENT % b"hi" + USAGE + b"data: [DONE]\n\n"
        app, pool, _broker, _calls = _app(monkeypatch, body=body, content_type="text/event-stream")
        _post(app, {"model": "gpt-x", "stream": True})
        row = _one_row(pool)
        assert (row[PROMPT], row[COMPLETION], row[TOTAL]) == (20000, 9000, 29000)
        assert "usage" not in json.loads(row[EXTRA])

    def test_a_one_shot_body_without_usage_stays_absent(self, monkeypatch):
        app, pool, _broker, _calls = _app(monkeypatch, body=b'{"choices": []}')
        _post(app, {"model": "gpt-x"})
        row = _one_row(pool)
        assert row[TOTAL] == 0 and json.loads(row[EXTRA])["usage"] == "absent"


class TestTapMetered:
    def _tap(self, body: bytes, *, kind="openai", streaming=True, ended=True):
        tap = UsageTap(provider_kind=kind, streaming=streaming)
        for i in range(0, len(body), 5):
            tap.feed(body[i : i + 5])
        tap.ended = ended
        return tap

    def test_a_cut_anthropic_stream_raises_its_output_to_the_estimate(self):
        body = (
            b'data: {"type":"message_start","message":{"usage":'
            b'{"input_tokens":100,"output_tokens":1}}}\n\n'
            + b'data: {"type":"content_block_delta","delta":{"text":"%s"}}\n\n'
            % (b"y" * 80)
        )
        usage, estimated = self._tap(body, kind="anthropic", ended=False).metered(
            request_bytes=4000, generation=True
        )
        assert estimated is True
        assert usage["prompt_tokens"] == 100  # the reported input stands
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 120

    def test_a_get_is_never_estimated(self):
        usage, estimated = self._tap(b"[]", streaming=False, ended=False).metered(
            request_bytes=0, generation=False
        )
        assert (usage, estimated) == (None, False)

    def test_a_cut_one_shot_body_counts_its_bytes(self):
        usage, estimated = self._tap(b"z" * 40, streaming=False, ended=False).metered(
            request_bytes=8, generation=True
        )
        assert estimated and (usage["prompt_tokens"], usage["completion_tokens"]) == (2, 10)

    def test_metadata_is_not_content(self):
        event = b'data: {"id":"%s","model":"m","choices":[{"delta":{"role":"assistant"}}]}\n\n'
        usage, _estimated = self._tap(event % (b"i" * 400), ended=True).metered(
            request_bytes=0, generation=True
        )
        assert usage["completion_tokens"] == 0
