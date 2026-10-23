"""ADR-099 D-4 (RS-1 lane Q): token usage of the inference proxy's calls.

``UsageTap`` reads the counts from a bounded copy of the relayed answer
(OpenAI, Anthropic and Google, streamed and one-shot); ``_parse_usage_dict``
learns the Anthropic and Google keys (it read only OpenAI's before, so an
Anthropic usage object counted zero tokens); the usage row carries venue,
session and ``origin='infer_proxy'`` in the 0126 columns.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from umrs.api.llm_usage import (
    _INSERT_SQL,
    UsageRecord,
    _insert_statement,
    _parse_usage_dict,
)
from umrs.quotas.usage_tap import (
    HEAD_BYTES,
    TAIL_BYTES,
    UsageTap,
    build_usage_record,
    record_proxy_usage,
)


def _sse(*events: dict, done: bool = True) -> bytes:
    body = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events)
    return body + (b"data: [DONE]\n\n" if done else b"")


def _tap(kind: str, body: bytes, *, streaming: bool, chunk: int = 7) -> UsageTap:
    tap = UsageTap(provider_kind=kind, streaming=streaming)
    for i in range(0, len(body), chunk):  # small chunks split keys across feeds
        tap.feed(body[i : i + chunk])
    return tap


OPENAI_STREAM = _sse(
    {"id": "c1", "model": "gpt-x", "choices": [{"delta": {"content": "hi"}}], "usage": None},
    {"id": "c1", "model": "gpt-x", "choices": [{"delta": {"content": " there"}}], "usage": None},
    {
        "id": "c1",
        "model": "gpt-x",
        "choices": [],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
            "total_tokens": 150,
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    },
)

ANTHROPIC_STREAM = (
    b"event: message_start\n"
    + b'data: {"type":"message_start","message":{"model":"claude-x","usage":'
    + b'{"input_tokens":100,"cache_read_input_tokens":20,"cache_creation_input_tokens":5,'
    + b'"output_tokens":1}}}\n\n'
    + b"event: content_block_delta\n"
    + b'data: {"type":"content_block_delta","delta":{"text":"the \\"usage\\": {1} word"}}\n\n'
    + b"event: message_delta\n"
    + b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}\n\n'
    + b"event: message_stop\n"
    + b'data: {"type":"message_stop"}\n\n'
)

GOOGLE_STREAM = _sse(
    {
        "candidates": [{"content": {"parts": [{"text": "a"}]}}],
        "modelVersion": "gemini-x",
        "usageMetadata": {"promptTokenCount": 50},
    },
    {
        "candidates": [{"content": {"parts": [{"text": "b"}]}}],
        "modelVersion": "gemini-x",
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 12,
            "thoughtsTokenCount": 8,
            "totalTokenCount": 70,
        },
    },
    done=False,
)


class TestUsageTapStreams:
    def test_openai_stream_reads_the_final_usage_chunk(self):
        tap = _tap("openai", OPENAI_STREAM, streaming=True)
        usage = tap.usage()
        assert usage["prompt_tokens"] == 120 and usage["completion_tokens"] == 30
        assert usage["total_tokens"] == 150 and usage["reasoning_tokens"] == 10
        assert tap.model() == "gpt-x"

    def test_anthropic_stream_merges_message_start_and_the_last_delta(self):
        usage = _tap("anthropic", ANTHROPIC_STREAM, streaming=True).usage()
        # input 100 + cache read 20 + cache write 5; the escaped "usage" in the
        # text is never mistaken for a usage object.
        assert usage["prompt_tokens"] == 125
        assert usage["completion_tokens"] == 42
        assert usage["total_tokens"] == 167
        assert usage["cached_read_tokens"] == 20 and usage["cached_write_tokens"] == 5

    def test_google_stream_reads_the_last_usage_metadata(self):
        tap = _tap("google", GOOGLE_STREAM, streaming=True)
        usage = tap.usage()
        assert usage["prompt_tokens"] == 50
        assert usage["completion_tokens"] == 20  # candidates + thoughts
        assert usage["reasoning_tokens"] == 8 and usage["total_tokens"] == 70
        assert tap.model() == "gemini-x"

    def test_stream_without_usage_answers_none(self):
        body = _sse({"choices": [{"delta": {"content": "x"}}], "usage": None})
        assert _tap("openai", body, streaming=True).usage() is None


class TestUsageTapOneShot:
    def test_openai_body(self):
        body = json.dumps(
            {"model": "m", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}
        ).encode()
        usage = _tap("openai_compat", body, streaming=False, chunk=1000).usage()
        assert usage["prompt_tokens"] == 7 and usage["total_tokens"] == 10

    def test_anthropic_body(self):
        body = json.dumps(
            {"model": "claude-x", "content": [], "usage": {"input_tokens": 9, "output_tokens": 4}}
        ).encode()
        usage = _tap("anthropic", body, streaming=False).usage()
        assert usage["prompt_tokens"] == 9 and usage["completion_tokens"] == 4
        assert usage["total_tokens"] == 13

    def test_google_body(self):
        body = json.dumps(
            {"candidates": [], "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 6}}
        ).encode()
        usage = _tap("google", body, streaming=False).usage()
        assert usage["prompt_tokens"] == 5 and usage["total_tokens"] == 11

    def test_error_body_has_no_usage(self):
        body = b'{"error": {"message": "bad request"}}'
        assert _tap("openai", body, streaming=False).usage() is None

    def test_anthropic_body_counts_only_the_top_level_usage(self):
        """Q6: a ``usage`` object inside tool input is not the call's usage
        (old code merged the head's first match: 5000 cache tokens extra)."""
        tool_input = {"usage": {"cache_creation_input_tokens": 5000, "output_tokens": 77}}
        body = json.dumps(
            {
                "model": "claude-x",
                "content": [{"type": "tool_use", "name": "t", "input": tool_input}],
                "usage": {"input_tokens": 9, "output_tokens": 4},
            }
        ).encode()
        usage = _tap("anthropic", body, streaming=False).usage()
        assert usage["prompt_tokens"] == 9 and usage["completion_tokens"] == 4
        assert usage["total_tokens"] == 13 and usage["cached_write_tokens"] == 0

    def test_openai_body_counts_only_the_top_level_usage(self):
        nested = {"prompt_tokens": 900, "completion_tokens": 900}
        body = json.dumps(
            {
                "model": "m",
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                "choices": [{"message": {"content": None, "usage": nested}}],
            }
        ).encode()
        usage = _tap("openai", body, streaming=False, chunk=1000).usage()
        assert usage["prompt_tokens"] == 7 and usage["total_tokens"] == 10

    def test_anthropic_stream_reads_message_start_not_an_earlier_usage_key(self):
        """A ``usage`` key ahead of ``message_start`` in the head (a ping, a
        content block) never feeds the input count."""
        stream = (
            b"event: ping\n"
            + b'data: {"type":"ping","usage":{"cache_creation_input_tokens":5000}}\n\n'
            + ANTHROPIC_STREAM
        )
        usage = _tap("anthropic", stream, streaming=True).usage()
        assert usage["prompt_tokens"] == 125 and usage["cached_write_tokens"] == 5


class TestBounds:
    def test_head_and_tail_stay_bounded_and_the_end_is_kept(self):
        filler = _sse(
            *({"choices": [{"delta": {"content": "x" * 200}}]} for _ in range(2000)), done=False
        )
        tail = _sse(
            {
                "choices": [],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        )
        tap = _tap("openai", filler + tail, streaming=True, chunk=4096)
        assert len(tap._head) == HEAD_BYTES and len(tap._tail) == TAIL_BYTES
        assert tap.usage()["total_tokens"] == 3

    def test_one_chunk_larger_than_the_tail(self):
        tap = UsageTap(provider_kind="openai", streaming=False)
        tap.feed(b"x" * (TAIL_BYTES * 3))
        assert len(tap._tail) == TAIL_BYTES and len(tap._head) == HEAD_BYTES


class TestParseUsageDict:
    def test_openai_shape_unchanged(self):
        out = _parse_usage_dict({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})
        assert (out["prompt_tokens"], out["completion_tokens"], out["total_tokens"]) == (3, 4, 7)

    def test_anthropic_keys_are_counted(self):
        out = _parse_usage_dict({"input_tokens": 10, "output_tokens": 5})
        assert out["prompt_tokens"] == 10 and out["completion_tokens"] == 5
        assert out["total_tokens"] == 15

    def test_google_keys_are_counted(self):
        out = _parse_usage_dict({"promptTokenCount": 2, "candidatesTokenCount": 3})
        assert out["prompt_tokens"] == 2 and out["total_tokens"] == 5


LEASE = SimpleNamespace(lease_id="lease-1", owner_id="alice", project="opossum", venue="rccs")


class _Cursor:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params=None):
        self.sink.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class _Conn:
    def __init__(self, sink):
        self.sink = sink

    def cursor(self):
        return _Cursor(self.sink)

    def commit(self):
        pass

    def rollback(self):
        pass


class _Pool:
    def __init__(self):
        self.sql: list = []

    def getconn(self):
        return _Conn(self.sql)

    def putconn(self, _conn):
        pass


class TestUsageRow:
    def test_one_row_with_origin_venue_and_session(self):
        pool = _Pool()
        record_proxy_usage(
            pool,
            lease=LEASE,
            session_id="sess-1",
            model="gpt-x",
            served_model="",
            source_id="src-a",
            path="/s/sess-1/v1/chat/completions?key=never-recorded",
            status=200,
            streaming=True,
            duration_ms=12.5,
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )
        assert len(pool.sql) == 1
        sql, params = pool.sql[0]
        assert "venue, session_id, origin" in sql
        assert params[-3:] == ("rccs", "sess-1", "infer_proxy")
        assert params[1:3] == ("alice", "opossum")
        assert params[9] == 7  # total_tokens
        extra = json.loads(params[19])
        assert extra["path"] == "/s/sess-1/v1/chat/completions"
        assert "never-recorded" not in json.dumps(params, default=str)

    def test_absent_usage_is_zero_tokens_and_marked(self):
        record = build_usage_record(
            lease=LEASE,
            session_id="s",
            model="",
            served_model="",
            source_id="site:job-9",
            path="/p",
            status=502,
            streaming=False,
            duration_ms=1.0,
            usage=None,
        )
        assert record.total_tokens == 0 and record.extra["usage"] == "absent"
        assert record.endpoint_job_id == "job-9" and record.endpoint_tier == "site"
        assert record.emit_metrics is False

    def test_umrs_llm_rows_keep_the_pre_0126_insert(self):
        sql, params = _insert_statement(UsageRecord(user_id="u", project_id="p", model="m"))
        assert sql is _INSERT_SQL and len(params) == 20
