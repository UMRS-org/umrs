"""
LLM Usage Extraction and Recording.

Parses token usage from OpenAI-compatible response payloads (both
oneshot JSON and the final SSE chunk) and persists the record to
``llm_usage_log`` in a fire-and-forget asyncio task so the proxy
latency is unaffected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------


@dataclass
class UsageRecord:
    """Single LLM request usage snapshot."""

    # Caller
    user_id: str
    project_id: str

    # Model / endpoint
    model: str
    served_model: str = ""
    endpoint_job_id: str = ""
    endpoint_tier: str = ""

    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_read_tokens: int = 0
    cached_write_tokens: int = 0

    # Thinking-model token breakdown (GLM-5.2, DeepSeek-V4-Pro, etc.)
    # ``completion_tokens`` INCLUDES ``reasoning_tokens``; the visible
    # response tokens are ``completion_tokens - reasoning_tokens``.
    reasoning_tokens: int = 0
    audio_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0

    # Request meta
    request_id: str = ""
    is_streaming: bool = False
    duration_ms: float = 0.0
    status_code: int = 200
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------
# Extraction helpers
# ------------------------------------------------------------------


def _parse_usage_dict(usage: dict) -> dict:
    """Normalise an OpenAI / Anthropic / vLLM ``usage`` object.

    Extracts the standard token counts plus the thinking-model
    breakdown (``reasoning_tokens``, ``audio_tokens``) and speculative
    decoding stats (``accepted/rejected_prediction_tokens``).

    For thinking models such as GLM-5.2, ``completion_tokens`` INCLUDES
    ``reasoning_tokens`` — so the visible response token count is
    ``completion_tokens - reasoning_tokens``.  We record both totals
    so billing can apply the correct rate per token class.
    """
    # OpenAI-style sub-dicts may be present but ``null`` (e.g. vLLM),
    # so coerce to ``{}`` before indexing.
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}

    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        # Anthropic-style cache (zero when absent).
        "cached_read_tokens": int(
            prompt_details.get("cached_tokens", 0)
            or usage.get("cache_read_input_tokens", 0)
        ),
        "cached_write_tokens": int(usage.get("cache_creation_input_tokens", 0)),
        # Thinking-model breakdown (GLM-5.2, DeepSeek-V4-Pro, etc.)
        "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0)),
        "audio_tokens": int(
            completion_details.get("audio_tokens", 0)
            or prompt_details.get("audio_tokens", 0)
        ),
        # Speculative decoding stats (vLLM ``spec_token_acceptance_rate``)
        "accepted_prediction_tokens": int(
            usage.get("accepted_prediction_tokens", 0)
        ),
        "rejected_prediction_tokens": int(
            usage.get("rejected_prediction_tokens", 0)
        ),
    }


def extract_usage_oneshot(body: bytes) -> dict | None:
    """Extract usage dict from a full JSON response body.

    Returns ``None`` if the response does not contain a ``usage`` block
    (e.g. error responses, model-list responses).
    """
    try:
        data = json.loads(body)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    return _parse_usage_dict(usage)


def extract_usage_streaming(chunks: list[bytes]) -> dict | None:
    """Extract usage from the final SSE data chunk(s).

    OpenAI streams end with an SSE event whose JSON ``data:`` payload
    contains a ``usage`` field (when ``stream_options.include_usage``
    is set by LiteLLM).

    We scan **backwards** through the collected chunks for the last
    non-empty ``data:`` line.
    """
    # Concatenate bytes then scan lines
    raw = b"".join(chunks)
    for line in reversed(raw.split(b"\n")):
        stripped = line.strip()
        if stripped.startswith(b"data:"):
            payload = stripped[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except (ValueError, json.JSONDecodeError):
                continue
            if isinstance(obj, dict) and "usage" in obj:
                usage = obj["usage"]
                if isinstance(usage, dict):
                    return _parse_usage_dict(usage)
    return None


# ------------------------------------------------------------------
# Async recording (fire-and-forget)
# ------------------------------------------------------------------

_INSERT_SQL = """
INSERT INTO llm_usage_log (
    request_id, user_id, project_id,
    model, served_model, endpoint_job_id, endpoint_tier,
    prompt_tokens, completion_tokens, total_tokens,
    cached_read_tokens, cached_write_tokens,
    reasoning_tokens, audio_tokens,
    accepted_prediction_tokens, rejected_prediction_tokens,
    is_streaming, duration_ms, status_code, extra
) VALUES (
    %s, %s, %s,
    %s, %s, %s, %s,
    %s, %s, %s,
    %s, %s,
    %s, %s,
    %s, %s,
    %s, %s, %s, %s::jsonb
)
"""


def _do_record_sync(pool, record: UsageRecord) -> None:  # noqa: ANN001
    """Execute the INSERT using a psycopg2 ThreadedConnectionPool.

    Runs in a thread so the async event loop is never blocked.
    Also increments Prometheus counters.
    """
    # Prometheus metrics (always, even if DB fails)
    try:
        from umrs.metrics import METRICS

        METRICS.llm_tokens(
            record.model,
            record.prompt_tokens,
            record.completion_tokens,
            user_id=record.user_id,
            project_id=record.project_id,
            cached_read=record.cached_read_tokens,
            cached_write=record.cached_write_tokens,
        )
        METRICS.llm_request(record.model, record.is_streaming)
        if record.duration_ms > 0:
            METRICS.llm_request_duration(record.model, record.duration_ms / 1000.0)
    except Exception:
        logger.debug("Prometheus metrics unavailable", exc_info=True)

    conn = None
    try:
        conn = pool.getconn()
        with conn.cursor() as cur:
            cur.execute(
                _INSERT_SQL,
                (
                    record.request_id,
                    record.user_id,
                    record.project_id,
                    record.model,
                    record.served_model or record.model,
                    record.endpoint_job_id,
                    record.endpoint_tier,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.cached_read_tokens,
                    record.cached_write_tokens,
                    record.reasoning_tokens,
                    record.audio_tokens,
                    record.accepted_prediction_tokens,
                    record.rejected_prediction_tokens,
                    record.is_streaming,
                    record.duration_ms,
                    record.status_code,
                    json.dumps(record.extra),
                ),
            )
        conn.commit()
    except Exception:
        logger.exception("Failed to record LLM usage for request %s", record.request_id)
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
    finally:
        if conn is not None:
            try:
                pool.putconn(conn)
            except Exception:
                pass


def record_usage(pool, record: UsageRecord) -> None:  # noqa: ANN001
    """Schedule a fire-and-forget task to persist *record*.

    The INSERT runs on the default executor (thread pool) so the async
    event loop is not blocked.  When no event loop is running (e.g. sync
    ``DATA_INGESTION`` workers), the INSERT runs inline on the caller thread.
    """
    if pool is None:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _do_record_sync, pool, record)
    except RuntimeError:
        _do_record_sync(pool, record)


def fill_usage(record: UsageRecord, usage: dict) -> None:
    """Copy parsed usage values into *record*.

    ``usage`` is the dict returned by ``_parse_usage_dict``; it always
    contains every key we care about (defaults to 0 when absent).
    """
    record.prompt_tokens = usage.get("prompt_tokens", 0)
    record.completion_tokens = usage.get("completion_tokens", 0)
    record.total_tokens = usage.get("total_tokens", 0)
    record.cached_read_tokens = usage.get("cached_read_tokens", 0)
    record.cached_write_tokens = usage.get("cached_write_tokens", 0)
    record.reasoning_tokens = usage.get("reasoning_tokens", 0)
    record.audio_tokens = usage.get("audio_tokens", 0)
    record.accepted_prediction_tokens = usage.get(
        "accepted_prediction_tokens", 0
    )
    record.rejected_prediction_tokens = usage.get(
        "rejected_prediction_tokens", 0
    )


class ProxyTimer:
    """Simple wall-clock timer for measuring proxy round-trip."""

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000
