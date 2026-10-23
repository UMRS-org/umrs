"""Token usage of the inference proxy's relayed calls (ADR-099 D-4).

The proxy streams provider answers through unbuffered. ``UsageTap`` keeps a
bounded copy while it does (the first 16 KiB and the last 64 KiB), which is
where providers put the counts:

- OpenAI-style (``openai``, ``openai_compat``, ``azure_openai``, the site
  source): the last ``usage`` object (the final chunk of a stream asked with
  ``stream_options.include_usage``, or the one-shot body's);
- Anthropic: ``message_start``'s ``usage`` (input and cache; that event
  only) merged with the last ``message_delta`` ``usage`` (output,
  cumulative), or the one-shot body's top-level ``usage`` (parsed as JSON
  when the tail holds the whole body, so a ``usage`` key inside content or
  tool input never counts);
- Google: the last ``usageMetadata``.

``record_proxy_usage`` writes one ``llm_usage_log`` row per relayed call
(``origin='infer_proxy'``) through ``api.llm_usage.record_usage``, after the
response was relayed. Bodies, headers and tokens are never recorded.

When the provider's counts cannot be trusted to be whole (a stream that
ended without usage, or any generation call the relay did not see to its
end: the client left before the final usage chunk) the row carries a
conservative estimate instead of zero, tagged ``extra.usage = "estimated"``
(ADR-099 D-4, Amendment 2): the prompt is the request body's bytes / 4
(:class:`PromptMeter`: a base64 run, an inline image, counts at most
:data:`BLOB_TOKENS`, Amendment 3), the completion the larger of the streamed
content bytes / 4 and the number of content events (a whole body's bytes / 4
when it is not an event stream).
A count the provider did report is kept, and raised to the estimate only
where the estimate is larger. Any other answer without usage (a one-shot
body, a listing) records zero tokens and ``extra.usage = "absent"``.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any

HEAD_BYTES = 16 * 1024
TAIL_BYTES = 64 * 1024

#: An event-stream line longer than this is counted whole as content.
LINE_BYTES = 256 * 1024

ORIGIN = "infer_proxy"
SITE_LABEL_PREFIX = "site:"

_DECODER = json.JSONDecoder()
_MODEL_RE = re.compile(r'(?<!\\)"(?:model|modelVersion)"\s*:\s*"([^"\\]{1,128})"')


def _key_re(key: str) -> re.Pattern[str]:
    # An unescaped key followed by an object: string content carries \" quotes.
    return re.compile(r'(?<!\\)"' + key + r'"\s*:\s*\{')


_USAGE_RE = _key_re("usage")
_GOOGLE_RE = _key_re("usageMetadata")


def _objects(text: str, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    """Every decodable object that follows ``pattern`` in ``text``, in order."""
    found: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        try:
            obj, _end = _DECODER.raw_decode(text, match.end() - 1)
        except ValueError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    return found


def _last(text: str, pattern: re.Pattern[str]) -> dict[str, Any] | None:
    objects = _objects(text, pattern)
    return objects[-1] if objects else None


def _sse_event(line: str) -> dict[str, Any] | None:
    """One SSE ``data:`` line as a JSON object, else None."""
    if not line.startswith("data:"):
        return None
    try:
        event = json.loads(line[5:])
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _message_start_usage(head: str) -> dict[str, Any] | None:
    """The ``usage`` of an Anthropic stream's ``message_start`` event only
    (never a ``usage`` key that happens to sit in content or tool input)."""
    for line in head.splitlines():
        event = _sse_event(line.strip())
        if event is not None and event.get("type") == "message_start":
            usage = (event.get("message") or {}).get("usage")
            return dict(usage) if isinstance(usage, dict) else None
    return None


#: Subtrees and keys of a stream event that are not generated text.
_NOT_CONTENT = frozenset(
    {
        "id",
        "object",
        "created",
        "model",
        "modelVersion",
        "responseId",
        "system_fingerprint",
        "role",
        "type",
        "index",
        "finish_reason",
        "stop_reason",
        "finishReason",
        "logprobs",
        "signature",
        "usage",
        "usageMetadata",
        "safetyRatings",
        "citationMetadata",
    }
)


def _text_bytes(node: Any) -> int:
    """UTF-8 bytes of the generated text in one decoded stream event."""
    if isinstance(node, str):
        return len(node.encode("utf-8", errors="replace"))
    if isinstance(node, dict):
        return sum(_text_bytes(v) for k, v in node.items() if k not in _NOT_CONTENT)
    if isinstance(node, list):
        return sum(_text_bytes(v) for v in node)
    return 0


def _event_bytes(data: bytes) -> int:
    """Content bytes of one ``data:`` payload; an undecodable one counts whole."""
    try:
        return _text_bytes(json.loads(data))
    except (ValueError, RecursionError):
        return len(data)


def _quarter(n: int) -> int:
    return math.ceil(max(0, n) / 4)


#: A run of base64 characters this long is binary data (an image, a file),
#: not text: no prose or code goes 1 KiB without a space or punctuation.
BLOB_RUN = 1024

#: The estimate's charge for one such run (ADR-099 D-4, Amendment 3): a
#: vision prompt's image costs ~0.3-5k tokens whatever its bytes, never
#: bytes / 4 (a 1.5 MB PNG would read as ~500k). Shorter runs count / 4.
BLOB_TOKENS = 4096

#: The base64 alphabet, the URL-safe one, and JSON's escaped ``\/``.
_RUN_RE = re.compile(rb"[A-Za-z0-9+/=_\\-]+")


class PromptMeter:
    """The prompt side of an estimate, fed the request body as it is sent:
    text bytes / 4, plus each base64 run of :data:`BLOB_RUN` or more at most
    :data:`BLOB_TOKENS` (a run may span chunks; nothing is kept but counts)."""

    def __init__(self) -> None:
        self._text = 0  # bytes outside long runs
        self._blob_tokens = 0
        self._run = 0  # the open run's length (it may continue in the next chunk)

    def feed(self, chunk: bytes) -> None:
        pos = 0
        for match in _RUN_RE.finditer(chunk):
            if match.start() > pos:
                self._close()
                self._text += match.start() - pos
            self._run += match.end() - match.start()
            pos = match.end()
        if pos < len(chunk):
            self._close()
            self._text += len(chunk) - pos

    def _close(self) -> None:
        if self._run >= BLOB_RUN:
            self._blob_tokens += _run_tokens(self._run)
        else:
            self._text += self._run
        self._run = 0

    def tokens(self) -> int:
        """The prompt estimate of what was fed so far."""
        if self._run >= BLOB_RUN:
            return _quarter(self._text) + self._blob_tokens + _run_tokens(self._run)
        return _quarter(self._text + self._run) + self._blob_tokens


def _run_tokens(run: int) -> int:
    return min(_quarter(run), BLOB_TOKENS)


def _merged(
    reported: dict[str, int] | None, estimate: dict[str, int]
) -> tuple[dict[str, int], bool]:
    """``reported`` raised to ``estimate`` where it is smaller (True when it
    was): a reported prompt count stands, the completion takes the larger."""
    if not reported:
        return dict(estimate), True
    merged = dict(reported)
    if not merged.get("prompt_tokens"):
        merged["prompt_tokens"] = estimate["prompt_tokens"]
    if estimate["completion_tokens"] > merged.get("completion_tokens", 0):
        merged["completion_tokens"] = estimate["completion_tokens"]
    whole = merged["prompt_tokens"] + merged["completion_tokens"]
    raised = merged != reported
    merged["total_tokens"] = max(int(merged.get("total_tokens") or 0), whole)
    return merged, raised


class UsageTap:
    """Bounded observer of a relayed response: head 16 KiB + tail 64 KiB,
    plus a running count of an event stream's generated text."""

    def __init__(self, *, provider_kind: str, streaming: bool) -> None:
        self.provider_kind = provider_kind or ""
        self.streaming = streaming
        self.ended = False  # the relay read the upstream answer to its end
        self._head = bytearray()
        self._tail = bytearray()
        self._total = 0
        self._line = bytearray()
        self._events = 0  # data events
        self._content_events = 0  # data events that carried text
        self._content_bytes = 0

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        self._total += len(chunk)
        room = HEAD_BYTES - len(self._head)
        if room > 0:
            self._head += chunk[:room]
        self._tail += chunk[-TAIL_BYTES:]
        if len(self._tail) > TAIL_BYTES:
            del self._tail[: len(self._tail) - TAIL_BYTES]
        if self.streaming:
            self._scan(chunk)

    def _scan(self, chunk: bytes) -> None:
        """Count the complete event-stream lines of ``chunk`` (bounded carry)."""
        self._line += chunk
        if b"\n" not in chunk:
            if len(self._line) > LINE_BYTES:
                self._count_data(bytes(self._line))
                self._line.clear()
            return
        *lines, rest = bytes(self._line).split(b"\n")
        self._line = bytearray(rest)
        for line in lines:
            self._count_line(line)

    def _count_line(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if data and data != b"[DONE]":
            self._count_data(data)

    def _count_data(self, data: bytes) -> None:
        self._events += 1
        size = _event_bytes(data)
        self._content_bytes += size
        self._content_events += 1 if size else 0

    def estimate(
        self, request_bytes: int, prompt_tokens: int | None = None
    ) -> dict[str, int]:
        """The conservative token estimate of this call (ADR-099 D-4); the
        prompt is ``prompt_tokens`` (a :class:`PromptMeter`'s) when given."""
        if self._line:
            self._count_line(bytes(self._line))
            self._line.clear()
        if self._events:
            completion = max(self._content_events, _quarter(self._content_bytes))
        else:
            completion = _quarter(self._total)
        prompt = _quarter(request_bytes) if prompt_tokens is None else prompt_tokens
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    def metered(
        self,
        *,
        request_bytes: int,
        generation: bool,
        prompt_tokens: int | None = None,
    ) -> tuple[dict[str, int] | None, bool]:
        """``(usage, estimated)`` for the call's row.

        The provider's counts when they are whole; the estimate (merged into
        what was reported) for a generation call (``generation``: a POST)
        that streamed without usage or that the relay did not see to its end.
        """
        reported = self.usage()
        whole = self.ended and (reported is not None or not self.streaming)
        if whole or not generation:
            return reported, False
        return _merged(reported, self.estimate(request_bytes, prompt_tokens))

    def _text(self, buf: bytearray) -> str:
        return bytes(buf).decode("utf-8", errors="replace")

    def raw_usage(self) -> dict[str, Any] | None:
        """The provider's own usage object (merged for Anthropic streams).

        A one-shot body held whole is parsed as JSON and only its top-level
        usage counts; the key search is the fallback for a body larger than
        the tail and for streams."""
        tail = self._text(self._tail)
        if self.provider_kind == "google":
            return self._top_level("usageMetadata") or _last(tail, _GOOGLE_RE)
        if self.provider_kind == "anthropic":
            return self._anthropic(tail)
        found = self._top_level("usage") or self._top_level("usageMetadata")
        return found or _last(tail, _USAGE_RE) or _last(tail, _GOOGLE_RE)

    def _top_level(self, key: str) -> dict[str, Any] | None:
        """``body[key]`` of a one-shot body the tail holds whole, else None."""
        if self.streaming or self._total > TAIL_BYTES:
            return None
        try:
            body = json.loads(bytes(self._tail))
        except ValueError:
            return None
        found = body.get(key) if isinstance(body, dict) else None
        return dict(found) if isinstance(found, dict) else None

    def _anthropic(self, tail: str) -> dict[str, Any] | None:
        """One-shot: the body's top-level ``usage``. Stream: ``message_start``'s
        usage (input, cache) merged with the last ``message_delta`` usage."""
        if not self.streaming:
            return self._top_level("usage") or _last(tail, _USAGE_RE)
        merged = _message_start_usage(self._text(self._head)) or {}
        merged.update(_last(tail, _USAGE_RE) or {})
        return merged or None

    def usage(self) -> dict[str, int] | None:
        """The ``_parse_usage_dict`` keys, or None when the answer had none."""
        from umrs.api.llm_usage import _parse_usage_dict

        raw = self.raw_usage()
        if not raw:
            return None
        parsed = _parse_usage_dict(raw)
        if not parsed["total_tokens"]:
            parsed["total_tokens"] = (
                parsed["prompt_tokens"] + parsed["completion_tokens"]
            )
        return parsed

    def model(self) -> str:
        """The model the answer names (``model`` / ``modelVersion``), else ""."""
        match = _MODEL_RE.search(self._text(self._head))
        return match.group(1) if match else ""


def _endpoint(source_id: str) -> tuple[str, str]:
    """(endpoint_job_id, endpoint_tier) for a source label."""
    if source_id.startswith(SITE_LABEL_PREFIX):
        return source_id[len(SITE_LABEL_PREFIX) :][:64], "site"
    return "", "user_source"


def build_usage_record(
    *,
    lease: Any,
    session_id: str,
    model: str,
    served_model: str,
    source_id: str,
    path: str,
    status: int,
    streaming: bool,
    duration_ms: float,
    usage: dict[str, int] | None,
    estimated: bool = False,
) -> Any:
    """The ``UsageRecord`` of one relayed call (no body, header or token)."""
    from umrs.api.llm_usage import UsageRecord

    job_id, tier = _endpoint(source_id)
    record = UsageRecord(
        user_id=str(getattr(lease, "owner_id", "") or ""),
        project_id=str(getattr(lease, "project", "") or ""),
        model=model or "",
        served_model=served_model or model or "",
        endpoint_job_id=job_id,
        endpoint_tier=tier,
        request_id=uuid.uuid4().hex,
        is_streaming=bool(streaming),
        duration_ms=float(duration_ms),
        status_code=int(status),
        extra={
            "source_id": source_id,
            "path": path.split("?", 1)[0],
            "lease_id": str(getattr(lease, "lease_id", "") or ""),
        },
        venue=str(getattr(lease, "venue", "") or ""),
        session_id=session_id,
        origin=ORIGIN,
        emit_metrics=False,
    )
    _fill(record, usage, estimated)
    return record


def _fill(record: Any, usage: dict[str, int] | None, estimated: bool) -> None:
    """The token columns, and ``extra.usage`` when the provider gave no whole
    count: ``absent`` (zero) or ``estimated`` (ADR-099 D-4)."""
    from umrs.api.llm_usage import fill_usage

    if usage is None:
        record.extra["usage"] = "absent"
        return
    fill_usage(record, usage)
    if estimated:
        record.extra["usage"] = "estimated"


def record_proxy_usage(
    pool: Any,
    *,
    lease: Any,
    session_id: str,
    model: str,
    served_model: str,
    source_id: str,
    path: str,
    status: int,
    streaming: bool,
    duration_ms: float,
    usage: dict[str, int] | None,
    estimated: bool = False,
) -> None:
    """One llm_usage_log row, origin='infer_proxy' (via api.llm_usage.record_usage)."""
    from umrs.api.llm_usage import record_usage

    record = build_usage_record(
        lease=lease,
        session_id=session_id,
        model=model,
        served_model=served_model,
        source_id=source_id,
        path=path,
        status=status,
        streaming=streaming,
        duration_ms=duration_ms,
        usage=usage,
        estimated=estimated,
    )
    record_usage(pool, record)


__all__ = [
    "BLOB_RUN",
    "BLOB_TOKENS",
    "HEAD_BYTES",
    "ORIGIN",
    "TAIL_BYTES",
    "PromptMeter",
    "UsageTap",
    "build_usage_record",
    "record_proxy_usage",
]
