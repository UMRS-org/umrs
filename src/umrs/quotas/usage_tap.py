"""Token usage of the inference proxy's relayed calls (ADR-099 D-4).

The proxy streams provider answers through unbuffered. ``UsageTap`` keeps a
bounded copy while it does (the first 16 KiB and the last 64 KiB), which is
where providers put the counts:

- OpenAI-style (``openai``, ``openai_compat``, ``azure_openai``, the site
  source): the last ``usage`` object (the final chunk of a stream asked with
  ``stream_options.include_usage``, or the one-shot body's);
- Anthropic: ``message_start``'s ``usage`` (input and cache) merged with the
  last ``message_delta`` ``usage`` (output, cumulative), or the one-shot
  body's single ``usage``;
- Google: the last ``usageMetadata``.

``record_proxy_usage`` writes one ``llm_usage_log`` row per relayed call
(``origin='infer_proxy'``) through ``api.llm_usage.record_usage``, after the
response was relayed. Bodies, headers and tokens are never recorded; a
response without usage records zero tokens and ``extra.usage = "absent"``.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

HEAD_BYTES = 16 * 1024
TAIL_BYTES = 64 * 1024

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


class UsageTap:
    """Bounded observer of a relayed response: head 16 KiB + tail 64 KiB."""

    def __init__(self, *, provider_kind: str, streaming: bool) -> None:
        self.provider_kind = provider_kind or ""
        self.streaming = streaming
        self._head = bytearray()
        self._tail = bytearray()

    def feed(self, chunk: bytes) -> None:
        if not chunk:
            return
        room = HEAD_BYTES - len(self._head)
        if room > 0:
            self._head += chunk[:room]
        self._tail += chunk[-TAIL_BYTES:]
        if len(self._tail) > TAIL_BYTES:
            del self._tail[: len(self._tail) - TAIL_BYTES]

    def _text(self, buf: bytearray) -> str:
        return bytes(buf).decode("utf-8", errors="replace")

    def raw_usage(self) -> dict[str, Any] | None:
        """The provider's own usage object (merged for Anthropic streams)."""
        tail = self._text(self._tail)
        if self.provider_kind == "google":
            return _last(tail, _GOOGLE_RE)
        if self.provider_kind == "anthropic":
            return self._anthropic(tail)
        return _last(tail, _USAGE_RE) or _last(tail, _GOOGLE_RE)

    def _anthropic(self, tail: str) -> dict[str, Any] | None:
        first = _objects(self._text(self._head), _USAGE_RE)
        last = _last(tail, _USAGE_RE)
        merged: dict[str, Any] = dict(first[0]) if first else {}
        merged.update(last or {})
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
) -> Any:
    """The ``UsageRecord`` of one relayed call (no body, header or token)."""
    from umrs.api.llm_usage import UsageRecord, fill_usage

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
    if usage is None:
        record.extra["usage"] = "absent"
    else:
        fill_usage(record, usage)
    return record


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
    )
    record_usage(pool, record)


__all__ = [
    "HEAD_BYTES",
    "ORIGIN",
    "TAIL_BYTES",
    "UsageTap",
    "build_usage_record",
    "record_proxy_usage",
]
