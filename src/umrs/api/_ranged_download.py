"""Ranged, inline-capable object streaming for ``GET|HEAD /umrs/files/download``.

ADR-115 D-1: one media read route. A browser ``<video>`` seeks with
``Range: bytes=…``; UMRS answers it with ``get_object(Range=…)`` instead of
streaming the whole object, so the edge never needs a presigned URL.

- single satisfiable range → 206 with ``Content-Range`` / ``Content-Length``
- unsatisfiable range → 416 with ``Content-Range: bytes */N``
- multi-range, malformed or non-``bytes`` → the full 200 (RFC 9110 §14.2
  lets a server ignore a Range it does not support)
- ``If-Range`` naming another validator → the full 200, so ranges of two
  object versions are never spliced
- ``?disposition=inline`` only for :data:`INLINE_TYPES`; anything else
  stays an attachment. Every answer carries ``Accept-Ranges``, ``ETag``,
  ``Last-Modified``, ``X-Content-Type-Options: nosniff`` and
  ``Cache-Control: private, no-store``.

Authorization is the route's job and runs before any call here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from email.utils import format_datetime
from typing import Any

from botocore.exceptions import ClientError
from fastapi.responses import Response, StreamingResponse

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

#: Content types a caller may ask to receive inline (ADR-115 D-1).
INLINE_TYPES = frozenset(
    {
        "video/mp4",
        "video/webm",
        "video/quicktime",
        "image/jpeg",
        "image/png",
        "application/json",
        "text/tab-separated-values",
        "text/csv",
    }
)

_CHUNK = 8 * 1024 * 1024


class RangeNotSatisfiable(Exception):
    """The Range names no byte of the object (answer 416)."""


@dataclass(frozen=True)
class ObjectMeta:
    """What ``head_object`` said about the object."""

    key: str
    size: int | None
    etag: str
    last_modified: str
    content_type: str


def _quoted(etag: str) -> str:
    text = (etag or "").strip()
    if not text or text.startswith(('"', "W/")):
        return text
    return f'"{text}"'


def _http_date(value: Any) -> str:
    return format_datetime(value, usegmt=True) if isinstance(value, datetime) else ""


def object_meta(head: dict[str, Any], key: str) -> ObjectMeta:
    """Normalise a ``head_object`` answer."""
    size = head.get("ContentLength")
    return ObjectMeta(
        key=key,
        size=int(size) if size is not None else None,
        etag=_quoted(str(head.get("ETag") or "")),
        last_modified=_http_date(head.get("LastModified")),
        content_type=str(head.get("ContentType") or "application/octet-stream"),
    )


def _int_or_none(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.isdigit() else None


def _suffix_span(length: int | None, size: int) -> tuple[int, int] | None:
    if length is None:
        return None
    if length == 0 or size == 0:
        raise RangeNotSatisfiable
    return max(0, size - length), size - 1


def _span(first: int | None, last_text: str, size: int) -> tuple[int, int] | None:
    if first is None:
        return None
    last = _int_or_none(last_text) if last_text.strip() else None
    if last_text.strip() and (last is None or last < first):
        return None  # malformed: ignored, the full object is served
    if first >= size:
        raise RangeNotSatisfiable
    return first, size - 1 if last is None else min(last, size - 1)


def parse_range(header: str | None, size: int | None) -> tuple[int, int] | None:
    """``(first, last)`` inclusive for one satisfiable byte range.

    ``None`` means "serve the whole object": no header, an unknown size, a
    non-``bytes`` unit, several ranges or a malformed spec.

    Raises:
        RangeNotSatisfiable: the single range starts past the end, or is an
            empty suffix.
    """
    if not header or size is None:
        return None
    unit, _, spec = header.strip().partition("=")
    if unit.strip().lower() != "bytes" or "," in spec or "-" not in spec:
        return None
    first_text, _, last_text = spec.strip().partition("-")
    if not first_text.strip():
        return _suffix_span(_int_or_none(last_text), size)
    return _span(_int_or_none(first_text), last_text, size)


def if_range_allows(if_range: str | None, meta: ObjectMeta) -> bool:
    """True unless ``If-Range`` names a validator the object no longer has.

    An entity tag must match strongly (a weak tag never does); a date must
    equal the object's ``Last-Modified``.
    """
    if not if_range:
        return True
    value = if_range.strip()
    if value.startswith("W/"):
        return False
    if value.startswith('"'):
        return bool(meta.etag) and value == meta.etag
    return bool(meta.last_modified) and value == meta.last_modified


def _safe_filename(key: str) -> str:
    name = key.rsplit("/", 1)[-1]
    return "".join("_" if ch in '"\\' or ord(ch) < 32 else ch for ch in name)


def content_disposition(meta: ObjectMeta, requested: str | None) -> str:
    """``inline`` only when asked for and the type is on the allowlist."""
    base_type = meta.content_type.split(";", 1)[0].strip().lower()
    inline = (requested or "").strip().lower() == "inline" and base_type in INLINE_TYPES
    kind = "inline" if inline else "attachment"
    return f'{kind}; filename="{_safe_filename(meta.key)}"'


def _base_headers(meta: ObjectMeta, disposition: str | None) -> dict[str, str]:
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Disposition": content_disposition(meta, disposition),
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "private, no-store",
    }
    if meta.etag:
        headers["ETag"] = meta.etag
    if meta.last_modified:
        headers["Last-Modified"] = meta.last_modified
    return headers


def _not_satisfiable(meta: ObjectMeta) -> Response:
    return Response(
        status_code=416,
        headers={
            "Content-Range": f"bytes */{meta.size}",
            "Accept-Ranges": "bytes",
            "Cache-Control": "private, no-store",
        },
    )


def _head(s3_client: Any, bucket: str, key: str) -> dict[str, Any]:
    try:
        return s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        raise_typed(PlatformError.files("not_found", f"File not found: {key}"), cause=e)


def _get(s3_client: Any, bucket: str, key: str, span: tuple[int, int] | None) -> Any:
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if span is not None:
        kwargs["Range"] = f"bytes={span[0]}-{span[1]}"
    try:
        return s3_client.get_object(**kwargs)
    except ClientError as e:
        logger.error("S3 error downloading %s/%s: %s", bucket, key, e)
        raise_typed(
            PlatformError.files("download_failed", "Failed to download file"), cause=e
        )


def iter_body(obj: Any, chunk_size: int = _CHUNK):
    """Stream S3 body chunks; close when done."""
    stream = obj["Body"]
    try:
        while chunk := stream.read(chunk_size):
            yield chunk
    finally:
        stream.close()


@dataclass(frozen=True)
class DownloadAsk:
    """The request's read options (``Range``, ``If-Range``, disposition, HEAD)."""

    range_header: str | None = None
    if_range: str | None = None
    disposition: str | None = None
    head_only: bool = False


def _span_for(ask: DownloadAsk, meta: ObjectMeta) -> tuple[int, int] | None:
    if not if_range_allows(ask.if_range, meta):
        return None
    return parse_range(ask.range_header, meta.size)


def _status_and_headers(
    meta: ObjectMeta, ask: DownloadAsk, span: tuple[int, int] | None
) -> tuple[int, dict[str, str]]:
    headers = _base_headers(meta, ask.disposition)
    if span is not None:
        headers["Content-Range"] = f"bytes {span[0]}-{span[1]}/{meta.size}"
        headers["Content-Length"] = str(span[1] - span[0] + 1)
        return 206, headers
    if meta.size is not None:
        headers["Content-Length"] = str(meta.size)
    return 200, headers


def stream_object(s3_client: Any, bucket: str, key: str, ask: DownloadAsk) -> Response:
    """Answer one authorized download: 200, 206, 416 or a bodiless HEAD.

    HEAD and 416 never call ``get_object``; a range reads only its bytes.
    """
    meta = object_meta(_head(s3_client, bucket, key), key)
    try:
        span = _span_for(ask, meta)
    except RangeNotSatisfiable:
        return _not_satisfiable(meta)
    status, headers = _status_and_headers(meta, ask, span)
    if ask.head_only:
        return Response(status_code=status, headers=headers, media_type=meta.content_type)
    obj = _get(s3_client, bucket, key, span)
    return StreamingResponse(
        iter_body(obj), status_code=status, media_type=meta.content_type, headers=headers
    )
