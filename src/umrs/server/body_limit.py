"""Cap the size of JSON request bodies (2026-10-21, cycle-2 umrs-api lane).

Found live through the gateway: every JSON API location inherits nginx's
``client_max_body_size 0`` (unlimited), and FastAPI reads and parses a JSON
body whole. One authenticated ``POST /gw/jobs`` with a 300 MB JSON body
raised the UMRS process from 238 MB to a 1.16 GB peak RSS (858 MB held
afterwards); a few concurrent ones exhaust the server that also runs the
ZMQ control plane and the durable job runtime.

JSON (and body-without-content-type) requests above ``UMRS_MAX_JSON_BODY_BYTES``
(default 64 MiB) answer a typed 413 before the body is read when the length
is declared (nginx always declares it), or as soon as a streamed body passes
the cap. Uploads keep their own limits: multipart, octet-stream and every
other non-JSON content type pass untouched.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import HTTPException

_DEFAULT_MAX_JSON_BODY_BYTES = 64 * 1024 * 1024
_JSON_TYPES = ("application/json", "+json", "text/json")


def max_json_body_bytes() -> int:
    """``UMRS_MAX_JSON_BODY_BYTES`` (default 64 MiB; 0 or invalid -> default)."""
    try:
        value = int(os.environ.get("UMRS_MAX_JSON_BODY_BYTES", "") or 0)
    except ValueError:
        value = 0
    return value if value > 0 else _DEFAULT_MAX_JSON_BODY_BYTES


def _header(scope: dict[str, Any], name: bytes) -> str:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def _is_json_body(scope: dict[str, Any]) -> bool:
    content_type = _header(scope, b"content-type").split(";", 1)[0].strip().lower()
    return not content_type or any(t in content_type for t in _JSON_TYPES)


def _declared_length(scope: dict[str, Any]) -> int | None:
    raw = _header(scope, b"content-length").strip()
    return int(raw) if raw.isdigit() else None


class _BodyTooLarge(HTTPException):
    """Raised from ``receive`` when a streamed JSON body passes the cap; an
    HTTPException, so a route reading the body answers the typed 413."""

    def __init__(self, limit: int) -> None:
        super().__init__(413, f"JSON request body exceeds {limit} bytes")


async def _send_413(send: Any, limit: int) -> None:
    body = json.dumps(
        {
            "error_id": "request.too_large",
            "message": f"JSON request body exceeds {limit} bytes",
        }
    ).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class JsonBodyLimitMiddleware:
    """Pure ASGI middleware: refuse JSON bodies above the configured cap."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or not _is_json_body(scope):
            await self.app(scope, receive, send)
            return
        limit = max_json_body_bytes()
        declared = _declared_length(scope)
        if declared is not None and declared > limit:
            await _send_413(send, limit)
            return
        await self._limited(scope, receive, send, limit)

    async def _limited(self, scope: dict[str, Any], receive: Any, send: Any, limit: int) -> None:
        seen = 0
        started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > limit:
                    raise _BodyTooLarge(limit)
            return message

        async def tracking_send(message: dict[str, Any]) -> None:
            nonlocal started
            started = started or message.get("type") == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not started:
                await _send_413(send, limit)
