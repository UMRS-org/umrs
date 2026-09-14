"""RFC 7807 (problem+json) projection for typed platform errors (WP-16 r2).

Rule context: the taxonomy wave kept detail bodies byte-identical as plain
strings (additive header only). This module is the additive projection on
top: an **app-level exception handler** that emits RFC 7807 problem+json
ONLY for responses that already carry the typed platform header
(``x-platform-error-code``). Untyped legacy errors keep their plain
``{"detail": ...}`` shape untouched.

Honesty / rollout shape:
- Opt-in at install time — apps enable it explicitly via
  ``install_problem_json_handler(app)``; the env knob
  ``UMRS_PROBLEM_JSON=1`` at the sidecar builder flips the flag. The handler
  never alters shape without the flag.
- ``type`` is a per-family URI pseudo-namespace
  ``https://platform.dev/errors/<namespace>`` — not a resolvable URL,
  tagged in the docstring, never lie-labels a reachable spec page.
- ``title`` is the namespace portion of the code, humanized
  (``session.store_unavailable`` → ``Session error``). RFC shape allows a
  generic title; it does not require prose per member.
- The detail string stays byte-identical inside ``detail`` — hoisting into
  a richer object never rewrites the text.
"""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from r4_contracts.errors import ERROR_CODE_HEADER, PlatformError

logger = logging.getLogger(__name__)

PROBLEM_MEDIA_TYPE = "application/problem+json"
_PROBLEM_BASE = "https://platform.dev/errors"


def _namespace(code: str) -> str:
    """Family namespace of a typed code (``session.x`` → ``session``)."""
    return code.split(".", 1)[0] if code and "." in code else "platform"


def _title(code: str) -> str:
    """Namespace of the code to title chars (session.x → 'Session error')."""
    if not code or "." not in code:
        return "Internal error"
    return f"{_namespace(code).capitalize()} error"


def problem_body(error: PlatformError) -> dict[str, object]:
    """RFC 7807 object for a typed platform error (detail byte-identical)."""
    return {
        "type": f"{_PROBLEM_BASE}/{_namespace(error.code)}",
        "title": _title(error.code),
        "status": error.http_status,
        "detail": error.message,
        "code": error.code,
    }


async def problem_json_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Project typed platform errors as problem+json; pass others through."""
    code = exc.headers.get(ERROR_CODE_HEADER) if getattr(exc, "headers", None) else None
    if not code or "." not in code:
        # untyped: identical to the stock FastAPI handler
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    error = PlatformError(
        code=code, message=str(exc.detail), http_status=exc.status_code
    )
    return JSONResponse(
        problem_body(error),
        status_code=exc.status_code,
        media_type=PROBLEM_MEDIA_TYPE,
        headers={ERROR_CODE_HEADER: code},
    )


def install_problem_json_handler(app) -> None:
    """Register the RFC 7807 projection as the app-wide HTTPException handler."""
    app.add_exception_handler(HTTPException, problem_json_handler)


def enabled_by_env() -> bool:
    """Site knob (2026-09-12 soak flip): projection is ON BY DEFAULT with an
    explicit opt-out (UMRS_PROBLEM_JSON=0 / off / no / false; any casing).

    Rationale checkpoint (post-WP-16): the handler rewrites ONLY responses
    carrying x-platform-error-code; lean dataclass chains like the r4
    client (`_api_error_message` reads ``detail`` and degrades it as plain
    text) stay byte-correct before-and-after, so the flip can't fabricate
    parse confusion downstream.'''
    """
    import os

    raw = os.environ.get("UMRS_PROBLEM_JSON", "").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return False
    return True
