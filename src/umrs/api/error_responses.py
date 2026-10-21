"""Structured API error responses for UMRS.

Provides a consistent JSON error envelope for all UMRS API endpoints.
Machine-readable error_id fields enable automation, while human messages
remain free to change between releases.

Integration with unified warning framework:
    from umrs.shared.warnings import UMRSWarning, WarningCategory

    # Raise a structured warning → converted to HTTP response
    raise UMRSWarning(
        category=WarningCategory.BLOCKING,
        message="Task is still running — download not available yet",
        details={"task_status": "running"}
    )

Response format:
    {
        "error_id": "warning.blocking",
        "message": "Task is still running — download not available yet",
        "request_id": "abc-123",
        "details": {"task_status": "running"},
        "suggestion": "Use ?blocking=true to wait, or retry later"
    }
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

# HTTP status → default error_id mapping
_STATUS_ERROR_MAP: dict[int, str] = {
    400: "request.bad_request",
    401: "auth.unauthorized",
    403: "auth.forbidden",
    404: "resource.not_found",
    408: "request.timeout",
    409: "resource.conflict",
    422: "input.validation_error",
    429: "request.rate_limited",
    500: "server.internal_error",
    502: "upstream.bad_gateway",
    503: "service.unavailable",
    504: "upstream.timeout",
}


def error_id_for_status(status_code: int) -> str:
    """Return a default error_id for an HTTP status code."""
    return _STATUS_ERROR_MAP.get(status_code, f"http.{status_code}")


def structured_error_response(
    status_code: int,
    message: str,
    *,
    error_id: str = "",
    request_id: str = "",
    details: dict[str, Any] | None = None,
    suggestion: str | None = None,
) -> JSONResponse:
    """Build a structured JSON error response."""
    body: dict[str, Any] = {
        "error_id": error_id or error_id_for_status(status_code),
        "message": message,
    }
    if request_id:
        body["request_id"] = request_id
    if details:
        body["details"] = details
    if suggestion:
        body["suggestion"] = suggestion
    return JSONResponse(status_code=status_code, content=body)


async def structured_http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """FastAPI exception handler that returns structured error JSON.

    Also handles UMRSWarning exceptions from the unified warning framework.
    """
    request_id = ""
    try:
        from umrs.server.correlation import get_request_id

        request_id = get_request_id()
    except Exception:
        pass

    message, details = _detail_parts(exc.detail)
    response = structured_error_response(
        exc.status_code,
        message,
        request_id=request_id,
        details=details,
    )
    # The exception's own headers ride along: the typed code
    # (x-platform-error-code), Retry-After, WWW-Authenticate (2026-10-21:
    # dropped here, so no client ever saw a typed code).
    if exc.headers:
        response.headers.update(exc.headers)
    return response


def _detail_parts(detail: Any) -> tuple[str, dict[str, Any] | None]:
    """``(message, details)`` for an HTTPException detail.

    A dict detail keeps its fields under ``details`` (a duplicate job's
    ``existing_job_id``) and lends its ``message``; it was a Python repr.
    """
    if isinstance(detail, str):
        return detail, None
    if isinstance(detail, dict):
        message = detail.get("message") or detail.get("detail") or detail.get("error")
        if not isinstance(message, str) or not message:
            message = json.dumps(detail, default=str)
        return message, detail
    return str(detail), None


async def structured_warning_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Convert a blocking UMRS warning exception to JSON."""
    from umrs.shared.warnings import UMRSWarningError, blocking_warning_response

    if not isinstance(exc, UMRSWarningError):
        raise exc

    request_id = ""
    try:
        from umrs.server.correlation import get_request_id

        request_id = get_request_id()
    except Exception:
        pass

    body = blocking_warning_response(exc.warning)
    if request_id:
        body["request_id"] = request_id

    headers: dict[str, str] = {}
    if exc.warning.retry_after is not None:
        headers["Retry-After"] = str(exc.warning.retry_after)

    return JSONResponse(
        status_code=exc.http_status,
        content=body,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Unhandled exceptions (2026-10-21, cycle-2 umrs-api lane)
# ---------------------------------------------------------------------------
# Without this handler Starlette answered any exception no route caught with
# a plain-text "Internal Server Error" 500: a stopped Postgres read as a
# server bug on every store-backed route, and a NUL in a JSON body (which
# Postgres text cannot hold) read the same way.

_NUL_MARKERS = ("NUL (0x00)", '"UTF8": 0x00')
_UNAVAILABLE_RETRY_AFTER = "5"


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """``exc`` and the exceptions it was raised from or while handling."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain and len(chain) < 16:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _database_error_types() -> tuple[type[BaseException], ...]:
    """Connection-level failures of either Postgres driver."""
    types: list[type[BaseException]] = []
    try:
        import psycopg2
        import psycopg2.pool

        types += [psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.pool.PoolError]
    except ImportError:
        pass
    try:
        import asyncpg

        types += [
            asyncpg.exceptions.PostgresConnectionError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.CannotConnectNowError,
        ]
    except ImportError:
        pass
    return tuple(types)


def _classify_unhandled(exc: BaseException) -> tuple[int, str]:
    """``(status, message)`` for an exception no route handled."""
    chain = _exception_chain(exc)
    if any(isinstance(e, _database_error_types()) for e in chain):
        return 503, "Database unavailable; retry shortly"
    if any(any(m in str(e) for m in _NUL_MARKERS) for e in chain):
        return 400, "Input contains a NUL (0x00) character, which cannot be stored"
    return 500, "Internal server error"


def _request_id_for(request: Request) -> str:
    from umrs.server.correlation import get_request_id

    request_id = get_request_id()  # "" outside a request context
    if request_id:
        return request_id
    header = request.headers.get("x-request-id") or request.headers.get("x-correlation-id")
    if header:
        return header
    import uuid

    return uuid.uuid4().hex


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Typed envelope for an exception no route caught (Starlette re-raises
    it after this response, so the traceback still reaches the server log;
    the client never sees it)."""
    status_code, message = _classify_unhandled(exc)
    response = structured_error_response(
        status_code, message, request_id=_request_id_for(request)
    )
    if status_code == 503:
        response.headers["Retry-After"] = _UNAVAILABLE_RETRY_AFTER
    return response
