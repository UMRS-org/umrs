"""Structured API error responses for UMRS.

Provides a consistent JSON error envelope for all UMRS API endpoints.
Machine-readable error_id fields enable automation, while human messages
remain free to change between releases.

Response format:
    {
        "error_id": "rag.service_unavailable",
        "message": "RAG service is not reachable",
        "request_id": "abc-123",
        "details": {}
    }
"""

from __future__ import annotations

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
    return JSONResponse(status_code=status_code, content=body)


async def structured_http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """FastAPI exception handler that returns structured error JSON.

    Replaces the default handler so all HTTPException responses include
    error_id and request_id for distributed tracing and automation.
    """
    request_id = ""
    try:
        from umrs.server.correlation import get_request_id
        request_id = get_request_id()
    except Exception:
        pass

    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

    return structured_error_response(
        exc.status_code,
        detail,
        request_id=request_id,
    )
