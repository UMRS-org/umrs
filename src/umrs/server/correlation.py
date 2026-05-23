"""Request correlation ID middleware for UMRS.

Captures X-Request-ID (from Gateway/NGINX) or X-Correlation-ID (from R4 SDK)
and makes it available to all handlers via contextvars. The ID is also injected
into the response headers for end-to-end traceability.

Precedence:
  1. X-Request-ID (set by NGINX $request_id) — canonical gateway-level ID
  2. X-Correlation-ID (set by R4 SDK) — client-level correlation
  3. Auto-generated UUID4 fallback

Usage in handlers:
    from umrs.server.correlation import get_request_id

    request_id = get_request_id()  # returns the active correlation ID
"""

from __future__ import annotations

import contextvars
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

# Context variable holding the current request's correlation ID.
_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

# Header names (case-insensitive in HTTP, but canonical forms for setting).
_HEADER_REQUEST_ID = "X-Request-ID"
_HEADER_CORRELATION_ID = "X-Correlation-ID"


def get_request_id() -> str:
    """Return the correlation ID for the current request context.

    Returns an empty string if called outside of a request context.
    """
    return _request_id_ctx.get()


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Extract or generate a correlation ID and propagate it."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Extract from incoming headers (NGINX X-Request-ID takes precedence)
        request_id = (
            request.headers.get(_HEADER_REQUEST_ID)
            or request.headers.get(_HEADER_CORRELATION_ID)
            or str(uuid.uuid4())
        )

        token = _request_id_ctx.set(request_id)
        try:
            response = await call_next(request)
            response.headers[_HEADER_REQUEST_ID] = request_id
            return response
        finally:
            _request_id_ctx.reset(token)
