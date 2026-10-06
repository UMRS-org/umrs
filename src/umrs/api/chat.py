"""UMRS Chat Proxy Router (ADR-012).

Routes ``/umrs/chat/*`` to the **Assistant Service** running inside the
active mini-RiVault container.  This is the platform-native conversational
surface — project-aware, tool-aware, MCP-aware — as opposed to the raw
OpenAI passthrough at ``/gw/llm/v1/*``.

The router reuses the same ``EndpointRegistry`` discovery as the LLM proxy
(``umrs/api/llm.py``) but forwards to the Assistant Service port instead of
the LiteLLM port directly.  The Assistant Service itself handles RAG context
injection, tool routing, and MCP, then calls LiteLLM as its upstream.

Architecture::

    Client → Gateway (NGINX) → UMRS /umrs/chat/* → this router
        → EndpointRegistry discovery (same as LLM)
        → proxy to Assistant Service (port 8090 inside mini-RiVault)
        → Assistant Service calls LiteLLM → vLLM
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from umrs.api._proxy_timeouts import ProxyTimeouts, resolve_proxy_timeouts
from umrs.api.typed_errors import PlatformError, raise_typed
from umrs.orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    RegisteredEndpoint,
    get_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/chat", tags=["chat-proxy"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_ASSISTANT_PORT = int(os.environ.get("UMRS_ASSISTANT_PORT", "8090"))
# Timeouts (W5/U1, 2026-10-05): connect, idle-between-chunks (streaming) and
# total (non-streaming deadline) are separate limits — CHAT_PROXY_CONNECT_
# TIMEOUT / CHAT_PROXY_IDLE_TIMEOUT (600 s) / CHAT_PROXY_TOTAL_TIMEOUT
# (7200 s). CHAT_PROXY_READ_TIMEOUT is a deprecated alias of the idle limit
# (see umrs.api._proxy_timeouts).
_TIMEOUTS: ProxyTimeouts = resolve_proxy_timeouts("CHAT_PROXY")
_CONNECT_TIMEOUT = _TIMEOUTS.connect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant_url(endpoint: RegisteredEndpoint) -> str:
    """Return the Assistant Service base URL for this endpoint.

    Delegates to the first-class ``RegisteredEndpoint.assistant_url``
    property so that service discovery is explicit — callers do not
    need to infer the Assistant port from the LiteLLM port.
    """
    return endpoint.assistant_url


def _discover_endpoint(
    registry: EndpointRegistry,
    project_id: str,
    model: str | None = None,
) -> RegisteredEndpoint:
    """Discover a ready mini-RiVault endpoint for the given project.

    If *model* is provided, attempts to match by model name first.
    Falls back to any ready endpoint for the project.
    """
    if model:
        endpoint = registry.select_endpoint(
            project_id, model, strategy="least_connections"
        )
        if endpoint:
            return endpoint

    # Fallback: any ready endpoint for this project
    all_eps = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )
    if all_eps:
        return min(all_eps, key=lambda ep: ep.active_requests)

    raise_typed(
        PlatformError.chatapi(
            "unavailable",
            f"No active endpoint for project '{project_id}'. "
            "Try again later or provision an endpoint.",
        ),
    )


def _build_upstream_headers(
    request: Request,
    user_id: str,
    project_id: str,
) -> dict[str, str]:
    """Build headers to forward to the Assistant Service."""
    headers: dict[str, str] = {
        "Content-Type": request.headers.get("content-type", "application/json"),
        "X-User-ID": user_id,
        "X-Project-ID": project_id,
        "X-Request-ID": request.headers.get("x-request-id", ""),
    }
    auth = request.headers.get("authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


def _get_caller_identity(request: Request) -> tuple[str, str]:
    """Extract user_id and project_id from request headers.

    Canonical tier chain in ``umrs.api._identity`` (R1).
    """
    from ._identity import verified_identity

    return verified_identity(request, require_project=True)


def _parse_body(body: bytes) -> dict:
    """Parse JSON body, returning empty dict on failure."""
    try:
        return json.loads(body) if body else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
) -> Response:
    """Proxy chat completions to the Assistant Service inside mini-RiVault.

    The Assistant Service handles:
    - RAG context injection (per-request ``project_id``)
    - Tool routing and MCP
    - Forwarding to LiteLLM → vLLM

    The request body is OpenAI-compatible.  Assistant-API extensions
    (``project_id``, ``rag``, ``rag_top_k``, ``rag_threshold``) are passed
    through to the Assistant Service.
    """
    user_id, project_id = _get_caller_identity(request)
    body = await request.body()
    parsed = _parse_body(body)
    model = parsed.get("model")
    is_streaming = parsed.get("stream", False)

    endpoint = _discover_endpoint(registry, project_id, model)
    upstream_base = _assistant_url(endpoint)
    upstream_url = f"{upstream_base}/v1/chat/completions"

    headers = _build_upstream_headers(request, user_id, project_id)

    if registry:
        registry.increment_requests(endpoint.job_id)

    if is_streaming:
        return StreamingResponse(
            _proxy_streaming(request, upstream_url, headers, body),
            media_type="text/event-stream",
        )

    try:
        resp = await _proxy_oneshot(upstream_url, body, headers)

        # Mark used inline
        if registry:
            registry.mark_used(endpoint.job_id)

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    finally:
        if registry:
            registry.decrement_requests(endpoint.job_id)


async def _proxy_oneshot(
    url: str,
    body: bytes,
    headers: dict[str, str],
) -> httpx.Response:
    """Non-streaming POST to the Assistant Service under the total deadline.

    The reply arrives only once the model has finished, so reads may wait up
    to ``CHAT_PROXY_TOTAL_TIMEOUT`` and the same value bounds the whole
    request as an asyncio deadline. Error mapping is unchanged: every
    upstream failure — httpx timeouts included — is ``chatapi.bad_gateway``
    (502; the chatapi taxonomy has no 504 code); the deadline uses the same
    code with its own message.
    """
    timeout = httpx.Timeout(
        connect=_TIMEOUTS.connect,
        read=_TIMEOUTS.total,
        write=_TIMEOUTS.connect,
        pool=5.0,
    )
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await asyncio.wait_for(
                client.post(url, content=body, headers=headers),
                timeout=_TIMEOUTS.total,
            )
    except asyncio.TimeoutError as exc:
        logger.error(
            "Assistant Service %s exceeded the total deadline of %.0fs",
            url,
            _TIMEOUTS.total,
        )
        raise_typed(
            PlatformError.chatapi("bad_gateway", "Assistant Service timed out"),
            cause=exc,
        )
    except httpx.RequestError as exc:
        logger.error("Error proxying POST %s: %s", url, exc)
        raise_typed(
            PlatformError.chatapi(
                "bad_gateway", "Failed to connect to Assistant Service"
            ),
            cause=exc,
        )


@router.get("/v1/models")
async def list_models(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
) -> Response:
    """Proxy model list request to the Assistant Service."""
    user_id, project_id = _get_caller_identity(request)
    endpoint = _discover_endpoint(registry, project_id)
    upstream_url = f"{_assistant_url(endpoint)}/v1/models"
    headers = _build_upstream_headers(request, user_id, project_id)

    try:
        async with httpx.AsyncClient(timeout=_CONNECT_TIMEOUT) as client:
            resp = await client.get(upstream_url, headers=headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except httpx.RequestError as exc:
        logger.error("Error proxying GET %s: %s", upstream_url, exc)
        raise_typed(
            PlatformError.chatapi(
                "bad_gateway", "Failed to connect to Assistant Service"
            ),
            cause=exc,
        )


@router.get("/manifest")
async def get_manifest(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
) -> Response:
    """Proxy manifest request to the Assistant Service for capability discovery."""
    user_id, project_id = _get_caller_identity(request)
    endpoint = _discover_endpoint(registry, project_id)
    upstream_url = f"{_assistant_url(endpoint)}/manifest"
    headers = _build_upstream_headers(request, user_id, project_id)

    try:
        async with httpx.AsyncClient(timeout=_CONNECT_TIMEOUT) as client:
            resp = await client.get(upstream_url, headers=headers)
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except httpx.RequestError as exc:
        logger.error("Error proxying GET %s: %s", upstream_url, exc)
        raise_typed(
            PlatformError.chatapi(
                "bad_gateway", "Failed to connect to Assistant Service"
            ),
            cause=exc,
        )


# ---------------------------------------------------------------------------
# Streaming proxy
# ---------------------------------------------------------------------------


async def _proxy_streaming(
    request: Request,
    url: str,
    headers: dict[str, str],
    body: bytes,
):
    """Proxy a streaming (SSE) request to the Assistant Service.

    Yields raw response bytes for the ``StreamingResponse`` body.  Only
    byte chunks are yielded — status/headers are handled by the enclosing
    ``StreamingResponse`` — so the SSE stream is not corrupted.

    Three failure modes are handled:

    * **Upstream connection failure** (``httpx.RequestError``): emit an
      OpenAI-style SSE error event followed by ``[DONE]``.
    * **Upstream timeout** (``httpx.TimeoutException``): same fail-open
      behaviour with a timeout-specific message.
    * **Client disconnect**: periodically check
      ``request.is_disconnected()`` and break out of the upstream loop so
      that the httpx connection is closed promptly.
    """
    # Connect timeout prevents hanging on dead upstreams; the read limit is
    # the idle gap between chunks (CHAT_PROXY_IDLE_TIMEOUT, per read), so a
    # stream that keeps producing chunks has no total limit.
    timeout = httpx.Timeout(
        connect=_TIMEOUTS.connect,
        read=_TIMEOUTS.idle,
        write=_TIMEOUTS.connect,
        pool=5.0,
    )
    chunk_count = 0
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST", url, content=body, headers=headers
            ) as resp:
                async for chunk in resp.aiter_raw():
                    if not chunk:
                        continue
                    chunk_count += 1
                    # Check for client disconnect every 16 chunks (~ once
                    # per second for typical SSE cadence) to amortise the
                    # cost of ``is_disconnected()``.
                    if chunk_count % 16 == 0 and await request.is_disconnected():
                        logger.info("Client disconnected mid-stream from %s", url)
                        return
                    yield chunk
    except httpx.TimeoutException as exc:
        logger.error("Streaming proxy timeout for %s: %s", url, exc)
        err = json.dumps({"error": {"message": f"Assistant Service timed out: {exc}"}})
        yield f"data: {err}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except httpx.RequestError as exc:
        logger.error("Streaming proxy error for %s: %s", url, exc)
        err = json.dumps(
            {"error": {"message": f"Assistant Service unavailable: {exc}"}}
        )
        yield f"data: {err}\n\n".encode()
        yield b"data: [DONE]\n\n"
