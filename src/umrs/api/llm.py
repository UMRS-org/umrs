"""
UMRS LLM Proxy Router.

Routes /umrs/llm/v1/* directly to the active LiteLLM endpoint,
resolving the endpoint from the local EndpointRegistry (no HTTP hop).

Implements:
- Model alias rewriting (catalog ↔ served name)
- Service key injection for LiteLLM auth
- Inline mark_used after successful proxy
- Streaming (SSE) support for chat completions
"""

from __future__ import annotations

import contextlib
import json
import logging
import os

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from umrs.orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    RegisteredEndpoint,
    get_registry,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/llm", tags=["llm-proxy"])

def _resolve_litellm_service_key() -> str | None:
    return os.environ.get("GATEWAY_LITELLM_SERVICE_KEY") or os.environ.get("LITELLM_MASTER_KEY")


# LiteLLM service key — injected as Authorization header when proxying.
# Prefer the Gateway-specific env var, but fall back to the mini-RiVault key
# used by readiness probes and Slurm-launched LiteLLM.
_LITELLM_SERVICE_KEY: str | None = _resolve_litellm_service_key()

# Timeouts
_CONNECT_TIMEOUT = float(os.environ.get("LLM_PROXY_CONNECT_TIMEOUT", "10"))
_READ_TIMEOUT = float(os.environ.get("LLM_PROXY_READ_TIMEOUT", "120"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_served_model_name(catalog_name: str) -> str:
    """Resolve catalog name to HuggingFace-style served name (best effort)."""
    mapping = {
        "qwen3-8b": "Qwen/Qwen3-8B",
        "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    }
    return mapping.get(catalog_name, catalog_name)


def _upstream_model_name(endpoint: RegisteredEndpoint) -> str:
    return str(endpoint.metadata.get("served_model_name") or endpoint.model)


def _recognized_model_names(endpoint: RegisteredEndpoint) -> set[str]:
    names = {endpoint.model, _resolve_served_model_name(endpoint.model)}
    served = endpoint.metadata.get("served_model_name")
    if served:
        names.add(str(served))
    return names


def _json_body(body: bytes) -> dict | None:
    try:
        decoded = json.loads(body)
    except (ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _model_needs_rewrite(req_model: str | None, endpoint: RegisteredEndpoint) -> bool:
    if not req_model:
        return False
    return req_model in _recognized_model_names(endpoint) and req_model != _upstream_model_name(endpoint)


def _discover_endpoint(
    registry: EndpointRegistry,
    project_id: str,
    model: str,
) -> RegisteredEndpoint:
    """
    Discover a ready endpoint for (project_id, model).

    Uses registry.get() for O(1) lookup — no HTTP round-trip.
    Falls back to scanning by served_model_name if direct lookup misses.

    Raises HTTPException if no endpoint is available.
    """
    endpoint = registry.get(project_id, model, prefer_warm=True)

    if not endpoint:
        # Try matching by served_model_name in metadata
        all_eps = registry.list_all(
            project_id=project_id,
            status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
        )
        for ep in all_eps:
            served = ep.metadata.get("served_model_name") or _resolve_served_model_name(ep.model)
            if served == model:
                endpoint = ep
                break

    if not endpoint:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No active endpoint for model '{model}'. "
            "Try again later or provision an endpoint.",
        )

    return endpoint


def _rewrite_model_name(body: bytes, endpoint: RegisteredEndpoint) -> bytes:
    """
    Rewrite catalog/alias names to the actual upstream model name.
    """
    body_json = _json_body(body)
    if body_json is None:
        return body

    req_model = body_json.get("model")
    if _model_needs_rewrite(req_model, endpoint):
        upstream_model = _upstream_model_name(endpoint)
        body_json["model"] = upstream_model
        logger.debug("Rewriting model for upstream: %s -> %s", req_model, upstream_model)
        return json.dumps(body_json).encode()

    return body


def _build_upstream_headers(
    request: Request,
    user_id: str,
    project_id: str,
) -> dict[str, str]:
    """Build headers for the upstream LiteLLM request."""
    headers: dict[str, str] = {}

    # Forward safe headers
    hop_by_hop = frozenset({
        "host", "connection", "keep-alive", "transfer-encoding",
        "upgrade", "authorization", "x-project-id", "x-user-id",
        "content-length",
    })
    for key, value in request.headers.items():
        if key.lower() not in hop_by_hop:
            headers[key] = value

    # Inject service key for LiteLLM auth
    if _LITELLM_SERVICE_KEY:
        headers["Authorization"] = f"Bearer {_LITELLM_SERVICE_KEY}"

    # Inject caller identity for audit
    if user_id:
        headers["X-User-ID"] = user_id
    if project_id:
        headers["X-Project-ID"] = project_id

    return headers


_PROXY_TIMEOUT = httpx.Timeout(
    connect=_CONNECT_TIMEOUT,
    read=_READ_TIMEOUT,
    write=_READ_TIMEOUT,
    pool=5.0,
)


async def _proxy_streaming(
    request: Request,
    upstream_url: str,
    headers: dict[str, str],
    body: bytes,
) -> StreamingResponse:
    """Stream upstream response to the caller.

    The httpx client must stay open until the generator is fully consumed
    by the ASGI server, so we close it in the generator's ``finally``.
    """
    client = httpx.AsyncClient(timeout=_PROXY_TIMEOUT)
    try:
        req = client.build_request(
            method=request.method,
            url=upstream_url,
            headers=headers,
            content=body,
        )
        response = await client.send(req, stream=True)
    except httpx.TimeoutException:
        await client.aclose()
        logger.error("Timeout proxying to %s", upstream_url)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request to upstream service timed out",
        ) from None
    except httpx.RequestError as exc:
        await client.aclose()
        logger.error("Error proxying to %s: %s", upstream_url, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to connect to upstream service",
        ) from exc

    async def _yield():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        _yield(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream"),
    )


async def _proxy_oneshot(
    request: Request,
    upstream_url: str,
    headers: dict[str, str],
    body: bytes,
) -> Response:
    """Non-streaming proxy: send request, return full response."""
    async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
        try:
            response = await client.request(
                method=request.method,
                url=upstream_url,
                headers=headers,
                content=body,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.headers.get("content-type"),
            )
        except httpx.TimeoutException:
            logger.error("Timeout proxying to %s", upstream_url)
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request to upstream service timed out",
            ) from None
        except httpx.RequestError as exc:
            logger.error("Error proxying to %s: %s", upstream_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to connect to upstream service",
            ) from exc


async def _proxy_to_litellm(
    request: Request,
    endpoint: RegisteredEndpoint,
    path: str,
    user_id: str,
    project_id: str,
) -> Response:
    """
    Proxy a request to the upstream LiteLLM endpoint.

    Handles both streaming and non-streaming responses.
    """
    upstream_url = f"{endpoint.litellm_url}/{path}"
    body = await request.body()

    # Model alias rewriting
    body = _rewrite_model_name(body, endpoint)

    headers = _build_upstream_headers(request, user_id, project_id)

    # Check if streaming
    is_streaming = False
    if body:
        with contextlib.suppress(ValueError, json.JSONDecodeError):
            is_streaming = json.loads(body).get("stream", False)

    if is_streaming:
        return await _proxy_streaming(request, upstream_url, headers, body)
    return await _proxy_oneshot(request, upstream_url, headers, body)


def _extract_model(body: dict) -> str:
    """Extract model name from request body."""
    model = body.get("model")
    if not model:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing 'model' field in request body",
        )
    return model


def _get_caller_identity(request: Request) -> tuple[str, str]:
    """
    Extract user_id and project_id from request headers.

    These are set by the NGINX auth sidecar via auth_request_set.
    """
    user_id = request.headers.get("x-user-id", "")
    verified_project_id = request.headers.get("x-verified-project-id", "").strip()
    project_id = verified_project_id or request.headers.get("x-project-id", "").strip()
    if not verified_project_id and os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Verified-Project-ID header (Gateway must set this)",
        )
    if not project_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Project-ID header (auth sidecar must set this)",
        )
    return user_id, project_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    prefer_system: bool = Query(True, description="Prefer system warm endpoints"),
    registry: EndpointRegistry = Depends(get_registry),
):
    """
    OpenAI-compatible chat completions endpoint.

    Discovers endpoint from local registry (no HTTP hop), proxies to LiteLLM.
    """
    user_id, project_id = _get_caller_identity(request)
    body = await request.json()
    model = _extract_model(body)

    logger.info("Chat completion: project=%s, model=%s", project_id, model)

    endpoint = _discover_endpoint(registry, project_id, model)
    response = await _proxy_to_litellm(request, endpoint, "v1/chat/completions", user_id, project_id)

    # Mark used inline (direct registry call, no HTTP round-trip)
    registry.mark_used(endpoint.job_id)

    return response


@router.post("/v1/completions")
async def completions(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """OpenAI-compatible completions endpoint (legacy)."""
    user_id, project_id = _get_caller_identity(request)
    body = await request.json()
    model = _extract_model(body)

    endpoint = _discover_endpoint(registry, project_id, model)
    response = await _proxy_to_litellm(request, endpoint, "v1/completions", user_id, project_id)
    registry.mark_used(endpoint.job_id)

    return response


@router.get("/v1/models")
async def list_models(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """
    List available models.

    Returns models from all active endpoints for the project.
    """
    user_id, project_id = _get_caller_identity(request)

    # Get any ready endpoint to query for models
    endpoints = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )

    if not endpoints:
        return {"object": "list", "data": []}

    # Use the first available endpoint
    endpoint = endpoints[0]
    return await _proxy_to_litellm(request, endpoint, "v1/models", user_id, project_id)


@router.post("/v1/embeddings")
async def embeddings(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """OpenAI-compatible embeddings endpoint."""
    user_id, project_id = _get_caller_identity(request)
    body = await request.json()
    model = _extract_model(body)

    endpoint = _discover_endpoint(registry, project_id, model)
    response = await _proxy_to_litellm(request, endpoint, "v1/embeddings", user_id, project_id)
    registry.mark_used(endpoint.job_id)

    return response
