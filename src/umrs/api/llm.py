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

import json
import logging
import os
from typing import Optional

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

# LiteLLM service key — injected as Authorization header when proxying.
# Set via GATEWAY_LITELLM_SERVICE_KEY env var (same as the Python Gateway used).
_LITELLM_SERVICE_KEY: Optional[str] = os.environ.get("GATEWAY_LITELLM_SERVICE_KEY")

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
    Bidirectional model alias rewriting.

    - If client sends served name (e.g. "Qwen/Qwen3-8B"), rewrite to catalog name
    - If client sends catalog name (e.g. "qwen3-8b"), rewrite to served name
    """
    served = endpoint.metadata.get("served_model_name") or _resolve_served_model_name(endpoint.model)
    catalog = endpoint.model

    if not served or served == catalog:
        return body

    try:
        body_json = json.loads(body)
    except (ValueError, json.JSONDecodeError):
        return body

    req_model = body_json.get("model")
    if not req_model:
        return body

    if req_model == served and served != catalog:
        body_json["model"] = catalog
        logger.debug("Rewriting model (served→catalog): %s → %s", req_model, catalog)
        return json.dumps(body_json).encode()

    if req_model == catalog and catalog != served:
        body_json["model"] = served
        logger.debug("Rewriting model (catalog→served): %s → %s", req_model, served)
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
        try:
            is_streaming = json.loads(body).get("stream", False)
        except (ValueError, json.JSONDecodeError):
            pass

    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT,
        read=_READ_TIMEOUT,
        write=_READ_TIMEOUT,
        pool=5.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            if is_streaming:
                req = client.build_request(
                    method=request.method,
                    url=upstream_url,
                    headers=headers,
                    content=body,
                )
                response = await client.send(req, stream=True)

                async def stream_response():
                    try:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                    finally:
                        await response.aclose()

                return StreamingResponse(
                    stream_response(),
                    status_code=response.status_code,
                    media_type=response.headers.get("content-type", "text/event-stream"),
                )
            else:
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
            )
        except httpx.RequestError as exc:
            logger.error("Error proxying to %s: %s", upstream_url, exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to connect to upstream service",
            )


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
    project_id = request.headers.get("x-project-id", "")
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
