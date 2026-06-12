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

import asyncio
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
from umrs.api.llm_usage import (
    ProxyTimer,
    UsageRecord,
    extract_usage_oneshot,
    extract_usage_streaming,
    fill_usage,
    record_usage,
)
from umrs.server.correlation import get_request_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/llm", tags=["llm-proxy"])

# ---------------------------------------------------------------------------
# DB pool for usage recording (set by http_sidecar startup)
# ---------------------------------------------------------------------------
_usage_db_pool = None


def set_usage_db_pool(pool) -> None:  # noqa: ANN001
    """Inject the DB pool for usage recording."""
    global _usage_db_pool
    _usage_db_pool = pool


def get_usage_db_pool():
    """Return the DB pool for usage recording (may be None)."""
    return _usage_db_pool


def _resolve_litellm_service_key() -> str | None:
    return os.environ.get("GATEWAY_LITELLM_SERVICE_KEY") or os.environ.get(
        "LITELLM_MASTER_KEY"
    )


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
    return req_model in _recognized_model_names(
        endpoint
    ) and req_model != _upstream_model_name(endpoint)


def _parse_variant_from_model(model: str) -> tuple[str, str | None]:
    """Parse a variant hint from a model name suffix.

    Convention: ``DeepSeek-V4-Pro-128k`` → model ``DeepSeek-V4-Pro``,
    variant ``128k``. The suffix is the last ``-`` delimited token
    that matches a known pattern (all-lowercase, or numeric-with-unit).

    Returns:
        (model, variant) tuple. variant is None if no suffix detected.
    """
    # Known variant suffixes: lowercase alphanumeric with optional unit suffix
    # e.g. "128k", "1m", "tep", "tep-128k", "256k"
    parts = model.rsplit("-", 1)
    if len(parts) < 2:
        return model, None
    base, suffix = parts
    # Heuristic: suffix is a variant if it's all lowercase and looks like
    # a context-length hint or a strategy tag (not a model name component)
    _VARIANT_PATTERN = __import__("re").compile(
        r"^\d+[km]?$|^tep(?:-\d+[km]?)?$|^canary$"
    )
    if _VARIANT_PATTERN.match(suffix):
        return base, suffix
    return model, None


def _discover_endpoint(
    registry: EndpointRegistry,
    project_id: str,
    model: str,
) -> RegisteredEndpoint:
    """
    Discover a ready endpoint for (project_id, model).

    Uses least-connections selection when multiple endpoints exist for
    the same model. Falls back to first-match if select_endpoint returns
    None (e.g. no ready endpoints).

    If the model name contains a variant suffix (e.g. "DeepSeek-V4-Pro-128k"),
    the variant is extracted and used to filter candidates.

    Raises HTTPException if no endpoint is available.
    """
    # Try variant-aware selection first
    base_model, variant = _parse_variant_from_model(model)
    endpoint = registry.select_endpoint(
        project_id, base_model, variant=variant, strategy="least_connections"
    )

    if not endpoint:
        # Try exact model name (no variant parsing)
        endpoint = registry.select_endpoint(
            project_id, model, strategy="least_connections"
        )

    if not endpoint:
        # Last resort: scan by served_model_name (R-15 fallback)
        all_eps = registry.list_all(
            project_id=project_id,
            status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
        )
        for ep in all_eps:
            served = ep.metadata.get("served_model_name") or _resolve_served_model_name(
                ep.model
            )
            if served == model or served == base_model:
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
        logger.debug(
            "Rewriting model for upstream: %s -> %s", req_model, upstream_model
        )
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
    hop_by_hop = frozenset(
        {
            "host",
            "connection",
            "keep-alive",
            "transfer-encoding",
            "upgrade",
            "authorization",
            "x-project-id",
            "x-user-id",
            "content-length",
        }
    )
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
    usage_record: UsageRecord | None = None,
    timer: ProxyTimer | None = None,
    registry: EndpointRegistry | None = None,
    job_id: str | None = None,
) -> StreamingResponse:
    """Stream upstream response to the caller.

    The httpx client must stay open until the generator is fully consumed
    by the ASGI server, so we close it in the generator's ``finally``.
    When *usage_record* is provided, the last few SSE chunks are
    inspected for a ``usage`` block and the record is persisted
    fire-and-forget after the stream ends.

    When *registry* and *job_id* are provided, ``decrement_requests``
    is called in the generator's ``finally`` block to accurately track
    load for the entire streaming duration.
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

    # Keep the last few chunks for usage extraction (streaming).
    # Only the tail matters — the final SSE event carries ``usage``.
    _tail_chunks: list[bytes] = []
    _MAX_TAIL = 5

    async def _yield():
        try:
            async for chunk in response.aiter_bytes():
                if usage_record is not None:
                    _tail_chunks.append(chunk)
                    if len(_tail_chunks) > _MAX_TAIL:
                        _tail_chunks.pop(0)
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()
            # Decrement active requests after stream is fully consumed
            if registry and job_id:
                registry.decrement_requests(job_id)
            # Fire-and-forget usage recording
            if usage_record is not None:
                usage_record.status_code = response.status_code
                if timer is not None:
                    usage_record.duration_ms = timer.elapsed_ms()
                usage = extract_usage_streaming(_tail_chunks)
                if usage:
                    fill_usage(usage_record, usage)
                record_usage(_usage_db_pool, usage_record)

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
    usage_record: UsageRecord | None = None,
    timer: ProxyTimer | None = None,
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
            # Extract and record usage (fire-and-forget)
            if usage_record is not None:
                usage_record.status_code = response.status_code
                if timer is not None:
                    usage_record.duration_ms = timer.elapsed_ms()
                usage = extract_usage_oneshot(response.content)
                if usage:
                    fill_usage(usage_record, usage)
                record_usage(_usage_db_pool, usage_record)

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
    registry: EndpointRegistry | None = None,
) -> Response:
    """
    Proxy a request to the upstream LiteLLM endpoint.

    Handles both streaming and non-streaming responses.
    Records token usage fire-and-forget when a DB pool is available.

    When *registry* is provided, increments ``active_requests`` before
    proxying and decrements after completion.  For streaming responses,
    decrement is deferred until the async generator is fully consumed
    (inside its ``finally`` block) so load tracking remains accurate.
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

    # Build usage record for billing
    timer = ProxyTimer()
    usage_record: UsageRecord | None = None
    if _usage_db_pool is not None:
        # Resolve endpoint tier from metadata
        ep_tier = (
            endpoint.metadata.get("tier") or endpoint.metadata.get("owner_type") or ""
        )
        # Requested model name (before rewrite)
        req_model = ""
        body_json = _json_body(body)
        if body_json:
            req_model = body_json.get("model", "")
        usage_record = UsageRecord(
            request_id=get_request_id(),
            user_id=user_id,
            project_id=project_id,
            model=req_model or endpoint.model,
            served_model=_upstream_model_name(endpoint),
            endpoint_job_id=endpoint.job_id,
            endpoint_tier=str(ep_tier),
            is_streaming=is_streaming,
        )

    if registry:
        registry.increment_requests(endpoint.job_id)

    if is_streaming:
        return await _proxy_streaming(
            request,
            upstream_url,
            headers,
            body,
            usage_record=usage_record,
            timer=timer,
            registry=registry,
            job_id=endpoint.job_id,
        )
    try:
        return await _proxy_oneshot(
            request,
            upstream_url,
            headers,
            body,
            usage_record=usage_record,
            timer=timer,
        )
    finally:
        if registry:
            registry.decrement_requests(endpoint.job_id)


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
    if (
        not verified_project_id
        and os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict"
    ):
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
    response = await _proxy_to_litellm(
        request, endpoint, "v1/chat/completions", user_id, project_id,
        registry=registry,
    )

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
    response = await _proxy_to_litellm(
        request, endpoint, "v1/completions", user_id, project_id,
        registry=registry,
    )
    registry.mark_used(endpoint.job_id)

    return response


def _synth_model_entry(endpoint: RegisteredEndpoint) -> dict:
    """Minimal OpenAI-compatible model object for a single endpoint.

    Used as a fallback when the endpoint's own ``/v1/models`` listing
    cannot be fetched, so the model still appears in the aggregate list.
    """
    model_id = _upstream_model_name(endpoint)
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": str(endpoint.metadata.get("tier") or "rivault"),
    }


async def _fetch_endpoint_models(endpoint: RegisteredEndpoint) -> list[dict]:
    """Fetch the ``/v1/models`` listing for a single endpoint.

    Returns the list of model objects reported by the endpoint's LiteLLM.
    Falls back to a synthesized single-model entry on any transport or
    JSON error so a transiently unreachable endpoint never removes its
    model from the aggregate list.
    """
    headers: dict[str, str] = {}
    if _LITELLM_SERVICE_KEY:
        headers["Authorization"] = f"Bearer {_LITELLM_SERVICE_KEY}"
    url = f"{endpoint.litellm_url}/v1/models"
    try:
        async with httpx.AsyncClient(timeout=_PROXY_TIMEOUT) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to list models for endpoint %s: %s", endpoint.job_id, exc
        )
        return [_synth_model_entry(endpoint)]

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data:
        return [_synth_model_entry(endpoint)]
    return [entry for entry in data if isinstance(entry, dict) and entry.get("id")]


@router.get("/v1/models")
async def list_models(
    request: Request,
    registry: EndpointRegistry = Depends(get_registry),
):
    """
    List available models.

    Aggregates models across all ready endpoints for the project so that
    every served model is discoverable (not just the first endpoint's).
    Entries are deduplicated by model id.
    """
    _user_id, project_id = _get_caller_identity(request)

    endpoints = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )

    if not endpoints:
        return {"object": "list", "data": []}

    results = await asyncio.gather(*(_fetch_endpoint_models(ep) for ep in endpoints))

    merged: dict[str, dict] = {}
    for entries in results:
        for entry in entries:
            model_id = entry.get("id")
            if model_id and model_id not in merged:
                merged[model_id] = entry

    # Apply deployment-scoped visibility (e.g. frontier-only on AI4S).
    from .model_visibility import is_model_visible

    data = [entry for entry in merged.values() if is_model_visible(entry.get("id"))]

    return {"object": "list", "data": data}


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
    response = await _proxy_to_litellm(
        request, endpoint, "v1/embeddings", user_id, project_id,
        registry=registry,
    )
    registry.mark_used(endpoint.job_id)

    return response
