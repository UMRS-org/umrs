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
import re

import httpx
from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse

from umrs.api._proxy_timeouts import ProxyTimeouts, resolve_proxy_timeouts
from umrs.api.llm_usage import (
    ProxyTimer,
    UsageRecord,
    extract_usage_oneshot,
    extract_usage_streaming,
    fill_usage,
    record_usage,
)
from umrs.api.typed_errors import PlatformError, raise_typed
from umrs.metrics import METRICS
from umrs.orchestrator.endpoint_access import EndpointCaller, usable_endpoints
from umrs.orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    RegisteredEndpoint,
    get_registry,
)
from umrs.server.correlation import get_request_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/llm", tags=["llm-proxy"])

# ---------------------------------------------------------------------------
# DB pool for usage recording (set by http_sidecar startup)
# ---------------------------------------------------------------------------
_usage_db_pool = None


def set_usage_db_pool(pool) -> None:
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

# Timeouts (W5/U1, 2026-10-05) — connect, idle-between-chunks (streaming) and
# total (non-streaming deadline) are separate limits; see _proxy_timeouts for
# the env names. LLM_PROXY_READ_TIMEOUT is a deprecated alias of the idle
# limit (it no longer caps non-streaming requests).
_TIMEOUTS: ProxyTimeouts = resolve_proxy_timeouts("LLM_PROXY")
# GET /v1/models fan-out keeps the pre-split 120 s read limit: one hung
# endpoint must not stall the aggregate listing for the idle/total limits.
_LIST_READ_TIMEOUT = float(os.environ.get("LLM_PROXY_LIST_TIMEOUT", "120"))


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
    # Include variant-suffixed names so the model rewriter can
    # rewrite disambiguated ids (e.g. ``GLM-5.2-FP8-pd``) back to
    # the upstream served model name.
    if endpoint.variant:
        # list() snapshot is load-bearing: names grows during this loop
        # (variant suffixes), iterating it raw raises RuntimeError
        # 'Set changed size during iteration'. Pinned by test_llm_proxy.
        for base in list(names):  # noqa: C408 — intentional snapshot copy
            names.add(f"{base}-{endpoint.variant}")
        names.add(f"{endpoint.model}-{endpoint.variant}")
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


def _list_ready_endpoints(
    registry: EndpointRegistry, project_id: str
) -> list[RegisteredEndpoint]:
    """List READY/DEGRADED endpoints for the project."""
    return registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )


def _variant_suffixed_fallback(
    registry: EndpointRegistry, project_id: str, model: str
) -> RegisteredEndpoint | None:
    """Match variant-suffixed model names from the picker.

    The model picker constructs ids like ``GLM-5.2-FP8-pd`` by
    suffixing the served model name with the endpoint's variant.
    We reverse this: find an endpoint whose variant-suffixed name
    matches the requested model.
    """
    for ep in _list_ready_endpoints(registry, project_id):
        served = ep.metadata.get("served_model_name") or _resolve_served_model_name(
            ep.model
        )
        if ep.variant:
            for candidate in (
                f"{served}-{ep.variant}",
                f"{ep.model}-{ep.variant}",
            ):
                if candidate == model:
                    return ep
    return None


def _served_model_fallback(
    registry: EndpointRegistry, project_id: str, model: str, base_model: str
) -> RegisteredEndpoint | None:
    """Last resort: scan by served_model_name (R-15 fallback)."""
    for ep in _list_ready_endpoints(registry, project_id):
        served = ep.metadata.get("served_model_name") or _resolve_served_model_name(
            ep.model
        )
        if served == model or served == base_model:
            return ep
    return None


def _discover_endpoint(
    registry: EndpointRegistry,
    project_id: str,
    model: str,
    caller: EndpointCaller | None = None,
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
        endpoint = _variant_suffixed_fallback(registry, project_id, model)

    if not endpoint:
        endpoint = _served_model_fallback(registry, project_id, model, base_model)

    if not endpoint:
        endpoint = _variant_pool_match(registry, project_id, model)

    if not endpoint and caller is not None:
        # ADR-078: the caller's own project first, then endpoints of other
        # projects whose access list admits the caller (user/group/system).
        endpoint = registry.select_shared_endpoint(
            caller, base_model, variant=variant
        ) or registry.select_shared_endpoint(caller, model)

    if not endpoint:
        raise_typed(
            PlatformError.llm(
                "unavailable",
                f"No active endpoint for model '{model}'. "
                "Try again later or provision an endpoint.",
            ),
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


def _streaming_timeout() -> httpx.Timeout:
    """Streaming: the read limit is the idle gap between chunks (per read)."""
    return httpx.Timeout(
        connect=_TIMEOUTS.connect,
        read=_TIMEOUTS.idle,
        write=_TIMEOUTS.idle,
        pool=5.0,
    )


def _oneshot_timeout() -> httpx.Timeout:
    """Non-streaming: the reply arrives only once the model has finished, so a
    per-read idle limit would cut long turns; reads may wait up to the total
    limit, which ``_proxy_oneshot`` also enforces as an overall deadline."""
    return httpx.Timeout(
        connect=_TIMEOUTS.connect,
        read=_TIMEOUTS.total,
        write=_TIMEOUTS.idle,
        pool=5.0,
    )


def _listing_timeout() -> httpx.Timeout:
    """Per-endpoint ``GET /v1/models`` during the aggregate listing."""
    return httpx.Timeout(
        connect=_TIMEOUTS.connect,
        read=_LIST_READ_TIMEOUT,
        write=_LIST_READ_TIMEOUT,
        pool=5.0,
    )


async def _open_stream(
    request: Request,
    upstream_url: str,
    headers: dict[str, str],
    body: bytes,
) -> tuple[httpx.AsyncClient, httpx.Response]:
    """Build and send the upstream streaming request (504/502-mapped)."""
    client = httpx.AsyncClient(timeout=_streaming_timeout())
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
        raise_typed(
            PlatformError.llm(
                "gateway_timeout", "Request to upstream service timed out"
            )
        )
    except httpx.RequestError as exc:
        await client.aclose()
        logger.error("Error proxying to %s: %s", upstream_url, exc)
        raise_typed(
            PlatformError.llm("bad_gateway", "Failed to connect to upstream service"),
            cause=exc,
        )
    return client, response


_CONTENT_TOKEN_RE = re.compile(rb'"content"\s*:\s*"((?:\\.|[^"\\])*)"')


async def _proxy_streaming(
    request: Request,
    upstream_url: str,
    headers: dict[str, str],
    body: bytes,
    usage_record: UsageRecord | None = None,
    timer: ProxyTimer | None = None,
    registry: EndpointRegistry | None = None,
    job_id: str | None = None,
    model: str = "",
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
    client, response = await _open_stream(request, upstream_url, headers, body)

    # Keep the last few chunks for usage extraction (streaming).
    # Only the tail matters — the final SSE event carries ``usage``.
    _tail_chunks: list[bytes] = []
    _MAX_TAIL = 5
    _ttft_observed = False
    _ttfc_observed = False
    _decode_start = None
    _content_delta_count = 0

    async def _yield():
        nonlocal _ttft_observed, _ttfc_observed, _decode_start, _content_delta_count
        try:
            async for chunk in response.aiter_bytes():
                if not _ttft_observed and timer is not None:
                    # P11 TTFT SLO (ADR-014 follow-up): observed exactly once
                    # per stream at the first upstream chunk; anchor = the
                    # ProxyTimer at the route boundary — chunk, not decoded
                    # token, same honest semantics as the retired gateway.
                    _ttft_observed = True
                    METRICS.llm_first_token(model, timer.elapsed_ms() / 1000)
                if chunk:
                    # P11 decode legs: first non-empty content delta (TTFC) +
                    # stream-end content-delta rate (decode pace).
                    content_match = _CONTENT_TOKEN_RE.search(chunk)
                    if content_match and content_match.group(1):
                        _content_delta_count += 1
                        if not _ttfc_observed and timer is not None:
                            _ttfc_observed = True
                            METRICS.llm_first_content(model, timer.elapsed_ms() / 1000)
                            _decode_start = timer.elapsed_ms()
                if usage_record is not None:
                    _tail_chunks.append(chunk)
                    if len(_tail_chunks) > _MAX_TAIL:
                        _tail_chunks.pop(0)
                yield chunk
        finally:
            # P11 decode-rate leg: per-stream content-delta rate at end.
            if timer is not None and _decode_start is not None:
                window_sec = max(0.0, timer.elapsed_ms() / 1000)
                if window_sec > 0:
                    METRICS.llm_decode_content_rate(
                        model, _content_delta_count / window_sec
                    )
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


# httpx hands the body back de-chunked and decoded, and this server frames and
# dates its own reply. Forwarding the upstream's framing made nginx refuse the
# reply whenever the model server answered chunked ("sent Content-Length and
# Transfer-Encoding at the same time" → 502, shown as 503; ollama, live
# 2026-10-09: a team role failed on a 200 from the model).
_UPSTREAM_FRAMING_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
        "content-encoding",
        "date",
        "server",
    }
)


def _oneshot_response_headers(response: httpx.Response) -> dict[str, str]:
    """The upstream's response headers, less framing that describes another body."""
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in _UPSTREAM_FRAMING_HEADERS
    }


def _record_oneshot_usage(
    response: httpx.Response,
    usage_record: UsageRecord | None,
    timer: ProxyTimer | None,
) -> None:
    """Fill and persist (fire-and-forget) the usage row of a one-shot reply."""
    if usage_record is None:
        return
    usage_record.status_code = response.status_code
    if timer is not None:
        usage_record.duration_ms = timer.elapsed_ms()
    usage = extract_usage_oneshot(response.content)
    if usage:
        fill_usage(usage_record, usage)
    record_usage(_usage_db_pool, usage_record)


async def _proxy_oneshot(
    request: Request,
    upstream_url: str,
    headers: dict[str, str],
    body: bytes,
    usage_record: UsageRecord | None = None,
    timer: ProxyTimer | None = None,
) -> Response:
    """Non-streaming proxy: send request, return full response.

    The whole request (send + full body read) runs under the total deadline
    (``LLM_PROXY_TOTAL_TIMEOUT``); exceeding it maps to the same 504 as an
    httpx timeout.
    """
    async with httpx.AsyncClient(timeout=_oneshot_timeout()) as client:
        try:
            response = await asyncio.wait_for(
                client.request(
                    method=request.method,
                    url=upstream_url,
                    headers=headers,
                    content=body,
                ),
                timeout=_TIMEOUTS.total,
            )
            _record_oneshot_usage(response, usage_record, timer)

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=_oneshot_response_headers(response),
                media_type=response.headers.get("content-type"),
            )
        except asyncio.TimeoutError:
            logger.error(
                "Upstream %s exceeded the total deadline of %.0fs",
                upstream_url,
                _TIMEOUTS.total,
            )
            raise_typed(
                PlatformError.llm(
                    "gateway_timeout", "Request to upstream service timed out"
                )
            )
        except httpx.TimeoutException:
            logger.error("Timeout proxying to %s", upstream_url)
            raise_typed(
                PlatformError.llm(
                    "gateway_timeout", "Request to upstream service timed out"
                )
            )
        except httpx.RequestError as exc:
            logger.error("Error proxying to %s: %s", upstream_url, exc)
            raise_typed(
                PlatformError.llm(
                    "bad_gateway", "Failed to connect to upstream service"
                ),
                cause=exc,
            )


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
    usage_record = _build_usage_record(
        body, endpoint, user_id, project_id, is_streaming
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
            model=endpoint.model or "",
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


def _build_usage_record(
    body: bytes,
    endpoint: RegisteredEndpoint,
    user_id: str,
    project_id: str,
    is_streaming: bool,
) -> UsageRecord | None:
    """Build a usage record for billing when a DB pool is available."""
    if _usage_db_pool is None:
        return None
    # Resolve endpoint tier from metadata
    ep_tier = endpoint.metadata.get("tier") or endpoint.metadata.get("owner_type") or ""
    # Requested model name (before rewrite)
    req_model = ""
    body_json = _json_body(body)
    if body_json:
        req_model = body_json.get("model", "")
    return UsageRecord(
        request_id=get_request_id(),
        user_id=user_id,
        project_id=project_id,
        model=req_model or endpoint.model,
        served_model=_upstream_model_name(endpoint),
        endpoint_job_id=endpoint.job_id,
        endpoint_tier=str(ep_tier),
        is_streaming=is_streaming,
    )


def _extract_model(body: dict) -> str:
    """Extract model name from request body."""
    model = body.get("model")
    if not model:
        raise_typed(
            PlatformError.llm("bad_request", "Missing 'model' field in request body"),
        )
    return model


def _get_caller_identity(request: Request) -> tuple[str, str]:
    """
    Extract user_id and project_id from request headers (canonical, R1).

    These are set by the NGINX auth sidecar via auth_request_set.
    """
    from ._identity import verified_identity

    return verified_identity(request, require_project=True)


_CIRCLES_TTL_SEC = 60.0
_circles_cache: dict[str, tuple[float, frozenset[str]]] = {}


def _cached_circles(request: Request, user_id: str) -> frozenset[str]:
    """The caller's active circle refs, cached per user for one minute.

    Called lazily — only when a circle-scoped endpoint is the deciding
    candidate — so the common routing path never takes a pool connection.
    """
    import time

    now = time.monotonic()
    hit = _circles_cache.get(user_id)
    if hit is not None and now - hit[0] < _CIRCLES_TTL_SEC:
        return hit[1]
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return frozenset()
    from .circles import circles_for_user

    conn = pool.getconn()
    try:
        circles = frozenset(circles_for_user(conn, user_id))
    except Exception as exc:  # noqa: BLE001 - no circles beats a failed request
        logger.warning("circle lookup for %s failed: %s", user_id, exc)
        return frozenset()
    finally:
        pool.putconn(conn)
    _circles_cache[user_id] = (now, circles)
    return circles


def _endpoint_caller(request: Request, user_id: str, project_id: str) -> EndpointCaller:
    """Routing identity for ADR-078 access checks (gateway-verified headers)."""
    admin = (request.headers.get("x-verified-is-admin") or "").strip().lower()
    return EndpointCaller(
        user_id=user_id or "",
        project_id=project_id or "",
        is_admin=admin in ("true", "1", "yes"),
        circles_resolver=lambda: _cached_circles(request, user_id) if user_id else (),
    )


def _variant_pool_match(
    registry: EndpointRegistry, project_id: str, model: str
) -> RegisteredEndpoint | None:
    """Pool YAML entry names are stored in variant (e.g. glm-5.2-v2)."""
    all_eps = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )
    variant_matches = [ep for ep in all_eps if ep.variant == model]
    if variant_matches:
        return min(variant_matches, key=lambda ep: ep.active_requests)
    return None


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

    endpoint = _discover_endpoint(
        registry, project_id, model, _endpoint_caller(request, user_id, project_id)
    )
    response = await _proxy_to_litellm(
        request,
        endpoint,
        "v1/chat/completions",
        user_id,
        project_id,
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

    endpoint = _discover_endpoint(
        registry, project_id, model, _endpoint_caller(request, user_id, project_id)
    )
    response = await _proxy_to_litellm(
        request,
        endpoint,
        "v1/completions",
        user_id,
        project_id,
        registry=registry,
    )
    registry.mark_used(endpoint.job_id)

    return response


def _synth_model_entry(endpoint: RegisteredEndpoint) -> dict:
    """Minimal OpenAI-compatible model object for a single endpoint.

    Used as a fallback when the endpoint's own ``/v1/models`` listing
    cannot be fetched, so the model still appears in the aggregate list.
    """
    base_id = endpoint.model or _upstream_model_name(endpoint)
    return {
        "id": base_id,
        "object": "model",
        "created": 0,
        "owned_by": str(endpoint.metadata.get("tier") or "rivault"),
        "catalog_model": endpoint.model,
    }


def _variant_is_redundant(variant: str, base_id: str) -> bool:
    """True when the variant duplicates the base model id.

    A variant that is equal (case-insensitively, separators normalized) to the
    base model name adds no disambiguation and only produces a confusing
    ``<Model>-<model>`` compound id (e.g. base ``Qwen3.6-35B-A3B-NVFP4`` with
    variant ``qwen3.6-35b-a3b-nvfp4``). Advertising that compound breaks model
    selection in clients, which resolve the clean served/catalog name via
    ``list_endpoints`` and provisioning checks.
    """

    def _norm(value: str) -> str:
        return value.strip().lower().replace("_", "-")

    return _norm(variant) == _norm(str(base_id))


def _positive_int(value: object) -> int | None:
    """``value`` when it is a positive int (bools excluded), else None."""
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _optional_bool(value: object) -> bool | None:
    """``value`` when it is a bool, else None (unknown)."""
    return value if isinstance(value, bool) else None


# Model-limit fields every listed entry carries (W5/U3, 2026-10-05; D-11).
# A value is null when the platform does not know it. Sources:
#   context_window            the served max_model_len — the upstream entry's
#                             own ``max_model_len`` (vLLM reports it), else
#                             the endpoint's registry metadata
#   max_output_tokens         serving-profile ``max_output_tokens``, else the
#                             model's generation_config ``max_new_tokens``
#   reasoning                 registry capability / model-family flag
#   supports_reasoning_effort registry ``reasoning_effort_levels`` non-empty
# The last three are recorded on the endpoint at provision time
# (``umrs.model_registry.capabilities.endpoint_limit_metadata``); endpoints
# provisioned before 2026-10-05 list them as null until re-provisioned.
_MODEL_LIMIT_FIELDS = (
    "context_window",
    "max_output_tokens",
    "reasoning",
    "supports_reasoning_effort",
)


def _model_limits(entry: dict, endpoint: RegisteredEndpoint) -> dict[str, object]:
    """Client-facing limits for one listed model (None = unknown)."""
    metadata = endpoint.metadata or {}
    return {
        "context_window": _positive_int(entry.get("max_model_len"))
        or _positive_int(metadata.get("max_model_len")),
        "max_output_tokens": _positive_int(metadata.get("max_output_tokens")),
        "reasoning": _optional_bool(metadata.get("reasoning")),
        "supports_reasoning_effort": _optional_bool(
            metadata.get("supports_reasoning_effort")
        ),
    }


def _with_model_limits(normalized: dict, endpoint: RegisteredEndpoint) -> dict:
    """Add the limit fields; a non-null value the upstream already sent wins."""
    limits = _model_limits(normalized, endpoint)
    for key in _MODEL_LIMIT_FIELDS:
        if normalized.get(key) is None:
            normalized[key] = limits[key]
    return normalized


def _normalize_model_list_entry(
    entry: dict,
    endpoint: RegisteredEndpoint,
) -> tuple[dict, str]:
    """Rewrite picker ids to catalog names, tag the base catalog model and
    attach the model-limit fields (``_MODEL_LIMIT_FIELDS``)."""
    catalog_model = endpoint.model
    upstream_id = entry.get("id")
    base_id = catalog_model or upstream_id
    if not base_id:
        return entry, ""

    normalized = _with_model_limits(
        {
            **entry,
            "catalog_model": catalog_model,
        },
        endpoint,
    )
    if endpoint.variant and not _variant_is_redundant(
        str(endpoint.variant), str(base_id)
    ):
        public_id = f"{base_id}-{endpoint.variant}"
        normalized["id"] = public_id
        return normalized, public_id

    normalized["id"] = base_id
    return normalized, str(base_id)


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
        async with httpx.AsyncClient(timeout=_listing_timeout()) as client:
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


def _merge_model_entries(
    endpoints: list,
    results: list[list[dict]],
) -> dict[str, dict]:
    """Deduplicate per-endpoint model entries by dedup key.

    When an endpoint has a variant (e.g. ``pd``, ``128k``), the variant
    is appended to the model id so each deployment variant appears as a
    separate entry in the model picker.  Without this, the picker shows
    only one entry for all variants (e.g. only one ``GLM-5.2-FP8`` for
    both ``-pd`` and non-``-pd`` deployments).
    """
    merged: dict[str, dict] = {}
    for ep, entries in zip(endpoints, results):
        for entry in entries:
            if not entry.get("id") and not ep.model:
                continue
            normalized, dedup_key = _normalize_model_list_entry(entry, ep)
            if not dedup_key:
                continue
            merged.setdefault(dedup_key, normalized)
    return merged


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
    user_id, project_id = _get_caller_identity(request)

    live = [EndpointStatus.READY, EndpointStatus.DEGRADED]
    endpoints = registry.list_all(project_id=project_id, status_filter=live)
    # ADR-078: plus endpoints other projects share with this caller.
    endpoints += usable_endpoints(
        registry.list_all(status_filter=live),
        _endpoint_caller(request, user_id, project_id),
        shared_only=True,
    )

    if not endpoints:
        return {"object": "list", "data": []}

    results = await asyncio.gather(*(_fetch_endpoint_models(ep) for ep in endpoints))

    merged = _merge_model_entries(endpoints, results)

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

    endpoint = _discover_endpoint(
        registry, project_id, model, _endpoint_caller(request, user_id, project_id)
    )
    response = await _proxy_to_litellm(
        request,
        endpoint,
        "v1/embeddings",
        user_id,
        project_id,
        registry=registry,
    )
    registry.mark_used(endpoint.job_id)

    return response
