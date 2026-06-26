"""Best-effort generative perception for catalog ingestion.

Reuses already-deployed inference endpoints (no new perception infrastructure):

- **Images** are captioned by a vision-capable chat model via the OpenAI
  ``image_url`` content format.
- **Text / documents** get an LLM summary of their *extracted text*.

Every function here is best-effort: on a missing endpoint, disabled flag, or any
error it returns ``None`` so ingestion degrades to deterministic metadata and
never fails. Deterministic extraction (columns, tags, raw text) remains the
ground truth; perception only augments it.
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_VISION_CAPTION = "vision_caption"
_LLM_SUMMARY = "llm_summary"

_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MAX_IMAGE_BYTES = 4 * 1024 * 1024
_DEFAULT_SUMMARY_MAX_CHARS = 12_000
_DEFAULT_SUMMARY_LIMIT = 500

_CAPTION_PROMPT = (
    "Describe this image for a scientific data catalog. State what it depicts, "
    "any visible text, labels, axes, or measurements, and notable features. "
    "Be concise and factual; do not speculate."
)
_SUMMARY_PROMPT = (
    "Summarize the following file content for a scientific data catalog so a "
    "researcher can decide whether to open it. Be concise and factual; do not "
    "invent details not present in the text."
)


@dataclass
class PerceptionResult:
    """Outcome of a perception call."""

    text: str
    model: str
    kind: str
    tags: list[str] = field(default_factory=list)


def perception_enabled() -> bool:
    """True when text/LLM perception is enabled (default on).

    Set ``UMRS_DATA_CATALOG_ENABLE_PERCEPTION=false`` to skip all generative
    perception (text summaries and image captions).
    """
    return os.getenv("UMRS_DATA_CATALOG_ENABLE_PERCEPTION", "true").lower() == "true"


def image_perception_enabled() -> bool:
    """True when image captioning is enabled (default off).

    Image captioning is the most expensive perception path. Text LLM summaries
    stay on when ``perception_enabled()``; enable image captioning explicitly
    via ``UMRS_DATA_CATALOG_ENABLE_IMAGE_PERCEPTION=true``.
    """
    return os.getenv("UMRS_DATA_CATALOG_ENABLE_IMAGE_PERCEPTION", "false").lower() == "true"


def run_perception(
    *,
    project_id: str,
    filename: str,
    modality: str,
    extracted_text: str,
    image_bytes: bytes | None = None,
    user_id: str | None = None,
) -> PerceptionResult | None:
    """
    Dispatch to the right perception path for a modality. Best-effort.

    Returns ``None`` when perception is disabled, unavailable, or fails.
    """
    if not perception_enabled():
        return None
    billing_user = user_id or project_id
    try:
        if modality == "image":
            if not image_bytes or not image_perception_enabled():
                return None
            return caption_image(
                project_id,
                filename=filename,
                image_bytes=image_bytes,
                user_id=billing_user,
            )
        if modality in ("text", "structured", "document") and extracted_text.strip():
            return summarize_text(
                project_id,
                filename=filename,
                extracted_text=extracted_text,
                user_id=billing_user,
            )
    except Exception:
        logger.exception("perception failed for %s (modality=%s)", filename, modality)
    return None


def caption_image(
    project_id: str,
    *,
    filename: str,
    image_bytes: bytes,
    user_id: str,
) -> PerceptionResult | None:
    """Caption an image with a vision-capable chat endpoint."""
    max_bytes = _int_env("UMRS_DATA_CATALOG_PERCEPTION_MAX_IMAGE_BYTES", _DEFAULT_MAX_IMAGE_BYTES)
    if len(image_bytes) > max_bytes:
        logger.debug("image %s exceeds perception cap (%d bytes)", filename, len(image_bytes))
        return None

    endpoint = _find_vision_endpoint(project_id)
    if endpoint is None:
        return None

    model = _served_model_name(endpoint)
    data_uri = _data_uri(filename, image_bytes)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": 512,
        "temperature": 0.2,
    }
    text = _chat_completion(
        endpoint,
        payload,
        project_id=project_id,
        user_id=user_id,
        perception_kind=_VISION_CAPTION,
    )
    if not text:
        return None
    return PerceptionResult(text=text.strip(), model=model, kind=_VISION_CAPTION)


def summarize_text(
    project_id: str,
    *,
    filename: str,
    extracted_text: str,
    user_id: str,
) -> PerceptionResult | None:
    """Summarize already-extracted text with a chat endpoint."""
    max_chars = _int_env("UMRS_DATA_CATALOG_SUMMARY_MAX_CHARS", _DEFAULT_SUMMARY_MAX_CHARS)
    snippet = extracted_text[:max_chars]

    endpoint = _find_text_endpoint(project_id)
    if endpoint is None:
        return None

    model = _served_model_name(endpoint)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SUMMARY_PROMPT},
            {"role": "user", "content": f"Filename: {filename}\n\n{snippet}"},
        ],
        "max_tokens": 400,
        "temperature": 0.2,
    }
    text = _chat_completion(
        endpoint,
        payload,
        project_id=project_id,
        user_id=user_id,
        perception_kind=_LLM_SUMMARY,
    )
    if not text:
        return None
    return PerceptionResult(text=text.strip(), model=model, kind=_LLM_SUMMARY)


# ---------------------------------------------------------------------------
# Endpoint discovery (reuses the in-process registry, no new infra)
# ---------------------------------------------------------------------------


def _find_vision_endpoint(project_id: str) -> Any | None:
    """Find a usable vision-capable chat endpoint (vLLM OpenAI, not BentoML)."""
    configured = os.getenv("UMRS_DATA_CATALOG_VISION_MODEL")
    for ep in _usable_endpoints(project_id):
        if _is_bentoml(ep):
            continue
        if configured and ep.model != configured:
            continue
        if _is_vision_capable(ep):
            return ep
    return None


def _find_text_endpoint(project_id: str) -> Any | None:
    """Find a usable chat endpoint for text summarization."""
    configured = os.getenv("UMRS_DATA_CATALOG_SUMMARY_MODEL")
    fallback = None
    for ep in _usable_endpoints(project_id):
        if _is_bentoml(ep) or ep.metadata.get("model_type") == "embedding":
            continue
        if configured and ep.model == configured:
            return ep
        if fallback is None:
            fallback = ep
    return None if configured else fallback


def _usable_endpoints(project_id: str) -> list[Any]:
    try:
        from umrs.orchestrator.endpoint_registry import EndpointStatus, get_registry

        registry = get_registry()
        return registry.list_all(
            project_id=project_id,
            status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
        )
    except Exception:
        logger.debug("endpoint registry unavailable for perception", exc_info=True)
        return []


def _is_vision_capable(endpoint: Any) -> bool:
    metadata = endpoint.metadata or {}
    if metadata.get("model_type") == "vision":
        return True
    capabilities = metadata.get("capabilities")
    return bool(isinstance(capabilities, dict) and capabilities.get("supports_vision"))


def _is_bentoml(endpoint: Any) -> bool:
    metadata = endpoint.metadata or {}
    return (
        metadata.get("serving_backend") == "bentoml"
        or metadata.get("service_kind") == "bentoml_vision"
    )


def _served_model_name(endpoint: Any) -> str:
    return str((endpoint.metadata or {}).get("served_model_name") or endpoint.model)


# ---------------------------------------------------------------------------
# Chat completion (synchronous; ingestion runs in a sync backend)
# ---------------------------------------------------------------------------


def _chat_completion(
    endpoint: Any,
    payload: dict[str, Any],
    *,
    project_id: str,
    user_id: str,
    perception_kind: str,
) -> str | None:
    import httpx
    import uuid

    from umrs.api.llm import get_usage_db_pool
    from umrs.api.llm_usage import ProxyTimer
    from umrs.server.correlation import get_request_id

    url = f"{endpoint.litellm_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    service_key = os.environ.get("GATEWAY_LITELLM_SERVICE_KEY") or os.environ.get(
        "LITELLM_MASTER_KEY"
    )
    if service_key:
        headers["Authorization"] = f"Bearer {service_key}"

    timeout = _float_env("UMRS_DATA_CATALOG_PERCEPTION_TIMEOUT", _DEFAULT_TIMEOUT)
    timer = ProxyTimer()
    status_code = 0
    body = b""
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        status_code = response.status_code
        body = response.content
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.warning("perception chat call to %s failed", url, exc_info=True)
        _record_perception_usage(
            pool=get_usage_db_pool(),
            endpoint=endpoint,
            project_id=project_id,
            user_id=user_id,
            model=str(payload.get("model", "")),
            body=body,
            status_code=status_code or 500,
            duration_ms=timer.elapsed_ms(),
            perception_kind=perception_kind,
            request_id=get_request_id() or uuid.uuid4().hex,
        )
        return None

    _record_perception_usage(
        pool=get_usage_db_pool(),
        endpoint=endpoint,
        project_id=project_id,
        user_id=user_id,
        model=str(payload.get("model", "")),
        body=body,
        status_code=status_code,
        duration_ms=timer.elapsed_ms(),
        perception_kind=perception_kind,
        request_id=get_request_id() or uuid.uuid4().hex,
    )

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.debug("unexpected chat response shape from %s", url)
        return None


def _record_perception_usage(
    *,
    pool: Any,
    endpoint: Any,
    project_id: str,
    user_id: str,
    model: str,
    body: bytes,
    status_code: int,
    duration_ms: float,
    perception_kind: str,
    request_id: str,
) -> None:
    """Persist token usage for a perception call (same path as LLM proxy)."""
    if pool is None:
        return

    from umrs.api.llm_usage import UsageRecord, extract_usage_oneshot, fill_usage, record_usage

    metadata = endpoint.metadata or {}
    record = UsageRecord(
        request_id=request_id,
        user_id=user_id,
        project_id=project_id,
        model=model,
        served_model=str(metadata.get("served_model_name") or endpoint.model),
        endpoint_job_id=str(getattr(endpoint, "job_id", "") or ""),
        endpoint_tier=str(metadata.get("tier") or metadata.get("owner_type") or ""),
        duration_ms=duration_ms,
        status_code=status_code,
        extra={"source": "data_catalog_perception", "kind": perception_kind},
    )
    usage = extract_usage_oneshot(body)
    if usage:
        fill_usage(record, usage)
    record_usage(pool, record)


def _data_uri(filename: str, image_bytes: bytes) -> str:
    mime, _ = mimetypes.guess_type(filename)
    mime = mime or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.getenv(name, str(default))))
    except ValueError:
        return default
