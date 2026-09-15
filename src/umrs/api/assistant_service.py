"""UMRS Assistant Service (ADR-012).

OpenAI-compatible, project-aware conversational surface that runs **inside**
mini-RiVault, co-located with LiteLLM/vLLM and the RAG service.

This service implements the "Assistant API" surface described in ADR-012:

- It exposes an OpenAI-compatible ``POST /v1/chat/completions`` endpoint so that
  any OpenAI-format client (OpenWebUI, ``r4 chat``, the TUI, IDE integrations)
  can consume it unchanged.
- It accepts a **per-request** ``project_id`` (never launch-bound) and, when
  present, augments the conversation with retrieved project knowledge by calling
  the co-located RAG service. RAG therefore stays *inside* mini-RiVault, per
  INV-09 / ADR-006.
- It **forwards** the (optionally augmented) request to the upstream
  OpenAI-compatible endpoint (LiteLLM), which remains a swappable implementation
  of the OpenAI contract.
- It exposes ``GET /manifest`` for capability discovery.

Boundary notes:
- This is the *augmented* surface. The raw OpenAI passthrough
  (``/gw/llm/v1/*`` → LiteLLM) remains a first-class public surface; the two are
  parallel products, not successive generations.
- The service holds no durable state. Persistent conversation/memory is owned by
  UMRS; this service consumes those APIs and never duplicates them.
- Augmentation is best-effort and fails open to a plain passthrough: any RAG
  error results in the original messages being forwarded unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

__version__ = "0.1.0"

# Request keys that are Assistant-API extensions and must be stripped before the
# payload is forwarded to the upstream OpenAI-compatible endpoint.
_ASSISTANT_ONLY_KEYS = ("project_id", "rag", "rag_top_k", "rag_threshold")


@dataclass(frozen=True)
class AssistantConfig:
    """Runtime configuration resolved from the environment.

    Resolved per request (not cached) so tests and redeploys observe env changes.
    """

    upstream_base_url: str
    upstream_api_key: str
    rag_url: str
    default_model: str
    request_timeout: float


def _resolve_config() -> AssistantConfig:
    """Resolve Assistant Service configuration from the environment."""
    upstream_base_url = os.getenv(
        "ASSISTANT_UPSTREAM_BASE_URL", "http://litellm:4000/v1"
    ).rstrip("/")

    upstream_api_key = ""
    for env_name in (
        "ASSISTANT_UPSTREAM_API_KEY",
        "LITELLM_MASTER_KEY",
        "VLLM_API_KEY",
    ):
        value = os.getenv(env_name)
        if value:
            upstream_api_key = value
            break

    rag_url = os.getenv("ASSISTANT_RAG_URL", "http://localhost:8081").rstrip("/")
    default_model = os.getenv("ASSISTANT_DEFAULT_MODEL", "rivault-gpt-oss-static")
    request_timeout = float(os.getenv("ASSISTANT_REQUEST_TIMEOUT", "120"))

    return AssistantConfig(
        upstream_base_url=upstream_base_url,
        upstream_api_key=upstream_api_key,
        rag_url=rag_url,
        default_model=default_model,
        request_timeout=request_timeout,
    )


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# =============================================================================
# Pure helpers (unit-testable, no network)
# =============================================================================


def _message_text(content: Any) -> str | None:
    """Normalize an OpenAI message ``content`` (string or content parts) to text."""
    if isinstance(content, str):
        return content if content.strip() else None
    if isinstance(content, list):
        text = " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
        return text or None
    return None


def extract_last_user_query(messages: list[dict[str, Any]]) -> str | None:
    """Return the content of the most recent ``user`` message, or None."""
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = _message_text(msg.get("content"))
        if text:
            return text
    return None


def _fit_sources_to_budget(
    sources: list[dict[str, Any]], max_context_tokens: int
) -> list[str]:
    """Render sources to numbered entries that fit within a token budget."""
    chars_per_token = 4  # rough approximation, mirrors r4 SDK inject_context
    parts: list[str] = []
    used_tokens = 0
    for i, source in enumerate(sources, 1):
        content = str(source.get("content", "")).strip()
        if not content:
            continue
        entry = f"[{i}] {content}"
        entry_tokens = len(entry) // chars_per_token
        if used_tokens + entry_tokens > max_context_tokens:
            remaining = max_context_tokens - used_tokens
            if remaining > 50:
                parts.append(entry[: remaining * chars_per_token] + "...")
            break
        parts.append(entry)
        used_tokens += entry_tokens
    return parts


def _source_label(source: dict[str, Any], index: int) -> str:
    metadata = source.get("metadata") or {}
    return str(metadata.get("source") or metadata.get("s3_object") or f"doc-{index}")


def build_context_block(
    sources: list[dict[str, Any]],
    *,
    max_context_tokens: int = 2000,
    context_prefix: str = "\n\n---\nRelevant context:\n",
    context_suffix: str = "\n---\n",
) -> str:
    """Assemble a context block from retrieved sources within a token budget.

    ``sources`` items are expected to carry a ``content`` field and an optional
    ``metadata`` dict. Returns an empty string when there is nothing to inject.
    """
    if not sources:
        return ""

    parts = _fit_sources_to_budget(sources, max_context_tokens)
    if not parts:
        return ""

    labels = [
        _source_label(source, i) for i, source in enumerate(sources[: len(parts)], 1)
    ]
    return (
        context_prefix
        + "\n".join(parts)
        + context_suffix
        + "Sources: "
        + ", ".join(labels)
    )


def inject_context_into_messages(
    messages: list[dict[str, Any]], context_block: str
) -> list[dict[str, Any]]:
    """Merge a context block into the system message (or prepend a new one)."""
    if not context_block:
        return messages

    merged: list[dict[str, Any]] = []
    system_found = False
    for msg in messages:
        if msg.get("role") == "system" and not system_found:
            merged.append(
                {
                    "role": "system",
                    "content": str(msg.get("content", "")) + context_block,
                }
            )
            system_found = True
        else:
            merged.append(msg)

    if not system_found:
        merged.insert(
            0,
            {
                "role": "system",
                "content": f"You are a helpful assistant.{context_block}",
            },
        )
    return merged


# =============================================================================
# Network helpers (thin, mockable)
# =============================================================================


def retrieve_context(
    config: AssistantConfig,
    query: str,
    *,
    project_id: str | None,
    top_k: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Call the co-located RAG service and return matching sources.

    Best-effort: returns an empty list on any error so the caller can fall back
    to a plain passthrough. ``project_id`` is mapped to the RAG ``xp_name`` scope
    (best-effort in increment 1; project→RAG scoping is refined later).
    """
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if project_id:
        payload["xp_name"] = project_id

    try:
        with httpx.Client(timeout=10.0, trust_env=False) as client:
            resp = client.post(f"{config.rag_url}/rag/search", json=payload)
        if resp.status_code != 200:
            return []
        results = resp.json().get("results", [])
    except Exception:
        return []

    sources: list[dict[str, Any]] = []
    for item in results:
        if float(item.get("score", 0.0)) < threshold:
            continue
        sources.append(
            {"content": item.get("content", ""), "metadata": item.get("metadata", {})}
        )
    return sources


def augment_messages(
    config: AssistantConfig,
    messages: list[dict[str, Any]],
    *,
    project_id: str | None,
    top_k: int,
    threshold: float,
) -> list[dict[str, Any]]:
    """Inject retrieved project context into ``messages`` (fails open)."""
    query = extract_last_user_query(messages)
    if not query:
        return messages
    sources = retrieve_context(
        config, query, project_id=project_id, top_k=top_k, threshold=threshold
    )
    return inject_context_into_messages(messages, build_context_block(sources))


# =============================================================================
# FastAPI application
# =============================================================================

app = FastAPI(title="UMRS Assistant Service", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "assistant"}


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    """Advertise this endpoint's conversational capabilities (ADR-012).

    The manifest exposes *capabilities* (boolean flags) rather than
    implementation details.  Clients discover what the Assistant Service
    can do without needing to know about LiteLLM, vLLM, or RAG internals.
    """
    config = _resolve_config()
    return {
        "service": "assistant",
        "version": __version__,
        "surfaces": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "manifest": "/manifest",
            "health": "/health",
        },
        "capabilities": {
            "assistant": True,
            "rag": True,
            "project_aware": True,
            "streaming": True,
            "vision": False,
            "mcp": False,
            "tools": False,
        },
        "default_model": config.default_model,
    }


@app.get("/v1/models")
def list_models() -> JSONResponse:
    """Proxy the upstream model list; fall back to the default model name."""
    config = _resolve_config()
    try:
        with httpx.Client(timeout=5.0, trust_env=False) as client:
            resp = client.get(
                f"{config.upstream_base_url}/models",
                headers=_auth_headers(config.upstream_api_key),
            )
        if resp.status_code == 200:
            return JSONResponse(content=resp.json())
    except Exception:
        pass
    return JSONResponse(
        content={
            "object": "list",
            "data": [{"id": config.default_model, "object": "model"}],
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """OpenAI-compatible chat completion with optional project-aware RAG.

    Accepts the standard OpenAI body plus Assistant-API extensions:
    ``project_id``, ``rag`` (default true), ``rag_top_k``, ``rag_threshold``.
    """
    config = _resolve_config()
    body: dict[str, Any] = await request.json()

    project_id = body.get("project_id")
    rag_enabled = body.get("rag", True)
    top_k = int(body.get("rag_top_k", 3))
    threshold = float(body.get("rag_threshold", 0.7))

    messages = body.get("messages") or []
    if rag_enabled and isinstance(messages, list) and messages:
        body["messages"] = augment_messages(
            config, messages, project_id=project_id, top_k=top_k, threshold=threshold
        )

    # Strip Assistant-only extensions before forwarding to the OpenAI upstream.
    forward_body = {k: v for k, v in body.items() if k not in _ASSISTANT_ONLY_KEYS}
    forward_body.setdefault("model", config.default_model)

    url = f"{config.upstream_base_url}/chat/completions"
    headers = _auth_headers(config.upstream_api_key)

    if forward_body.get("stream"):
        return StreamingResponse(
            _stream_upstream(config, url, forward_body, headers),
            media_type="text/event-stream",
        )

    try:
        async with httpx.AsyncClient(
            timeout=config.request_timeout, trust_env=False
        ) as client:
            resp = await client.post(url, json=forward_body, headers=headers)
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"error": {"message": f"upstream request failed: {exc}"}},
        )
    return JSONResponse(status_code=resp.status_code, content=resp.json())


async def _stream_upstream(
    config: AssistantConfig,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
):
    """Proxy a streaming (SSE) chat completion from the upstream endpoint."""
    async with httpx.AsyncClient(timeout=None, trust_env=False) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            async for chunk in resp.aiter_raw():
                if chunk:
                    yield chunk


def main() -> None:
    import uvicorn

    bind_host = os.getenv("ASSISTANT_BIND_HOST", "127.0.0.1")
    bind_port = int(os.getenv("ASSISTANT_PORT", "8090"))
    uvicorn.run(
        "umrs.api.assistant_service:app", host=bind_host, port=bind_port, reload=False
    )


__all__ = [
    "app",
    "main",
    "AssistantConfig",
    "extract_last_user_query",
    "build_context_block",
    "inject_context_into_messages",
    "retrieve_context",
    "augment_messages",
]
