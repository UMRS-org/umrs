"""Site LiteLLM chat helper (U3 S1 item 4, ADR-117).

One public chat-completion call, its reply-text reader and the service-key
resolver, used by the agent executor, the RAG answer path, knowledge
extraction and the companion turn executor. Moved from
``knowledge.agent_executor`` (``_default_litellm_chat``) without a behaviour
change; ``api/llm.py`` uses the same key resolver.
"""

from __future__ import annotations

import os
from typing import Any


def resolve_service_key() -> str | None:
    """The site LiteLLM service key: GATEWAY_LITELLM_SERVICE_KEY, else LITELLM_MASTER_KEY."""
    return os.environ.get("GATEWAY_LITELLM_SERVICE_KEY") or os.environ.get(
        "LITELLM_MASTER_KEY"
    )


def chat_completion(
    messages: list[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    max_tokens: int | None = 700,
    timeout: float = 60.0,
    temperature: float = 0.3,
    response_format: dict[str, Any] | None = None,
    reasoning_fallback: bool = False,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """Chat via the site LiteLLM upstream (same auth as api/llm.py).

    ``temperature`` stays 0.3 for the loop and extraction callers; the RAG
    answer path passes 0.0 (ADR-093 D-4). ``max_tokens=None`` sends no
    output limit at all (the provider's own default): the RAG answer of a
    large reader (ADR-093 D-14, rikyu's kimi-k3 and glm-5.3).

    ``response_format`` (ADR-116 D-2) is sent as given, e.g.
    ``{"type": "json_object"}``; None sends no field. ``reasoning_fallback``
    returns the message's ``reasoning_content`` (or ``reasoning``) when a
    reasoning model left ``content`` empty; without it the reply text is
    ``str(content)`` as before. ``extra_body`` adds request fields as given
    (knowledge extraction sends ``chat_template_kwargs`` to switch a Qwen3
    model's thinking off, 2026-11-14); None adds nothing.
    """
    key = api_key or resolve_service_key()
    if not base_url:
        raise RuntimeError("no litellm base url configured for agent executor")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    import httpx

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if response_format is not None:
        body["response_format"] = response_format
    if extra_body:
        body.update(extra_body)
    resp = httpx.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=timeout,
        trust_env=False,
    )
    resp.raise_for_status()
    message = resp.json()["choices"][0]["message"]
    if reasoning_fallback:
        return message_text(message)
    return str(message["content"])


def message_text(message: dict[str, Any]) -> str:
    """The reply text of one chat message: ``content``, else the reasoning
    field a reasoning model filled instead (``reasoning_content`` or
    ``reasoning``), else ``""``. Never the string ``"None"``."""
    for key in ("content", "reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


__all__ = ["chat_completion", "message_text", "resolve_service_key"]
