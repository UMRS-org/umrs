"""Companion turn executor (wave T3c) — in-sidecar async roundtrip.

Execution lives inside the UMRS sidecar process by design: the spawned
venue worker (m4) holds liveness/session only and carries neither DB
credentials nor auth material (env contract), while the attempt registry,
site LiteLLM upstream, and service keys all resolve in-process here. The
agentic/tool executor venue comes later; this is the honest chat
roundtrip milestone.

Semantics:

- Called from the m5 HTTP surface (``begin_turn``) via :func:`kick` —
  one asyncio task per accepted attempt; task refs are pinned on
  ``app.state.companion_turn_tasks`` so the loop cannot GC mid-flight.
- Exactly-once is delegated to the registry: executes only attempts in
  ``in_flight`` state; commits write ``committed_response_ref`` +
  ``response_content`` (ADR-028 replay plane); failures mark `fail`.
- Stale attempts from a reloaded sidecar stay ``in_flight`` (a reaper is
  the recorded follow-up; reads are SSE-reattach safe today).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

COMPANION_SYSTEM_PROMPT = (
    "You are the R4 platform companion: a precise research-operations "
    "assistant. Answer directly; say what you do not know; never invent "
    "platform state, job ids, or results."
)

DEFAULT_MODEL = "rivault-gpt-oss-static"
DEFAULT_LITELLM_URL = "http://127.0.0.1:11434"

ChatFn = Callable[..., str]


@dataclass(frozen=True)
class ChatTarget:
    """Resolved chat upstream: real endpoint (base_url + served model name)."""

    base_url: str
    model: str


def resolve_chat_target(
    model: str, *, project_id: str = "default"
) -> ChatTarget | None:
    """Registry-first chat target resolution (T3c follow-up).

    Route through the platform's own endpoint selection — READY/DEGRADED
    endpoints on the cluster — instead of the site-LiteLLM static alias
    (absent on this deployment: T5's live-roundtrip blocker). Returns None
    on any registry failure so kick() can fall back to the static chain
    (UMRS_COMPANION_LITELLM_URL → UMRS_AGENT_LITELLM_URL → default).
    """
    try:
        from umrs.orchestrator.endpoint_registry import get_registry

        registry = get_registry()
        endpoint = registry.select_endpoint(
            project_id, model, strategy="least_connections"
        )
        if endpoint is None:
            return None
        base_url = (getattr(endpoint, "litellm_url", "") or "").rstrip("/")
        if not base_url:
            return None
        metadata = getattr(endpoint, "metadata", {}) or {}
        served = metadata.get("served_model_name") or model
        return ChatTarget(base_url=base_url, model=served)
    except Exception:
        logger.debug(
            "companion chat target: registry resolution skipped", exc_info=True
        )
        return None


def project_from_registry_key(registry_key: str) -> str:
    """Extract project from '<user>@<project>' registry key (default-safe)."""
    if "@" in registry_key:
        project = registry_key.split("@", 1)[1].strip()
        if project:
            return project
    return "default"


def _default_chat() -> ChatFn:
    """Site LiteLLM chat (same upstream/auth pattern as api/llm.py)."""
    from umrs.knowledge.agent_executor import _default_litellm_chat

    return _default_litellm_chat


def resolve_model() -> str:
    return (
        os.environ.get("UMRS_COMPANION_MODEL")
        or os.environ.get("UMRS_AGENT_MODEL")
        or DEFAULT_MODEL
    )


def resolve_base_url() -> str:
    return (
        os.environ.get("UMRS_COMPANION_LITELLM_URL")
        or os.environ.get("UMRS_AGENT_LITELLM_URL")
        or DEFAULT_LITELLM_URL
    )


async def execute_turn(
    turns: Any,
    attempt_id: str,
    *,
    chat: ChatFn,
    model: str,
    base_url: str,
    max_tokens: int = 1024,
) -> None:
    """Run one attempt roundtrip; commit content or mark failed (terminal)."""
    attempt = turns.get(attempt_id)
    if attempt is None or attempt.state != "in_flight":
        return
    messages = [
        {"role": "system", "content": COMPANION_SYSTEM_PROMPT},
        {"role": "user", "content": attempt.user_message},
    ]
    try:
        content = await asyncio.to_thread(
            chat,
            messages,
            model=model,
            base_url=base_url,
            api_key=None,
            max_tokens=max_tokens,
        )
    except Exception as exc:  # noqa: BLE001 — any upstream failure fails the attempt
        logger.warning("companion turn %s failed: %s", attempt_id, exc)
        turns.fail(attempt_id)
        return
    turns.complete(
        attempt_id,
        committed_response_ref=f"attempt:{attempt_id}",
        response_content=content,
    )


def kick(
    turns: Any,
    attempt_id: str,
    *,
    app: Any = None,
    chat: ChatFn | None = None,
    model: str | None = None,
    base_url: str | None = None,
    registry_key: str = "",
) -> asyncio.Task:
    """Schedule the roundtrip for one accepted attempt (fire-and-pin).

    Target order: (1) registry-resolved live endpoint, (2) explicit
    model/base_url args, (3) UMRS_COMPANION_* env, (4) UMRS_AGENT_* env,
    (5) static default. Order (1) is what unblocks live roundtrips on
    cluster deployments where no site LiteLLM listens locally.
    """
    resolved_model = model or resolve_model()
    resolved_base = base_url or resolve_base_url()
    if base_url is None:
        target = resolve_chat_target(
            resolved_model, project_id=project_from_registry_key(registry_key)
        )
        if target is not None:
            resolved_model, resolved_base = target.model, target.base_url
    task = asyncio.ensure_future(
        execute_turn(
            turns,
            attempt_id,
            chat=chat or _default_chat(),
            model=resolved_model,
            base_url=resolved_base,
        )
    )
    if app is not None:
        _pin_task(app, task)
    return task


def _pin_task(app: Any, task: Awaitable) -> None:
    """Hold a strong ref on the app so the task cannot be GC'd mid-flight."""
    tasks = getattr(app.state, "companion_turn_tasks", None)
    if tasks is None:
        tasks = set()
        app.state.companion_turn_tasks = tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
