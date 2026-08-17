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
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

COMPANION_SYSTEM_PROMPT = (
    "You are the R4 platform companion: a precise research-operations "
    "assistant. Answer directly; say what you do not know; never invent "
    "platform state, job ids, or results."
)

DEFAULT_MODEL = "rivault-gpt-oss-static"
DEFAULT_LITELLM_URL = "http://127.0.0.1:11434"

ChatFn = Callable[..., str]


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
    chat: Optional[ChatFn] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> asyncio.Task:
    """Schedule the roundtrip for one accepted attempt (fire-and-pin)."""
    task = asyncio.ensure_future(
        execute_turn(
            turns,
            attempt_id,
            chat=chat or _default_chat(),
            model=model or resolve_model(),
            base_url=base_url or resolve_base_url(),
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
