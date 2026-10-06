"""Upstream timeout limits for the OpenAI-compatible proxies (W5 / U1).

Shared by the LLM proxy (``api/llm.py``, env prefix ``LLM_PROXY``) and the
assistant chat proxy (``api/chat.py``, env prefix ``CHAT_PROXY``). Not a
router module: it only resolves numbers from the environment.

Three separate limits, all in seconds (2026-10-05):

* **connect** (``<PREFIX>_CONNECT_TIMEOUT``, default 10) — opening the TCP
  connection to the upstream. Unchanged by W5.
* **idle** (``<PREFIX>_IDLE_TIMEOUT``, default 600) — the longest silence
  allowed between two upstream reads of a *streaming* reply. httpx applies
  its read timeout per read, so a stream that keeps producing chunks never
  hits it, however long the whole turn runs; the first chunk of a large
  prompt may take minutes, which the default covers.
* **total** (``<PREFIX>_TOTAL_TIMEOUT``, default 7200) — the overall deadline
  for one *non-streaming* request. A non-streaming reply arrives only after
  the model has finished (a reasoning turn has been measured at 93 min), so
  the per-read limit cannot be the guard there; the proxies enforce this
  value as an asyncio deadline around the whole request instead.

Deprecated alias: ``<PREFIX>_READ_TIMEOUT`` is the pre-2026-10-05 name of the
single read limit (default 120 s, applied to streaming and non-streaming
alike). It is still honoured as the **idle** value when
``<PREFIX>_IDLE_TIMEOUT`` is unset, and a warning is logged; it no longer
caps non-streaming requests (those follow the total limit). Remove the alias
once no deployment sets it (none in the repo did on 2026-10-05).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT_SEC = 10.0
DEFAULT_IDLE_TIMEOUT_SEC = 600.0
DEFAULT_TOTAL_TIMEOUT_SEC = 7200.0


@dataclass(frozen=True)
class ProxyTimeouts:
    """Resolved upstream limits for one proxy (seconds, all > 0)."""

    connect: float  # TCP connect to the upstream
    idle: float  # max silence between two reads of a streaming reply
    total: float  # overall deadline of one non-streaming request


def _seconds(name: str, raw: str) -> float:
    """Parse one positive seconds value; raise with the variable name on error."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number of seconds, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0 seconds, got {raw!r}")
    return value


def _env_seconds(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    alias: str | None = None,
) -> float:
    """Value of *name*, else of the deprecated *alias*, else *default*."""
    raw = env.get(name)
    if raw not in (None, ""):
        return _seconds(name, str(raw))
    alias_raw = env.get(alias) if alias else None
    if alias and alias_raw not in (None, ""):
        logger.warning(
            "%s is deprecated (2026-10-05): it now sets only the idle limit "
            "between streamed chunks; set %s instead",
            alias,
            name,
        )
        return _seconds(alias, str(alias_raw))
    return default


def resolve_proxy_timeouts(
    prefix: str, env: Mapping[str, str] | None = None
) -> ProxyTimeouts:
    """Resolve connect / idle / total limits for the proxy named by *prefix*.

    Args:
        prefix: env prefix, ``"LLM_PROXY"`` or ``"CHAT_PROXY"``.
        env: mapping to read (defaults to ``os.environ``; tests pass a dict).

    Returns:
        The resolved :class:`ProxyTimeouts`.

    Raises:
        ValueError: a set variable is not a positive number of seconds.
    """
    source = os.environ if env is None else env
    return ProxyTimeouts(
        connect=_env_seconds(
            source, f"{prefix}_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT_SEC
        ),
        idle=_env_seconds(
            source,
            f"{prefix}_IDLE_TIMEOUT",
            DEFAULT_IDLE_TIMEOUT_SEC,
            alias=f"{prefix}_READ_TIMEOUT",
        ),
        total=_env_seconds(
            source, f"{prefix}_TOTAL_TIMEOUT", DEFAULT_TOTAL_TIMEOUT_SEC
        ),
    )
