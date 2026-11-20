"""Check a session's web choice against the site's policy (U3 S1 item 5).

The loop start route checks its runs' web choice before anything is reserved
(ADR-099 D-2). It reached through the frontend venue service for this; the
check needs only the policy store, the site, the owner and project, the
policy environment and the legacy default, so it lives in core. The venue
service delegates here. Unreadable rules fail closed with
``PolicyError("unavailable")``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from .model import PolicyError, WebChoice
from .resolve import resolve_session_policy, web_choice_required
from .store import effective_limits_for

logger = logging.getLogger(__name__)


def web_choice(web_policy: str, domains: list[str] | None = None) -> WebChoice:
    """A request's ``web_policy`` + ``web_allow_domains`` as a :class:`WebChoice`."""
    return WebChoice(
        mode=(web_policy or "").strip(),
        domains=tuple(d.strip() for d in (domains or []) if d and d.strip()),
    )


def limits_for(
    store: Any, *, site: str, owner: str, project: str, env: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Effective limits of a session on ``site``; unreadable rules fail closed."""
    try:
        return effective_limits_for(store, venue=site, user_id=owner, project_id=project, env=env)
    except Exception as exc:
        logger.warning("session policy rules unreadable", exc_info=True)
        raise PolicyError(
            "unavailable", f"session policy rules unreadable: {type(exc).__name__}"
        ) from exc


def check_web_choice(
    store: Any,
    *,
    site: str,
    owner: str,
    project: str,
    web_policy: str,
    web_allow_domains: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    legacy_default: str = "",
) -> dict[str, Any]:
    """The policy a start with this web choice would get, or the typed
    :class:`PolicyError` it would raise; nothing is reserved."""
    return resolve_session_policy(
        limits_for(store, site=site, owner=owner, project=project, env=env),
        web_choice(web_policy, web_allow_domains),
        require_choice=web_choice_required(env),
        legacy_default=legacy_default,
        now=datetime.now(timezone.utc),
    )


__all__ = ["check_web_choice", "limits_for", "web_choice"]
