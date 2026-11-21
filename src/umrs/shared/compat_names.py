"""Neutral UMRS names with R4 fallbacks (U3 S1 item 11, ADR-117).

Core reads environment variables and request headers that carry R4 names
(``R4_GATEWAY_URL``, ``x-r4-actor``, ...). U3 S6 moves core to neutral
``UMRS_*`` / ``x-umrs-*`` names; until every sender switches, both are read:

- an environment value: ``UMRS_<NAME>`` first, then ``R4_<NAME>``;
- a header value: ``x-umrs-<name>`` first, then ``x-r4-<name>``.

The actor and scope headers mark agent and work-level requests, which get
**fewer** rights. They fail closed: a request counts as an agent request when
*either* header names a non-person actor, and as scoped when *either* scope
header is non-empty. A sender cannot shed a restriction by also sending the
other name.
"""

from __future__ import annotations

import os
from collections.abc import Mapping


def env(name: str, default: str = "", environ: Mapping[str, str] | None = None) -> str:
    """``UMRS_<name>``, else ``R4_<name>``, else ``default`` (empty values skipped)."""
    source = os.environ if environ is None else environ
    for key in (f"UMRS_{name}", f"R4_{name}"):
        value = source.get(key)
        if value:
            return value
    return default


def _header_values(headers: Mapping[str, str], name: str) -> list[str]:
    lowered = {str(k).lower(): v for k, v in headers.items()}
    return [
        str(lowered.get(f"{prefix}-{name}") or "").strip()
        for prefix in ("x-umrs", "x-r4")
    ]


def header_values(headers: Mapping[str, str], name: str) -> list[str]:
    """Both names' values, ``x-umrs-<name>`` first (blank for an absent one):
    for callers that must see a conflict between the two (U3 S6a)."""
    return _header_values(headers, name)


def header(headers: Mapping[str, str], name: str, default: str = "") -> str:
    """``x-umrs-<name>``, else ``x-r4-<name>``, else ``default`` (blank values skipped)."""
    for value in _header_values(headers, name):
        if value:
            return value
    return default


def is_agent_request(headers: Mapping[str, str]) -> bool:
    """Fail closed: either actor header that is not the person makes it an agent.

    Uses the one actor rule (:func:`umrs.knowledge.actors.normalize_actor`:
    only an absent, empty or ``user`` value is the person).
    """
    from umrs.knowledge.actors import AGENT_ACTOR, normalize_actor

    return any(
        normalize_actor(v) == AGENT_ACTOR for v in _header_values(headers, "actor")
    )


def is_scoped_request(headers: Mapping[str, str]) -> bool:
    """Fail closed: either scope header being non-empty makes the request scoped."""
    return any(_header_values(headers, "scope"))


__all__ = ["env", "header", "header_values", "is_agent_request", "is_scoped_request"]
