"""Public service-model badge for one site (ADR-106 amendment, 2026-11-07).

``GET /umrs/status/service-badge`` (``umrs.api.status_badge``) answers a
shields.io endpoint document saying how many of this site's service roles
are up. It is anonymous, so this module decides *only* from integers and
fixed words: no project, user, host, port, partition, digest or model name
ever reaches the output, whatever the roster or the registry hold.

- **Roles**: ``SERVICE_ROLES`` whose ``svc-<role>`` alias resolves on this
  site's roster (``provision_roster.role_alias_entries``).
- **Kept up**: some entry behind the role has a ``service`` block with
  ``replicas >= 1``, or a ``keep_alive`` block (the relay keeps it alive).
  ``service.replicas: 0`` and no block at all are *on demand* (ADR-109 D-4).
  As in the relay, a ``service`` block wins over ``keep_alive``.
- **Live**: a READY row the service tier owns. A ``service`` replica carries
  the recipe digest the relay records (``service_controller.DIGEST_KEY``);
  a ``keep_alive`` successor, and a ``service`` replica started before the
  digest was recorded, is a system-tier row (admin-only to provision) of the
  policy's project and submitter. ADR-109 D-2 co-served rows count
  through their owned primary. A user's own endpoint of the same model in
  any project never makes a role "up". DEGRADED is tracked separately.

Foundation module: no ``umrs.api`` / ``umrs.server`` imports.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .co_serve import CO_SERVED_BY_KEY
from .endpoint_registry import EndpointStatus, RegisteredEndpoint
from .endpoint_relay import KeepAlivePolicy, _non_negative, _owned, keep_alive_policies
from .provision_roster import (
    ROSTER_UNREADABLE,
    ROSTER_UNSET,
    SERVICE_ROLES,
    role_alias_entries,
    roster_state,
)
from .service_controller import DIGEST_KEY

logger = logging.getLogger(__name__)

#: The exact keys of every answer (pinned by the leak test).
BADGE_KEYS = ("schemaVersion", "label", "message", "color", "cacheSeconds")
LABEL = "models"
CACHE_SECONDS = 300
#: The only colours the badge ever shows.
COLORS = frozenset({"brightgreen", "yellow", "red", "lightgrey"})

#: Registry metadata the provision path records (``api/endpoints.py``).
_TIER_KEY = "tier"
_OWNER_KEY = "owner_id"
_SYSTEM_TIER = "system"


def badge(message: str, color: str) -> dict[str, Any]:
    """A shields endpoint document; ``color`` is one of ``COLORS``."""
    if color not in COLORS:
        raise ValueError(f"badge colour outside the fixed set: {color}")
    return {
        "schemaVersion": 1,
        "label": LABEL,
        "message": message,
        "color": color,
        "cacheSeconds": CACHE_SECONDS,
    }


NONE_CONFIGURED = badge("none configured", "lightgrey")
UNKNOWN = badge("unknown", "lightgrey")
IDLE = badge("idle", "lightgrey")
#: UMRS answers this (HTTP 503) when the registry cannot be read.
DOWN = badge("down", "red")


@dataclass(frozen=True)
class RoleState:
    """One configured role: kept up or on demand, and what of it is live."""

    kept_up: bool
    ready: bool
    degraded: bool


def keeps_up(entry: dict[str, Any]) -> bool:
    """``service`` with ``replicas >= 1`` (default 1), else a ``keep_alive`` block."""
    service = entry.get("service")
    if isinstance(service, dict):
        return _non_negative(service.get("replicas"), 1) >= 1
    return isinstance(entry.get("keep_alive"), dict)


def _relay_lineage(policy: KeepAlivePolicy, meta: dict[str, Any]) -> bool:
    """A system-tier row (admin-only to provision) of the policy's submitter."""
    if meta.get(_TIER_KEY) != _SYSTEM_TIER:
        return False
    return policy.user is None or meta.get(_OWNER_KEY) == policy.user


def _tier_marked(policy: KeepAlivePolicy, ep: RegisteredEndpoint) -> bool:
    """A service-tier row: the relay's digest (``service`` only), or the relay
    lineage. ``service`` replicas started before the job-id fix (401fa3a8,
    2026-11-04) carry no digest; the controller still owns them, so they
    count through the lineage too."""
    meta = ep.metadata or {}
    if policy.service and meta.get(DIGEST_KEY):
        return True
    return _relay_lineage(policy, meta)


def _names(ep: RegisteredEndpoint, model: str) -> bool:
    return ep.model == model or (ep.metadata or {}).get("served_model_name") == model


def _co_served_rows(
    model: str, primaries: list[RegisteredEndpoint], rows: list[RegisteredEndpoint]
) -> list[RegisteredEndpoint]:
    """ADR-109 D-2: the rows of ``model`` co-served in one of ``primaries``' jobs."""
    jobs = {ep.job_id for ep in primaries}
    co_rows = [ep for ep in rows if (ep.metadata or {}).get(CO_SERVED_BY_KEY) in jobs]
    return [ep for ep in co_rows if _names(ep, model)]


def _owned_rows(
    model: str, entry: dict[str, Any], rows: list[RegisteredEndpoint]
) -> list[RegisteredEndpoint]:
    """The rows of ``model`` the service tier of ``entry`` owns."""
    policies = keep_alive_policies([entry])
    if not policies:
        return []  # no lifecycle block (or no project): nothing is service-owned
    policy = policies[0]
    primaries = [ep for ep in rows if _owned(policy, ep) and _tier_marked(policy, ep)]
    if model == policy.model:
        return primaries
    return _co_served_rows(model, primaries, rows)


def role_state(
    sources: list[tuple[str, dict[str, Any]]], rows: list[RegisteredEndpoint]
) -> RoleState | None:
    """The state of one role from its ``(model, entry)`` sources; ``None`` if unconfigured."""
    if not sources:
        return None
    owned = [ep for model, entry in sources for ep in _owned_rows(model, entry, rows)]
    return RoleState(
        kept_up=any(keeps_up(entry) for _, entry in sources),
        ready=any(ep.status == EndpointStatus.READY for ep in owned),
        degraded=any(ep.status == EndpointStatus.DEGRADED for ep in owned),
    )


def role_states(rows: Iterable[RegisteredEndpoint]) -> list[RoleState]:
    """Every configured service role of this site's roster, in role order."""
    listed = list(rows)
    states = (
        role_state(role_alias_entries(f"svc-{role}"), listed) for role in SERVICE_ROLES
    )
    return [state for state in states if state is not None]


def _kept_up_badge(kept: list[RoleState]) -> dict[str, Any]:
    total = len(kept)
    up = sum(1 for state in kept if state.ready)
    if up == total:
        return badge(f"{up}/{total} up", "brightgreen")
    if up == 0 and not any(state.degraded for state in kept):
        return badge(f"0/{total} up", "red")
    return badge(f"{up}/{total} up", "yellow")


def badge_for(states: list[RoleState]) -> dict[str, Any]:
    """The badge for a site's configured role states."""
    if not states:
        return NONE_CONFIGURED
    kept = [state for state in states if state.kept_up]
    if kept:
        return _kept_up_badge(kept)
    up = sum(1 for state in states if state.ready)
    return badge(f"{up} up", "brightgreen") if up else IDLE


def service_badge(
    read_rows: Callable[[], Iterable[RegisteredEndpoint]],
) -> dict[str, Any]:
    """The site's badge. ``read_rows`` is the one registry read, made only
    when a readable roster is published; its errors propagate (the route
    answers ``DOWN``), a roster surprise after it reads "unknown"."""
    state = roster_state()
    if state == ROSTER_UNSET:
        return NONE_CONFIGURED
    if state == ROSTER_UNREADABLE:
        return UNKNOWN
    rows = list(read_rows())
    try:
        return badge_for(role_states(rows))
    except Exception:  # a roster surprise reads "unknown", never a 500
        logger.warning("service badge: roster evaluation failed", exc_info=True)
        return UNKNOWN
