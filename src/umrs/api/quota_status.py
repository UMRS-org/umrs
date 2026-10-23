"""Quota status read model (ADR-051 D-2/D-3).

One honest summary route: the verified caller sees their own two resource
poles — CPU-process units (live agent jobs) and disk bytes (catalog-attributed
unique content). Read-only, no accounting decisions here; caps come straight
from the same env knobs the enforcement edges read, so the number shown is
exactly the number enforced.

ADR-099 (RS-1) adds two remote-session poles from the caller's effective
session policy at this site: ``sessions`` (live remote sessions against
``sessions.concurrent_user``) and ``tokens_today`` (the inference proxy's
tokens today against ``budget.tokens_per_day_user``). On these two poles
``limit`` is ``null`` for no limit, as in the policy (the two older poles
keep their ``0`` = uncapped). A pole whose numbers cannot be read is
``null``, never invented.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ._scope_identity import get_caller_scope
from .agent_quota import agent_max_processes, count_active_agent_units
from .disk_ledger import DiskLedger, disk_bytes_max
from .typed_errors import PlatformError, raise_typed


class PoleRow(BaseModel):
    current: int = Field(..., description="Live usage right now")
    limit: int = Field(..., description="Resolved site cap (0 = uncapped)")


class LimitPoleRow(BaseModel):
    """A session-policy pole (ADR-099): ``limit`` null = no limit."""

    current: int = Field(..., description="Live usage right now")
    limit: int | None = Field(
        None, description="The effective session-policy limit (null = no limit)"
    )


class QuotaUnitsResponse(BaseModel):
    user_id: str
    cpu_process: PoleRow
    disk_bytes: PoleRow
    sessions: LimitPoleRow | None = Field(
        None, description="Live remote sessions at this site (ADR-099 D-6)"
    )
    tokens_today: LimitPoleRow | None = Field(
        None, description="Inference-proxy tokens today at this site (ADR-099 D-5)"
    )


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/umrs/quota", tags=["quota"])


def _pool(request: Request) -> Any:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.quota("unavailable", "database pool unavailable"))
    return pool


@router.get("/units", response_model=QuotaUnitsResponse)
def quota_units(request: Request) -> QuotaUnitsResponse:
    """Per-user gauge for both ADR-051 caps (caller-attributed, no fabrication).

    # auth: verified
    # Caller-attributed; anonymous refused (see body).

    The route requires a verified identity: in strict mode the shared scope
    layer answers anonymous callers with 401 before this body runs; in mvp
    (dev) mode an anonymous scope arrives with no user — refuse it with the
    same canonical code the other verified lanes emit. An anonymous gauge
    would be fabricated data.
    """
    caller = get_caller_scope(request)
    if not caller.user_id:
        raise_typed(
            PlatformError.scopeidentity(
                "unauthorized", "verified identity required for quota summary"
            )
        )
    pool = _pool(request)
    cpu_current = count_active_agent_units(pool, user_id=caller.user_id)
    conn = pool.getconn()
    try:
        disk_usage = DiskLedger(conn).usage(caller.user_id)
    finally:
        pool.putconn(conn)
    sessions, tokens_today = _remote_session_poles(pool, caller.user_id)
    return QuotaUnitsResponse(
        user_id=caller.user_id,
        cpu_process=PoleRow(current=cpu_current, limit=agent_max_processes()),
        disk_bytes=PoleRow(current=disk_usage, limit=disk_bytes_max()),
        sessions=sessions,
        tokens_today=tokens_today,
    )


def _remote_session_poles(
    pool: Any, user_id: str
) -> tuple[LimitPoleRow | None, LimitPoleRow | None]:
    """(sessions, tokens_today) from the caller's effective session policy."""
    from umrs.jobs.store import configured_site_name
    from umrs.quotas.enforcer import policy_limits

    venue = configured_site_name()
    try:
        limits = policy_limits(pool, venue=venue, user_id=user_id)
    except Exception:  # the poles stay null, never invented
        logger.warning("session policy unreadable for quota units", exc_info=True)
        return None, None
    return (
        _sessions_pole(pool, venue, user_id, limits),
        _tokens_pole(pool, venue, user_id, limits),
    )


def _sessions_pole(
    pool: Any, venue: str, user_id: str, limits: dict[str, Any]
) -> LimitPoleRow | None:
    from umrs.quotas.session_cap import count_live_sessions, session_limits

    try:
        current = count_live_sessions(pool, venue=venue, user_id=user_id)
    except Exception:
        logger.warning("live sessions unreadable for quota units", exc_info=True)
        return None
    return LimitPoleRow(
        current=current, limit=session_limits(limits)["concurrent_user"]
    )


def _tokens_pole(
    pool: Any, venue: str, user_id: str, limits: dict[str, Any]
) -> LimitPoleRow | None:
    from umrs.quotas.enforcer import day_window, day_zone, sum_tokens
    from umrs.session_policy import limit_value

    zone = day_zone(os.environ.get("UMRS_QUOTA_DAY_TZ", "UTC"))
    start, _end = day_window(time.time(), zone)
    try:
        current = sum_tokens(
            pool, venue=venue, scope="user", subject=user_id, since=start
        )
    except Exception:
        logger.warning("token usage unreadable for quota units", exc_info=True)
        return None
    limit = ((limits or {}).get("budget") or {}).get("tokens_per_day_user")
    return LimitPoleRow(current=current, limit=limit_value(limit))


__all__ = ["router"]
