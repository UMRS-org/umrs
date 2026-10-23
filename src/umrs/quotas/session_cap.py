"""Concurrent remote sessions per user, and optionally per project (ADR-099 D-6).

At venue start (chat sessions and plan-run workers alike) UMRS counts the
user's live ``frontend_session`` jobs at this venue (``planned``, ``queued``,
``running``) and refuses when the count is >= ``sessions.concurrent_user``
of the start's resolved policy (site default 12, from
``UMRS_SESSION_CONCURRENT_PER_USER``, parsed by ``session_policy.site_defaults``
only), and the project's likewise when
``sessions.concurrent_project`` is set: ``quota.concurrent_sessions_exceeded``
(429).

Only remote sessions count: the cap protects shared server capacity, and a
local session is neither on the server nor observable by it. Two starts
racing can overshoot by one; accepted (ADR-099 D-6).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from umrs.api.agent_quota import LIVE_JOB_STATUSES
from umrs.api.typed_errors import PlatformError

if TYPE_CHECKING:  # lane P's start hook (umrs.venues.start_checks)
    from umrs.venues.start_checks import StartCheck

SESSION_JOB_TYPE = "frontend_session"

_SCOPES = (
    ("user", "concurrent_user", "owner", "user_id"),
    ("project", "concurrent_project", "project", "project_id"),
)


def session_limits(policy: Mapping[str, Any] | None) -> dict[str, int | None]:
    """``sessions`` of a resolved policy; a missing key is the site default.

    The site default and every value are read the one way the policy reads
    them (``session_policy``: ``site_defaults`` parses
    ``UMRS_SESSION_CONCURRENT_PER_USER``, ``limit_value`` reads a count:
    null = no limit, a positive integer = the cap)."""
    from umrs.session_policy import limit_value, site_defaults

    sessions = dict((policy or {}).get("sessions") or {})
    if "concurrent_user" not in sessions:
        sessions["concurrent_user"] = site_defaults()["sessions"]["concurrent_user"]
    return {
        "concurrent_user": limit_value(sessions.get("concurrent_user")),
        "concurrent_project": limit_value(sessions.get("concurrent_project")),
    }


def count_live_sessions(
    pool: Any, *, venue: str, user_id: str = "", project_id: str = ""
) -> int:
    """Live ``frontend_session`` jobs at ``venue`` (of one user and/or project)."""
    if pool is None:
        return 0
    clauses = ["job_type = %s", "status = ANY(%s)", "venue = %s"]
    params: list[Any] = [SESSION_JOB_TYPE, list(LIVE_JOB_STATUSES), venue]
    for column, value in (("user_id", user_id), ("project_id", project_id)):
        if value:
            clauses.append(f"{column} = %s")
            params.append(value)
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM jobs WHERE " + " AND ".join(clauses),
                    tuple(params),
                )
                row = cur.fetchone()
    finally:
        pool.putconn(conn)
    return int(row[0]) if row else 0


def _refusal(
    scope: str, subject: str, venue: str, limit: int, used: int
) -> HTTPException:
    where = venue or "this site"
    message = (
        f"concurrent session limit reached for {scope} {subject} at {where}: "
        f"{used} of {limit} live sessions; stop one (r4 session stop <id>) "
        "or ask a platform admin to raise the limit"
    )
    error = PlatformError.quota(
        "concurrent_sessions_exceeded",
        message,
        scope=scope,
        subject=subject,
        limit=limit,
        used=used,
    )
    return HTTPException(**error.to_http_parts())


def check_concurrent_sessions(pool: Any, check: StartCheck) -> None:
    """Raise HTTPException(quota.concurrent_sessions_exceeded) at the cap."""
    limits = session_limits(check.policy)
    for scope, key, attr, column in _SCOPES:
        limit = limits[key]
        subject = str(getattr(check, attr, "") or "")
        if limit is None or not subject:
            continue
        used = count_live_sessions(pool, venue=check.venue, **{column: subject})
        if used >= limit:
            raise _refusal(scope, subject, check.venue, limit, used)


__all__ = [
    "SESSION_JOB_TYPE",
    "check_concurrent_sessions",
    "count_live_sessions",
    "session_limits",
]
