"""Live session counts and the session caps a policy sets (U3 S1 item 9).

The quota status route reads these; the venue start check
(``session_cap``, whose start-lock half moves to ``umrs_r4`` in U3 S5)
enforces them. Moved verbatim from ``session_cap``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from umrs.api.agent_quota import LIVE_JOB_STATUSES

SESSION_JOB_TYPE = "frontend_session"


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


__all__ = ["SESSION_JOB_TYPE", "count_live_sessions", "session_limits"]
