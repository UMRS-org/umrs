"""Admin view of remote-session token usage (ADR-099 D-4/D-5, RS-1 lane Q).

    GET /umrs/admin/usage?user=&project=&venue=&days=1

One row per subject and day from ``llm_usage_log`` rows the inference proxy
wrote (``origin='infer_proxy'``) at one site: prompt, completion and total
tokens, the number of calls, and the subject's daily token limit from the
effective session policy (``null`` = no limit, or unreadable when
``limits`` says so). ``user`` narrows to one user (scope ``user``);
``project`` alone groups by project (scope ``project``); neither lists
every user. Days are calendar days in ``UMRS_QUOTA_DAY_TZ`` (default UTC).

Access: the admin credential of the RS-0 session routes
(``admin_sessions.require_session_admin``). Admin views never include
request or response content: the ledger holds none.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from .admin_sessions import require_session_admin
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/admin/usage", tags=["admin-usage"])

MAX_DAYS = 90
MAX_ROWS = 2000

_SQL = """
SELECT {column},
       to_char(date_trunc('day', event_time AT TIME ZONE %s), 'YYYY-MM-DD') AS day,
       COALESCE(SUM(prompt_tokens), 0),
       COALESCE(SUM(completion_tokens), 0),
       COALESCE(SUM(total_tokens), 0),
       COUNT(*)
  FROM llm_usage_log
 WHERE origin = 'infer_proxy' AND venue = %s AND event_time >= %s{filters}
 GROUP BY 1, 2
 ORDER BY 2 DESC, 1
 LIMIT {limit}
"""


def _zone_name() -> str:
    return (os.environ.get("UMRS_QUOTA_DAY_TZ", "UTC") or "UTC").strip()


def _query(user: str, project: str) -> tuple[str, str, list[str]]:
    """(scope, the SQL with its filters, the filter values)."""
    scope = "project" if project and not user else "user"
    filters = ""
    values: list[str] = []
    for column, value in (("user_id", user), ("project_id", project)):
        if value:
            filters += f" AND {column} = %s"
            values.append(value)
    column = "project_id" if scope == "project" else "user_id"
    sql = _SQL.format(column=column, filters=filters, limit=MAX_ROWS)
    return scope, sql, values


def usage_rows(
    pool: Any, *, venue: str, user: str = "", project: str = "", days: int = 1
) -> tuple[str, list[dict[str, Any]]]:
    """(scope, rows) of the ledger's ``infer_proxy`` usage at ``venue``."""
    from umrs.quotas.enforcer import day_window, day_zone

    zone = day_zone(_zone_name())
    start, _end = day_window(time.time() - 86400 * (days - 1), zone)
    scope, sql, values = _query(user, project)
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql, (str(zone), venue, start, *values))
                fetched = cur.fetchall()
    finally:
        pool.putconn(conn)
    return scope, [_row(scope, r) for r in fetched or []]


def _row(scope: str, r: Any) -> dict[str, Any]:
    return {
        "scope": scope,
        "subject": str(r[0]),
        "day": str(r[1]),
        "prompt_tokens": int(r[2] or 0),
        "completion_tokens": int(r[3] or 0),
        "total_tokens": int(r[4] or 0),
        "requests": int(r[5] or 0),
        "limit": None,
    }


def _limit_for(pool: Any, venue: str, scope: str, subject: str) -> int | None:
    from umrs.quotas.enforcer import policy_limits

    if scope == "project":
        limits = policy_limits(pool, venue=venue, user_id="", project_id=subject)
    else:
        limits = policy_limits(pool, venue=venue, user_id=subject)
    value = ((limits or {}).get("budget") or {}).get(f"tokens_per_day_{scope}")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def attach_limits(pool: Any, venue: str, rows: list[dict[str, Any]]) -> str:
    """Fill each row's ``limit``; "ok", or "unavailable" when unreadable."""
    cache: dict[tuple[str, str], int | None] = {}
    try:
        for row in rows:
            key = (row["scope"], row["subject"])
            if key not in cache:
                cache[key] = _limit_for(pool, venue, *key)
            row["limit"] = cache[key]
    except Exception:  # noqa: BLE001 — usage still shows; the limits say why not
        return "unavailable"
    return "ok"


def _venue(venue: str) -> str:
    from umrs.jobs.store import configured_site_name

    return venue.strip() or configured_site_name()


@router.get("", response_model=dict[str, Any])
def admin_usage(
    request: Request,
    user: str = Query("", description="one user (scope user)"),
    project: str = Query("", description="one project (alone: scope project)"),
    venue: str = Query("", description="site name (default: this site)"),
    days: int = Query(1, ge=1, le=MAX_DAYS),
    admin: str = Depends(require_session_admin),
) -> dict[str, Any]:
    """Remote-session token usage per subject and day.
    # auth: admin(jwt|verified-admin)
    """
    del admin
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.quota("unavailable", "database pool unavailable"))
    site = _venue(venue)
    try:
        scope, rows = usage_rows(
            pool, venue=site, user=user.strip(), project=project.strip(), days=days
        )
    except Exception as exc:  # noqa: BLE001 — a typed 503 naming the failure
        raise_typed(
            PlatformError.quota("unavailable", f"usage ledger unreadable: {exc}"),
            cause=exc,
        )
    limits = attach_limits(pool, site, rows)
    return {"venue": site, "scope": scope, "days": days, "limits": limits, "rows": rows}


__all__ = ["attach_limits", "router", "usage_rows"]
