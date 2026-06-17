"""
LLM Usage Query API.

Exposes usage summaries and per-request export for billing dashboards,
admin reports, and the ``r4 usage`` CLI.

Endpoints
---------
- GET /umrs/usage/me         Per-user usage summary (own data)
- GET /umrs/usage/summary    Admin usage summary (any user/project)
- GET /umrs/usage/records    Raw record export (admin)

All reads use the shared ``umrs_db_pool`` (psycopg2 ThreadedConnectionPool).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request, status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/usage", tags=["usage"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_pool(request: Request):
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Usage database unavailable",
        )
    return pool


def _caller_user_id(request: Request) -> str:
    uid = request.headers.get("x-user-id", "").strip()
    if not uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-ID header",
        )
    return uid


def _is_admin(request: Request) -> bool:
    """Check if caller has admin role (set by auth sidecar).

    The auth sidecar emits ``X-Verified-Is-Admin: true|false`` after
    JWT validation (see gateway/sidecar/main.py).  We honour that
    signal and fall back to ``X-User-Role`` for older sidecar
    revisions or direct-UMRS clients.
    """
    verified_admin = request.headers.get(
        "x-verified-is-admin", ""
    ).strip().lower()
    if verified_admin == "true":
        return True

    role = request.headers.get("x-user-role", "").strip().lower()
    return role in ("admin", "operator")


def _date_to_ts(d: date | None) -> datetime | None:
    if d is None:
        return None
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


# ------------------------------------------------------------------
# GET /umrs/usage/me — own usage summary
# ------------------------------------------------------------------


@router.get("/me")
async def my_usage(
    request: Request,
    since: date | None = Query(None, description="Start date (inclusive)"),
    until: date | None = Query(None, description="End date (exclusive)"),
    group_by: str = Query("model", description="Group by: model | day | model_day"),
):
    """Return token usage summary for the authenticated user."""
    pool = _get_pool(request)
    user_id = _caller_user_id(request)

    rows = _query_summary(
        pool, user_id=user_id, since=since, until=until, group_by=group_by
    )
    return {"user_id": user_id, "group_by": group_by, "rows": rows}


# ------------------------------------------------------------------
# GET /umrs/usage/summary — admin summary (any user/project)
# ------------------------------------------------------------------


@router.get("/summary")
async def usage_summary(
    request: Request,
    user_id: str | None = Query(None),
    project_id: str | None = Query(None),
    model: str | None = Query(None),
    since: date | None = Query(None, description="Start date (inclusive)"),
    until: date | None = Query(None, description="End date (exclusive)"),
    group_by: str = Query(
        "model", description="Group by: model | day | user | user_model | project"
    ),
):
    """Return token usage summary (admin-level, filterable)."""
    pool = _get_pool(request)
    caller = _caller_user_id(request)

    # Non-admins can only see their own data
    if not _is_admin(request):
        user_id = caller

    rows = _query_summary(
        pool,
        user_id=user_id,
        project_id=project_id,
        model=model,
        since=since,
        until=until,
        group_by=group_by,
    )
    return {"group_by": group_by, "rows": rows}


# ------------------------------------------------------------------
# GET /umrs/usage/records — raw record export
# ------------------------------------------------------------------


@router.get("/records")
async def usage_records(
    request: Request,
    user_id: str | None = Query(None),
    project_id: str | None = Query(None),
    model: str | None = Query(None),
    since: date | None = Query(None),
    until: date | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Return raw usage records (admin or own data)."""
    pool = _get_pool(request)
    caller = _caller_user_id(request)

    if not _is_admin(request):
        user_id = caller

    rows = _query_records(
        pool,
        user_id=user_id,
        project_id=project_id,
        model=model,
        since=since,
        until=until,
        limit=limit,
        offset=offset,
    )
    return {"records": rows, "limit": limit, "offset": offset}


# ------------------------------------------------------------------
# Query helpers (sync, run in pool.getconn/putconn)
# ------------------------------------------------------------------


_GROUP_COLUMNS = {
    "model": "model",
    "day": "event_time::date",
    "user": "user_id",
    "user_model": "user_id, model",
    "project": "project_id",
    "model_day": "model, event_time::date",
}


def _query_summary(
    pool,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    model: str | None = None,
    since: date | None = None,
    until: date | None = None,
    group_by: str = "model",
) -> list[dict]:
    group_col = _GROUP_COLUMNS.get(group_by, "model")
    where_parts = []
    params: list = []

    if user_id:
        where_parts.append("user_id = %s")
        params.append(user_id)
    if project_id:
        where_parts.append("project_id = %s")
        params.append(project_id)
    if model:
        where_parts.append("model = %s")
        params.append(model)
    if since:
        where_parts.append("event_time >= %s")
        params.append(_date_to_ts(since))
    if until:
        where_parts.append("event_time < %s")
        params.append(_date_to_ts(until))

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
        SELECT {group_col},
               count(*)            AS request_count,
               sum(prompt_tokens)  AS total_prompt_tokens,
               sum(completion_tokens) AS total_completion_tokens,
               sum(total_tokens)   AS total_tokens,
               sum(cached_read_tokens) AS total_cached_read_tokens,
               sum(cached_write_tokens) AS total_cached_write_tokens,
               avg(duration_ms)    AS avg_duration_ms
        FROM llm_usage_log
        {where_clause}
        GROUP BY {group_col}
        ORDER BY total_tokens DESC
        LIMIT 200
    """  # noqa: S608

    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        pool.putconn(conn)


def _query_records(
    pool,
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    model: str | None = None,
    since: date | None = None,
    until: date | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    where_parts = []
    params: list = []

    if user_id:
        where_parts.append("user_id = %s")
        params.append(user_id)
    if project_id:
        where_parts.append("project_id = %s")
        params.append(project_id)
    if model:
        where_parts.append("model = %s")
        params.append(model)
    if since:
        where_parts.append("event_time >= %s")
        params.append(_date_to_ts(since))
    if until:
        where_parts.append("event_time < %s")
        params.append(_date_to_ts(until))

    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    sql = f"""
        SELECT id, request_id, event_time, user_id, project_id,
               model, served_model, endpoint_job_id, endpoint_tier,
               prompt_tokens, completion_tokens, total_tokens,
               cached_read_tokens, cached_write_tokens,
               is_streaming, duration_ms, status_code
        FROM llm_usage_log
        {where_clause}
        ORDER BY event_time DESC
        LIMIT %s OFFSET %s
    """  # noqa: S608

    params.extend([limit, offset])

    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [desc[0] for desc in cur.description]
            rows = []
            for row in cur.fetchall():
                d = dict(zip(cols, row))
                # Serialize datetimes for JSON
                if isinstance(d.get("event_time"), datetime):
                    d["event_time"] = d["event_time"].isoformat()
                rows.append(d)
            return rows
    finally:
        pool.putconn(conn)
