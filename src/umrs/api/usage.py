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

import logging
from datetime import date, datetime, timezone

from fastapi import APIRouter, Query, Request

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/usage", tags=["usage"])


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _get_pool(request: Request):
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.usage("unavailable", "Usage database unavailable"))
    return pool


def _caller(request: Request) -> tuple[str, bool]:
    """``(user_id, is_admin)`` of the caller.

    Strict mode: the HMAC-verified gateway identity (401 when absent or
    unsigned). Admin used to come from an unverified ``X-Verified-Is-Admin``
    or an ``X-User-Role: admin`` header, and the user from a bare
    ``X-User-ID`` (H-1 item 7, 2026-10-17); ``X-User-Role`` has no sender
    and is no longer read. Mvp mode keeps the legacy header trust.
    """
    from ._verified_headers import _strict_mode_env, resolve_caller

    headers = request.headers
    uid = headers.get("x-user-id", "").strip()
    if not uid:
        raise_typed(PlatformError.auth("missing_header", "Missing X-User-ID header"))
    admin_header = headers.get("x-verified-is-admin")
    if not _strict_mode_env():
        return uid, (admin_header or "").strip().lower() == "true"
    caller = resolve_caller(
        uid,
        headers.get("x-verified-project-id"),
        admin_header,
        headers.get("x-verified-hmac"),
    )
    return caller.user_id, caller.is_admin


def _require_admin(request: Request) -> None:
    _uid, admin = _caller(request)
    if not admin:
        raise_typed(
            PlatformError.auth(
                "admin_required", "Admin token required for billing aggregation"
            )
        )


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
    user_id, _admin = _caller(request)

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
    caller, admin = _caller(request)

    # Non-admins can only see their own data
    if not admin:
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
    caller, admin = _caller(request)

    if not admin:
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
# GET /umrs/usage/billing — per-user billing aggregation (admin)
# ------------------------------------------------------------------


@router.get("/billing")
async def usage_billing(
    request: Request,
    project_id: str | None = Query(None, description="Filter by project"),
    model: str | None = Query(None, description="Filter by model"),
    since: date | None = Query(None, description="Start date (inclusive)"),
    until: date | None = Query(None, description="End date (exclusive)"),
):
    """Return per-user billing aggregation.

    Groups by ``user_id`` and returns token consumption broken down by:
    - ``prompt_tokens`` — input tokens
    - ``completion_tokens`` — output tokens (includes reasoning)
    - ``reasoning_tokens`` — thinking-model reasoning tokens (subset of completion)
    - ``cached_read_tokens`` — cache hits (billed at lower rate)
    - ``cached_write_tokens`` — cache writes

    Requires admin authentication (HMAC-verified in strict mode).
    """
    _require_admin(request)
    pool = _get_pool(request)
    where_clause, params = _billing_where(project_id, model, since, until)

    sql = f"""
        SELECT user_id,
               count(*)            AS request_count,
               sum(prompt_tokens)  AS total_prompt_tokens,
               sum(completion_tokens) AS total_completion_tokens,
               sum(total_tokens)   AS total_tokens,
               sum(cached_read_tokens) AS total_cached_read_tokens,
               sum(cached_write_tokens) AS total_cached_write_tokens,
               sum(reasoning_tokens) AS total_reasoning_tokens,
               sum(audio_tokens) AS total_audio_tokens,
               sum(accepted_prediction_tokens) AS total_accepted_prediction_tokens,
               sum(rejected_prediction_tokens) AS total_rejected_prediction_tokens,
               avg(duration_ms)    AS avg_duration_ms
        FROM llm_usage_log
        {where_clause}
        GROUP BY user_id
        ORDER BY total_tokens DESC
    """

    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [desc[0] for desc in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        return {"group_by": "user", "rows": rows}
    finally:
        pool.putconn(conn)


# ------------------------------------------------------------------
# Query helpers (sync, run in pool.getconn/putconn)
# ------------------------------------------------------------------


def _billing_where(
    project_id: str | None,
    model: str | None,
    since: date | None,
    until: date | None,
) -> tuple[str, list]:
    """WHERE clause + params for the billing aggregation filters."""
    filters = (
        ("project_id = %s", project_id),
        ("model = %s", model),
        ("event_time >= %s", _date_to_ts(since)),
        ("event_time < %s", _date_to_ts(until)),
    )
    where_parts = [clause for clause, value in filters if value]
    params: list = [value for _clause, value in filters if value]
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    return where_clause, params


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
               sum(reasoning_tokens) AS total_reasoning_tokens,
               sum(audio_tokens) AS total_audio_tokens,
               sum(accepted_prediction_tokens) AS total_accepted_prediction_tokens,
               sum(rejected_prediction_tokens) AS total_rejected_prediction_tokens,
               avg(duration_ms)    AS avg_duration_ms
        FROM llm_usage_log
        {where_clause}
        GROUP BY {group_col}
        ORDER BY total_tokens DESC
        LIMIT 200
    """

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
               reasoning_tokens, audio_tokens,
               accepted_prediction_tokens, rejected_prediction_tokens,
               is_streaming, duration_ms, status_code
        FROM llm_usage_log
        {where_clause}
        ORDER BY event_time DESC
        LIMIT %s OFFSET %s
    """

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
