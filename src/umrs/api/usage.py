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

from umrs.billing import compute_cost, load_pricing

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
# GET /umrs/usage/cost — per-user cost breakdown (admin)
# ------------------------------------------------------------------


@router.get("/cost")
async def usage_cost(
    request: Request,
    project_id: str | None = Query(None, description="Filter by project"),
    model: str | None = Query(None, description="Filter by model"),
    since: date | None = Query(None, description="Start date (inclusive)"),
    until: date | None = Query(None, description="End date (exclusive)"),
):
    """Return per-user USD cost breakdown.

    Applies the billing cost model (see ``umrs/src/umrs/billing``) to
    the aggregated token usage, returning per-user cost by token class:

    - ``prompt_cost`` — non-cached input tokens
    - ``completion_cost`` — visible output tokens (excludes reasoning)
    - ``cached_read_cost`` — cache hits (billed at lower rate)
    - ``cached_write_cost`` — cache writes
    - ``reasoning_cost`` — thinking-model reasoning tokens
    - ``audio_cost`` — audio tokens
    - ``total_cost`` — sum of the above

    Requires admin authentication.
    """
    if not _is_admin(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin token required for cost aggregation",
        )

    pool = _get_pool(request)
    _caller_user_id(request)  # validate caller identity

    rows = _query_cost(
        pool,
        project_id=project_id,
        model=model,
        since=since,
        until=until,
    )
    return {"group_by": "user", "rows": rows}


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

    Requires admin authentication.
    """
    if not _is_admin(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin token required for billing aggregation",
        )

    pool = _get_pool(request)
    _caller_user_id(request)  # validate caller identity

    where_parts = []
    params: list = []

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
    """  # noqa: S608

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
               reasoning_tokens, audio_tokens,
               accepted_prediction_tokens, rejected_prediction_tokens,
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


def _query_cost(
    pool,
    *,
    project_id: str | None = None,
    model: str | None = None,
    since: date | None = None,
    until: date | None = None,
) -> list[dict]:
    """Aggregate per-user token usage and apply the cost model.

    Returns a list of dicts, one per (user, model), containing:

    - ``user_id``
    - ``model``
    - ``request_count``
    - Per-class token totals (prompt, completion, reasoning, cache, audio)
    - Per-class USD costs (``prompt_cost``, ``completion_cost`` etc.)
    - ``total_cost`` — sum of all per-class costs
    """
    pricing = load_pricing()

    where_parts = []
    params: list = []

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
        SELECT user_id,
               model,
               count(*)            AS request_count,
               sum(prompt_tokens)  AS total_prompt_tokens,
               sum(completion_tokens) AS total_completion_tokens,
               sum(total_tokens)   AS total_tokens,
               sum(cached_read_tokens) AS total_cached_read_tokens,
               sum(cached_write_tokens) AS total_cached_write_tokens,
               sum(reasoning_tokens) AS total_reasoning_tokens,
               sum(audio_tokens)   AS total_audio_tokens,
               avg(duration_ms)    AS avg_duration_ms
        FROM llm_usage_log
        {where_clause}
        GROUP BY user_id, model
        ORDER BY user_id, model
    """  # noqa: S608

    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [desc[0] for desc in cur.description]
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        pool.putconn(conn)

    # Apply the cost model to each (user, model) row
    out: list[dict] = []
    for row in rows:
        model_name = str(row.get("model") or "")
        rates = pricing.rates_for(model_name)
        cost = compute_cost(
            prompt_tokens=int(row.get("total_prompt_tokens", 0)),
            completion_tokens=int(row.get("total_completion_tokens", 0)),
            cached_read_tokens=int(row.get("total_cached_read_tokens", 0)),
            cached_write_tokens=int(row.get("total_cached_write_tokens", 0)),
            reasoning_tokens=int(row.get("total_reasoning_tokens", 0)),
            audio_tokens=int(row.get("total_audio_tokens", 0)),
            rates=rates,
        )
        out.append({**row, **cost})
    return out
