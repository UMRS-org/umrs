"""SQL for shaped job-list reads (ADR-105 D-6, slice S6a).

``JobStore.list_jobs_page`` and ``JobStore.summarize_jobs`` run the SQL
built here. The builders are pure (no connection, no I/O) so they are
tested as strings and parameter lists; the Postgres test in
``tests/test_jobs_read_shaping.py`` runs them for real.

Order and cursor. Job pages are ordered ``created_at DESC, job_id DESC``,
a total order (``job_id`` is the primary key), so a keyset position
"strictly after the last row seen" never repeats or skips a row even when
many jobs share one ``created_at``. In Postgres the row-value comparison
``(created_at, job_id) < (%s, %s)`` is exactly "comes later in that order".

Scope. The caller's scope (project and, for a non-admin, owner) arrives in
``JobReadFilters`` from the route, never from a cursor: the cursor only
carries a position. Every value is a bound parameter (``%s``); no value
is ever formatted into the SQL text.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

#: The one order every shaped page uses (and ``JobStore.list_jobs``).
JOB_ORDER_SQL = "ORDER BY created_at DESC, job_id DESC"

#: ``limit`` bounds accepted by ``page_sql`` (the route asks for limit + 1).
MIN_LIMIT = 1
MAX_LIMIT = 1001


@dataclass(frozen=True)
class JobReadFilters:
    """The WHERE filters of a job-list read.

    ``user_id`` filters when it is not None (an empty string still filters,
    so a blank owner can never widen a non-admin read to the project).
    ``project_id``, ``status`` and ``job_type`` filter only when truthy
    (None or ``""`` means "no filter"), as ``JobStore.list_jobs`` does.
    ``status`` and ``job_type`` are the enum *values* (plain strings).
    """

    project_id: str | None = None
    user_id: str | None = None
    status: str | None = None
    job_type: str | None = None


@dataclass(frozen=True)
class JobKeysetPosition:
    """The last row a page returned: the next page starts strictly after it."""

    created_at: datetime
    job_id: str


def where_clause(
    filters: JobReadFilters, after: JobKeysetPosition | None = None
) -> tuple[str, list[Any]]:
    """The ``WHERE`` text and its parameters.

    Conditions, in this exact order, each added only when its filter is
    active (see ``JobReadFilters``):

    1. ``user_id = %s``
    2. ``project_id = %s``
    3. ``status = %s``
    4. ``job_type = %s``
    5. ``(created_at, job_id) < (%s, %s)`` when ``after`` is given, with
       parameters ``after.created_at`` then ``after.job_id``.

    They are joined with ``" AND "`` and prefixed with ``"WHERE "``; the
    parameters are in the same order. No active condition: ``("", [])``.
    Example: ``JobReadFilters(project_id="p", user_id="u")`` gives
    ``("WHERE user_id = %s AND project_id = %s", ["u", "p"])``.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if filters.user_id is not None:
        conditions.append("user_id = %s")
        params.append(filters.user_id)
    if filters.project_id:
        conditions.append("project_id = %s")
        params.append(filters.project_id)
    if filters.status:
        conditions.append("status = %s")
        params.append(filters.status)
    if filters.job_type:
        conditions.append("job_type = %s")
        params.append(filters.job_type)
    if after is not None:
        conditions.append("(created_at, job_id) < (%s, %s)")
        params.extend([after.created_at, after.job_id])

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    return where, params


def page_sql(
    filters: JobReadFilters,
    *,
    after: JobKeysetPosition | None,
    limit: int,
    offset: int = 0,
) -> tuple[str, list[Any]]:
    """``SELECT * FROM jobs`` for one page, and its parameters.

    The SQL, whitespace-normalised (``" ".join(sql.split())``), is exactly
    ``"SELECT * FROM jobs <where> ORDER BY created_at DESC, job_id DESC
    LIMIT %s OFFSET %s"`` where ``<where>`` is ``where_clause(filters,
    after)[0]``; with no WHERE text there is a single space between
    ``jobs`` and ``ORDER``. Parameters: the WHERE parameters, then
    ``limit``, then ``offset``.

    Raises ``ValueError`` when ``limit`` is not an ``int`` between
    ``MIN_LIMIT`` and ``MAX_LIMIT`` inclusive, or ``offset`` is not an
    ``int`` >= 0 (``bool`` is rejected for both).
    """
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit < MIN_LIMIT
        or limit > MAX_LIMIT
    ):
        raise ValueError(f"limit must be an int between {MIN_LIMIT} and {MAX_LIMIT}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative int")

    where, params = where_clause(filters, after)
    sql = f"SELECT * FROM jobs {where} {JOB_ORDER_SQL} LIMIT %s OFFSET %s"
    return sql, [*params, limit, offset]


def summary_sql(filters: JobReadFilters) -> tuple[str, list[Any]]:
    """The grouped aggregate for ``summary=true``, and its parameters.

    Whitespace-normalised SQL, exactly: ``"SELECT status, job_type,
    COUNT(*) AS n, MIN(created_at) AS oldest, MAX(created_at) AS newest
    FROM jobs <where> GROUP BY status, job_type ORDER BY status,
    job_type"`` with ``<where>`` from ``where_clause(filters)`` (no keyset;
    a single space when empty). Parameters: the WHERE parameters.
    """
    where, params = where_clause(filters)
    sql = (
        "SELECT status, job_type, COUNT(*) AS n, "
        "MIN(created_at) AS oldest, MAX(created_at) AS newest "
        f"FROM jobs {where} "
        "GROUP BY status, job_type ORDER BY status, job_type"
    )
    return sql, params


def position_to_fields(position: JobKeysetPosition) -> dict[str, str]:
    """The cursor payload of a position: ``{"c": created_at.isoformat(),
    "j": job_id}``."""
    return {"c": position.created_at.isoformat(), "j": position.job_id}


def position_from_fields(fields: dict[str, str]) -> JobKeysetPosition:
    """Parse ``position_to_fields`` output back into a position.

    ``fields`` must have exactly the keys ``"c"`` and ``"j"``, both ``str``
    values; ``"c"`` must parse with ``datetime.fromisoformat``; ``"j"`` must be a non-empty
    string of at most 128 characters. Anything else raises ``ValueError``
    whose message is exactly ``"cursor is malformed"``. Round trip:
    ``position_from_fields(position_to_fields(p)) == p``.
    """
    if set(fields) != {"c", "j"}:
        raise ValueError("cursor is malformed")

    created_text = fields["c"]
    job_id = fields["j"]
    if (
        not isinstance(created_text, str)
        or not isinstance(job_id, str)
        or not job_id
        or len(job_id) > 128
    ):
        raise ValueError("cursor is malformed")

    try:
        created_at = datetime.fromisoformat(created_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("cursor is malformed") from exc
    return JobKeysetPosition(created_at=created_at, job_id=job_id)
