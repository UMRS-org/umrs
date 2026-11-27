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
    raise NotImplementedError("a105-s6a-01")


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
    raise NotImplementedError("a105-s6a-01")


def summary_sql(filters: JobReadFilters) -> tuple[str, list[Any]]:
    """The grouped aggregate for ``summary=true``, and its parameters.

    Whitespace-normalised SQL, exactly: ``"SELECT status, job_type,
    COUNT(*) AS n, MIN(created_at) AS oldest, MAX(created_at) AS newest
    FROM jobs <where> GROUP BY status, job_type ORDER BY status,
    job_type"`` with ``<where>`` from ``where_clause(filters)`` (no keyset;
    a single space when empty). Parameters: the WHERE parameters.
    """
    raise NotImplementedError("a105-s6a-01")


def position_to_fields(position: JobKeysetPosition) -> dict[str, str]:
    """The cursor payload of a position: ``{"c": created_at.isoformat(),
    "j": job_id}``."""
    raise NotImplementedError("a105-s6a-01")


def position_from_fields(fields: dict[str, str]) -> JobKeysetPosition:
    """Parse ``position_to_fields`` output back into a position.

    ``fields`` must have exactly the keys ``"c"`` and ``"j"``, both ``str``
    values; ``"c"`` must parse with ``datetime.fromisoformat``; ``"j"`` must be a non-empty
    string of at most 128 characters. Anything else raises ``ValueError``
    whose message is exactly ``"cursor is malformed"``. Round trip:
    ``position_from_fields(position_to_fields(p)) == p``.
    """
    raise NotImplementedError("a105-s6a-01")
