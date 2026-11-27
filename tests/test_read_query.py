"""Job read SQL builders (ADR-105 D-6 S6a, task a105-s6a-01)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from _a105_s6a_pending import pending
from umrs.jobs.read_query import (
    MAX_LIMIT,
    JobKeysetPosition,
    JobReadFilters,
    page_sql,
    position_from_fields,
    position_to_fields,
    summary_sql,
    where_clause,
)

pytestmark = pending("a105-s6a-01")

T = datetime(2026, 11, 20, 8, 15, 30, 123456)
AFTER = JobKeysetPosition(created_at=T, job_id="job-b")


def _norm(sql: str) -> str:
    return " ".join(sql.split())


def test_no_filter_no_where() -> None:
    assert where_clause(JobReadFilters()) == ("", [])


def test_example_from_the_doc() -> None:
    assert where_clause(JobReadFilters(project_id="p", user_id="u")) == (
        "WHERE user_id = %s AND project_id = %s",
        ["u", "p"],
    )


def test_all_filters_in_fixed_order_then_keyset() -> None:
    f = JobReadFilters(project_id="p", user_id="u", status="running", job_type="workflow_run")
    sql, params = where_clause(f, AFTER)
    assert _norm(sql) == (
        "WHERE user_id = %s AND project_id = %s AND status = %s AND job_type = %s"
        " AND (created_at, job_id) < (%s, %s)"
    )
    assert params == ["u", "p", "running", "workflow_run", T, "job-b"]


def test_keyset_alone() -> None:
    sql, params = where_clause(JobReadFilters(), AFTER)
    assert _norm(sql) == "WHERE (created_at, job_id) < (%s, %s)"
    assert params == [T, "job-b"]


def test_empty_user_id_still_filters() -> None:
    """A blank owner never widens a non-admin read to the whole project."""
    sql, params = where_clause(JobReadFilters(user_id=""))
    assert sql == "WHERE user_id = %s"
    assert params == [""]


@pytest.mark.parametrize("field", ["project_id", "status", "job_type"])
def test_empty_other_filters_are_absent(field: str) -> None:
    assert where_clause(JobReadFilters(**{field: ""})) == ("", [])


def test_values_are_never_in_the_sql_text() -> None:
    evil = "x'; DROP TABLE jobs; --"
    f = JobReadFilters(project_id=evil, user_id=evil, status=evil, job_type=evil)
    sql, params = where_clause(f, JobKeysetPosition(created_at=T, job_id=evil))
    assert evil not in sql
    assert params.count(evil) == 5


def test_page_sql_shape_without_filters() -> None:
    sql, params = page_sql(JobReadFilters(), after=None, limit=10)
    assert _norm(sql) == (
        "SELECT * FROM jobs ORDER BY created_at DESC, job_id DESC LIMIT %s OFFSET %s"
    )
    assert params == [10, 0]


def test_page_sql_with_filters_keyset_and_offset() -> None:
    sql, params = page_sql(JobReadFilters(project_id="p"), after=AFTER, limit=5, offset=3)
    assert _norm(sql) == (
        "SELECT * FROM jobs WHERE project_id = %s AND (created_at, job_id) < (%s, %s)"
        " ORDER BY created_at DESC, job_id DESC LIMIT %s OFFSET %s"
    )
    assert params == ["p", T, "job-b", 5, 3]


@pytest.mark.parametrize("limit", [0, -1, MAX_LIMIT + 1, True, 2.0, "5"])
def test_page_sql_rejects_bad_limit(limit: object) -> None:
    with pytest.raises(ValueError):
        page_sql(JobReadFilters(), after=None, limit=limit)  # type: ignore[arg-type]


@pytest.mark.parametrize("offset", [-1, True, 1.0, "0"])
def test_page_sql_rejects_bad_offset(offset: object) -> None:
    with pytest.raises(ValueError):
        page_sql(JobReadFilters(), after=None, limit=1, offset=offset)  # type: ignore[arg-type]


def test_page_sql_accepts_the_bounds() -> None:
    assert page_sql(JobReadFilters(), after=None, limit=1)[1] == [1, 0]
    assert page_sql(JobReadFilters(), after=None, limit=MAX_LIMIT)[1] == [MAX_LIMIT, 0]


def test_summary_sql_shape() -> None:
    sql, params = summary_sql(JobReadFilters(project_id="p", user_id="u"))
    assert _norm(sql) == (
        "SELECT status, job_type, COUNT(*) AS n, MIN(created_at) AS oldest,"
        " MAX(created_at) AS newest FROM jobs WHERE user_id = %s AND project_id = %s"
        " GROUP BY status, job_type ORDER BY status, job_type"
    )
    assert params == ["u", "p"]


def test_summary_sql_without_filters() -> None:
    sql, params = summary_sql(JobReadFilters())
    assert _norm(sql) == (
        "SELECT status, job_type, COUNT(*) AS n, MIN(created_at) AS oldest,"
        " MAX(created_at) AS newest FROM jobs GROUP BY status, job_type"
        " ORDER BY status, job_type"
    )
    assert params == []


def test_position_fields_round_trip() -> None:
    assert position_to_fields(AFTER) == {"c": T.isoformat(), "j": "job-b"}
    assert position_from_fields(position_to_fields(AFTER)) == AFTER
    aware = JobKeysetPosition(created_at=T.replace(tzinfo=timezone.utc), job_id="j")
    assert position_from_fields(position_to_fields(aware)) == aware


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"c": T.isoformat()},
        {"j": "job"},
        {"c": T.isoformat(), "j": "job", "x": "1"},
        {"c": "not-a-time", "j": "job"},
        {"c": T.isoformat(), "j": ""},
        {"c": T.isoformat(), "j": "x" * 129},
        {"c": 5, "j": "job"},
        {"c": T.isoformat(), "j": 7},
    ],
)
def test_position_from_bad_fields(fields: dict) -> None:
    with pytest.raises(ValueError, match="^cursor is malformed$"):
        position_from_fields(fields)


def test_position_accepts_128_char_job_id() -> None:
    p = position_from_fields({"c": T.isoformat(), "j": "x" * 128})
    assert p.job_id == "x" * 128
