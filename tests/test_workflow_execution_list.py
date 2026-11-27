"""Execution list rows, pages and groups (ADR-105 D-6 S6a, task a105-s6a-05)."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest
from _a105_s6a_pending import pending
from umrs.api.workflow_execution_list import (
    EXECUTION_FIELDS,
    ExecutionPosition,
    execution_row,
    page_executions,
    position_from_fields,
    position_to_fields,
    summary_groups,
)
from umrs.jobs.read_summary import SummaryGroup
from umrs.server.workflow_execution_registry import ExecutionRecord, StepProgress

pytestmark = pending("a105-s6a-05")


def _rec(eid: str, ts: float, status: str = "running", wf: str = "wf") -> ExecutionRecord:
    return ExecutionRecord(
        execution_id=eid, workflow_name=wf, user_id="alice", status=status, submitted_at=ts
    )


def test_row_has_exactly_the_declared_keys() -> None:
    rec = ExecutionRecord(
        execution_id="wf-1",
        workflow_name="align",
        user_id="alice",
        status="running",
        submitted_at=100.5,
        started_at=101.0,
        progress=0.5,
        durable_job_id="job-9",
        workflow_version="1.2.0",
        error=None,
    )
    rec.steps = {
        "a": StepProgress(name="a", status="succeeded"),
        "b": StepProgress(name="b", status="failed"),
        "c": StepProgress(name="c", status="skipped"),
        "d": StepProgress(name="d", status="running"),
        "e": StepProgress(name="e", status="pending"),
    }
    row = execution_row(rec)
    assert set(row) == EXECUTION_FIELDS
    assert row == {
        "execution_id": "wf-1",
        "workflow_name": "align",
        "workflow_version": "1.2.0",
        "user_id": "alice",
        "status": "running",
        "progress": 0.5,
        "submitted_at": 100.5,
        "started_at": 101.0,
        "finished_at": None,
        "error": None,
        "steps_done": 3,
        "steps_total": 5,
        "durable_job_id": "job-9",
    }


def test_row_without_steps() -> None:
    row = execution_row(_rec("wf-2", 1.0))
    assert row["steps_done"] == 0
    assert row["steps_total"] == 0


def test_page_orders_newest_first_with_id_tiebreak() -> None:
    recs = [_rec("a", 1.0), _rec("c", 2.0), _rec("b", 2.0), _rec("d", 0.5)]
    page, nxt = page_executions(recs, after=None, limit=10)
    assert [r.execution_id for r in page] == ["c", "b", "a", "d"]
    assert nxt is None
    assert [r.execution_id for r in recs] == ["a", "c", "b", "d"]  # input untouched


def test_page_returns_next_position_only_when_more_remain() -> None:
    recs = [_rec(f"e{i}", float(i)) for i in range(5)]
    page, nxt = page_executions(recs, after=None, limit=2)
    assert [r.execution_id for r in page] == ["e4", "e3"]
    assert nxt == ExecutionPosition(submitted_at=3.0, execution_id="e3")
    page, nxt = page_executions(recs, after=None, limit=5)
    assert len(page) == 5
    assert nxt is None


def test_walking_pages_with_ties_visits_each_record_once() -> None:
    recs = [_rec(f"x{i:03d}", float(i // 7)) for i in range(50)]
    seen: list[str] = []
    after = None
    for _ in range(100):
        page, after = page_executions(recs, after=after, limit=6)
        seen.extend(r.execution_id for r in page)
        if after is None:
            break
    assert len(seen) == 50
    assert sorted(seen) == sorted(r.execution_id for r in recs)


def test_after_is_strict() -> None:
    recs = [_rec("a", 1.0), _rec("b", 1.0), _rec("c", 2.0)]
    page, nxt = page_executions(
        recs, after=ExecutionPosition(submitted_at=1.0, execution_id="b"), limit=10
    )
    assert [r.execution_id for r in page] == ["a"]
    assert nxt is None


def test_after_past_the_end_gives_an_empty_page() -> None:
    page, nxt = page_executions(
        [_rec("a", 5.0)], after=ExecutionPosition(submitted_at=0.0, execution_id=""), limit=3
    )
    assert page == []
    assert nxt is None


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_page_rejects_bad_limit(limit: object) -> None:
    with pytest.raises(ValueError):
        page_executions([], after=None, limit=limit)  # type: ignore[arg-type]


def test_summary_groups() -> None:
    recs = [
        _rec("a", 10.0, "succeeded", "align"),
        _rec("b", 30.0, "succeeded", "align"),
        _rec("c", 20.0, "failed", "align"),
        _rec("d", 5.0, "succeeded", "blast"),
    ]
    utc = timezone.utc
    assert summary_groups(recs) == [
        SummaryGroup(
            "failed",
            "align",
            1,
            datetime.fromtimestamp(20.0, tz=utc),
            datetime.fromtimestamp(20.0, tz=utc),
        ),
        SummaryGroup(
            "succeeded",
            "align",
            2,
            datetime.fromtimestamp(10.0, tz=utc),
            datetime.fromtimestamp(30.0, tz=utc),
        ),
        SummaryGroup(
            "succeeded",
            "blast",
            1,
            datetime.fromtimestamp(5.0, tz=utc),
            datetime.fromtimestamp(5.0, tz=utc),
        ),
    ]


def test_summary_groups_empty() -> None:
    assert summary_groups([]) == []


def test_position_round_trip_is_exact() -> None:
    p = ExecutionPosition(submitted_at=1764230400.123456789, execution_id="wf-abc")
    fields = position_to_fields(p)
    assert fields == {"t": repr(p.submitted_at), "e": "wf-abc"}
    assert position_from_fields(fields) == p


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"t": "1.0"},
        {"e": "wf"},
        {"t": "1.0", "e": "wf", "x": "y"},
        {"t": "soon", "e": "wf"},
        {"t": "nan", "e": "wf"},
        {"t": "inf", "e": "wf"},
        {"t": "1.0", "e": ""},
        {"t": "1.0", "e": "x" * 129},
        {"t": 1.0, "e": "wf"},
        {"t": "1.0", "e": 3},
    ],
)
def test_position_from_bad_fields(fields: dict) -> None:
    with pytest.raises(ValueError, match="^cursor is malformed$"):
        position_from_fields(fields)


def test_position_accepts_128_char_id() -> None:
    p = position_from_fields({"t": "2.5", "e": "x" * 128})
    assert p == ExecutionPosition(submitted_at=2.5, execution_id="x" * 128)
    assert math.isfinite(p.submitted_at)
