"""``fold_summary`` (ADR-105 D-6 S6a, task a105-s6a-04): summary=true shape."""

from __future__ import annotations

from datetime import datetime, timezone

from _a105_s6a_pending import pending
from umrs.jobs.read_summary import SummaryGroup, fold_summary

pytestmark = pending("a105-s6a-04")

D1 = datetime(2026, 11, 1, 9, 0)
D2 = datetime(2026, 11, 5, 12, 30)
D3 = datetime(2026, 11, 20, 17, 30)


def test_empty_input_gives_the_zero_summary() -> None:
    assert fold_summary([]) == {
        "total": 0,
        "by_status": {},
        "by_type": {},
        "oldest": None,
        "newest": None,
    }


def test_counts_fold_by_status_and_type() -> None:
    out = fold_summary(
        [
            SummaryGroup("completed", "knowledge_embed", 3, D2, D3),
            SummaryGroup("completed", "workflow_run", 2, D1, D2),
            SummaryGroup("failed", "knowledge_embed", 1, D2, D2),
            SummaryGroup("failed", "workflow_run", 1, D3, D3),
        ]
    )
    assert out["total"] == 7
    assert out["by_status"] == {"completed": 5, "failed": 2}
    assert out["by_type"] == {"knowledge_embed": 4, "workflow_run": 3}
    assert out["oldest"] == D1.isoformat()
    assert out["newest"] == D3.isoformat()
    assert set(out) == {"total", "by_status", "by_type", "oldest", "newest"}


def test_map_keys_are_sorted() -> None:
    out = fold_summary(
        [
            SummaryGroup("running", "zeta", 1, D1, D1),
            SummaryGroup("completed", "alpha", 1, D1, D1),
            SummaryGroup("queued", "mid", 1, D1, D1),
        ]
    )
    assert list(out["by_status"]) == ["completed", "queued", "running"]
    assert list(out["by_type"]) == ["alpha", "mid", "zeta"]


def test_zero_count_groups_add_nothing() -> None:
    out = fold_summary(
        [
            SummaryGroup("completed", "a", 2, D2, D2),
            SummaryGroup("failed", "b", 0, D1, D3),
            SummaryGroup("cancelled", "c", -1, D1, D3),
        ]
    )
    assert out["total"] == 2
    assert out["by_status"] == {"completed": 2}
    assert out["by_type"] == {"a": 2}
    assert out["oldest"] == D2.isoformat()
    assert out["newest"] == D2.isoformat()


def test_missing_timestamps_are_skipped() -> None:
    out = fold_summary(
        [
            SummaryGroup("completed", "a", 1, None, None),
            SummaryGroup("failed", "a", 1, D2, None),
        ]
    )
    assert out["oldest"] == D2.isoformat()
    assert out["newest"] is None


def test_aware_datetimes_render_with_offset() -> None:
    t = datetime(2026, 11, 2, 3, 4, 5, tzinfo=timezone.utc)
    out = fold_summary([SummaryGroup("running", "wf", 1, t, t)])
    assert out["oldest"] == "2026-11-02T03:04:05+00:00"
    assert out["newest"] == "2026-11-02T03:04:05+00:00"


def test_a_generator_is_consumed_once() -> None:
    def gen():
        yield SummaryGroup("completed", "a", 2, D1, D2)
        yield SummaryGroup("completed", "b", 3, D2, D3)

    out = fold_summary(gen())
    assert out["total"] == 5
    assert out["by_status"] == {"completed": 5}
    assert out["oldest"] == D1.isoformat()
    assert out["newest"] == D3.isoformat()
