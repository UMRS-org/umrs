"""``summary=true`` aggregates for list reads (ADR-105 D-6, slice S6a).

A list read with ``summary=true`` answers counts instead of rows. Both the
job list (``GET /umrs/jobs``) and the workflow executions list
(``GET /umrs/workflows/executions``) produce *groups* (one per
(status, kind) pair, with a row count ``n`` and the oldest and newest timestamp in
the group) and fold them here into one summary of a fixed shape:

    {
      "total": 7,
      "by_status": {"completed": 5, "failed": 2},
      "by_type": {"knowledge_embed": 4, "workflow_run": 3},
      "oldest": "2026-11-01T09:00:00",
      "newest": "2026-11-20T17:30:00",
    }

``kind`` is the job type for jobs and the workflow name for executions;
both land under ``by_type``. Pure: no I/O, no clock.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any, NamedTuple


class SummaryGroup(NamedTuple):
    """One (status, kind) group: its row count and time range."""

    status: str
    kind: str
    n: int
    oldest: datetime | None
    newest: datetime | None


def fold_summary(groups: Iterable[SummaryGroup]) -> dict[str, Any]:
    """Fold groups into the summary shape shown in the module doc.

    Rules (the tests pin each one):

    - ``total`` is the sum of every group's ``n``.
    - ``by_status`` maps each status to the sum of the counts of its groups;
      ``by_type`` does the same for ``kind``. A group whose ``n`` is 0
      (or negative) adds nothing and creates no key.
    - Both maps are plain dicts whose keys are inserted in ascending
      (sorted) order, so the JSON output is stable.
    - ``oldest`` is the earliest non-None ``oldest`` of the groups with a
      positive ``n``, rendered with ``datetime.isoformat()``; ``newest`` is
      the latest non-None ``newest``, rendered the same way. Either is
      ``None`` when no counted group carries that timestamp.
    - No groups (or only empty ones): ``{"total": 0, "by_status": {},
      "by_type": {}, "oldest": None, "newest": None}``.
    - The input iterable is consumed once (it may be a generator).
    """
    counted = [group for group in groups if group.n > 0]
    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for group in counted:
        by_status[group.status] = by_status.get(group.status, 0) + group.n
        by_type[group.kind] = by_type.get(group.kind, 0) + group.n
    oldest = min((g.oldest for g in counted if g.oldest is not None), default=None)
    newest = max((g.newest for g in counted if g.newest is not None), default=None)
    return {
        "total": sum(group.n for group in counted),
        "by_status": dict(sorted(by_status.items())),
        "by_type": dict(sorted(by_type.items())),
        "oldest": _iso(oldest),
        "newest": _iso(newest),
    }


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None
