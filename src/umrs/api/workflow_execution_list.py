"""Rows, pages and summary groups for ``GET /umrs/workflows/executions``.

ADR-105 D-6, slice S6a: the HTTP twin of the ZMQ ``WORKFLOW_EXECUTIONS``
list (``server/workflow_registry_handler._handle_workflow_executions``),
with the same source of truth (the in-process execution registry) and the
read shaping of the job list: ``fields=`` projection, an opaque cursor and
``summary=true`` aggregates.

The route (``api/workflows.py``) decides *which* records the caller may see
(``can_access_execution`` on each record, owner pinned for non-admins) and
hands only the visible ones to this module. Nothing here widens or narrows
visibility. Pure: no I/O, no registry access, no clock.

Order: ``submitted_at`` descending, then ``execution_id`` descending (a
total order, so a keyset position never repeats or skips a record).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from umrs.jobs.read_summary import SummaryGroup
from umrs.server.workflow_execution_registry import ExecutionRecord

#: Step states that count as done in ``steps_done`` (as the ZMQ list does).
DONE_STEP_STATES = frozenset({"succeeded", "failed", "skipped"})

#: Every key of an ``execution_row``; the allowed ``fields=`` names.
EXECUTION_FIELDS = frozenset(
    {
        "execution_id",
        "workflow_name",
        "workflow_version",
        "user_id",
        "status",
        "progress",
        "submitted_at",
        "started_at",
        "finished_at",
        "error",
        "steps_done",
        "steps_total",
        "durable_job_id",
    }
)


@dataclass(frozen=True)
class ExecutionPosition:
    """The last record a page returned: the next page starts strictly after it."""

    submitted_at: float
    execution_id: str


def execution_row(record: ExecutionRecord) -> dict[str, Any]:
    """One list row: exactly the keys of ``EXECUTION_FIELDS``.

    Values come straight from the record's attributes of the same name
    (times stay epoch floats or None). ``steps_total`` is
    ``len(record.steps)``; ``steps_done`` counts the steps whose ``status``
    is in ``DONE_STEP_STATES``.
    """
    return {
        "execution_id": record.execution_id,
        "workflow_name": record.workflow_name,
        "workflow_version": record.workflow_version,
        "user_id": record.user_id,
        "status": record.status,
        "progress": record.progress,
        "submitted_at": record.submitted_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
        "error": record.error,
        "steps_done": sum(
            step.status in DONE_STEP_STATES for step in record.steps.values()
        ),
        "steps_total": len(record.steps),
        "durable_job_id": record.durable_job_id,
    }


def page_executions(
    records: Sequence[ExecutionRecord],
    *,
    after: ExecutionPosition | None,
    limit: int,
) -> tuple[list[ExecutionRecord], ExecutionPosition | None]:
    """One page of ``records`` and the position to continue from.

    1. Sort a copy of ``records`` by ``(submitted_at, execution_id)``
       descending (do not mutate the input).
    2. With ``after``, keep only the records strictly after it in that
       order: ``(r.submitted_at, r.execution_id) < (after.submitted_at,
       after.execution_id)``.
    3. Take the first ``limit`` records.
    4. The second value is the position of the last record taken when at
       least one more record remains after it, else None.

    Raise ``ValueError`` when ``limit`` is not an ``int`` >= 1 (``bool`` is
    rejected).
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be an integer >= 1")

    ordered = sorted(
        records,
        key=lambda record: (record.submitted_at, record.execution_id),
        reverse=True,
    )
    if after is not None:
        after_key = (after.submitted_at, after.execution_id)
        ordered = [
            record
            for record in ordered
            if (record.submitted_at, record.execution_id) < after_key
        ]

    page = ordered[:limit]
    if len(ordered) <= limit:
        return page, None
    last = page[-1]
    return page, ExecutionPosition(last.submitted_at, last.execution_id)


def summary_groups(records: Sequence[ExecutionRecord]) -> list[SummaryGroup]:
    """One ``SummaryGroup`` per distinct ``(status, workflow_name)``.

    ``kind`` is the workflow name; ``n`` the number of records;
    ``oldest`` / ``newest`` the smallest / largest ``submitted_at`` of the
    group as timezone-aware UTC datetimes
    (``datetime.fromtimestamp(ts, tz=timezone.utc)``). Groups are sorted by
    ``(status, kind)``. No records: ``[]``.
    """
    grouped: dict[tuple[str, str], list[ExecutionRecord]] = {}
    for record in records:
        key = (record.status, record.workflow_name)
        grouped.setdefault(key, []).append(record)

    groups: list[SummaryGroup] = []
    for status, kind in sorted(grouped):
        group = grouped[(status, kind)]
        groups.append(
            SummaryGroup(
                status=status,
                kind=kind,
                n=len(group),
                oldest=datetime.fromtimestamp(
                    min(record.submitted_at for record in group), tz=timezone.utc
                ),
                newest=datetime.fromtimestamp(
                    max(record.submitted_at for record in group), tz=timezone.utc
                ),
            )
        )
    return groups


def position_to_fields(position: ExecutionPosition) -> dict[str, str]:
    """The cursor payload: ``{"t": repr(submitted_at), "e": execution_id}``
    (``repr`` of a float round-trips exactly through ``float()``)."""
    return {"t": repr(position.submitted_at), "e": position.execution_id}


def position_from_fields(fields: dict[str, str]) -> ExecutionPosition:
    """Parse ``position_to_fields`` output back into a position.

    ``fields`` must have exactly the keys ``"t"`` and ``"e"``, both ``str``
    values; ``"t"`` must parse with ``float()`` to a finite number; ``"e"`` must be a non-empty
    string of at most 128 characters. Anything else raises ``ValueError``
    whose message is exactly ``"cursor is malformed"``.
    """
    if set(fields) != {"t", "e"}:
        raise ValueError("cursor is malformed")

    raw_submitted_at = fields["t"]
    execution_id = fields["e"]
    if (
        not isinstance(raw_submitted_at, str)
        or not isinstance(execution_id, str)
        or not execution_id
        or len(execution_id) > 128
    ):
        raise ValueError("cursor is malformed")

    try:
        submitted_at = float(raw_submitted_at)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("cursor is malformed") from exc
    if not math.isfinite(submitted_at):
        raise ValueError("cursor is malformed")
    return ExecutionPosition(submitted_at=submitted_at, execution_id=execution_id)
