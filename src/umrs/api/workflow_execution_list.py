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

from collections.abc import Sequence
from dataclasses import dataclass
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
    raise NotImplementedError("a105-s6a-05")


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
    raise NotImplementedError("a105-s6a-05")


def summary_groups(records: Sequence[ExecutionRecord]) -> list[SummaryGroup]:
    """One ``SummaryGroup`` per distinct ``(status, workflow_name)``.

    ``kind`` is the workflow name; ``n`` the number of records;
    ``oldest`` / ``newest`` the smallest / largest ``submitted_at`` of the
    group as timezone-aware UTC datetimes
    (``datetime.fromtimestamp(ts, tz=timezone.utc)``). Groups are sorted by
    ``(status, kind)``. No records: ``[]``.
    """
    raise NotImplementedError("a105-s6a-05")


def position_to_fields(position: ExecutionPosition) -> dict[str, str]:
    """The cursor payload: ``{"t": repr(submitted_at), "e": execution_id}``
    (``repr`` of a float round-trips exactly through ``float()``)."""
    raise NotImplementedError("a105-s6a-05")


def position_from_fields(fields: dict[str, str]) -> ExecutionPosition:
    """Parse ``position_to_fields`` output back into a position.

    ``fields`` must have exactly the keys ``"t"`` and ``"e"``, both ``str``
    values; ``"t"`` must parse with ``float()`` to a finite number; ``"e"`` must be a non-empty
    string of at most 128 characters. Anything else raises ``ValueError``
    whose message is exactly ``"cursor is malformed"``.
    """
    raise NotImplementedError("a105-s6a-05")
