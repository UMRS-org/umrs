"""Response models shared by the shaped list reads (ADR-105 D-6, slice S6a).

``GET /umrs/jobs`` (``job_list``) and ``GET /umrs/workflows/executions``
(``workflow_executions``) accept the same three read-shaping inputs:

- ``fields=a,b``: rows carry only those keys (plus the row's id);
- ``cursor=<next_cursor>``: the page after the one that returned it;
- ``summary=true``: counts by status and type instead of rows.

Additive: a caller that sends none of them gets the rows it got before,
plus ``next_cursor`` when more rows follow.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ReadSummary(BaseModel):
    """``summary=true`` aggregates over every row the read's filters match."""

    total: int = Field(..., description="Rows matched (all pages)")
    by_status: dict[str, int] = Field(
        default_factory=dict, description="Row count per status"
    )
    by_type: dict[str, int] = Field(
        default_factory=dict,
        description="Row count per job type (jobs) or workflow name (executions)",
    )
    oldest: str | None = Field(
        default=None, description="Earliest creation/submit time (ISO 8601)"
    )
    newest: str | None = Field(
        default=None, description="Latest creation/submit time (ISO 8601)"
    )


class WorkflowExecutionListResponse(BaseModel):
    """``GET /umrs/workflows/executions`` (operation ``workflow_executions``)."""

    status: str = Field(..., description="'ok'")
    count: int = Field(..., description="Rows in this page (0 with summary=true)")
    executions: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Rows, newest first: execution_id, workflow_name, workflow_version, "
            "user_id, status, progress, submitted_at, started_at, finished_at "
            "(epoch seconds), error, steps_done, steps_total, durable_job_id; "
            "only the fields= names (and execution_id) when fields= is given"
        ),
    )
    next_cursor: str | None = Field(
        default=None,
        description="Pass as cursor= for the next page; null on the last page",
    )
    summary: ReadSummary | None = Field(
        default=None, description="Present with summary=true"
    )
