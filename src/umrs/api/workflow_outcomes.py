"""Workflow outcome REST API (MEM-D1).

Provides:
- GET  /umrs/workflows/outcomes                          List outcomes (with filters)
- GET  /umrs/workflows/outcomes/{outcome_id}             Get single outcome
- GET  /umrs/workflows/{workflow_name}/outcomes/summary   Aggregate summary for template
- GET  /umrs/workflows/outcomes/by-model/{model_name}    Summary by model
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..jobs.workflow_outcome import (
    OutcomeSummary,
    StepMetric,
    WorkflowOutcome,
    WorkflowOutcomeStore,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/workflows", tags=["workflow-outcomes"])


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_outcome_store: WorkflowOutcomeStore | None = None


def get_outcome_store() -> WorkflowOutcomeStore:
    global _outcome_store
    if _outcome_store is None:
        db_url = os.environ.get("DATABASE_URL") or os.environ.get("UMRS_DB_URL")
        _outcome_store = WorkflowOutcomeStore(db_url=db_url)
    return _outcome_store


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class StepMetricResponse(BaseModel):
    step_name: str
    status: str = ""
    duration_sec: float | None = None
    model: str | None = None
    tokens: int | None = None


class OutcomeResponse(BaseModel):
    outcome_id: str
    job_id: str
    workflow_name: str
    workflow_version: str | None = None
    user_id: str
    project_id: str
    status: str
    execution_duration_sec: float | None = None
    step_count: int | None = None
    failed_step_name: str | None = None
    model_used: str | None = None
    model_version: str | None = None
    embedding_model: str | None = None
    total_tokens: int | None = None
    tool_call_count: int | None = None
    retrieval_doc_count: int | None = None
    resource_usage: dict[str, Any] = Field(default_factory=dict)
    failure_reason: str | None = None
    failure_category: str | None = None
    step_metrics: list[StepMetricResponse] = Field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str | None = None


class OutcomeSummaryResponse(BaseModel):
    workflow_name: str
    total_runs: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    success_rate: float = 0.0
    avg_duration_sec: float | None = None
    avg_tokens: float | None = None
    failure_breakdown: dict[str, int] = Field(default_factory=dict)


def _to_response(outcome: WorkflowOutcome) -> OutcomeResponse:
    d = outcome.to_dict()
    return OutcomeResponse(**d)


def _summary_to_response(s: OutcomeSummary) -> OutcomeSummaryResponse:
    return OutcomeSummaryResponse(**s.to_dict())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/outcomes")
async def list_outcomes(
    workflow_name: str | None = Query(None),
    model_used: str | None = Query(None),
    user_id: str | None = Query(None),
    outcome_status: str | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=500),
    store: WorkflowOutcomeStore = Depends(get_outcome_store),
) -> list[OutcomeResponse]:
    """List workflow outcomes with optional filters."""
    outcomes = store.list_outcomes(
        workflow_name=workflow_name,
        model_used=model_used,
        user_id=user_id,
        status=outcome_status,
        limit=limit,
    )
    return [_to_response(o) for o in outcomes]


@router.get("/outcomes/{outcome_id}")
async def get_outcome(
    outcome_id: str,
    store: WorkflowOutcomeStore = Depends(get_outcome_store),
) -> OutcomeResponse:
    """Retrieve a specific workflow outcome."""
    outcome = store.get_outcome(outcome_id)
    if not outcome:
        raise HTTPException(status_code=404, detail="Outcome not found")
    return _to_response(outcome)


@router.get("/{workflow_name}/outcomes/summary")
async def workflow_summary(
    workflow_name: str,
    store: WorkflowOutcomeStore = Depends(get_outcome_store),
) -> OutcomeSummaryResponse:
    """Aggregate statistics for a workflow template."""
    summary = store.summarize(workflow_name)
    return _summary_to_response(summary)


@router.get("/outcomes/by-model/{model_name}")
async def model_summary(
    model_name: str,
    store: WorkflowOutcomeStore = Depends(get_outcome_store),
) -> OutcomeSummaryResponse:
    """Aggregate statistics for all workflows using a specific model."""
    summary = store.summarize_by_model(model_name)
    return _summary_to_response(summary)
