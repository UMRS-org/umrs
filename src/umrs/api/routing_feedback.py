"""Routing feedback REST API (MEM-D2).

Provides:
- POST  /umrs/routing-feedback                            Record feedback
- PATCH /umrs/routing-feedback/{feedback_id}               Update user feedback
- GET   /umrs/routing-feedback                            List feedback
- GET   /umrs/routing-feedback/{feedback_id}               Get single feedback
- GET   /umrs/routing-feedback/summary/{model_name}        Model routing summary
- DELETE /umrs/routing-feedback/{feedback_id}              Delete feedback
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from ..jobs.routing_feedback import RoutingFeedbackStore
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/routing-feedback", tags=["routing-feedback"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_store: RoutingFeedbackStore | None = None


def get_store() -> RoutingFeedbackStore:
    global _store
    if _store is None:
        db_url = os.environ.get("DATABASE_URL") or os.environ.get("UMRS_DB_URL")
        _store = RoutingFeedbackStore(db_url=db_url)
    return _store


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RecordFeedbackRequest(BaseModel):
    job_id: str
    step_name: str
    model_selected: str
    outcome: str = ""
    user_accepted: bool | None = None
    user_correction: str | None = None
    step_type: str = ""
    routing_reason: str = ""
    latency_ms: float | None = None
    tokens_used: int | None = None
    quality_score: float | None = None
    project_id: str = ""
    user_id: str = ""


class UpdateFeedbackRequest(BaseModel):
    user_accepted: bool
    user_correction: str | None = None
    quality_score: float | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    job_id: str
    step_name: str
    model_selected: str
    outcome: str = ""
    user_accepted: bool | None = None
    user_correction: str | None = None
    step_type: str = ""
    routing_reason: str = ""
    latency_ms: float | None = None
    tokens_used: int | None = None
    quality_score: float | None = None
    project_id: str = ""
    user_id: str = ""
    created_at: str | None = None


class SummaryResponse(BaseModel):
    model_name: str
    total_selections: int = 0
    succeeded: int = 0
    failed: int = 0
    user_accepted: int = 0
    user_rejected: int = 0
    user_corrected: int = 0
    avg_latency_ms: float | None = None
    avg_quality_score: float | None = None
    acceptance_rate: float = 0.0
    correction_targets: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
def record_feedback(req: RecordFeedbackRequest):
    store = get_store()
    fb = store.record_feedback(
        job_id=req.job_id,
        step_name=req.step_name,
        model_selected=req.model_selected,
        outcome=req.outcome,
        user_accepted=req.user_accepted,
        user_correction=req.user_correction,
        step_type=req.step_type,
        routing_reason=req.routing_reason,
        latency_ms=req.latency_ms,
        tokens_used=req.tokens_used,
        quality_score=req.quality_score,
        project_id=req.project_id,
        user_id=req.user_id,
    )
    return FeedbackResponse(**fb.to_dict())


@router.patch("/{feedback_id}")
def update_feedback(feedback_id: str, req: UpdateFeedbackRequest):
    store = get_store()
    ok = store.update_user_feedback(
        feedback_id=feedback_id,
        user_accepted=req.user_accepted,
        user_correction=req.user_correction,
        quality_score=req.quality_score,
    )
    if not ok:
        raise_typed(PlatformError.routing("not_found", "Feedback not found"))
    return {"status": "updated", "feedback_id": feedback_id}


@router.get("", response_model=list[FeedbackResponse])
def list_feedback(
    model_name: str | None = Query(None),
    job_id: str | None = Query(None),
    step_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    store = get_store()
    results = store.list_feedback(
        model_name=model_name,
        job_id=job_id,
        step_type=step_type,
        limit=limit,
    )
    return [FeedbackResponse(**f.to_dict()) for f in results]


@router.get("/summary/{model_name}", response_model=SummaryResponse)
def model_summary(model_name: str):
    store = get_store()
    s = store.summarize_model(model_name)
    return SummaryResponse(**s.to_dict())


@router.get("/{feedback_id}", response_model=FeedbackResponse)
def get_feedback(feedback_id: str):
    store = get_store()
    fb = store.get_feedback(feedback_id)
    if not fb:
        raise_typed(PlatformError.routing("not_found", "Feedback not found"))
    return FeedbackResponse(**fb.to_dict())


@router.delete("/{feedback_id}")
def delete_feedback(feedback_id: str):
    store = get_store()
    ok = store.delete_feedback(feedback_id)
    if not ok:
        raise_typed(PlatformError.routing("not_found", "Feedback not found"))
    return {"status": "deleted", "feedback_id": feedback_id}
