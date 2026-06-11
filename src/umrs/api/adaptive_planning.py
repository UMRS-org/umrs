"""Adaptive planning REST API (MEM-D3).

Provides:
- GET   /umrs/planning/suggest/{workflow_name}    Get recommendations
- POST  /umrs/planning/heuristics                 Store a heuristic
- GET   /umrs/planning/heuristics                 List heuristics
- PATCH /umrs/planning/heuristics/{id}/deactivate Deactivate a heuristic
- DELETE /umrs/planning/heuristics/{id}           Delete a heuristic
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..jobs.adaptive_planning import AdaptivePlanningEngine
from ..jobs.routing_feedback import RoutingFeedbackStore
from ..jobs.workflow_outcome import WorkflowOutcomeStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/planning", tags=["planning"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_engine: AdaptivePlanningEngine | None = None


def get_engine() -> AdaptivePlanningEngine:
    global _engine
    if _engine is None:
        db_url = os.environ.get("DATABASE_URL") or os.environ.get("UMRS_DB_URL")
        outcome_store = WorkflowOutcomeStore(db_url=db_url)
        routing_store = RoutingFeedbackStore(db_url=db_url)
        _engine = AdaptivePlanningEngine(outcome_store, routing_store)
    return _engine


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RecommendationResponse(BaseModel):
    recommendation_id: str
    category: str
    confidence: float
    description: str
    current_value: Any = None
    suggested_value: Any = None
    evidence: dict[str, Any] = Field(default_factory=dict)


class SuggestionResponse(BaseModel):
    workflow_name: str
    recommendations: list[RecommendationResponse] = Field(default_factory=list)
    heuristics_used: list[str] = Field(default_factory=list)
    outcome_count: int = 0
    generated_at: str | None = None


class StoreHeuristicRequest(BaseModel):
    workflow_name: str
    rule_type: str
    condition: str
    action: str
    confidence: float = 0.0
    sample_count: int = 0


class HeuristicResponse(BaseModel):
    heuristic_id: str
    workflow_name: str
    rule_type: str
    condition: str
    action: str
    confidence: float = 0.0
    sample_count: int = 0
    is_active: bool = True
    created_at: str | None = None
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/suggest/{workflow_name}", response_model=SuggestionResponse)
def suggest(workflow_name: str):
    engine = get_engine()
    result = engine.suggest(workflow_name)
    return SuggestionResponse(**result.to_dict())


@router.post(
    "/heuristics", response_model=HeuristicResponse, status_code=status.HTTP_201_CREATED
)
def store_heuristic(req: StoreHeuristicRequest):
    engine = get_engine()
    h = engine.store_heuristic(
        workflow_name=req.workflow_name,
        rule_type=req.rule_type,
        condition=req.condition,
        action=req.action,
        confidence=req.confidence,
        sample_count=req.sample_count,
    )
    return HeuristicResponse(**h.to_dict())


@router.get("/heuristics", response_model=list[HeuristicResponse])
def list_heuristics(workflow_name: str | None = Query(None)):
    engine = get_engine()
    results = engine.list_heuristics(workflow_name=workflow_name)
    return [HeuristicResponse(**h.to_dict()) for h in results]


@router.patch("/heuristics/{heuristic_id}/deactivate")
def deactivate_heuristic(heuristic_id: str):
    engine = get_engine()
    ok = engine.deactivate_heuristic(heuristic_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Heuristic not found")
    return {"status": "deactivated", "heuristic_id": heuristic_id}


@router.delete("/heuristics/{heuristic_id}")
def delete_heuristic(heuristic_id: str):
    engine = get_engine()
    ok = engine.delete_heuristic(heuristic_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Heuristic not found")
    return {"status": "deleted", "heuristic_id": heuristic_id}
