"""Step replay REST API (MEM-B5).

Provides:
- POST /umrs/replay                         Create a replay from a trace
- GET  /umrs/replay/{replay_id}             Get replay result
- GET  /umrs/replay                         List replays
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..jobs.execution_trace_store import ExecutionTraceStore
from ..jobs.step_replay import ReplayRequest, StepReplayEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/replay", tags=["replay"])


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_engine: StepReplayEngine | None = None


def get_engine() -> StepReplayEngine:
    global _engine
    if _engine is None:
        db_url = os.environ.get("DATABASE_URL") or os.environ.get("UMRS_DB_URL")
        trace_store = ExecutionTraceStore(db_url=db_url)
        _engine = StepReplayEngine(trace_store)
    return _engine


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ReplayCreateRequest(BaseModel):
    source_trace_id: str
    user_id: str
    reason: str = ""
    param_overrides: dict[str, Any] | None = None
    new_job_id: str | None = None


class ReplayResponse(BaseModel):
    replay_id: str
    source_trace_id: str
    new_trace_id: str
    new_job_id: str
    step_name: str
    inputs_hash: str | None = None
    inputs_snapshot: dict[str, Any] | None = None
    inputs_ref: str | None = None
    param_overrides: dict[str, Any] | None = None
    user_id: str = ""
    reason: str = ""
    status: str = "created"
    created_at: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=ReplayResponse, status_code=status.HTTP_201_CREATED)
def create_replay(req: ReplayCreateRequest):
    engine = get_engine()
    result = engine.replay(ReplayRequest(
        source_trace_id=req.source_trace_id,
        user_id=req.user_id,
        reason=req.reason,
        param_overrides=req.param_overrides,
        new_job_id=req.new_job_id,
    ))
    return ReplayResponse(**result.to_dict())


@router.get("", response_model=list[ReplayResponse])
def list_replays(
    source_trace_id: str | None = Query(None),
    user_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    engine = get_engine()
    results = engine.list_replays(
        source_trace_id=source_trace_id,
        user_id=user_id,
        limit=limit,
    )
    return [ReplayResponse(**r.to_dict()) for r in results]


@router.get("/{replay_id}", response_model=ReplayResponse)
def get_replay(replay_id: str):
    engine = get_engine()
    r = engine.get_replay(replay_id)
    if not r:
        raise HTTPException(status_code=404, detail="Replay not found")
    return ReplayResponse(**r.to_dict())
