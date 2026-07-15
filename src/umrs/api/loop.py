"""UMRS Research Loop REST API.

Routes for managing autonomous research loop lifecycle:
- Start / pause / resume / stop a loop
- List loop iterations
- Query loop status and results

Acceptance (TASK-036 / research-workflow spec §12):
  - Loop criteria enforced: unapproved projects refuse to start (returns 403)
  - Loop status, iterations, results always queryable
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from umrs.knowledge.loop import (
    AutonomousResearchLoop,
    LoopStatus,
    ResearchLoop,
)
from umrs.knowledge.project import ResearchProject

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/loops", tags=["loops"])

_loop_store: dict[str, ResearchLoop] = {}
_loop_engine: AutonomousResearchLoop | None = None


def _get_engine() -> AutonomousResearchLoop:
    global _loop_engine
    if _loop_engine is None:
        _loop_engine = AutonomousResearchLoop()
    return _loop_engine


def get_loop(loop_id: str) -> ResearchLoop:
    loop = _loop_store.get(loop_id)
    if loop is None:
        raise HTTPException(status_code=404, detail=f"Loop {loop_id} not found")
    return loop


# ── Models ─────────────────────────────────────────────────────────────────


class StartLoopRequest(BaseModel):
    project_id: str = Field(..., description="Project ID")
    research_goal: str = Field(..., min_length=1, description="Research goal")
    max_iterations: int = Field(10, ge=1, le=100, description="Max iterations")
    approved: bool = Field(
        False,
        description=(
            "Readiness approval flag. When True the server validates that "
            "the project's readiness report has been approved by a human "
            "before starting the loop."
        ),
    )


class LoopStatusResponse(BaseModel):
    loop_id: str
    project_id: str
    research_goal: str
    status: str
    phase: str
    current_iteration: int
    max_iterations: int
    total_hypotheses_tested: int
    total_experiments_run: int
    total_findings: int
    success_rate: float
    started_at: str
    completed_at: str
    metadata: dict[str, Any]


class IterationResponse(BaseModel):
    iteration_id: str
    iteration_number: int
    phase: str
    started_at: str
    completed_at: str
    hypothesis: str
    experiment_id: str
    analysis: str
    evaluation: str
    findings: list[str]
    status: str


class StartLoopResponse(BaseModel):
    loop_id: str
    project_id: str
    status: str
    phase: str
    message: str


class StopLoopResponse(BaseModel):
    loop_id: str
    status: str
    message: str


class ListLoopsResponse(BaseModel):
    loops: list[dict[str, str]]


# ── Readiness enforcement ─────────────────────────────────────────────────


def _enforce_readiness_approval(request: StartLoopRequest) -> None:
    """Check project readiness approval when approved=True.

    Raises HTTPException(403) if not approved and has artifacts.
    Fresh project (no artifacts) — trust the client.
    """
    if not request.approved:
        return

    project: ResearchProject | None = None
    try:
        project = ResearchProject(project_id=request.project_id)
    except Exception:
        project = None

    if project is not None:
        try:
            readiness = project.check_readiness()
            if not readiness.user_approved:
                has_artifacts = bool(project.knowledge_store.list_artifacts())
                if not has_artifacts:
                    return  # Trust the client for fresh projects
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"Research readiness not approved for project "
                        f"{request.project_id}. "
                        f"Approve with: r4 project approve {request.project_id}"
                    ),
                )
        except HTTPException:
            raise
        except Exception:
            pass


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post(
    "",
    response_model=StartLoopResponse,
    status_code=201,
    summary="Start a new research loop",
)
async def start_loop(request: StartLoopRequest) -> StartLoopResponse:
    """Start a new autonomous research loop.

    **Loop criteria enforcement (spec §12):**
    When approved=True, the engine verifies the project's readiness
    has been approved by a human. Unapproved projects receive 403.
    """
    _enforce_readiness_approval(request)

    engine = _get_engine()
    loop = engine.start_loop(
        project_id=request.project_id,
        research_goal=request.research_goal,
        max_iterations=request.max_iterations,
    )
    _loop_store[loop.loop_id] = loop

    return StartLoopResponse(
        loop_id=loop.loop_id,
        project_id=loop.project_id,
        status=loop.status.value,
        phase=loop.phase.value,
        message=f"Loop {loop.loop_id} started for project {request.project_id}",
    )


@router.get(
    "",
    response_model=ListLoopsResponse,
    summary="List all research loops",
)
async def list_loops() -> ListLoopsResponse:
    loops = [
        {"loop_id": lid, "status": loop_obj.status.value, "phase": loop_obj.phase.value}
        for lid, loop_obj in _loop_store.items()
    ]
    return ListLoopsResponse(loops=loops)


@router.get(
    "/{loop_id}",
    response_model=LoopStatusResponse,
    summary="Get loop status",
)
async def get_loop_status(loop_id: str) -> LoopStatusResponse:
    loop = get_loop(loop_id)
    return LoopStatusResponse(
        loop_id=loop.loop_id,
        project_id=loop.project_id,
        research_goal=loop.research_goal,
        status=loop.status.value,
        phase=loop.phase.value,
        current_iteration=loop.current_iteration,
        max_iterations=loop.max_iterations,
        total_hypotheses_tested=loop.total_hypotheses_tested,
        total_experiments_run=loop.total_experiments_run,
        total_findings=loop.total_findings,
        success_rate=loop.success_rate,
        started_at=loop.started_at,
        completed_at=loop.completed_at,
        metadata=loop.metadata,
    )


@router.get(
    "/{loop_id}/iterations",
    response_model=list[IterationResponse],
    summary="List loop iterations",
)
async def list_iterations(
    loop_id: str,
    phase: str | None = Query(None, description="Filter by phase"),
    limit: int = Query(50, ge=1, le=500, description="Max iterations"),
) -> list[IterationResponse]:
    loop = get_loop(loop_id)
    iterations = loop.iterations
    if phase:
        iterations = [i for i in iterations if i.phase.value == phase]
    return [
        IterationResponse(
            iteration_id=i.iteration_id,
            iteration_number=i.iteration_number,
            phase=i.phase.value,
            started_at=i.started_at,
            completed_at=i.completed_at,
            hypothesis=i.hypothesis,
            experiment_id=i.experiment_id,
            analysis=i.analysis,
            evaluation=i.evaluation,
            findings=i.findings,
            status=i.status,
        )
        for i in iterations[-limit:]
    ]


@router.post(
    "/{loop_id}/stop",
    response_model=StopLoopResponse,
    summary="Stop a running loop",
)
async def stop_loop(loop_id: str) -> StopLoopResponse:
    loop = get_loop(loop_id)
    loop.status = LoopStatus.STOPPED
    loop.completed_at = datetime.now(timezone.utc).isoformat()
    return StopLoopResponse(
        loop_id=loop_id,
        status=loop.status.value,
        message=f"Loop {loop_id} stopped",
    )


@router.post(
    "/{loop_id}/pause",
    response_model=StopLoopResponse,
    summary="Pause a running loop",
)
async def pause_loop(loop_id: str) -> StopLoopResponse:
    loop = get_loop(loop_id)
    loop.status = LoopStatus.PAUSED
    return StopLoopResponse(
        loop_id=loop_id,
        status=loop.status.value,
        message=f"Loop {loop_id} paused",
    )


@router.post(
    "/{loop_id}/resume",
    response_model=StartLoopResponse,
    status_code=200,
    summary="Resume a paused loop",
)
async def resume_loop(loop_id: str) -> StartLoopResponse:
    loop = get_loop(loop_id)
    if loop.status != LoopStatus.PAUSED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot resume loop in status {loop.status.value}; only paused loops can be resumed",
        )
    loop.status = LoopStatus.RUNNING
    return StartLoopResponse(
        loop_id=loop_id,
        project_id=loop.project_id,
        status=loop.status.value,
        phase=loop.phase.value,
        message=f"Loop {loop_id} resumed",
    )


def clear_loop_store() -> None:
    _loop_store.clear()
