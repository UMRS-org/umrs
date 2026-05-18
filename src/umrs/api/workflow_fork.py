"""UMRS API routes for workflow fork (MEM-B8)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/workflow-fork", tags=["workflow-fork"])

# Forker instance — injected at startup
_forker = None


def set_forker(forker) -> None:
    global _forker
    _forker = forker


def _get_forker():
    if _forker is None:
        raise HTTPException(status_code=503, detail="Workflow forker not initialized")
    return _forker


@router.post("")
async def fork_workflow(req: dict):
    """Fork a completed workflow run with modified parameters."""
    from ..jobs.workflow_fork import WorkflowForkError, WorkflowForkRequest

    forker = _get_forker()
    try:
        result = forker.fork(WorkflowForkRequest(
            source_job_id=req["source_job_id"],
            user_id=req["user_id"],
            param_overrides=req.get("param_overrides", {}),
            description=req.get("description", ""),
        ))
        return result.to_dict()
    except WorkflowForkError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/lineage/{job_id}")
async def get_fork_lineage(job_id: str):
    """Get the full fork lineage chain for a job."""
    from ..jobs.workflow_fork import WorkflowForkError

    forker = _get_forker()
    try:
        chain = forker.get_fork_lineage(job_id)
        return {"job_id": job_id, "lineage_chain": chain}
    except WorkflowForkError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
