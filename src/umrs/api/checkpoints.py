"""Checkpoint REST API (MEM-B6).

Provides:
- POST /umrs/checkpoints                             Save a checkpoint
- GET  /umrs/checkpoints/{checkpoint_id}              Get a checkpoint
- GET  /umrs/checkpoints/job/{job_id}                 List checkpoints for a job
- GET  /umrs/checkpoints/job/{job_id}/latest          Latest checkpoint for resumption
- DELETE /umrs/checkpoints/job/{job_id}               Delete checkpoints for a job
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from .typed_errors import PlatformError, raise_typed

from ..jobs.checkpoint_store import Checkpoint, CheckpointStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/checkpoints", tags=["checkpoints"])


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_checkpoint_store: CheckpointStore | None = None


def get_checkpoint_store() -> CheckpointStore:
    global _checkpoint_store
    if _checkpoint_store is None:
        db_url = os.environ.get("DATABASE_URL") or os.environ.get("UMRS_DB_URL")
        _checkpoint_store = CheckpointStore(db_url=db_url)
    return _checkpoint_store


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SaveCheckpointRequest(BaseModel):
    job_id: str
    attempt_id: str
    step_name: str
    state_uri: str | None = None
    state_inline: dict[str, Any] | None = None
    state_size_bytes: int = 0
    iteration: int = 0
    progress: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CheckpointResponse(BaseModel):
    checkpoint_id: str
    job_id: str
    attempt_id: str
    step_name: str
    sequence_no: int
    state_uri: str | None = None
    state_size_bytes: int = 0
    state_inline: dict[str, Any] | None = None
    iteration: int = 0
    progress: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


def _to_response(ckpt: Checkpoint) -> CheckpointResponse:
    d = ckpt.to_dict()
    return CheckpointResponse(**d)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_201_CREATED)
async def save_checkpoint(
    req: SaveCheckpointRequest,
    store: CheckpointStore = Depends(get_checkpoint_store),
) -> CheckpointResponse:
    """Save a checkpoint for a long-running step."""
    ckpt = store.save_checkpoint(
        job_id=req.job_id,
        attempt_id=req.attempt_id,
        step_name=req.step_name,
        state_uri=req.state_uri,
        state_inline=req.state_inline,
        state_size_bytes=req.state_size_bytes,
        iteration=req.iteration,
        progress=req.progress,
        metadata=req.metadata,
    )
    return _to_response(ckpt)


@router.get("/{checkpoint_id}")
async def get_checkpoint(
    checkpoint_id: str,
    store: CheckpointStore = Depends(get_checkpoint_store),
) -> CheckpointResponse:
    """Retrieve a specific checkpoint."""
    ckpt = store.get_checkpoint(checkpoint_id)
    if not ckpt:
        raise_typed(PlatformError.checkpoints("not_found", "Checkpoint not found"))
    return _to_response(ckpt)


@router.get("/job/{job_id}")
async def list_checkpoints(
    job_id: str,
    attempt_id: str | None = Query(None),
    store: CheckpointStore = Depends(get_checkpoint_store),
) -> list[CheckpointResponse]:
    """List checkpoints for a job, optionally filtered by attempt."""
    ckpts = store.list_checkpoints(job_id, attempt_id)
    return [_to_response(c) for c in ckpts]


@router.get("/job/{job_id}/latest")
async def latest_checkpoint(
    job_id: str,
    store: CheckpointStore = Depends(get_checkpoint_store),
) -> CheckpointResponse:
    """Get the latest checkpoint for job resumption."""
    ckpt = store.select_resume_checkpoint(job_id)
    if not ckpt:
        raise_typed(PlatformError.checkpoints("not_found", "No checkpoints found for job"))
    return _to_response(ckpt)


@router.delete("/job/{job_id}", status_code=status.HTTP_200_OK)
async def delete_checkpoints(
    job_id: str,
    attempt_id: str | None = Query(None),
    store: CheckpointStore = Depends(get_checkpoint_store),
) -> dict[str, Any]:
    """Delete checkpoints for a job."""
    count = store.delete_checkpoints(job_id, attempt_id)
    return {"deleted": count, "job_id": job_id}
