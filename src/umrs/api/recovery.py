"""UMRS API routes for session crash recovery (MEM-A4)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/recovery", tags=["recovery"])

# Store instance — injected at startup
_recoverer = None


def set_recoverer(recoverer) -> None:
    global _recoverer
    _recoverer = recoverer


def _get_recoverer():
    if _recoverer is None:
        raise HTTPException(status_code=503, detail="Session recoverer not initialized")
    return _recoverer


@router.get("/sessions")
async def recover_sessions(job_id: str | None = None):
    """Recover in-flight session state from execution traces.

    If job_id is given, recovers only that job.
    Otherwise, finds all jobs with LOST attempts.
    """
    recoverer = _get_recoverer()
    job_ids = [job_id] if job_id else None
    result = recoverer.recover_sessions(lost_job_ids=job_ids)
    return result.to_dict()


@router.get("/sessions/{job_id}")
async def recover_single_session(job_id: str):
    """Recover session state for a specific job."""
    recoverer = _get_recoverer()
    result = recoverer.recover_sessions(lost_job_ids=[job_id])
    if not result.recovered_sessions:
        raise HTTPException(status_code=404, detail="No session traces found for job")
    return result.recovered_sessions[0].to_dict()
