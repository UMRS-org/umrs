"""UMRS API routes for vector versioning (MEM-C2)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/vectors", tags=["vectors"])

# Store instance — injected at startup
_store = None


def set_store(store) -> None:
    global _store
    _store = store


def _get_store():
    if _store is None:
        raise HTTPException(status_code=503, detail="Vector version store not initialized")
    return _store


@router.post("/versions")
async def register_version(req: dict):
    """Register a new embedding model version."""
    store = _get_store()
    version = store.register_version(
        model_name=req["model_name"],
        model_version=req["model_version"],
        model_path=req.get("model_path"),
        embedding_dimension=req.get("embedding_dimension", 0),
        set_active=req.get("set_active", True),
    )
    return version.to_dict()


@router.get("/versions")
async def list_versions(model_name: str | None = None):
    """List all registered embedding model versions."""
    store = _get_store()
    versions = store.list_versions(model_name)
    return {"versions": [v.to_dict() for v in versions]}


@router.get("/versions/active/{model_name}")
async def get_active_version(model_name: str):
    """Get the active version for a model."""
    store = _get_store()
    version = store.get_active_version(model_name)
    if version is None:
        raise HTTPException(status_code=404, detail="No active version found")
    return version.to_dict()


@router.get("/summary")
async def version_summary(model_name: str | None = None):
    """Get a summary of embedding model versions and vector counts."""
    store = _get_store()
    summary = store.get_version_summary(model_name)
    return summary.to_dict()


@router.post("/reembed")
async def create_reembed_job(req: dict):
    """Create a re-embedding job to migrate vectors between model versions."""
    store = _get_store()
    job = store.create_reembed_job(
        source_model=req["source_model"],
        source_version=req["source_version"],
        target_model=req["target_model"],
        target_version=req["target_version"],
        target_dimension=req.get("target_dimension", 0),
        job_id=req.get("job_id"),
    )
    return job.to_dict()


@router.get("/reembed")
async def list_reembed_jobs(status: str | None = None):
    """List re-embedding jobs, optionally filtered by status."""
    store = _get_store()
    jobs = store.list_reembed_jobs(status)
    return {"jobs": [j.to_dict() for j in jobs]}


@router.get("/reembed/{reembed_id}")
async def get_reembed_job(reembed_id: str):
    """Get a single re-embedding job."""
    store = _get_store()
    job = store.get_reembed_job(reembed_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Re-embed job not found")
    return job.to_dict()


@router.put("/reembed/{reembed_id}/progress")
async def update_reembed_progress(reembed_id: str, req: dict):
    """Update progress of a re-embedding job."""
    store = _get_store()
    job = store.update_reembed_progress(
        reembed_id=reembed_id,
        processed=req["processed_vectors"],
        failed=req.get("failed_vectors", 0),
        status=req.get("status"),
        error=req.get("error"),
    )
    if job is None:
        raise HTTPException(status_code=404, detail="Re-embed job not found")
    return job.to_dict()
