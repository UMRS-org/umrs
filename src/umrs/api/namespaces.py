"""UMRS API routes for knowledge namespace isolation (MEM-C4)."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/namespaces", tags=["namespaces"])

# Store instance — injected at startup
_store = None


def set_store(store) -> None:
    global _store
    _store = store


def _get_store():
    if _store is None:
        raise HTTPException(status_code=503, detail="Namespace store not initialized")
    return _store


@router.post("")
async def create_namespace(req: dict):
    """Create a knowledge namespace."""
    store = _get_store()
    ns = store.create_namespace(
        name=req["name"],
        scope=req["scope"],
        project_id=req.get("project_id"),
        experiment_id=req.get("experiment_id"),
        owner_id=req.get("owner_id"),
        description=req.get("description", ""),
    )
    return ns.to_dict()


@router.get("")
async def list_namespaces(
    project_id: str | None = Query(None),
    scope: str | None = Query(None),
    active_only: bool = Query(True),
):
    """List namespaces with optional filters."""
    store = _get_store()
    namespaces = store.list_namespaces(
        project_id=project_id,
        scope=scope,
        active_only=active_only,
    )
    return {"namespaces": [ns.to_dict() for ns in namespaces]}


@router.get("/{namespace_id}")
async def get_namespace(namespace_id: str):
    """Get a single namespace."""
    store = _get_store()
    ns = store.get_namespace(namespace_id)
    if ns is None:
        raise HTTPException(status_code=404, detail="Namespace not found")
    return ns.to_dict()


@router.delete("/{namespace_id}")
async def deactivate_namespace(namespace_id: str):
    """Soft-delete (deactivate) a namespace."""
    store = _get_store()
    ok = store.deactivate_namespace(namespace_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Namespace not found")
    return {"status": "deactivated", "namespace_id": namespace_id}


@router.post("/visibility")
async def create_visibility_policy(req: dict):
    """Create a namespace visibility policy for a job or workflow."""
    store = _get_store()
    policy = store.create_visibility_policy(
        job_id=req.get("job_id"),
        workflow_name=req.get("workflow_name"),
        visible_namespaces=req.get("visible_namespaces"),
        include_default=req.get("include_default", True),
        include_project=req.get("include_project", True),
        include_experiment=req.get("include_experiment", False),
    )
    return policy.to_dict()


@router.get("/visibility/resolve")
async def resolve_visibility(
    job_id: str | None = Query(None),
    workflow_name: str | None = Query(None),
    project_id: str | None = Query(None),
    experiment_id: str | None = Query(None),
):
    """Resolve visible namespaces for a job or workflow."""
    store = _get_store()
    namespaces = store.resolve_visible_namespaces(
        job_id=job_id,
        workflow_name=workflow_name,
        project_id=project_id,
        experiment_id=experiment_id,
    )
    return {"visible_namespaces": namespaces}
