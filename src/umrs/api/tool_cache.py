"""UMRS API routes for session-scoped tool output cache (MEM-A3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from .endpoints import _caller_is_admin
from .files import get_project_id

logger = logging.getLogger(__name__)


def _namespaced(project_id: str, session_id: str) -> str:
    """Project-scoped session namespace.

    Previously the session_id alone formed the cache key: knowledge of one
    cross-tenant session_id granted read/wipe rights over that session's
    cached tool outputs (audit HIGH). Binding the key to the authenticated
    project is the UMRS-side fix; the gateway already forwards verified
    project headers (wave-2 gateway close).
    """
    return f"{project_id}\x1f{session_id}"


router = APIRouter(prefix="/umrs/tool-cache", tags=["tool-cache"])

# Store instance — injected at startup
_store = None


def set_store(store) -> None:
    global _store
    _store = store


def _get_store():
    if _store is None:
        raise HTTPException(status_code=503, detail="Tool output cache not initialized")
    return _store


@dataclass
class PutCacheRequest:
    session_id: str
    tool_name: str
    inputs: Any
    output: Any
    output_ref: str | None = None
    ttl_seconds: int = 300


@router.post("")
async def put_cached_output(
    req: dict,
    project_id: str = Depends(get_project_id),
):
    """Cache a tool output for a session (project-namespaced)."""
    store = _get_store()
    entry = store.put(
        session_id=_namespaced(project_id, req["session_id"]),
        tool_name=req["tool_name"],
        inputs=req.get("inputs"),
        output=req.get("output"),
        output_ref=req.get("output_ref"),
        ttl_seconds=req.get("ttl_seconds", 300),
    )
    return entry.to_dict()


@router.get("/session/{session_id}")
async def list_session_cache(
    session_id: str,
    project_id: str = Depends(get_project_id),
):
    """List cached outputs for a session in the caller's project."""
    store = _get_store()
    entries = store.list_session(_namespaced(project_id, session_id))
    return {"entries": [e.to_dict() for e in entries]}


@router.get("/session/{session_id}/lookup")
async def lookup_cached_output(
    session_id: str,
    tool_name: str,
    inputs_hash: str,
    project_id: str = Depends(get_project_id),
):
    """Look up a cached output within the caller's project namespace."""
    store = _get_store()
    # Direct hash lookup (bypass re-hashing)
    from umrs.jobs.tool_output_cache import _compute_inputs_hash

    entry = store.get(
        _namespaced(project_id, session_id),
        tool_name,
        {"_raw_hash": inputs_hash},
    )
    if entry is None:
        raise HTTPException(status_code=404, detail="Cache miss")
    return entry.to_dict()


@router.delete("/session/{session_id}")
async def invalidate_session_cache(
    session_id: str,
    tool_name: str | None = None,
    project_id: str = Depends(get_project_id),
):
    """Invalidate cached entries only within the caller's project."""
    store = _get_store()
    count = store.invalidate(_namespaced(project_id, session_id), tool_name)
    return {"invalidated": count}


@router.post("/cleanup")
async def cleanup_expired(is_admin: bool = Depends(_caller_is_admin)):
    """Remove all expired cache entries (admin-only: cross-project op)."""
    if not is_admin:
        raise HTTPException(
            status_code=403,
            detail="cleanup requires admin role (cross-project operation)",
        )
    store = _get_store()
    count = store.cleanup_expired()
    return {"cleaned": count}
