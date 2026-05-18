"""UMRS API routes for session-scoped tool output cache (MEM-A3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

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
async def put_cached_output(req: dict):
    """Cache a tool output for a session."""
    store = _get_store()
    entry = store.put(
        session_id=req["session_id"],
        tool_name=req["tool_name"],
        inputs=req.get("inputs"),
        output=req.get("output"),
        output_ref=req.get("output_ref"),
        ttl_seconds=req.get("ttl_seconds", 300),
    )
    return entry.to_dict()


@router.get("/session/{session_id}")
async def list_session_cache(session_id: str):
    """List all cached outputs for a session."""
    store = _get_store()
    entries = store.list_session(session_id)
    return {"entries": [e.to_dict() for e in entries]}


@router.get("/session/{session_id}/lookup")
async def lookup_cached_output(
    session_id: str, tool_name: str, inputs_hash: str
):
    """Look up a specific cached output by inputs hash."""
    store = _get_store()
    # Direct hash lookup (bypass re-hashing)
    from umrs.jobs.tool_output_cache import _compute_inputs_hash

    entry = store.get(session_id, tool_name, {"_raw_hash": inputs_hash})
    if entry is None:
        raise HTTPException(status_code=404, detail="Cache miss")
    return entry.to_dict()


@router.delete("/session/{session_id}")
async def invalidate_session_cache(
    session_id: str, tool_name: str | None = None
):
    """Invalidate cached entries for a session."""
    store = _get_store()
    count = store.invalidate(session_id, tool_name)
    return {"invalidated": count}


@router.post("/cleanup")
async def cleanup_expired():
    """Remove all expired cache entries."""
    store = _get_store()
    count = store.cleanup_expired()
    return {"cleaned": count}
