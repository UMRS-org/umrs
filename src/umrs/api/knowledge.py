"""UMRS API routes for cross-experiment knowledge search (MEM-C5)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge"])

# Engine instance — injected at startup
_engine = None


def set_engine(engine) -> None:
    global _engine
    _engine = engine


def _get_engine():
    if _engine is None:
        raise HTTPException(
            status_code=503, detail="Knowledge search engine not initialized"
        )
    return _engine


@router.get("/search")
async def search_knowledge(
    query: str = Query(..., description="Search query"),
    scope: str = Query("project", description="Search scope: project|experiment|user"),
    project_id: str | None = Query(None),
    namespaces: str | None = Query(None, description="Comma-separated namespace names"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search across knowledge namespaces with source attribution."""
    engine = _get_engine()
    ns_list = (
        [n.strip() for n in namespaces.split(",") if n.strip()] if namespaces else None
    )
    result = engine.search(
        query=query,
        scope=scope,
        project_id=project_id,
        namespaces=ns_list,
        limit=limit,
    )
    return result.to_dict()
