"""UMRS API routes for the knowledge graph registry (multi-graph Phase 1/3).

Routes (mounted under the gateway ``/gw/knowledge/`` prefix — no nginx edits):

  GET   /umrs/knowledge/graphs            — list (project/status filters, archive-aware)
  POST  /umrs/knowledge/graphs            — create a named graph
  GET   /umrs/knowledge/graphs/{graph_id} — graph detail
  PATCH /umrs/knowledge/graphs/{graph_id} — metadata update / archive (no hard delete)

Backward compatibility: listing with ``project_id`` materializes the project's
deterministic default graph first, so existing projects appear automatically.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from umrs.knowledge.graph_registry import GRAPH_STATUSES, GraphRegistryStore

from .jobs import CallerIdentity, get_caller_identity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge/graphs", tags=["knowledge-graphs"])


def _registry(request: Request) -> GraphRegistryStore:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return GraphRegistryStore(pool)


class CreateGraphRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    project_id: Optional[str] = None
    kind: str = "research"
    ontology_id: Optional[str] = None
    ontology_version: Optional[str] = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class UpdateGraphRequest(BaseModel):
    description: Optional[str] = None
    kind: Optional[str] = None
    ontology_id: Optional[str] = None
    ontology_version: Optional[str] = None
    status: Optional[str] = Field(
        default=None, description="Lifecycle: active | archived"
    )


@router.get("")
async def list_graphs(
    request: Request,
    project_id: Optional[str] = Query(None),
    status: Optional[str] = Query("active"),
    q: Optional[str] = Query(None, description="Name/description search"),
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    """List registered knowledge graphs with optional filters.

    When ``project_id`` is given, its default graph is materialized first
    so legacy projects are listed without any manual migration.
    """
    registry = _registry(request)
    if project_id:
        registry.ensure_default_graph(project_id)
    if q:
        graphs = registry.search_graphs(q, project_id=project_id, limit=limit)
        if status:
            graphs = [g for g in graphs if g.status == status]
    else:
        graphs = registry.list_graphs(project_id=project_id, status=status, limit=limit)
    return {"graphs": [g.to_dict() for g in graphs], "total": len(graphs)}


@router.post("", status_code=201)
async def create_graph(
    body: CreateGraphRequest,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Create a named, independent knowledge graph."""
    registry = _registry(request)
    graph = registry.create_graph(
        body.name,
        description=body.description,
        owner_id=identity.user_id or "",
        project_id=body.project_id,
        kind=body.kind,
        ontology_id=body.ontology_id,
        ontology_version=body.ontology_version,
        provenance=body.provenance,
    )
    return graph.to_dict()


@router.get("/{graph_id}")
async def get_graph(graph_id: str, request: Request) -> dict[str, Any]:
    """Fetch one graph by id (or resolve by exact name)."""
    registry = _registry(request)
    graph = registry.resolve_graph(graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")
    return graph.to_dict()


@router.patch("/{graph_id}")
async def update_graph(
    graph_id: str,
    body: UpdateGraphRequest,
    request: Request,
) -> dict[str, Any]:
    """Update metadata or lifecycle status. Archiving is the only 'delete'."""
    registry = _registry(request)
    if registry.get_graph(graph_id) is None:
        raise HTTPException(status_code=404, detail=f"Graph {graph_id} not found")
    if body.status is not None:
        if body.status not in GRAPH_STATUSES:
            raise HTTPException(
                status_code=400, detail=f"Invalid status {body.status!r}"
            )
        registry.set_status(graph_id, body.status)
    graph = registry.update_graph(
        graph_id,
        description=body.description,
        kind=body.kind,
        ontology_id=body.ontology_id,
        ontology_version=body.ontology_version,
    )
    assert graph is not None
    return graph.to_dict()
