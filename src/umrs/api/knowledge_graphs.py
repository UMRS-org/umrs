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

from .typed_errors import PlatformError, raise_typed
from pydantic import BaseModel, Field

from umrs.knowledge.access_scope import (
    CallerScope,
    graph_readable,
    graph_writable,
    project_writable,
)
from umrs.knowledge.graph_registry import GRAPH_STATUSES, GraphRegistryStore

from ._knowledge_mutation_guard import enforce_registry_write_authorization
from ._scope_identity import get_caller_scope
from .jobs import CallerIdentity, get_caller_identity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge/graphs", tags=["knowledge-graphs"])


def _registry(request: Request) -> GraphRegistryStore:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.knowledgegraphs("unavailable", "Database unavailable"))
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
    scope = get_caller_scope(request)
    visible = [
        g
        for g in graphs
        if graph_readable(
            kind=g.kind, project_id=g.project_id, owner_id=g.owner_id, caller=scope
        )
    ]
    return {"graphs": [g.to_dict() for g in visible], "total": len(visible)}


@router.post("", status_code=201)
async def create_graph(
    body: CreateGraphRequest,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Create a named, independent knowledge graph.

    Three-tier boundary: system graphs are admin-only; user/personal graphs
    are normalized to the caller's personal namespace; project-like graphs
    require the caller to belong to the target project.
    """
    enforce_registry_write_authorization(request)
    registry = _registry(request)
    scope = CallerScope(
        user_id=identity.user_id or "",
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    project_id = body.project_id
    if not graph_writable(
        kind=body.kind,
        project_id=project_id if project_id is not None else scope.project_id,
        owner_id=scope.user_id,
        caller=scope,
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_denied",
                "message": (
                    f"Cannot create a '{body.kind}' graph in this namespace — "
                    "system graphs are admin-only, project graphs require "
                    "project membership, personal graphs are owner-only."
                ),
            },
        )
    kind_reduced = (body.kind or "").strip().lower()
    if kind_reduced in ("user", "personal"):
        project_id = None  # owner-only graphs are not project-bound
    graph = registry.create_graph(
        body.name,
        description=body.description,
        owner_id=identity.user_id or "",
        project_id=project_id,
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
    scope = get_caller_scope(request)
    if graph is None or not graph_readable(
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        caller=scope,
    ):
        raise_typed(PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found"))
    return graph.to_dict()


@router.get("/{graph_id}/content")
async def get_graph_content(
    graph_id: str,
    request: Request,
    node_limit: int = Query(200, ge=1, le=1000),
    edge_limit: int = Query(400, ge=1, le=2000),
) -> dict[str, Any]:
    """Bounded nodes+edges for one registered graph (webview visualization).

    Read-only; no mutation guard. ``graph_id`` may be an id or exact name
    (same resolution as the detail route). 404 when the graph is unknown
    or outside the caller's tier (existence is not disclosed).
    """
    registry = _registry(request)
    try:
        content = registry.graph_content(
            graph_id, node_limit=node_limit, edge_limit=edge_limit
        )
    except KeyError:
        raise_typed(PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found"))
    scope = get_caller_scope(request)
    if not graph_readable(
        kind=str(content.get("kind") or ""),
        project_id=content.get("project_id")
        if isinstance(content.get("project_id"), str)
        else None,
        owner_id=str(content.get("owner_id") or ""),
        caller=scope,
    ):
        raise_typed(PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found"))
    return content


@router.patch("/{graph_id}")
async def update_graph(
    graph_id: str,
    body: UpdateGraphRequest,
    request: Request,
) -> dict[str, Any]:
    """Update metadata or lifecycle status. Archiving is the only 'delete'.

    Tier rule: system graphs are admin-managed, user graphs owner-only,
    project graphs archive-able by project members.
    """
    enforce_registry_write_authorization(request)
    registry = _registry(request)
    existing = registry.get_graph(graph_id)
    if existing is None:
        raise_typed(PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found"))
    scope = get_caller_scope(request)
    if not graph_writable(
        kind=existing.kind,
        project_id=existing.project_id,
        owner_id=existing.owner_id,
        caller=scope,
    ):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_denied",
                "message": "Graph lifecycle is outside the caller's access tier.",
            },
        )
    if body.status is not None:
        if body.status not in GRAPH_STATUSES:
            raise_typed(
                PlatformError.knowledgegraphs("bad_request", f"Invalid status {body.status!r}")
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
