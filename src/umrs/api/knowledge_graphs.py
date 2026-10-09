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
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from umrs.knowledge.store import KnowledgeStore

from umrs.knowledge.access_scope import (
    CallerScope,
    graph_readable,
    graph_writable,
)
from umrs.knowledge.graph_registry import GRAPH_STATUSES, GraphRegistryStore
from umrs.shared.access_scopes import access_admits

from ._access import checked_access, resource_caller
from ._access import graph_read_allowed as _graph_read_allowed
from ._knowledge_mutation_guard import (
    AGENT_ACTOR,
    enforce_registry_write_authorization,
    request_actor,
)
from ._scope_identity import get_caller_scope
from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge/graphs", tags=["knowledge-graphs"])


def _registry(request: Request) -> GraphRegistryStore:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.knowledgegraphs("unavailable", "Database unavailable")
        )
    return GraphRegistryStore(pool)


def _shared_read_admits(
    request: Request, stored_access: Any, scope: CallerScope
) -> bool:
    """ADR-080: does the row's access list admit this caller (beyond the tier
    ladder)? Group membership resolves lazily against the request pool."""
    caller = resource_caller(
        request,
        user_id=scope.user_id,
        project_id=scope.project_id,
        is_admin=scope.is_admin,
    )
    return access_admits(stored_access, caller)


def _stored_project(kind: str | None, project_id: str | None, scope: CallerScope) -> str | None:
    """The project a new graph is stored under.

    Owner-only graphs are not project-bound and system graphs stay global.
    Any other graph created without a project gets the caller's project, the
    one create_graph's write check admitted. Persisting NULL left a graph no
    non-admin could ever update, archive or decide proposals on, its creator
    included (2026-10-09).
    """
    kind_reduced = (kind or "").strip().lower()
    if kind_reduced in ("user", "personal"):
        return None
    if project_id is None and kind_reduced != "system":
        return scope.project_id or None
    return project_id


class AccessUpdate(BaseModel):
    """ADR-080 access list update (validated against the canonical grammar)."""

    access: list[str] = Field(
        default_factory=list,
        description=(
            "user:<id> | project:<id> | circle:<proj>/<id> | system; [] = tier ladder only"
        ),
    )


def _base_graph_list(
    registry: GraphRegistryStore,
    *,
    project_id: str | None,
    status: str | None,
    q: str | None,
    limit: int,
) -> list[Any]:
    """The tier-ladder listing (project default materialized on demand)."""
    if project_id:
        registry.ensure_default_graph(project_id)
    if q:
        graphs = registry.search_graphs(q, project_id=project_id, limit=limit)
        if status:
            graphs = [g for g in graphs if g.status == status]
        return graphs
    return registry.list_graphs(project_id=project_id, status=status, limit=limit)


class CreateGraphRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = ""
    project_id: str | None = None
    kind: str = "research"
    ontology_id: str | None = None
    ontology_version: str | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)


class UpdateGraphRequest(BaseModel):
    description: str | None = None
    kind: str | None = None
    ontology_id: str | None = None
    ontology_version: str | None = None
    status: str | None = Field(default=None, description="Lifecycle: active | archived")


@router.get("")
async def list_graphs(
    request: Request,
    project_id: str | None = Query(None),
    status: str | None = Query("active"),
    q: str | None = Query(None, description="Name/description search"),
    limit: int = Query(200, ge=1, le=1000),
) -> dict[str, Any]:
    """List registered knowledge graphs with optional filters.

    When ``project_id`` is given, its default graph is materialized first
    so legacy projects are listed without any manual migration.
    """
    registry = _registry(request)
    graphs = _base_graph_list(
        registry, project_id=project_id, status=status, q=q, limit=limit
    )
    scope = get_caller_scope(request)
    visible = [
        g
        for g in graphs
        if graph_readable(
            kind=g.kind, project_id=g.project_id, owner_id=g.owner_id, caller=scope
        )
    ]
    seen = {g.graph_id for g in visible}
    visible.extend(
        _shared_visible_graphs(request, registry, status=status, scope=scope, seen=seen)
    )
    return {"graphs": [g.to_dict() for g in visible], "total": len(visible)}


def _shared_visible_graphs(
    request: Request,
    registry: GraphRegistryStore,
    *,
    status: str | None,
    scope: CallerScope,
    seen: set[str],
) -> list[Any]:
    """ADR-080 rows outside the caller's tier whose access admits them."""
    out: list[Any] = []
    for shared in registry.list_shared_graphs(status=status):
        if shared.graph_id in seen:
            continue
        if _shared_read_admits(request, shared.access, scope):
            out.append(shared)
            seen.add(shared.graph_id)
    return out


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
    project_id = _stored_project(body.kind, project_id, scope)
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
    if graph is None or not _graph_read_allowed(
        request,
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        stored_access=graph.access,
        scope=scope,
    ):
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    payload = graph.to_dict()
    # ADR-049 B-part-2: traversal surfaces carry the resolved ontology
    # (registry-first); unresolvable references named, never smudged.
    payload["ontology"] = _resolved_ontology_block(request, payload)
    return payload


def _resolved_ontology_block(request: Request, meta: dict[str, Any]) -> dict[str, Any]:
    """Registry-first ontology block for one graph meta (ADR-049 B-part-2).

    Shares the knowledge-query payload shape
    ({graph_id, ontology} | {graph_id, ontology_error}) so consumers have
    ONE language across query and traversal surfaces.
    """
    from .knowledge_query import _ontology_blocks_for

    pool = getattr(request.app.state, "umrs_db_pool", None)
    blocks = _ontology_blocks_for([meta], pool)
    if blocks:
        return blocks[0]
    return {"graph_id": meta.get("graph_id", "?"), "ontology_error": "unresolved"}


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
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    scope = get_caller_scope(request)
    meta = registry.resolve_graph(graph_id)
    if not _graph_read_allowed(
        request,
        kind=str(content.get("kind") or ""),
        project_id=content.get("project_id")
        if isinstance(content.get("project_id"), str)
        else None,
        owner_id=str(content.get("owner_id") or ""),
        stored_access=meta.access if meta is not None else [],
        scope=scope,
    ):
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
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
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
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
                PlatformError.knowledgegraphs(
                    "bad_request", f"Invalid status {body.status!r}"
                )
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


@router.get("/{graph_id}/access")
async def get_graph_access(graph_id: str, request: Request) -> dict[str, Any]:
    """The graph's ADR-080 access list (any caller who may read the graph).
    # auth: verified
    """
    registry = _registry(request)
    graph = registry.resolve_graph(graph_id)
    scope = get_caller_scope(request)
    if graph is None or not _graph_read_allowed(
        request,
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        stored_access=graph.access,
        scope=scope,
    ):
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    return {"graph_id": graph.graph_id, "access": list(graph.access)}


@router.put("/{graph_id}/access")
async def set_graph_access(
    graph_id: str,
    body: AccessUpdate,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Replace the graph's access list (write-tier owners or admin; audited).

    The list names who may READ the graph beyond its tier ladder;
    ``system`` entries are admin-only. Lifecycle writes are unaffected.
    # auth: verified
    """
    enforce_registry_write_authorization(request)
    registry = _registry(request)
    graph = registry.resolve_graph(graph_id)
    scope = CallerScope(
        user_id=identity.user_id or "",
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    if graph is None or not graph_writable(
        kind=graph.kind,
        project_id=graph.project_id,
        owner_id=graph.owner_id,
        caller=scope,
    ):
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Graph {graph_id} not found")
        )
    caller = resource_caller(
        request,
        user_id=identity.user_id,
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    access = checked_access(body.access, caller, error=PlatformError.knowledgegraphs)
    updated = registry.set_access(graph.graph_id, access)
    from .audit_events import emit_audit

    emit_audit(
        "knowledgegraph.access_changed",
        project_id=graph.project_id or "",
        details={"graph_id": graph.graph_id, "access": access, "by": scope.user_id},
    )
    return (updated or graph).to_dict()


class EstablishResponse(BaseModel):
    node_id: str | None = None
    edge_id: str | None = None
    status: str
    established_by: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float | None = None


class EstablishBody(BaseModel):
    """D-35 evidence refs on tiered rows: the human's promotion carries
    the evidence that justified it (+ optionally the settled confidence).

    Both optional — absent means the review block records the actor only
    (never an invented ref or confidence)."""

    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float | None = Field(None, ge=0.0, le=1.0)


def _refuse_agent_promotion(request: Request) -> None:
    """D-35: proposed → established is a human decision, never the agent's."""
    if request_actor(request) == AGENT_ACTOR:
        raise_typed(
            PlatformError.knowledgegraphs(
                "forbidden",
                "Knowledge promotion (proposed → established) is a human "
                "decision; agents propose, humans establish.",
            )
        )


async def _establish_target(request: Request) -> KnowledgeStore:
    from ._asyncpg_pool import knowledge_asyncpg_pool

    pool = await knowledge_asyncpg_pool(request)
    if pool is None:
        raise_typed(
            PlatformError.knowledgegraphs("unavailable", "Knowledge store unavailable")
        )
    from umrs.knowledge.store import KnowledgeStore

    return KnowledgeStore(pool)


@router.post("/{graph_id}/nodes/{node_id}/establish", response_model=EstablishResponse)
async def establish_node(
    graph_id: str,
    node_id: str,
    request: Request,
    body: EstablishBody | None = None,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> EstablishResponse:
    """Promote one node proposed → established (D-35 two-tier knowledge).

    Human-only (agent actor refused by name); idempotent on established
    rows; 404 when the node does not exist. The optional body carries
    the promotion's evidence refs and settled confidence (D-35 tail:
    evidence links on tiered rows).
    """
    _refuse_agent_promotion(request)
    store = await _establish_target(request)
    established_by = (identity.user_id or "user").strip() or "user"
    refs = list(body.evidence_refs) if body else []
    confidence = body.confidence if body else None
    ok = await store.establish_node(
        node_id,
        established_by=established_by,
        evidence_refs=refs,
        confidence=confidence,
    )
    if not ok:
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Node {node_id} not found")
        )
    return EstablishResponse(
        node_id=node_id,
        status="established",
        established_by=established_by,
        evidence_refs=refs,
        confidence=confidence,
    )


@router.post("/{graph_id}/edges/{edge_id}/establish", response_model=EstablishResponse)
async def establish_edge(
    graph_id: str,
    edge_id: str,
    request: Request,
    body: EstablishBody | None = None,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> EstablishResponse:
    """Promote one edge proposed → established (same rules as nodes)."""
    _refuse_agent_promotion(request)
    store = await _establish_target(request)
    established_by = (identity.user_id or "user").strip() or "user"
    refs = list(body.evidence_refs) if body else []
    confidence = body.confidence if body else None
    ok = await store.establish_edge(
        edge_id,
        established_by=established_by,
        evidence_refs=refs,
        confidence=confidence,
    )
    if not ok:
        raise_typed(
            PlatformError.knowledgegraphs("not_found", f"Edge {edge_id} not found")
        )
    return EstablishResponse(
        edge_id=edge_id,
        status="established",
        established_by=established_by,
        evidence_refs=refs,
        confidence=confidence,
    )
