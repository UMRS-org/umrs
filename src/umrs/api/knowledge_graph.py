"""UMRS API routes for the knowledge graph (Phase 2, Step 3.1).

Exposes the knowledge_graph_api subsystem as HTTP API endpoints.

Routes:
  POST   /umrs/knowledge-graph/nodes         — Add a node
  POST   /umrs/knowledge-graph/edges         — Add an edge
  GET    /umrs/knowledge-graph/nodes/{id}     — Get a node
  GET    /umrs/knowledge-graph/nodes/{id}/outgoing — Get outgoing edges
  GET    /umrs/knowledge-graph/nodes/{id}/incoming — Get incoming edges
  GET    /umrs/knowledge-graph/nodes/{id}/neighbors — Get neighbors
  GET    /umrs/knowledge-graph/nodes/{id}/related   — Get related nodes
  GET    /umrs/knowledge-graph/path           — Find path between nodes
  GET    /umrs/knowledge-graph/query          — Query the graph
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from umrs.api.files import get_project_id
from umrs.knowledge_graph_api import (
    InMemoryKnowledgeGraph,
    RelationType,
)

logger = logging.getLogger(__name__)


def _sat_router_retired() -> None:
    """ADR-026 execution: the sat_kg facade is retired (R5 tail, 2026-08-23).

    Every /umrs/knowledge-graph/* route now 410s and points to the canonical
    multigraph. Tables and module stay intact for read-compat/migration hooks.
    """
    raise HTTPException(
        status_code=status.HTTP_410_GONE,
        detail=(
            "The sat_kg graph facade is retired (ADR-026 / R5, 2026-08-23). "
            "Use the canonical multigraph: /umrs/knowledge/graph, "
            "/umrs/knowledge/graphs, or client.knowledge / r4 knowledge *."
        ),
    )


router = APIRouter(
    prefix="/umrs/knowledge-graph",
    tags=["knowledge-graph"],
    dependencies=[Depends(_sat_router_retired)],
)

# Graph instances — injected at startup or populated by POST /umrs/knowledge/understand
_graph: InMemoryKnowledgeGraph | None = None
_project_graphs: dict[str, InMemoryKnowledgeGraph] = {}


def set_graph(graph: InMemoryKnowledgeGraph) -> None:
    global _graph
    _graph = graph


def register_project_graph(project_id: str, graph: InMemoryKnowledgeGraph) -> None:
    """Store a project-scoped graph for HTTP traversal endpoints."""
    _project_graphs[project_id] = graph


def _get_graph(project_id: str | None = None) -> InMemoryKnowledgeGraph:
    if project_id and project_id in _project_graphs:
        return _project_graphs[project_id]
    if _graph is not None:
        return _graph
    raise HTTPException(
        status_code=503,
        detail="Knowledge graph not initialized. Run POST /umrs/knowledge/understand first.",
    )


# ── Request/Response Models ──────────────────────────────────────────────


class AddNodeRequest(BaseModel):
    node_id: str = Field(..., description="Node ID (typically an object_id)")
    node_type: str = Field(..., description="Node type (paper, code, hypothesis, etc.)")
    title: str = Field("", description="Human-readable title")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    proposal_id: str | None = Field(
        default=None,
        description="Approved graph-extension proposal — required for agent actors",
    )


class AddEdgeRequest(BaseModel):
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relation_type: str = Field(
        ..., description="Relation type (cites, implements, uses, etc.)"
    )
    weight: float = Field(1.0, description="Edge weight")
    confidence: float = Field(1.0, description="Confidence score (0.0–1.0)")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    proposal_id: str | None = Field(
        default=None,
        description="Approved graph-extension proposal — required for agent actors",
    )


class NodeResponse(BaseModel):
    node_id: str
    node_type: str
    title: str
    metadata: dict[str, Any]


class EdgeResponse(BaseModel):
    edge_id: str
    source_id: str
    target_id: str
    relation_type: str
    weight: float
    confidence: float
    metadata: dict[str, Any]
    created_at: str


class PathResponse(BaseModel):
    source_id: str
    target_id: str
    path: list[str] | None
    found: bool


class QueryResponse(BaseModel):
    nodes: list[NodeResponse]
    total: int


def _resolve_graph(project_id: str = Depends(get_project_id)) -> InMemoryKnowledgeGraph:
    return _get_graph(project_id)


# ── Routes ────────────────────────────────────────────────────────────────


def _guard_direct_write(
    request: Request, project_id: str | None, proposal_id: str | None
) -> None:
    """0069 agent-mutation guard for direct node/edge writes."""
    from umrs.api._knowledge_mutation_guard import (
        enforce_graph_mutation_authorization,
    )
    from umrs.knowledge.graph_registry import default_graph_id

    target = default_graph_id(project_id or "default")
    enforce_graph_mutation_authorization(
        request, target_graph_id=target, proposal_id=proposal_id
    )


@router.post("/nodes", response_model=NodeResponse)
async def add_node(
    req: AddNodeRequest,
    request: Request,
    project_id: str | None = Depends(get_project_id),
    graph: InMemoryKnowledgeGraph = Depends(_resolve_graph),
) -> NodeResponse:
    """Add a node to the knowledge graph."""
    _guard_direct_write(request, project_id, req.proposal_id)
    graph.add_node(
        node_id=req.node_id,
        node_type=req.node_type,
        title=req.title,
        metadata=req.metadata,
    )
    node = graph.get_node(req.node_id)
    return NodeResponse(**node.to_dict())


@router.post("/edges")
async def add_edge(
    req: AddEdgeRequest,
    request: Request,
    project_id: str | None = Depends(get_project_id),
    graph: InMemoryKnowledgeGraph = Depends(_resolve_graph),
) -> dict[str, Any]:
    """Add an edge to the knowledge graph."""
    _guard_direct_write(request, project_id, req.proposal_id)
    try:
        relation_type = RelationType(req.relation_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid relation type: {req.relation_type}",
        )
    edge_id = graph.add_edge(
        source_id=req.source_id,
        target_id=req.target_id,
        relation_type=relation_type,
        weight=req.weight,
        confidence=req.confidence,
        metadata=req.metadata,
    )
    return {"edge_id": edge_id}


@router.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_node(
    node_id: str,
    graph: InMemoryKnowledgeGraph = Depends(_resolve_graph),
) -> NodeResponse:
    """Get a node by ID."""
    node = graph.get_node(node_id)
    if node is None:
        raise HTTPException(
            status_code=404,
            detail=f"Node {node_id} not found",
        )
    return NodeResponse(**node.to_dict())


@router.get("/nodes/{node_id}/outgoing", response_model=list[EdgeResponse])
async def get_outgoing(
    node_id: str,
    graph: InMemoryKnowledgeGraph = Depends(_resolve_graph),
) -> list[EdgeResponse]:
    """Get all outgoing edges from a node."""
    edges = graph.get_outgoing(node_id)
    return [EdgeResponse(**edge.to_dict()) for edge in edges]


@router.get("/nodes/{node_id}/incoming", response_model=list[EdgeResponse])
async def get_incoming(
    node_id: str,
    graph: InMemoryKnowledgeGraph = Depends(_resolve_graph),
) -> list[EdgeResponse]:
    """Get all incoming edges to a node."""
    edges = graph.get_incoming(node_id)
    return [EdgeResponse(**edge.to_dict()) for edge in edges]


@router.get("/nodes/{node_id}/neighbors")
async def get_neighbors(
    node_id: str,
    graph: InMemoryKnowledgeGraph = Depends(_resolve_graph),
) -> dict[str, Any]:
    """Get all neighbor node IDs."""
    neighbors = graph.get_neighbors(node_id)
    return {"node_id": node_id, "neighbors": neighbors}


@router.get("/nodes/{node_id}/related")
async def get_related(
    node_id: str,
    relation_type: str = Query(..., description="Relation type"),
    graph: InMemoryKnowledgeGraph = Depends(_resolve_graph),
) -> dict[str, Any]:
    """Get all node IDs related to the given node with the given relation type."""
    try:
        rel_type = RelationType(relation_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid relation type: {relation_type}",
        )
    related = graph.get_related(node_id, rel_type)
    return {"node_id": node_id, "related": related}


@router.get("/path", response_model=PathResponse)
async def find_path(
    source_id: str = Query(..., description="Source node ID"),
    target_id: str = Query(..., description="Target node ID"),
    max_depth: int = Query(5, ge=1, le=20, description="Maximum search depth"),
    graph: InMemoryKnowledgeGraph = Depends(_resolve_graph),
) -> PathResponse:
    """Find a path between two nodes using BFS."""
    path = graph.find_path(source_id, target_id, max_depth=max_depth)
    return PathResponse(
        source_id=source_id,
        target_id=target_id,
        path=path,
        found=path is not None,
    )


@router.get("/query", response_model=QueryResponse)
async def query_graph(
    node_type: str | None = Query(None, description="Filter by node type"),
    relation_type: str | None = Query(None, description="Filter by relation type"),
    graph: InMemoryKnowledgeGraph = Depends(_resolve_graph),
) -> QueryResponse:
    """Query the graph for nodes matching the given criteria."""
    rel_type: RelationType | None = None
    if relation_type:
        try:
            rel_type = RelationType(relation_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid relation type: {relation_type}",
            )
    nodes = graph.query(node_type=node_type, relation_type=rel_type)
    return QueryResponse(
        nodes=[NodeResponse(**node.to_dict()) for node in nodes],
        total=len(nodes),
    )
