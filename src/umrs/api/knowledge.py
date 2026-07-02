"""UMRS API routes for knowledge search and RC1 project workflow."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from umrs.api.files import S3Config, get_project_id, get_s3_config
from umrs.graph_reasoning import GraphNavigator
from umrs.knowledge.knowledge_correction_controller import (
    KnowledgeCorrectionController,
)
from umrs.knowledge.knowledge_dashboard_controller import (
    KnowledgeDashboardController,
    ObjectDetail,
)
from umrs.knowledge.project import ProjectPhase, ResearchProject
from umrs.knowledge.research_readiness_controller import ResearchReadinessController
from umrs.knowledge_graph_api import InMemoryKnowledgeGraph, KnowledgeEdge
from umrs.knowledge_model import InMemoryKnowledgeModelStore
from umrs.scientific_memory import InMemoryScientificMemoryStore
from umrs.scientific_reasoning_engine import ScientificReasoningEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge"])

# Cross-experiment search engine — injected at startup (MEM-C5)
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


def build_research_project(
    project_id: str,
    config: S3Config,
) -> ResearchProject:
    """Construct a catalog-backed project for HTTP workflow handlers."""
    return ResearchProject(
        project_id=project_id,
        s3_client=config.get_client(),
        s3_bucket=config.bucket,
    )


class UnderstandResponse(BaseModel):
    project_id: str
    phase: str
    artifact_count: int
    extracted_count: int
    graph_nodes: int
    graph_edges: int
    coverage_score: float


class VerifyRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Verification question")


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


# ── M3.5 — Interactive Scientific Knowledge Verification ──────────────────

# Shared in-memory stores for the M3.5 controllers.  A real deployment would
# inject Postgres-backed stores via set_* hooks; this wiring exposes the
# controllers through the HTTP sidecar so the TUI can reach them.
_m35_knowledge_model = InMemoryKnowledgeModelStore()
_m35_knowledge_graph = InMemoryKnowledgeGraph()
_m35_scientific_memory = InMemoryScientificMemoryStore()
_m35_navigator = GraphNavigator(graph=_m35_knowledge_graph)
_m35_reasoning_engine = ScientificReasoningEngine(navigator=_m35_navigator)

_dashboard_controller = KnowledgeDashboardController(
    knowledge_model=_m35_knowledge_model,
    knowledge_graph=_m35_knowledge_graph,
    scientific_memory=_m35_scientific_memory,
)
_correction_controller = KnowledgeCorrectionController(
    knowledge_graph=_m35_knowledge_graph,
)
_readiness_controller = ResearchReadinessController(
    knowledge_model=_m35_knowledge_model,
    knowledge_graph=_m35_knowledge_graph,
)


class RelatedObjectRef(BaseModel):
    object_id: str
    object_type: str
    title: str
    relation: str
    confidence: float


class ObjectProvenance(BaseModel):
    source: str
    extractor: str
    timestamp: str
    confidence: float


class ObjectDetailResponse(BaseModel):
    object_id: str
    object_type: str
    title: str
    status: str
    metadata: dict[str, Any]
    provenance: ObjectProvenance
    related_objects: list[RelatedObjectRef]
    graph_neighbors: list[RelatedObjectRef]
    created_at: str


class CorrectionRequest(BaseModel):
    correction_type: str = Field(..., description="Type of correction")
    corrected_by: str = Field(..., description="User making the correction")
    reason: str = Field(..., description="Why the correction is needed")
    affected_object_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    approved: bool = Field(..., description="Approval decision")
    approved_by: str = Field(..., description="User making the decision")


def _object_title(obj: ObjectDetail | None) -> str:
    if obj is None:
        return ""
    return obj.title or obj.object_id


def _build_related_ref(edge: KnowledgeEdge, center_id: str) -> RelatedObjectRef:
    """Build a RelatedObjectRef from a graph edge centered on center_id."""
    neighbor_id = edge.target_id if edge.source_id == center_id else edge.source_id
    relation = edge.relation_type.value
    node = _m35_knowledge_graph.get_node(neighbor_id)
    if node is None:
        obj = _m35_knowledge_model.get_object(neighbor_id)
        title = obj.content.get("title", neighbor_id) if obj else neighbor_id
        node_type = obj.object_type.value if obj else "unknown"
    else:
        title = node.title or neighbor_id
        node_type = node.node_type
    return RelatedObjectRef(
        object_id=neighbor_id,
        object_type=node_type,
        title=title,
        relation=relation,
        confidence=edge.confidence,
    )


def _build_object_detail_response(detail: ObjectDetail) -> dict[str, Any]:
    """Convert controller ObjectDetail into the TUI-facing shape."""
    edges = list(_m35_knowledge_graph.get_outgoing(detail.object_id)) + list(
        _m35_knowledge_graph.get_incoming(detail.object_id)
    )
    related = [_build_related_ref(edge, detail.object_id) for edge in edges]

    avg_confidence = 0.0
    if edges:
        avg_confidence = sum(edge.confidence for edge in edges) / len(edges)

    return ObjectDetailResponse(
        object_id=detail.object_id,
        object_type=detail.object_type,
        title=detail.title,
        status=detail.status,
        metadata=detail.metadata,
        provenance=ObjectProvenance(
            source=detail.object_id,
            extractor="knowledge_graph",
            timestamp=detail.updated_at or detail.created_at,
            confidence=round(avg_confidence, 4),
        ),
        related_objects=related,
        graph_neighbors=related,
        created_at=detail.created_at,
    ).model_dump()


@router.get("/dashboard")
async def get_knowledge_dashboard() -> dict[str, Any]:
    """Get project knowledge summary for the M3.5 dashboard."""
    return _dashboard_controller.get_knowledge_summary().to_dict()


@router.get("/objects")
async def list_knowledge_objects(
    object_type: str | None = Query(None, description="Filter by object type"),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """List scientific objects for the M3.5 Knowledge Explorer."""
    return [
        obj.to_dict()
        for obj in _dashboard_controller.list_objects(
            object_type=object_type,
            limit=limit,
        )
    ]


@router.get("/objects/{object_id}", response_model=ObjectDetailResponse)
async def get_knowledge_object(object_id: str) -> ObjectDetailResponse:
    """Get detailed view of a scientific object."""
    detail = _dashboard_controller.get_object_detail(object_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Object {object_id} not found")
    return _build_object_detail_response(detail)


@router.get("/graph/{object_id}/neighbors")
async def get_graph_neighbors(object_id: str) -> list[dict[str, Any]]:
    """Get graph neighbors for a scientific object."""
    edges = list(_m35_knowledge_graph.get_outgoing(object_id)) + list(
        _m35_knowledge_graph.get_incoming(object_id)
    )
    return [_build_related_ref(edge, object_id).model_dump() for edge in edges]


@router.post("/correction")
async def submit_knowledge_correction(req: CorrectionRequest) -> dict[str, Any]:
    """Record a human knowledge correction."""
    record = _correction_controller.correct(
        correction_type=req.correction_type,
        corrected_by=req.corrected_by,
        reason=req.reason,
        affected_object_ids=req.affected_object_ids,
        details=req.details,
    )
    return record.to_dict()


@router.get("/readiness")
async def get_research_readiness() -> dict[str, Any]:
    """Get the research readiness report."""
    return _readiness_controller.evaluate_readiness().to_dict()


@router.post("/approval")
async def submit_approval(req: ApprovalRequest) -> dict[str, Any]:
    """Approve or reject autonomous research."""
    if req.approved:
        report = _readiness_controller.evaluate_readiness()
        record = _readiness_controller.approve(
            user_id=req.approved_by,
            readiness_report=report,
        )
    else:
        record = _readiness_controller.reject(
            user_id=req.approved_by,
            reason="",
        )
    return record.to_dict()


@router.post("/understand", response_model=UnderstandResponse)
async def understand_project(
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> UnderstandResponse:
    """Extract structured knowledge and build the project knowledge graph."""
    project = build_research_project(project_id, config)
    try:
        project.understand()
    except Exception as exc:
        logger.exception("understand failed for project %s", project_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    from umrs.api.knowledge_graph import register_project_graph

    register_project_graph(project_id, project.umrs_knowledge_graph.graph)

    artifacts = project.knowledge_store.list_artifacts()
    coverage_score = (
        project.coverage_dashboard.coverage_score
        if project.coverage_dashboard is not None
        else 0.0
    )
    return UnderstandResponse(
        project_id=project_id,
        phase=project.phase.value,
        artifact_count=len(artifacts),
        extracted_count=len(project.extracted_knowledge),
        graph_nodes=len(project.knowledge_graph.nodes),
        graph_edges=len(project.knowledge_graph.edges),
        coverage_score=coverage_score,
    )


@router.post("/verify")
async def verify_project_knowledge(
    body: VerifyRequest,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> dict[str, Any]:
    """Answer a knowledge verification question with evidence."""
    project = build_research_project(project_id, config)
    if project.phase not in (
        ProjectPhase.READY,
        ProjectPhase.PLANNING,
        ProjectPhase.RESEARCHING,
        ProjectPhase.COMPLETED,
    ):
        try:
            project.understand()
            from umrs.api.knowledge_graph import register_project_graph

            register_project_graph(project_id, project.umrs_knowledge_graph.graph)
        except Exception as exc:
            logger.exception("auto-understand failed for project %s", project_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        answer = project.verify(body.question)
    except Exception as exc:
        logger.exception("verify failed for project %s", project_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return answer.to_dict()
