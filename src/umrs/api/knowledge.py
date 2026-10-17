"""UMRS API routes for knowledge search and RC1 project workflow."""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from umrs.api._asyncpg_pool import knowledge_asyncpg_pool
from umrs.api.files import get_project_id
from umrs.graph_reasoning import GraphNavigator
from umrs.jobs.models import JobType
from umrs.knowledge.canonical_graph_view import (
    default_view as _canonical_default_view,
)
from umrs.knowledge.graph_types import KnowledgeEdge
from umrs.knowledge.knowledge_correction_controller import (
    KnowledgeCorrectionController,
)
from umrs.knowledge.knowledge_dashboard_controller import (
    KnowledgeDashboardController,
    ObjectDetail,
)
from umrs.knowledge.project import ProjectPhase, ResearchProject
from umrs.knowledge.readiness_approvals import ReadinessApprovalStore
from umrs.knowledge.verification_store import VerificationStore
from umrs.knowledge_model import InMemoryKnowledgeModelStore
from umrs.scientific_memory import InMemoryScientificMemoryStore
from umrs.scientific_reasoning_engine import ScientificReasoningEngine
from umrs.shared.storage.s3_config import S3Config, get_s3_config

from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge"])

# Cross-experiment search engine — injected at startup (MEM-C5)
_engine = None


def set_engine(engine) -> None:
    global _engine
    _engine = engine


def _get_engine():
    if _engine is None:
        raise_typed(
            PlatformError.knowledge(
                "unavailable", "Knowledge search engine not initialized"
            )
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


class UnderstandRequest(BaseModel):
    """Optional body — agent-originated understand requires a proposal."""

    proposal_id: str | None = Field(
        default=None,
        description=(
            "Approved graph-extension proposal id — required for agent actors"
        ),
    )


class VerifyRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Verification question")


def _readable_project(request: Request, project_id: str | None, route: str) -> str:
    """The explicit or caller project, fail-closed (audit B10): 403 when there
    is none or it is outside the caller's access tier."""
    from umrs.knowledge.access_scope import project_readable

    return _scoped_project(request, project_id, route, project_readable)


def _scoped_project(
    request: Request, project_id: str | None, route: str, allowed: Any
) -> str:
    """The explicit or caller project when ``allowed(project, caller)``; 403
    when there is none or the predicate refuses it."""
    from umrs.api._scope_identity import request_caller_scope

    caller = request_caller_scope(request)
    effective_project = project_id or caller.project_id
    if not effective_project:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_required",
                "message": f"{route} requires a project scope (authenticated or explicit).",
            },
        )
    if not allowed(effective_project, caller):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_denied",
                "message": "Project is outside the caller's access tier.",
            },
        )
    return effective_project


@router.get("/search")
async def search_knowledge(
    request: Request,
    query: str = Query(..., description="Search query"),
    scope: str = Query("project", description="Search scope: project|experiment|user"),
    project_id: str | None = Query(None),
    namespaces: str | None = Query(None, description="Comma-separated namespace names"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search across knowledge namespaces with source attribution.

    Project binding is mandatory (audit B10): this endpoint previously had
    no identity dependency at all — omitting ``project_id`` searched every
    project's memory_items. Anonymous or unscoped calls fail closed.
    """
    effective_project = _readable_project(request, project_id, "knowledge/search")
    engine = _get_engine()
    ns_list = (
        [n.strip() for n in namespaces.split(",") if n.strip()] if namespaces else None
    )
    result = engine.search(
        query=query,
        scope=scope,
        project_id=effective_project,
        namespaces=ns_list,
        limit=limit,
    )
    return result.to_dict()


# ── D5/D6 — Unified academic-source search (arXiv | S2 | CrossRef | CiNii | J-STAGE) ─


class ExternalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    sources: list[str] | None = Field(
        default=None,
        description="Subset of arxiv, semantic_scholar, crossref, cinii, jstage",
    )
    limit_per_source: int = Field(default=5, ge=1, le=25)
    project_id: str | None = None


@router.post("/external/search")
async def post_external_search(
    req: ExternalSearchRequest,
    request: Request,
    caller_project: str = Depends(get_project_id),
):
    """Fan out one query across academic sources including JA-native ones.

    Defaults include CiNii Research and J-STAGE so Japanese-language
    research content is first-class alongside arXiv/S2/CrossRef. Matches
    the r4 SDK's `KnowledgeClient.external_search()` contract.

    H-1 (2026-10-18): this server-side egress was anonymous; it now needs
    the verified project (401 without it in strict mode), and each search is
    logged with the verified user and project. The body ``project_id`` reads
    no project data.
    """
    from umrs.knowledge.external import ExternalKnowledgeSearcher, ExternalSourceType

    default = ["arxiv", "semantic_scholar", "crossref", "cinii", "jstage"]
    names = req.sources or default
    try:
        source_types = [ExternalSourceType(n) for n in names]
    except ValueError:
        raise_typed(
            PlatformError.knowledge(
                "bad_request",
                f"unknown source in {names!r}; allowed: {', '.join(default)}",
            )
        )

    from umrs.api._scope_identity import request_caller_scope

    caller = request_caller_scope(request)
    logger.info(
        "external search egress user=%s project=%s sources=%s",
        caller.user_id,
        caller_project,
        names,
    )
    searcher = ExternalKnowledgeSearcher()
    results = searcher.search_all(
        req.query,
        max_results_per_source=req.limit_per_source,
        sources=source_types,
    )
    return {
        "query": req.query,
        "sources": names,
        "results": [
            {
                "source_type": r.source_type.value,
                "status": r.status.value,
                "total_results": r.total_results,
                "error_message": r.error_message,
                "search_duration_sec": r.search_duration_sec,
                "references": [ref.to_dict() for ref in r.references],
            }
            for r in results
        ],
    }


# ── M3.5 — Interactive Scientific Knowledge Verification ──────────────────

# Shared in-memory stores for the M3.5 controllers.  A real deployment would
# inject Postgres-backed stores via set_* hooks; this wiring exposes the
# controllers through the HTTP sidecar so the TUI can reach them.
_m35_knowledge_model = InMemoryKnowledgeModelStore()

# ADR-026 migration: the shared m35 graph is a canonical-backed view with
# project '' root (memory-only until a request attaches the shared pool).
_m35_knowledge_graph = _canonical_default_view(project_id="default")
_m35_scientific_memory = InMemoryScientificMemoryStore()
_m35_navigator = GraphNavigator(graph=_m35_knowledge_graph)
_m35_reasoning_engine = ScientificReasoningEngine(navigator=_m35_navigator)


async def _m35_maybe_refresh(request) -> None:
    """Attach the shared pool on first request and reload canonical rows."""
    pool = getattr(getattr(request, "app", None), "state", None)
    pool = getattr(pool, "umrs_knowledge_asyncpg_pool", None)
    if pool is None:
        return
    from ..knowledge.store import KnowledgeStore as _KS

    if _m35_knowledge_graph._store is None:
        _m35_knowledge_graph._store = _KS(pool)
    await _m35_knowledge_graph.refresh()


_dashboard_controller = KnowledgeDashboardController(
    knowledge_model=_m35_knowledge_model,
    knowledge_graph=_m35_knowledge_graph,
    scientific_memory=_m35_scientific_memory,
)
_correction_controller = KnowledgeCorrectionController(
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


class ReadinessApproveRequest(BaseModel):
    """Per-project readiness approval; the approver is the verified caller."""

    project_id: str | None = Field(
        None, description="Project to approve (default: the caller's)"
    )
    reason: str = Field("", max_length=2000, description="Why (audited)")


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


async def _explorer_caller(request: Request) -> Any:
    """Fail-closed scope entry gate for M3.5 Explorer routes (R2, 2026-08-22).

    Explorer controllers are module-global (no per-project namespace), so the
    three-tier predicates cannot be applied per-object; the convergent gate is
    strict-mode authentication (401 when unauthenticated) with mvp parity
    (anonymous callers allowed). Returns the resolved scope for logging.
    """
    from umrs.api._scope_identity import request_caller_scope

    scope = request_caller_scope(request)
    if scope.anonymous:
        logger.debug("explorer route used by anonymous caller (mvp mode)")
    try:
        await _m35_maybe_refresh(request)
    except Exception:
        logger.debug("m35 refresh skipped", exc_info=True)
    return scope


@router.get("/dashboard")
async def get_knowledge_dashboard(request: Request) -> dict[str, Any]:
    """Get project knowledge summary for the M3.5 dashboard."""
    await _explorer_caller(request)
    return _dashboard_controller.get_knowledge_summary().to_dict()


@router.get("/objects")
async def list_knowledge_objects(
    request: Request,
    object_type: str | None = Query(None, description="Filter by object type"),
    limit: int = Query(100, ge=1, le=1000),
) -> list[dict[str, Any]]:
    """List scientific objects for the M3.5 Knowledge Explorer."""
    await _explorer_caller(request)
    return [
        obj.to_dict()
        for obj in _dashboard_controller.list_objects(
            object_type=object_type,
            limit=limit,
        )
    ]


@router.get("/objects/{object_id}", response_model=ObjectDetailResponse)
async def get_knowledge_object(
    object_id: str, request: Request
) -> ObjectDetailResponse:
    """Get detailed view of a scientific object."""
    await _explorer_caller(request)
    detail = _dashboard_controller.get_object_detail(object_id)
    if detail is None:
        raise_typed(
            PlatformError.knowledge("not_found", f"Object {object_id} not found")
        )
    return _build_object_detail_response(detail)


@router.get("/graph/{object_id}/neighbors")
async def get_graph_neighbors(object_id: str, request: Request) -> list[dict[str, Any]]:
    """Get graph neighbors for a scientific object."""
    await _explorer_caller(request)
    edges = list(_m35_knowledge_graph.get_outgoing(object_id)) + list(
        _m35_knowledge_graph.get_incoming(object_id)
    )
    return [_build_related_ref(edge, object_id).model_dump() for edge in edges]


@router.post("/correction")
async def submit_knowledge_correction(
    req: CorrectionRequest, request: Request
) -> dict[str, Any]:
    """Record a human knowledge correction (write-gated: R2 scope)."""
    await _explorer_caller(request)
    record = _correction_controller.correct(
        correction_type=req.correction_type,
        corrected_by=req.corrected_by,
        reason=req.reason,
        affected_object_ids=req.affected_object_ids,
        details=req.details,
    )
    return record.to_dict()


def _durable_node_payload(node: Any) -> dict[str, Any]:
    """KnowledgeStore row → Studio/webview node shape (id/type/label aliases)."""
    content = node.content if isinstance(node.content, dict) else {}
    label = node.label or node.node_id
    return {
        "id": node.node_id,
        "node_id": node.node_id,
        "name": label,
        "label": label,
        "title": label,
        "type": node.node_type,
        "node_type": node.node_type,
        "graph_id": node.graph_id,
        "created_at": node.created_at.isoformat() if node.created_at else None,
        "metadata": content,
    }


def _durable_edge_payload(edge: Any) -> dict[str, Any]:
    """KnowledgeStore row → Studio/webview edge shape (from/to/label aliases)."""
    return {
        "edge_id": edge.edge_id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relation_type": edge.relation_type,
        "from": edge.source_id,
        "to": edge.target_id,
        "label": edge.relation_type,
        "created_at": edge.created_at.isoformat() if edge.created_at else None,
    }


async def _durable_graph_rows(
    request: Request,
    project_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Durable Postgres graph rows for the project; empty when unavailable.

    KNOWLEDGE_INGEST/KNOWLEDGE_UNDERSTAND durable jobs write here — without
    this union the Studio graph stayed blind to everything ingest produced.
    """
    from umrs.knowledge.graph_registry import GraphRegistryStore
    from umrs.knowledge.store import KnowledgeStore

    sync_pool = getattr(request.app.state, "umrs_db_pool", None)
    if sync_pool is not None:
        # Materialize the default graph so legacy projects appear in the registry.
        GraphRegistryStore(sync_pool).ensure_default_graph(project_id)
    pool = await knowledge_asyncpg_pool(request)
    if pool is None:
        return [], []
    store = KnowledgeStore(pool)
    nodes = (await store.list_nodes(project_id))[:4000]
    edges = (await store.list_edges(project_id))[:8000]
    return (
        [_durable_node_payload(node) for node in nodes],
        [_durable_edge_payload(edge) for edge in edges],
    )


@router.get("/graph")
async def get_project_knowledge_graph(
    request: Request,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
    durable_only: bool = Query(
        False,
        description=(
            "Authoritative view: durable Postgres rows only (delegates to the "
            "/graph-pg path). Default False = union view."
        ),
    ),
) -> dict[str, Any]:
    """Project knowledge graph (nodes + typed edges) for the Studio/webview.

    Union of two sources (default): the RC1 in-memory project graph
    (reconstructed per request) and the durable Postgres store that
    KNOWLEDGE_INGEST / KNOWLEDGE_UNDERSTAND durable jobs populate. RC1 rows
    win on node_id collisions (richer local metadata). Smoke-free read.
    ``durable_only=true`` returns durable rows only (``/graph-pg`` parity).
    """
    from umrs.api._scope_identity import request_caller_scope
    from umrs.knowledge.access_scope import project_readable as _scope_readable

    if not _scope_readable(project_id, request_caller_scope(request)):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_denied",
                "message": "Project is outside the caller's access tier.",
            },
        )

    project = build_research_project(project_id, config)
    graph = project.knowledge_graph
    nodes: dict[str, dict[str, Any]] = {
        node.node_id: {**node.to_dict(), "id": node.node_id}
        for node in graph.nodes.values()
    }
    edges: dict[tuple[str, str, str], dict[str, Any]] = {
        (edge.source_id, edge.target_id, edge.relation_type.value): {
            **edge.to_dict(),
            "from": edge.source_id,
            "to": edge.target_id,
            "label": edge.relation_type.value,
        }
        for edge in graph.edges
    }
    try:
        durable_nodes, durable_edges = await _durable_graph_rows(request, project_id)
    except Exception as exc:  # noqa: BLE001 — read-only union; degrade to RC1 view
        logger.warning("durable graph union failed for %s: %s", project_id, exc)
        durable_nodes, durable_edges = [], []
    if durable_only:
        return {
            "project_id": project_id,
            "backend": "postgres",
            "nodes": durable_nodes,
            "edges": durable_edges,
        }
    for node in durable_nodes:
        nodes.setdefault(node["id"], node)
    for edge in durable_edges:
        key = (edge["from"], edge["to"], edge["label"])
        edges.setdefault(key, edge)
    return {
        "project_id": project_id,
        "phase": project.phase.value,
        "nodes": list(nodes.values()),
        "edges": list(edges.values()),
    }


@router.get("/graph-pg", response_model=dict[str, Any])
async def get_project_graph_pg(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    node_limit: int = Query(1000, le=10000),
    _get_project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Serve the durable Postgres knowledge graph (knowledge_nodes/edges).

    Unlike ``GET /graph`` (project graph reconstructed per request), this
    reads the durable store populated by KNOWLEDGE_UNDERSTAND durable jobs.
    """
    from umrs.api._scope_identity import request_caller_scope
    from umrs.knowledge.access_scope import project_readable as _scope_readable

    if not _scope_readable(project_id, request_caller_scope(request)):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_denied",
                "message": "Project is outside the caller's access tier.",
            },
        )
    import dataclasses

    from umrs.knowledge.graph_registry import GraphRegistryStore
    from umrs.knowledge.store import KnowledgeStore

    sync_pool = getattr(request.app.state, "umrs_db_pool", None)
    if sync_pool is not None:
        # Materialize the default graph so legacy projects appear in the registry.
        GraphRegistryStore(sync_pool).ensure_default_graph(project_id)
    pool = await knowledge_asyncpg_pool(request)
    store = KnowledgeStore(pool)
    nodes = (await store.list_nodes(project_id))[:node_limit]
    edges = (await store.list_edges(project_id))[:node_limit]
    return {
        "project_id": project_id,
        "backend": "postgres",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": [dataclasses.asdict(n) for n in nodes],
        "edges": [dataclasses.asdict(e) for e in edges],
    }


def _readiness_level_for(artifact_count: int, latest: Any | None) -> tuple[str, str]:
    """(readiness_level, blocker) from the project's artifacts and latest report."""
    if artifact_count == 0:
        return "not_ready", "no artifacts ingested; run r4 project ingest"
    if latest is None:
        return "not_ready", "no verification report yet; run r4 project verify"
    if latest.degraded:
        return "not_ready", "the latest verification ran degraded"
    if latest.confidence >= 0.8:
        return "ready", ""
    if latest.confidence >= 0.5:
        return "partially_ready", ""
    return "not_ready", f"verification confidence {latest.confidence:.2f} is below 0.50"


def project_readiness_payload(
    project_id: str, artifact_count: int, latest: Any | None
) -> dict[str, Any]:
    """Project readiness in the knowledge/readiness.py level vocabulary.

    ``partially_ready`` can run (gaps remain), as in the CLI and the TUI.
    """
    level, blocker = _readiness_level_for(artifact_count, latest)
    checks = [
        {
            "check": "artifacts",
            "passed": artifact_count > 0,
            "details": f"{artifact_count} ingested",
        },
        {
            "check": "verification",
            "passed": latest is not None and not latest.degraded,
            "details": "none"
            if latest is None
            else f"confidence {latest.confidence:.2f}",
        },
    ]
    return {
        "project_id": project_id,
        "readiness_level": level,
        "can_run": level in ("ready", "partially_ready"),
        "confidence": latest.confidence if latest is not None else 0.0,
        "readiness_checks": checks,
        "blockers": [blocker] if blocker else [],
        "artifact_count": artifact_count,
        "has_verification": latest is not None,
    }


@router.get("/readiness")
async def get_research_readiness(
    request: Request,
    project_id: str | None = Query(
        None, description="Project ID (default: the caller's)"
    ),
) -> dict[str, Any]:
    """Research readiness for one project, from its artifacts and latest
    verification report.

    It returned the module-global explorer report (project "default", counted
    from an in-memory model nothing fills) whatever project was asked for, so
    every client saw "not_ready" (TUI audit 2026-10-17).
    """
    await _explorer_caller(request)
    effective = _readable_project(request, project_id, "knowledge/readiness")
    from umrs.knowledge.store import KnowledgeStore

    try:
        pool = await knowledge_asyncpg_pool(request)
        artifacts, _total = await KnowledgeStore(pool).list_artifacts(
            project_id=effective
        )
        reports = await VerificationStore(pool).list_reports(
            project_id=effective, limit=1
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("readiness failed for project %s", effective)
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)
    latest = reports[0] if reports else None
    payload = project_readiness_payload(effective, len(artifacts), latest)
    payload.update(await _approval_fields(pool, effective, latest))
    return payload


async def _approval_fields(
    pool: Any, project_id: str, latest: Any | None
) -> dict[str, Any]:
    """The project's latest unexpired readiness approval, as payload fields.

    The approval holds only while the latest verification report is clean and
    its confidence is still at least the approved threshold; otherwise it is
    reported as ``approval_stale`` with ``user_approved`` false.  A failed
    lookup reads as "not approved" so readiness itself still answers.
    """
    try:
        approval = await ReadinessApprovalStore(pool).latest_active(project_id)
    except Exception:
        logger.warning(
            "readiness approval lookup failed for %s", project_id, exc_info=True
        )
        approval = None
    if approval is None:
        return {"user_approved": False, "approved_at": None}
    row = approval.to_dict()
    if not approval.holds_for(latest):
        return {
            "user_approved": False,
            "approved_at": None,
            "approval_stale": True,
            "approval_threshold": row["threshold"],
        }
    return {
        "user_approved": True,
        "approved_at": row["approved_at"],
        "approved_by": row["approved_by"],
        "approval_expires_at": row["expires_at"],
        "approval_threshold": row["threshold"],
    }


def _refuse_agent_approval(request: Request) -> None:
    """Readiness approval is the person's decision, never the agent's own."""
    from umrs.api._knowledge_mutation_guard import AGENT_ACTOR, request_actor

    if request_actor(request) == AGENT_ACTOR:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "approval_forbidden_for_agent",
                "message": "Agents may not approve readiness; the user runs "
                "the approval themselves.",
            },
        )


@router.post("/readiness/approve")
async def approve_project_readiness(
    body: ReadinessApproveRequest, request: Request
) -> dict[str, Any]:
    """Approve one project's knowledge base for autonomous operation.

    The caller must be able to write the project and must not be an agent
    actor (403 otherwise). One ``readiness_approvals`` row records the
    approver, the latest verification confidence as the threshold, the reason
    and an expiry; the readiness route reports the newest unexpired row as
    ``user_approved`` while the confidence stays at or above that threshold.
    # auth: verified
    """
    from umrs.api._scope_identity import request_caller_scope
    from umrs.knowledge.access_scope import project_writable

    _refuse_agent_approval(request)
    effective = _scoped_project(
        request, body.project_id, "knowledge/readiness/approve", project_writable
    )
    approver = request_caller_scope(request).user_id or "anonymous"
    try:
        pool = await knowledge_asyncpg_pool(request)
        reports = await VerificationStore(pool).list_reports(
            project_id=effective, limit=1
        )
        approval = await ReadinessApprovalStore(pool).record(
            project_id=effective,
            confidence=reports[0].confidence if reports else 0.0,
            approved_by=approver,
            reason=body.reason,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("readiness approval failed for project %s", effective)
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)
    logger.info(
        "AUDIT: readiness approval project=%s by=%s threshold=%.2f",
        effective,
        approver,
        approval.threshold,
    )
    return {"user_approved": True, **approval.to_dict()}


@router.post("/approval", status_code=410)
async def submit_approval(request: Request) -> dict[str, Any]:
    """Retired: use ``POST /umrs/knowledge/readiness/approve``.

    It held ONE approval for the whole server in the module-global explorer
    controller, with a client-asserted ``approved_by`` and no project (TUI
    audit 2026-10-17, round 2). The caller is still authenticated first.
    """
    from umrs.api._scope_identity import request_caller_scope

    request_caller_scope(request)
    raise HTTPException(
        status_code=410,
        detail={
            "error": "gone",
            "message": "Readiness approval is per project: "
            "POST /umrs/knowledge/readiness/approve {project_id, reason}.",
        },
    )


@router.post("/understand", response_model=UnderstandResponse)
async def understand_project(
    request: Request,
    body: UnderstandRequest | None = None,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> UnderstandResponse:
    """Extract structured knowledge and build the project knowledge graph.

    Agent-originated calls (x-r4-actor: agent) require an approved
    graph-extension proposal targeting the project's default graph (0069).
    The default graph is materialized so existing projects appear in the
    registry without manual migration.
    """
    from umrs.api._knowledge_mutation_guard import (
        enforce_graph_mutation_authorization,
        mark_proposal_executed,
        request_actor,
        request_session_id,
    )
    from umrs.knowledge.graph_registry import GraphRegistryStore, default_graph_id

    proposal_id = body.proposal_id if body else None
    enforce_graph_mutation_authorization(
        request,
        target_graph_id=default_graph_id(project_id),
        proposal_id=proposal_id,
    )

    project = build_research_project(project_id, config)
    try:
        project.understand()
    except Exception as exc:
        logger.exception("understand failed for project %s", project_id)
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)

    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is not None:
        GraphRegistryStore(pool).ensure_default_graph(project_id)
    mark_proposal_executed(
        request,
        proposal_id,
        actor=request_actor(request),
        session_id=request_session_id(request),
        operation="understand",
        extra={"project_id": project_id},
    )

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
        except Exception as exc:
            logger.exception("auto-understand failed for project %s", project_id)
            raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)

    try:
        answer = project.verify(body.question)
    except Exception as exc:
        logger.exception("verify failed for project %s", project_id)
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)
    return answer.to_dict()


# ── TASK-025: Knowledge Verification endpoints ────────────────────────────────


class TransitionFindingRequest(BaseModel):
    """Transition a finding to a new status."""

    finding_key: str = Field(..., description="Finding identifier")
    from_status: str = Field(..., description="Current status of the finding")
    new_status: str = Field(
        ...,
        description="New status: acknowledged | resolved | waived",
    )
    evidence: str = Field(
        "", description="Evidence for the transition", max_length=10000
    )


class ApproveRequest(BaseModel):
    """Approve knowledge with optional force override."""

    report_id: str = Field(..., description="Verification report to approve")
    force: bool = Field(
        False, description="Force approval despite low confidence (audited)"
    )


class ExportRequest(BaseModel):
    """Export knowledge bundle."""

    project_id: str = Field(..., description="Project to export")
    allow_unsigned: bool = Field(
        default=False,
        description="Produce a bundle even without UMRS_SECRET_KEY (manifest unsigned=true)",
    )
    include_raw: bool = Field(
        default=False,
        description="Carry the artifacts' raw bytes (owner/admin; catalog licence required)",
    )
    acknowledge_sensitive: bool = Field(
        default=False,
        description="With include_raw on a sensitive project: explicit, audited acknowledgement",
    )
    chunk_text: bool = Field(
        default=False,
        description="Carry each chunk's md5-verified full text (r4-kb v1.3, ADR-093 "
        "D-12); a sensitive project: its owner only, audited",
    )


class VerifyJobRequest(BaseModel):
    """Trigger knowledge verification job."""

    verifier_names: list[str] | None = Field(
        None, description="Specific verifier names to run"
    )


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def _approval_decision(
    *,
    force: bool,
    confidence_ok: bool,
    unresolved_count: int,
    confidence: float,
    threshold: float,
) -> tuple[bool, str]:
    """Approve / reject with reason; force always approves."""
    blocking_ok = unresolved_count == 0
    if force:
        return True, "approved (force)"
    if not confidence_ok and not blocking_ok:
        return False, (
            f"confidence {confidence:.2f} below threshold {threshold} and "
            f"{unresolved_count} blocking finding(s)"
        )
    if not confidence_ok:
        return False, f"confidence {confidence:.2f} below threshold {threshold}"
    if not blocking_ok:
        return False, f"{unresolved_count} unresolved blocking finding(s)"
    return True, "approved"


# NOTE: pool acquisition for knowledge stores lives in
# umrs.api._asyncpg_pool.knowledge_asyncpg_pool — the shared ``umrs_db_pool``
# is psycopg2 and MUST NOT be used with asyncpg stores (no acquire()).


def _report_summary(r) -> dict[str, Any]:
    """One verification report → API dict."""
    return {
        "report_id": r.report_id,
        "spec_name": r.spec_name,
        "version": r.version,
        "confidence": r.confidence,
        "total_checks": r.total_checks,
        "failed_checks": r.failed_checks,
        "degraded": r.degraded,
        "degraded_reason": r.degraded_reason,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def _readiness_from_latest(latest: dict[str, Any]) -> tuple[str, float]:
    """Map one report summary to (readiness_level, confidence)."""
    confidence = latest["confidence"]
    if latest["degraded"]:
        return "degraded", confidence
    if confidence >= 0.8:
        return "ready", confidence
    if confidence >= 0.5:
        return "partial", confidence
    return "insufficient", confidence


def _reconcile_payload(report: Any, *, sample_limit: int) -> dict[str, Any]:
    """Reconciliation report → API dict (kept async-free for unit tests)."""
    assets = sorted(report.assets, key=lambda a: (a.state, a.document_id))[
        :sample_limit
    ]
    return {
        "project_id": report.project_id,
        "catalogue_available": report.catalogue_available,
        "counts": report.counts,
        "orphaned_vector_rows": len(report.orphaned_vectors),
        "orphaned_chunk_nodes": len(report.orphaned_chunks),
        "assets_sampled": len(assets),
        "assets": [
            {
                "document_id": asset.document_id,
                "state": asset.state,
                "reasons": asset.reasons,
                "chunk_nodes": asset.chunk_count,
                "has_catalogue": asset.has_catalogue,
                "has_embeddings": asset.has_embeddings,
            }
            for asset in assets
        ],
    }


@router.get("/reconcile", response_model=dict[str, Any])
async def knowledge_reconcile(
    request: Request,
    sample: int = Query(50, ge=0, le=500),
    project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """ADE-031 plane reconciliation/checker (normative §Operational enforcement).

    Read-only scan over the catalogue/canonical/vector planes for the
    project. Output is the checker's authoritative consistency state: UI
    surfaces are read-side projections consuming this payload only.
    """
    await _explorer_caller(request)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.knowledge(
                "unavailable", "reconciliation requires the shared umrs_db pool"
            )
        )
    from umrs.knowledge.plane_reconciler import scan_project

    try:
        report = scan_project(pool, project_id)
    except Exception as exc:
        logger.exception("plane reconciliation failed for %s", project_id)
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)
    return _reconcile_payload(report, sample_limit=sample)


@router.get("/status", response_model=dict[str, Any])
async def knowledge_status(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    _get_project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Get knowledge status: artifacts, graph stats, verification state."""
    # ?project_id= was never compared with the caller (2026-10-17).
    project_id = _readable_project(request, project_id, "knowledge/status")
    try:
        from umrs.knowledge.store import KnowledgeStore

        pool = await knowledge_asyncpg_pool(request)
        store = KnowledgeStore(pool)
        artifact_rows, _total = await store.list_artifacts(project_id=project_id)
        artifacts = [dataclasses.asdict(a) for a in artifact_rows]
        active_count = sum(1 for a in artifacts if a.get("status") == "active")

        ver = VerificationStore(pool)
        reports = await ver.list_reports(project_id=project_id, limit=25)
        report_summaries = [_report_summary(r) for r in reports]

        readiness_level = "unverified"
        confidence = 0.0
        if report_summaries:
            readiness_level, confidence = _readiness_from_latest(report_summaries[0])

        return {
            "project_id": project_id,
            "artifacts_ingested": len(artifacts),
            "artifacts_active": active_count,
            "reports": report_summaries,
            "total_reports": len(reports),
            "readiness_level": readiness_level,
            "confidence": confidence,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get knowledge status")
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)


async def _list_findings_filtered(ver, project_id, report_id, status, severity):
    """Filter wrapper — keeps the handler body single-expression."""
    kwargs: dict[str, Any] = {
        "project_id": project_id,
        "status": status,
        "severity": severity,
        "limit": 500,
    }
    if report_id:
        kwargs["report_id"] = report_id
    return await ver.list_findings(**kwargs)


def _findings_payload(findings) -> dict[str, Any]:
    """Serialize findings + status-count summary."""
    findings_data = [_finding_dict(f) for f in findings]
    return {
        "findings": findings_data,
        "total": len(findings_data),
        "open_count": sum(1 for f in findings if f.status in ("open", "acknowledged")),
        "resolved_count": sum(1 for f in findings if f.status == "resolved"),
        "waived_count": sum(1 for f in findings if f.status == "waived"),
    }


def _finding_dict(f) -> dict[str, Any]:
    """One finding → API dict."""
    return {
        "finding_key": f.finding_key,
        "report_id": f.report_id,
        "category": f.category,
        "severity": f.severity,
        "status": f.status,
        "title": f.title,
        "description": f.description,
        "evidence": f.evidence,
        "artifact_id": f.artifact_id,
        "waived_evidence": f.waived_evidence,
        "waiver_expires": (f.waiver_expires.isoformat() if f.waiver_expires else None),
        "created_at": (f.created_at.isoformat() if f.created_at else None),
        "updated_at": (f.updated_at.isoformat() if f.updated_at else None),
    }


@router.get("/findings", response_model=dict[str, Any])
async def knowledge_findings(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    report_id: str | None = Query(None, description="Filter by report"),
    status: str | None = Query(None, description="Filter by status"),
    severity: str | None = Query(None, description="Filter by severity"),
    _get_project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """List verification findings with filters.

    It never filtered by project: any caller got up to 500 findings from every
    project (2026-10-17). Findings are now the readable project's only.
    """
    project_id = _readable_project(request, project_id, "knowledge/findings")
    try:
        pool = await knowledge_asyncpg_pool(request)
        ver = VerificationStore(pool)

        findings = await _list_findings_filtered(
            ver, project_id, report_id, status, severity
        )
        return _findings_payload(findings)
    except Exception as exc:
        logger.exception("Failed to list findings")
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)


async def _report_in_scope(ver: Any, report_id: str, caller: Any, allowed: Any) -> Any:
    """The report when ``allowed(report.project_id, caller)``, else None."""
    report = await ver.get_report(report_id)
    if report is None or not allowed(report.project_id, caller):
        return None
    return report


async def _require_writable_finding(
    ver: Any, finding_key: str, request: Request
) -> None:
    """404 unless the finding's report is in a project the caller may write.

    The route wrote any project's finding by key (H-1 A.2, 2026-10-17); a
    finding of another project reads as not found, as an absent one does.
    """
    from umrs.api._scope_identity import request_caller_scope
    from umrs.knowledge.access_scope import project_writable

    caller = request_caller_scope(request)
    try:
        finding = await ver.get_finding(finding_key)
    except ValueError:
        finding = None
    report = (
        await _report_in_scope(ver, finding.report_id, caller, project_writable)
        if finding is not None
        else None
    )
    if report is None:
        raise_typed(
            PlatformError.knowledge("not_found", f"Finding not found: {finding_key}")
        )


@router.patch("/findings/{finding_key}", response_model=dict[str, Any])
async def transition_finding(
    finding_key: str,
    body: TransitionFindingRequest,
    request: Request,
    _get_project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Transition a finding status (acknowledged, resolved, waived) in a
    project the caller may write."""
    try:
        pool = await knowledge_asyncpg_pool(request)
        ver = VerificationStore(pool)
        await _require_writable_finding(ver, finding_key, request)

        finding = await ver.transition_finding(
            finding_key=finding_key,
            from_status=body.from_status,
            to_status=body.new_status,
        )

        # Store evidence for the transition
        if body.evidence and body.new_status == "waived":
            await ver.waive_finding(
                finding_key=finding_key,
                evidence=body.evidence,
            )

        return {
            "finding_key": finding.finding_key,
            "status": finding.status,
            "report_id": finding.report_id,
            "updated_at": (
                finding.updated_at.isoformat() if finding.updated_at else None
            ),
        }
    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Finding transition failed: %s", exc)
        raise_typed(PlatformError.knowledge("bad_request", str(exc)), cause=exc)
    except Exception as exc:
        logger.exception("Failed to transition finding")
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)


@router.post("/approve", response_model=dict[str, Any])
async def approve_knowledge(
    body: ApproveRequest,
    request: Request,
    _get_project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Approve knowledge with confidence threshold and blocking findings check.

    Force approval is audited (logged). The report must be in a project the
    caller may read: any report was read by id (H-1 A.2, 2026-10-17); another
    project's reads as not found.
    """
    from umrs.api._scope_identity import request_caller_scope
    from umrs.knowledge.access_scope import project_readable

    report_id = body.report_id
    force = body.force
    confidence_threshold = 0.8

    try:
        caller = request_caller_scope(request)
        pool = await knowledge_asyncpg_pool(request)
        ver = VerificationStore(pool)
        report = await _report_in_scope(ver, report_id, caller, project_readable)

        if report is None:
            raise_typed(
                PlatformError.knowledge("not_found", f"Report not found: {report_id}")
            )

        # Confidence check
        confidence_ok = report.confidence >= confidence_threshold

        # Blocking findings check (unresolved only)
        blocking = await ver.list_blocking_findings(report_id)
        unresolved_blocking = [
            b for b in blocking if b.status not in ("resolved", "waived")
        ]

        approved, reason = _approval_decision(
            force=force,
            confidence_ok=confidence_ok,
            unresolved_count=len(unresolved_blocking),
            confidence=report.confidence,
            threshold=confidence_threshold,
        )
        if force:
            logger.info(
                "AUDIT: Force approve report=%s confidence=%.2f "
                "threshold=%.2f blocking=%d",
                report_id,
                report.confidence,
                confidence_threshold,
                len(unresolved_blocking),
            )

        return {
            "approved": approved,
            "report_id": report_id,
            "confidence": report.confidence,
            "threshold": confidence_threshold,
            "blocking_findings": len(unresolved_blocking),
            "reason": reason,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to approve knowledge")
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)


@router.get("/readiness/graph", response_model=dict[str, Any])
async def knowledge_readiness(
    request: Request,
    project_id: str = Query(..., description="Project ID"),
    _get_project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Get knowledge readiness: artifact count + verification status.
    # auth: verified

    Renamed from ``GET /readiness`` (shadowed duplicate; G-P2-9): the
    research-readiness controller handler registered earlier owns that path.
    """
    # ?project_id= was never compared with the caller (2026-10-17).
    project_id = _readable_project(request, project_id, "knowledge/readiness/graph")
    try:
        from umrs.knowledge.store import KnowledgeStore

        pool = await knowledge_asyncpg_pool(request)
        store = KnowledgeStore(pool)
        artifact_rows, _total = await store.list_artifacts(project_id=project_id)
        artifacts = [dataclasses.asdict(a) for a in artifact_rows]
        active_count = sum(1 for a in artifacts if a.get("status") == "active")

        ver = VerificationStore(pool)
        reports = await ver.list_reports(project_id=project_id, limit=1)

        readiness_level = "unverified"
        confidence = 0.0
        if reports:
            latest = reports[0]
            confidence = latest.confidence
            if latest.degraded:
                readiness_level = "degraded"
            elif latest.confidence >= 0.8:
                readiness_level = "ready"
            elif latest.confidence >= 0.5:
                readiness_level = "partial"
            else:
                readiness_level = "insufficient"

        return {
            "project_id": project_id,
            "artifact_count": len(artifacts),
            "artifacts_active": active_count,
            "readiness_level": readiness_level,
            "confidence": confidence,
            "has_verification": len(reports) > 0,
        }
    except Exception as exc:
        logger.exception("Failed to get knowledge readiness")
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)


def _raw_caller(request: Request) -> CallerIdentity:
    """The verified caller (gateway headers, HMAC-checked in strict mode);
    resolved only for raw exports — plain exports keep their contract."""
    headers = request.headers
    return get_caller_identity(
        caller_user_id=None,
        caller_project_id=None,
        is_admin=False,
        x_user_id=headers.get("x-user-id"),
        x_verified_project_id=headers.get("x-verified-project-id"),
        x_verified_is_admin=headers.get("x-verified-is-admin"),
        x_verified_hmac=headers.get("x-verified-hmac"),
    )


def _authorize_raw(request: Request, body: ExportRequest) -> dict[str, Any]:
    """ADR-032 raw-materials gate (before the job; typed refusals). The
    acknowledgement of a sensitive project's raw release is audited."""
    if not body.include_raw:
        return {}
    from ._raw_export_gate import authorize_raw_export
    from .audit_events import emit_audit

    identity = _raw_caller(request)
    ack = authorize_raw_export(
        getattr(request.app.state, "umrs_db_pool", None),
        project_id=body.project_id,
        user_id=identity.user_id,
        is_admin=identity.is_admin,
        acknowledge_sensitive=body.acknowledge_sensitive,
    )
    emit_audit(
        "knowledge.export.raw",
        actor=identity.user_id,
        user_id=identity.user_id,
        project_id=body.project_id,
        details={"acknowledgement": ack} if ack else {"data_class": "non_sensitive"},
    )
    return ack


def _authorize_chunk_text(
    request: Request, body: ExportRequest
) -> tuple[dict[str, Any], str]:
    """ADR-093 D-12 / MD-5 chunk-text gate (before the job; typed refusals):
    ``(acknowledgement, verified caller)``. The release is audited; the job
    is submitted under the verified caller, whom the export engine checks
    against the project's owner again (it trusts no acknowledgement a spec
    carries, review chunk-data-2)."""
    if not body.chunk_text:
        return {}, ""
    from ._raw_export_gate import authorize_chunk_text_export
    from .audit_events import emit_audit

    identity = _raw_caller(request)
    ack = authorize_chunk_text_export(
        getattr(request.app.state, "umrs_db_pool", None),
        project_id=body.project_id,
        user_id=identity.user_id,
    )
    emit_audit(
        "knowledge.export.chunk_text",
        actor=identity.user_id,
        user_id=identity.user_id,
        project_id=body.project_id,
        details={"acknowledgement": ack} if ack else {"data_class": "non_sensitive"},
    )
    return ack, identity.user_id


@router.post("/export", response_model=dict[str, Any])
async def export_knowledge(
    body: ExportRequest,
    request: Request,
    _get_project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Trigger knowledge bundle export (.kb format).

    The export publishes a downloadable archive (D-40 transport), so it acts
    on the caller's verified project only — a body naming another project is
    refused, never silently exported.
    """
    if body.project_id != _get_project_id:
        raise_typed(
            PlatformError.knowledge(
                "project_mismatch",
                f"export of project {body.project_id!r} refused: the caller's "
                f"verified project is {_get_project_id!r}",
            )
        )
    raw_ack = _authorize_raw(request, body)
    chunk_ack, submitter = _authorize_chunk_text(request, body)
    try:
        from umrs.server.durable_job_runtime import get_runtime

        runtime = get_runtime()
        if runtime is None:
            raise_typed(
                PlatformError.knowledge(
                    "unavailable", "UMRS durable job runtime not available"
                )
            )

        spec: dict[str, Any] = {
            "project_id": body.project_id,
            "allow_unsigned": body.allow_unsigned,
            # Each export is a new snapshot: without a per-request id the
            # spec-derived idempotency key collided with the project's first
            # export forever ("Job already exists", observed live 2026-09-24).
            "request_id": uuid4().hex,
            "include_raw": body.include_raw,
            "raw_acknowledgement": raw_ack,
            "chunk_text": body.chunk_text,
            "chunk_text_acknowledgement": chunk_ack,
        }

        # Unified ingest governance (R6.4).
        from .governance import guard_ingest

        guard_ingest(
            body.project_id,
            "knowledge.export",
            user_id=request.headers.get("x-user-id", ""),
            details={"kind": "export"},
        )

        # A plain export is owned by its caller too: submitted under "" its
        # job was unreadable to the very user who started it (403 on
        # GET /jobs/{id}, found 2026-10-12 running ADR-093 S8).
        job_id = await runtime.submit_job_sync(
            job_type=JobType.KNOWLEDGE_EXPORT,
            project_id=body.project_id,
            spec=spec,
            user_id=submitter or request.headers.get("x-user-id", ""),
        )

        logger.info("Knowledge export queued: %s", job_id)
        from umrs.knowledge.bundle_archive import export_archive_key

        return {
            "job_id": job_id,
            "bundle_id": job_id,
            "status": "processing",
            # Downloadable once the job succeeds (r4 files download <key>).
            "archive_key": export_archive_key(body.project_id, job_id),
        }
    except Exception as exc:
        logger.exception("Failed to queue knowledge export")
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)


@router.post("/verify/trigger", response_model=dict[str, Any])
async def trigger_verify(
    body: VerifyJobRequest = VerifyJobRequest(),
    project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Trigger a KNOWLEDGE_VERIFY durable job.
    # auth: verified

    Runs all verifiers in the registry against the project's knowledge.
    Renamed from ``POST /verify`` (shadowed duplicate; G-P2-9): the
    synchronous QA-with-evidence handler registered earlier owns that path.
    """
    try:
        from umrs.server.durable_job_runtime import get_runtime

        runtime = get_runtime()
        if runtime is None:
            raise_typed(
                PlatformError.knowledge(
                    "unavailable", "UMRS durable job runtime not available"
                )
            )

        spec: dict[str, Any] = {
            "project_id": project_id,
            "verifier_names": (body.verifier_names if body.verifier_names else None),
            "request_id": uuid4().hex,  # re-runnable (see export's spec)
        }

        job_id = await runtime.submit_job_sync(
            job_type=JobType.KNOWLEDGE_VERIFY,
            project_id=project_id,
            spec=spec,
        )

        logger.info("Knowledge verify queued: %s", job_id)
        return {
            "job_id": job_id,
            "job_type": JobType.KNOWLEDGE_VERIFY.value,
            "status": "queued",
        }
    except Exception as exc:
        logger.exception("Failed to queue knowledge verify")
        raise_typed(PlatformError.knowledge("failure", str(exc)), cause=exc)
