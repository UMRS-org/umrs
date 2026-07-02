"""UMRS API routes for knowledge search and RC1 project workflow."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from umrs.api.files import S3Config, get_project_id, get_s3_config
from umrs.knowledge.project import ProjectPhase, ResearchProject

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
    if project.phase not in (ProjectPhase.READY, ProjectPhase.PLANNING, ProjectPhase.RESEARCHING, ProjectPhase.COMPLETED):
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
