"""Knowledge API router — durable job layer routes.

Routes for triggering KNOWLEDGE_INGEST, KNOWLEDGE_VERIFY durable jobs,
listing artifacts, and running lint checks.  These complement the
M3.5 knowledge routes in ``api/knowledge.py`` (which use the
InMemoryKnowledgeGraph research model).

Prefix: ``/umrs/k2`` (versioned to avoid conflict with existing ``/umrs/knowledge``).

No umrs.api or umrs.server imports — strict boundary.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from umrs.api.files import S3Config, get_project_id, get_s3_config
from umrs.jobs.models import JobType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/k2", tags=["knowledge-v2"])

# -- Pydantic models --


class IngestRequest(BaseModel):
    sources: list[dict[str, Any]] = Field(..., description="Source specs (kind, uri/path)")
    project_id: Optional[str] = Field(None, description="Project ID override")
    allow_domains: Optional[list[str]] = Field(None, description="Domain allowlist for remote sources")


class LintRequest(BaseModel):
    files: Optional[list[str]] = Field(None, description="List of file paths to lint")
    source_code: Optional[dict[str, str]] = Field(
        None, description="Map of file_path -> source_code content"
    )


class LintResponse(BaseModel):
    findings: list[dict[str, Any]]
    total: int
    blocking: int
    warnings: int
    info: int


class ArtifactListResponse(BaseModel):
    artifacts: list[dict[str, Any]]
    total: int
    page: int
    page_size: int


class EventListResponse(BaseModel):
    events: list[dict[str, Any]]
    total: int


class VerifyRequest(BaseModel):
    project_id: Optional[str] = Field(None, description="Project to verify")


class VerifyResponse(BaseModel):
    job_id: str
    status: str  # queued | running | completed | failed


# -- Routes --


@router.post("/ingest")
async def trigger_ingest(
    body: IngestRequest,
    project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Trigger a KNOWLEDGE_INGEST durable job.

    Accepts source specs, optionally with allow_domains for remote sources.
    Returns job_id for tracking.
    """
    from umrs.server.durable_job_runtime import get_runtime

    runtime = get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="UMRS durable job runtime not available")

    sources = body.sources if body.sources else []
    allow_domains = body.allow_domains if body.allow_domains else []

    spec: dict[str, Any] = {
        "sources": sources,
        "allow_domains": allow_domains,
        "project_id": project_id,
    }

    try:
        job_id = await runtime.submit_job_sync(
            job_type=JobType.KNOWLEDGE_INGEST,
            project_id=project_id,
            spec=spec,
        )
        logger.info("Knowledge ingest queued: %s", job_id)
        return {"job_id": job_id, "job_type": JobType.KNOWLEDGE_INGEST.value, "status": "queued"}
    except Exception as exc:
        logger.exception("Failed to queue knowledge ingest")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/artifacts")
async def list_artifacts(
    project_id: str = Depends(get_project_id),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    kind: Optional[str] = Query(None, description="Filter by artifact kind"),
    status: Optional[str] = Query(None, description="Filter by status"),
) -> ArtifactListResponse:
    """List knowledge artifacts (paginated)."""
    try:
        from umrs.knowledge.store import KnowledgeStore
        from umrs.server.job_tracking_handler import build_durable_job_db_pool_from_env

        pool = build_durable_job_db_pool_from_env()
        if pool is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        store = KnowledgeStore(pool)
        # TODO: implement paged list_artifacts with filters
        artifacts = await store.list_artifacts(project_id=project_id)

        total = len(artifacts)
        start = (page - 1) * page_size
        end = start + page_size
        page_artifacts = artifacts[start:end]

        return ArtifactListResponse(
            artifacts=page_artifacts,
            total=total,
            page=page,
            page_size=page_size,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list artifacts")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/lint")
async def run_lint(
    body: LintRequest,
    project_id: str = Depends(get_project_id),
) -> LintResponse:
    """Run lint checks on files.

    Accepts either file paths or source_code dict.
    Returns structured findings.
    """
    try:
        from umrs.knowledge.lint_engine import (
            check_complexity,
            check_docstrings,
            check_import_boundary,
            check_syntax,
            check_unused_variables,
        )

        all_findings: list[dict[str, Any]] = []

        if body.source_code:
            for file_path, source in body.source_code.items():
                all_findings.extend(check_syntax(source, file_path))
                all_findings.extend(check_complexity(source, file_path))
                all_findings.extend(check_unused_variables(source, file_path))
                all_findings.extend(check_import_boundary(source, file_path))
                all_findings.extend(check_docstrings(source, file_path))
        elif body.files:
            for file_path in body.files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    all_findings.extend(check_syntax(source, file_path))
                    all_findings.extend(check_complexity(source, file_path))
                    all_findings.extend(check_unused_variables(source, file_path))
                    all_findings.extend(check_import_boundary(source, file_path))
                    all_findings.extend(check_docstrings(source, file_path))
                except OSError:
                    pass

        blocking = sum(1 for f in all_findings if f["severity"] == "BLOCKING")
        warnings = sum(1 for f in all_findings if f["severity"] == "WARNING")
        info = sum(1 for f in all_findings if f["severity"] == "INFO")

        return LintResponse(
            findings=all_findings,
            total=len(all_findings),
            blocking=blocking,
            warnings=warnings,
            info=info,
        )
    except Exception as exc:
        logger.exception("Lint check failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/verify", response_model=VerifyResponse)
async def trigger_verify(
    body: VerifyRequest = VerifyRequest(),
    project_id: str = Depends(get_project_id),
) -> VerifyResponse:
    """Trigger a KNOWLEDGE_VERIFY durable job.

    Runs all verifiers in the registry against the project's knowledge.
    """
    from umrs.server.durable_job_runtime import get_runtime

    runtime = get_runtime()
    if runtime is None:
        raise HTTPException(status_code=503, detail="UMRS durable job runtime not available")

    spec: dict[str, Any] = {
        "project_id": project_id,
        "verifier_names": body.verifier_names if hasattr(body, "verifier_names") and body.verifier_names else None,
    }

    try:
        job_id = await runtime.submit_job_sync(
            job_type=JobType.KNOWLEDGE_VERIFY,
            project_id=project_id,
            spec=spec,
        )
        logger.info("Knowledge verify queued: %s", job_id)
        return VerifyResponse(job_id=job_id, status="queued")
    except Exception as exc:
        logger.exception("Failed to queue knowledge verify")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/readiness")
async def knowledge_readiness(
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> dict[str, Any]:
    """Get knowledge readiness score for a project.

    Combines artifact count, graph coverage, and verification status.
    """
    try:
        from umrs.knowledge.store import KnowledgeStore
        from umrs.server.job_tracking_handler import build_durable_job_db_pool_from_env

        pool = build_durable_job_db_pool_from_env()
        if pool is None:
            raise HTTPException(status_code=503, detail="Database unavailable")

        store = KnowledgeStore(pool)
        artifacts = await store.list_artifacts(project_id=project_id)
        active_count = sum(1 for a in artifacts if a.get("status") == "active")

        # TODO: Add graph coverage and verification status
        return {
            "project_id": project_id,
            "artifact_count": len(artifacts),
            "active_artifacts": active_count,
            "readiness_score": round(min(1.0, active_count / max(1, active_count)), 2),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Knowledge readiness check failed")
        raise HTTPException(status_code=500, detail=str(exc))
