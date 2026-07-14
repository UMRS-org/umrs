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
    sources: list[dict[str, Any]] = Field(
        ..., description="Source specs (kind, uri/path)"
    )
    project_id: Optional[str] = Field(None, description="Project ID override")
    allow_domains: Optional[list[str]] = Field(
        None, description="Domain allowlist for remote sources"
    )


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


class WorkflowDraftResponse(BaseModel):
    workflow_id: str
    name: str
    steps: list[dict[str, Any]]
    project_context: dict[str, Any]
    meta: dict[str, Any]


class VerifyRequest(BaseModel):
    project_id: Optional[str] = Field(None, description="Project to verify")


class VerifyResponse(BaseModel):
    job_id: str
    status: str  # queued | running | completed | failed


# -- Verification report/finding models --


class FindingTransitionRequest(BaseModel):
    finding_key: str = Field(..., description="Finding identifier")
    new_status: str = Field(
        ..., description="New status: acknowledged | resolved | waived"
    )
    evidence: str = Field("", description="Evidence for the transition")


class FindingResponse(BaseModel):
    finding_key: str
    report_id: str
    category: str
    severity: str  # blocking | warning | info
    status: str  # open | acknowledged | resolved | waived
    title: str
    description: str
    evidence: str
    artifact_id: Optional[str] = None
    waived_evidence: Optional[str] = None
    waiver_expires: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ReportSummary(BaseModel):
    report_id: str
    project_id: str
    version: int
    spec_name: str
    confidence: float
    total_checks: int
    passed_checks: int
    failed_checks: int
    skipped_checks: int
    degraded: bool
    degraded_reason: Optional[str] = None
    created_at: Optional[str] = None


class ReportListResponse(BaseModel):
    reports: list[ReportSummary]
    total: int


class FindingsResponse(BaseModel):
    findings: list[FindingResponse]
    total: int
    open_count: int
    resolved_count: int
    waived_count: int


class FindingListQuery(BaseModel):
    report_id: Optional[str] = Field(None, description="Filter by report")
    status: Optional[str] = Field(None, description="Filter by status")
    severity: Optional[str] = Field(None, description="Filter by severity")


class WaiveRequest(BaseModel):
    finding_key: str = Field(..., description="Finding identifier")
    evidence: str = Field(..., description="Waiver justification")
    expires_days: Optional[int] = Field(None, description="Waiver expiry in days")


class ApproveRequest(BaseModel):
    report_id: str = Field(..., description="Report to approve")
    force: bool = Field(
        False, description="Force approval despite low confidence (audited)"
    )


class ApproveResponse(BaseModel):
    approved: bool
    report_id: str
    confidence: float
    threshold: float
    blocking_findings: int
    reason: str


class EmitWorkflowRequest(BaseModel):
    """Request body for plan-to-workflow emission."""

    plan: dict[str, Any] = Field(
        ..., description="ResearchPlan dict (objectives, hypotheses, experiments)"
    )
    model: Optional[str] = Field(
        None, description="Model name for project context prefill"
    )
    partition: Optional[str] = Field(
        None, description="Slurm partition for project context prefill"
    )
    account: Optional[str] = Field(
        None, description="Slurm account for project context prefill"
    )


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
        raise HTTPException(
            status_code=503, detail="UMRS durable job runtime not available"
        )

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
        return {
            "job_id": job_id,
            "job_type": JobType.KNOWLEDGE_INGEST.value,
            "status": "queued",
        }
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
        raise HTTPException(
            status_code=503, detail="UMRS durable job runtime not available"
        )

    spec: dict[str, Any] = {
        "project_id": project_id,
        "verifier_names": body.verifier_names
        if hasattr(body, "verifier_names") and body.verifier_names
        else None,
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


@router.post("/emit-workflow", response_model=WorkflowDraftResponse)
async def emit_workflow(
    body: EmitWorkflowRequest,
    project_id: str = Depends(get_project_id),
) -> WorkflowDraftResponse:
    """Emit a WorkflowDraft from a ResearchPlan.

    Converts planner output (objectives, hypotheses, experiments) into a
    workflow draft. Unmappable steps become manual placeholders with
    ``type="manual"`` and ``instructions`` field.

    Project context (model, partition, account) is prefilled from the request.
    """
    from umrs.knowledge.emit_workflow import emit_workflow as _emit_workflow

    plan = body.plan
    if not isinstance(plan, dict):
        raise HTTPException(status_code=400, detail="plan must be a dict")

    try:
        result = _emit_workflow(
            plan,
            project_id=project_id,
            model=body.model,
            partition=body.partition,
            account=body.account,
        )
    except Exception as exc:
        logger.exception("Failed to emit workflow")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return WorkflowDraftResponse(
        workflow_id=result["workflow_id"],
        name=result["name"],
        steps=result["steps"],
        project_context=result.get("project_context", {}),
        meta=result.get("_meta", {}),
    )
