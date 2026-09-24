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
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from umrs.api.files import get_project_id
from umrs.jobs.models import JobType
from umrs.knowledge.graph_registry import default_graph_id
from umrs.shared.storage.s3_config import S3Config, get_s3_config

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/k2", tags=["knowledge-v2"])

# -- Pydantic models --


class IngestRequest(BaseModel):
    sources: list[dict[str, Any]] = Field(
        ..., description="Source specs (kind, uri/path)"
    )
    project_id: str | None = Field(None, description="Project ID override")
    allow_domains: list[str] | None = Field(
        None, description="Domain allowlist for remote sources"
    )
    proposal_id: str | None = Field(
        None,
        description=(
            "Approved graph-extension proposal id — required when the caller "
            "is an agent (x-r4-actor: agent); human-initiated calls omit it"
        ),
    )
    graph_id: str | None = Field(
        None,
        description=(
            "Target knowledge graph (id or name) for this ingest. When absent, "
            "knowledge lands in the project's default graph. Named graphs must "
            "exist and be active; their registry entry may carry a different "
            "or no project association for shared graphs."
        ),
    )


class LintRequest(BaseModel):
    files: list[str] | None = Field(None, description="List of file paths to lint")
    source_code: dict[str, str] | None = Field(
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
    project_id: str | None = Field(None, description="Project to verify")


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
    artifact_id: str | None = None
    waived_evidence: str | None = None
    waiver_expires: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


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
    degraded_reason: str | None = None
    created_at: str | None = None


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
    report_id: str | None = Field(None, description="Filter by report")
    status: str | None = Field(None, description="Filter by status")
    severity: str | None = Field(None, description="Filter by severity")


class WaiveRequest(BaseModel):
    finding_key: str = Field(..., description="Finding identifier")
    evidence: str = Field(..., description="Waiver justification")
    expires_days: int | None = Field(None, description="Waiver expiry in days")


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
    model: str | None = Field(
        None, description="Model name for project context prefill"
    )
    partition: str | None = Field(
        None, description="Slurm partition for project context prefill"
    )
    account: str | None = Field(
        None, description="Slurm account for project context prefill"
    )


# -- Routes --


def _resolve_ingest_target(
    request: Request, project_id: str, graph_ref: str | None
) -> str:
    """Resolve the ingest target graph (default: the project's default graph).

    Raises 400 for unknown archived/missing named graphs (a typo must never
    silently reinterpret into the default graph).
    """
    from umrs.knowledge.graph_registry import GraphRegistryStore

    if not graph_ref:
        return default_graph_id(project_id)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.knowledge2(
                "unavailable",
                "Graph registry requires the database (named-graph target)",
            )
        )
    graph = GraphRegistryStore(pool).resolve_graph(graph_ref)
    if graph is None:
        raise_typed(
            PlatformError.knowledge2(
                "unknown_graph", f"Unknown target graph {graph_ref!r}"
            )
        )
    if graph.status != "active":
        raise_typed(
            PlatformError.knowledge2(
                "graph_not_active", f"Graph {graph_ref!r} is {graph.status}"
            )
        )
    return graph.graph_id


# Source kinds a REMOTE caller may name on the HTTP ingest route. The job
# executes inside the server, so server-path kinds (local/local_file/
# local_dir — and a missing kind, which the job defaults to local_file) would
# read the SERVER's filesystem into the caller's project, and a caller-named
# s3_uri would read any bucket key with the server's credentials. Workstation
# files ride inline_text (small UTF-8) or POST /umrs/knowledge/upload (bytes
# staged under the caller's project prefix, which submits its own s3_uri job
# server-side). Remote kinds stay egress-gated by allow_domains.
REMOTE_INGEST_SOURCE_KINDS = frozenset({"inline_text", "http_url", "git_repo"})


def _refuse_server_side_source_kinds(sources: list[dict[str, Any]]) -> None:
    """Typed 400 naming the first source whose kind is not remote-safe."""
    for index, src in enumerate(sources):
        kind = str((src or {}).get("kind") or "")
        if kind not in REMOTE_INGEST_SOURCE_KINDS:
            raise_typed(
                PlatformError.knowledge2(
                    "source_kind_refused",
                    f"sources[{index}] kind {kind or '(missing)'!r} is not accepted "
                    f"over HTTP (accepted: {sorted(REMOTE_INGEST_SOURCE_KINDS)}). "
                    "Paths resolve on the UMRS server, not your workstation — send "
                    "file bytes via POST /umrs/knowledge/upload or small text as "
                    "inline_text.",
                )
            )


@router.post("/ingest")
async def trigger_ingest(
    body: IngestRequest,
    request: Request,
    project_id: str = Depends(get_project_id),
) -> dict[str, Any]:
    """Trigger a KNOWLEDGE_INGEST durable job.

    Accepts source specs, optionally with allow_domains for remote sources.
    Returns job_id for tracking. Agent-originated ingests require an approved
    graph-extension proposal (0069 mutation guard).
    """
    from umrs.api._knowledge_mutation_guard import (
        enforce_graph_mutation_authorization,
        mark_proposal_executed,
        request_actor,
        request_session_id,
    )
    from umrs.server.durable_job_runtime import get_runtime

    if body.project_id and body.project_id != project_id:
        # Fail loud instead of silently writing into the verified (JWT)
        # project — the historic silent-override was an audit P1: the UI
        # showed the selected project while data landed elsewhere.
        raise HTTPException(
            status_code=409,
            detail={
                "error": "project_mismatch",
                "message": (
                    f"Requested project {body.project_id!r} does not match the "
                    f"authenticated project {project_id!r}. Re-login with a "
                    "project-scoped token or omit the override."
                ),
                "requested_project_id": body.project_id,
                "verified_project_id": project_id,
            },
        )

    _refuse_server_side_source_kinds(body.sources or [])

    runtime = get_runtime()
    if runtime is None:
        raise_typed(
            PlatformError.knowledge2(
                "unavailable", "UMRS durable job runtime not available"
            )
        )

    target_graph_id = _resolve_ingest_target(request, project_id, body.graph_id)
    enforce_graph_mutation_authorization(
        request,
        target_graph_id=target_graph_id,
        proposal_id=body.proposal_id,
    )

    sources = body.sources if body.sources else []
    allow_domains = body.allow_domains if body.allow_domains else []

    # Unified ingest governance (R6.4): quota + audit before job submit.
    from umrs.api.governance import guard_ingest

    guard_ingest(
        project_id,
        "knowledge.ingest",
        user_id=request.headers.get("x-user-id", ""),
        details={
            "sources": len(sources),
            "graph_id": target_graph_id,
            "allow_domains": allow_domains,
        },
    )

    spec: dict[str, Any] = {
        "sources": sources,
        "allow_domains": allow_domains,
        "project_id": project_id,
        # Always carry the resolved target graph: without it the understand
        # chain tagged nodes graph_id='' (shadow graph), making fresh
        # ingestion invisible to knowledge/query — audit B2.
        "graph_id": target_graph_id,
        # D-35 producer leg: agent-originated extractions land their
        # nodes/edges as 'proposed' (the understand chain reads this).
        "origin_actor": request_actor(request),
    }
    if body.proposal_id:
        spec["proposal_id"] = body.proposal_id

    try:
        job_id = await runtime.submit_job_sync(
            job_type=JobType.KNOWLEDGE_INGEST,
            project_id=project_id,
            spec=spec,
        )
        logger.info("Knowledge ingest queued: %s", job_id)
        mark_proposal_executed(
            request,
            body.proposal_id,
            actor=request_actor(request),
            session_id=request_session_id(request),
            operation="knowledge_ingest",
            extra={"job_id": job_id},
        )
        return {
            "job_id": job_id,
            "job_type": JobType.KNOWLEDGE_INGEST.value,
            "status": "queued",
        }
    except Exception as exc:
        logger.exception("Failed to queue knowledge ingest")
        raise_typed(PlatformError.knowledge2("failure", str(exc)), cause=exc)


@router.get("/artifacts")
async def list_artifacts(
    request: Request,
    project_id: str = Depends(get_project_id),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=100),
    kind: str | None = Query(None, description="Filter by artifact kind"),
    status: str | None = Query(None, description="Filter by status"),
) -> ArtifactListResponse:
    """List knowledge artifacts (paginated)."""
    try:
        # Reuse the shared app-level pool (built once in http_sidecar.py) —
        # do NOT call build_durable_job_db_pool_from_env() here, which opens
        # a brand-new connection pool on every request and leaks it.
        import dataclasses

        from umrs.api._asyncpg_pool import knowledge_asyncpg_pool
        from umrs.knowledge.store import KnowledgeStore

        pool = await knowledge_asyncpg_pool(request)
        store = KnowledgeStore(pool)
        artifact_rows, total = await store.list_artifacts(
            project_id=project_id, page=page, page_size=page_size
        )
        page_artifacts = [dataclasses.asdict(a) for a in artifact_rows]

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
        raise_typed(PlatformError.knowledge2("failure", str(exc)))


def _lint_source(checks: list, source: str, file_path: str) -> list[dict[str, Any]]:
    """Run every lint check over one source blob."""
    findings: list[dict[str, Any]] = []
    for check in checks:
        findings.extend(check(source, file_path))
    return findings


def _lint_request_checks() -> list:
    """Lazily import the lint checks (keeps module import cheap)."""
    from umrs.knowledge.lint_engine import (
        check_complexity,
        check_docstrings,
        check_import_boundary,
        check_syntax,
        check_unused_variables,
    )

    return [
        check_syntax,
        check_complexity,
        check_unused_variables,
        check_import_boundary,
        check_docstrings,
    ]


def _collect_lint_findings(body: LintRequest) -> list[dict[str, Any]]:
    """Run lint checks over the request payload (source_code or files)."""
    checks = _lint_request_checks()
    findings: list[dict[str, Any]] = []
    if body.source_code:
        for file_path, source in body.source_code.items():
            findings.extend(_lint_source(checks, source, file_path))
        return findings
    for file_path in body.files or []:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            findings.extend(_lint_source(checks, source, file_path))
        except OSError:
            pass
    return findings


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
        all_findings = _collect_lint_findings(body)
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
        raise_typed(PlatformError.knowledge2("failure", str(exc)))


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
        raise_typed(
            PlatformError.knowledge2(
                "unavailable", "UMRS durable job runtime not available"
            )
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
        raise_typed(PlatformError.knowledge2("failure", str(exc)))


def _verified_artifact_count(rows: Any) -> int:
    """Artifacts whose metadata carries an active verification marker.

    Honesty contract: the 'verified' iv age is metadata.status == 'verified'
    or metadata.verified (truthy); rows missing the field stay uncounted
    (no invented status).
    """
    if not rows:
        return 0
    count = 0
    for row in rows:
        metadata = (
            getattr(row, "metadata_", None) or getattr(row, "metadata", None) or {}
        )
        if not isinstance(metadata, dict):
            continue
        if metadata.get("status") == "verified" or metadata.get("verified") is True:
            count += 1
    return count


@router.get("/readiness")
async def knowledge_readiness(
    request: Request,
    project_id: str = Depends(get_project_id),
    config: S3Config = Depends(get_s3_config),
) -> dict[str, Any]:
    """Get knowledge readiness score for a project.

    Combines artifact count, graph coverage, and verification status.
    """
    try:
        # Reuse the shared app-level pool (built once in http_sidecar.py) —
        # do NOT call build_durable_job_db_pool_from_env() here, which opens
        # a brand-new connection pool on every request and leaks it.
        from umrs.api._asyncpg_pool import knowledge_asyncpg_pool
        from umrs.knowledge.store import KnowledgeStore

        pool = await knowledge_asyncpg_pool(request)
        store = KnowledgeStore(pool)
        artifact_rows, _total = await store.list_artifacts(project_id=project_id)
        active_count = sum(1 for a in artifact_rows if a.status == "active")
        verified_count = _verified_artifact_count(artifact_rows)
        graph_node_count = await store.count_graph_nodes(project_id=project_id)

        # 2026-09-19 (TODO sweep): coverage + verification status per the
        # docstring — computed from actual row counts, never falsified.
        # artifact_fields stay byte-compatible (additive new fields alone).
        total = max(1, len(artifact_rows))
        return {
            "project_id": project_id,
            "artifact_count": len(artifact_rows),
            "active_artifacts": active_count,
            "verified_artifacts": verified_count,
            "graph_node_count": graph_node_count,
            "graph_coverage": round(min(1.0, graph_node_count / total), 2),
            "verification_coverage": round(min(1.0, verified_count / total), 2),
            "readiness_score": round(
                min(
                    1.0,
                    (active_count / total) * 0.5
                    + (verified_count / total) * 0.3
                    + min(1.0, graph_node_count / total) * 0.2,
                ),
                2,
            ),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Knowledge readiness check failed")
        raise_typed(PlatformError.knowledge2("failure", str(exc)))


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
        raise_typed(PlatformError.knowledge2("bad_request", "plan must be a dict"))

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
        raise_typed(PlatformError.knowledge2("failure", str(exc)), cause=exc)

    return WorkflowDraftResponse(
        workflow_id=result["workflow_id"],
        name=result["name"],
        steps=result["steps"],
        project_context=result.get("project_context", {}),
        meta=result.get("_meta", {}),
    )
