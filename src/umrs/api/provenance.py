"""Provenance REST API (COR-005, MEM-B4, MEM-B7).

Provides:
- GET /umrs/provenance/{job_id}           — Full provenance record for a job
- GET /umrs/provenance/{job_id}/lineage   — Lineage graph (DAG) for a job
- GET /umrs/provenance/diff               — Diff two provenance records
- GET /umrs/provenance                    — List provenance for recent completed jobs
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ..jobs.lineage_graph import build_lineage_graph
from ..jobs.models import JobStatus, JobType
from ..jobs.provenance import build_provenance
from ..jobs.provenance_diff import compute_provenance_diff
from ..jobs.store import JobNotFoundError, JobStore
from .jobs import CallerIdentity, get_caller_identity, get_job_store
from .traces import get_trace_store
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/provenance", tags=["provenance"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AttemptSummaryResponse(BaseModel):
    attempt_id: str
    attempt_no: int
    backend: str
    backend_ref: str | None = None
    status: str
    started_at: str | None = None
    ended_at: str | None = None
    exit_reason: str | None = None
    duration_seconds: float | None = None


class ModelRefResponse(BaseModel):
    name: str
    path: str | None = None
    version: str | None = None
    source: str | None = None


class ArtifactRefResponse(BaseModel):
    uri: str
    media_type: str = ""
    size_bytes: int | None = None
    sha256: str | None = None


class ProvenanceResponse(BaseModel):
    job_id: str
    job_type: str
    project_id: str
    user_id: str
    status: str
    spec: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    completed_at: str | None = None
    total_duration_seconds: float | None = None
    execution_authority: str | None = None
    runtime_wrapper: str | None = None
    umrs_instance_id: str | None = None
    model: ModelRefResponse | None = None
    attempts: list[AttemptSummaryResponse] = Field(default_factory=list)
    total_attempts: int = 0
    commit_attempt_id: str | None = None
    artifacts: list[ArtifactRefResponse] = Field(default_factory=list)
    restart_policy: str | None = None
    max_attempts: int = 0
    idempotency_key: str | None = None
    workflow_name: str | None = None
    execution_id: str | None = None
    parent_job_id: str | None = None


class ProvenanceListResponse(BaseModel):
    records: list[ProvenanceResponse]
    total: int
    limit: int
    offset: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# Declared before "/{job_id}": FastAPI matches in order, and "/{job_id}"
# captured "diff" as a job id ("Job not found: diff") — cycle-2 N7.
@router.get(
    "/diff",
    summary="Diff two provenance records (MEM-B7)",
)
def diff_provenance(
    job_a: str = Query(..., description="Job ID A (baseline)"),
    job_b: str = Query(..., description="Job ID B (comparison)"),
    store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
    trace_store=Depends(get_trace_store),
) -> dict[str, Any]:
    """Compare two job provenance records and their execution traces.

    Returns structured diff showing job-level, step-level, and artifact changes.
    """
    # Validate access to both jobs
    for jid in (job_a, job_b):
        try:
            job = store.get_job(jid)
        except JobNotFoundError:
            raise_typed(PlatformError.provenance("not_found", f"Job not found: {jid}"))
        if not caller.can_access_job(job):
            # Another user's job reads like an unknown id (existence hiding).
            raise_typed(PlatformError.provenance("not_found", f"Job not found: {jid}"))

    # Build provenance records
    job_obj_a = store.get_job(job_a)
    attempts_a = store.get_attempts_for_job(job_a)
    prov_a = build_provenance(job_obj_a, attempts_a).to_dict()

    job_obj_b = store.get_job(job_b)
    attempts_b = store.get_attempts_for_job(job_b)
    prov_b = build_provenance(job_obj_b, attempts_b).to_dict()

    # Fetch traces for step-level diff
    traces_a = [t.to_dict() for t in trace_store.get_traces_for_job(job_a)]
    traces_b = [t.to_dict() for t in trace_store.get_traces_for_job(job_b)]

    diff = compute_provenance_diff(prov_a, prov_b, traces_a, traces_b)
    return diff.to_dict()


@router.get(
    "/{job_id}",
    response_model=ProvenanceResponse,
    summary="Get provenance for a job",
)
def get_provenance(
    job_id: str,
    store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Return the full provenance record for a durable job."""
    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        raise_typed(PlatformError.provenance("not_found", f"Job not found: {job_id}"))

    if not caller.can_access_job(job):
        # Another user's job reads like an unknown id (existence hiding).
        raise_typed(PlatformError.provenance("not_found", f"Job not found: {job_id}"))

    attempts = store.get_attempts_for_job(job_id)
    record = build_provenance(job, attempts)
    return record.to_dict()


@router.get(
    "/{job_id}/lineage",
    summary="Get lineage graph for a job",
)
def get_provenance_lineage(
    job_id: str,
    store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
    trace_store=Depends(get_trace_store),
) -> dict[str, Any]:
    """Return a structured lineage graph (DAG) for all execution traces of a job.

    The graph includes:
    - Nodes: each execution step with status, timing, model refs, artifacts
    - Edges: parent-child relationships between steps
    - Metrics: depth, critical path, status aggregates
    """
    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        raise_typed(PlatformError.provenance("not_found", f"Job not found: {job_id}"))

    if not caller.can_access_job(job):
        # Another user's job reads like an unknown id (existence hiding).
        raise_typed(PlatformError.provenance("not_found", f"Job not found: {job_id}"))

    traces = trace_store.get_traces_for_job(job_id)
    graph = build_lineage_graph(job_id, traces)
    return graph.to_dict()


@router.get(
    "",
    response_model=ProvenanceListResponse,
    summary="List provenance records",
)
def list_provenance(
    project_id: str | None = Query(None),
    job_type: str | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """List provenance records for jobs matching the given filters."""
    jt = None
    if job_type:
        try:
            jt = JobType(job_type)
        except ValueError:
            raise_typed(
                PlatformError.provenance("bad_request", f"Invalid job_type: {job_type}")
            )

    js = None
    if status_filter:
        try:
            js = JobStatus(status_filter)
        except ValueError:
            raise_typed(
                PlatformError.provenance(
                    "bad_request", f"Invalid status: {status_filter}"
                )
            )

    effective_project = project_id or caller.project_id

    # Non-admin callers MUST NOT query another project's jobs.
    if project_id and project_id != caller.project_id and not caller.is_admin:
        raise_typed(
            PlatformError.auth(
                "admin_required",
                "Cross-project provenance listing requires admin privileges",
            )
        )

    jobs = store.list_jobs(
        project_id=effective_project,
        status=js,
        job_type=jt,
        limit=limit,
        offset=offset,
    )

    records = []
    for job in jobs:
        if not caller.can_access_job(job):
            continue
        attempts = store.get_attempts_for_job(job.job_id)
        record = build_provenance(job, attempts)
        records.append(record.to_dict())

    return {
        "records": records,
        "total": len(records),
        "limit": limit,
        "offset": offset,
    }
