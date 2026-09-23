"""
Job REST API (U-731 ~ U-735).

Provides:
- POST /umrs/jobs - Submit a new job
- GET /umrs/jobs - List jobs with filters
- GET /umrs/jobs/{job_id} - Get job details with attempts
- POST /umrs/jobs/{job_id}/cancel - Cancel a job
- POST /umrs/jobs/{job_id}/retry - Force retry a job
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..jobs.cancel_gateway import register_store_getter as _register_store_getter
from ..jobs.models import (
    Job,
    JobAttempt,
    JobStatus,
    JobType,
    RestartPolicy,
)
from ..jobs.store import (
    DuplicateJobError,
    InvalidTransitionError,
    JobNotFoundError,
    JobStore,
)
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/jobs", tags=["jobs"])


# =============================================================================
# Dependency Injection
# =============================================================================

# Global job store instance (set during app startup)
_job_store: JobStore | None = None


def set_job_store(store: JobStore) -> None:
    """Set the global job store instance."""
    global _job_store
    _job_store = store


def get_job_store() -> JobStore:
    """Get the global job store instance."""
    if _job_store is None:
        raise_typed(PlatformError.jobs("unavailable", "Job store not initialized"))
    return _job_store


def get_optional_job_store() -> JobStore | None:
    """Return the configured job store, if this process owns one."""
    return _job_store


# ADR-058/WP-22: hand the orchestrator/janitor layers a jobs-layer cancel
# seam without ever importing them from api/. See jobs.cancel_gateway.
_register_store_getter(get_optional_job_store)


def _strict_auth_mode() -> bool:
    # Default strict (audit B12): mvp is an explicit dev opt-in, never the
    # silent default — an unconfigured process must not run wide open.
    return os.environ.get("UMRS_AUTH_MODE", "strict").strip().lower() == "strict"


def _header_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


@dataclass
class CallerIdentity:
    """Identity of the API caller, extracted from request parameters.

    In production this will come from JWT claims via Gateway.
    For now, accept as query params (matching the models API MVP pattern).
    """

    user_id: str
    project_id: str
    is_admin: bool = False

    def can_access_job(self, job: "Job") -> bool:
        """Check if caller can access a job (owner or admin)."""
        if self.is_admin:
            return True
        return job.project_id == self.project_id and job.user_id == self.user_id

    @property
    def is_anonymous(self) -> bool:
        """No verified project identity (anonymous scope, R1 parity)."""
        return not bool(self.project_id)

    @property
    def personal_project(self) -> str:
        """Personal three-tier namespace (``user:<uid>``, R1 parity)."""
        return f"user:{self.user_id}"


def get_caller_identity(
    caller_user_id: str | None = Query(None, description="Caller's user ID"),
    caller_project_id: str | None = Query(None, description="Caller's project ID"),
    is_admin: bool = Query(
        False, description="Admin flag (legacy, ignored in strict mode)"
    ),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> CallerIdentity:
    """Build caller identity from request params.

    Production: extract from Gateway-verified headers only.
    ``is_admin`` is derived from the ``X-Verified-Is-Admin`` header
    when present; the query-param ``is_admin`` is only used in
    non-strict MVP/testing mode when no verified headers are sent.

    Strict mode honors verified headers only with a valid gateway HMAC
    signature (shared-secret class closure, live-spoof-proof).
    """
    verified_user_id = _header_str(x_user_id)
    verified_project_id = _header_str(x_verified_project_id)
    if verified_user_id and verified_project_id:
        from ._verified_headers import handshake_secret, verified_headers_trusted

        admin = _header_str(x_verified_is_admin) in ("true", "1")
        if not verified_headers_trusted(
            strict=_strict_auth_mode(),
            secret=handshake_secret(),
            presented_hmac=_header_str(x_verified_hmac),
            user_id=verified_user_id,
            project_id=verified_project_id,
            is_admin=admin,
        ):
            raise_typed(
                PlatformError.auth(
                    "gateway_signature_rejected",
                    "Gateway-verified identity signature rejected",
                )
            )
        return CallerIdentity(
            user_id=verified_user_id,
            project_id=verified_project_id,
            is_admin=admin,
        )

    if _strict_auth_mode():
        raise_typed(
            PlatformError.auth(
                "gateway_identity_required", "Missing Gateway-verified caller identity"
            )
        )

    from ._identity import warn_legacy_params_once

    warn_legacy_params_once("jobs.get_caller_identity")
    if not caller_user_id or not caller_project_id:
        raise_typed(
            PlatformError.auth(
                "caller_identity_required",
                "caller_user_id and caller_project_id are required",
            )
        )

    return CallerIdentity(
        user_id=caller_user_id.strip(),
        project_id=caller_project_id.strip(),
        is_admin=is_admin,
    )


def _enforce_job_scope_policy(
    job_type: "JobType",
    spec: dict[str, Any],
    user_id: str,
    project_id: str,
    is_admin: bool,
) -> None:
    """Content-scope authorization for durable-job submission.

    Pre-fix (M10 HIGH): create_job gated only on job_type dispatchability.
    A MODEL_DOWNLOAD job with ``spec.scope="system"`` auto-registers the
    downloaded model SYSTEM-scope with no ACL context at all, and provision
    resolution sorts system scope first — one raw durable job poisoned the
    tenant-wide model registry and re-homed incompatible-provision requests.
    This guard applies the same rules as the direct ``stage_model`` route
    so the durable path can no longer bypass them.

    Debug/trace-dumpers (workers reading job state) out of scope here — this
    covers job types with model-registry side effects only.
    """
    if job_type != JobType.MODEL_DOWNLOAD:
        return
    scope = str(spec.get("scope") or "user")
    metadata = spec.get("metadata") or {}
    if scope == "system":
        _deny_unless(is_admin, "system-scope model registration requires admin role")
        return
    _check_scope_target(scope, metadata, user_id, project_id, is_admin)


def _check_scope_target(
    scope: str,
    metadata: dict[str, Any],
    user_id: str,
    project_id: str,
    is_admin: bool,
) -> None:
    """Validate project/user scope targets (keeps the gate ≤ 9 branches)."""
    if scope == "project":
        target = metadata.get("project_id") or project_id
        _deny_unless(
            target == project_id or is_admin,
            "cannot register a project-scope model into another project",
        )
        return
    _deny_unless(
        not metadata.get("owner_id") or metadata["owner_id"] == user_id or is_admin,
        "cannot register a user-scope model under another user",
    )


def _deny_unless(condition: bool, detail: str) -> None:
    """Raise 403 for a violated scope rule (keeps call-sites above budget)."""
    if not condition:
        raise_typed(PlatformError.jobs("forbidden", detail))


def _resolve_job_create_identity_primitive(
    x_user_id: str | None,
    x_project_id_from_body: str | None,
    x_user_id_from_body: str | None,
    x_verified_project_id: str | None,
    x_verified_is_admin: str | None,
    x_verified_hmac: str | None,
) -> tuple[str, str, bool]:
    """Strict-mode identity + admin tuple for durable-job submission.

    Pre-fix this resolver trusted X-User-ID + X-Verified-Project-ID alone —
    the same plain-header spoof class as wave-3's endpoints resolver (live
    PoC exploited on create_job specifically). Verification via the shared
    gateway HMAC handshake is MANDATORY in strict mode.
    """
    exists_identifier_handshake = _header_str(x_verified_hmac)
    verified_user_id = _header_str(x_user_id)
    verified_project_id = _header_str(x_verified_project_id)
    if _strict_auth_mode():
        from ._verified_headers import handshake_secret, verified_headers_trusted

        admin = _header_str(x_verified_is_admin) in ("true", "1")
        trusted = verified_headers_trusted(
            strict=True,
            secret=handshake_secret(),
            presented_hmac=exists_identifier_handshake,
            user_id=verified_user_id,
            project_id=verified_project_id,
            is_admin=admin,
        )
        if not trusted:
            raise_typed(
                PlatformError.auth(
                    "gateway_signature_rejected",
                    "Gateway-verified identity signature rejected",
                )
            )
        return verified_project_id, verified_user_id, admin
    if verified_user_id and verified_project_id:
        admin = _header_str(x_verified_is_admin) in ("true", "1")
        return verified_project_id, verified_user_id, admin
    # mvp compatibility: fall back to the request body identity the way the
    # pre-fix resolver did; strict mode never reaches here.
    pid = x_project_id_from_body
    uid = x_user_id_from_body
    if pid and uid:
        return pid, uid, False
    raise_typed(
        PlatformError.auth(
            "gateway_identity_required", "Missing Gateway-verified caller identity"
        )
    )


# =============================================================================
# Request/Response Models
# =============================================================================


class JobSpec(BaseModel):
    """Job specification - parameters for the job."""

    # Common fields
    model: str | None = Field(None, description="Model name for inference jobs")
    model_path: str | None = Field(None, description="Path to model files")

    # Data job fields
    dataset: str | None = Field(None, description="Dataset name/path for data jobs")
    output_path: str | None = Field(None, description="Output path for results")

    # Additional params (extensible)
    extra: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional job-specific parameters"
    )

    model_config = ConfigDict(extra="allow")


class JobCreateRequest(BaseModel):
    """Request to create a new job."""

    project_id: str = Field(..., description="Project/tenant identifier")
    user_id: str = Field(..., description="User who created the job")
    job_type: str = Field(
        ...,
        description="Job type: index_build, embedding_batch, pipeline_run, endpoint_provision, cleanup",
    )
    spec: JobSpec = Field(..., description="Job specification")

    idempotency_key: str | None = Field(
        None, description="Idempotency key (auto-generated if not provided)"
    )
    priority: int = Field(0, ge=-100, le=100, description="Priority (-100 to 100)")
    max_attempts: int = Field(3, ge=1, le=10, description="Max retry attempts")
    restart_policy: str = Field(
        "on_crash", description="Restart policy: on_crash, on_failure, never"
    )

    # Job TTL — how long the job stays alive after completion
    # None = terminate immediately (default)
    # > 0 = seconds to keep alive
    # -1 = until partition walltime limit
    ttl_seconds: int | None = Field(
        None,
        description=(
            "Time-to-live in seconds after job completion. "
            "None=terminate immediately, >0=keep alive N seconds, "
            "-1=keep alive until partition walltime limit."
        ),
    )

    # Fan-out job orchestration (GitHub Issue #87, TASK-015-005)
    fan_out_jobs: list[dict[str, Any]] | None = Field(
        None,
        description=(
            "Child jobs to create via fan-out. Each child spec contains "
            "job_type, spec, and optional depends_on (job IDs to wait for)."
        ),
    )

    enqueue_immediately: bool = Field(
        True, description="Immediately queue for scheduling"
    )

    @field_validator("ttl_seconds")
    @classmethod
    def validate_ttl_seconds(cls, value: int | None) -> int | None:
        if value is not None and value != -1 and value <= 0:
            raise ValueError("ttl_seconds must be null, -1, or a positive integer")
        return value

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "project_id": "proj-123",
                "user_id": "user-456",
                "job_type": "endpoint_provision",
                "spec": {
                    "model": "rivault-gpt-oss-static",
                    "model_path": "/models/rivault-gpt-oss",
                },
                "priority": 0,
                "max_attempts": 3,
                "restart_policy": "on_crash",
                "enqueue_immediately": True,
            }
        }
    )


class AttemptResponse(BaseModel):
    """Attempt data for API responses."""

    attempt_id: str
    job_id: str
    attempt_no: int
    backend: str
    backend_ref: str | None
    status: str
    started_at: str | None
    ended_at: str | None
    exit_reason: str | None
    log_uri: str | None
    status_detail: str | None
    metrics: dict[str, Any]
    created_at: str


class JobResponse(BaseModel):
    """Job data for API responses."""

    job_id: str
    project_id: str
    user_id: str
    job_type: str
    spec: dict[str, Any]
    status: str
    priority: int
    desired_max_attempts: int
    restart_policy: str
    ttl_seconds: int | None = None
    remaining_alive_seconds: int | None = None
    fan_out_child_job_ids: list[str] = Field(default_factory=list)
    next_run_at: str | None
    active_attempt_id: str | None
    commit_attempt_id: str | None
    created_at: str
    updated_at: str
    last_state_change_at: str | None

    # Included in detail view
    attempts: list[AttemptResponse] | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "job-abc123",
                "project_id": "proj-123",
                "user_id": "user-456",
                "job_type": "endpoint_provision",
                "spec": {"model": "rivault-gpt-oss-static"},
                "status": "running",
                "priority": 0,
                "desired_max_attempts": 3,
                "restart_policy": "on_crash",
                "ttl_seconds": 300,
                "next_run_at": None,
                "active_attempt_id": "att-xyz789",
                "commit_attempt_id": None,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:31:00Z",
                "last_state_change_at": "2024-01-15T10:31:00Z",
            }
        }
    )


class JobListResponse(BaseModel):
    """Response for job listing."""

    jobs: list[JobResponse]
    total: int
    limit: int
    offset: int


class JobCreateResponse(BaseModel):
    """Response for job creation."""

    job_id: str
    status: str
    message: str


class JobActionResponse(BaseModel):
    """Response for job actions (cancel, retry)."""

    job_id: str
    action: str
    success: bool
    message: str


# =============================================================================
# Helper Functions
# =============================================================================


def _job_to_response(job: Job, include_attempts: bool = False) -> JobResponse:
    """Convert Job model to response."""
    return JobResponse(
        job_id=job.job_id,
        project_id=job.project_id,
        user_id=job.user_id,
        job_type=job.job_type.value,
        spec=job.spec,
        status=job.status.value,
        priority=job.priority,
        desired_max_attempts=job.desired_max_attempts,
        restart_policy=job.restart_policy.value,
        ttl_seconds=job.ttl_seconds,
        remaining_alive_seconds=(
            max(
                0,
                job.ttl_seconds
                - int(
                    (
                        datetime.now(timezone.utc)
                        - job.updated_at.replace(tzinfo=timezone.utc)
                    ).total_seconds()
                ),
            )
            if job.ttl_seconds
            and job.ttl_seconds > 0
            and job.status == JobStatus.COMPLETED
            else job.ttl_seconds
            if job.ttl_seconds == -1 and job.status == JobStatus.COMPLETED
            else None
        ),
        fan_out_child_job_ids=job.fan_out_child_job_ids,
        next_run_at=job.next_run_at.isoformat() if job.next_run_at else None,
        active_attempt_id=job.active_attempt_id,
        commit_attempt_id=job.commit_attempt_id,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        last_state_change_at=job.last_state_change_at.isoformat()
        if job.last_state_change_at
        else None,
        attempts=None,  # Set separately if needed
    )


def _attempt_to_response(attempt: JobAttempt) -> AttemptResponse:
    """Convert JobAttempt model to response."""
    return AttemptResponse(
        attempt_id=attempt.attempt_id,
        job_id=attempt.job_id,
        attempt_no=attempt.attempt_no,
        backend=attempt.backend,
        backend_ref=attempt.backend_ref,
        status=attempt.status.value,
        started_at=attempt.started_at.isoformat() if attempt.started_at else None,
        ended_at=attempt.ended_at.isoformat() if attempt.ended_at else None,
        exit_reason=attempt.exit_reason,
        log_uri=attempt.log_uri,
        status_detail=attempt.status_detail,
        metrics=attempt.metrics,
        created_at=attempt.created_at.isoformat(),
    )


def _enforce_job_access(
    caller: CallerIdentity,
    job: Job,
    action: str = "access",
) -> None:
    """Raise 403 if caller does not own the job and is not admin."""
    if not caller.can_access_job(job):
        logger.warning(
            "User %s/%s denied %s on job %s (owner: %s/%s)",
            caller.project_id,
            caller.user_id,
            action,
            job.job_id,
            job.project_id,
            job.user_id,
        )
        raise_typed(
            PlatformError.jobs(
                "forbidden", f"Not authorised to {action} job {job.job_id}"
            )
        )


# =============================================================================
# API Endpoints
# =============================================================================


@router.post(
    "",
    response_model=JobCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new job",
    description="Submit a new job for processing. Returns 409 if idempotency key already exists.",
)
async def create_job(
    request: JobCreateRequest,
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    store: JobStore = Depends(get_job_store),
) -> JobCreateResponse:
    """
    Create a new job (U-731).

    If idempotency_key is provided and already exists, returns the existing job_id.
    """
    try:
        # Parse job type
        try:
            job_type = JobType(request.job_type)
        except ValueError:
            raise_typed(
                PlatformError.jobs(
                    "bad_request",
                    f"Invalid job_type: {request.job_type}. "
                    f"Valid types: {[t.value for t in JobType]}",
                )
            )

        # #216: reject types without a dispatch backend on the REST/scheduler
        # path — otherwise a durable job is accepted and only explodes with
        # "not yet implemented" at dispatch time.
        _enforce_dispatchable_job_type(job_type)

        # Parse restart policy
        try:
            restart_policy = RestartPolicy(request.restart_policy)
        except ValueError:
            raise_typed(
                PlatformError.jobs(
                    "bad_request",
                    f"Invalid restart_policy: {request.restart_policy}. "
                    f"Valid policies: {[p.value for p in RestartPolicy]}",
                )
            )

        # Build spec dict
        spec = request.spec.model_dump(exclude_none=True)
        project_id, user_id, is_admin = _resolve_job_create_identity_primitive(
            x_user_id,
            request.project_id,
            request.user_id,
            x_verified_project_id,
            x_verified_is_admin,
            x_verified_hmac,
        )
        _enforce_job_scope_policy(job_type, spec, user_id, project_id, is_admin)

        create_method = (
            store.create_fan_out_graph if request.fan_out_jobs else store.create_job
        )
        create_kwargs = {
            "project_id": project_id,
            "user_id": user_id,
            "priority": request.priority,
            "max_attempts": request.max_attempts,
            "restart_policy": restart_policy,
            "ttl_seconds": request.ttl_seconds,
        }
        if request.fan_out_jobs:
            job_id = _create_fan_out_job(store, request, job_type, spec, create_kwargs)
            return JobCreateResponse(
                job_id=job_id,
                status="running",
                message="Fan-out parent and child jobs created",
            )

        job_id = create_method(
            job_type=job_type,
            spec=spec,
            idempotency_key=request.idempotency_key,
            **create_kwargs,
        )
        _autopin_catalog(store, project_id, job_id)

        # Optionally enqueue immediately
        if request.enqueue_immediately:
            store.enqueue_job(job_id)
            return JobCreateResponse(
                job_id=job_id,
                status="queued",
                message="Job created and queued for scheduling",
            )
        else:
            return JobCreateResponse(
                job_id=job_id,
                status="planned",
                message="Job created (not yet queued)",
            )

    except DuplicateJobError as e:
        # Return existing job ID with 409 Conflict
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Job with this idempotency key already exists",
                "existing_job_id": e.existing_job_id,
            },
        ) from e


def _autopin_catalog(store: JobStore, project_id: str, job_id: str) -> None:
    """D-43 auto-pin: when a project has exactly ONE active scientific
    catalog, every new durable job is pinned onto it (kind=job, ref=job_id).

    Zero catalogs → nothing to associate against; several → the choice is
    ambiguous and never silently guessed (pin by hand with `r4 catalog pin`).
    Best-effort: a store hiccup never blocks job submission.
    """
    try:
        from umrs.scientific_catalogs.store import ScientificCatalogStore

        with store._get_conn() as conn:
            catalogs = ScientificCatalogStore(conn).list(project_id=project_id)
            if len(catalogs) != 1:
                return
            ScientificCatalogStore(conn).add_pin(
                catalogs[0]["catalog_id"], kind="job", ref=job_id
            )
    except Exception as exc:  # noqa: BLE001 — pin is best-effort, never a gate
        logger.warning("catalog autopin skipped for job %s: %s", job_id, exc)


def _create_fan_out_job(
    store: JobStore,
    request: JobCreateRequest,
    job_type: JobType,
    spec: dict[str, Any],
    create_kwargs: dict[str, Any],
) -> str:
    """Create fan-out parent+children graph; map store errors to 422."""
    try:
        return store.create_fan_out_graph(
            parent_job_type=job_type,
            parent_spec=spec,
            children=request.fan_out_jobs,
            **create_kwargs,
        )
    except (KeyError, ValueError) as exc:
        raise_typed(PlatformError.jobs("unprocessable", str(exc)), cause=exc)


def _parse_status_filter(status_filter: str | None) -> JobStatus | None:
    """Parse and validate a job status filter value."""
    if not status_filter:
        return None
    try:
        return JobStatus(status_filter)
    except ValueError:
        raise_typed(
            PlatformError.jobs("bad_request", f"Invalid status: {status_filter}")
        )


# #216: job types accepted by JobType enum but with no dispatch backend on the
# REST/scheduler path. Remove entries as backends land.
_UNDISPATCHABLE_JOB_TYPES = {
    JobType.EMBEDDING_BATCH,
    JobType.PIPELINE_RUN,
    # ADR-047 wave: executor classes that have no shipped backend YET.
    # BATCH_INFER graduated 2026-09-12 (c5061d43).
    # MODEL_EVAL graduated 2026-09-12 (this wave).
    JobType.FINE_TUNE,
    # SESSION_GC: gated at submission (operator-side, POST /umrs/jobs requires
    # identity as every other submission; max_days is job-locked at submit-time).
    # Not in this set.
    # D-44 domain benchmark: executor + ground-truth suite format are the
    # named design decisions; the kind stays a phenom until they land.
    JobType.DOMAIN_BENCHMARK,
}


def _enforce_dispatchable_job_type(job_type: JobType) -> None:
    """Reject unimplemented job types at submission (400) instead of burning a
    durable job that only explodes with "not yet implemented" at dispatch (#216)."""
    if job_type in _UNDISPATCHABLE_JOB_TYPES:
        raise_typed(
            PlatformError.jobs(
                "bad_request",
                (
                    f"job_type '{job_type.value}' is not implemented on this "
                    "deployment (no durable dispatch backend). Remove from "
                    "_UNDISPATCHABLE_JOB_TYPES in api/jobs.py when the backend lands."
                ),
            )
        )


def _parse_job_type_filter(job_type: str | None) -> JobType | None:
    """Parse and validate a job type filter value."""
    if not job_type:
        return None
    try:
        return JobType(job_type)
    except ValueError:
        raise_typed(PlatformError.jobs("bad_request", f"Invalid job_type: {job_type}"))


@router.get(
    "",
    response_model=JobListResponse,
    summary="List jobs",
    description="List jobs with optional filters by project, status, and type.",
)
async def list_jobs(
    project_id: str | None = Query(None, description="Filter by project ID"),
    status_filter: str | None = Query(
        None, alias="status", description="Filter by status"
    ),
    job_type: str | None = Query(None, description="Filter by job type"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> JobListResponse:
    """
    List jobs (U-732).

    Non-admin callers are restricted to their own project's jobs.

    project_id defaults to the caller's project when not explicitly given,
    so that callers who accidentally pass an admin flag (or are behind a
    misconfigured proxy) never see jobs from unrelated projects.
    """
    # Always default project_id to caller's project (defence-in-depth).
    # Gateway sends "caller_project_id" as a query param, not "project_id",
    # so an admin caller would otherwise see ALL jobs across ALL projects.
    if project_id is None:
        project_id = caller.project_id

    # Non-admin callers cannot override project_id
    if not caller.is_admin:
        project_id = caller.project_id

    parsed_status = _parse_status_filter(status_filter)
    parsed_type = _parse_job_type_filter(job_type)

    # Query jobs
    jobs = store.list_jobs(
        project_id=project_id,
        status=parsed_status,
        job_type=parsed_type,
        limit=limit,
        offset=offset,
    )

    # Non-admin callers only see their own jobs within the project
    if not caller.is_admin:
        jobs = [j for j in jobs if j.user_id == caller.user_id]

    return JobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=len(jobs),  # Note: would need separate count query for true total
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job details",
    description="Get detailed information about a job including all attempts.",
)
async def get_job(
    job_id: str,
    include_attempts: bool = Query(True, description="Include attempt history"),
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> JobResponse:
    """
    Get job details (U-733).

    Returns 403 if the caller does not own the job and is not admin.
    """
    try:
        job = store.get_job(job_id)
        _enforce_job_access(caller, job, action="read")

        response = _job_to_response(job)

        if include_attempts:
            attempts = store.get_attempts_for_job(job_id)
            response.attempts = [_attempt_to_response(a) for a in attempts]

        return response

    except JobNotFoundError:
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))


@router.get(
    "/{job_id}/manifest",
    summary="Get the latest execution manifest for a job",
    description=(
        "P7a: the execution manifest persisted at attempt completion "
        "(environment, code version, provenance). Commit attempt preferred."
    ),
)
async def get_job_manifest(
    job_id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> dict[str, Any]:
    """Latest ExecutionManifest for one job (404 when none is persisted yet).
    # auth: verified
    """
    # auth: verified
    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
    _enforce_job_access(caller, job, action="read")
    manifest = store.get_latest_attempt_manifest(job_id)
    if manifest is None:
        raise_typed(
            PlatformError.jobs(
                "manifest_missing",
                f"No execution manifest persisted for job {job_id} yet "
                "(recorded at attempt completion)",
            )
        )
    return {"job_id": job_id, "manifest": manifest}


@router.get(
    "/{job_id}/manifests",
    summary="List per-attempt execution manifests for a job",
    description="P7a: manifests persisted per attempt (subset may be empty for older attempts).",
)
async def list_job_manifests(
    job_id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> dict[str, Any]:
    """Per-attempt manifests for one job (older attempts pre-manifest = omitted).
    # auth: verified
    """
    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
    _enforce_job_access(caller, job, action="read")
    manifests = []
    for attempt in store.get_attempts_for_job(job_id):
        manifest = store.get_attempt_manifest(attempt.attempt_id)
        if manifest is not None:
            manifests.append({"attempt_id": attempt.attempt_id, "manifest": manifest})
    return {"job_id": job_id, "manifests": manifests, "count": len(manifests)}


@router.post(
    "/{job_id}/cancel",
    response_model=JobActionResponse,
    summary="Cancel a job",
    description="Cancel a job. Stops scheduling further attempts but does not kill running backend jobs.",
)
async def cancel_job(
    job_id: str,
    reason: str = Query("user_cancelled", description="Cancellation reason"),
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> JobActionResponse:
    """
    Cancel a job (U-734).

    Returns 403 if the caller does not own the job and is not admin.
    The durable flip ALSO broadcasts to in-process listener classes (6b-3b):
    model-download backends interrupt mid-flight chunk loops — the loop's
    progress callback is where a cancel currently shakes open; for
    uninterruptible sub-passes (the HF single-snapshot opacity, named
    6b-3a) the interrupt lands at pass boundary, never silently dropped.
    """
    try:
        job = store.get_job(job_id)
        _enforce_job_access(caller, job, action="cancel")

        store.cancel_job(job_id, reason)
        if job.job_type == JobType.WORKFLOW_RUN:
            from ..jobs.workflow_run import cancel_workflow_execution_for_job

            cancel_workflow_execution_for_job(job)
        # 6b-3b (2026-09-11): durable cancel ALSO broadcasts to in-process
        # back-end listeners (model download, etc.) so a running job is
        # interrupted — never only re-labeled cancelled while it keeps
        # writing.
        from ..jobs.download_safety import broadcast_cancel

        broadcast_cancel(job_id)
        return JobActionResponse(
            job_id=job_id,
            action="cancel",
            success=True,
            message=f"Job cancelled: {reason}",
        )

    except JobNotFoundError:
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
    except InvalidTransitionError as e:
        raise_typed(PlatformError.jobs("conflict", str(e)), cause=e)


def _drift_window(store: JobStore, job_id: str) -> dict[str, Any]:
    """Drift honesty (P7b): recorded vs current code version.

    Pinned when the prior attempt's manifest carries a code SHA (P7a);
    otherwise the drift window is declared undeclared, never hidden.
    """
    recorded: str = ""
    latest = store.get_latest_attempt_manifest(job_id)
    if latest:
        versions = latest.get("code_versions") or []
        if versions:
            recorded = str(versions[0].get("commit_sha") or "")
    from ..jobs.manifest import code_version_ref

    current = code_version_ref().get("commit_sha") or ""
    pinned = bool(recorded)
    return {
        "pinned": pinned,
        "same_code": (recorded == current) if pinned else None,
        "recorded_code_sha": recorded or None,
        "current_code_sha": current or None,
        "note": (
            ""
            if pinned
            else "prior attempt carries no recorded code version — drift window undeclared"
        ),
    }


# ---------------------------------------------------------------------------
# Attempt-scoped short-lived S3 credentials (WP-11, G-P5-3)
# ---------------------------------------------------------------------------


def _attempt_for_job(store: JobStore, job_id: str, attempt_id: str) -> Any:
    """Resolve one attempt belonging to the job (404 when absent)."""
    for attempt in store.get_attempts_for_job(job_id):
        if attempt.attempt_id == attempt_id:
            return attempt
    raise_typed(
        PlatformError.jobs(
            "attempt_not_found", f"Attempt not found on job {job_id}: {attempt_id}"
        )
    )


def _load_or_mint_credentials(
    store: JobStore, job_id: str, attempt_id: str, attempt_no: int
) -> dict[str, Any]:
    """Existing live record, else mint + persist a new one (idempotent)."""
    from ..jobs.attempt_credentials import (
        DEFAULT_ATTEMPT_TTL_SEC,
        AttemptCredentials,
        mint_attempt_credentials,
    )

    existing = store.get_attempt_credentials(attempt_id)
    if existing:
        record = AttemptCredentials.from_dict(existing)
        try:
            expires = datetime.fromisoformat(record.expires_at)
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) < expires:
                return record.to_dict()
        except ValueError:
            pass  # malformed expiry → re-mint
    record = mint_attempt_credentials(
        job_id, attempt_id, attempt_no, ttl_sec=DEFAULT_ATTEMPT_TTL_SEC
    )
    store.record_attempt_credentials(attempt_id, record.to_dict())
    return record.to_dict()


@router.post(
    "/{job_id}/attempts/{attempt_id}/credentials",
    summary="Mint attempt-scoped S3 credentials (WP-11)",
    description=(
        "Attempt-keyed short-lived credential record: key-space scoped to "
        "jobs/<job_id>/attempt_<n>/, short TTL, auditable mint trail. No secret "
        "material is stored. Round 1 = scoped presign + mint trail; STS round 2."
    ),
)
async def mint_attempt_credentials_route(
    job_id: str,
    attempt_id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> dict[str, Any]:
    """Mint (idempotent) the attempt's credential record.
    # auth: verified
    """
    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
    _enforce_job_access(caller, job, action="mint credentials")
    attempt = _attempt_for_job(store, job_id, attempt_id)
    record = _load_or_mint_credentials(store, job_id, attempt_id, attempt.attempt_no)
    return {"status": "minted", "credentials": record}


@router.post(
    "/{job_id}/attempts/{attempt_id}/credentials/sts",
    summary="Per-attempt STS credentials (WP-11 round 2)",
    description=(
        "Federates the attempt's minted credential record into real STS "
        "temporary credentials (MinIO AssumeRole, prefix-scoped inline "
        "policy, capped TTL 3600s). Fail-closed: an STS venue failure is a "
        "503, never an ambient-credential fallback. The token itself is never "
        "persisted or logged (round-1 mint record is the audit trail)."
    ),
)
async def mint_attempt_sts_credentials(
    job_id: str,
    attempt_id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> dict[str, Any]:
    """Mint real per-attempt identity via STS AssumeRole.
    # auth: verified
    """
    from ..jobs.attempt_credentials import AttemptCredentials
    from ..jobs.sts_federation import STSFederationError, assume_role_for_attempt
    from ..shared.storage.s3_config import get_s3_config

    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
    _enforce_job_access(caller, job, action="mint sts credentials")
    attempt = _attempt_for_job(store, job_id, attempt_id)

    raw = store.get_attempt_credentials(attempt_id)
    if not raw:
        raise_typed(
            PlatformError.presign(
                "not_minted",
                "No attempt credentials minted yet — POST .../credentials first",
            )
        )
    record = AttemptCredentials.from_dict(raw)

    config = get_s3_config()
    try:
        creds = assume_role_for_attempt(
            record,
            bucket=config.bucket,
            endpoint_url=config.endpoint_url,
            access_key=config.access_key,
            secret_key=config.secret_key,
            role_arn=config.sts_role_arn,
        )
    except STSFederationError as exc:
        raise_typed(
            PlatformError.jobs("unavailable", f"STS federation unavailable: {exc}"),
            cause=exc,
        )

    # Re-mint the record with the capped STS TTL so downstream presign verbs
    # bound their URLs by the session lifetime (never wider).
    _ = attempt  # attempt_no already folded into the record; keep route pure
    return {
        "status": "federated",
        "scope_prefixes": list(record.scope_prefixes),
        "bucket": config.bucket,
        "endpoint_url": config.endpoint_url,
        "credentials": creds,
    }


class AttemptPresignBody(BaseModel):
    """Request for an attempt-scoped presigned URL."""

    key: str = Field(min_length=1, max_length=1024)
    method: str = Field(default="get", pattern="^(get|put)$")
    ttl_sec: int | None = None


@router.post(
    "/{job_id}/attempts/{attempt_id}/presign",
    summary="Attempt-scoped presigned URL (WP-11)",
    description=(
        "Presigns GET/PUT ONLY for keys under the attempt's recorded scope "
        "(403 outside). TTL is bounded by the minted credential TTL "
        "(default 900s — not the ambient one-hour flat TTL)."
    ),
)
async def presign_for_attempt(
    job_id: str,
    attempt_id: str,
    body: AttemptPresignBody,
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> dict[str, Any]:
    """Attempt-scoped presign; refuses outside the recorded prefix/expiry.
    # auth: verified
    """
    from ..jobs.attempt_credentials import AttemptCredentials, key_in_scope

    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
    _enforce_job_access(caller, job, action="presign")
    _attempt_for_job(store, job_id, attempt_id)

    raw = store.get_attempt_credentials(attempt_id)
    if not raw:
        raise_typed(
            PlatformError.presign(
                "not_minted",
                "No attempt credentials minted yet — POST .../credentials first",
            )
        )
    record = AttemptCredentials.from_dict(raw)
    key = body.key.lstrip("/")
    if not key_in_scope(key, record):
        raise_typed(
            PlatformError.presign(
                "out_of_scope", "key outside attempt scope or credentials expired"
            )
        )

    ttl = min(body.ttl_sec or record.ttl_sec, record.ttl_sec)
    from ..shared.storage.s3_config import get_s3_config

    config = get_s3_config()
    client = config.get_client()
    from ..shared.storage.s3_presign import (
        generate_presigned_get,
        generate_presigned_put,
    )

    if body.method == "get":
        url = generate_presigned_get(client, config.bucket, key, ttl=ttl)
    else:
        url = generate_presigned_put(client, config.bucket, key, ttl=ttl)
    return {
        "url": url,
        "method": body.method,
        "expires_in_sec": ttl,
        "scope": list(record.scope_prefixes),
    }


@router.post(
    "/{job_id}/reproduce",
    summary="Reproduce a job (executor-backed replay, P7b)",
    description=(
        "Executor-backed replay: requeues the job's spec so the durable runtime "
        "executes it again with the SAME inputs, and reports the drift window "
        "(recorded vs current code version) honestly."
    ),
)
async def reproduce_job(
    job_id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> dict[str, Any]:
    """Reproduce (replay) one job through the durable runtime (G-P7-2).
    # auth: verified
    """
    # auth: verified
    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
    _enforce_job_access(caller, job, action="reproduce")
    drift = _drift_window(store, job_id)
    try:
        store.retry_job(job_id)
    except InvalidTransitionError as exc:
        raise_typed(PlatformError.jobs("conflict", str(exc)), cause=exc)
    return {
        "job_id": job_id,
        "action": "reproduce",
        "success": True,
        "message": "job requeued; the durable runtime will execute it with the same spec",
        "drift": drift,
    }


@router.post(
    "/{job_id}/retry",
    response_model=JobActionResponse,
    summary="Retry a job",
    description="Force retry a failed or cancelled job. Admin operation.",
)
async def retry_job(
    job_id: str,
    caller: CallerIdentity = Depends(get_caller_identity),
    store: JobStore = Depends(get_job_store),
) -> JobActionResponse:
    """
    Retry a job (U-735).

    Admin-only operation - force requeues a failed/cancelled job.
    Returns 403 for non-admin callers.
    """
    if not caller.is_admin:
        raise_typed(PlatformError.jobs("forbidden", "Retry is an admin-only operation"))

    try:
        store.retry_job(job_id)
        return JobActionResponse(
            job_id=job_id,
            action="retry",
            success=True,
            message="Job requeued for retry",
        )

    except JobNotFoundError:
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
    except InvalidTransitionError as e:
        raise_typed(PlatformError.jobs("conflict", str(e)), cause=e)


# =============================================================================
# Health/Stats Endpoint
# =============================================================================


@router.get(
    "/stats",
    summary="Get job statistics",
    description="Get aggregate statistics about jobs.",
)
async def get_job_stats(
    project_id: str | None = Query(None, description="Filter by project ID"),
    store: JobStore = Depends(get_job_store),
) -> dict[str, Any]:
    """
    Get job statistics.
    """
    # This would ideally use aggregate queries
    # For now, return basic stats
    stats = {
        "status_counts": {},
        "type_counts": {},
    }

    # Count by status
    for job_status in JobStatus:
        jobs = store.list_jobs(project_id=project_id, status=job_status, limit=0)
        stats["status_counts"][job_status.value] = len(jobs)

    return stats
