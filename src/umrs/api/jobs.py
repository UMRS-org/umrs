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
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Job store not initialized",
        )
    return _job_store


def get_optional_job_store() -> JobStore | None:
    """Return the configured job store, if this process owns one."""
    return _job_store


def _strict_auth_mode() -> bool:
    return os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict"


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
        return (
            job.project_id == self.project_id
            and job.user_id == self.user_id
        )


def get_caller_identity(
    caller_user_id: str | None = Query(None, description="Caller's user ID"),
    caller_project_id: str | None = Query(None, description="Caller's project ID"),
    is_admin: bool = Query(False, description="Admin flag (legacy, ignored in strict mode)"),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
) -> CallerIdentity:
    """Build caller identity from request params.

    Production: extract from Gateway-verified headers only.
    ``is_admin`` is derived from the ``X-Verified-Is-Admin`` header
    when present; the query-param ``is_admin`` is only used in
    non-strict MVP/testing mode when no verified headers are sent.
    """
    verified_user_id = _header_str(x_user_id)
    verified_project_id = _header_str(x_verified_project_id)
    if verified_user_id and verified_project_id:
        admin = _header_str(x_verified_is_admin) in ("true", "1")
        return CallerIdentity(
            user_id=verified_user_id,
            project_id=verified_project_id,
            is_admin=admin,
        )

    if _strict_auth_mode():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Gateway-verified caller identity",
        )

    if not caller_user_id or not caller_project_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="caller_user_id and caller_project_id are required",
        )

    return CallerIdentity(
        user_id=caller_user_id.strip(),
        project_id=caller_project_id.strip(),
        is_admin=is_admin,
    )


def _resolve_job_create_identity(
    request: "JobCreateRequest",
    x_user_id: str | None,
    x_verified_project_id: str | None,
) -> tuple[str, str]:
    verified_user_id = _header_str(x_user_id)
    verified_project_id = _header_str(x_verified_project_id)
    if verified_user_id and verified_project_id:
        return verified_project_id, verified_user_id

    if _strict_auth_mode():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Gateway-verified caller identity",
        )
    return request.project_id, request.user_id


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

    class Config:
        extra = "allow"  # Allow additional fields


class JobCreateRequest(BaseModel):
    """Request to create a new job."""

    project_id: str = Field(..., description="Project/tenant identifier")
    user_id: str = Field(..., description="User who created the job")
    job_type: str = Field(
        ...,
        description="Job type: index_build, embedding_batch, pipeline_run, endpoint_provision, cleanup"
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

    enqueue_immediately: bool = Field(
        True, description="Immediately queue for scheduling"
    )

    class Config:
        json_schema_extra = {
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
    next_run_at: str | None
    active_attempt_id: str | None
    commit_attempt_id: str | None
    created_at: str
    updated_at: str
    last_state_change_at: str | None

    # Included in detail view
    attempts: list[AttemptResponse] | None = None

    class Config:
        json_schema_extra = {
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
                "next_run_at": None,
                "active_attempt_id": "att-xyz789",
                "commit_attempt_id": None,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:31:00Z",
                "last_state_change_at": "2024-01-15T10:31:00Z",
            }
        }


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
        next_run_at=job.next_run_at.isoformat() if job.next_run_at else None,
        active_attempt_id=job.active_attempt_id,
        commit_attempt_id=job.commit_attempt_id,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        last_state_change_at=job.last_state_change_at.isoformat() if job.last_state_change_at else None,
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
    caller: CallerIdentity, job: Job, action: str = "access",
) -> None:
    """Raise 403 if caller does not own the job and is not admin."""
    if not caller.can_access_job(job):
        logger.warning(
            "User %s/%s denied %s on job %s (owner: %s/%s)",
            caller.project_id, caller.user_id, action,
            job.job_id, job.project_id, job.user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorised to {action} job {job.job_id}",
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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job_type: {request.job_type}. "
                       f"Valid types: {[t.value for t in JobType]}",
            ) from None

        # Parse restart policy
        try:
            restart_policy = RestartPolicy(request.restart_policy)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid restart_policy: {request.restart_policy}. "
                       f"Valid policies: {[p.value for p in RestartPolicy]}",
            ) from None

        # Build spec dict
        spec = request.spec.model_dump(exclude_none=True)
        project_id, user_id = _resolve_job_create_identity(
            request,
            x_user_id,
            x_verified_project_id,
        )

        # Create job
        job_id = store.create_job(
            project_id=project_id,
            user_id=user_id,
            job_type=job_type,
            spec=spec,
            idempotency_key=request.idempotency_key,
            priority=request.priority,
            max_attempts=request.max_attempts,
            restart_policy=restart_policy,
        )

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
    """
    # Non-admin callers can only see their own project's jobs
    if not caller.is_admin:
        project_id = caller.project_id
    # Parse filters
    parsed_status = None
    if status_filter:
        try:
            parsed_status = JobStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            ) from None

    parsed_type = None
    if job_type:
        try:
            parsed_type = JobType(job_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job_type: {job_type}",
            ) from None

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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        ) from None


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
    Note: This stops scheduling but does NOT kill running backend jobs.
    """
    try:
        job = store.get_job(job_id)
        _enforce_job_access(caller, job, action="cancel")

        store.cancel_job(job_id, reason)
        if job.job_type == JobType.WORKFLOW_RUN:
            from ..jobs.workflow_run import cancel_workflow_execution_for_job

            cancel_workflow_execution_for_job(job)
        return JobActionResponse(
            job_id=job_id,
            action="cancel",
            success=True,
            message=f"Job cancelled: {reason}",
        )

    except JobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        ) from None
    except InvalidTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e


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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Retry is an admin-only operation",
        )

    try:
        store.retry_job(job_id)
        return JobActionResponse(
            job_id=job_id,
            action="retry",
            success=True,
            message="Job requeued for retry",
        )

    except JobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        ) from None
    except InvalidTransitionError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e


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
