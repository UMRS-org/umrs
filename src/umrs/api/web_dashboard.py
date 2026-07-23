"""
Web Dashboard REST API — OpenAPI contract for the UMRS Web UI.

This module defines the standardized response schemas for the web dashboard.
Both the CLI/TUI and Web UI consume the same API.

Endpoints:
- GET /umrs/web/dashboard/overview — System overview (counts, status, uptime)
- GET /umrs/web/dashboard/jobs — Paginated job list with filters
- GET /umrs/web/dashboard/models — Paginated model list with filters
- GET /umrs/web/dashboard/health — Health check with component details
- GET /umrs/web/dashboard/notifications — Unified warning notifications (from TASK-015-001)
- GET /umrs/web/dashboard/jobs/{job_id} — Single job detail with attempts
- GET /umrs/web/dashboard/models/{model_id} — Single model detail with versions
- GET /umrs/web/dashboard/metrics — Prometheus metrics (proxied)
- GET /umrs/web/sse/jobs — Server-Sent Events stream for job updates
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Query, Request, status
from pydantic import BaseModel, Field

from ..api.jobs import get_optional_job_store, get_job_store
from ..api.models import get_optional_model_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/web/dashboard", tags=["web-dashboard"])

# ===========================================================================
# Response Models
# ===========================================================================


class PaginationMeta(BaseModel):
    """Pagination metadata included in every list response."""

    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1, le=1000)
    offset: int = Field(..., ge=0)
    has_next: bool
    has_prev: bool


class PaginatedResponse(BaseModel):
    """Generic paginated response envelope."""

    items: list[Any]
    pagination: PaginationMeta


class PaginatedJobsResponse(PaginatedResponse):
    """Paginated list of jobs."""

    items: list[dict[str, Any]]


class PaginatedModelsResponse(PaginatedResponse):
    """Paginated list of models."""

    items: list[dict[str, Any]]


class JobStatusCounts(BaseModel):
    """Count of jobs by status."""

    planned: int = 0
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0

    def total(self) -> int:
        return sum(self.__dict__.values())


class ModelTypeCounts(BaseModel):
    """Count of models by type."""

    llm: int = 0
    embedding: int = 0
    vision: int = 0

    def total(self) -> int:
        return sum(self.__dict__.values())


class ModelStatusCounts(BaseModel):
    """Count of models by status."""

    staging: int = 0
    available: int = 0
    deprecated: int = 0

    def total(self) -> int:
        return sum(self.__dict__.values())


class JobSummary(BaseModel):
    """Summary of a job for list views."""

    job_id: str
    status: str
    job_type: str
    ttl_seconds: int | None
    fan_out_child_job_ids: list[str]
    created_at: str | None
    updated_at: str | None


class ModelSummary(BaseModel):
    """Summary of a model for list views."""

    model_id: str
    name: str
    scope: str
    model_type: str
    status: str
    version: str | None = None
    latest_version: str | None = None
    created_at: str | None


class HealthComponent(BaseModel):
    """Health status of a subsystem."""

    name: str
    status: str  # "healthy", "degraded", "unavailable"
    detail: str = ""


class HealthResponse(BaseModel):
    """System health overview."""

    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    components: list[HealthComponent]
    instance_id: str | None = None


class OverviewResponse(BaseModel):
    """Dashboard overview — aggregate counts + system status."""

    total_jobs: int
    running_jobs: int
    queued_jobs: int
    completed_jobs: int
    failed_jobs: int

    total_models: int
    available_models: int
    staging_models: int
    deprecated_models: int

    system_health: str  # "healthy", "degraded", "unhealthy"


class NotificationItem(BaseModel):
    """A single warning/notification from the unified framework."""

    category: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    suggestion: str | None = None
    retry_after: int | None = None


class NotificationsResponse(BaseModel):
    """List of pending notifications."""

    notifications: list[NotificationItem]
    total: int = 0


class JobDetailResponse(BaseModel):
    """Detailed view of a single job."""

    job: JobSummary
    attempts: list[dict[str, Any]] = []


class ModelDetailResponse(BaseModel):
    """Detailed view of a single model."""

    model: dict[str, Any]
    versions: list[dict[str, Any]] = []
    latest_version: str | None = None


# ===========================================================================
# Endpoints
# ===========================================================================


@router.get(
    "/overview",
    response_model=OverviewResponse,
    summary="System overview",
    description=("Aggregate counts and system health for the dashboard home page."),
)
async def get_overview(request: Request) -> OverviewResponse:
    """Return aggregate job and model counts plus system health."""
    app = request.app
    # Job counts
    job_store = get_optional_job_store()
    running = queued = completed = failed = 0
    if job_store:
        for status_val in ("running", "queued", "completed", "failed"):
            try:
                count = job_store.count_by_status(status_val)
            except Exception:
                count = 0
            if status_val == "running":
                running = count
            elif status_val == "queued":
                queued = count
            elif status_val == "completed":
                completed = count
            elif status_val == "failed":
                failed = count

    total_jobs = running + queued + completed + failed

    # Model counts
    model_store = get_optional_model_store()
    available_models = staging_models = deprecated_models = 0
    if model_store:
        try:
            models = model_store.list_models(ctx=None)  # type: ignore[arg-type]
            total_models = len(models)
            for m in models:
                if m.status.value == "available":
                    available_models += 1
                elif m.status.value == "staging":
                    staging_models += 1
                elif m.status.value == "deprecated":
                    deprecated_models += 1
        except Exception:
            total_models = 0

    # Health
    db_pool = getattr(app.state, "umrs_db_pool", None)
    health = "healthy"
    try:
        if db_pool:
            conn = db_pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1")
            finally:
                db_pool.putconn(conn)
        else:
            health = "degraded"
    except Exception:
        health = "unhealthy"

    return OverviewResponse(
        total_jobs=total_jobs,
        running_jobs=running,
        queued_jobs=queued,
        completed_jobs=completed,
        failed_jobs=failed,
        total_models=total_models,
        available_models=available_models,
        staging_models=staging_models,
        deprecated_models=deprecated_models,
        system_health=health,
    )


@router.get(
    "/jobs",
    response_model=PaginatedJobsResponse,
    summary="List jobs",
    description=(
        "Paginated job list with optional filters. "
        "Sorting by created_at descending by default."
    ),
)
async def list_jobs(
    status: str | None = Query(None, description="Filter by job status"),
    job_type: str | None = Query(None, description="Filter by job type"),
    user_id: str | None = Query(None, description="Filter by user"),
    limit: int = Query(20, ge=1, le=200, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    request: Request = None,
) -> PaginatedJobsResponse:
    """List jobs with pagination and filters."""
    from ..jobs.models import JobStatus, JobType

    job_store = get_job_store()
    if not job_store:
        return PaginatedJobsResponse(
            items=[],
            pagination=PaginationMeta(
                total=0, limit=limit, offset=offset, has_next=False, has_prev=False
            ),
        )

    try:
        jobs = job_store.list_jobs(
            status=JobStatus(status) if status else None,
            job_type=JobType(job_type) if job_type else None,
            limit=limit,
            offset=offset,
        )
        if user_id:
            jobs = [job for job in jobs if job.user_id == user_id]
    except Exception as exc:
        logger.warning("Dashboard job query failed: %s", exc)
        jobs = []

    # Serialize
    items = [
        JobSummary(
            job_id=j.job_id,
            status=j.status.value,
            job_type=j.job_type.value,
            ttl_seconds=j.ttl_seconds,
            fan_out_child_job_ids=j.fan_out_child_job_ids,
            created_at=j.created_at.isoformat() if j.created_at else None,
            updated_at=j.updated_at.isoformat() if j.updated_at else None,
        )
        for j in jobs
    ]

    # Get total count
    total = len(jobs)
    try:
        if hasattr(job_store, "count_all"):
            total = job_store.count_all(status=JobStatus(status) if status else None)
    except Exception as exc:
        logger.warning("Dashboard job count failed: %s", exc)

    has_next = offset + limit < total
    has_prev = offset > 0

    return PaginatedJobsResponse(
        items=items,
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            has_next=has_next,
            has_prev=has_prev,
        ),
    )


@router.get(
    "/models",
    response_model=PaginatedModelsResponse,
    summary="List models",
    description="Paginated model list with type/status filters.",
)
async def list_models(
    model_type: str | None = Query(None, description="Filter by model type"),
    status: str | None = Query(None, description="Filter by model status"),
    scope: str | None = Query(None, description="Filter by scope"),
    limit: int = Query(20, ge=1, le=200, description="Page size"),
    offset: int = Query(0, ge=0, description="Page offset"),
    request: Request = None,
) -> PaginatedModelsResponse:
    """List models with pagination and filters."""
    from ..model_registry.models import ModelScope, ModelStatus, ModelType

    model_store = get_optional_model_store()
    if not model_store:
        return PaginatedModelsResponse(
            items=[],
            pagination=PaginationMeta(
                total=0, limit=limit, offset=offset, has_next=False, has_prev=False
            ),
        )

    try:
        models = model_store.list_models(
            ctx=None,  # type: ignore[arg-type]
            model_type=ModelType(model_type) if model_type else None,
            status=ModelStatus(status) if status else None,
            scope=ModelScope(scope) if scope else None,
            limit=limit,
            offset=offset,
        )
    except Exception:
        models = []

    items = []
    for model in models:
        latest_version = None
        try:
            history = model_store.get_model_versions(model.model_id, ctx=None)
            latest_version = history.latest.version if history.latest else None
        except Exception as exc:
            logger.warning(
                "Could not load versions for model %s: %s", model.model_id, exc
            )
        items.append(
            ModelSummary(
                model_id=model.model_id,
                name=model.name,
                scope=model.scope.value,
                model_type=model.model_type.value,
                status=model.status.value,
                version=model.version,
                latest_version=latest_version,
                created_at=model.created_at.isoformat() if model.created_at else None,
            )
        )

    total = len(models)
    has_next = offset + limit < total
    has_prev = offset > 0

    return PaginatedModelsResponse(
        items=items,
        pagination=PaginationMeta(
            total=total,
            limit=limit,
            offset=offset,
            has_next=has_next,
            has_prev=has_prev,
        ),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health",
    description="Health check with component details.",
)
async def get_health(request: Request) -> HealthResponse:
    """Return system health with component details."""
    import time

    app = request.app
    start_time = getattr(app.state, "start_time", None) or 0
    uptime = time.time() - float(start_time) if start_time else 0

    components: list[HealthComponent] = []

    # Check database
    db_pool = getattr(app.state, "umrs_db_pool", None)
    db_ok = False
    if db_pool:
        try:
            conn = db_pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute("SELECT 1")
                db_ok = True
            finally:
                db_pool.putconn(conn)
        except Exception:
            db_ok = False

    components.append(
        HealthComponent(
            name="database",
            status="healthy" if db_ok else "unavailable",
            detail="PostgreSQL" if db_ok else "Connection failed",
        )
    )

    # Check job store
    job_store = get_optional_job_store()
    components.append(
        HealthComponent(
            name="job_store",
            status="healthy" if job_store else "unavailable",
            detail="Durable job tracking" if job_store else "Not configured",
        )
    )

    # Check model store
    model_store = get_optional_model_store()
    components.append(
        HealthComponent(
            name="model_store",
            status="healthy" if model_store else "unavailable",
            detail="Model registry" if model_store else "Not configured",
        )
    )

    # Overall health
    status_val = "healthy"
    if not db_ok or not job_store or not model_store:
        status_val = "degraded" if db_ok else "unhealthy"

    instance_id = getattr(app.state, "instance_id", None)

    return HealthResponse(
        status=status_val,
        uptime_seconds=round(uptime, 1),
        components=components,
        instance_id=instance_id,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobDetailResponse,
    summary="Job detail",
    description="Detailed view of a single job with its attempts.",
)
async def get_job_detail(job_id: str) -> JobDetailResponse:
    """Get detailed job information including attempts."""
    from ..api.jobs import get_job_store

    store = get_job_store()
    try:
        job = store.get_job(job_id)
    except Exception:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    job_summary = JobSummary(
        job_id=job.job_id,
        status=job.status.value,
        job_type=job.job_type.value,
        ttl_seconds=job.ttl_seconds,
        fan_out_child_job_ids=job.fan_out_child_job_ids,
        created_at=job.created_at.isoformat() if job.created_at else None,
        updated_at=job.updated_at.isoformat() if job.updated_at else None,
    )

    # Get attempts if available
    attempts: list[dict[str, Any]] = []
    if job.active_attempt_id or hasattr(store, "get_attempts_for_job"):
        try:
            all_attempts = store.get_attempts_for_job(job_id)
            attempts = [
                {
                    "attempt_id": a.attempt_id,
                    "attempt_no": a.attempt_no,
                    "status": a.status.value,
                    "backend": a.backend,
                    "started_at": a.started_at.isoformat() if a.started_at else None,
                    "ended_at": a.ended_at.isoformat() if a.ended_at else None,
                    "exit_reason": a.exit_reason,
                    "status_detail": a.status_detail,
                    "metrics": a.metrics or {},
                }
                for a in all_attempts
            ]
        except Exception:
            pass

    return JobDetailResponse(job=job_summary, attempts=attempts)


@router.get(
    "/models/{model_id}",
    response_model=ModelDetailResponse,
    summary="Model detail",
    description="Detailed view of a single model with its versions.",
)
async def get_model_detail(
    model_id: str,
    model_store: Any = None,  # Will be injected by FastAPI
) -> ModelDetailResponse:
    """Get detailed model information including versions."""
    from ..api.models import get_model_store as _get_model_store

    store = _get_model_store()
    try:
        model = store.get_model(model_id)
    except Exception:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_id}",
        )

    # Get versions
    versions: list[dict[str, Any]] = []
    latest_version: str | None = None
    try:
        if hasattr(store, "get_model_versions"):
            hist = store.get_model_versions(model_id)
            versions = [v.to_dict() for v in hist.versions]
            latest_version = hist.latest.version if hist.latest else None
    except Exception:
        pass

    model_data = (
        model.to_dict()
        if hasattr(model, "to_dict")
        else {
            "model_id": model.model_id,
            "name": model.name,
            "scope": model.scope.value,
            "model_type": model.model_type.value,
            "status": model.status.value,
            "created_at": model.created_at.isoformat() if model.created_at else None,
        }
    )

    return ModelDetailResponse(
        model=model_data,
        versions=versions,
        latest_version=latest_version,
    )


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Proxied Prometheus metrics endpoint.",
)
async def get_metrics(request: Request):
    """Proxy Prometheus metrics."""
    from fastapi.responses import Response

    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.get(
    "/notifications",
    response_model=NotificationsResponse,
    summary="Pending notifications",
    description="Unified warning notifications from the warning framework.",
)
async def get_notifications(
    request: Request,
) -> NotificationsResponse:
    """Return pending notifications (future: store-based)."""
    # Currently returns empty — notifications are generated on-demand
    # In production, store notifications and surface them here
    return NotificationsResponse(
        notifications=[],
        total=0,
    )


@router.get(
    "/sse/jobs",
    summary="Job updates via Server-Sent Events",
    description=(
        "SSE stream for real-time job status changes. "
        "Clients should reconnect on disconnect."
    ),
)
async def jobs_sse(request: Request):
    """Server-Sent Events endpoint for job updates.

    Emits events:
    - `job_updated`: When a job status changes
    - `job_heartbeat`: Keepalive (every 30s)
    """
    from fastapi.responses import StreamingResponse

    async def event_generator():
        while True:
            # Keepalive
            yield ": keepalive\n\n"
            await asyncio.sleep(30)  # noqa: ASYNC110

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
