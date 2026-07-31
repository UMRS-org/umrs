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


class WorkflowExecutionCounts(BaseModel):
    """Aggregate workflow execution counts (U1 dashboard section).

    Source: the in-process workflow execution registry. Counts reflect
    executions tracked by this UMRS process since start (interim until W1
    makes the registry durable).
    """

    total: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0


class ConnectorInventory(BaseModel):
    """Available data-intake connector types (U1/U3 dashboard section).

    U1: static inventory of connector implementations UMRS can run.
    U3: per-connector availability probe via umrs.connectors.introspect
    (import errors surface as ``unavailable`` + detail map). Per-source sync
    state (cursors, health) lands with P2 intake hardening; until then
    ``sync_state`` is None on purpose so dashboards can render "n/a".
    """

    available_types: list[str]
    sync_state: str | None = None
    unavailable: list[str] = []
    details: dict[str, dict] = {}


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

    # U1 sections — additive, nullable: None means the source is
    # unavailable in this process (e.g. no workflow runner).
    workflows: WorkflowExecutionCounts | None = None
    connectors: ConnectorInventory | None = None


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
    (
        total_models,
        available_models,
        staging_models,
        deprecated_models,
    ) = _model_status_counts()

    # Health
    health = _probe_db_health(getattr(app.state, "umrs_db_pool", None))

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
        workflows=_workflow_execution_counts(),
        connectors=_connector_inventory(),
    )


# Connector implementers ship one module per source under umrs.connectors.
# Kept as a constant so the dashboard and CLI render the same list.
_CONNECTOR_TYPES = [
    "confluence",
    "federated",
    "filesystem",
    "gdrive",
    "github",
    "notion",
    "slack",
    "web_crawl",
]


def _workflow_execution_counts() -> WorkflowExecutionCounts | None:
    """Aggregate workflow executions; None when the registry is unavailable."""
    try:
        from ..server.workflow_execution_registry import get_execution_registry

        records = get_execution_registry().list()
    except Exception:  # noqa: BLE001 — dashboard must degrade, not 500
        logger.debug("workflow execution registry unavailable", exc_info=True)
        return None
    counts = WorkflowExecutionCounts(total=len(records))
    for record in records:
        status_value = getattr(record, "status", "")
        if status_value == "succeeded":
            counts.completed += 1
        elif status_value == "failed":
            counts.failed += 1
        elif status_value == "cancelled":
            counts.cancelled += 1
        elif not record.is_terminal():
            counts.running += 1  # non-terminal: running/pending/etc.
    return counts


def _connector_inventory() -> ConnectorInventory | None:
    """Connector inventory with availability probe (U3).

    Falls back to the static U1 type list when the introspection module is
    unavailable; None only when the connectors package itself is missing.
    """
    try:
        from ..connectors import introspect
    except Exception:  # noqa: BLE001
        logger.debug("connector introspection unavailable", exc_info=True)
        return ConnectorInventory(available_types=list(_CONNECTOR_TYPES))

    report = introspect.inventory()
    return ConnectorInventory(
        available_types=report.available,
        unavailable=report.unavailable,
        details={c.name: c.to_dict() for c in report.connectors},
    )


# ── Overview/health/list helpers (complexity decomposition) ─────────────


def _model_status_counts() -> tuple[int, int, int, int]:
    """Return (total, available, staging, deprecated) model counts."""
    model_store = get_optional_model_store()
    if not model_store:
        return 0, 0, 0, 0
    try:
        models = model_store.list_models(ctx=None)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001 — degrade to zero counts
        return 0, 0, 0, 0
    counts = {"available": 0, "staging": 0, "deprecated": 0}
    for m in models:
        if m.status.value in counts:
            counts[m.status.value] += 1
    return len(models), counts["available"], counts["staging"], counts["deprecated"]


def _probe_db(db_pool: Any) -> bool:
    """True when the shared Postgres pool answers SELECT 1."""
    if not db_pool:
        return False
    try:
        conn = db_pool.getconn()
        try:
            conn.cursor().execute("SELECT 1")
        finally:
            db_pool.putconn(conn)
        return True
    except Exception:  # noqa: BLE001
        return False


def _probe_db_health(db_pool: Any) -> str:
    """healthy / degraded / unhealthy for the overview payload."""
    if not db_pool:
        return "degraded"
    return "healthy" if _probe_db(db_pool) else "unhealthy"


def _presence_component(name: str, store: Any, *, healthy_detail: str) -> HealthComponent:
    """Health component whose state is pure store presence."""
    return HealthComponent(
        name=name,
        status="healthy" if store else "unavailable",
        detail=healthy_detail if store else "Not configured",
    )


def _overall_health(db_ok: bool, job_ok: bool, model_ok: bool) -> str:
    """Aggregate component states into one status string."""
    if db_ok and job_ok and model_ok:
        return "healthy"
    return "degraded" if db_ok else "unhealthy"


def _fetch_jobs(
    job_store: Any,
    status: str | None,
    job_type: str | None,
    user_id: str | None,
    limit: int,
    offset: int,
) -> list[Any]:
    """Store query + in-memory user filter; empty on any failure."""
    from ..jobs.models import JobStatus, JobType

    try:
        jobs = job_store.list_jobs(
            status=JobStatus(status) if status else None,
            job_type=JobType(job_type) if job_type else None,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dashboard job query failed: %s", exc)
        return []
    if user_id:
        jobs = [job for job in jobs if job.user_id == user_id]
    return jobs


def _job_total(job_store: Any, jobs: list[Any], status_filter: Any) -> int:
    """Store-backed total when available, else the window size."""
    try:
        if hasattr(job_store, "count_all"):
            return job_store.count_all(status=status_filter)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dashboard job count failed: %s", exc)
    return len(jobs)


def _job_summary(j: Any) -> JobSummary:
    """Serialize a durable job row into a JobSummary."""
    return JobSummary(
        job_id=j.job_id,
        status=j.status.value,
        job_type=j.job_type.value,
        ttl_seconds=j.ttl_seconds,
        fan_out_child_job_ids=j.fan_out_child_job_ids,
        created_at=j.created_at.isoformat() if j.created_at else None,
        updated_at=j.updated_at.isoformat() if j.updated_at else None,
    )


def _model_summary(model_store: Any, model: Any) -> ModelSummary:
    """Serialize a model row, resolving the latest version best-effort."""
    latest_version = None
    try:
        history = model_store.get_model_versions(model.model_id, ctx=None)
        latest_version = history.latest.version if history.latest else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not load versions for model %s: %s", model.model_id, exc
        )
    return ModelSummary(
        model_id=model.model_id,
        name=model.name,
        scope=model.scope.value,
        model_type=model.model_type.value,
        status=model.status.value,
        version=model.version,
        latest_version=latest_version,
        created_at=model.created_at.isoformat() if model.created_at else None,
    )


def _attempt_dicts(store: Any, job: Any) -> list[dict[str, Any]]:
    """Attempt dicts for a job; empty when the store lacks the surface."""
    if not (job.active_attempt_id or hasattr(store, "get_attempts_for_job")):
        return []
    try:
        all_attempts = store.get_attempts_for_job(job.job_id)
    except Exception:  # noqa: BLE001
        return []
    return [
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

    jobs = _fetch_jobs(job_store, status, job_type, user_id, limit, offset)
    items = [_job_summary(j) for j in jobs]
    total = _job_total(job_store, jobs, JobStatus(status) if status else None)
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

    items = [_model_summary(model_store, model) for model in models]

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

    db_ok = _probe_db(getattr(app.state, "umrs_db_pool", None))
    components = [
        HealthComponent(
            name="database",
            status="healthy" if db_ok else "unavailable",
            detail="PostgreSQL" if db_ok else "Connection failed",
        ),
        _presence_component(
            "job_store",
            get_optional_job_store(),
            healthy_detail="Durable job tracking",
        ),
        _presence_component(
            "model_store",
            get_optional_model_store(),
            healthy_detail="Model registry",
        ),
    ]
    status_val = _overall_health(
        db_ok,
        components[1].status == "healthy",
        components[2].status == "healthy",
    )

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

    job_summary = _job_summary(job)
    attempts = _attempt_dicts(store, job)

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
