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
- GET /umrs/web/dashboard/workflows — Workflow executions panel
- GET /umrs/web/dashboard/connectors — Connector intake panel
- GET /umrs/web/dashboard/knowledge — Per-project knowledge panel
- GET /umrs/web/dashboard/metrics — Prometheus metrics (proxied)
- GET /umrs/web/dashboard/sse/jobs — Server-Sent Events stream for job updates
- GET /umrs/web/dashboard/sse/workflows — Workflow execution SSE snapshots
- GET /umrs/web/dashboard/sse/intake — Connector intake SSE snapshots

Scope (hardening plan H-1 item 5, 2026-10-17): every route except /health and
/metrics resolves the gateway-verified caller (401 in strict mode without
it). Non-admin callers see their own jobs (as GET /umrs/jobs), the models,
notifications, workflow executions and knowledge of their verified project;
another project's job or model reads as not found. Connector sync rows carry
no project and are admin-only. Admins see every project; anonymous callers
exist only in dev (mvp) mode and keep the unscoped view.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ..api.jobs import get_job_store, get_optional_job_store
from ..api.models import get_optional_model_store
from ..knowledge.access_scope import CallerScope, project_readable
from ._scope_identity import get_caller_scope
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/web/dashboard", tags=["web-dashboard"])


# ===========================================================================
# Caller scope (H-1 item 5)
# ===========================================================================


def _unrestricted(scope: CallerScope) -> bool:
    """Admins (and anonymous dev-mode callers) see every project's rows."""
    return scope.is_admin or scope.anonymous


def _project_filter(scope: CallerScope) -> str | None:
    """The caller's verified project, or None (no filter) when unrestricted."""
    return None if _unrestricted(scope) else scope.project_id


def _job_owner_filter(scope: CallerScope) -> tuple[str | None, str | None]:
    """(project, user) job filter: the caller's own jobs, as GET /umrs/jobs."""
    if _unrestricted(scope):
        return None, None
    return scope.project_id, scope.user_id or None


def _job_visible(job: Any, scope: CallerScope) -> bool:
    """May the caller see this job (its project and, when known, its owner)?"""
    project, user = _job_owner_filter(scope)
    if project is None:
        return True
    return job.project_id == project and user in (None, job.user_id)


def _model_access_context(scope: CallerScope) -> Any:
    """Model-registry ACL context for the caller (verified project only).

    Deliberately stricter than ``api.models.get_access_context``: the
    ``X-Verified-Project-Memberships`` header is not covered by the gateway
    HMAC (``sign_claims`` binds user, project and admin only), so a replayed
    signed tuple could carry any membership list. The dashboard trusts the
    signed project alone; a multi-project user sees other projects' models
    here only after switching the active project. Revisit once the gateway
    signs the memberships list.
    """
    from ..model_registry.store import AccessContext

    return AccessContext(
        user_id=scope.user_id,
        is_admin=_unrestricted(scope),
        project_memberships={scope.project_id} if scope.project_id else set(),
    )


def _visible_executions(scope: CallerScope) -> list[Any]:
    """Workflow executions the caller may read (raises when the registry fails)."""
    from ..server.workflow_execution_registry import get_execution_registry
    from .workflow_auth import can_access_execution

    if _unrestricted(scope):
        return get_execution_registry().list()
    records = get_execution_registry().list(user_id=scope.user_id)
    return [
        r
        for r in records
        if can_access_execution(scope.user_id, scope.project_id or None, record=r)
    ]


def _require_readable_project(project_id: str, scope: CallerScope) -> None:
    """403 when the requested project is outside the caller's access tier."""
    if not project_readable(project_id, scope):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "scope_denied",
                "message": "Project is outside the caller's access tier.",
            },
        )


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
    # P2: per-connector sync state hydrated from connector_sync_state
    # (cursor age, docs/errors, DLQ depth). Empty until syncs run (P2
    # runner I2) — render as "n/a "honestly.
    sync: dict[str, dict] = {}


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
async def get_overview(
    request: Request, scope: CallerScope = Depends(get_caller_scope)
) -> OverviewResponse:
    """Return aggregate job and model counts plus system health."""
    app = request.app
    running, queued, completed, failed = _job_status_counts(scope)
    total_jobs = running + queued + completed + failed

    # Model counts
    (
        total_models,
        available_models,
        staging_models,
        deprecated_models,
    ) = _model_status_counts(scope)

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
        workflows=_workflow_execution_counts(scope),
        connectors=_connector_inventory(scope),
    )


def _job_status_counts(scope: CallerScope) -> tuple[int, int, int, int]:
    """(running, queued, completed, failed) counts of the caller's jobs.

    Same owner filter as GET /jobs (admins: platform-wide). H-1 (2026-10-18):
    this asked for ``count_by_status``, which no JobStore implements, so
    every caller saw zeros; ``JobStore.count_jobs`` now backs it.
    """
    job_store = get_optional_job_store()
    count_jobs = getattr(job_store, "count_jobs", None)
    if not callable(count_jobs):
        return 0, 0, 0, 0
    from ..jobs.models import JobStatus

    project, owner = _job_owner_filter(scope)
    counts = []
    for status_val in (
        JobStatus.RUNNING,
        JobStatus.QUEUED,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
    ):
        try:
            counts.append(
                count_jobs(project_id=project, user_id=owner, status=status_val)
            )
        except Exception:  # noqa: BLE001 — dashboard probes degrade to honest empty
            counts.append(0)
    return counts[0], counts[1], counts[2], counts[3]


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


def _workflow_execution_counts(scope: CallerScope) -> WorkflowExecutionCounts | None:
    """Aggregate the caller's visible workflow executions; None when the
    registry is unavailable."""
    try:
        records = _visible_executions(scope)
    except Exception:
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


def _connector_inventory(scope: CallerScope) -> ConnectorInventory | None:
    """Connector inventory with availability probe (U3).

    Falls back to the static U1 type list when the introspection module is
    unavailable; None only when the connectors package itself is missing.
    Per-instance sync rows carry no project: admins only.
    """
    try:
        from ..connectors import introspect
    except Exception:
        logger.debug("connector introspection unavailable", exc_info=True)
        return ConnectorInventory(available_types=list(_CONNECTOR_TYPES))

    report = introspect.inventory()
    return ConnectorInventory(
        available_types=report.available,
        unavailable=report.unavailable,
        details={c.name: c.to_dict() for c in report.connectors},
        sync=_connector_sync_view() if _unrestricted(scope) else {},
    )


def _connector_sync_view() -> dict[str, dict]:
    """Per-connector sync state for the overview payload (P2).

    Degrades to {} when the store is unavailable (lib-only runner yet to
    execute a sync, or no DB) — dashboards render "n/a "honestly.
    """
    try:
        from ..connectors.sync_state import build_default_store

        store = build_default_store()
        return {
            row.name: {
                "last_sync_at": row.last_sync_at,
                "status": row.last_status,
                "docs_synced": row.docs_synced,
                "errors": row.errors_count,
                "dlq": len(row.dlq),
            }
            for row in store.list()
        }
    except Exception:
        logger.debug("connector sync state unavailable", exc_info=True)
        return {}


# ── Overview/health/list helpers (complexity decomposition) ─────────────


def _model_status_counts(scope: CallerScope) -> tuple[int, int, int, int]:
    """Return (total, available, staging, deprecated) counts of the models
    the caller may read."""
    model_store = get_optional_model_store()
    if not model_store:
        return 0, 0, 0, 0
    try:
        models = model_store.list_models(ctx=_model_access_context(scope))
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


def _presence_component(
    name: str, store: Any, *, healthy_detail: str
) -> HealthComponent:
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
    page: tuple[int, int],
    scope: CallerScope,
) -> list[Any]:
    """Store query scoped to the caller + in-memory user filter; empty on
    any failure."""
    from ..jobs.models import JobStatus, JobType

    project, owner = _job_owner_filter(scope)
    try:
        jobs = job_store.list_jobs(
            project_id=project,
            status=JobStatus(status) if status else None,
            job_type=JobType(job_type) if job_type else None,
            limit=page[0],
            offset=page[1],
            user_id=owner,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dashboard job query failed: %s", exc)
        return []
    if user_id:
        jobs = [job for job in jobs if job.user_id == user_id]
    return jobs


def _job_count_owner(
    scope: CallerScope, user_id: str | None
) -> tuple[str | None, str | None] | None:
    """(project, user) the job-list total counts under: the owner filter
    narrowed by the ``user_id`` query, as ``_fetch_jobs`` narrows the page.
    None when a restricted caller names another user (the list is empty)."""
    project, owner = _job_owner_filter(scope)
    if not user_id:
        return project, owner
    if owner is not None and user_id != owner:
        return None
    return project, user_id


def _job_total(
    job_store: Any,
    jobs: list[Any],
    filters: tuple[Any, str | None, str | None],
    scope: CallerScope,
) -> int:
    """Store-backed total of the caller's jobs (admins: platform-wide) under
    the list's (status, job_type, user_id) filters, else the window size.

    H-1 (2026-10-18): was ``count_all``, which no JobStore implements; the
    total is now ``count_jobs`` under the owner filter and the list filters.
    """
    status_filter, job_type, user_id = filters
    owner_filter = _job_count_owner(scope, user_id)
    if owner_filter is None:
        return 0
    count_jobs = getattr(job_store, "count_jobs", None)
    try:
        if callable(count_jobs):
            from ..jobs.models import JobType

            return count_jobs(
                project_id=owner_filter[0],
                user_id=owner_filter[1],
                status=status_filter,
                job_type=JobType(job_type) if job_type else None,
            )
    except Exception as exc:  # noqa: BLE001 — e.g. an unknown job_type (page empty too)
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
        logger.warning("Could not load versions for model %s: %s", model.model_id, exc)
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
    scope: CallerScope = Depends(get_caller_scope),
) -> PaginatedJobsResponse:
    """List the caller's jobs (admin: all) with pagination and filters."""
    from ..jobs.models import JobStatus

    job_store = get_job_store()
    if not job_store:
        return PaginatedJobsResponse(
            items=[],
            pagination=PaginationMeta(
                total=0, limit=limit, offset=offset, has_next=False, has_prev=False
            ),
        )

    jobs = _fetch_jobs(job_store, status, job_type, user_id, (limit, offset), scope)
    # The envelope holds dicts: a JobSummary model failed validation (500).
    items = [_job_summary(j).model_dump() for j in jobs]
    status_filter = JobStatus(status) if status else None
    total = _job_total(job_store, jobs, (status_filter, job_type, user_id), scope)
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
    caller: CallerScope = Depends(get_caller_scope),
) -> PaginatedModelsResponse:
    """List the models the caller may read, with pagination and filters."""
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
            ctx=_model_access_context(caller),
            model_type=ModelType(model_type) if model_type else None,
            status=ModelStatus(status) if status else None,
            scope=ModelScope(scope) if scope else None,
            limit=limit,
            offset=offset,
        )
    except Exception:  # noqa: BLE001 — dashboard probes degrade to honest empty
        models = []

    items = [_model_summary(model_store, model).model_dump() for model in models]

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
async def get_job_detail(
    job_id: str, scope: CallerScope = Depends(get_caller_scope)
) -> JobDetailResponse:
    """Get detailed job information including attempts (the caller's own
    jobs; another project's or user's job reads as not found)."""
    store = get_job_store()
    try:
        job = store.get_job(job_id)
    except Exception:  # noqa: BLE001 — dashboard probes degrade to honest empty
        job = None
    if job is None or not _job_visible(job, scope):
        raise_typed(
            PlatformError.dashboard("not_found", f"Job not found: {job_id}"),
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
    scope: CallerScope = Depends(get_caller_scope),
) -> ModelDetailResponse:
    """Get detailed model information including versions (models the caller
    may read; any other reads as not found)."""
    from ..api.models import get_model_store as _get_model_store

    store = _get_model_store()
    try:
        model = store.get_model(model_id, ctx=_model_access_context(scope))
    except Exception:  # noqa: BLE001 — dashboard probes degrade to honest empty
        raise_typed(
            PlatformError.dashboard("not_found", f"Model not found: {model_id}"),
        )

    # Get versions
    versions: list[dict[str, Any]] = []
    latest_version: str | None = None
    try:
        if hasattr(store, "get_model_versions"):
            hist = store.get_model_versions(model_id)
            versions = [v.to_dict() for v in hist.versions]
            latest_version = hist.latest.version if hist.latest else None
    except Exception:  # noqa: BLE001, S110 — dashboard probes degrade to honest empty
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
    scope: CallerScope = Depends(get_caller_scope),
) -> NotificationsResponse:
    """Return pending notifications — store-backed since 0109 (D-36 leg B).

    Unread rows from ``platform_notifications`` (budget-exhausted stops
    and kindred operator-visible events), newest-first. The store being
    absent (no pool) returns the honest empty list — never fabricated.
    Non-admin callers see their verified project's rows only.
    """
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return NotificationsResponse(notifications=[], total=0)
    import asyncio

    from umrs.platform_notifications import NotificationStore

    def _load():
        conn = pool.getconn()
        try:
            return NotificationStore(conn).list(
                project_id=_project_filter(scope), unread_only=True, limit=50
            )
        finally:
            pool.putconn(conn)

    rows = await asyncio.to_thread(_load)
    items = [
        NotificationItem(
            category=str(row.get("kind") or ""),
            message=str(row.get("title") or ""),
            details={
                "id": row.get("id"),
                "project_id": row.get("project_id"),
                "created_at": row.get("created_at"),
                **(row.get("detail") or {}),
            },
            suggestion=None,
            retry_after=None,
        )
        for row in rows
    ]
    return NotificationsResponse(notifications=items, total=len(items))


@router.get(
    "/sse/jobs",
    dependencies=[Depends(get_caller_scope)],
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
            await asyncio.sleep(30)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ===========================================================================
# U6: Dashboard panels (workflows / connectors / knowledge)
# ===========================================================================


class WorkflowRunSummary(BaseModel):
    """One workflow execution row for the dashboard."""

    execution_id: str
    workflow_name: str
    status: str
    submitted_at: float
    finished_at: float | None = None
    gate_count: int = 0


class WorkflowsPanelResponse(BaseModel):
    """Workflow executions panel: counts by status + recent rows."""

    counts: dict[str, int]
    recent: list[WorkflowRunSummary]
    degraded: bool = False
    degraded_reason: str = ""


class ConnectorSyncRow(BaseModel):
    """One connector's sync-state row for the intake panel."""

    status: str
    last_sync_at: str | None = None
    docs_synced: int = 0
    errors: int = 0
    dlq: int = 0


class ConnectorsPanelResponse(BaseModel):
    """Connector intake panel (null-safe when the store is unavailable)."""

    connectors: dict[str, ConnectorSyncRow] | None = None
    degraded: bool = False
    degraded_reason: str = ""


class KnowledgePanelResponse(BaseModel):
    """Per-project knowledge stats panel."""

    project_id: str
    artifact_count: int = 0
    active_artifact_count: int = 0
    document_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    last_confidence: float | None = None
    degraded: bool = False
    degraded_reason: str = ""


def _workflow_summary(record: Any) -> WorkflowRunSummary:
    """Execution record → dashboard row."""
    return WorkflowRunSummary(
        execution_id=record.execution_id,
        workflow_name=record.workflow_name,
        status=record.status,
        submitted_at=record.submitted_at,
        finished_at=record.finished_at,
        gate_count=len(getattr(record, "gate_decisions", {}) or {}),
    )


def _workflows_panel_payload(
    scope: CallerScope, *, recent_limit: int = 25
) -> WorkflowsPanelResponse:
    """Counts + recent executions the caller may read (null-safe when the
    registry is unavailable)."""
    try:
        records = _visible_executions(scope)
    except Exception as exc:  # noqa: BLE001 — dashboard probes degrade to honest empty
        logger.warning("Workflow panel degraded: %s", exc)
        return WorkflowsPanelResponse(
            counts={},
            recent=[],
            degraded=True,
            degraded_reason=str(exc),
        )

    counts: dict[str, int] = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    recent = [_workflow_summary(r) for r in records[:recent_limit]]
    return WorkflowsPanelResponse(counts=counts, recent=recent)


@router.get(
    "/workflows",
    response_model=WorkflowsPanelResponse,
    summary="Workflow executions panel",
    description="Counts by status plus the most recent workflow executions.",
)
async def get_workflows_panel(
    scope: CallerScope = Depends(get_caller_scope),
) -> WorkflowsPanelResponse:
    """Workflow executions panel (W1 registry view for the dashboard)."""
    return _workflows_panel_payload(scope)


def _connectors_panel_payload(scope: CallerScope) -> ConnectorsPanelResponse:
    """Connector intake rows from the sync-state store (null-safe).

    The rows carry no project: admins only, others get an empty map.
    """
    if not _unrestricted(scope):
        return ConnectorsPanelResponse(connectors={})
    try:
        from ..connectors.sync_state import build_default_store

        rows = build_default_store().list()
    except Exception as exc:  # noqa: BLE001 — dashboard probes degrade to honest empty
        logger.warning("Connectors panel degraded: %s", exc)
        return ConnectorsPanelResponse(
            connectors=None,
            degraded=True,
            degraded_reason=str(exc),
        )

    connectors = {
        row.name: ConnectorSyncRow(
            status=row.last_status,
            last_sync_at=row.last_sync_at,
            docs_synced=row.docs_synced,
            errors=row.errors_count,
            dlq=len(row.dlq),
        )
        for row in rows
    }
    return ConnectorsPanelResponse(connectors=connectors)


@router.get(
    "/connectors",
    response_model=ConnectorsPanelResponse,
    summary="Connector intake panel",
    description="Per-connector sync state (status, last sync, counts, DLQ depth).",
)
async def get_connectors_panel(
    scope: CallerScope = Depends(get_caller_scope),
) -> ConnectorsPanelResponse:
    """Connector intake panel (I-sync state view)."""
    return _connectors_panel_payload(scope)


async def _knowledge_panel_payload(
    project_id: str, store: Any, verification_store: Any
) -> KnowledgePanelResponse:
    """Per-project knowledge aggregates from the shared stores."""
    artifacts, _total = await store.list_artifacts(
        project_id=project_id, status=None, page=1, page_size=1000
    )
    active = [a for a in artifacts if getattr(a, "status", "active") == "active"]
    documents = await store.list_irs(project_id=project_id)
    nodes = await store.list_nodes(project_id=project_id)
    edges = await store.list_edges(project_id=project_id)

    last_confidence: float | None = None
    if verification_store is not None:
        try:
            reports = await verification_store.list_reports(
                project_id=project_id, limit=1
            )
            if reports:
                last_confidence = float(getattr(reports[0], "confidence", 0.0))
        except Exception:  # noqa: BLE001 — dashboard probes degrade to honest empty
            logger.debug("verification store unavailable for project %s", project_id)

    return KnowledgePanelResponse(
        project_id=project_id,
        artifact_count=_total,
        active_artifact_count=len(active),
        document_count=len(documents),
        node_count=len(nodes),
        edge_count=len(edges),
        last_confidence=last_confidence,
    )


@router.get(
    "/knowledge",
    response_model=KnowledgePanelResponse,
    summary="Knowledge panel",
    description="Per-project knowledge stats (artifacts, documents, graph, confidence).",
)
async def get_knowledge_panel(
    request: Request,
    project_id: str = Query(
        ..., description="Project scope (knowledge is project-bound)"
    ),
    scope: CallerScope = Depends(get_caller_scope),
) -> KnowledgePanelResponse:
    """Per-project knowledge panel for the dashboard (readable projects only)."""
    _require_readable_project(project_id, scope)
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return KnowledgePanelResponse(
            project_id=project_id,
            degraded=True,
            degraded_reason="database unavailable",
        )
    from ..knowledge.store import KnowledgeStore
    from ..knowledge.verification_store import VerificationStore

    return await _knowledge_panel_payload(
        project_id, KnowledgeStore(pool), VerificationStore(pool)
    )


# ---------------------------------------------------------------------------
# U6: SSE channels (snapshot stream + keepalive)
# ---------------------------------------------------------------------------


def _sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


async def _sse_snapshot_stream(
    *,
    snapshot_fn,
    channel: str,
    interval: float,
) -> Any:
    """SSE stream yielding one snapshot frame immediately, then keepalives.

    First frame: ``event: <channel>-snapshot`` with a JSON payload from
    ``snapshot_fn()`` — clients render instantly without waiting a full
    poll cycle. Errors inside the snapshot never break the stream (the
    frame carries ``degraded`` payloads).
    """
    payload = json.dumps(snapshot_fn(), default=str)
    yield f"event: {channel}-snapshot\ndata: {payload}\n\n"
    while True:
        yield ": keepalive\n\n"
        await asyncio.sleep(interval)


def _workflow_sse_payload(scope: CallerScope) -> dict[str, Any]:
    """JSON-safe view of the caller's workflows panel."""
    panel = _workflows_panel_payload(scope, recent_limit=10)
    return json.loads(panel.model_dump_json())


def _intake_sse_payload(scope: CallerScope) -> dict[str, Any]:
    """JSON-safe view of the caller's connectors panel."""
    panel = _connectors_panel_payload(scope)
    return json.loads(panel.model_dump_json())


@router.get(
    "/sse/workflows",
    summary="Workflow execution SSE snapshots",
    description="Immediate workflow panel snapshot frame + keepalives.",
)
async def workflows_sse(
    request: Request, scope: CallerScope = Depends(get_caller_scope)
):
    """SSE channel for workflow executions (snapshot-on-connect)."""
    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        _sse_snapshot_stream(
            snapshot_fn=lambda: _workflow_sse_payload(scope),
            channel="workflows",
            interval=float(os.getenv("UMRS_SSE_POLL_SECONDS", "30")),
        ),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )


@router.get(
    "/sse/intake",
    summary="Connector intake SSE snapshots",
    description="Immediate intake panel snapshot frame + keepalives.",
)
async def intake_sse(request: Request, scope: CallerScope = Depends(get_caller_scope)):
    """SSE channel for connector intake (snapshot-on-connect)."""
    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        _sse_snapshot_stream(
            snapshot_fn=lambda: _intake_sse_payload(scope),
            channel="intake",
            interval=float(os.getenv("UMRS_SSE_POLL_SECONDS", "30")),
        ),
        media_type="text/event-stream",
        headers=_sse_headers(),
    )
