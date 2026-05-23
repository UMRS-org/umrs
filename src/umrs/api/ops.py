"""Operational introspection REST API (COR-006).

Provides operator-facing inspection of the running UMRS instance:

- GET  /umrs/ops/health       — Subsystem health summary
- GET  /umrs/ops/topology     — Active runtime topology (jobs, transports, bridges)
- GET  /umrs/ops/timeline     — Job-attempt timeline for recent activity
- GET  /umrs/ops/ping         — Lightweight liveness check (no admin required)
- POST /umrs/ops/reload       — Trigger configuration reload (admin-only)
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/ops", tags=["ops"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class SubsystemHealth(BaseModel):
    name: str
    status: str  # "healthy" | "degraded" | "unavailable"
    detail: str = ""
    last_activity: str | None = None


class OpsHealthResponse(BaseModel):
    instance_id: str
    uptime_seconds: float
    subsystems: list[SubsystemHealth]
    overall: str  # "healthy" | "degraded" | "unhealthy"


class TopologyEntry(BaseModel):
    component: str
    role: str
    status: str
    detail: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class OpsTopologyResponse(BaseModel):
    instance_id: str
    timestamp: str
    components: list[TopologyEntry]
    active_jobs: int = 0
    active_attempts: int = 0


class TimelineEvent(BaseModel):
    timestamp: str
    job_id: str
    job_type: str
    event: str  # "created" | "queued" | "attempt_started" | "succeeded" | "failed" | "lost"
    attempt_id: str | None = None
    attempt_no: int | None = None
    backend: str | None = None
    detail: str = ""


class OpsTimelineResponse(BaseModel):
    events: list[TimelineEvent]
    total: int
    window_minutes: int


# ---------------------------------------------------------------------------
# Module-level state (set during app startup)
# ---------------------------------------------------------------------------

_boot_time: float = time.monotonic()
_instance_id: str = ""


def set_ops_context(instance_id: str) -> None:
    """Set the UMRS instance ID for ops responses (called at startup)."""
    global _instance_id, _boot_time
    _instance_id = instance_id
    _boot_time = time.monotonic()


def _uptime() -> float:
    return round(time.monotonic() - _boot_time, 1)


def _lazy_get_optional_job_store() -> Any:
    """Lazy import to avoid triggering umrs.jobs circular import at module level."""
    from .jobs import get_optional_job_store
    return get_optional_job_store()


def require_ops_admin(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    caller_user_id: str | None = Query(None, description="Caller's user ID"),
    caller_project_id: str | None = Query(None, description="Caller's project ID"),
    is_admin: bool = Query(False, description="Admin flag (legacy, ignored in strict mode)"),
) -> None:
    """Enforce admin-only access for ops endpoints.

    Ops endpoints expose internal infrastructure details (instance IDs,
    transport config, active backends, job timelines). Only admin callers
    may access them.

    Uses the same verified-header pattern as ``get_caller_identity`` but
    avoids importing it directly to prevent circular imports in tests.
    """
    import os

    # Check Gateway-verified headers first (production path)
    if x_user_id and x_verified_project_id:
        admin = (x_verified_is_admin or "").strip() in ("true", "1")
        if not admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Ops endpoints require admin privileges",
            )
        return

    # Strict mode: reject if no verified headers
    if os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Gateway-verified caller identity",
        )

    # MVP/test mode: check query-param is_admin
    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Ops endpoints require admin privileges",
        )
    return


# ---------------------------------------------------------------------------
# Subsystem health probes
# ---------------------------------------------------------------------------

def _probe_job_store(store: Any) -> SubsystemHealth:
    if store is None:
        return SubsystemHealth(name="job_store", status="unavailable", detail="not initialized")
    try:
        # Lightweight probe: list 1 job
        store.list_jobs(limit=1)
        return SubsystemHealth(name="job_store", status="healthy", detail="connected")
    except Exception as exc:
        return SubsystemHealth(name="job_store", status="degraded", detail=str(exc)[:200])


def _probe_scheduler() -> SubsystemHealth:
    """Check if the scheduler loop is presumably alive.

    We cannot import the scheduler directly (that would violate internal
    boundary rules), so we rely on the environment hint that the process
    is configured to run a scheduler.
    """
    if os.environ.get("UMRS_SCHEDULER_ENABLED", "true").lower() == "false":
        return SubsystemHealth(name="scheduler", status="unavailable", detail="disabled by config")
    return SubsystemHealth(name="scheduler", status="healthy", detail="enabled")


def _probe_transport() -> SubsystemHealth:
    """Report transport configuration health."""
    transport = os.environ.get("UMRS_TRANSPORT", "http")
    tunnel = os.environ.get("UMRS_TUNNEL_MODE", "false").lower() == "true"
    detail = f"transport={transport}"
    if tunnel:
        detail += ", tunnel_mode=true (zmq bound to 127.0.0.1)"
    return SubsystemHealth(name="transport", status="healthy", detail=detail)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=OpsHealthResponse,
    summary="Subsystem health summary",
)
def ops_health(
    _admin: None = Depends(require_ops_admin),
    store=Depends(lambda: _lazy_get_optional_job_store()),
) -> dict[str, Any]:
    """Return health status of all UMRS subsystems (admin-only)."""
    subsystems = [
        _probe_job_store(store),
        _probe_scheduler(),
        _probe_transport(),
    ]

    status_levels = {"healthy": 0, "degraded": 1, "unavailable": 2}
    worst = max(subsystems, key=lambda s: status_levels.get(s.status, 2))
    overall = worst.status if worst.status != "unavailable" else "unhealthy"

    return {
        "instance_id": _instance_id,
        "uptime_seconds": _uptime(),
        "subsystems": [s.model_dump() for s in subsystems],
        "overall": overall,
    }


@router.get(
    "/topology",
    response_model=OpsTopologyResponse,
    summary="Active runtime topology",
)
def ops_topology(
    _admin: None = Depends(require_ops_admin),
    store=Depends(lambda: _lazy_get_optional_job_store()),
) -> dict[str, Any]:
    """Return the active runtime topology (admin-only)."""
    components: list[dict[str, Any]] = []

    # Report gateway mode
    gw_mode = os.environ.get("GATEWAY_MODE", "nginx")
    components.append({
        "component": "gateway",
        "role": "policy_edge",
        "status": "configured",
        "detail": f"mode={gw_mode}",
    })

    # Report transport
    transport = os.environ.get("UMRS_TRANSPORT", "http")
    components.append({
        "component": "transport",
        "role": "data_plane",
        "status": "active",
        "detail": f"type={transport}",
    })

    active_jobs = 0
    active_attempts = 0

    if store is not None:
        try:
            from ..jobs.models import JobStatus
            running_jobs = store.list_jobs(status=JobStatus.RUNNING, limit=1000)
            active_jobs = len(running_jobs)

            active_atts = store.get_active_attempts()
            active_attempts = len(active_atts)

            # Report active backends
            backends: dict[str, int] = {}
            for att in active_atts:
                backends[att.backend] = backends.get(att.backend, 0) + 1
            for backend, count in backends.items():
                components.append({
                    "component": f"backend:{backend}",
                    "role": "execution_authority",
                    "status": "active",
                    "detail": f"{count} active attempt(s)",
                    "metadata": {"count": count},
                })
        except Exception as exc:
            logger.warning("Failed to query job store for topology: %s", exc)

    return {
        "instance_id": _instance_id,
        "timestamp": datetime.utcnow().isoformat(),
        "components": components,
        "active_jobs": active_jobs,
        "active_attempts": active_attempts,
    }


@router.get(
    "/timeline",
    response_model=OpsTimelineResponse,
    summary="Job-attempt timeline",
)
def ops_timeline(
    window_minutes: int = Query(60, ge=1, le=1440, description="Look-back window in minutes"),
    limit: int = Query(100, ge=1, le=500),
    _admin: None = Depends(require_ops_admin),
    store=Depends(lambda: _lazy_get_optional_job_store()),
) -> dict[str, Any]:
    """Return a timeline of recent job/attempt events (admin-only)."""
    events: list[dict[str, Any]] = []

    if store is None:
        return {"events": [], "total": 0, "window_minutes": window_minutes}

    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)

    try:
        # Get recently updated jobs
        recent_jobs = store.list_jobs(limit=limit * 2)

        for job in recent_jobs:
            if job.updated_at and job.updated_at < cutoff:
                continue

            # Job creation event
            if job.created_at and job.created_at >= cutoff:
                events.append({
                    "timestamp": job.created_at.isoformat(),
                    "job_id": job.job_id,
                    "job_type": job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type),
                    "event": "created",
                    "detail": f"status={job.status.value if hasattr(job.status, 'value') else job.status}",
                })

            # Attempt events
            attempts = store.get_attempts_for_job(job.job_id)
            for att in attempts:
                jt = job.job_type.value if hasattr(job.job_type, "value") else str(job.job_type)

                if att.started_at and att.started_at >= cutoff:
                    events.append({
                        "timestamp": att.started_at.isoformat(),
                        "job_id": job.job_id,
                        "job_type": jt,
                        "event": "attempt_started",
                        "attempt_id": att.attempt_id,
                        "attempt_no": att.attempt_no,
                        "backend": att.backend,
                        "detail": att.status_detail or "",
                    })

                if att.ended_at and att.ended_at >= cutoff:
                    att_status = att.status.value if hasattr(att.status, "value") else str(att.status)
                    event_type = "succeeded" if att_status == "succeeded" else (
                        "lost" if att_status == "lost" else "failed"
                    )
                    events.append({
                        "timestamp": att.ended_at.isoformat(),
                        "job_id": job.job_id,
                        "job_type": jt,
                        "event": event_type,
                        "attempt_id": att.attempt_id,
                        "attempt_no": att.attempt_no,
                        "backend": att.backend,
                        "detail": att.exit_reason or "",
                    })

    except Exception as exc:
        logger.warning("Failed to build timeline: %s", exc)

    # Sort by timestamp descending
    events.sort(key=lambda e: e["timestamp"], reverse=True)
    events = events[:limit]

    return {
        "events": events,
        "total": len(events),
        "window_minutes": window_minutes,
    }


# ---------------------------------------------------------------------------
# Lightweight ping (no admin required) — used by umrs-client over HTTP
# ---------------------------------------------------------------------------

class OpsPingResponse(BaseModel):
    status: str
    server_ready: bool
    uptime_seconds: float = 0.0


@router.get(
    "/ping",
    response_model=OpsPingResponse,
    summary="Lightweight readiness check",
)
def ops_ping() -> dict[str, Any]:
    """Return server readiness without requiring admin privileges.

    Intended for ``umrs-client ping --gateway-url`` to verify connectivity
    via the HTTP Gateway when ZMQ is not directly reachable.

    NOTE: This endpoint is unauthenticated — do not expose internal state
    beyond a boolean readiness flag.
    """
    return {
        "status": "ok",
        "server_ready": True,
        "uptime_seconds": 0.0,  # omitted from unauthenticated response
    }


# ---------------------------------------------------------------------------
# Configuration reload via HTTP (admin-only)
# ---------------------------------------------------------------------------

class OpsReloadRequest(BaseModel):
    """Optional request body for reload.  Currently empty — the server
    re-reads its own ``configure_file`` from disk."""
    pass


class OpsReloadResponse(BaseModel):
    status: str
    summary: str = ""
    error: str | None = None


# Module-level references set during startup so reload can access server state.
_reload_configure_file: str | None = None
_reload_buckets_cfg: Any = None
_reload_user_config_map: Any = None


def set_reload_context(
    configure_file: str,
    buckets_cfg: Any,
    user_config_map: Any,
) -> None:
    """Provide server state references needed by the HTTP reload endpoint.

    Called once during UMRS server startup, after the ZMQ main loop state
    is initialised.
    """
    global _reload_configure_file, _reload_buckets_cfg, _reload_user_config_map
    _reload_configure_file = configure_file
    _reload_buckets_cfg = buckets_cfg
    _reload_user_config_map = user_config_map


@router.post(
    "/reload",
    response_model=OpsReloadResponse,
    summary="Trigger configuration reload",
)
def ops_reload(
    _admin: None = Depends(require_ops_admin),
) -> dict[str, Any]:
    """Re-read the server configuration from disk and re-provision users.

    This is the HTTP equivalent of the ZMQ ``RELOAD`` command.  The server
    re-reads ``configure_file``, creates any missing MinIO/Postgres
    resources, and updates the live ``buckets_cfg``.

    Requires admin privileges.
    """
    if _reload_configure_file is None or _reload_buckets_cfg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reload context not initialised — server may still be starting",
        )

    from umrs.server.maintenance import reload_configuration

    outcome = reload_configuration(
        _reload_configure_file,
        _reload_buckets_cfg,
        _reload_user_config_map,
        log_output=True,
    )

    if "error" in outcome:
        return {
            "status": "error",
            "summary": outcome.get("summary", ""),
            "error": outcome["error"],
        }
    return {
        "status": "ok",
        "summary": outcome.get("summary", "Reload complete"),
    }


# ---------------------------------------------------------------------------
# Audit trace inspection
# ---------------------------------------------------------------------------


@router.get("/audit/trace")
async def audit_trace(
    request_id: str = Query("", description="Correlation/request ID to filter by"),
    event: str = Query("", description="Event name substring filter"),
    actor: str = Query("", description="Actor/user filter"),
    project_id: str = Query("", description="Project ID filter"),
    since: float = Query(0.0, description="Unix timestamp lower bound"),
    until: float = Query(0.0, description="Unix timestamp upper bound"),
    limit: int = Query(50, ge=1, le=500, description="Maximum results"),
):
    """Query audit events for distributed trace reconstruction.

    Uses persistent Postgres store when available, falls back to in-memory
    ring buffer. Filter by request_id to see all operations in a single
    request chain. Use since/until for time-window queries.
    """
    from .audit_events import query_audit_events

    events = query_audit_events(
        request_id=request_id,
        event=event,
        actor=actor,
        since=since,
        until=until,
        limit=limit,
    )
    # Apply project_id filter post-query (ring buffer doesn't support it natively)
    if project_id:
        events = [e for e in events if e.get("project_id") == project_id]
    return {"events": events, "count": len(events), "persistent": _audit_store_available()}


@router.delete("/audit/retention")
async def audit_retention_cleanup(
    days: int = Query(30, ge=1, le=365, description="Delete events older than N days"),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """Delete audit events older than the specified retention window.

    Requires admin authentication via x-admin-token header.
    """
    expected = os.environ.get("UMRS_ADMIN_TOKEN", "")
    if not expected or admin_token != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required")

    from .audit_events import _audit_store

    if _audit_store is None:
        return {"status": "skipped", "message": "No persistent audit store configured"}

    cutoff = time.time() - (days * 86400)
    deleted = _audit_store.delete_before(cutoff)
    return {
        "status": "ok",
        "deleted": deleted,
        "retention_days": days,
        "cutoff_ts": cutoff,
    }


def _audit_store_available() -> bool:
    """Check if persistent audit store is configured."""
    from .audit_events import _audit_store
    return _audit_store is not None


# ---------------------------------------------------------------------------
# Governance introspection
# ---------------------------------------------------------------------------


@router.get("/governance/ownership")
async def governance_ownership(
    owner_id: str = Query("", description="Filter by owner"),
    project_id: str = Query("", description="Filter by project"),
    resource_type: str = Query("", description="Filter by resource type"),
):
    """List resource ownership records."""
    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    if owner_id:
        records = store.list_owned(
            owner_id, resource_type=resource_type, project_id=project_id,
        )
    elif project_id:
        records = store.list_by_project(project_id, resource_type=resource_type)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide owner_id or project_id as filter",
        )

    return {"records": records, "count": len(records)}


@router.post("/governance/ownership/transfer")
async def governance_transfer(
    resource_type: str = Query(..., description="Resource type"),
    resource_id: str = Query(..., description="Resource identifier"),
    new_owner_id: str = Query(..., description="New owner user ID"),
    new_project_id: str | None = Query(None, description="Move to new project"),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """Transfer ownership of a resource (admin-only)."""
    expected = os.environ.get("UMRS_ADMIN_TOKEN", "")
    if not expected or admin_token != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required")

    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    updated = store.transfer_owner(
        resource_type, resource_id, new_owner_id,
        new_project_id=new_project_id,
    )
    if not updated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No ownership record: {resource_type}/{resource_id}",
        )

    from .audit_events import emit_audit
    emit_audit(
        "governance.ownership_transferred",
        details={
            "resource_type": resource_type,
            "resource_id": resource_id,
            "new_owner": new_owner_id,
            "new_project": new_project_id,
        },
    )
    return {"status": "transferred", "resource_type": resource_type, "resource_id": resource_id}


@router.get("/governance/quotas")
async def governance_quotas(
    project_id: str = Query(..., description="Project ID"),
):
    """List quotas for a project."""
    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    quotas = store.list_quotas(project_id)
    return {"project_id": project_id, "quotas": quotas}


@router.put("/governance/quotas")
async def governance_set_quota(
    project_id: str = Query(..., description="Project ID"),
    resource_type: str = Query(..., description="Resource type (endpoints, mcp_servers, etc.)"),
    max_count: int = Query(..., ge=0, description="Maximum allowed count"),
    warn_count: int | None = Query(None, ge=0, description="Soft warning threshold (optional)"),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """Set quota for a project + resource type (admin-only).

    When warn_count is set, the system emits a governance.quota_warning
    audit event when usage reaches the warning threshold but does not
    block operations until max_count is reached.
    """
    expected = os.environ.get("UMRS_ADMIN_TOKEN", "")
    if not expected or admin_token != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required")

    if warn_count is not None and warn_count >= max_count:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"warn_count ({warn_count}) must be less than max_count ({max_count})",
        )

    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    store.set_quota(project_id, resource_type, max_count, warn_count=warn_count)

    from .audit_events import emit_audit
    emit_audit(
        "governance.quota_set",
        project_id=project_id,
        details={
            "resource_type": resource_type,
            "max_count": max_count,
            "warn_count": warn_count,
        },
    )
    return {
        "status": "ok",
        "project_id": project_id,
        "resource_type": resource_type,
        "max_count": max_count,
        "warn_count": warn_count,
    }


@router.get("/governance/orphans")
async def governance_orphans(
    resource_type: str = Query(..., description="Resource type to check"),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """Detect orphaned ownership records for a resource type (admin-only).

    Compares governance ownership records against live registries.
    """
    expected = os.environ.get("UMRS_ADMIN_TOKEN", "")
    if not expected or admin_token != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required")

    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    # Gather known live IDs from the relevant registry
    known_ids: list[str] = []
    if resource_type == "endpoints":
        try:
            from .endpoints import get_registry
            registry = get_registry()
            all_endpoints = registry.list_all()
            known_ids = [ep.job_id for ep in all_endpoints]
        except Exception:
            logger.warning("Cannot enumerate live endpoints for orphan check")
    elif resource_type == "mcp_servers":
        try:
            from .mcp import _get_store as _get_mcp_store
            mcp_store = _get_mcp_store()
            servers = mcp_store.list_servers()
            known_ids = [s.server_id for s in servers]
        except Exception:
            logger.warning("Cannot enumerate live MCP servers for orphan check")

    orphans = store.find_orphans(resource_type, known_ids)
    return {"resource_type": resource_type, "orphans": orphans, "count": len(orphans)}


@router.delete("/governance/orphans")
async def governance_cleanup_orphans(
    resource_type: str = Query(..., description="Resource type to clean up"),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """Remove orphaned ownership records for a resource type (admin-only).

    Compares governance ownership records against live registries,
    then deletes stale entries. Emits audit event for the cleanup.
    """
    expected = os.environ.get("UMRS_ADMIN_TOKEN", "")
    if not expected or admin_token != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required")

    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    known_ids: list[str] = []
    if resource_type == "endpoints":
        try:
            from .endpoints import get_registry
            registry = get_registry()
            all_endpoints = registry.list_all()
            known_ids = [ep.job_id for ep in all_endpoints]
        except Exception:
            logger.warning("Cannot enumerate live endpoints for orphan cleanup")
    elif resource_type == "mcp_servers":
        try:
            from .mcp import _get_store as _get_mcp_store
            mcp_store = _get_mcp_store()
            servers = mcp_store.list_servers()
            known_ids = [s.server_id for s in servers]
        except Exception:
            logger.warning("Cannot enumerate live MCP servers for orphan cleanup")

    deleted = store.cleanup_orphans(resource_type, known_ids)

    from .audit_events import emit_audit
    emit_audit(
        "governance.orphans_cleaned",
        details={
            "resource_type": resource_type,
            "deleted": deleted,
            "live_count": len(known_ids),
        },
    )

    return {"resource_type": resource_type, "deleted": deleted}


# ---------------------------------------------------------------------------
# Schema version introspection
# ---------------------------------------------------------------------------


@router.get("/schema/versions")
async def schema_versions():
    """Report current schema versions for all persistent stores.

    Useful for deployment validation, migration audits, and
    upgrade compatibility checks.
    """
    versions: dict[str, Any] = {}

    # Audit schema
    try:
        from .audit_store import AUDIT_STORE_SCHEMA_VERSION
        versions["audit"] = {
            "code_version": AUDIT_STORE_SCHEMA_VERSION,
            "status": "active" if _audit_store_available() else "unavailable",
        }
    except ImportError:
        versions["audit"] = {"status": "not_installed"}

    # Governance schema
    try:
        from .governance import GOVERNANCE_SCHEMA_VERSION, get_governance_store
        store = get_governance_store()
        versions["governance"] = {
            "code_version": GOVERNANCE_SCHEMA_VERSION,
            "status": "active" if store is not None else "unavailable",
        }
    except ImportError:
        versions["governance"] = {"status": "not_installed"}

    # MCP schema
    try:
        from ..mcp.schema import MCP_SCHEMA_VERSION
        versions["mcp"] = {"code_version": MCP_SCHEMA_VERSION}
    except ImportError:
        versions["mcp"] = {"status": "not_installed"}

    return {"schemas": versions}
