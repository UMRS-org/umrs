"""Operational introspection REST API (COR-006).

Provides operator-facing inspection of the running UMRS instance:

- GET /umrs/ops/health       — Subsystem health summary
- GET /umrs/ops/topology     — Active runtime topology (jobs, transports, bridges)
- GET /umrs/ops/timeline     — Job-attempt timeline for recent activity
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
