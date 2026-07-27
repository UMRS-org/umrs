"""Operational introspection REST API (COR-006).

Provides operator-facing inspection of the running UMRS instance:

- GET  /umrs/ops/health       — Subsystem health summary
- GET  /umrs/ops/topology     — Active runtime topology (jobs, transports, bridges)
- GET  /umrs/ops/timeline     — Job-attempt timeline for recent activity
- GET  /umrs/ops/ping         — Lightweight liveness check (no admin required)
- POST /umrs/ops/reload       — Trigger configuration reload (any authenticated user)
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
    is_admin: bool = Query(
        False, description="Admin flag (legacy, ignored in strict mode)"
    ),
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


def require_authenticated(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    caller_user_id: str | None = Query(None, description="Caller's user ID"),
) -> None:
    """Require a valid authenticated caller but not necessarily admin.

    Used for endpoints like reload that any authenticated user may invoke.
    """
    import os

    # Gateway-verified headers present → authenticated
    if x_user_id and x_verified_project_id:
        return

    # Strict mode: reject if no verified headers
    if os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Gateway-verified caller identity",
        )

    # MVP/test mode: accept if any caller identity is provided
    if caller_user_id:
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
    )


# ---------------------------------------------------------------------------
# Subsystem health probes
# ---------------------------------------------------------------------------


def _probe_job_store(store: Any) -> SubsystemHealth:
    if store is None:
        return SubsystemHealth(
            name="job_store", status="unavailable", detail="not initialized"
        )
    try:
        # Lightweight probe: list 1 job
        store.list_jobs(limit=1)
        return SubsystemHealth(name="job_store", status="healthy", detail="connected")
    except Exception as exc:
        return SubsystemHealth(
            name="job_store", status="degraded", detail=str(exc)[:200]
        )


def _probe_scheduler() -> SubsystemHealth:
    """Check if the scheduler loop is presumably alive.

    We cannot import the scheduler directly (that would violate internal
    boundary rules), so we rely on the environment hint that the process
    is configured to run a scheduler.
    """
    if os.environ.get("UMRS_SCHEDULER_ENABLED", "true").lower() == "false":
        return SubsystemHealth(
            name="scheduler", status="unavailable", detail="disabled by config"
        )
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
    components.append(
        {
            "component": "gateway",
            "role": "policy_edge",
            "status": "configured",
            "detail": f"mode={gw_mode}",
        }
    )

    # Report transport
    transport = os.environ.get("UMRS_TRANSPORT", "http")
    components.append(
        {
            "component": "transport",
            "role": "data_plane",
            "status": "active",
            "detail": f"type={transport}",
        }
    )

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
                components.append(
                    {
                        "component": f"backend:{backend}",
                        "role": "execution_authority",
                        "status": "active",
                        "detail": f"{count} active attempt(s)",
                        "metadata": {"count": count},
                    }
                )
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
    window_minutes: int = Query(
        60, ge=1, le=1440, description="Look-back window in minutes"
    ),
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
                events.append(
                    {
                        "timestamp": job.created_at.isoformat(),
                        "job_id": job.job_id,
                        "job_type": job.job_type.value
                        if hasattr(job.job_type, "value")
                        else str(job.job_type),
                        "event": "created",
                        "detail": f"status={job.status.value if hasattr(job.status, 'value') else job.status}",
                    }
                )

            # Attempt events
            attempts = store.get_attempts_for_job(job.job_id)
            for att in attempts:
                jt = (
                    job.job_type.value
                    if hasattr(job.job_type, "value")
                    else str(job.job_type)
                )

                if att.started_at and att.started_at >= cutoff:
                    events.append(
                        {
                            "timestamp": att.started_at.isoformat(),
                            "job_id": job.job_id,
                            "job_type": jt,
                            "event": "attempt_started",
                            "attempt_id": att.attempt_id,
                            "attempt_no": att.attempt_no,
                            "backend": att.backend,
                            "detail": att.status_detail or "",
                        }
                    )

                if att.ended_at and att.ended_at >= cutoff:
                    att_status = (
                        att.status.value
                        if hasattr(att.status, "value")
                        else str(att.status)
                    )
                    event_type = (
                        "succeeded"
                        if att_status == "succeeded"
                        else ("lost" if att_status == "lost" else "failed")
                    )
                    events.append(
                        {
                            "timestamp": att.ended_at.isoformat(),
                            "job_id": job.job_id,
                            "job_type": jt,
                            "event": event_type,
                            "attempt_id": att.attempt_id,
                            "attempt_no": att.attempt_no,
                            "backend": att.backend,
                            "detail": att.exit_reason or "",
                        }
                    )

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
# Lightweight ping (no admin required) — used by umrs over HTTP
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

    Intended for ``umrs ping --gateway-url`` to verify connectivity
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
# Client connection info (unauthenticated, non-secret)
# ---------------------------------------------------------------------------


class ConnectInfoZMQ(BaseModel):
    host: str
    control_port: int
    data_port_range: dict[str, int]
    auth_type: str
    tunnel_mode: bool


class ConnectInfoServices(BaseModel):
    minio_url: str | None = None
    rag_url: str | None = None


class ConnectInfoResponse(BaseModel):
    zmq: ConnectInfoZMQ
    services: ConnectInfoServices | None = None
    s3cfg_content: str | None = None


@router.get(
    "/connect-info",
    response_model=ConnectInfoResponse,
    summary="ZMQ connection parameters for client auto-discovery",
)
def ops_connect_info(
    x_user_id: str | None = Header(None, alias="X-User-ID"),
) -> dict[str, Any]:
    """Return ZMQ transport parameters so clients can auto-configure.

    This endpoint is unauthenticated for basic discovery (ports, auth
    mode, bind address).  When an ``X-User-ID`` header is provided
    (forwarded by the Gateway after JWT validation), the response
    additionally includes the user's ``s3cfg_content`` so the client
    can write a local S3 configuration file at login time.
    """
    from umrs.shared.connection_profiles import parse_data_port_range

    control_port = int(os.environ.get("UMRS_CONTROL_PORT", "5555"))
    tunnel_mode = os.environ.get("UMRS_TUNNEL_MODE", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    auth_type = os.environ.get("UMRS_ZMQ_AUTH_TYPE", "none").lower()
    # Prefer UMRS_ADVERTISE_HOST (client-facing) over UMRS_SERVER_HOST (bind).
    # 0.0.0.0 is a valid bind address but never a valid client destination.
    host = (
        os.environ.get("UMRS_ADVERTISE_HOST")
        or os.environ.get("UMRS_SERVER_HOST")
        or ("127.0.0.1" if tunnel_mode else "0.0.0.0")
    )

    # Data port range: explicit env or default based on control port
    data_range_env = os.environ.get("UMRS_DATA_PORT_RANGE")
    data_base_env = os.environ.get("UMRS_DATA_BASE_PORT")
    data_port_env = os.environ.get("UMRS_DATA_PORT")

    if data_range_env:
        dr = parse_data_port_range(data_range_env)
        data_start, data_end = dr.start, dr.end
    elif data_base_env:
        data_start = int(data_base_env)
        data_end = data_start + 7
    elif data_port_env:
        data_start = int(data_port_env)
        data_end = data_start
    else:
        data_start = control_port + 1
        data_end = control_port + 8

    result: dict[str, Any] = {
        "zmq": {
            "host": host,
            "control_port": control_port,
            "data_port_range": {"start": data_start, "end": data_end},
            "auth_type": auth_type,
            "tunnel_mode": tunnel_mode,
        },
    }

    # Include service URLs when configured (allows clients to discover
    # the correct MinIO/RAG endpoint instead of guessing from the host).
    minio_url = os.environ.get("UMRS_MINIO_URL") or os.environ.get("MINIO_ENDPOINT")
    rag_url = os.environ.get("RAG_URL")
    if minio_url or rag_url:
        result["services"] = {}
        if minio_url:
            result["services"]["minio_url"] = minio_url
        if rag_url:
            result["services"]["rag_url"] = rag_url

    # Include per-user S3 credentials when caller identity is known.
    if x_user_id:
        s3cfg = _get_or_create_user_s3cfg(x_user_id)
        if s3cfg:
            result["s3cfg_content"] = s3cfg

    return result


def _read_user_s3cfg(username: str) -> str | None:
    """Read existing s3cfg content for *username* from the server filesystem.

    Searches several locations in order (most authoritative first):

    1. ``~/.config/umrs/s3cfg_<username>``  (authoritative — written by
       ``minio_s3_create_user``; always reflects the most recent MinIO
       credentials)
    2. ``<umrs_config_dir>/user_files/s3cfg_<username>``  (mirror copy —
       kept in sync by ``_mirror_s3cfg_to_user_files``; pruned when stale
       by ``_prune_stale_user_files_s3cfg``)
    3. ``<configure_file_dir>/user_files/s3cfg_<username>``  (legacy
       provisioning fallback)
    4. ``<configure_file_dir>/s3cfg_<username>``
    5. ``<umrs_config_dir>/s3cfg_<username>``

    Returns ``None`` if no file is found.
    """
    candidates: list[str] = []

    # Priority 1: Authoritative path (written by minio_s3_create_user on creation,
    # updated on secret rotation).  Always trust this copy over mirrors.
    candidates.append(
        os.path.expanduser(f"~/.config/umrs/s3cfg_{username}"),
    )

    # Priority 2-3: UMRS_CONFIG mirrors
    umrs_config_dir = os.environ.get("UMRS_CONFIG", "")
    if umrs_config_dir:
        umrs_config_dir = os.path.expanduser(umrs_config_dir)
        candidates.append(
            os.path.join(umrs_config_dir, "user_files", f"s3cfg_{username}"),
        )
        candidates.append(
            os.path.join(umrs_config_dir, f"s3cfg_{username}"),
        )

    # Priority 4-5: Legacy provisioning fallback paths
    configure_file = os.environ.get("UMRS_CONFIGURE_FILE", "")
    if configure_file:
        cfg_dir = os.path.dirname(os.path.expanduser(configure_file))
        candidates.append(
            os.path.join(cfg_dir, "user_files", f"s3cfg_{username}"),
        )
        candidates.append(os.path.join(cfg_dir, f"s3cfg_{username}"))

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return f.read()
            except OSError:
                pass

    return None


def _get_or_create_user_s3cfg(username: str) -> str | None:
    """Return the user's s3cfg content, auto-provisioning a MinIO user if needed.

    First tries to read an existing s3cfg from disk (via ``_read_user_s3cfg``).
    If none exists and ``UMRS_ADMIN_S3_CONFIGURATION_FILE`` is set, creates a
    new MinIO IAM user with a per-user list policy and returns the generated
    s3cfg content.  Returns ``None`` on any failure — login still succeeds
    without s3cfg.
    """
    # Try existing file first (fast path)
    existing = _read_user_s3cfg(username)
    if existing:
        return existing

    # Auto-provision: create MinIO user on first login
    admin_s3cfg = os.environ.get("UMRS_ADMIN_S3_CONFIGURATION_FILE", "")
    if not admin_s3cfg:
        # No admin config → cannot auto-provision
        return None

    try:
        from umrs.shared.storage.s3 import connect_to_s3
        from umrs.shared.storage.minio import minio_s3_create_user

        user_s3cfg_path = os.path.expanduser(f"~/.config/umrs/s3cfg_{username}")

        s3_connection, _ = connect_to_s3(admin_s3cfg)
        try:
            _alias, s3cfg_content = minio_s3_create_user(
                username,
                s3_connection,
                user_s3cfg_path,
                admin_s3cfg,
            )
        finally:
            s3_connection.close()

        return s3cfg_content
    except Exception as e:
        # Best-effort: MinIO may be unavailable; login proceeds without s3cfg
        print(f"Warning: auto-provision s3cfg for '{username}' failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Configuration reload via HTTP (any authenticated user)
# ---------------------------------------------------------------------------


class OpsReloadRequest(BaseModel):
    """Optional request body for reload.  When ``user_config`` is provided,
    the server persists it to the user's config file before re-reading."""

    user_config: dict | None = None


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


def _persist_user_config_for_reload(
    user_config: dict | None, x_user_id: str | None
) -> list[str]:
    """Persist uploaded user config to disk before reload.

    Returns a list of human-readable error messages; empty on success.
    """
    persist_errors: list[str] = []
    if not user_config or not x_user_id:
        return persist_errors

    import json as _json
    import logging
    import tempfile
    import traceback

    from umrs.server.file_handling import persist_uploaded_user_config

    log = logging.getLogger(__name__)
    tmp_fd = None
    tmp_path: str | None = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="umrs_user_config_",
            suffix=".json",
        )
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            _json.dump(user_config, f, indent=2, ensure_ascii=False)
        tmp_fd = None

        ok, _stored_user, _stored_path = persist_uploaded_user_config(
            tmp_path,
            _reload_configure_file,
            _reload_user_config_map,
            expected_username=x_user_id,
        )
        if not ok:
            persist_errors.append(
                f"Failed to persist uploaded config for '{x_user_id}'."
            )
    except Exception as exc:
        err_detail = f"{exc}\n{traceback.format_exc()}"
        log.warning("Failed to persist user config for %s: %s", x_user_id, err_detail)
        persist_errors.append(
            f"Failed to persist uploaded config for '{x_user_id}': {exc}"
        )
    finally:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return persist_errors


def _build_reload_response(
    outcome: dict[str, Any],
    x_user_id: str | None,
    persist_errors: list[str],
) -> dict[str, Any]:
    """Build the reload endpoint response from the reload outcome."""
    if persist_errors or "error" in outcome:
        summary = outcome.get("summary", "")
        error_text = "\n".join(persist_errors + [outcome.get("error", "")]).strip()
        return {
            "status": "error",
            "summary": summary,
            "error": error_text or "Reload failed",
        }

    summary = outcome.get("summary", "Reload complete")
    if x_user_id and _reload_buckets_cfg is not None:
        from umrs.shared.pretty_print import pretty_print_cfg_for_user

        summary = pretty_print_cfg_for_user(
            _reload_buckets_cfg,
            x_user_id,
            print_output=False,
        )

    return {"status": "ok", "summary": summary}


@router.post(
    "/reload",
    response_model=OpsReloadResponse,
    summary="Trigger configuration reload",
)
def ops_reload(
    body: OpsReloadRequest = OpsReloadRequest(),
    _auth: None = Depends(require_authenticated),
    x_user_id: str | None = Header(None, alias="X-User-ID"),
) -> dict[str, Any]:
    """Re-read the server configuration from disk and re-provision users.

    This is the HTTP equivalent of the ZMQ ``RELOAD`` command.  The server
    re-reads ``configure_file``, creates any missing MinIO/Postgres
    resources, and updates the live ``buckets_cfg``.

    When ``user_config`` is provided in the request body, the server
    persists it to the user's config file before reloading — this is how
    the client synchronises locally-added experiences to the server.

    Requires authentication.  When ``X-User-ID`` is provided, the
    returned summary is filtered to only show that user's buckets.
    """
    if _reload_configure_file is None or _reload_buckets_cfg is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reload context not initialised — server may still be starting",
        )

    try:
        persist_errors = _persist_user_config_for_reload(body.user_config, x_user_id)

        from umrs.server.maintenance import reload_configuration

        outcome = reload_configuration(
            _reload_configure_file,
            _reload_buckets_cfg,
            _reload_user_config_map,
            log_output=True,
        )

        return _build_reload_response(outcome, x_user_id, persist_errors)
    except Exception as exc:
        import traceback

        detail = f"{exc}\n\n{traceback.format_exc()}"
        logging.getLogger(__name__).exception("Unhandled exception in ops_reload")
        return {
            "status": "error",
            "summary": "Unhandled exception during reload",
            "error": detail,
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
    _auth: None = Depends(require_authenticated),
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
    return {
        "events": events,
        "count": len(events),
        "persistent": _audit_store_available(),
    }


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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required"
        )

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
    _auth: None = Depends(require_authenticated),
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
            owner_id,
            resource_type=resource_type,
            project_id=project_id,
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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required"
        )

    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    updated = store.transfer_owner(
        resource_type,
        resource_id,
        new_owner_id,
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
    return {
        "status": "transferred",
        "resource_type": resource_type,
        "resource_id": resource_id,
    }


@router.get("/governance/quotas")
async def governance_quotas(
    project_id: str = Query(..., description="Project ID"),
    _auth: None = Depends(require_authenticated),
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
    resource_type: str = Query(
        ..., description="Resource type (endpoints, mcp_servers, etc.)"
    ),
    max_count: int = Query(..., ge=0, description="Maximum allowed count"),
    warn_count: int | None = Query(
        None, ge=0, description="Soft warning threshold (optional)"
    ),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """Set quota for a project + resource type (admin-only).

    When warn_count is set, the system emits a governance.quota_warning
    audit event when usage reaches the warning threshold but does not
    block operations until max_count is reached.
    """
    expected = os.environ.get("UMRS_ADMIN_TOKEN", "")
    if not expected or admin_token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required"
        )

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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required"
        )

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
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required"
        )

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
# Resource sharing
# ---------------------------------------------------------------------------


@router.post("/governance/shares")
async def governance_create_share(
    resource_type: str = Query(..., description="Resource type"),
    resource_id: str = Query(..., description="Resource ID"),
    granted_to: str = Query(..., description="User or group ID to share with"),
    granted_to_type: str = Query("user", description="'user' or 'group'"),
    access_level: str = Query("read", description="'read' or 'write'"),
    expires_at: str | None = Query(None, description="ISO 8601 expiry (optional)"),
    user_id: str = Header("", alias="x-user-id"),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """Share a resource with a user or group.

    The caller must be the resource owner or an admin.
    """
    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    if granted_to_type not in ("user", "group"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="granted_to_type must be 'user' or 'group'",
        )
    if access_level not in ("read", "write"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="access_level must be 'read' or 'write'",
        )

    # Authorization: owner or admin
    expected_admin = os.environ.get("UMRS_ADMIN_TOKEN", "")
    is_admin = bool(expected_admin and admin_token == expected_admin)

    if not is_admin:
        owner = store.get_owner(resource_type, resource_id)
        if not owner or owner.get("owner_id") != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the resource owner or admin can create shares",
            )

    share = store.create_share(
        resource_type,
        resource_id,
        granted_to,
        granted_to_type=granted_to_type,
        access_level=access_level,
        granted_by=user_id or "admin",
        expires_at=expires_at,
    )

    from .audit_events import emit_audit

    emit_audit(
        "governance.share_created",
        actor=user_id or "admin",
        details={
            "resource_type": resource_type,
            "resource_id": resource_id,
            "granted_to": granted_to,
            "granted_to_type": granted_to_type,
            "access_level": access_level,
        },
    )

    return share


@router.delete("/governance/shares")
async def governance_revoke_share(
    resource_type: str = Query(..., description="Resource type"),
    resource_id: str = Query(..., description="Resource ID"),
    granted_to: str = Query(..., description="User or group ID to revoke"),
    granted_to_type: str = Query("user", description="'user' or 'group'"),
    user_id: str = Header("", alias="x-user-id"),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """Revoke a specific share. Caller must be resource owner or admin."""
    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    expected_admin = os.environ.get("UMRS_ADMIN_TOKEN", "")
    is_admin = bool(expected_admin and admin_token == expected_admin)

    if not is_admin:
        owner = store.get_owner(resource_type, resource_id)
        if not owner or owner.get("owner_id") != user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the resource owner or admin can revoke shares",
            )

    deleted = store.revoke_share(
        resource_type,
        resource_id,
        granted_to,
        granted_to_type=granted_to_type,
    )

    if deleted:
        from .audit_events import emit_audit

        emit_audit(
            "governance.share_revoked",
            actor=user_id or "admin",
            details={
                "resource_type": resource_type,
                "resource_id": resource_id,
                "granted_to": granted_to,
                "granted_to_type": granted_to_type,
            },
        )

    return {"revoked": deleted}


@router.get("/governance/shares")
async def governance_list_shares(
    resource_type: str = Query("", description="Filter by resource type"),
    resource_id: str = Query("", description="Filter by resource ID"),
    user_id: str = Header("", alias="x-user-id"),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """List shares for a resource or for the current user."""
    from .governance import get_governance_store

    store = get_governance_store()
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Governance store not initialized",
        )

    if resource_type and resource_id:
        # List shares for a specific resource (owner or admin)
        expected_admin = os.environ.get("UMRS_ADMIN_TOKEN", "")
        is_admin = bool(expected_admin and admin_token == expected_admin)

        if not is_admin:
            owner = store.get_owner(resource_type, resource_id)
            if not owner or owner.get("owner_id") != user_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Only the resource owner or admin can list shares",
                )

        shares = store.list_shares_for_resource(resource_type, resource_id)
    else:
        # List shares granted to the current user
        shares = store.list_shares_for_user(user_id)

    return {"shares": shares, "count": len(shares)}


# ---------------------------------------------------------------------------
# Governance analytics
# ---------------------------------------------------------------------------


@router.get("/governance/analytics")
async def governance_analytics(
    project_id: str = Query("", description="Filter by project (empty = all)"),
    since: float = Query(0.0, description="Unix timestamp lower bound"),
    admin_token: str = Header("", alias="x-admin-token"),
):
    """Aggregate governance analytics: usage trends, quota health, event distribution.

    Returns a summary suitable for dashboards and operational monitoring.
    Requires admin authentication.
    """
    expected = os.environ.get("UMRS_ADMIN_TOKEN", "")
    if not expected or admin_token != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required"
        )

    from .governance import get_governance_store
    from .audit_events import query_audit_events

    analytics: dict[str, Any] = {
        "quota_utilization": [],
        "resource_counts": {},
        "event_summary": {},
        "recent_denials": [],
        "recent_warnings": [],
    }

    # Quota utilization across projects
    store = get_governance_store()
    if store is not None:
        cur = store.conn.cursor()
        try:
            if project_id:
                cur.execute(
                    "SELECT project_id, resource_type, max_count, current_count, warn_count "
                    "FROM resource_quotas WHERE project_id = %s ORDER BY resource_type",
                    (project_id,),
                )
            else:
                cur.execute(
                    "SELECT project_id, resource_type, max_count, current_count, warn_count "
                    "FROM resource_quotas ORDER BY project_id, resource_type"
                )
            for row in cur.fetchall():
                pct = (row[3] / row[2] * 100) if row[2] > 0 else 0
                analytics["quota_utilization"].append(
                    {
                        "project_id": row[0],
                        "resource_type": row[1],
                        "max_count": row[2],
                        "current_count": row[3],
                        "warn_count": row[4],
                        "utilization_pct": round(pct, 1),
                    }
                )
        except Exception:
            store.conn.rollback()
        finally:
            cur.close()

        # Resource counts by type
        cur2 = store.conn.cursor()
        try:
            if project_id:
                cur2.execute(
                    "SELECT resource_type, COUNT(*) FROM resource_ownership "
                    "WHERE project_id = %s GROUP BY resource_type ORDER BY resource_type",
                    (project_id,),
                )
            else:
                cur2.execute(
                    "SELECT resource_type, COUNT(*) FROM resource_ownership "
                    "GROUP BY resource_type ORDER BY resource_type"
                )
            analytics["resource_counts"] = {row[0]: row[1] for row in cur2.fetchall()}
        except Exception:
            store.conn.rollback()
        finally:
            cur2.close()

    # Governance event distribution from audit trail
    gov_events = query_audit_events(
        event="governance.",
        since=since,
        limit=500,
    )
    if project_id:
        gov_events = [e for e in gov_events if e.get("project_id") == project_id]

    event_counts: dict[str, int] = {}
    for ev in gov_events:
        name = ev.get("event", "unknown")
        event_counts[name] = event_counts.get(name, 0) + 1
    analytics["event_summary"] = event_counts

    # Recent denials and warnings
    analytics["recent_denials"] = [
        e for e in gov_events if e.get("outcome") == "denied"
    ][:20]
    analytics["recent_warnings"] = [
        e for e in gov_events if e.get("event") == "governance.quota_warning"
    ][:20]

    return analytics


# ---------------------------------------------------------------------------
# Schema version introspection
# ---------------------------------------------------------------------------


@router.get("/schema/versions")
async def schema_versions(
    _auth: None = Depends(require_authenticated),
):
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
