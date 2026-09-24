"""Frontend-venue session runtime routes (wave 2026-09-21, ADR-046 A2).

Owner routes (gateway-verified identity, same ``x-user-id`` contract as the
sibling companion-session routers):

    POST /umrs/companion/sessions/{sid}/venue/start   submit the session job
    GET  /umrs/companion/sessions/{sid}/venue         runtime status (+lease)
    POST /umrs/companion/sessions/{sid}/venue/stop    teardown + revoke

Worker rail (lease-token principal — see ``_rail_identity``):

    GET  /umrs/companion/sessions/{sid}/rail/snapshot
    POST /umrs/companion/sessions/{sid}/rail/snapshot
    POST /umrs/companion/sessions/{sid}/rail/queue/pop
    POST /umrs/companion/sessions/{sid}/rail/heartbeat
    POST /umrs/companion/sessions/{sid}/rail/approvals
    GET  /umrs/companion/sessions/{sid}/rail/approvals/{approval_id}
    ANY  /umrs/companion/sessions/{sid}/rail/gw/{workflows|jobs}/…   (rail gw)

Rail gw: the sandbox's platform verbs (the companion's workflow/job tools)
are re-dispatched in-process to ``/umrs/<family>/…`` with the lease owner's
identity minted the way the gateway sidecar mints it (HMAC over the
identity tuple), so policy, quota and typed errors are the user path's own.
Only the allowlisted families pass; the sandbox never carries a JWT.

The rail reuses the exact row helpers of the owner routes with the lease
owner as the principal, so queue/snapshot semantics never fork.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from umrs.venues.frontend_service import FrontendVenueService, VenueError

from ._rail_identity import (
    RailAuthError,
    RailPrincipal,
    bearer_token,
    resolve_rail_principal,
)
from .companion_sessions import (
    SessionSnapshotBody,
    _pool,
    _uid,
    queue_mutate_row,
    snapshot_detail_row,
    upsert_snapshot_row,
)
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/companion/sessions", tags=["companion-sessions"])


# ── service wiring ─────────────────────────────────────────────────────────


def _site_name() -> str:
    """The claiming site for session jobs (the store's own site, else config)."""
    try:
        from umrs.orchestrator.site_config import load_site_config

        return str(load_site_config().name or "")
    except Exception:  # unresolved → the service refuses to submit
        logger.warning("site name unresolved for frontend_session jobs", exc_info=True)
        return ""


def _quota_check(pool: Any) -> Any:
    from .agent_quota import enforce_spawn_budget

    def check(user_id: str) -> None:
        enforce_spawn_budget(pool, user_id=user_id)

    return check


def get_venue_service(request: Request) -> FrontendVenueService:
    """Process-cached service (tests inject ``app.state.companion_venue_service``)."""
    cached = getattr(request.app.state, "companion_venue_service", None)
    if cached is not None:
        return cached
    from umrs.venues.frontend_config import FrontendVenueConfig
    from umrs.venues.frontend_podman import FrontendPodmanProvisioner
    from umrs.venues.venue_sessions import VenueSessionStore

    from .jobs import get_optional_job_store
    from .secret_broker import get_broker

    config = FrontendVenueConfig.from_env()
    pool = getattr(request.app.state, "umrs_db_pool", None)
    broker = get_broker()
    service = FrontendVenueService(
        config=config,
        broker=broker,
        store=VenueSessionStore(pool),
        provisioner=FrontendPodmanProvisioner(broker, session_root=config.session_root),
        pool=pool,
        job_store=get_optional_job_store(),
        site_name=_site_name(),
        quota_check=_quota_check(pool),
    )
    request.app.state.companion_venue_service = service
    return service


def _typed(exc: VenueError) -> None:
    raise_typed(
        PlatformError.companionvenue(exc.code, exc.message, **exc.detail), cause=exc
    )


def _unexpected(exc: Exception, verb: str) -> None:
    """An unhandled failure inside the venue service answers a typed 503 naming
    the exception (never a bare "Internal Server Error": the live receipt of
    2026-09-21 had nothing to read); the traceback goes to the server log."""
    logger.exception("companion venue %s failed", verb)
    raise_typed(
        PlatformError.companionvenue(
            "unavailable", f"venue {verb} failed: {type(exc).__name__}: {exc}"
        ),
        cause=exc,
    )


# ── bodies ─────────────────────────────────────────────────────────────────


class VenueStartBody(BaseModel):
    source_refs: list[str] = Field(default_factory=list)
    default_source: str = ""
    web_policy: str = ""
    web_allow_domains: list[str] = Field(default_factory=list)
    project: str = ""


class HeartbeatBody(BaseModel):
    phase: str = "idle"
    turn_seq: int = 0
    partial: str = ""
    detail: str = ""
    extend_seconds: int = 0


class RailApprovalBody(BaseModel):
    turn_id: str = ""
    action: dict[str, Any] = Field(default_factory=dict)
    approval_id: str | None = None


# ── owner routes ───────────────────────────────────────────────────────────


def _refuse_cpu_venue_on_frontend_rail(
    request: Request, uid: str, session_id: str
) -> None:
    """D-48 (2026-09-22): sessions whose registry venue is cpu:<host> may
    not start on the frontend rail — the cpu worker rail is the ROADMAP
    3.2 leg; fail honest, never a silent frontend placement."""
    from .companion_sessions import snapshot_detail_row

    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return
    conn = pool.getconn()
    try:
        row = snapshot_detail_row(conn, uid, session_id)
    except Exception:  # noqa: BLE001 — venue lookup must not block a start
        return
    finally:
        pool.putconn(conn)
    if row and str(row.get("venue") or "").startswith("cpu:"):
        raise_typed(
            PlatformError.companionvenue(
                "bad_request",
                f"session venue {row['venue']!r} may not start on the frontend rail "
                "(cpu worker rail is the ROADMAP 3.2 leg)",
            )
        )


@router.post("/{session_id}/venue/start", response_model=dict[str, Any])
async def start_venue(
    session_id: str, body: VenueStartBody, request: Request
) -> dict[str, Any]:
    """Submit the session's ``frontend_session`` job; returns the ``provisioning``
    row (poll GET …/venue until running|failed). Design F-2, ADR-046 A2.
    # auth: verified
    """
    uid = _uid(request)
    _pool(request)
    _refuse_cpu_venue_on_frontend_rail(request, uid, session_id)
    service = get_venue_service(request)
    try:
        return await asyncio.to_thread(
            service.start,
            owner=uid,
            session_id=session_id,
            source_refs=body.source_refs,
            default_source=body.default_source,
            web_policy=body.web_policy,
            web_allow_domains=body.web_allow_domains,
            project=body.project,
        )
    except VenueError as exc:
        _typed(exc)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — unexpected → typed 503 naming the cause
        _unexpected(exc, "start")


@router.get("/{session_id}/venue", response_model=dict[str, Any])
async def venue_status(session_id: str, request: Request) -> dict[str, Any]:
    """Runtime row + lease projection for the owner (404 otherwise).
    # auth: verified
    """
    uid = _uid(request)
    service = get_venue_service(request)
    try:
        status = service.status(owner=uid, session_id=session_id)
    except VenueError as exc:
        _typed(exc)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — unexpected → typed 503 naming the cause
        _unexpected(exc, "status")
    if status is None:
        raise_typed(
            PlatformError.companionvenue(
                "not_found", "no venue runtime for this session"
            )
        )
    return status


@router.post("/{session_id}/venue/stop", response_model=dict[str, Any])
async def stop_venue(session_id: str, request: Request) -> dict[str, Any]:
    """Tear the sandbox down and revoke its lease (idempotent).
    # auth: verified
    """
    uid = _uid(request)
    service = get_venue_service(request)
    try:
        return await asyncio.to_thread(service.stop, owner=uid, session_id=session_id)
    except VenueError as exc:
        _typed(exc)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — unexpected → typed 503 naming the cause
        _unexpected(exc, "stop")


@router.post("/{session_id}/venue/pause", response_model=dict[str, Any])
async def pause_venue(session_id: str, request: Request) -> dict[str, Any]:
    """Pause the sandbox worker between turns (D-09 verb; idempotent).
    # auth: verified
    """
    uid = _uid(request)
    service = get_venue_service(request)
    try:
        return await asyncio.to_thread(service.pause, owner=uid, session_id=session_id)
    except VenueError as exc:
        _typed(exc)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — unexpected → typed 503 naming the cause
        _unexpected(exc, "pause")


@router.post("/{session_id}/venue/resume", response_model=dict[str, Any])
async def resume_venue(session_id: str, request: Request) -> dict[str, Any]:
    """Resume a paused sandbox worker (D-09 verb; only from paused).
    # auth: verified
    """
    uid = _uid(request)
    service = get_venue_service(request)
    try:
        return await asyncio.to_thread(service.resume, owner=uid, session_id=session_id)
    except VenueError as exc:
        _typed(exc)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — unexpected → typed 503 naming the cause
        _unexpected(exc, "resume")


# ── worker rail ────────────────────────────────────────────────────────────


def rail_principal(request: Request, session_id: str) -> RailPrincipal:
    """FastAPI dependency: lease-token principal or a named 401."""
    service = get_venue_service(request)
    client_ip = request.client.host if request.client else ""
    try:
        return resolve_rail_principal(
            token=bearer_token(request.headers),
            session_id=session_id,
            client_ip=client_ip,
            broker=service.broker,
            ip_check=service.config.rail_ip_check,
        )
    except RailAuthError as exc:
        raise_typed(PlatformError.companionvenue("unauthorized", exc.reason), cause=exc)


@router.get("/{session_id}/rail/snapshot", response_model=dict[str, Any])
async def rail_get_snapshot(
    session_id: str,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """Worker bootstrap: the owner's mirrored snapshot for this session.
    # auth: lease-token
    """
    pool = _pool(request)
    conn = pool.getconn()
    try:
        detail = snapshot_detail_row(conn, principal.owner, session_id)
    finally:
        pool.putconn(conn)
    if detail is None:
        raise_typed(PlatformError.companionvenue("not_found", "snapshot not found"))
    return detail


@router.post("/{session_id}/rail/snapshot", response_model=dict[str, Any])
async def rail_put_snapshot(
    session_id: str,
    body: SessionSnapshotBody,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """Worker mirror (LWW). Venue is pinned to ``frontend`` — never body-chosen.
    # auth: lease-token
    """
    pool = _pool(request)
    conn = pool.getconn()
    try:
        applied = upsert_snapshot_row(
            conn,
            principal.owner,
            session_id,
            name=body.name,
            payload=body.payload,
            updated_at=body.updated_at,
            venue="frontend",
            state=body.state,
        )
    finally:
        pool.putconn(conn)
    return {"status": "ok", "applied": applied}


@router.post("/{session_id}/rail/queue/pop", response_model=dict[str, Any])
async def rail_queue_pop(
    session_id: str,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """Pop the next queued turn (F9 owner-scope queue) for the worker.
    # auth: lease-token
    """
    pool = _pool(request)
    conn = pool.getconn()
    try:
        return queue_mutate_row(
            conn, principal.owner, session_id, action="pop", client_id="venue-worker"
        )
    finally:
        pool.putconn(conn)


@router.post("/{session_id}/rail/heartbeat", response_model=dict[str, Any])
async def rail_heartbeat(
    session_id: str,
    body: HeartbeatBody,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """Renew the lease + record worker runtime (phase/turn_seq/partial).
    # auth: lease-token
    """
    service = get_venue_service(request)
    return service.heartbeat(
        session_id=session_id,
        lease_id=principal.lease_id,
        phase=body.phase,
        turn_seq=body.turn_seq,
        partial=body.partial,
        detail=body.detail,
        extend_seconds=body.extend_seconds,
    )


# ── rail gw: platform verbs for the sandbox (workflows, jobs) ───────────────

# D-42 (2026-09-22): remote sessions reach knowledge/catalog/rag READ-ONLY —
# a sandbox reads and retrieves but knowledge writes stay on the lease
# owner's own surfaces (proposals ride the durable approval inbox, not the
# rail). workflows/jobs keep the pre-existing write verbs for the lease rail.
RAIL_GW_FAMILIES: frozenset[str] = frozenset(
    {"workflows", "jobs", "knowledge", "catalog", "scientific-catalogs", "rag"}
)
RAIL_GW_READ_ONLY_FAMILIES: frozenset[str] = frozenset(
    {"knowledge", "catalog", "scientific-catalogs", "rag"}
)
RAIL_GW_METHODS: tuple[str, ...] = ("GET", "POST", "DELETE")
_RAIL_GW_DROP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "authorization",
        "cookie",
        "x-user-id",
        "x-project-id",
        "x-verified-project-id",
        "x-verified-is-admin",
        "x-verified-hmac",
        "x-verified-project-memberships",
        "x-api-key",
    }
)


def _lease_project(principal: RailPrincipal) -> str:
    return str(getattr(principal.lease, "project", "") or "").strip()


def _snapshot_detail(request: Request, principal: RailPrincipal) -> dict[str, Any]:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return {}
    conn = pool.getconn()
    try:
        return snapshot_detail_row(conn, principal.owner, principal.session_id) or {}
    finally:
        pool.putconn(conn)


def _snapshot_project(detail: dict[str, Any]) -> str:
    try:
        payload = json.loads(detail.get("payload") or "{}")
        metadata = payload.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        config = (metadata or {}).get("config") or {}
        return str(config.get("project_id") or "").strip()
    except (TypeError, ValueError, AttributeError):
        return ""


def rail_project_for(principal: RailPrincipal, request: Request) -> str:
    """Project the sandbox acts in: the lease's project (session config at
    start), else the snapshot's recorded project, else the deployment default."""
    return (
        _lease_project(principal)
        or _snapshot_project(_snapshot_detail(request, principal))
        or os.environ.get("JWT_DEFAULT_PROJECT", "default").strip()
        or "default"
    )


def mint_rail_identity(owner: str, project: str) -> dict[str, str]:
    """Gateway-equivalent verified identity headers for the lease owner."""
    from r4_contracts.handshake_hmac import sign_claims

    from umrs.shared.handshake import handshake_secret

    headers = {
        "x-user-id": owner,
        "x-project-id": project,
        "x-verified-project-id": project,
        "x-verified-is-admin": "false",
    }
    secret = handshake_secret()
    if secret:
        headers["x-verified-hmac"] = sign_claims(secret, owner, project, False)
    return headers


def _forwardable_headers(request: Request) -> dict[str, str]:
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _RAIL_GW_DROP_HEADERS
        and not k.lower().startswith("x-verified")
    }


def _created_job_id(store: Any, family: str, payload: Any) -> str:
    """The durable job a successful platform POST created, if the body names one.

    ``/umrs/jobs`` answers ``job_id``; ``/umrs/workflows/…/run`` answers an
    ``execution_id`` whose job carries it in ``spec.execution_id``.
    """
    if not isinstance(payload, dict):
        return ""
    job_id = str(payload.get("job_id") or "")
    if job_id or family != "workflows":
        return job_id
    execution_id = str(payload.get("execution_id") or "")
    if not execution_id:
        return ""
    from umrs.jobs.models import JobType

    return (
        store.find_job_id_by_spec_value(
            JobType.WORKFLOW_RUN, "execution_id", execution_id
        )
        or ""
    )


def stamp_rail_origin(store: Any, session_id: str, family: str, upstream: Any) -> str:
    """Stamp ``spec.origin = {session_id, venue: frontend}`` on a job the sandbox
    just created through the rail (lineage only; no cascade, ADR-046 A2).
    Returns the stamped job id ("" when nothing applied)."""
    if store is None:
        return ""
    try:
        job_id = _created_job_id(store, family, upstream.json())
        if job_id and store.stamp_spec_origin(
            job_id, {"session_id": session_id, "venue": "frontend"}
        ):
            return job_id
    except Exception:  # lineage is best-effort, never the response
        logger.warning("rail gw origin stamp failed for %s", session_id, exc_info=True)
    return ""


def _refuse_rail_family(family: str, method: str) -> None:
    """Allowlisted families only; read-only families refuse writes (D-42)."""
    if family not in RAIL_GW_FAMILIES:
        raise_typed(
            PlatformError.companionvenue(
                "forbidden", f"rail gw: /umrs/{family} is not reachable from a sandbox"
            )
        )
    if method != "GET" and family in RAIL_GW_READ_ONLY_FAMILIES:
        raise_typed(
            PlatformError.companionvenue(
                "forbidden",
                f"rail gw: /umrs/{family} is read-only from a sandbox (D-42); "
                "knowledge writes ride the session's approval rail",
            )
        )


def _run_rail_gw_hook(
    request: Request, principal: Any, path: str, upstream: Any
) -> None:
    hook = getattr(request.app.state, "companion_venue_rail_gw_hook", None)
    if hook is None:
        return
    try:
        hook(principal, request.method, path, upstream)
    except Exception:
        logger.warning("rail gw hook failed", exc_info=True)


@router.api_route("/{session_id}/rail/gw/{path:path}", methods=list(RAIL_GW_METHODS))
async def rail_gateway(
    session_id: str,
    path: str,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> Response:
    """Re-dispatch an allowlisted platform verb with the lease owner's identity.
    # auth: lease-token
    """
    family = path.split("/", 1)[0]
    _refuse_rail_family(family, request.method)
    headers = _forwardable_headers(request)
    headers.update(
        mint_rail_identity(principal.owner, rail_project_for(principal, request))
    )
    body = await request.body()
    transport = httpx.ASGITransport(app=request.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://rail.internal"
    ) as client:
        upstream = await client.request(
            request.method,
            f"/umrs/{path}",
            params=request.query_params,
            content=body,
            headers=headers,
            timeout=httpx.Timeout(120.0),
        )
    logger.info(
        "rail_gw sid=%s owner=%s %s /umrs/%s -> %s",
        session_id,
        principal.owner,
        request.method,
        path,
        upstream.status_code,
    )
    _run_rail_gw_hook(request, principal, path, upstream)
    if request.method == "POST" and upstream.status_code < 300:
        stamp_rail_origin(
            get_venue_service(request).job_store, session_id, family, upstream
        )
    passthrough = {
        k: v
        for k, v in upstream.headers.items()
        if k.lower() in {"content-type", "x-platform-error-code", "x-correlation-id"}
    }
    return Response(
        content=upstream.content, status_code=upstream.status_code, headers=passthrough
    )


def _approvals_store(request: Request, conn: Any):
    """T4 inbox on the borrowed connection (tests inject a factory on app.state)."""
    factory = getattr(request.app.state, "companion_venue_approvals_factory", None)
    if factory is not None:
        return factory(conn)
    from umrs.companion_trust.chat_turn_approvals import ChatTurnApprovalsStore

    return ChatTurnApprovalsStore(conn)


@router.post("/{session_id}/rail/approvals", response_model=dict[str, Any])
async def rail_request_approval(
    session_id: str,
    body: RailApprovalBody,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """Worker raises a durable tool-grain intent (T4 inbox) for the owner.
    # auth: lease-token
    """
    pool = _pool(request)
    conn = pool.getconn()
    try:
        record = _approvals_store(request, conn).request(
            session_id,
            body.turn_id,
            body.action,
            approval_id=body.approval_id,
            user_id=principal.owner,
        )
        return {"session_id": session_id, "approval": record}
    finally:
        pool.putconn(conn)


@router.get("/{session_id}/rail/approvals/{approval_id}", response_model=dict[str, Any])
async def rail_get_approval(
    session_id: str,
    approval_id: str,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """Worker polls one intent's decision (session-scoped; 404 across sessions).
    # auth: lease-token
    """
    pool = _pool(request)
    conn = pool.getconn()
    try:
        record = _approvals_store(request, conn).get(approval_id)
    finally:
        pool.putconn(conn)
    if record is None or record.get("session_id") != session_id:
        raise_typed(PlatformError.companionvenue("not_found", "approval not found"))
    return {"session_id": session_id, "approval": record}
