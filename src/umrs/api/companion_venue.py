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
import hashlib
import logging
import os
import re
from typing import Any
from urllib.parse import unquote

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from umrs.venues.frontend_service import FrontendVenueService, VenueError

from ._rail_identity import (
    RAIL_ATTESTATION_HEADER,
    RailAuthError,
    RailPrincipal,
    attest,
    bearer_token,
    resolve_rail_principal,
    revoke_attestation,
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
    """The start's quota check (ADR-099 D-6): the live remote-session cap of
    the start's resolved policy. ``frontend_session`` left the ADR-051
    agent-process budget, which the other spawn points keep enforcing."""
    from umrs.quotas.session_cap import check_concurrent_sessions

    return lambda check: check_concurrent_sessions(pool, check)


def get_venue_service(request: Request) -> FrontendVenueService:
    """Process-cached service (tests inject ``app.state.companion_venue_service``)."""
    cached = getattr(request.app.state, "companion_venue_service", None)
    if cached is not None:
        return cached
    service = build_venue_service(getattr(request.app.state, "umrs_db_pool", None))
    request.app.state.companion_venue_service = service
    return service


def build_venue_service(pool: Any) -> FrontendVenueService:
    """The venue service over ``pool`` (the routes' and, slice 8b, the plan
    loop's: one construction, never a second configuration)."""
    from umrs.venues.frontend_config import FrontendVenueConfig
    from umrs.venues.frontend_podman import FrontendPodmanProvisioner
    from umrs.venues.venue_sessions import VenueSessionStore

    from .jobs import get_optional_job_store
    from .secret_broker import get_broker

    config = FrontendVenueConfig.from_env()
    broker = get_broker()
    return FrontendVenueService(
        config=config,
        broker=broker,
        store=VenueSessionStore(pool),
        provisioner=FrontendPodmanProvisioner(broker, session_root=config.session_root),
        pool=pool,
        job_store=get_optional_job_store(),
        site_name=_site_name(),
        quota_check=_quota_check(pool),
    )


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
    # Live tool-activity lines the owner watches next to the partial text
    # (r4 venue worker, 2026-09-27); [] when the worker pre-dates the field.
    tool_activity: list[str] = Field(default_factory=list)


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


def _checked_start_project(request: Request, requested: str) -> str:
    """The session's project: one the verified caller may write (the start's
    project becomes the job's and the lease's, hence the rail identity and the
    ADR-062 data-class gate; it was honoured unchecked, H-1 2026-10-17)."""
    from umrs.knowledge.access_scope import writable_project_for

    from ._scope_identity import request_caller_scope

    project = writable_project_for(requested, request_caller_scope(request))
    if project is None:
        raise_typed(
            PlatformError.companionvenue(
                "forbidden",
                f"no write access to project {requested!r} for this session",
            )
        )
    return project


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
    project = _checked_start_project(request, body.project)
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
            project=project,
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
        tool_activity=body.tool_activity,
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
        RAIL_ATTESTATION_HEADER,
    }
)
# ADR-092 slice 8: a plan-run worker reaches the traces family for its own
# run only, and only the record verbs (the run, its attempts, its records,
# finishing an attempt); never another run, never start, resume or cancel.
_PLAN_RUN_TAILS = re.compile(
    r"^(|/attempts|/records|/attempts/[A-Za-z0-9_.:-]+/finish)$"
)


def _lease_project(principal: RailPrincipal) -> str:
    return str(getattr(principal.lease, "project", "") or "").strip()


def rail_project_for(principal: RailPrincipal, request: Request) -> str:
    """Project the sandbox acts in: the lease's project, checked when the
    lease was issued, else the deployment default.

    The session snapshot's recorded project was a fallback, but the client
    writes that snapshot, so it could name any project and the rail minted
    signed identity for it (H-1, 2026-10-17). Leases now always carry the
    checked project; the default only covers leases issued before the fix.
    """
    del request  # kept for the route's call shape
    return (
        _lease_project(principal)
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


def _fully_decoded(path: str) -> str:
    """``path`` percent-decoded until it stops changing (at most three times)."""
    for _ in range(3):
        decoded = unquote(path)
        if decoded == path:
            break
        path = decoded
    return path


def _rail_path(path: str) -> str:
    """The upstream path, refused when a segment could leave its family.

    The route sees the path already decoded (``%2e%2e`` is ``..``), and
    httpx resolves ``..`` in ``/umrs/{path}``: ``rag/../secrets/probe``
    passed the family check as ``rag`` and reached ``/umrs/secrets/probe``
    as the lease owner, and ``jobs/../knowledge/...`` wrote to a read-only
    family (D-42). No dot segment, however encoded, reaches upstream.
    """
    segments = _fully_decoded(path).replace("\\", "/").split("/")
    if any(segment in (".", "..") for segment in segments):
        raise_typed(
            PlatformError.companionvenue(
                "forbidden", "rail gw: a path may not contain '.' or '..' segments"
            )
        )
    return path


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


def _own_plan_run_path(request: Request, principal: RailPrincipal, path: str) -> None:
    """ADR-092 slice 8: the traces family is a plan-run worker's own run's
    record verbs, nothing else. The run is the one the session's job was
    started for (the record UMRS wrote), never one the sandbox names."""
    run_id, _job_id = get_venue_service(request).plan_run_binding(principal.session_id)
    prefix = f"traces/plan-runs/{run_id}"
    if (
        not run_id
        or not path.startswith(prefix)
        or not _PLAN_RUN_TAILS.match(path[len(prefix) :])
    ):
        raise_typed(
            PlatformError.companionvenue(
                "forbidden",
                "rail gw: /umrs/traces is reachable from a sandbox only for the record of "
                "the plan run its session hosts (ADR-092 slice 8)",
            )
        )


def _rail_family_scope(
    request: Request, principal: RailPrincipal, family: str, path: str
) -> bool:
    """Refuse what the sandbox may not reach; True = attest the call (slice 8)."""
    if family == "traces":
        _own_plan_run_path(request, principal, path)
        return True
    _refuse_rail_family(family, request.method)
    return False


async def _dispatch(
    request: Request, path: str, headers: dict[str, str], attest_session: str
) -> Any:
    """The in-process re-dispatch; an attestation lives only for this call."""
    handle = attest(request.app.state, attest_session) if attest_session else ""
    if handle:
        headers[RAIL_ATTESTATION_HEADER] = handle
    body = await request.body()
    transport = httpx.ASGITransport(app=request.app)
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://rail.internal"
        ) as client:
            return await client.request(
                request.method,
                f"/umrs/{path}",
                params=request.query_params,
                content=body,
                headers=headers,
                timeout=httpx.Timeout(120.0),
            )
    finally:
        if handle:
            revoke_attestation(request.app.state, handle)


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
    path = _rail_path(path)
    family = path.split("/", 1)[0]
    attested = _rail_family_scope(request, principal, family, path)
    headers = _forwardable_headers(request)
    headers.update(
        mint_rail_identity(principal.owner, rail_project_for(principal, request))
    )
    upstream = await _dispatch(
        request, path, headers, principal.session_id if attested else ""
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
    if request.method == "POST" and upstream.status_code < 300 and not attested:
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


# ── plan gates (ADR-092 slice 8) ───────────────────────────────────────────
#
# A gate a server-hosted plan run raises goes to the UMRS durable approvals
# inbox (ADR-066, ``companion_approval_inbox``): the worker raises the
# intent and polls it; the run's owner answers it from any client through
# the inbox's own decide route. The rail has no decide verb: a worker never
# answers its own gate.

PLAN_GATE_KIND = "plan_gate"
PLAN_GATE_LOOP_PREFIX = "plan-run:"
_MAX_GATE_TEXT = 2000


class RailPlanGateBody(BaseModel):
    gate: str = Field(min_length=1, max_length=200)
    trigger_class: str = Field(default="", max_length=64)
    prompt: str = ""


def _bound_run_job(request: Request, principal: RailPrincipal) -> tuple[str, Any]:
    """``(run_id, job)`` of the plan run this session hosts; 403 otherwise."""
    service = get_venue_service(request)
    run_id, job_id = service.plan_run_binding(principal.session_id)
    if not run_id or not job_id or service.job_store is None:
        raise_typed(
            PlatformError.companionvenue(
                "forbidden",
                "plan gates: this session hosts no plan run (ADR-092 slice 8)",
            )
        )
    try:
        return run_id, service.job_store.get_job(job_id)
    except Exception as exc:  # noqa: BLE001 — a missing record is named
        raise_typed(
            PlatformError.companionvenue(
                "not_found", f"plan run record {job_id} not found"
            ),
            cause=exc,
        )


def plan_gate_approval_id(job_id: str, attempt_id: str, gate: str) -> str:
    """One intent per (run record, attempt, gate): a retried request finds it."""
    digest = hashlib.sha256(f"{job_id}|{attempt_id}|{gate}".encode()).hexdigest()
    return f"pgate-{digest[:20]}"


def _inbox() -> Any:
    from umrs.knowledge import tool_approvals

    return tool_approvals


def _gate_view(record: dict[str, Any]) -> dict[str, Any]:
    view = {
        key: record.get(key)
        for key in ("loop_id", "status", "decision", "decided_by", "decided_at")
    }
    view["approval_id"] = record.get("id") or record.get("approval_id")
    return view


@router.post("/{session_id}/rail/plan-gates", response_model=dict[str, Any])
async def rail_request_plan_gate(
    session_id: str,
    body: RailPlanGateBody,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """The worker raises a plan gate as an ADR-066 inbox intent for the owner.
    # auth: lease-token
    """
    run_id, job = _bound_run_job(request, principal)
    attempt_id = str(getattr(job, "active_attempt_id", "") or "")
    if not attempt_id:
        raise_typed(
            PlatformError.companionvenue(
                "conflict", f"plan gates: run {run_id} has no active attempt"
            )
        )
    action = {
        "kind": PLAN_GATE_KIND,
        "gate": body.gate,
        "class": body.trigger_class,
        "prompt": body.prompt[:_MAX_GATE_TEXT],
        "run_id": run_id,
        "untrusted": True,
    }
    record = _inbox().request_approval(
        f"{PLAN_GATE_LOOP_PREFIX}{run_id}",
        0,
        action,
        approval_id=plan_gate_approval_id(job.job_id, attempt_id, body.gate),
        user_id=principal.owner,
        job_id=job.job_id,
        attempt_id=attempt_id,
    )
    return {"session_id": session_id, "run_id": run_id, "approval": _gate_view(record)}


@router.get(
    "/{session_id}/rail/plan-gates/{approval_id}", response_model=dict[str, Any]
)
async def rail_get_plan_gate(
    session_id: str,
    approval_id: str,
    request: Request,
    principal: RailPrincipal = Depends(rail_principal),
) -> dict[str, Any]:
    """The worker polls its gate's decision (its own run's gates only; 404 else).
    # auth: lease-token
    """
    run_id, job = _bound_run_job(request, principal)
    record = _inbox().get(approval_id)
    if (
        record is None
        or record.get("job_id") != job.job_id
        or record.get("loop_id") != f"{PLAN_GATE_LOOP_PREFIX}{run_id}"
    ):
        raise_typed(PlatformError.companionvenue("not_found", "plan gate not found"))
    return {"session_id": session_id, "run_id": run_id, "approval": _gate_view(record)}
