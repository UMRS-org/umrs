"""Frontend-venue session runtime routes (wave 2026-09-21, ADR-046 A2).

Owner routes (gateway-verified identity, same ``x-user-id`` contract as the
sibling companion-session routers):

    POST /umrs/companion/sessions/{sid}/venue/start   provision the sandbox
    GET  /umrs/companion/sessions/{sid}/venue         runtime status (+lease)
    POST /umrs/companion/sessions/{sid}/venue/stop    teardown + revoke

Worker rail (lease-token principal — see ``_rail_identity``):

    GET  /umrs/companion/sessions/{sid}/rail/snapshot
    POST /umrs/companion/sessions/{sid}/rail/snapshot
    POST /umrs/companion/sessions/{sid}/rail/queue/pop
    POST /umrs/companion/sessions/{sid}/rail/heartbeat
    POST /umrs/companion/sessions/{sid}/rail/approvals
    GET  /umrs/companion/sessions/{sid}/rail/approvals/{approval_id}

The rail reuses the exact row helpers of the owner routes with the lease
owner as the principal, so queue/snapshot semantics never fork.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, Request
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


def get_venue_service(request: Request) -> FrontendVenueService:
    """Process-cached service (tests inject ``app.state.companion_venue_service``)."""
    cached = getattr(request.app.state, "companion_venue_service", None)
    if cached is not None:
        return cached
    from umrs.venues.frontend_config import FrontendVenueConfig
    from umrs.venues.frontend_podman import FrontendPodmanProvisioner
    from umrs.venues.venue_sessions import VenueSessionStore

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
    )
    request.app.state.companion_venue_service = service
    return service


def _typed(exc: VenueError) -> None:
    raise_typed(
        PlatformError.companionvenue(exc.code, exc.message, **exc.detail), cause=exc
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


@router.post("/{session_id}/venue/start", response_model=dict[str, Any])
async def start_venue(
    session_id: str, body: VenueStartBody, request: Request
) -> dict[str, Any]:
    """Provision the sandbox runtime for a mirrored session (design F-2).
    # auth: verified
    """
    uid = _uid(request)
    _pool(request)
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
