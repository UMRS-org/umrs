"""Source grant routes (R3-1, ADR-046 R3 + ADR-041 lineage).

    POST /umrs/user-sources/grants            create (owner only, target = own source)
    GET  /umrs/user-sources/grants/mine       owner's grants on their sources
    GET  /umrs/user-sources/grants/for-me     live+unexpired grants held by the caller
    POST /umrs/user-sources/grants/{id}/revoke  owner revoke (tombstone, never delete)

Rules: verified-identity envelope (G-6 discipline); source must be the
caller's sealed row (never grant someone else's source); grantee must
differ from the owner; audit rows land in ``user_source_audit`` visible to
both sides (owner via owner-claim; grantee via the for-me list).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from umrs.grants.source_store import SourceGrantStore
from umrs.knowledge.access_scope import CallerScope

from ._scope_identity import get_caller_scope
from .circles import CircleNotFound, CirclesStore
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/umrs/user-sources/grants",
    tags=["source-grants"],
    dependencies=[Depends(get_caller_scope)],
)


def _store(request: Request) -> SourceGrantStore:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.sourcegrants(
                "unavailable", "source grants require the shared umrs_db pool"
            )
        )
    return SourceGrantStore(pool)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sealed_meta(request: Request, name: str, owner: str):
    from umrs.security.sealed_sources import SealedSourceStore

    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.sourcegrants("unavailable", "shared pool required"))
    meta = SealedSourceStore(pool).get_meta(owner, name)
    if meta is None:
        raise_typed(
            PlatformError.sourcegrants(
                "not_found", f"Source {name!r} not found for the caller"
            )
        )
    return meta


def _validate_circle_grantee(request: Request, grantee: str) -> None:
    """ADR-086: circle grantees must resolve to a live circle — dangling
    'circle:' values must not accrete (404-shaped, never invented)."""
    ref = grantee[len("circle:") :]
    project, sep, circle = ref.partition("/")
    if not sep or not project or not circle:
        raise_typed(
            PlatformError.sourcegrants(
                "bad_request",
                "circle grantee must be 'circle:<project_id>/<circle_id>'",
            )
        )
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.sourcegrants(
                "unavailable", "circle grantees require the shared umrs_db pool"
            )
        )
    conn = pool.getconn()
    try:
        crow = CirclesStore(conn).get(project, circle)
    except CircleNotFound:
        raise_typed(PlatformError.sourcegrants("not_found", f"circle {ref} not found"))
    finally:
        pool.putconn(conn)
    if crow["status"] != "active":
        raise_typed(PlatformError.sourcegrants("conflict", f"circle {ref} is retired"))


class CreateGrantRequest(BaseModel):
    source_name: str = Field(..., description="pushed source the caller owns")
    grantee: str = Field(
        ...,
        min_length=1,
        description="user id, or 'circle:<project_id>/<circle_id>' (ADR-086)",
    )
    allow_paths: list[str] = Field(
        default_factory=list, description="narrows the source opt-ins (intersection)"
    )
    budget_usd: float | None = Field(None, description="R3-reserved; null = uncapped")
    expires_at: datetime | None = Field(None, description="optional auto-expiry")


@router.post("", status_code=status.HTTP_201_CREATED)
def create_grant(
    body: CreateGrantRequest,
    request: Request,
    caller: CallerScope = Depends(get_caller_scope),
) -> dict[str, Any]:
    """# auth: verified"""
    if not caller.user_id:
        raise_typed(
            PlatformError.sourcegrants("unauthorized", "user identity required")
        )
    if body.grantee == caller.user_id:
        raise_typed(
            PlatformError.sourcegrants("bad_request", "cannot grant to yourself")
        )
    if body.grantee.startswith("circle:"):
        _validate_circle_grantee(request, body.grantee)
    meta = _sealed_meta(request, body.source_name, caller.user_id)

    store = _store(request)
    # Intersection rule: grant paths must be a subset of the source's own opt-ins.
    source_paths = set(meta.allow_paths)
    requested = [p for p in body.allow_paths if p in source_paths]
    try:
        grant = store.create_grant(
            source_id=meta.id,
            owner=caller.user_id,
            grantee=body.grantee,
            allow_paths=tuple(requested),
            budget_usd=body.budget_usd,
            expires_at=body.expires_at,
            now=_now(),
        )
    except Exception as exc:
        # Live-grant uniqueness violation → honest conflict.
        if "source_grants_live_unique" in str(exc):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "grant_exists",
                    "message": "a live grant already covers this source/grantee",
                },
            ) from exc
        raise
    _audit(
        request,
        action="grant_create",
        owner=caller.user_id,
        source_id=meta.id,
        detail={
            "grantee": body.grantee,
            "allow_paths": requested,
            "grant_id": grant.grant_id,
        },
    )
    return grant.to_dict()


@router.get("/mine")
def list_mine(
    request: Request,
    caller: CallerScope = Depends(get_caller_scope),
) -> list[dict[str, Any]]:
    """# auth: verified"""
    if not caller.user_id:
        raise_typed(
            PlatformError.sourcegrants("unauthorized", "user identity required")
        )
    store = _store(request)
    return [g.to_dict() for g in store.list_for_owner(caller.user_id)]


@router.get("/for-me")
def list_for_me(
    request: Request,
    caller: CallerScope = Depends(get_caller_scope),
) -> list[dict[str, Any]]:
    """# auth: verified"""
    if not caller.user_id:
        raise_typed(
            PlatformError.sourcegrants("unauthorized", "user identity required")
        )
    store = _store(request)
    # ADR-086: my user id plus every active circle:'<project>/<id>' I am in.
    from .circles import circles_for_user

    pool = getattr(request.app.state, "umrs_db_pool", None)
    grantees: list[str] = [caller.user_id]
    if pool is not None:
        conn = pool.getconn()
        try:
            grantees += [
                f"circle:{ref}" for ref in circles_for_user(conn, caller.user_id)
            ]
        finally:
            pool.putconn(conn)
    if len(grantees) == 1:
        return [g.to_dict() for g in store.list_for_grantee(caller.user_id)]
    return [g.to_dict() for g in store.list_for_grantees(tuple(grantees))]


@router.post("/{grant_id}/revoke")
def revoke_grant(
    grant_id: str,
    request: Request,
    caller: CallerScope = Depends(get_caller_scope),
) -> dict[str, Any]:
    """# auth: verified"""
    if not caller.user_id:
        raise_typed(
            PlatformError.sourcegrants("unauthorized", "user identity required")
        )
    store = _store(request)
    grant = store.get_grant(grant_id)
    if grant is None:
        raise_typed(PlatformError.sourcegrants("not_found", "Grant not found"))
    if grant.owner != caller.user_id:
        raise_typed(
            PlatformError.sourcegrants("not_found", "Grant not found")
        )  # owner-only; no leakage
    if not store.revoke_grant(grant_id, actor=caller.user_id, now=_now()):
        raise_typed(
            PlatformError.sourcegrants(
                "not_found", "Grant not found or already revoked"
            )
        )

    # Cascade: any live lease referencing a now-revoked source for that grantee dies.
    from ..jobs.secret_broker import build_default_broker

    broker = build_default_broker()
    revoked = broker.revoke_by_source(grant.source_id, actor=caller.user_id)
    _audit(
        request,
        action="grant_revoke",
        owner=caller.user_id,
        source_id=grant.source_id,
        detail={
            "grantee": grant.grantee,
            "grant_id": grant_id,
            "leases_revoked": revoked,
        },
    )
    return {"grant_id": grant_id, "revoked": True, "leases_revoked": revoked}


def _audit(
    request: Request, *, action: str, owner: str, source_id: str, detail: dict
) -> None:
    from umrs.security.sealed_sources import SealedSourceStore

    SealedSourceStore(request.app.state.umrs_db_pool).audit(
        action=action, owner=owner, source_id=source_id, detail=detail
    )
