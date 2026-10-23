"""Admin session control over the frontend venue (C-1, remote-session plan RS-0).

    GET  /umrs/admin/sessions                 every user's sessions
                                              (filters: user, project, state, host)
    GET  /umrs/admin/sessions/{sid}           one session of any owner
    POST /umrs/admin/sessions/{sid}/stop      stop it (the owner stop path,
                                              actor = the admin, cause = admin)

The routes sit on the existing ``FrontendVenueService`` (the owner routes'
process-cached instance): no second session store, no new topology. Pause,
resume and host drain are RS-2.

Access: admin only, with the credentials the endpoint registry's admin
routes already take (``endpoints._caller_is_admin``): an admin JWT bearer
(``roles``/``groups`` holds ``admin``; signed with the server's JWT secret,
which ``umrs-server admin endpoint|session`` send), or a gateway-signed
admin identity (``X-Verified-*`` + HMAC). Strict and mvp alike: a request
with neither is a typed 401, a non-admin a typed 403 (audited as
``venue.admin.denied``). No gateway location serves ``/umrs/admin``; the
operator CLI calls UMRS directly. A credential whose account is disabled
is a typed 403 too, whatever its roles say (ADR-097 Amendment D-8,
2026-10-23; audited as ``venue.admin.denied`` with reason
``account_disabled``).

Admin views leave out the worker's live text and tool lines (the user's
conversation, not control data); see ``frontend_service.admin_view``.
"""

from __future__ import annotations

import asyncio
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel

from umrs.venues.frontend_service import SESSION_HOST, VenueError
from umrs.venues.session_audit import audit_venue
from umrs.venues.venue_sessions import VALID_STATES

from .companion_venue import _typed, _unexpected, get_venue_service
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/admin/sessions", tags=["admin-sessions"])

MAX_LIST = 1000


# ── admin credential ───────────────────────────────────────────────────────


def _refuse(code: str, message: str) -> NoReturn:
    raise_typed(PlatformError.companionvenue(code, message))


def _denied(actor: str) -> NoReturn:
    audit_venue(
        "admin.denied", actor=actor, session_id="", outcome="denied", scope="system"
    )
    _refuse("forbidden", "admin session control requires the admin role")


def _refuse_disabled(actor: str) -> None:
    """A disabled account's admin credential is refused (ADR-097 Amendment
    D-8, 2026-10-23): the JWT used to outlive its owner's disable."""
    from .endpoints import _admin_disabled

    if not _admin_disabled(actor):
        return
    audit_venue(
        "admin.denied",
        actor=actor,
        session_id="",
        outcome="denied",
        scope="system",
        reason="account_disabled",
    )
    _refuse("forbidden", f"account '{actor}' is disabled")


def _admin_from_claims(claims: dict[str, Any]) -> str:
    from .endpoints import _claim_is_admin

    actor = str(claims.get("sub") or "")
    if not _claim_is_admin(claims):
        _denied(actor)
    _refuse_disabled(actor)
    return actor or "admin"


def _admin_from_headers(
    uid: str | None, pid: str | None, is_admin: str | None, sig: str | None
) -> str:
    from ._verified_headers import resolve_caller

    if not (uid or "").strip():
        _refuse(
            "unauthorized",
            "admin credential required: an admin JWT bearer or a "
            "gateway-verified admin identity",
        )
    caller = resolve_caller(uid, pid, is_admin, sig)  # strict: bad signature → 401
    if caller is None:
        _refuse("unauthorized", "admin credential required (no verified project)")
    if not caller.is_admin:
        _denied(caller.user_id)
    _refuse_disabled(caller.user_id)
    return caller.user_id


def require_session_admin(
    authorization: str | None = Header(None),
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> str:
    """FastAPI dependency: the admin's user id, else a typed 401/403."""
    from .endpoints import _jwt_claims

    claims = _jwt_claims(authorization, required=False)  # a bad JWT is a 401
    if claims is not None:
        return _admin_from_claims(claims)
    return _admin_from_headers(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )


# ── routes ─────────────────────────────────────────────────────────────────


class AdminStopBody(BaseModel):
    note: str = ""


def _checked_state(state: str) -> str:
    state = state.strip()
    if state and state not in VALID_STATES:
        _refuse("bad_request", f"state must be one of {', '.join(VALID_STATES)}")
    return state


@router.get("", response_model=dict[str, Any])
async def admin_list_sessions(
    request: Request,
    user: str = Query("", description="owner user id"),
    project: str = Query("", description="the session job's project"),
    state: str = Query("", description="provisioning|running|paused|failed|stopped"),
    host: str = Query("", description=f"session host (today only {SESSION_HOST!r})"),
    limit: int = Query(100, ge=1, le=MAX_LIST),
    admin: str = Depends(require_session_admin),
) -> dict[str, Any]:
    """Every user's sessions, newest first.
    # auth: admin(jwt|verified-admin)
    """
    filters = {
        "user": user.strip(),
        "project": project.strip(),
        "state": _checked_state(state),
    }
    sessions: list[dict[str, Any]] = []
    if host.strip() in ("", SESSION_HOST):
        service = get_venue_service(request)
        sessions = await _guarded(
            "admin list",
            service.admin_list,
            user_id=filters["user"],
            state=filters["state"],
            project=filters["project"],
            limit=limit,
        )
    audit_venue(
        "admin.list",
        actor=admin,
        session_id="",
        scope="system",
        filters={**filters, "host": host.strip()},
        count=len(sessions),
    )
    return {"sessions": sessions, "total": len(sessions), "hosts": [SESSION_HOST]}


@router.get("/{session_id}", response_model=dict[str, Any])
async def admin_show_session(
    session_id: str, request: Request, admin: str = Depends(require_session_admin)
) -> dict[str, Any]:
    """One session of any owner (job, lease, owner, project).
    # auth: admin(jwt|verified-admin)
    """
    service = get_venue_service(request)
    status = await _guarded("admin status", service.admin_status, session_id)
    if status is None:
        _refuse("not_found", "no venue runtime for this session")
    audit_venue(
        "admin.show",
        actor=admin,
        session_id=session_id,
        owner=str(status.get("user_id") or ""),
        project=str(status.get("project") or ""),
        scope="system",
    )
    return status


@router.post("/{session_id}/stop", response_model=dict[str, Any])
async def admin_stop_session(
    session_id: str,
    request: Request,
    body: AdminStopBody | None = None,
    admin: str = Depends(require_session_admin),
) -> dict[str, Any]:
    """Stop any user's session: teardown, lease revoked, job settled
    (idempotent; audited as ``venue.session.stop`` with cause ``admin``).
    # auth: admin(jwt|verified-admin)
    """
    service = get_venue_service(request)
    note = (body.note if body else "").strip()
    return await _guarded(
        "admin stop", service.admin_stop, session_id=session_id, admin=admin, note=note
    )


async def _guarded(verb: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a service call off the loop; venue errors stay typed."""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except VenueError as exc:
        _typed(exc)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — unexpected → typed 503 naming the cause
        _unexpected(exc, verb)
