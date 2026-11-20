"""Admin credential for session and usage administration (U3 S1 item 6).

The admin session, session-policy and usage routes share one FastAPI
dependency: an admin JWT bearer, or a gateway-verified admin identity, from
an account that is not disabled. Moved verbatim from ``admin_sessions``
(the venue admin router moves to ``umrs_r4`` in U3 S5); the typed
``companionvenue.*`` codes and the ``admin.denied`` audit events are unchanged.
"""

from __future__ import annotations

from typing import Any, NoReturn

from fastapi import Header

from umrs.venues.session_audit import audit_venue

from .typed_errors import PlatformError, raise_typed

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


__all__ = ["require_session_admin"]
