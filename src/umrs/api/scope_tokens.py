"""Scope tokens over HTTP (ADR-108 D-3, W2b).

- ``POST /umrs/scopes``: issue a token for a work level on the caller's
  verified identity; with ``parent_token``, a child that never exceeds its
  parent.
- ``POST /umrs/scopes/revoke``: revoke a path and everything beneath it, for
  the caller's own user and project.
- :func:`scope_of` / :func:`require_grant`: what the routes a level calls use
  to read ``x-r4-scope``.

A token never adds a right. It must match the request's verified user and
project, and it only narrows what that caller may do: grants say which
writes the level may make. A request without the header is unchanged.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from umrs.scopes.store import store_for
from umrs.scopes.tokens import (
    HEADER,
    ScopeClaims,
    ScopeError,
    check_path,
    decode,
    issue,
)

from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/scopes", tags=["scopes"])


class IssueBody(BaseModel):
    path: str = Field(..., min_length=1, max_length=600)
    grants: list[str] = Field(default_factory=list, max_length=64)
    parent_token: str | None = Field(default=None, max_length=4096)
    ttl_seconds: int | None = None
    repo: str = ""
    branch: str = ""
    commit: str = ""


class RevokeBody(BaseModel):
    path: str = Field(..., min_length=1, max_length=600)


def _secret() -> str:
    from umrs.shared.handshake import handshake_secret

    return handshake_secret() or ""


def _refuse(exc: ScopeError) -> None:
    raise_typed(PlatformError.scopes(exc.code, str(exc)), cause=exc)


def _verified(token: str, request: Request) -> ScopeClaims:
    """A token's claims when it is signed, unexpired and not revoked."""
    claims = decode(token, _secret())
    if store_for(request.app).is_revoked(claims.user, claims.project, claims.path):
        raise ScopeError("revoked", f"scope {claims.path} was revoked")
    return claims


def _identity_check(identity: CallerIdentity) -> None:
    if not identity.user_id or not identity.project_id:
        raise ScopeError("forbidden", "a scope needs a verified user and project")


@router.post("")
def issue_scope(
    body: IssueBody,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Issue a scope token for a work level.
    # auth: verified
    """
    try:
        _identity_check(identity)
        parent = _verified(body.parent_token, request) if body.parent_token else None
        token, claims = issue(
            user=identity.user_id,
            project=identity.project_id,
            path=body.path,
            grants=body.grants,
            secret=_secret(),
            parent=parent,
            ttl=body.ttl_seconds,
            repo=body.repo,
            branch=body.branch,
            commit=body.commit,
        )
    except ScopeError as exc:
        _refuse(exc)
    return {"status": "ok", "token": token, "scope": claims.record()}


@router.post("/revoke")
def revoke_scope(
    body: RevokeBody,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Revoke a scope path and everything beneath it (the caller's own).
    # auth: verified
    """
    try:
        _identity_check(identity)
        path = check_path(body.path)
        store_for(request.app).revoke(identity.user_id, identity.project_id, path)
    except ScopeError as exc:
        _refuse(exc)
    return {"status": "ok", "revoked": path}


def scope_of(request: Request, *, user: str, project: str) -> ScopeClaims | None:
    """The request's scope (``x-r4-scope``), checked against its verified
    user and project; None without the header; a typed refusal otherwise."""
    token = (request.headers.get(HEADER) or "").strip()
    if not token:
        return None
    try:
        claims = _verified(token, request)
        if (claims.user, claims.project) != (user, project):
            raise ScopeError("forbidden", "the scope belongs to another user or project")
    except ScopeError as exc:
        _refuse(exc)
    return claims


def verified_scope(request: Request, *, project: str) -> ScopeClaims | None:
    """:func:`scope_of` for a route that has no verified user of its own:
    the user comes from the gateway-HMAC-verified headers
    (``request_caller_scope``), read only when the request is scoped, so an
    unscoped request is untouched."""
    if not (request.headers.get(HEADER) or "").strip():
        return None
    from ._scope_identity import request_caller_scope

    caller = request_caller_scope(request)
    return scope_of(request, user=caller.user_id, project=project)


def require_grant(claims: ScopeClaims | None, wanted: str) -> None:
    """Refuse a scoped request whose scope does not grant ``wanted``."""
    from umrs.scopes.tokens import allows

    if claims is not None and not allows(claims.grants, wanted):
        raise_typed(
            PlatformError.scopes(
                "grant_missing", f"scope {claims.path} does not grant {wanted}"
            )
        )
