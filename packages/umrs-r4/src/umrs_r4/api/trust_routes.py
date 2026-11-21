"""Companion trust routes (wave T3c, ADR-035): the r4 extension's part.

U3 S4b moved them out of core ``umrs.api.companion``: the per-user-scope
trust allowlist (``/umrs/companion/trust*``) and the session archive
tombstone (``/umrs/companion/sessions/{session_id}/archive``). The
extension registers ``router`` through ``ExtensionContext.add_router``;
core keeps the companion approval routes.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from umrs.api.identity import strict_user_header
from umrs.api.typed_errors import PlatformError, raise_typed
from umrs.shared.scope_grammar import ScopeError, fallback_chain
from umrs.shared.scope_grammar import parse as parse_scope

from umrs_r4.trust.store import SessionArchivedError

logger = logging.getLogger(__name__)

_strict_headers = strict_user_header

router = APIRouter(prefix="/umrs/companion", tags=["companion"])

# Per-user-scope TRUST allowlist (wave T3c): module-level store handle, set
# at sidecar boot the same way the research-session store is.
# ADR-053 (1.1): the singleton lives in umrs_r4.trust.access.


def _require_trust_store():
    from umrs_r4.trust.access import get_trust_store as _shared_get

    store = _shared_get()
    if store is None:
        raise_typed(
            PlatformError.companion(
                "unavailable",
                "Companion trust store unavailable (no DB pool configured)",
            )
        )
    return store


# ── typed error mapping ─────────────────────────────────────────────────────


def _http_error_from(exc: Exception) -> HTTPException:
    """Map typed venue errors to HTTP statuses deterministically."""
    if isinstance(exc, SessionArchivedError):
        return HTTPException(status_code=410, detail=str(exc))
    if isinstance(exc, HTTPException):
        return exc
    return HTTPException(status_code=500, detail=str(exc))


def _assert_mirrored_session_owner_order(session_id: str, user_id: str) -> None:
    """403 on cross-owner archive (only when a mirrored row identifies the
    real owner; row-absent → caller's tombstone lands under its own name).

    Reads `companion_session_snapshots` directly (0080/0084 lockstep — owner
    column name is `user_id`, same F1-fix lived across routes).
    """
    from umrs.shared.pg import get_default_sync_pool

    pool = get_default_sync_pool()
    if pool is None:
        return  # dev-tier without the pool: local-truth lane rules already block
    conn = pool.getconn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id FROM companion_session_snapshots WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
    finally:
        pool.putconn(conn)
    if row is None:
        return  # never had a mirrored row — caller's own tombstone territory
    if row[0] != user_id:
        raise_typed(
            PlatformError.companion(
                "forbidden", "archive refused: session is not owned by this caller"
            )
        )


def _strict_caller(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> dict[str, str]:
    """Gate + return (user_id, project_id) for scope-authorizing routes."""
    uid = _strict_headers(x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac)
    return {"user_id": uid, "project_id": (x_verified_project_id or "").strip()}


# ── request/response models ──────────────────────────────────────────────────


class TrustGrantRequest(BaseModel):
    tool_name: str = Field(..., min_length=1)
    scope: str = Field("system", min_length=1)
    ttl_sec: int = Field(1800, ge=60, le=86400)


class TrustListResponse(BaseModel):
    rows: list[dict[str, Any]] = []
    count: int = 0


def _validated_scope(raw: str) -> str:
    """Canonical scope grammar (umrs.shared.scope_grammar). Legacy 'global'
    is REJECTED on this write path — callers must pass 'system'."""
    try:
        return str(parse_scope(raw, allow_legacy=False))
    except ScopeError as exc:
        raise_typed(PlatformError.companion("unprocessable", str(exc)), cause=exc)


def _authorize_scope_write(scope: str, x_user_id: str, caller_project: str) -> None:
    """Scope-write authorization (matrix row: companion trust).

    Rows are always user-pinned, but scope strings claim a domain and must
    not be mintable for domains the caller doesn't inhabit:
    - project:<pid>  → pid must be the caller's verified project
    - circle:<p/cid> → not resolvable on this trust-write path → 422
    - group:<gid>    → superseded by circle: (ADR-086) → 422
    - user:<uid>     → uid must equal the caller
    - session:<sid>  → allowed (caller-pinned rows)
    - system         → allowed (per-user blanket, still user-pinned)
    """
    claim = parse_scope(scope)
    if claim.kind == "project":
        if caller_project and claim.id == caller_project:
            return
        raise_typed(
            PlatformError.companion(
                "forbidden",
                (
                    f"cannot grant trust scoped to project {claim.id!r} — not the "
                    "caller's verified project"
                ),
            )
        )
    if claim.kind in ("circle", "group"):
        raise_typed(
            PlatformError.companion(
                "unprocessable",
                f"{claim.kind}-scoped trust is not writable on this path"
                " (audience-scoped trust resolves nowhere here)",
            )
        )
    if claim.kind == "user" and claim.id != x_user_id:
        raise_typed(
            PlatformError.companion(
                "forbidden",
                f"cannot grant trust scoped to user {claim.id!r} — not the caller",
            )
        )


# ── per-user-scope TRUST allowlist (wave T3c, ADR-035) ───────────────────────


@router.get("/trust", response_model=TrustListResponse)
async def list_trust_rows(
    scope: str | None = None,
    x_user_id: str = Depends(_strict_headers),
) -> TrustListResponse:
    """List the caller's unexpired allowlist rows (owner-pinned)."""
    store = _require_trust_store()
    resolved = _validated_scope(scope) if scope else None
    rows = store.list(x_user_id, scope=resolved)
    return TrustListResponse(rows=[r.to_dict() for r in rows], count=len(rows))


@router.put("/trust", status_code=status.HTTP_201_CREATED)
async def grant_trust_row(
    body: TrustGrantRequest,
    caller: dict = Depends(_strict_caller),
) -> dict:
    """Grant/refresh a trust row for the caller (owner-pinned, TTL-capped,
    scope-write authorized against the verified caller identity)."""
    store = _require_trust_store()
    scope = _validated_scope(body.scope)
    _authorize_scope_write(scope, caller["user_id"], caller["project_id"])
    try:
        row = store.grant(
            caller["user_id"],
            body.tool_name,
            scope=scope,
            ttl_sec=body.ttl_sec,
            created_by=caller["user_id"],
        )
    except SessionArchivedError as exc:
        # Gone: once the tombstone exists, late grant attempts must fail
        # closed (410) — the archived session's scopes are terminal.
        raise_typed(PlatformError.companion("gone", str(exc)), cause=exc)
    return row.to_dict()


@router.get("/trust/check")
async def check_trust_row(
    tool_name: str,
    scope: str | None = None,
    x_user_id: str = Depends(_strict_headers),
) -> dict:
    """Autopilot consult: is this tool allowlisted NOW for the caller+scope?

    Returns ``{"allowed": bool, "expires_at": ...}``. The companion calls this
    instead of trusting its own process-global TRUST mode.
    """
    store = _require_trust_store()
    resolved = _validated_scope(scope) if scope else "system"
    row = None
    # Precedence walk: session > project > group > user > system (one module,
    # one chain — see umrs.shared.scope_grammar).
    for candidate in fallback_chain(resolved):
        row = store.check(x_user_id, tool_name, scope=candidate)
        if row is not None:
            break
    return {
        "allowed": row is not None,
        "scope": (row.scope if row else resolved),
        "expires_at": (row.expires_at if row else ""),
    }


@router.delete("/trust/{tool_name}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_trust_row(
    tool_name: str,
    scope: str | None = None,
    x_user_id: str = Depends(_strict_headers),
) -> None:
    """Revoke a trust row (owner-pinned)."""
    store = _require_trust_store()
    resolved = _validated_scope(scope) if scope else "system"
    store.revoke(x_user_id, tool_name, scope=resolved)


@router.post("/sessions/{session_id}/archive")
async def archive_companion_session(
    session_id: str,
    caller: dict = Depends(_strict_caller),
) -> dict:
    """Archive a companion session — tombstone + session-scope trust revoke
    # auth: verified
    + one audit row per revoked grant, ONE transaction (schema v2).

    This IS the client `session archive` lifecycle hook: bridge, CLI, and
    ACP converge here so the cascade is enforced where the trust rows live,
    never in whichever client triggered the archive. A separate bulk-revoke
    verb (if ever added) stays an explicit user action, NOT part of archive.

    F5 audit-block (2026-09-21): when a mirrored snapshot row records its
    own user_id, cross-owner archives refuse: the caller can't permanently
    traverse someone else's session-death OR trust cascade. No probe-row?
    Tombstone lands under this caller's own name only.
    """
    store = _require_trust_store()
    _assert_mirrored_session_owner_order(session_id, caller["user_id"])
    revoked = store.archive_session(session_id, archived_by=caller["user_id"])
    return {
        "session_id": session_id,
        "archived": True,
        "revoked_grants": revoked,
        "revoked_count": len(revoked),
    }
