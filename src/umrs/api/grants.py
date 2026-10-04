"""Asset grant routes (ADR-041).

Sharing happens by version-pinned grants with tombstone-on-revoke, not by
copy/re-scope (P10-C2). Create/list/revoke require write authority on the
asset (the static scope ladder, never a grant — grants authorize READS
only); ``GET /mine`` shows the caller what it received.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from umrs.grants import ASSET_KINDS, GRANTEE_KINDS, AssetGrant, GrantStore
from umrs.knowledge.access_scope import CallerScope, graph_writable

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/grants", tags=["asset-grants"])


class GrantBody(BaseModel):
    """Create-grant request. ``version_pin=None`` follows the current asset."""

    asset_kind: str
    asset_id: str = Field(min_length=1, max_length=256)
    grantee_kind: str
    grantee_id: str = Field(min_length=1, max_length=256)
    version_pin: str | None = Field(default=None, max_length=128)
    expires_at: datetime | None = None


# ── Identity + pool helpers ───────────────────────────────────────────────


def _caller(request: Request) -> CallerScope:
    uid = request.headers.get("x-user-id", "").strip()
    if not uid:
        raise_typed(PlatformError.grants("unauthorized", "identity header missing"))
    pid = request.headers.get("x-verified-project-id", "").strip()
    admin = request.headers.get("x-verified-is-admin", "").strip() in ("true", "1")
    return CallerScope(user_id=uid, project_id=pid, is_admin=admin)


def _pool(request: Request) -> Any:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(PlatformError.grants("unavailable", "database pool unavailable"))
    return pool


# ── Asset write-authority checks (static ladder only; grants never stack) ──


def _model_writable(pool: Any, asset_id: str, caller: CallerScope) -> bool:
    from umrs.model_registry.store import AccessContext, ModelStore
    from umrs.model_registry.store.errors import ModelNotFoundError

    conn = pool.getconn()
    try:
        store = ModelStore(conn)
        try:
            model = store.get_model(asset_id)
        except ModelNotFoundError:
            return False
        ctx = AccessContext(
            user_id=caller.user_id,
            is_admin=caller.is_admin,
            project_memberships={caller.project_id} if caller.project_id else set(),
        )
        return store.acl.can_write(model, ctx)
    finally:
        pool.putconn(conn)


def _graph_writable(pool: Any, asset_id: str, caller: CallerScope) -> bool:
    from umrs.knowledge.graph_registry import GraphRegistryStore

    meta = GraphRegistryStore(pool).get_graph(asset_id)
    if meta is None:
        return False
    return graph_writable(
        kind=meta.kind,
        project_id=meta.project_id,
        owner_id=meta.owner_id,
        caller=caller,
    )


def _endpoint_writable(asset_id: str, caller: CallerScope) -> bool:
    """WP-06 r2: endpoints grant-writable by the provisioning project's member.

    Wired via the in-process EndpointRegistry (no duplicate dialect query):
    caller's project must match the endpoint's project_id, OR the endpoint's
    recorded owner_id when one is pinned (admin wins anywhere).
    """
    from umrs.orchestrator.endpoint_registry import get_registry

    registry = get_registry()
    caller_pid = caller.project_id or ""
    for ep in registry.list_all(project_id=caller_pid) + list(
        registry.list_all(project_id="") or []
    ):
        if ep.job_id != asset_id:
            continue
        owner = str(ep.metadata.get("owner_id") or "")
        if owner and caller.user_id:
            return owner == caller.user_id
        return bool(caller_pid) and ep.project_id == caller_pid
    return False


def _workflow_entry(asset_id: str) -> dict | None:
    """Catalog lookup for the workflow name (module catalog, never a new dialect)."""
    from ..server import workflow_registry_handler as _wrh

    try:
        catalog = _wrh._load_catalog()
    except Exception:
        return None
    for entry in (catalog or {}).get("workflows", []):
        if str(entry.get("name") or "") == asset_id:
            return entry
    return None


def _workflow_writable(asset_id: str, caller: CallerScope) -> bool:
    """WP-06 r2: workflows are owner-writable (authority mirrors workflow_delete)."""
    entry = _workflow_entry(asset_id)
    if not isinstance(entry, dict):
        return False
    if str(entry.get("scope") or "") == "system":
        return False  # system-scope = admin-owned; admin wins above
    owner = str(entry.get("user_id") or "").strip()
    return bool(owner) and bool(caller.user_id) and owner == caller.user_id


def _asset_writable(
    pool: Any, asset_kind: str, asset_id: str, caller: CallerScope
) -> bool:
    if caller.is_admin:
        return True
    if asset_kind == "model":
        return _model_writable(pool, asset_id, caller)
    if asset_kind == "knowledge_graph":
        return _graph_writable(pool, asset_id, caller)
    if asset_kind == "endpoint":
        return _endpoint_writable(asset_id, caller)
    if asset_kind == "workflow":
        return _workflow_writable(asset_id, caller)
    return False


def _validate_circle_grantee(pool: Any, grantee_id: str) -> None:
    """ADR-086: a circle grantee is '<project>/<circle>' and must resolve to
    an ACTIVE circle (400 malformed, 404 unknown, 409 retired — never a
    dangling audience accreting in the store)."""
    from .circles import CircleNotFound, CirclesStore

    project, sep, circle = grantee_id.partition("/")
    if not sep or not project or not circle:
        raise_typed(
            PlatformError.grants(
                "bad_request",
                "circle grantee_id must be '<project_id>/<circle_id>'",
            )
        )
    gconn = pool.getconn()
    try:
        crow = CirclesStore(gconn).get(project, circle)
    except CircleNotFound:
        raise_typed(PlatformError.grants("not_found", f"circle {grantee_id} not found"))
    finally:
        pool.putconn(gconn)
    if crow["status"] != "active":
        raise_typed(PlatformError.grants("conflict", f"circle {grantee_id} is retired"))


# ── Routes ─────────────────────────────────────────────────────────────────


@router.post("", response_model=dict[str, Any])
async def create_grant(body: GrantBody, request: Request) -> dict[str, Any]:
    """Create a grant on an asset the caller controls (static write ladder).
    # auth: verified
    """
    caller = _caller(request)
    pool = _pool(request)
    if body.asset_kind not in ASSET_KINDS or body.grantee_kind not in GRANTEE_KINDS:
        raise_typed(
            PlatformError.grants("bad_request", "unknown asset_kind or grantee_kind")
        )
    if body.grantee_kind == "circle":
        _validate_circle_grantee(pool, body.grantee_id)
    if not _asset_writable(pool, body.asset_kind, body.asset_id, caller):
        # 404-shaped: the asset is absent or not the caller's to share.
        raise_typed(PlatformError.grants("not_found", "asset not found"))
    conn = pool.getconn()
    try:
        store = GrantStore(conn)
        try:
            grant = store.create_grant(
                asset_kind=body.asset_kind,
                asset_id=body.asset_id,
                grantee_kind=body.grantee_kind,
                grantee_id=body.grantee_id,
                created_by=caller.user_id,
                version_pin=body.version_pin,
                expires_at=body.expires_at,
            )
        except Exception as exc:  # live-grant uniqueness violation → conflict
            raise_typed(PlatformError.grants("conflict", str(exc)), cause=exc)
    finally:
        pool.putconn(conn)
    return {"status": "ok", "grant": grant.to_dict()}


def _serialize(grants: list[AssetGrant]) -> dict[str, Any]:
    return {"grants": [g.to_dict() for g in grants]}


@router.get("", response_model=dict[str, Any])
async def list_grants_for_asset(
    request: Request, asset_kind: str = "", asset_id: str = ""
) -> dict[str, Any]:
    """Live + tombstoned grants on one asset (asset writers only).
    # auth: verified
    """
    caller = _caller(request)
    pool = _pool(request)
    if not asset_kind or not asset_id:
        raise_typed(
            PlatformError.grants("bad_request", "asset_kind and asset_id required")
        )
    if not _asset_writable(pool, asset_kind, asset_id, caller):
        raise_typed(PlatformError.grants("not_found", "asset not found"))
    conn = pool.getconn()
    try:
        grants = GrantStore(conn).list_for_asset(
            asset_kind, asset_id, include_tombstoned=True
        )
    finally:
        pool.putconn(conn)
    return _serialize(grants)


@router.get("/mine", response_model=dict[str, Any])
async def list_my_grants(request: Request) -> dict[str, Any]:
    """Live grants received by the caller (as user and via project).
    # auth: verified
    """
    caller = _caller(request)
    pool = _pool(request)
    conn = pool.getconn()
    try:
        store = GrantStore(conn)
        grants = store.list_for_grantee("user", caller.user_id)
        if caller.project_id:
            grants = grants + store.list_for_grantee("project", caller.project_id)
        # ADR-086: plus every grant riding an ACTIVE circle the caller is in
        from .circles import circles_for_user

        for ref in circles_for_user(conn, caller.user_id):
            grants = grants + store.list_for_grantee("circle", ref)
    finally:
        pool.putconn(conn)
    return _serialize(grants)


@router.post("/{grant_id}/revoke", response_model=dict[str, Any])
async def revoke_grant(grant_id: str, request: Request) -> dict[str, Any]:
    """Tombstone a grant. Revoke = tombstone, never DELETE (audit survives).
    # auth: verified
    """
    caller = _caller(request)
    pool = _pool(request)
    conn = pool.getconn()
    try:
        store = GrantStore(conn)
        grant = store.get_grant(grant_id)
        if grant is None:
            raise_typed(PlatformError.grants("not_found", "grant not found"))
        if not _asset_writable(pool, grant.asset_kind, grant.asset_id, caller):
            raise_typed(PlatformError.grants("not_found", "grant not found"))
        revoked = store.revoke_grant(grant_id, now=datetime.now(timezone.utc))
    finally:
        pool.putconn(conn)
    return {"status": "ok", "revoked": revoked}
