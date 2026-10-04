"""Asset index HTTP surface (Review F4, plan track 1 item 6).

Routes under ``/umrs/assets`` (gateway path ``/gw/assets/*``):

- GET /umrs/assets           → full snapshot (kinds filter, project filter)
- GET /umrs/assets/{kind}/{asset_id} → one row with lineage when present

Consistent with every other degraded-able surface: a store that raises is a
``warnings`` entry in the response; an entirely absent store just yields no
rows of its kind (see asset_index.fanout comments).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from umrs.asset_index import build_asset_index

from ._scope_identity import get_caller_scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/assets", tags=["assets"])


class AssetSnapshotResponse(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    # Additive (2026-09-16): snapshot-level content hash — the caller keeps it
    # for the /patch wire rail's `base` state (never a server-held cookie).
    snapshot_hash: str = ""


class AssetRowResponse(BaseModel):
    row: dict[str, Any]


def _gather_stores() -> dict[str, Any]:
    """Pull every available store handle; each accessor is independently fallible."""
    stores: dict[str, Any] = {}

    def _grab(key: str, fn) -> None:
        try:
            value = fn()
            if value is not None:
                stores[key] = value
        except Exception:
            logger.debug("asset index: %s store unavailable", key, exc_info=True)

    from umrs.api import jobs as jobs_api
    from umrs.api import mcp as mcp_api
    from umrs.api import models as models_api
    from umrs.api import research_sessions as sessions_api

    _grab("job_store", jobs_api.get_optional_job_store)
    _grab("model_store", models_api.get_optional_model_store)
    _grab("mcp_store", lambda: getattr(mcp_api, "_mcp_store", None))
    _grab("session_store", sessions_api.get_session_store)

    # Artifact store is constructed on demand from the job pool (artifact_routes
    # pattern: pool.getconn as factory + putconn as release).
    try:
        from umrs.artifact_registry.store import ArtifactStore

        job_store = stores.get("job_store")
        if job_store is not None and getattr(job_store, "db_pool", None):
            stores["artifact_store"] = ArtifactStore(
                job_store.db_pool.getconn, conn_release=job_store.db_pool.putconn
            )
    except (ImportError, TypeError, AttributeError):
        logger.debug("artifact store construction skipped; artifact kind omitted")

    try:
        from umrs.api.loop import _loop_store

        stores["loop_store"] = _loop_store
    except ImportError:
        logger.debug("loop store accessor not present; loops kind skipped")

    return stores


def _caller_circle_refs(request: Request, caller: Any) -> frozenset[str]:
    """Active circle refs for the caller (ADR-086 authority); empty without
    a pool (circle rows stay hidden — the safe edge, never fabricated)."""
    user_id = getattr(caller, "user_id", "") or ""
    if not user_id:
        return frozenset()
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        return frozenset()
    from .circles import circles_for_user

    conn = pool.getconn()
    try:
        return frozenset(circles_for_user(conn, user_id))
    finally:
        pool.putconn(conn)


@router.get("", response_model=AssetSnapshotResponse)
async def snapshot(
    request: Request,
    kinds: str | None = Query(None, description="comma-separated kind filter"),
    project_id: str | None = Query(None),
    caller: Any = Depends(get_caller_scope),
) -> AssetSnapshotResponse:
    """Cross-subsystem asset index snapshot (read-model, F4).

    # auth: verified

    Callers see only rows for scopes they inhabit (system + verified project
    + own user rows + ACTIVE circles from the ADR-086 membership authority);
    nothing else ever leaks.
    """
    stores = _gather_stores()
    kind_list = [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None
    # Non-admin project filter is bound to the caller's verified project.
    bound_project = (
        project_id
        if getattr(caller, "is_admin", False)
        else (getattr(caller, "project_id", "") or project_id)
    )
    rows, warnings = build_asset_index(
        project_id=bound_project,
        kinds=kind_list,
        artifact_store=stores.get("artifact_store"),
        model_store=stores.get("model_store"),
        mcp_store=stores.get("mcp_store"),
        session_store=stores.get("session_store"),
        loop_store=stores.get("loop_store"),
        job_store=stores.get("job_store"),
        caller=caller,
        caller_circles=_caller_circle_refs(request, caller),
    )
    row_dicts = [r.to_dict() for r in rows]
    return AssetSnapshotResponse(
        rows=row_dicts,
        warnings=[w.to_dict() for w in warnings],
        count=len(row_dicts),
        snapshot_hash=_snapshot_hash(
            {f"{d['kind']}:{d['asset_id']}": _row_shape_hash(d) for d in row_dicts}
        ),
    )


# ── catalog patch lane (2026-09-16) ─────────────────────────────────────────
# Contract: caller keeps `base[{kind}:{asset_id}] = sha1(row-shape)`, posts it;
# the server recomposes the same snapshot and diffs. No server-side client
# state anywhere — the request body IS the state (honest wire rail).
def _row_shape_hash(row_dict: dict[str, Any]) -> str:
    import hashlib
    import json

    canonical = json.dumps(row_dict, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _snapshot_hash(pairs: dict[str, str]) -> str:
    import hashlib

    canonical = "\n".join(f"{k}={pairs[k]}" for k in sorted(pairs))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


class CatalogPatchRequest(BaseModel):
    base: dict[str, str] = Field(
        default_factory=dict,
        description='caller-held map {"kind:asset_id": shape-hash} from the last snapshot it saw',
        max_length=10_000,
    )
    kinds: str | None = None
    project_id: str | None = None


class CatalogPatchResponse(BaseModel):
    snapshot_hash: str
    upserts: list[dict[str, Any]]
    tombstones: list[str]
    same: bool
    warnings: list[dict[str, Any]]
    count: int


def _gather_and_diff(
    request: Request, body: CatalogPatchRequest, caller: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[Any]]:
    stores = _gather_stores()
    bound_project = (
        body.project_id
        if getattr(caller, "is_admin", False)
        else (getattr(caller, "project_id", "") or body.project_id)
    )
    kind_list = (
        [k.strip() for k in body.kinds.split(",") if k.strip()] if body.kinds else None
    )
    rows, warnings = build_asset_index(
        project_id=bound_project,
        kinds=kind_list,
        artifact_store=stores.get("artifact_store"),
        model_store=stores.get("model_store"),
        mcp_store=stores.get("mcp_store"),
        session_store=stores.get("session_store"),
        loop_store=stores.get("loop_store"),
        job_store=stores.get("job_store"),
        caller=caller,
        caller_circles=_caller_circle_refs(request, caller),
    )
    shape: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for row in rows:
        d = row.to_dict()
        key = f"{d['kind']}:{d['asset_id']}"
        shape[key] = d
        hashes[key] = _row_shape_hash(d)
    return shape, hashes, warnings


@router.post("/patch", response_model=CatalogPatchResponse)
async def catalog_patch(
    body: CatalogPatchRequest,
    request: Request,
    caller: Any = Depends(get_caller_scope),
) -> CatalogPatchResponse:
    """Incmental wire rail over the catalog snapshot (F4 follow-through).

    # auth: verified

    Caller sends its last seen base (key -> shape hash) from a snapshot
    materialization; server recomposes the same visible set right now and
    returns only what changed, plus the new snapshot-level hash to keep.
    Same visibility rules as the snapshot route.
    """
    shape, hashes, warnings = _gather_and_diff(request, body, caller)
    base = body.base or {}
    upserts = [shape[k] for k, cur in hashes.items() if base.get(k) != cur]
    tombstones = sorted(k for k in base.keys() - set(hashes.keys()))
    same = not upserts and not tombstones
    return CatalogPatchResponse(
        snapshot_hash=_snapshot_hash(hashes),
        upserts=upserts,
        tombstones=tombstones,
        same=same,
        warnings=[w.to_dict() for w in warnings],
        count=len(shape),
    )


@router.get("/{kind}/{asset_id}", response_model=AssetRowResponse)
async def get_asset(
    request: Request,
    kind: str,
    asset_id: str,
    caller: Any = Depends(get_caller_scope),
) -> AssetRowResponse:
    """Fetch one asset row by kind+id (caller-filtered)."""
    stores = _gather_stores()
    rows, warnings = build_asset_index(
        kinds=[kind],
        artifact_store=stores.get("artifact_store"),
        model_store=stores.get("model_store"),
        mcp_store=stores.get("mcp_store"),
        session_store=stores.get("session_store"),
        loop_store=stores.get("loop_store"),
        job_store=stores.get("job_store"),
        caller=caller,
        caller_circles=_caller_circle_refs(request, caller),
    )
    for row in rows:
        if row.asset_id == asset_id:
            return AssetRowResponse(row=row.to_dict())
    raise HTTPException(
        status_code=404,
        detail={
            "error": "asset_not_found",
            "kind": kind,
            "asset_id": asset_id,
            "warnings": [w.to_dict() for w in warnings],
        },
    )
