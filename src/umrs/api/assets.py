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
from typing import Any, Optional

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
        except Exception:  # noqa: BLE001 — grab failures show as absent stores
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


@router.get("", response_model=AssetSnapshotResponse)
async def snapshot(
    kinds: Optional[str] = Query(None, description="comma-separated kind filter"),
    project_id: Optional[str] = Query(None),
    caller: Any = Depends(get_caller_scope),
) -> AssetSnapshotResponse:
    """Cross-subsystem asset index snapshot (read-model, F4).

    Callers see only rows for scopes they inhabit (system + verified project
    + own user rows); group rows are admin-only until a group-membership
    authority exists — fail closed, never silent leakage.
    """
    stores = _gather_stores()
    kind_list = [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None
    # Non-admin project filter is bound to the caller's verified project.
    bound_project = project_id if getattr(caller, "is_admin", False) else (
        getattr(caller, "project_id", "") or project_id
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
    )
    return AssetSnapshotResponse(
        rows=[r.to_dict() for r in rows],
        warnings=[w.to_dict() for w in warnings],
        count=len(rows),
    )


@router.get("/{kind}/{asset_id}", response_model=AssetRowResponse)
async def get_asset(
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
    )
    for row in rows:
        if row.asset_id == asset_id:
            return AssetRowResponse(row=row.to_dict())
    raise HTTPException(
        status_code=404,
        detail={"error": "asset_not_found", "kind": kind, "asset_id": asset_id, "warnings": [w.to_dict() for w in warnings]},
    )
