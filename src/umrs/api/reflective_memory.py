"""Reflective memory inspection REST API (MEM-D4).

Provides:
- GET    /umrs/reflective-memory                    Inspect all reflective memory
- DELETE /umrs/reflective-memory/{tier}/{entry_id}  Delete a specific entry
- POST   /umrs/reflective-memory/{tier}/reset       Reset (deactivate) a tier

Access (H-1, 2026-10-17): admin-only. The routes took no identity, so any
caller read every project's workflow outcomes and routing feedback, deleted
entries, or deactivated every planning heuristic. This memory is
platform-global (heuristics are derived across all projects and carry no
project) and the inspector has no project filter, so a caller-scoped read is
not possible here; the module is the operator inspection surface. No gateway
location serves it and the r4 ``/memory`` command is retired. Dev (mvp)
callers without identity headers keep the old open behaviour.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ..jobs.adaptive_planning import AdaptivePlanningEngine
from ..jobs.reflective_memory import ReflectiveMemoryInspector
from ..jobs.routing_feedback import RoutingFeedbackStore
from ..jobs.workflow_outcome import WorkflowOutcomeStore
from ._verified_headers import CallerIdentity, get_caller
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)


def _require_admin(caller: CallerIdentity | None = Depends(get_caller)) -> None:
    """Gateway-signed admin only (the admin flag is part of the HMAC, so a
    forged flag is a 401 in ``get_caller``); anonymous dev (mvp) passes."""
    if caller is not None and not caller.is_admin:
        raise_typed(
            PlatformError.verified(
                "forbidden", "reflective memory is an operator surface"
            ),
        )


router = APIRouter(
    prefix="/umrs/reflective-memory",
    tags=["reflective-memory"],
    dependencies=[Depends(_require_admin)],
)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------

_inspector: ReflectiveMemoryInspector | None = None


def get_inspector() -> ReflectiveMemoryInspector:
    global _inspector
    if _inspector is None:
        db_url = os.environ.get("DATABASE_URL") or os.environ.get("UMRS_DB_URL")
        outcome_store = WorkflowOutcomeStore(db_url=db_url)
        routing_store = RoutingFeedbackStore(db_url=db_url)
        planning_engine = AdaptivePlanningEngine(outcome_store, routing_store)
        _inspector = ReflectiveMemoryInspector(
            outcome_store=outcome_store,
            routing_store=routing_store,
            planning_engine=planning_engine,
        )
    return _inspector


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class EntryResponse(BaseModel):
    entry_id: str
    tier: str
    source_store: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class SnapshotResponse(BaseModel):
    entries: list[EntryResponse] = Field(default_factory=list)
    total_outcomes: int = 0
    total_routing_feedback: int = 0
    total_heuristics: int = 0
    active_heuristics: int = 0
    generated_at: str | None = None


VALID_TIERS = {"outcome", "routing", "heuristic"}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_model=SnapshotResponse)
def inspect(
    tier: str | None = Query(None),
    workflow_name: str | None = Query(None),
    model_name: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    if tier and tier not in VALID_TIERS:
        raise_typed(
            PlatformError.reflective(
                "bad_request", f"Invalid tier: {tier!r}; must be one of {VALID_TIERS}"
            ),
        )
    inspector = get_inspector()
    snap = inspector.inspect(
        tier=tier,
        workflow_name=workflow_name,
        model_name=model_name,
        limit=limit,
    )
    return SnapshotResponse(**snap.to_dict())


@router.delete("/{tier}/{entry_id}")
def delete_entry(tier: str, entry_id: str):
    if tier not in VALID_TIERS:
        raise_typed(PlatformError.reflective("bad_request", f"Invalid tier: {tier!r}"))
    inspector = get_inspector()
    ok = inspector.delete_entry(entry_id, tier)
    if not ok:
        raise_typed(
            PlatformError.reflective(
                "not_found", "Entry not found or deletion not supported"
            ),
        )
    return {"status": "deleted", "entry_id": entry_id, "tier": tier}


@router.post("/{tier}/reset")
def reset_tier(tier: str):
    if tier not in VALID_TIERS:
        raise_typed(PlatformError.reflective("bad_request", f"Invalid tier: {tier!r}"))
    inspector = get_inspector()
    count = inspector.reset_tier(tier)
    return {"status": "reset", "tier": tier, "entries_affected": count}
