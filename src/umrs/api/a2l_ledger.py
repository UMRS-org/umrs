"""A2L conservation-ledger API (ADR-034).

Exposes the UMRS-bounded a2l conservation domain at ``/umrs/a2l/ledger/*`` —
a NEW prefix so the standalone a2l service (proxied at ``/gw/a2l/*``) keeps
serving during the parallel-run phase.

Read models are compositional: ``GET .../jobs/{job_id}/view`` joins the
authoritative jobs table with a2l's conserved rows at query time and never
materializes a copy of execution state.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from umrs.a2l import (
    AgentExecutionCreate,
    AgentExecutionUpdate,
    ArtifactDerive,
    DecisionCreate,
)
from umrs.a2l.store import ConservationStore, init_store

from ._asyncpg_pool import knowledge_asyncpg_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/a2l/ledger", tags=["a2l-ledger"])


# ── Store dependency (lazy app-cached pool, knowledge parity) ─────────────


async def _get_store(request: Request) -> ConservationStore:
    cached = getattr(request.app.state, "a2l_conservation_store", None)
    if cached is not None:
        return cached
    pool = await knowledge_asyncpg_pool(request)
    store = init_store(pool)
    request.app.state.a2l_conservation_store = store
    return store


# ── Request bodies (pydantic mirrors of the contracts DTOs) ───────────────


class DeriveBody(BaseModel):
    parents: list[str] = Field(..., min_length=1)
    operation: str
    type: str
    uri: str
    user_id: str
    input_data_hash: str
    params: dict[str, Any] = Field(default_factory=dict)
    type_schema: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    project_id: Optional[str] = None
    artifact_class: Optional[str] = None
    job_id: str = ""
    attempt_id: str = ""


class ExecutionCreateBody(BaseModel):
    agent_name: str
    user_id: str
    agent_version: str = ""
    model_ref: str = ""
    prompt_hash: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    project_id: Optional[str] = None
    job_id: str = ""
    attempt_id: str = ""


class ExecutionPatchBody(BaseModel):
    status: Optional[str] = None
    execution_trace: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None


class DecisionBody(BaseModel):
    decided_by: str
    decision: str
    rationale: str = ""
    job_id: str = ""
    attempt_id: str = ""
    artifact_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Artifacts ─────────────────────────────────────────────────────────────


@router.post("/artifacts/derive", status_code=status.HTTP_201_CREATED)
async def derive_artifact(body: DeriveBody, store=Depends(_get_store)) -> dict[str, Any]:
    """Derive a conserved artifact from parents (lineage edges atomically)."""
    req = ArtifactDerive.from_dict(body.model_dump())
    try:
        record = await store.derive_artifact(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record.to_dict()


@router.get("/artifacts")
async def list_artifacts(
    job_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    store=Depends(_get_store),
) -> dict[str, Any]:
    items = await store.list_artifacts(job_id=job_id, user_id=user_id, limit=limit, offset=offset)
    return {"artifacts": [a.to_dict() for a in items], "limit": limit, "offset": offset}


@router.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str, store=Depends(_get_store)) -> dict[str, Any]:
    record = await store.get_artifact(artifact_id)
    if record is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return record.to_dict()


@router.patch("/artifacts/{artifact_id}/status")
async def patch_artifact_status(
    artifact_id: str, body: dict[str, Any], store=Depends(_get_store)
) -> dict[str, Any]:
    new_status = str(body.get("status", ""))
    if not new_status:
        raise HTTPException(status_code=400, detail="status is required")
    try:
        record = await store.update_artifact_status(
            artifact_id, new_status, error=str(body.get("error") or "")
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return record.to_dict()


# ── Agent executions ──────────────────────────────────────────────────────


@router.post("/agent-executions", status_code=status.HTTP_201_CREATED)
async def create_execution(body: ExecutionCreateBody, store=Depends(_get_store)) -> dict[str, Any]:
    req = AgentExecutionCreate.from_dict(body.model_dump())
    record = await store.create_execution(req)
    return record.to_dict()


@router.get("/agent-executions")
async def list_executions(
    job_id: Optional[str] = Query(None),
    attempt_id: Optional[str] = Query(None),
    agent_name: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    store=Depends(_get_store),
) -> dict[str, Any]:
    items = await store.list_executions(
        job_id=job_id, attempt_id=attempt_id, agent_name=agent_name,
        user_id=user_id, limit=limit, offset=offset,
    )
    return {"executions": [e.to_dict() for e in items], "limit": limit, "offset": offset}


@router.get("/agent-executions/{execution_id}")
async def get_execution(execution_id: str, store=Depends(_get_store)) -> dict[str, Any]:
    record = await store.get_execution(execution_id)
    if record is None:
        raise HTTPException(status_code=404, detail="execution not found")
    return record.to_dict()


@router.patch("/agent-executions/{execution_id}")
async def patch_execution(
    execution_id: str, body: ExecutionPatchBody, store=Depends(_get_store)
) -> dict[str, Any]:
    patch = AgentExecutionUpdate.from_dict(body.model_dump())
    try:
        record = await store.update_execution(execution_id, patch)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record.to_dict()


# ── Decisions (conserved human/operator additions) ────────────────────────


@router.post("/decisions", status_code=status.HTTP_201_CREATED)
async def create_decision(body: DecisionBody, store=Depends(_get_store)) -> dict[str, Any]:
    req = DecisionCreate(**body.model_dump())
    record = await store.create_decision(req)
    return {
        "id": record.id,
        "decided_by": record.decided_by,
        "decision": record.decision,
        "job_id": record.job_id,
        "attempt_id": record.attempt_id,
        "created_at": record.created_at,
    }


@router.get("/decisions")
async def list_decisions(
    job_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    store=Depends(_get_store),
) -> dict[str, Any]:
    items = await store.list_decisions(job_id=job_id, limit=limit)
    return {"decisions": [
        {
            "id": d.id, "decided_by": d.decided_by, "decision": d.decision,
            "rationale": d.rationale, "job_id": d.job_id,
            "attempt_id": d.attempt_id, "artifact_id": d.artifact_id,
            "payload": d.payload, "created_at": d.created_at,
        }
        for d in items
    ]}


# ── Compositional job view (join-not-copy read model) ─────────────────────


@router.get("/jobs/{job_id}/view")
async def job_view(job_id: str, store=Depends(_get_store)) -> dict[str, Any]:
    """Compositional history view for one job: spine state + conserved rows."""
    view = await store.job_conservation_view(job_id)
    if view["spine"] is None and not (view["artifacts"] or view["executions"] or view["decisions"]):
        raise HTTPException(status_code=404, detail=f"no conservation data for job {job_id}")
    return view
