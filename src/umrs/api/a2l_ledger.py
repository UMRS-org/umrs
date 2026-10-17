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
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from umrs.a2l import (
    AgentExecutionCreate,
    AgentExecutionUpdate,
    ArtifactDerive,
    ArtifactRecord,
    DecisionCreate,
)
from umrs.a2l.store import ConservationStore, compute_artifact_id, init_store
from umrs.knowledge.access_scope import (
    SYSTEM_SPACE,
    CallerScope,
    personal_project,
    project_readable,
    project_writable,
)

from ._asyncpg_pool import knowledge_asyncpg_pool
from .jobs import CallerIdentity, get_caller_identity, get_optional_job_store
from .typed_errors import PlatformError, raise_typed

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


# ── Access (2026-10-09) ───────────────────────────────────────────────────
# These routes took no caller identity: on the UMRS port they answered an
# anonymous request, listed every user's rows, and wrote as whatever user_id
# a body named. Every route now runs on the verified identity: reads are
# limited to the caller's readable projects (knowledge.access_scope), writes
# are made as the caller, and a row the caller may not see answers 404.


def _readable_projects(identity: CallerIdentity) -> list[str] | None:
    """None = every project (admin); else the namespaces the caller reads."""
    if identity.is_admin:
        return None
    namespaces = (identity.project_id, SYSTEM_SPACE, personal_project(identity.user_id))
    return [p for p in namespaces if p]


def _visible(identity: CallerIdentity, project_id: str | None, *, write: bool = False) -> bool:
    caller = CallerScope.from_identity(identity)
    check = project_writable if write else project_readable
    return check(project_id or "", caller)


def _as_caller(
    identity: CallerIdentity, user_id: str, project_id: str | None
) -> tuple[str, str]:
    """``(user_id, project_id)`` for a write: the verified caller's, or an
    admin's explicit choice. A body naming another user is refused."""
    if not identity.is_admin and user_id and user_id != identity.user_id:
        raise_typed(PlatformError.a2l("forbidden", "user_id must be the verified caller"))
    project = project_id or identity.project_id
    if not _visible(identity, project, write=True):
        raise_typed(PlatformError.a2l("forbidden", f"project {project!r} is not writable"))
    return (user_id or identity.user_id), project


def _uri_hash(uri: str) -> str:
    """Content identity when the caller gives none: the URI's SHA-256."""
    import hashlib

    return hashlib.sha256(uri.encode("utf-8")).hexdigest()


def _job_for(identity: CallerIdentity, job_id: str) -> Any:
    """The job when the caller may read it; else 404 (never reveals it exists)."""
    store = get_optional_job_store()
    job = None
    if store is not None:
        try:
            job = store.get_job(job_id)
        except Exception:  # noqa: BLE001 — an unknown job reads as not found
            job = None
    if job is None or not identity.can_access_job(job):
        raise_typed(PlatformError.a2l("not_found", f"no conservation data for job {job_id}"))
    return job


# ── Request bodies (pydantic mirrors of the contracts DTOs) ───────────────


class CreateBody(BaseModel):
    """A root (ingested) artifact; ``r4 artifact create`` sends type/uri/metadata."""

    type: str
    uri: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    user_id: str = ""
    input_data_hash: str = ""
    artifact_class: str | None = None
    job_id: str = ""
    attempt_id: str = ""


class DeriveBody(BaseModel):
    # `r4 artifact derive` sends type/uri/metadata/parents: the rest defaults
    # (the caller; the URI's hash as the content identity).
    parents: list[str] = Field(..., min_length=1)
    operation: str = "derive"
    type: str
    uri: str
    user_id: str = ""
    input_data_hash: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    type_schema: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    artifact_class: str | None = None
    job_id: str = ""
    attempt_id: str = ""


class ExecutionCreateBody(BaseModel):
    agent_name: str
    user_id: str = ""  # the verified caller when omitted
    agent_version: str = ""
    model_ref: str = ""
    prompt_hash: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    project_id: str | None = None
    job_id: str = ""
    attempt_id: str = ""


class ExecutionPatchBody(BaseModel):
    status: str | None = None
    execution_trace: dict[str, Any] | None = None
    error: str | None = None
    completed_at: str | None = None


class DecisionBody(BaseModel):
    decided_by: str
    decision: str
    rationale: str = ""
    job_id: str = ""
    attempt_id: str = ""
    artifact_id: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Artifacts ─────────────────────────────────────────────────────────────


@router.get("/health")
async def ledger_health() -> dict[str, str]:
    """Liveness for `r4 artifact` (`/gw/a2l/health`)."""
    return {"status": "ok"}


@router.post("/artifacts", status_code=status.HTTP_201_CREATED)
async def create_artifact(
    body: CreateBody,
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Register a root artifact — admin/ingestion only; agents derive."""
    if not identity.is_admin:
        raise_typed(PlatformError.a2l("forbidden", "creating artifacts is admin-only; derive instead"))
    content = body.input_data_hash or _uri_hash(body.uri)
    record = ArtifactRecord(
        id=compute_artifact_id(content, "ingest", body.type),
        type=body.type,
        uri=body.uri,
        user_id=body.user_id or identity.user_id,
        metadata=body.metadata,
        project_id=body.project_id or identity.project_id,
        artifact_class=body.artifact_class,
        job_id=body.job_id,
        attempt_id=body.attempt_id,
    )
    try:
        record = await store.create_artifact(record)
    except ValueError as exc:
        raise_typed(PlatformError.a2l("bad_request", str(exc)), cause=exc)
    return record.to_dict()


@router.post("/artifacts/derive", status_code=status.HTTP_201_CREATED)
async def derive_artifact(
    body: DeriveBody,
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Derive a conserved artifact from parents (lineage edges atomically)."""
    user_id, project_id = _as_caller(identity, body.user_id, body.project_id)
    req = ArtifactDerive.from_dict(
        {
            **body.model_dump(),
            "user_id": user_id,
            "project_id": project_id,
            "input_data_hash": body.input_data_hash or _uri_hash(body.uri),
        }
    )
    try:
        record = await store.derive_artifact(req)
    except ValueError as exc:
        raise_typed(PlatformError.a2l("bad_request", str(exc)), cause=exc)
    return record.to_dict()


@router.get("/artifacts")
async def list_artifacts(
    job_id: str | None = Query(None),
    user_id: str | None = Query(None),
    project_id: str | None = Query(None),
    artifact_type: str | None = Query(None, alias="type"),
    artifact_status: str | None = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    readable = _readable_projects(identity)
    if project_id:
        if not _visible(identity, project_id):
            raise_typed(PlatformError.a2l("forbidden", f"project {project_id!r} is not readable"))
        readable = [project_id]
    items = await store.list_artifacts(
        job_id=job_id,
        user_id=user_id,
        artifact_type=artifact_type,
        status=artifact_status,
        readable_projects=readable,
        limit=limit,
        offset=offset,
    )
    return {"artifacts": [a.to_dict() for a in items], "limit": limit, "offset": offset}


async def _visible_artifact(
    store: Any, identity: CallerIdentity, artifact_id: str, *, write: bool = False
) -> Any:
    record = await store.get_artifact(artifact_id)
    if record is None or not _visible(identity, record.project_id, write=write):
        raise_typed(PlatformError.a2l("not_found", "artifact not found"))
    return record


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    return (await _visible_artifact(store, identity, artifact_id)).to_dict()


@router.get("/artifacts/{artifact_id}/lineage")
async def artifact_lineage(
    artifact_id: str,
    direction: str = Query("both", pattern="^(up|down|both)$"),
    max_depth: int = Query(3, ge=1, le=10),
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Ancestors and/or descendants over the derive edges; nodes the caller
    may not read are left out, with the edges that touch them."""
    await _visible_artifact(store, identity, artifact_id)
    graph = await store.lineage(artifact_id, direction=direction, max_depth=max_depth)
    nodes = [n for n in graph["nodes"] if _visible(identity, n.get("project_id"))]
    kept = {n["id"] for n in nodes}
    edges = [e for e in graph["edges"] if e["parent_id"] in kept and e["child_id"] in kept]
    return {"root_id": artifact_id, "nodes": nodes, "edges": edges}


@router.patch("/artifacts/{artifact_id}/status")
async def patch_artifact_status(
    artifact_id: str,
    body: dict[str, Any],
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    await _visible_artifact(store, identity, artifact_id, write=True)
    new_status = str(body.get("status", ""))
    if not new_status:
        raise_typed(PlatformError.a2l("bad_request", "status is required"))
    try:
        record = await store.update_artifact_status(
            artifact_id, new_status, error=str(body.get("error") or "")
        )
    except ValueError as exc:
        raise_typed(PlatformError.a2l("conflict", str(exc)), cause=exc)
    return record.to_dict()


# ── Agent executions ──────────────────────────────────────────────────────


@router.post("/agent-executions", status_code=status.HTTP_201_CREATED)
async def create_execution(
    body: ExecutionCreateBody,
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    user_id, project_id = _as_caller(identity, body.user_id, body.project_id)
    req = AgentExecutionCreate.from_dict(
        {**body.model_dump(), "user_id": user_id, "project_id": project_id}
    )
    record = await store.create_execution(req)
    return record.to_dict()


@router.get("/agent-executions")
async def list_executions(
    job_id: str | None = Query(None),
    attempt_id: str | None = Query(None),
    agent_name: str | None = Query(None),
    user_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    items = await store.list_executions(
        job_id=job_id,
        attempt_id=attempt_id,
        agent_name=agent_name,
        user_id=user_id,
        readable_projects=_readable_projects(identity),
        limit=limit,
        offset=offset,
    )
    return {
        "executions": [e.to_dict() for e in items],
        "limit": limit,
        "offset": offset,
    }


async def _visible_execution(
    store: Any, identity: CallerIdentity, execution_id: str, *, write: bool = False
) -> Any:
    record = await store.get_execution(execution_id)
    if record is None or not _visible(identity, record.project_id, write=write):
        raise_typed(PlatformError.a2l("not_found", "execution not found"))
    return record


@router.get("/agent-executions/{execution_id}")
async def get_execution(
    execution_id: str,
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    return (await _visible_execution(store, identity, execution_id)).to_dict()


@router.patch("/agent-executions/{execution_id}")
async def patch_execution(
    execution_id: str,
    body: ExecutionPatchBody,
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    await _visible_execution(store, identity, execution_id, write=True)
    patch = AgentExecutionUpdate.from_dict(body.model_dump())
    try:
        record = await store.update_execution(execution_id, patch)
    except ValueError as exc:
        raise_typed(PlatformError.a2l("not_found", str(exc)), cause=exc)
    return record.to_dict()


# ── Decisions (conserved human/operator additions) ────────────────────────


@router.post("/decisions", status_code=status.HTTP_201_CREATED)
async def create_decision(
    body: DecisionBody,
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    if not identity.is_admin and body.decided_by != identity.user_id:
        raise_typed(PlatformError.a2l("forbidden", "decided_by must be the verified caller"))
    if body.job_id:
        _job_for(identity, body.job_id)
    if body.artifact_id:
        # H-1 A.2 (2026-10-18): the job was checked, the artifact was not, so a
        # decision could point at another project's artifact. Same rule as the
        # artifact routes: one the caller may not read is not found.
        await _visible_artifact(store, identity, body.artifact_id)
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
    job_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    if job_id:
        _job_for(identity, job_id)
    elif not identity.is_admin:
        raise_typed(PlatformError.a2l("bad_request", "job_id is required"))
    items = await store.list_decisions(job_id=job_id, limit=limit)
    return {
        "decisions": [
            {
                "id": d.id,
                "decided_by": d.decided_by,
                "decision": d.decision,
                "rationale": d.rationale,
                "job_id": d.job_id,
                "attempt_id": d.attempt_id,
                "artifact_id": d.artifact_id,
                "payload": d.payload,
                "created_at": d.created_at,
            }
            for d in items
        ]
    }


# ── Compositional job view (join-not-copy read model) ─────────────────────


@router.get("/jobs/{job_id}/view")
async def job_view(
    job_id: str,
    store=Depends(_get_store),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Compositional history view for one job: spine state + conserved rows."""
    _job_for(identity, job_id)
    view = await store.job_conservation_view(job_id)
    if view["spine"] is None and not (
        view["artifacts"] or view["executions"] or view["decisions"]
    ):
        raise_typed(
            PlatformError.a2l("not_found", f"no conservation data for job {job_id}")
        )
    return view
