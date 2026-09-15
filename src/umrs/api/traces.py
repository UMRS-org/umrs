"""Execution trace REST API (MEM-B1 / MEM-A1).

Provides:
- POST /umrs/traces                      Create a trace
- GET  /umrs/traces/{trace_id}           Get a single trace
- PATCH /umrs/traces/{trace_id}          Update trace status/outputs
- GET  /umrs/traces/job/{job_id}         List traces for a job
- GET  /umrs/traces/execution/{exec_id}  List traces for workflow execution
- GET  /umrs/traces/{trace_id}/lineage   Get ancestor chain
- GET  /umrs/traces/{trace_id}/children  Get child traces
- DELETE /umrs/traces/expired            Cleanup expired TTL traces
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..jobs.execution_trace_store import ExecutionTraceStore, TraceRecord
from ..jobs.store import JobStore
from .jobs import CallerIdentity, get_caller_identity, get_job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/traces", tags=["execution-traces"])

# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------

_trace_store: ExecutionTraceStore | None = None


def get_trace_store() -> ExecutionTraceStore:
    global _trace_store
    if _trace_store is None:
        db_url = os.getenv("UMRS_DB_URL", "").strip() or None
        _trace_store = ExecutionTraceStore(db_url=db_url)
    return _trace_store


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class TraceCreateRequest(BaseModel):
    """Request to create a new execution trace."""

    job_id: str
    step_name: str
    step_type: str = "generic"
    attempt_id: str | None = None
    execution_id: str | None = None
    parent_trace_id: str | None = None
    inputs: dict[str, Any] | None = None
    model_ref: dict[str, Any] | None = None
    ttl_seconds: int | None = None


class TraceUpdateRequest(BaseModel):
    """Request to update a trace (status transitions, outputs)."""

    status: str | None = None
    progress: float | None = None
    outputs: dict[str, Any] | None = None
    outputs_ref: str | None = None
    artifact_refs: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    retrieval_refs: list[dict[str, Any]] | None = None
    routing_decision: dict[str, Any] | None = None
    error: str | None = None
    error_detail: dict[str, Any] | None = None


class TraceResponse(BaseModel):
    """Single execution trace."""

    trace_id: str
    job_id: str
    step_name: str
    step_type: str = "generic"
    attempt_id: str | None = None
    execution_id: str | None = None
    parent_trace_id: str | None = None
    status: str = "pending"
    progress: float = 0.0
    inputs_hash: str | None = None
    outputs_hash: str | None = None
    inputs_ref: str | None = None
    outputs_ref: str | None = None
    model_ref: dict[str, Any] | None = None
    artifact_refs: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_refs: list[dict[str, Any]] = Field(default_factory=list)
    routing_decision: dict[str, Any] | None = None
    error: str | None = None
    error_detail: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None
    created_at: str | None = None
    ttl_expires_at: str | None = None


class TraceListResponse(BaseModel):
    """List of execution traces."""

    traces: list[TraceResponse]
    count: int


class CleanupResponse(BaseModel):
    """Result of TTL cleanup."""

    removed: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(record: TraceRecord) -> TraceResponse:
    d = record.to_dict()
    return TraceResponse(**d)


def _verify_job_access(
    job_id: str,
    store: JobStore,
    caller: CallerIdentity,
) -> None:
    """Verify caller can access the job's traces."""
    from ..jobs.store import JobNotFoundError

    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )
    if not caller.can_access_job(job):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access traces for this job",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=TraceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an execution trace",
)
def create_trace(
    req: TraceCreateRequest,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Create a new execution trace for a step within a job."""
    _verify_job_access(req.job_id, job_store, caller)

    record = trace_store.create_trace(
        job_id=req.job_id,
        step_name=req.step_name,
        step_type=req.step_type,
        attempt_id=req.attempt_id,
        execution_id=req.execution_id,
        parent_trace_id=req.parent_trace_id,
        inputs=req.inputs,
        model_ref=req.model_ref,
        ttl_seconds=req.ttl_seconds,
    )
    return record.to_dict()


@router.get(
    "/{trace_id}",
    response_model=TraceResponse,
    summary="Get an execution trace",
)
def get_trace(
    trace_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
) -> dict[str, Any]:
    """Get a single execution trace by ID."""
    record = trace_store.get_trace(trace_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace not found: {trace_id}",
        )
    return record.to_dict()


@router.patch(
    "/{trace_id}",
    response_model=TraceResponse,
    summary="Update an execution trace",
)
def update_trace(
    trace_id: str,
    req: TraceUpdateRequest,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
) -> dict[str, Any]:
    """Update trace status, outputs, or metadata."""
    existing = trace_store.get_trace(trace_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace not found: {trace_id}",
        )

    # Apply status transition
    if req.status == "running":
        record = trace_store.start_trace(trace_id)
    elif req.status == "succeeded":
        record = trace_store.complete_trace(
            trace_id,
            outputs=req.outputs,
            outputs_ref=req.outputs_ref,
            artifact_refs=req.artifact_refs,
            tool_calls=req.tool_calls,
            retrieval_refs=req.retrieval_refs,
            routing_decision=req.routing_decision,
        )
    elif req.status == "failed":
        record = trace_store.fail_trace(
            trace_id,
            error=req.error or "Unknown error",
            error_detail=req.error_detail,
        )
    elif req.status == "skipped":
        record = trace_store.skip_trace(trace_id)
    elif req.progress is not None:
        record = trace_store.update_progress(trace_id, req.progress)
    else:
        # Generic field updates
        updates: dict[str, Any] = {}
        for field_name in (
            "outputs_ref",
            "artifact_refs",
            "tool_calls",
            "retrieval_refs",
            "routing_decision",
        ):
            val = getattr(req, field_name, None)
            if val is not None:
                updates[field_name] = val
        record = trace_store._update_trace(trace_id, **updates) if updates else existing

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace not found after update: {trace_id}",
        )
    return record.to_dict()


@router.get(
    "/job/{job_id}",
    response_model=TraceListResponse,
    summary="List traces for a job",
)
def list_traces_for_job(
    job_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Get all execution traces for a specific job."""
    _verify_job_access(job_id, job_store, caller)
    traces = trace_store.get_traces_for_job(job_id)
    return {
        "traces": [t.to_dict() for t in traces],
        "count": len(traces),
    }


@router.get(
    "/execution/{execution_id}",
    response_model=TraceListResponse,
    summary="List traces for a workflow execution",
)
def list_traces_for_execution(
    execution_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
) -> dict[str, Any]:
    """Get all execution traces for a workflow execution ID."""
    traces = trace_store.get_traces_for_execution(execution_id)
    return {
        "traces": [t.to_dict() for t in traces],
        "count": len(traces),
    }


@router.get(
    "/{trace_id}/lineage",
    response_model=TraceListResponse,
    summary="Get trace lineage (ancestor chain)",
)
def get_trace_lineage(
    trace_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
) -> dict[str, Any]:
    """Get the full ancestor chain for a trace (root first)."""
    chain = trace_store.get_lineage(trace_id)
    if not chain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace not found: {trace_id}",
        )
    return {
        "traces": [t.to_dict() for t in chain],
        "count": len(chain),
    }


@router.get(
    "/{trace_id}/children",
    response_model=TraceListResponse,
    summary="Get child traces",
)
def get_trace_children(
    trace_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
) -> dict[str, Any]:
    """Get direct child traces."""
    children = trace_store.get_children(trace_id)
    return {
        "traces": [t.to_dict() for t in children],
        "count": len(children),
    }


@router.delete(
    "/expired",
    response_model=CleanupResponse,
    summary="Cleanup expired traces",
)
def cleanup_expired_traces(
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
) -> dict[str, Any]:
    """Remove execution traces that have passed their TTL."""
    count = trace_store.cleanup_expired()
    return {"removed": count}


# ---------------------------------------------------------------------------
# Replay endpoints (MEM-B5)
# ---------------------------------------------------------------------------


class InputSnapshotResponse(BaseModel):
    """Stored input snapshot for a trace."""

    trace_id: str
    inputs_hash: str | None = None
    inputs_snapshot: dict[str, Any] | None = None
    inputs_ref: str | None = None


class OutputSnapshotResponse(BaseModel):
    """Stored output snapshot for a trace."""

    trace_id: str
    outputs_hash: str | None = None
    outputs_snapshot: dict[str, Any] | None = None
    outputs_ref: str | None = None


@router.get(
    "/{trace_id}/inputs",
    response_model=InputSnapshotResponse,
    summary="Get input snapshot for replay",
)
def get_trace_inputs(
    trace_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
) -> dict[str, Any]:
    """Get the stored input snapshot for deterministic replay."""
    record = trace_store.get_trace(trace_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace not found: {trace_id}",
        )
    return {
        "trace_id": trace_id,
        "inputs_hash": record.inputs_hash,
        "inputs_snapshot": record.inputs_snapshot,
        "inputs_ref": record.inputs_ref,
    }


@router.get(
    "/{trace_id}/outputs",
    response_model=OutputSnapshotResponse,
    summary="Get output snapshot",
)
def get_trace_outputs(
    trace_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
) -> dict[str, Any]:
    """Get the stored output snapshot for a completed trace."""
    record = trace_store.get_trace(trace_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace not found: {trace_id}",
        )
    return {
        "trace_id": trace_id,
        "outputs_hash": record.outputs_hash,
        "outputs_snapshot": record.outputs_snapshot,
        "outputs_ref": record.outputs_ref,
    }


@router.get(
    "/step/{step_name}",
    response_model=TraceListResponse,
    summary="Find traces by step name",
)
def find_traces_by_step(
    step_name: str,
    trace_status: str | None = Query(None, alias="status"),
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
) -> dict[str, Any]:
    """Find all traces for a given step name (for replay comparison)."""
    traces = trace_store.find_traces_by_step(step_name, trace_status)
    return {
        "traces": [t.to_dict() for t in traces],
        "count": len(traces),
    }
