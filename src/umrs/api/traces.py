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
- POST /umrs/traces/plan-runs/{run_id}/attempts           Open a plan run's attempt
- POST /umrs/traces/plan-runs/{run_id}/records            Append one plan record
- GET  /umrs/traces/plan-runs/{run_id}                    The run: definition, attempts
- GET  /umrs/traces/plan-runs/{run_id}/records            The run's records
- POST /umrs/traces/plan-runs/{run_id}/attempts/{id}/finish  Close an attempt
(The server-venue verbs of a plan run, ``…/{run_id}/venue`` and
``…/{run_id}/cancel``, live in ``plan_run_venue.py`` under the same prefix.)

ADR-092 slice 8: a run hosted on the server venue is written (open, append,
finish) only by its own worker, through the rail, attested for the run's
recorded venue session; its owner reads it. A venue worker never writes a
client-hosted run.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from ..jobs import plan_run_records
from ..jobs.execution_trace_store import ExecutionTraceStore, TraceRecord
from ..jobs.store import JobStore
from ._rail_identity import attested_session
from .jobs import CallerIdentity, get_caller_identity, get_job_store
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/traces", tags=["execution-traces"])

# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------

_trace_store: ExecutionTraceStore | None = None


def get_trace_store() -> ExecutionTraceStore:
    global _trace_store
    if _trace_store is None:
        from ._asyncpg_pool import sync_store_dsn

        # H-10.1: UMRS_DB_URL alone left umrs-front traces in memory.
        _trace_store = ExecutionTraceStore(db_url=sync_store_dsn())
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
    ttl_seconds: int | None = Field(default=None, gt=0)


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
        raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
    if not caller.can_access_job(job):
        raise_typed(
            PlatformError.traces(
                "forbidden", "Not authorized to access traces for this job"
            )
        )


def _can_read_job(job_id: str, store: JobStore, caller: CallerIdentity) -> bool:
    from ..jobs.store import JobNotFoundError

    try:
        job = store.get_job(job_id)
    except JobNotFoundError:
        return False
    return caller.can_access_job(job)


def _readable_trace(
    trace_id: str,
    trace_store: ExecutionTraceStore,
    job_store: JobStore,
    caller: CallerIdentity,
) -> TraceRecord:
    """The trace, when the verified caller may read its job; otherwise 404.

    The same access as the job's own traces (owner or admin), answered as
    "not found" so a trace of another project or user is never revealed,
    not even its existence.
    """
    record = trace_store.get_trace(trace_id)
    if record is None or not _can_read_job(record.job_id, job_store, caller):
        raise_typed(PlatformError.traces("not_found", f"Trace not found: {trace_id}"))
    return record


def _readable(
    traces: list[TraceRecord], job_store: JobStore, caller: CallerIdentity
) -> list[TraceRecord]:
    """Only the traces of jobs the verified caller may read."""
    job_ids = sorted({t.job_id for t in traces})
    jobs = job_store.get_jobs_batch(job_ids) if job_ids else {}
    allowed = {jid for jid, job in jobs.items() if caller.can_access_job(job)}
    return [t for t in traces if t.job_id in allowed]


def _trace_list(traces: list[TraceRecord]) -> dict[str, Any]:
    return {"traces": [t.to_dict() for t in traces], "count": len(traces)}


def _write_failed(what: str, exc: Exception) -> None:
    """A trace store write failure, named; never acknowledged as written."""
    logger.warning("trace %s failed: %s", what, exc)
    raise_typed(
        PlatformError.jobs("unavailable", f"trace {what} failed: {type(exc).__name__}"),
        cause=exc,
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
    if req.parent_trace_id:
        _readable_trace(req.parent_trace_id, trace_store, job_store, caller)
    try:
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
    except Exception as exc:  # noqa: BLE001 — named, never acknowledged
        _write_failed("create", exc)
    return record.to_dict()


@router.get(
    "/{trace_id}",
    response_model=TraceResponse,
    summary="Get an execution trace",
)
def get_trace(
    trace_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Get a single execution trace by ID (the caller's jobs only)."""
    return _readable_trace(trace_id, trace_store, job_store, caller).to_dict()


@router.patch(
    "/{trace_id}",
    response_model=TraceResponse,
    summary="Update an execution trace",
)
def update_trace(
    trace_id: str,
    req: TraceUpdateRequest,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Update trace status, outputs, or metadata (the caller's jobs only)."""
    existing = _readable_trace(trace_id, trace_store, job_store, caller)
    try:
        record = _apply_update(trace_store, trace_id, req, existing)
    except Exception as exc:  # noqa: BLE001 — named, never acknowledged
        _write_failed("update", exc)
    if record is None:
        raise_typed(PlatformError.traces("not_found", f"Trace not found after update: {trace_id}"))
    return record.to_dict()


def _apply_update(
    trace_store: ExecutionTraceStore,
    trace_id: str,
    req: TraceUpdateRequest,
    existing: TraceRecord,
) -> TraceRecord | None:
    """The status transition (or field update) a PATCH asks for."""
    if req.status == "running":
        return trace_store.start_trace(trace_id)
    if req.status == "succeeded":
        return trace_store.complete_trace(
            trace_id,
            outputs=req.outputs,
            outputs_ref=req.outputs_ref,
            artifact_refs=req.artifact_refs,
            tool_calls=req.tool_calls,
            retrieval_refs=req.retrieval_refs,
            routing_decision=req.routing_decision,
        )
    if req.status == "failed":
        return trace_store.fail_trace(
            trace_id, error=req.error or "Unknown error", error_detail=req.error_detail
        )
    if req.status == "skipped":
        return trace_store.skip_trace(trace_id)
    return _apply_fields(trace_store, trace_id, req, existing)


def _apply_fields(
    trace_store: ExecutionTraceStore,
    trace_id: str,
    req: TraceUpdateRequest,
    existing: TraceRecord,
) -> TraceRecord | None:
    """No transition: a progress update, else the fields the PATCH sets."""
    if req.progress is not None:
        return trace_store.update_progress(trace_id, req.progress)
    fields = ("outputs_ref", "artifact_refs", "tool_calls", "retrieval_refs", "routing_decision")
    updates = {name: getattr(req, name) for name in fields if getattr(req, name) is not None}
    return trace_store._update_trace(trace_id, **updates) if updates else existing


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
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """The execution's traces of jobs the caller may read (no others, not counted)."""
    traces = trace_store.get_traces_for_execution(execution_id)
    return _trace_list(_readable(traces, job_store, caller))


@router.get(
    "/{trace_id}/lineage",
    response_model=TraceListResponse,
    summary="Get trace lineage (ancestor chain)",
)
def get_trace_lineage(
    trace_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """The ancestor chain (root first) of a trace the caller may read; an
    ancestor in a job the caller may not read is left out."""
    _readable_trace(trace_id, trace_store, job_store, caller)
    chain = trace_store.get_lineage(trace_id)
    return _trace_list(_readable(chain, job_store, caller))


@router.get(
    "/{trace_id}/children",
    response_model=TraceListResponse,
    summary="Get child traces",
)
def get_trace_children(
    trace_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Direct children of a trace the caller may read (readable jobs only)."""
    _readable_trace(trace_id, trace_store, job_store, caller)
    children = trace_store.get_children(trace_id)
    return _trace_list(_readable(children, job_store, caller))


@router.delete(
    "/expired",
    response_model=CleanupResponse,
    summary="Cleanup expired traces",
)
def cleanup_expired_traces(
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Remove execution traces that have passed their TTL (every user's: admin only)."""
    if not caller.is_admin:
        raise_typed(
            PlatformError.auth("admin_required", "Expired-trace cleanup is an admin operation")
        )
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
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Get the stored input snapshot for deterministic replay."""
    record = _readable_trace(trace_id, trace_store, job_store, caller)
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
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Get the stored output snapshot for a completed trace."""
    record = _readable_trace(trace_id, trace_store, job_store, caller)
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
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """The step's traces in jobs the caller may read (for replay comparison)."""
    traces = trace_store.find_traces_by_step(step_name, trace_status)
    return _trace_list(_readable(traces, job_store, caller))


# ---------------------------------------------------------------------------
# Team plan run records (ADR-092 slice 7, MD-4)
# ---------------------------------------------------------------------------
#
# A team plan runs in r4; its durable record is a client-hosted
# research_loop_job (one per run, one attempt per r4 attempt) plus one
# execution_traces row per portable journal record, keyed by the record's
# D-8 content key. Every route is the caller's own run only: the job is
# found by a key naming the caller's user in the caller's project, and
# access is checked as for every job's traces.


class PlanRunAttemptRequest(BaseModel):
    """Open the run's next attempt (creating its durable job on the first)."""

    attempt: int = Field(ge=1, le=100_000)
    definition: dict[str, Any] = Field(default_factory=dict)
    required: bool = False


class PlanRunRecordRequest(BaseModel):
    """One portable journal record, as its JSON text (key order kept)."""

    attempt_id: str
    record: str


class PlanRunFinishRequest(BaseModel):
    """The attempt's outcome: success | partial | failed | cancelled."""

    status: str
    error: str = ""
    summary: dict[str, Any] = Field(default_factory=dict)


def _plan_run_refusal(exc: plan_run_records.PlanRunRecordError) -> None:
    raise_typed(PlatformError.jobs(exc.code, exc.message), cause=exc)


def _writer_checked(job: Any, request: Request) -> Any:
    """The run's job, when this request may write its records (slice 8)."""
    try:
        plan_run_records.check_writer(job, attested_session(request.app.state, request.headers))
    except plan_run_records.PlanRunRecordError as exc:
        _plan_run_refusal(exc)
    return job


def _plan_run_job(run_id: str, store: JobStore, caller: CallerIdentity) -> Any:
    """The caller's run job; 404 when this caller has none by that id."""
    try:
        job = plan_run_records.find_run_job(store, caller.project_id, caller.user_id, run_id)
    except plan_run_records.PlanRunRecordError as exc:
        _plan_run_refusal(exc)
    if job is None:
        raise_typed(
            PlatformError.jobs(
                "not_found", f"No durable record of plan run {run_id} for this caller"
            )
        )
    if not caller.can_access_job(job):
        raise_typed(
            PlatformError.traces("forbidden", "Not authorized to access this plan run's records")
        )
    return job


def _check_opener(
    run_id: str, request: Request, store: JobStore, caller: CallerIdentity
) -> None:
    attested = attested_session(request.app.state, request.headers)
    try:
        job = plan_run_records.find_run_job(store, caller.project_id, caller.user_id, run_id)
        if job is not None:
            plan_run_records.check_writer(job, attested)
        elif attested:
            raise plan_run_records.PlanRunRecordError(
                "forbidden", f"a venue worker never creates a plan run's record ({run_id})"
            )
    except plan_run_records.PlanRunRecordError as exc:
        _plan_run_refusal(exc)


@router.post(
    "/plan-runs/{run_id}/attempts",
    status_code=status.HTTP_201_CREATED,
    summary="Open a team plan run's next attempt (ADR-092 slice 7)",
)
def open_plan_run_attempt(
    run_id: str,
    req: PlanRunAttemptRequest,
    request: Request,
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Create or find the run's client-hosted research_loop_job; open attempt N+1.

    A venue-hosted run's attempt is opened by its own worker only; a worker
    never creates a run (its record exists before its worker starts).
    # auth: verified
    """
    if caller.is_anonymous:
        raise_typed(PlatformError.auth("project_required", "A plan run record needs a project"))
    _check_opener(run_id, request, job_store, caller)
    try:
        opened = plan_run_records.open_attempt(
            job_store,
            project_id=caller.project_id,
            user_id=caller.user_id,
            run_id=run_id,
            attempt_label=req.attempt,
            definition=req.definition,
            required=req.required,
        )
    except plan_run_records.PlanRunRecordError as exc:
        _plan_run_refusal(exc)
    return {"run_id": run_id, **opened.to_dict()}


@router.post(
    "/plan-runs/{run_id}/records",
    status_code=status.HTTP_201_CREATED,
    summary="Append one record of a team plan run (ADR-092 slice 7)",
)
def append_plan_run_record(
    run_id: str,
    req: PlanRunRecordRequest,
    request: Request,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """One execution_traces row, keyed by the record's D-8 content key.
    # auth: verified
    """
    job = _writer_checked(_plan_run_job(run_id, job_store, caller), request)
    try:
        trace = plan_run_records.append_record(
            job_store, trace_store, job, attempt_id=req.attempt_id, record_text=req.record
        )
    except plan_run_records.PlanRunRecordError as exc:
        _plan_run_refusal(exc)
    except Exception as exc:  # noqa: BLE001 — a store failure is named, never acknowledged
        logger.warning("plan run %s: record write failed: %s", run_id, exc)
        raise_typed(
            PlatformError.jobs("unavailable", f"durable record write failed: {type(exc).__name__}"),
            cause=exc,
        )
    return {
        "run_id": run_id,
        "trace_id": trace.trace_id,
        "call_key": trace.inputs_hash,
        "outputs_hash": trace.outputs_hash,
    }


@router.get(
    "/plan-runs/{run_id}",
    summary="A team plan run's durable record: definition and attempts",
)
def get_plan_run(
    run_id: str,
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """The run's definition and attempts (the caller's own run).
    # auth: verified
    """
    job = _plan_run_job(run_id, job_store, caller)
    return plan_run_records.run_view(job_store, job)


@router.get(
    "/plan-runs/{run_id}/records",
    summary="A team plan run's records, in write order",
)
def list_plan_run_records(
    run_id: str,
    trace_store: ExecutionTraceStore = Depends(get_trace_store),
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """The run's records in write order, verbatim (the caller's own run).
    # auth: verified
    """
    job = _plan_run_job(run_id, job_store, caller)
    records = plan_run_records.list_records(trace_store, job)
    return {"run_id": run_id, "job_id": job.job_id, "records": records, "count": len(records)}


@router.post(
    "/plan-runs/{run_id}/attempts/{attempt_id}/finish",
    summary="Close a team plan run's attempt with its outcome",
)
def finish_plan_run_attempt(
    run_id: str,
    attempt_id: str,
    req: PlanRunFinishRequest,
    request: Request,
    job_store: JobStore = Depends(get_job_store),
    caller: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Close the attempt with the run's outcome (its writer only, slice 8).
    # auth: verified
    """
    job = _writer_checked(_plan_run_job(run_id, job_store, caller), request)
    try:
        closed = plan_run_records.finish_attempt(
            job_store,
            job,
            attempt_id=attempt_id,
            status=req.status,
            error=req.error,
            summary=req.summary,
        )
    except plan_run_records.PlanRunRecordError as exc:
        _plan_run_refusal(exc)
    return {"run_id": run_id, "attempt_id": attempt_id, **closed}

