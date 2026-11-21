"""Team plan run records over the execution-trace store (ADR-092 slice 7, MD-4).

U3 S5: moved out of core ``umrs.api.traces`` with the plan-run cluster; the
extension mounts ``router`` (same ``/umrs/traces`` prefix) after core's.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field
from umrs.api.jobs import CallerIdentity, get_caller_identity, get_job_store
from umrs.api.traces import get_trace_store
from umrs.api.typed_errors import PlatformError, raise_typed
from umrs.jobs.execution_trace_store import ExecutionTraceStore
from umrs.jobs.store import JobStore

from umrs_r4.api._rail_identity import attested_session
from umrs_r4.jobs import plan_run_records

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/traces", tags=["execution-traces"])


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
    if job is None or not caller.can_access_job(job):
        # Another user's run reads like an unknown id (existence hiding).
        raise_typed(
            PlatformError.jobs(
                "not_found", f"No durable record of plan run {run_id} for this caller"
            )
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

