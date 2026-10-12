"""A client-hosted team plan run's durable record (ADR-092 slice 7, MD-4).

A team plan runs in r4 (the agent harness is R4's, CONSTITUTION §1.3); its
durable record is UMRS's (§1.3, §6: one job store). MD-4 names the home:
a ``research_loop_job`` plus ``execution_traces`` keyed by input hash, with
no new table.

The run is one ``research_loop_job`` whose spec says ``hosting: client``:
the request-driven tier (as ``vision_infer``): ``RestartPolicy.NEVER``,
never enqueued, never hosted here (``validate_research_loop_spec`` refuses
it by name), its attempts opened and closed by the client's HTTP calls. Each
r4 attempt is one UMRS attempt (ADR-002): opening attempt N+1 supersedes an
attempt still active (its client is gone or moved, so at most one is active)
and reopens a finished job; a UMRS restart marks the active attempt LOST and,
NEVER retried, the job FAILED, and a write to a LOST or finished attempt is
refused (attempt-scoped writes). A cancelled job is never reopened.

Each portable journal record (an agent call, a human's gate answer, a
``--from-phase`` marker) is one ``execution_traces`` row of the run's job
and the attempt that wrote it: ``inputs_hash`` is the record's D-8 call
key, ``outputs_snapshot`` carries the record's JSON text verbatim and
``outputs_hash`` its sha256. Text, not a JSON object: JSONB reorders object
keys, and a replayed output must keep its key order (its readers' prompts
render it with ``|json`` and their keys hash the rendered prompt).

ADR-092 slice 8: a run hosted on the server venue is the same record with
``hosting: venue``: its frontend-venue worker opens and closes its attempts
through the rail, and only that worker writes it (``check_writer``: the
rail's attestation of the run's recorded venue session); its owner reads
it, resumes it or cancels it (``umrs.venues.plan_runs``).

Foundation module: no ``umrs.api`` or ``umrs.server`` imports.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .execution_trace_store import ExecutionTraceStore, TraceRecord
from .models import AttemptStatus, Job, JobStatus, JobType, RestartPolicy
from .store import (
    AttemptNotFoundError,
    DuplicateJobError,
    InvalidTransitionError,
    JobStore,
)

HOSTING_KEY = "hosting"
HOSTING_CLIENT = "client"
# ADR-092 slice 8: a run its frontend-venue worker hosts (the server venue).
HOSTING_VENUE = "venue"
PLAN_RUN_HOSTINGS = frozenset({HOSTING_CLIENT, HOSTING_VENUE})
BACKEND = "r4_local"
VENUE_BACKEND = "r4_venue"
REF_PREFIX = "research_loop_job|"
RECORD_FORMAT = "r4.plan.journal/1"
STEP_TYPES = {"agent": "plan_agent", "gate": "plan_gate", "invalidate": "plan_invalidate"}
PLAN_STEP_TYPES = frozenset(STEP_TYPES.values())
MAX_RECORD_BYTES = 512 * 1024
MAX_DEFINITION_BYTES = 256 * 1024
MAX_SUMMARY_BYTES = 16 * 1024
MAX_KEY_CHARS = 64  # execution_traces.inputs_hash is VARCHAR(64)
MAX_STEP_NAME_CHARS = 512
MAX_ERROR_CHARS = 2000
FINISH_STATUSES = frozenset({"success", "partial", "failed", "cancelled"})
_COMMITTED = frozenset({"success", "partial"})
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REOPENABLE = frozenset(
    {JobStatus.PLANNED, JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.FAILED}
)

CLIENT_HOSTED_REFUSAL = (
    "client_hosted_run: this research_loop_job records a team plan run that "
    "its client (r4) hosts (ADR-092 slice 7); UMRS never hosts or requeues it. "
    "Resume it from r4 (r4 team plan resume <run_id>)."
)


class PlanRunRecordError(Exception):
    """A refusal by name; ``code`` is a ``jobs.*`` typed-error name."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class OpenedAttempt:
    job_id: str
    attempt_id: str
    attempt_no: int
    created: bool
    superseded: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "attempt_no": self.attempt_no,
            "created": self.created,
            "superseded": list(self.superseded),
        }


def is_client_hosted(spec: Any) -> bool:
    """True for a research_loop_job its client hosts (a team plan run)."""
    return isinstance(spec, dict) and spec.get(HOSTING_KEY) == HOSTING_CLIENT


def is_venue_hosted(spec: Any) -> bool:
    """True for a plan run its frontend-venue worker hosts (slice 8)."""
    return isinstance(spec, dict) and spec.get(HOSTING_KEY) == HOSTING_VENUE


def is_plan_run_record(spec: Any) -> bool:
    """True for a team plan run's record, however hosted: UMRS never hosts it
    as a loop; its client or its venue worker opens and closes its attempts."""
    return isinstance(spec, dict) and spec.get(HOSTING_KEY) in PLAN_RUN_HOSTINGS


def idempotency_key(user_id: str, run_id: str) -> str:
    """The run's job key within its project: one job per (user, run id)."""
    digest = hashlib.sha256(f"{user_id}\0{run_id}".encode()).hexdigest()
    return f"plan_run:{digest[:48]}"


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise PlanRunRecordError(
            "bad_request",
            f"invalid plan run id {str(run_id)[:80]!r}: [A-Za-z0-9_.-], at most 128",
        )
    return run_id


def backend_ref(run_id: str, attempt_label: int, *, driver: str = "client") -> str:
    return f"{REF_PREFIX}plan_run={run_id}|driver={driver}|attempt={int(attempt_label)}"


def find_run_job(store: JobStore, project_id: str, user_id: str, run_id: str) -> Job | None:
    """The caller's run job, or None (never another user's: the key names the user)."""
    key = idempotency_key(user_id, check_run_id(run_id))
    job = store.get_job_by_idempotency_key(project_id, key)
    if job is not None and not is_plan_run_record(job.spec):
        raise PlanRunRecordError("conflict", f"job {job.job_id} is not a plan run record")
    return job


def open_attempt(
    store: JobStore,
    *,
    project_id: str,
    user_id: str,
    run_id: str,
    attempt_label: int,
    definition: Any,
    required: bool = False,
) -> OpenedAttempt:
    """Create (or find) the run's job, then open its next attempt, RUNNING.

    The definition (the plan, its params, the resolution manifest) is kept
    from the first open only: a job's spec is immutable (ADR-002).
    """
    job, created = _job_for(store, project_id, user_id, run_id, definition, required)
    if job.status not in _REOPENABLE:
        raise PlanRunRecordError(
            "conflict",
            f"plan_run_cancelled: the durable record of run {run_id} ({job.job_id}) is "
            f"{job.status.value}; start a new run",
        )
    venue = is_venue_hosted(job.spec)
    try:
        superseded = store.reopen_client_job(job.job_id)
        attempt_id = store.create_attempt(job.job_id, backend=VENUE_BACKEND if venue else BACKEND)
        store.mark_attempt_submitted(
            attempt_id, backend_ref(run_id, attempt_label, driver="venue" if venue else "client")
        )
        host = "venue-hosted" if venue else "client-hosted"
        store.mark_attempt_running(
            attempt_id, status_detail=f"{host} plan run, attempt-{int(attempt_label)}"
        )
    except InvalidTransitionError as exc:
        raise PlanRunRecordError(
            "conflict", f"attempt_not_opened: run {run_id} ({job.job_id}): {exc}"
        ) from exc
    attempt = store.get_attempt(attempt_id)
    return OpenedAttempt(job.job_id, attempt_id, attempt.attempt_no, created, tuple(superseded))


def _job_for(
    store: JobStore,
    project_id: str,
    user_id: str,
    run_id: str,
    definition: Any,
    required: bool,
) -> tuple[Job, bool]:
    job = find_run_job(store, project_id, user_id, run_id)
    if job is not None:
        return job, False
    spec = {
        HOSTING_KEY: HOSTING_CLIENT,
        "plan_run_id": run_id,
        "durable_required": bool(required),
        "definition": _bounded_definition(definition),
    }
    try:
        job_id = store.create_job(
            project_id,
            user_id,
            JobType.RESEARCH_LOOP,
            spec,
            idempotency_key=idempotency_key(user_id, run_id),
            max_attempts=1,
            restart_policy=RestartPolicy.NEVER,
        )
    except DuplicateJobError as exc:  # a concurrent first open won
        return store.get_job(exc.existing_job_id), False
    return store.get_job(job_id), True


def _bounded_definition(definition: Any) -> dict[str, Any]:
    if not isinstance(definition, dict):
        raise PlanRunRecordError("bad_request", "definition must be a JSON object")
    size = len(json.dumps(definition, default=str).encode())
    if size > MAX_DEFINITION_BYTES:
        raise PlanRunRecordError(
            "bad_request", f"definition_too_large: {size} bytes > {MAX_DEFINITION_BYTES}"
        )
    return definition


def active_attempt(store: JobStore, job: Job, attempt_id: str) -> None:
    """Refuse unless ``attempt_id`` is the job's active, RUNNING attempt."""
    try:
        attempt = store.get_attempt(attempt_id)
    except AttemptNotFoundError as exc:
        raise PlanRunRecordError("attempt_not_found", f"attempt {attempt_id} not found") from exc
    if attempt.job_id != job.job_id:
        raise PlanRunRecordError(
            "attempt_not_found", f"attempt {attempt_id} is not an attempt of {job.job_id}"
        )
    if attempt.status != AttemptStatus.RUNNING or job.active_attempt_id != attempt_id:
        raise PlanRunRecordError(
            "conflict",
            f"attempt_not_active: attempt {attempt_id} is {attempt.status.value} (lost on a "
            "UMRS restart, superseded by a later attempt, or finished); its records are "
            "refused (ADR-002 attempt-scoped writes)",
        )


def append_record(
    store: JobStore, traces: ExecutionTraceStore, job: Job, *, attempt_id: str, record_text: Any
) -> TraceRecord:
    """One portable journal record as one execution_traces row (D-8 key)."""
    record = parse_record(record_text)
    active_attempt(store, job, attempt_id)
    kind = str(record.get("kind") or "agent")
    ok = bool(record.get("ok")) or kind == "invalidate"
    model = str(record.get("model") or "")
    return traces.record_completed(
        job_id=job.job_id,
        attempt_id=attempt_id,
        step_name=str(record.get("node") or record.get("label") or kind)[:MAX_STEP_NAME_CHARS],
        step_type=STEP_TYPES[kind],
        inputs_hash=record["call_key"],
        status="succeeded" if ok else "failed",
        outputs_snapshot={"format": RECORD_FORMAT, "record": record_text},
        outputs_hash=hashlib.sha256(record_text.encode()).hexdigest(),
        model_ref={"name": model} if model else None,
        retrieval_refs=_retrieval_refs(record.get("retrieval")),
        error=None if ok else _record_error(record),
    )


def _retrieval_refs(retrieval: Any) -> list[dict[str, Any]]:
    """The record's retrieval refs: its object entries, or none."""
    if not isinstance(retrieval, list):
        return []
    return [r for r in retrieval if isinstance(r, dict)]


def _record_error(record: dict[str, Any]) -> str | None:
    """A failed record's reason, bounded; ``None`` when it named none."""
    return str(record.get("reason") or "")[:MAX_ERROR_CHARS] or None


def parse_record(record_text: Any) -> dict[str, Any]:
    """The record a client sent: a JSON object of a portable kind with a key."""
    record = _record_object(record_text)
    key = record.get("call_key")
    if not isinstance(key, str) or not key or len(key) > MAX_KEY_CHARS:
        raise PlanRunRecordError(
            "bad_request", f"record call_key must be a string of 1-{MAX_KEY_CHARS} characters"
        )
    kind = str(record.get("kind") or "agent")
    if kind not in STEP_TYPES:
        raise PlanRunRecordError(
            "bad_request", f"record kind {kind!r} is not portable ({sorted(STEP_TYPES)})"
        )
    return record


def _record_object(record_text: Any) -> dict[str, Any]:
    """The record text as a bounded JSON object, refused by name otherwise."""
    if not isinstance(record_text, str):
        raise PlanRunRecordError("bad_request", "record must be the record's JSON text")
    size = len(record_text.encode())
    if size > MAX_RECORD_BYTES:
        raise PlanRunRecordError(
            "bad_request", f"record_too_large: {size} bytes > {MAX_RECORD_BYTES}"
        )
    try:
        record = json.loads(record_text)
    except ValueError as exc:
        raise PlanRunRecordError("bad_request", f"record is not JSON: {exc}") from exc
    if not isinstance(record, dict):
        raise PlanRunRecordError("bad_request", "record must be a JSON object")
    return record


def list_records(traces: ExecutionTraceStore, job: Job) -> list[dict[str, Any]]:
    """The run's records in write order, each with its verbatim text."""
    rows = []
    for trace in traces.get_traces_for_job(job.job_id):
        snapshot = trace.outputs_snapshot or {}
        if trace.step_type not in PLAN_STEP_TYPES or snapshot.get("format") != RECORD_FORMAT:
            continue
        rows.append(
            {
                "trace_id": trace.trace_id,
                "attempt_id": trace.attempt_id,
                "call_key": trace.inputs_hash,
                "status": trace.status,
                "outputs_hash": trace.outputs_hash,
                "created_at": trace.created_at.isoformat() if trace.created_at else None,
                "record": snapshot.get("record"),
            }
        )
    return rows


def run_view(store: JobStore, job: Job) -> dict[str, Any]:
    """The run's job: its definition and its attempts (r4's labels included)."""
    spec = job.spec or {}
    view = {
        "run_id": spec.get("plan_run_id"),
        "job_id": job.job_id,
        "job_type": job.job_type.value,
        "status": job.status.value,
        "hosting": spec.get(HOSTING_KEY),
        "durable_required": bool(spec.get("durable_required")),
        "definition": spec.get("definition") or {},
        "active_attempt_id": job.active_attempt_id,
        "commit_attempt_id": job.commit_attempt_id,
        "attempts": [_attempt_row(a) for a in store.get_attempts_for_job(job.job_id)],
    }
    if is_venue_hosted(spec):
        view["venue_session_id"] = spec.get("venue_session_id")
        view["caps"] = spec.get("caps") or {}
    return view


def _attempt_row(attempt: Any) -> dict[str, Any]:
    ref = _parse_ref(attempt.backend_ref or "")
    label = ref.get("attempt", "")
    metrics = getattr(attempt, "metrics", None) or {}
    return {
        "attempt_id": attempt.attempt_id,
        "attempt_no": attempt.attempt_no,
        "label": int(label) if label.isdigit() else None,
        "status": attempt.status.value,
        "exit_reason": attempt.exit_reason,
        # The outcome its writer reported at finish (slice 8: a server-hosted
        # run's result reaches its client only through here).
        "plan_status": metrics.get("plan_status") if isinstance(metrics, dict) else None,
        "summary": metrics.get("summary") if isinstance(metrics, dict) else None,
    }


def _parse_ref(ref: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in ref.split("|"):
        key, sep, value = part.partition("=")
        if sep:
            out[key] = value
    return out


def finish_attempt(
    store: JobStore, job: Job, *, attempt_id: str, status: str, error: str = "", summary: Any = None
) -> dict[str, Any]:
    """Close the attempt with the run's outcome; success and partial commit it."""
    if status not in FINISH_STATUSES:
        raise PlanRunRecordError(
            "bad_request", f"status must be one of {sorted(FINISH_STATUSES)}, not {status!r}"
        )
    active_attempt(store, job, attempt_id)
    metrics = {"plan_status": status, "summary": _bounded_summary(summary)}
    if status in _COMMITTED:
        store.mark_attempt_succeeded(attempt_id, commit=True, metrics=metrics)
    else:
        store.update_attempt_metrics(attempt_id, metrics)
        reason = f"plan_{status}: {error or status}"[:MAX_ERROR_CHARS]
        store.mark_attempt_failed(attempt_id, exit_reason=reason)
    return {"job_id": job.job_id, "job_status": store.get_job(job.job_id).status.value}


def _bounded_summary(summary: Any) -> dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    text = json.dumps(summary, default=str)
    if len(text.encode()) > MAX_SUMMARY_BYTES:
        return {"truncated": True}
    return summary


# ── slice 8: the server venue ───────────────────────────────────────────────

VENUE_SESSION_PREFIX = "planrun-"
VENUE_HOSTED_WRITE_REFUSAL = (
    "venue_hosted_run: run {run} is hosted by its server-venue worker (ADR-092 slice 8); "
    "only that worker writes its records. Read it, resume it or cancel it through the "
    "venue routes (r4 team plan result|resume|cancel {run})."
)


def venue_session_id(user_id: str, run_id: str) -> str:
    """The frontend-venue session that hosts ``run_id`` for ``user_id``.

    One per (user, run): a resume starts a fresh worker under the same id
    (the previous one has ended), so the run's record names its worker's
    session from its first start, immutably (ADR-002).
    """
    digest = hashlib.sha256(f"{user_id}\0{check_run_id(run_id)}".encode()).hexdigest()
    return f"{VENUE_SESSION_PREFIX}{digest[:24]}"


def venue_run_job(
    store: JobStore,
    *,
    project_id: str,
    user_id: str,
    run_id: str,
    definition: Any,
    caps: dict[str, Any],
) -> tuple[Job, bool]:
    """The venue-hosted run's job: created on its first start, found on a resume.

    A run recorded as client-hosted is refused by name (its hosting is part
    of its immutable spec); so is a cancelled one. A first start needs the
    run's definition; a resume uses the recorded one.
    """
    job = find_run_job(store, project_id, user_id, run_id)
    if job is not None:
        return _resumable_venue_job(job, run_id), False
    if not isinstance(definition, dict) or not definition.get("plan"):
        raise PlanRunRecordError(
            "not_found",
            f"No durable record of plan run {run_id} for this caller; start it with its "
            "definition",
        )
    spec = {
        HOSTING_KEY: HOSTING_VENUE,
        "plan_run_id": run_id,
        "durable_required": True,
        "definition": _bounded_definition(definition),
        "venue_session_id": venue_session_id(user_id, run_id),
        "caps": dict(caps or {}),
    }
    try:
        job_id = store.create_job(
            project_id,
            user_id,
            JobType.RESEARCH_LOOP,
            spec,
            idempotency_key=idempotency_key(user_id, run_id),
            max_attempts=1,
            restart_policy=RestartPolicy.NEVER,
        )
    except DuplicateJobError as exc:  # a concurrent first start won
        return _resumable_venue_job(store.get_job(exc.existing_job_id), run_id), False
    return store.get_job(job_id), True


def _resumable_venue_job(job: Job, run_id: str) -> Job:
    if not is_venue_hosted(job.spec):
        raise PlanRunRecordError(
            "conflict",
            f"hosting_mismatch: run {run_id} ({job.job_id}) is recorded as client-hosted; "
            "its client resumes it (r4 team plan resume <run_id>). Start a new run on the "
            "server venue.",
        )
    if job.status not in _REOPENABLE:
        raise PlanRunRecordError(
            "conflict",
            f"plan_run_cancelled: the durable record of run {run_id} ({job.job_id}) is "
            f"{job.status.value}; start a new run",
        )
    return job


def check_writer(job: Job, venue_session: str | None) -> None:
    """Who may write a run's records (open, append, finish).

    A venue-hosted run: only its own worker, attested by the rail as the
    run's recorded venue session (never its owner's client, never another
    session's worker). A client-hosted run: never a venue worker.
    """
    spec = job.spec or {}
    run = str(spec.get("plan_run_id") or job.job_id)
    if is_venue_hosted(spec):
        if not venue_session or venue_session != spec.get("venue_session_id"):
            raise PlanRunRecordError("forbidden", VENUE_HOSTED_WRITE_REFUSAL.format(run=run))
        return
    if venue_session:
        raise PlanRunRecordError(
            "forbidden",
            f"client_hosted_run: run {run} is hosted by its client; a venue worker never "
            "writes its records",
        )


__all__ = [
    "BACKEND",
    "CLIENT_HOSTED_REFUSAL",
    "HOSTING_CLIENT",
    "HOSTING_VENUE",
    "PLAN_RUN_HOSTINGS",
    "OpenedAttempt",
    "PlanRunRecordError",
    "append_record",
    "backend_ref",
    "check_run_id",
    "check_writer",
    "find_run_job",
    "finish_attempt",
    "idempotency_key",
    "is_client_hosted",
    "is_plan_run_record",
    "is_venue_hosted",
    "list_records",
    "open_attempt",
    "parse_record",
    "run_view",
    "venue_run_job",
    "venue_session_id",
]
