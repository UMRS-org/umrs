"""Session crash recovery (MEM-A4).

On UMRS restart, recovers in-flight session state from persisted
``ExecutionTraceStore`` rows and re-links them to the durable
Job/Attempt model via the ``Reconciler``.

This module extends the existing reconciler with session-level
recovery: after the base reconciler marks stale attempts as LOST,
the session recoverer inspects ``execution_traces`` to build a
``RecoveredSession`` that the application layer can use to resume
or report what was in progress.

This module lives in ``umrs.jobs`` because it reads/writes durable state
tied to jobs and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .execution_trace_store import ExecutionTraceStore, TraceRecord
from .models import JobStatus
from .store import JobStore

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@dataclass
class RecoveredStep:
    """A single step recovered from execution traces."""

    trace_id: str
    step_name: str
    step_type: str
    status: str
    progress: float
    has_inputs: bool
    has_outputs: bool
    has_checkpoint: bool = False
    error: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "step_name": self.step_name,
            "step_type": self.step_type,
            "status": self.status,
            "progress": self.progress,
            "has_inputs": self.has_inputs,
            "has_outputs": self.has_outputs,
            "has_checkpoint": self.has_checkpoint,
            "error": self.error,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
        }


@dataclass
class RecoveredSession:
    """Aggregate recovery information for a single job/execution."""

    job_id: str
    execution_id: str | None
    job_status: str
    can_retry: bool
    total_steps: int
    completed_steps: int
    failed_steps: int
    running_steps: int
    last_completed_step: str | None = None
    first_failed_step: str | None = None
    resume_from_step: str | None = None
    steps: list[RecoveredStep] = field(default_factory=list)
    recovered_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "job_status": self.job_status,
            "can_retry": self.can_retry,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "running_steps": self.running_steps,
            "last_completed_step": self.last_completed_step,
            "first_failed_step": self.first_failed_step,
            "resume_from_step": self.resume_from_step,
            "steps": [s.to_dict() for s in self.steps],
            "recovered_at": _iso(self.recovered_at),
        }


@dataclass
class SessionRecoveryResult:
    """Result of session recovery across all stale jobs."""

    recovered_sessions: list[RecoveredSession] = field(default_factory=list)
    resumable_jobs: list[str] = field(default_factory=list)
    terminal_jobs: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total_recovered(self) -> int:
        return len(self.recovered_sessions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recovered_sessions": [s.to_dict() for s in self.recovered_sessions],
            "resumable_jobs": self.resumable_jobs,
            "terminal_jobs": self.terminal_jobs,
            "errors": [{"job_id": j, "error": e} for j, e in self.errors],
            "total_recovered": self.total_recovered,
        }


@dataclass
class _RecoveredStepSummary:
    completed: list[RecoveredStep]
    failed: list[RecoveredStep]
    running: list[RecoveredStep]
    last_completed_step: str | None
    first_failed_step: str | None
    resume_from_step: str | None


class SessionRecoverer:
    """Recovers in-flight session state from execution traces after crash.

    Should be called AFTER the base ``Reconciler.reconcile_on_boot()``
    has already marked stale attempts as LOST and requeued eligible jobs.
    """

    def __init__(
        self,
        job_store: JobStore,
        trace_store: ExecutionTraceStore,
        checkpoint_store: Any | None = None,
    ) -> None:
        self._job_store = job_store
        self._trace_store = trace_store
        self._checkpoint_store = checkpoint_store

    def recover_sessions(
        self, lost_job_ids: list[str] | None = None
    ) -> SessionRecoveryResult:
        """Recover session state for jobs that were in-flight during crash.

        Args:
            lost_job_ids: Specific job IDs to recover. If None, queries
                the job store for all jobs in SUBMITTED/RUNNING/QUEUED states.

        Returns:
            SessionRecoveryResult with per-job recovery details.
        """
        result = SessionRecoveryResult()

        if lost_job_ids is None:
            lost_job_ids = self._find_recoverable_job_ids()

        if not lost_job_ids:
            logger.info("No jobs to recover session state for")
            return result

        logger.info("Recovering session state for %d jobs", len(lost_job_ids))

        for job_id in lost_job_ids:
            try:
                session = self._recover_single_job(job_id)
                if session is not None:
                    result.recovered_sessions.append(session)
                    if session.can_retry:
                        result.resumable_jobs.append(job_id)
                    else:
                        result.terminal_jobs.append(job_id)
            except Exception as exc:
                logger.error("Error recovering session for job %s: %s", job_id, exc)
                result.errors.append((job_id, str(exc)))

        logger.info(
            "Session recovery complete: %d recovered (%d resumable, %d terminal, %d errors)",
            result.total_recovered,
            len(result.resumable_jobs),
            len(result.terminal_jobs),
            len(result.errors),
        )
        return result

    def _find_recoverable_job_ids(self) -> list[str]:
        """Find job IDs that might have in-flight session state."""
        job_ids = []
        # Look for jobs that were recently active (from reconciler's LOST marking)
        if self._job_store.db_enabled:
            try:
                with self._job_store._get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT DISTINCT j.job_id
                            FROM jobs j
                            JOIN attempts a ON a.job_id = j.job_id
                            WHERE a.status = 'LOST'
                            ORDER BY j.job_id
                            LIMIT 1000
                            """,
                        )
                        job_ids = [row[0] for row in cur.fetchall()]
            except Exception as exc:
                logger.warning("DB query for recoverable jobs failed: %s", exc)
        else:
            # In-memory fallback: check job store's in-memory state
            for job_id, job in getattr(self._job_store, "_jobs", {}).items():
                if job.status in (
                    JobStatus.SUBMITTED,
                    JobStatus.RUNNING,
                    JobStatus.QUEUED,
                ):
                    job_ids.append(job_id)
        return job_ids

    def _recover_single_job(self, job_id: str) -> RecoveredSession | None:
        """Recover session state for a single job."""
        traces = self._trace_store.get_traces_for_job(job_id)
        if not traces:
            return None

        recovered_steps = [self._trace_to_recovered_step(trace) for trace in traces]
        summary = _recovered_step_summary(recovered_steps)
        can_retry, job_status = self._job_retry_status(job_id)

        return RecoveredSession(
            job_id=job_id,
            execution_id=_execution_id_from_traces(traces),
            job_status=job_status,
            can_retry=can_retry,
            total_steps=len(recovered_steps),
            completed_steps=len(summary.completed),
            failed_steps=len(summary.failed),
            running_steps=len(summary.running),
            last_completed_step=summary.last_completed_step,
            first_failed_step=summary.first_failed_step,
            resume_from_step=summary.resume_from_step,
            steps=recovered_steps,
        )

    def _trace_to_recovered_step(self, trace: TraceRecord) -> RecoveredStep:
        return RecoveredStep(
            trace_id=trace.trace_id,
            step_name=trace.step_name,
            step_type=trace.step_type,
            status=trace.status,
            progress=trace.progress,
            has_inputs=trace.inputs_snapshot is not None
            or trace.inputs_ref is not None,
            has_outputs=trace.outputs_snapshot is not None
            or trace.outputs_ref is not None,
            has_checkpoint=self._has_checkpoint(trace),
            error=trace.error,
            started_at=trace.started_at,
            finished_at=trace.finished_at,
        )

    def _has_checkpoint(self, trace: TraceRecord) -> bool:
        if self._checkpoint_store is None:
            return False
        try:
            ckpt = self._checkpoint_store.latest_checkpoint(
                trace.attempt_id or "", trace.step_name
            )
            return ckpt is not None
        except Exception:
            return False

    def _job_retry_status(self, job_id: str) -> tuple[bool, str]:
        try:
            job = self._job_store.get_job(job_id)
        except Exception:
            return False, "unknown"
        if job is None:
            return False, "unknown"
        job_status = (
            job.status.value if hasattr(job.status, "value") else str(job.status)
        )
        return job.can_retry(), job_status


def _execution_id_from_traces(traces: list[TraceRecord]) -> str | None:
    return next((trace.execution_id for trace in traces if trace.execution_id), None)


def _recovered_step_summary(steps: list[RecoveredStep]) -> _RecoveredStepSummary:
    completed = _steps_with_status(steps, {"succeeded"})
    failed = _steps_with_status(steps, {"failed"})
    running = _steps_with_status(steps, {"running", "pending"})
    return _RecoveredStepSummary(
        completed=completed,
        failed=failed,
        running=running,
        last_completed_step=_last_step_name(completed),
        first_failed_step=_first_step_name(failed),
        resume_from_step=_resume_from_step(steps),
    )


def _steps_with_status(
    steps: list[RecoveredStep], statuses: set[str]
) -> list[RecoveredStep]:
    return [step for step in steps if step.status in statuses]


def _last_step_name(steps: list[RecoveredStep]) -> str | None:
    return steps[-1].step_name if steps else None


def _first_step_name(steps: list[RecoveredStep]) -> str | None:
    return steps[0].step_name if steps else None


def _resume_from_step(steps: list[RecoveredStep]) -> str | None:
    return next((step.step_name for step in steps if step.status != "succeeded"), None)
