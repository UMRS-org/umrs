"""
Reconciler - Boot-time crash recovery (U-713, U-721-U-723).

On UMRS startup:
1. Generate new umrs_instance_id
2. Find all attempts with status IN ('SUBMITTED', 'RUNNING')
   where umrs_instance_id != current (stale from previous instance)
3. Mark them LOST
4. Requeue parent jobs (if retries remain and policy allows)

NO backend cleanup (scancel/delete) - rely on attempt-scoped isolation.
"""

from __future__ import annotations

import logging
from datetime import datetime

from .models import AttemptStatus, Job, JobStatus, JobType
from .store import JobStore
from umrs.metrics import METRICS

logger = logging.getLogger(__name__)


class ReconcileResult:
    """Result of reconciliation process."""

    def __init__(self):
        self.lost_attempts: list[str] = []
        self.requeued_jobs: list[str] = []
        self.failed_jobs: list[str] = []  # No more retries
        self.model_download_recoveries: list[str] = []
        self.errors: list[tuple[str, str]] = []  # (id, error_message)

    @property
    def total_processed(self) -> int:
        return len(self.lost_attempts)

    def __str__(self) -> str:
        return (
            f"ReconcileResult(lost={len(self.lost_attempts)}, "
            f"requeued={len(self.requeued_jobs)}, "
            f"failed={len(self.failed_jobs)}, "
            f"model_download_recoveries={len(self.model_download_recoveries)}, "
            f"errors={len(self.errors)})"
        )


class Reconciler:
    """
    Boot-time crash recovery.

    Marks in-flight attempts as LOST and requeues their parent jobs
    if retries remain. Does NOT perform any backend cleanup.
    """

    def __init__(self, job_store: JobStore):
        """
        Initialize Reconciler.

        Args:
            job_store: JobStore instance (provides umrs_instance_id)
        """
        self.job_store = job_store
        self.umrs_instance_id = job_store.umrs_instance_id

    def reconcile_on_boot(self) -> ReconcileResult:
        """
        Perform boot-time reconciliation.

        This should be called once at UMRS startup, before the scheduler
        starts processing new jobs.

        Returns:
            ReconcileResult with details of what was processed
        """
        import time as _time

        _start = _time.monotonic()
        result = ReconcileResult()

        logger.info(f"Starting boot reconciliation (instance={self.umrs_instance_id})")

        # Find stale attempts (from previous UMRS instance or crashed state)
        stale_attempts = self._find_stale_attempts()

        if not stale_attempts:
            logger.info("No stale attempts found - clean startup")
            METRICS.reconciler_run("ok")
            dur = _time.monotonic() - _start
            METRICS.reconciler_duration(dur, "ok")
            METRICS.reconciler_latency(dur, "ok")
            METRICS.set_reconciler_queue_depth(0)
            return result

        logger.warning(f"Found {len(stale_attempts)} stale attempts to reconcile")
        METRICS.reconciler_backlog(len(stale_attempts))
        METRICS.set_reconciler_queue_depth(len(stale_attempts))

        # Process each stale attempt
        for attempt_id, job_id, attempt_no, old_instance_id in stale_attempts:
            try:
                self._process_stale_attempt(
                    attempt_id, job_id, attempt_no, old_instance_id, result
                )
            except Exception as e:
                logger.error(f"Error reconciling attempt {attempt_id}: {e}")
                result.errors.append((attempt_id, str(e)))

        logger.info(f"Boot reconciliation complete: {result}")
        outcome = "error" if result.errors else "ok"
        METRICS.reconciler_run(outcome)
        dur = _time.monotonic() - _start
        METRICS.reconciler_duration(dur, outcome)
        METRICS.reconciler_latency(dur, outcome)
        return result

    def _find_stale_attempts(self) -> list[tuple[str, str, int, str | None]]:
        """
        Find attempts that are stale (from previous UMRS instance).

        Returns list of (attempt_id, job_id, attempt_no, umrs_instance_id)
        """
        conn = self.job_store.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Find all in-flight attempts that either:
                # - Have a different umrs_instance_id (from previous instance)
                # - Have NULL umrs_instance_id (legacy or failed to set)
                cur.execute(
                    """
                    SELECT
                        attempt_id,
                        job_id,
                        attempt_no,
                        umrs_instance_id
                    FROM job_attempts
                    WHERE status IN (%s, %s)
                      AND (umrs_instance_id IS NULL OR umrs_instance_id != %s)
                    ORDER BY created_at ASC
                """,
                    (
                        AttemptStatus.SUBMITTED.value,
                        AttemptStatus.RUNNING.value,
                        self.umrs_instance_id,
                    ),
                )

                return cur.fetchall()
        finally:
            self.job_store.db_pool.putconn(conn)

    def _process_stale_attempt(
        self,
        attempt_id: str,
        job_id: str,
        attempt_no: int,
        old_instance_id: str | None,
        result: ReconcileResult,
    ) -> None:
        """
        Process a single stale attempt.

        1. Mark attempt as LOST
        2. Check if job can be retried
        3. Requeue job if allowed
        """
        logger.info(
            f"Processing stale attempt {attempt_id} "
            f"(job={job_id}, no={attempt_no}, old_instance={old_instance_id})"
        )

        # Mark attempt as LOST
        ignore_reason = f"umrs_restart_no_cleanup (prev={old_instance_id})"
        self.job_store.mark_attempt_lost(attempt_id, ignore_reason)
        result.lost_attempts.append(attempt_id)
        METRICS.job_lost("reconciler")
        METRICS.reconciler_items("lost")

        # Get job to check retry policy
        job = self.job_store.get_job(job_id)
        if job.job_type == JobType.MODEL_DOWNLOAD:
            self._record_model_download_recovery(job, attempt_id, result)

        # Only process if job is still in RUNNING status
        # (could have been cancelled or completed by other means)
        if job.status != JobStatus.RUNNING:
            logger.debug(f"Job {job_id} not in RUNNING state, skipping requeue")
            return

        # Check if we should retry
        if job.can_retry(attempt_no):
            # Calculate backoff delay
            next_run = datetime.utcnow()  # Run immediately on crash recovery

            # Requeue the job
            self._requeue_job(job_id, next_run)
            result.requeued_jobs.append(job_id)
            METRICS.retry_attempt("reconciler")
            METRICS.reconciler_items("requeued")
            logger.info(f"Requeued job {job_id} (attempt {attempt_no} was lost)")
        else:
            # No more retries - mark job as failed
            self._fail_job(
                job_id, f"exhausted retries after crash (last attempt: {attempt_no})"
            )
            result.failed_jobs.append(job_id)
            METRICS.job_failed("reconciler")
            METRICS.reconciler_items("failed_exhausted")
            logger.warning(f"Job {job_id} failed - no more retries after crash")

    def _record_model_download_recovery(
        self,
        job: Job,
        attempt_id: str,
        result: ReconcileResult,
    ) -> None:
        """Record explicit restart policy for partial model downloads."""
        source = job.spec.get("source", "huggingface").lower()
        recovery_mode = (
            "resume_on_retry" if source == "huggingface" else "retry_in_place"
        )
        result.model_download_recoveries.append(f"{attempt_id}:{recovery_mode}")
        logger.info(
            "Model download attempt %s lost; recovery=%s source=%s target=%s",
            attempt_id,
            recovery_mode,
            source,
            job.spec.get("target_path", "<missing>"),
        )

    def _requeue_job(self, job_id: str, next_run: datetime) -> None:
        """
        Requeue a job for scheduling.

        Transitions: RUNNING → QUEUED
        """
        conn = self.job_store.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs SET
                        status = %s,
                        active_attempt_id = NULL,
                        next_run_at = %s,
                        last_state_change_at = NOW()
                    WHERE job_id = %s
                """,
                    (JobStatus.QUEUED.value, next_run, job_id),
                )
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.job_store.db_pool.putconn(conn)

    def _fail_job(self, job_id: str, reason: str) -> None:
        """
        Mark a job as failed (no more retries).

        Transitions: RUNNING → FAILED
        """
        conn = self.job_store.db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE jobs SET
                        status = %s,
                        active_attempt_id = NULL,
                        last_state_change_at = NOW()
                    WHERE job_id = %s
                """,
                    (JobStatus.FAILED.value, job_id),
                )
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.job_store.db_pool.putconn(conn)


def reconcile_on_startup(job_store: JobStore) -> ReconcileResult:
    """
    Convenience function for boot-time reconciliation.

    Call this once at UMRS startup before starting the scheduler.

    Example:
        job_store = JobStore(db_pool)
        result = reconcile_on_startup(job_store)
        logger.info(f"Reconciled: {result}")
        # Now safe to start scheduler
    """
    reconciler = Reconciler(job_store)
    return reconciler.reconcile_on_boot()
