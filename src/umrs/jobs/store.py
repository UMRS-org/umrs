"""
JobStore - Single source of truth for job state transitions (U-711).

All status transitions go through this class - no raw SQL updates elsewhere.
This ensures consistent state machine behavior and audit logging.

Key methods:
- create_job(spec, idempotency_key) -> job_id
- enqueue_job(job_id, when=now)
- claim_next_jobs(limit) -> [job_id]
- create_attempt(job_id) -> attempt_id
- mark_attempt_submitted/running/succeeded/failed/lost()
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from .models import (
    AttemptStatus,
    Job,
    JobAttempt,
    JobStatus,
    JobType,
    RestartPolicy,
    compute_idempotency_key,
    generate_attempt_id,
    generate_job_id,
)
from .schema import ensure_schema

logger = logging.getLogger(__name__)


class JobStoreError(Exception):
    """Base exception for JobStore errors."""


class DuplicateJobError(JobStoreError):
    """Raised when idempotency key already exists."""

    def __init__(self, existing_job_id: str):
        self.existing_job_id = existing_job_id
        super().__init__(f"Job already exists: {existing_job_id}")


class JobNotFoundError(JobStoreError):
    """Raised when job is not found."""


class AttemptNotFoundError(JobStoreError):
    """Raised when attempt is not found."""


class InvalidTransitionError(JobStoreError):
    """Raised when state transition is invalid."""


class JobStore:
    """
    Durable job store with state machine enforcement.

    All state transitions go through this class to ensure consistency.
    Supports Postgres advisory locks for Invariant C (no concurrent attempts).
    """

    def __init__(self, db_pool, umrs_instance_id: str | None = None):
        """
        Initialize JobStore.

        Args:
            db_pool: psycopg2 connection pool
            umrs_instance_id: Unique ID for this UMRS instance (for crash detection)
        """
        self.db_pool = db_pool
        self.umrs_instance_id = umrs_instance_id or str(uuid.uuid4())
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure database schema exists."""
        conn = self.db_pool.getconn()
        try:
            ensure_schema(conn)
        finally:
            self.db_pool.putconn(conn)

    @staticmethod
    def _advisory_lock_key(job_id: str) -> int:
        """Return a stable int32 advisory-lock key for ``job_id``."""
        digest = hashlib.sha256(job_id.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], byteorder="big", signed=False) & 0x7FFFFFFF

    @contextmanager
    def _get_conn(self):
        """Get a connection from the pool."""
        conn = self.db_pool.getconn()
        try:
            yield conn
        finally:
            self.db_pool.putconn(conn)

    @contextmanager
    def _advisory_lock(self, conn, job_id: str):
        """
        Acquire advisory lock for a job (Invariant C).

        This prevents concurrent attempts on the same job,
        even across multiple UMRS instances.
        """
        lock_key = self._advisory_lock_key(job_id)

        with conn.cursor() as cur:
            # Try to acquire lock (non-blocking)
            cur.execute("SELECT pg_try_advisory_lock(%s)", (lock_key,))
            acquired = cur.fetchone()[0]

            if not acquired:
                raise JobStoreError(f"Could not acquire lock for job {job_id}")

            try:
                yield
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (lock_key,))

    # =========================================================================
    # Job Lifecycle Methods
    # =========================================================================

    def create_job(
        self,
        project_id: str,
        user_id: str,
        job_type: JobType,
        spec: dict[str, Any],
        idempotency_key: str | None = None,
        priority: int = 0,
        max_attempts: int = 3,
        restart_policy: RestartPolicy = RestartPolicy.ON_CRASH,
    ) -> str:
        """
        Create a new job.

        Args:
            project_id: Project/tenant identifier
            user_id: User who created the job
            job_type: Type of job
            spec: Job-specific parameters
            idempotency_key: Optional; computed if not provided
            priority: Higher = more urgent (default 0)
            max_attempts: Maximum retry attempts
            restart_policy: When to retry

        Returns:
            job_id of created job

        Raises:
            DuplicateJobError: If idempotency_key already exists for this project
        """
        job_id = generate_job_id()

        if idempotency_key is None:
            idempotency_key = compute_idempotency_key(
                project_id, user_id, job_type, spec
            )

        now = datetime.utcnow()

        with self._get_conn() as conn, conn.cursor() as cur:
            try:
                cur.execute("""
                        INSERT INTO jobs (
                            job_id, project_id, user_id, job_type, spec,
                            idempotency_key, status, priority,
                            desired_max_attempts, restart_policy,
                            created_at, updated_at
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s,
                            %s, %s
                        )
                    """, (
                    job_id, project_id, user_id, job_type.value, json.dumps(spec),
                    idempotency_key, JobStatus.PLANNED.value, priority,
                    max_attempts, restart_policy.value,
                    now, now,
                ))
                conn.commit()
                logger.info(f"Created job {job_id} (type={job_type.value})")
                return job_id

            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                # Find existing job
                cur.execute("""
                        SELECT job_id FROM jobs
                        WHERE project_id = %s AND idempotency_key = %s
                    """, (project_id, idempotency_key))
                row = cur.fetchone()
                existing_id = row[0] if row else "unknown"
                raise DuplicateJobError(existing_id) from None

    def get_job(self, job_id: str) -> Job:
        """Get job by ID."""
        with self._get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
            if not row:
                raise JobNotFoundError(f"Job not found: {job_id}")
            return self._row_to_job(row)

    def get_job_by_idempotency_key(
        self, project_id: str, idempotency_key: str
    ) -> Job | None:
        """Get job by idempotency key (for duplicate detection)."""
        with self._get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                    SELECT * FROM jobs
                    WHERE project_id = %s AND idempotency_key = %s
                """, (project_id, idempotency_key))
            row = cur.fetchone()
            return self._row_to_job(row) if row else None

    def enqueue_job(
        self, job_id: str, when: datetime | None = None
    ) -> None:
        """
        Enqueue a job for scheduling.

        Transitions: PLANNED → QUEUED
        """
        when = when or datetime.utcnow()
        now = datetime.utcnow()

        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                    UPDATE jobs SET
                        status = %s,
                        next_run_at = %s,
                        last_state_change_at = %s
                    WHERE job_id = %s AND status = %s
                    RETURNING job_id
                """, (
                JobStatus.QUEUED.value,
                when,
                now,
                job_id,
                JobStatus.PLANNED.value,
            ))

            if cur.fetchone() is None:
                conn.rollback()
                raise InvalidTransitionError(
                    f"Cannot enqueue job {job_id} (not in PLANNED status)"
                )
            conn.commit()
            logger.debug(f"Enqueued job {job_id} for {when}")

    def claim_next_jobs(self, limit: int = 10) -> list[str]:
        """
        Claim QUEUED jobs ready to run.

        Returns job_ids of jobs that are:
        - status = QUEUED
        - next_run_at <= now
        - ordered by priority DESC, next_run_at ASC
        """
        now = datetime.utcnow()

        with self._get_conn() as conn, conn.cursor() as cur:
            # Use FOR UPDATE SKIP LOCKED for concurrent schedulers
            cur.execute("""
                    SELECT job_id FROM jobs
                    WHERE status = %s AND next_run_at <= %s
                    ORDER BY priority DESC, next_run_at ASC
                    LIMIT %s
                    FOR UPDATE SKIP LOCKED
                """, (JobStatus.QUEUED.value, now, limit))

            return [row[0] for row in cur.fetchall()]

    def list_jobs(
        self,
        project_id: str | None = None,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs with optional filters."""
        conditions = []
        params = []

        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if status:
            conditions.append("status = %s")
            params.append(status.value)
        if job_type:
            conditions.append("job_type = %s")
            params.append(job_type.value)

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        with self._get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                    SELECT * FROM jobs
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                """, params + [limit, offset])

            return [self._row_to_job(row) for row in cur.fetchall()]

    # =========================================================================
    # Attempt Lifecycle Methods
    # =========================================================================

    def create_attempt(self, job_id: str, backend: str) -> str:
        """
        Create a new attempt for a job.

        Must hold advisory lock for the job.
        Transitions job: QUEUED → RUNNING
        """
        attempt_id = generate_attempt_id()
        now = datetime.utcnow()

        with self._get_conn() as conn, self._advisory_lock(conn, job_id):
            with conn.cursor() as cur:
                # Get current attempt count
                cur.execute("""
                        SELECT COALESCE(MAX(attempt_no), 0) FROM job_attempts
                        WHERE job_id = %s
                    """, (job_id,))
                max_attempt = cur.fetchone()[0]
                attempt_no = max_attempt + 1

                # Create attempt
                cur.execute("""
                        INSERT INTO job_attempts (
                            attempt_id, job_id, attempt_no, backend, status,
                            umrs_instance_id, scheduler_claimed_at, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s
                        )
                    """, (
                    attempt_id, job_id, attempt_no, backend,
                    AttemptStatus.CREATED.value,
                    self.umrs_instance_id, now, now,
                ))

                # Update job state
                cur.execute("""
                        UPDATE jobs SET
                            status = %s,
                            active_attempt_id = %s,
                            last_state_change_at = %s
                        WHERE job_id = %s
                    """, (
                    JobStatus.RUNNING.value,
                    attempt_id,
                    now,
                    job_id,
                ))

                conn.commit()
                logger.info(f"Created attempt {attempt_id} (job={job_id}, no={attempt_no})")
                return attempt_id

    def get_attempt(self, attempt_id: str) -> JobAttempt:
        """Get attempt by ID."""
        with self._get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM job_attempts WHERE attempt_id = %s",
                (attempt_id,)
            )
            row = cur.fetchone()
            if not row:
                raise AttemptNotFoundError(f"Attempt not found: {attempt_id}")
            return self._row_to_attempt(row)

    def get_attempts_for_job(self, job_id: str) -> list[JobAttempt]:
        """Get all attempts for a job, ordered by attempt_no."""
        with self._get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                    SELECT * FROM job_attempts
                    WHERE job_id = %s
                    ORDER BY attempt_no ASC
                """, (job_id,))
            return [self._row_to_attempt(row) for row in cur.fetchall()]

    def get_active_attempts(self) -> list[JobAttempt]:
        """Get all attempts in SUBMITTED or RUNNING status."""
        with self._get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                    SELECT * FROM job_attempts
                    WHERE status IN (%s, %s)
                    ORDER BY created_at ASC
                """, (AttemptStatus.SUBMITTED.value, AttemptStatus.RUNNING.value))
            return [self._row_to_attempt(row) for row in cur.fetchall()]

    def mark_attempt_submitted(
        self, attempt_id: str, backend_ref: str
    ) -> None:
        """
        Mark attempt as submitted to backend.

        Transitions: CREATED → SUBMITTED
        """
        now = datetime.utcnow()

        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                    UPDATE job_attempts SET
                        status = %s,
                        backend_ref = %s,
                        started_at = %s,
                        status_detail = 'submitted to backend'
                    WHERE attempt_id = %s AND status = %s
                    RETURNING attempt_id
                """, (
                AttemptStatus.SUBMITTED.value,
                backend_ref,
                now,
                attempt_id,
                AttemptStatus.CREATED.value,
            ))

            if cur.fetchone() is None:
                conn.rollback()
                raise InvalidTransitionError(
                    f"Cannot mark attempt {attempt_id} as submitted"
                )
            conn.commit()
            logger.debug(f"Attempt {attempt_id} submitted (ref={backend_ref})")

    def mark_attempt_running(
        self, attempt_id: str, status_detail: str | None = None
    ) -> None:
        """
        Mark attempt as running.

        Transitions: SUBMITTED → RUNNING
        """
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                    UPDATE job_attempts SET
                        status = %s,
                        status_detail = COALESCE(%s, 'running')
                    WHERE attempt_id = %s AND status = %s
                    RETURNING attempt_id
                """, (
                AttemptStatus.RUNNING.value,
                status_detail,
                attempt_id,
                AttemptStatus.SUBMITTED.value,
            ))

            if cur.fetchone() is None:
                conn.rollback()
                raise InvalidTransitionError(
                    f"Cannot mark attempt {attempt_id} as running"
                )
            conn.commit()

    def mark_attempt_succeeded(
        self,
        attempt_id: str,
        commit: bool = True,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        """
        Mark attempt as succeeded.

        Transitions: SUBMITTED/RUNNING → SUCCEEDED
        If commit=True, also sets commit_attempt_id on parent job.
        """
        now = datetime.utcnow()

        with self._get_conn() as conn, conn.cursor() as cur:
            # Update attempt
            cur.execute("""
                    UPDATE job_attempts SET
                        status = %s,
                        ended_at = %s,
                        exit_reason = 'success',
                        status_detail = 'completed successfully',
                        metrics = COALESCE(metrics, '{}')::jsonb || %s::jsonb
                    WHERE attempt_id = %s AND status IN (%s, %s)
                    RETURNING job_id
                """, (
                AttemptStatus.SUCCEEDED.value,
                now,
                json.dumps(metrics or {}),
                attempt_id,
                AttemptStatus.SUBMITTED.value,
                AttemptStatus.RUNNING.value,
            ))

            row = cur.fetchone()
            if row is None:
                conn.rollback()
                raise InvalidTransitionError(
                    f"Cannot mark attempt {attempt_id} as succeeded"
                )

            job_id = row[0]

            # Update job
            update_fields = ["status = %s", "active_attempt_id = NULL", "last_state_change_at = %s"]
            update_values = [JobStatus.COMPLETED.value, now]

            if commit:
                update_fields.append("commit_attempt_id = %s")
                update_values.append(attempt_id)

            update_values.append(job_id)

            cur.execute(f"""
                    UPDATE jobs SET {', '.join(update_fields)}
                    WHERE job_id = %s
                """, update_values)

            conn.commit()
            logger.info(f"Attempt {attempt_id} succeeded (commit={commit})")

    def mark_attempt_failed(
        self,
        attempt_id: str,
        exit_reason: str,
        retry_delay: int | None = None,
    ) -> bool:
        """
        Mark attempt as failed.

        Transitions: CREATED/SUBMITTED/RUNNING → FAILED

        Args:
            attempt_id: Attempt to mark failed
            exit_reason: Reason for failure
            retry_delay: If provided and retries available, requeue job

        Returns:
            True if job was requeued for retry
        """
        now = datetime.utcnow()
        requeued = False

        with self._get_conn() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Update attempt
            cur.execute("""
                    UPDATE job_attempts SET
                        status = %s,
                        ended_at = %s,
                        exit_reason = %s,
                        status_detail = 'failed'
                    WHERE attempt_id = %s AND status IN (%s, %s, %s)
                    RETURNING job_id, attempt_no
                """, (
                AttemptStatus.FAILED.value,
                now,
                exit_reason,
                attempt_id,
                AttemptStatus.CREATED.value,
                AttemptStatus.SUBMITTED.value,
                AttemptStatus.RUNNING.value,
            ))

            row = cur.fetchone()
            if row is None:
                conn.rollback()
                raise InvalidTransitionError(
                    f"Cannot mark attempt {attempt_id} as failed"
                )

            job_id = row["job_id"]
            attempt_no = row["attempt_no"]

            # Get job to check retry policy
            cur.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
            job_row = cur.fetchone()
            job = self._row_to_job(job_row)

            # Check if we should retry
            if job.can_retry(attempt_no):
                delay = retry_delay or job.get_backoff_delay(attempt_no)
                next_run = now + timedelta(seconds=delay)

                cur.execute("""
                        UPDATE jobs SET
                            status = %s,
                            active_attempt_id = NULL,
                            next_run_at = %s,
                            last_state_change_at = %s
                        WHERE job_id = %s
                    """, (JobStatus.QUEUED.value, next_run, now, job_id))

                requeued = True
                logger.info(
                    f"Attempt {attempt_id} failed, job {job_id} requeued for {next_run}"
                )
            else:
                # No more retries
                cur.execute("""
                        UPDATE jobs SET
                            status = %s,
                            active_attempt_id = NULL,
                            last_state_change_at = %s
                        WHERE job_id = %s
                    """, (JobStatus.FAILED.value, now, job_id))

                logger.warning(
                    f"Attempt {attempt_id} failed, job {job_id} exhausted retries"
                )

            conn.commit()
            return requeued

    def mark_attempt_lost(
        self,
        attempt_id: str,
        ignore_reason: str = "umrs_restart",
    ) -> None:
        """
        Mark attempt as LOST (UMRS crash recovery).

        Transitions: SUBMITTED/RUNNING → LOST
        Does NOT automatically requeue - caller must handle that.
        """
        now = datetime.utcnow()

        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                    UPDATE job_attempts SET
                        status = %s,
                        ended_at = %s,
                        exit_reason = 'lost on umrs restart',
                        ignored_at = %s,
                        ignore_reason = %s,
                        status_detail = 'marked lost during reconciliation'
                    WHERE attempt_id = %s AND status IN (%s, %s)
                    RETURNING job_id
                """, (
                AttemptStatus.LOST.value,
                now,
                now,
                ignore_reason,
                attempt_id,
                AttemptStatus.SUBMITTED.value,
                AttemptStatus.RUNNING.value,
            ))

            if cur.fetchone() is None:
                conn.rollback()
                # May have already been marked - not an error
                logger.debug(f"Attempt {attempt_id} already processed")
                return

            conn.commit()
            logger.info(f"Marked attempt {attempt_id} as LOST ({ignore_reason})")

    def update_attempt_status_detail(
        self, attempt_id: str, status_detail: str
    ) -> None:
        """Update status_detail for progress tracking."""
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                    UPDATE job_attempts SET status_detail = %s
                    WHERE attempt_id = %s
                """, (status_detail, attempt_id))
            conn.commit()

    def update_attempt_metrics(
        self, attempt_id: str, metrics: dict[str, Any]
    ) -> None:
        """Merge metrics into attempt record."""
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                    UPDATE job_attempts SET
                        metrics = COALESCE(metrics, '{}')::jsonb || %s::jsonb
                    WHERE attempt_id = %s
                """, (json.dumps(metrics), attempt_id))
            conn.commit()

    # =========================================================================
    # Job Control Methods
    # =========================================================================

    def cancel_job(self, job_id: str, reason: str = "user_cancelled") -> None:
        """
        Cancel a job (stop scheduling further attempts).

        Does NOT kill running backend jobs.
        """
        now = datetime.utcnow()

        with self._get_conn() as conn, conn.cursor() as cur:
            # Mark job as cancelled
            cur.execute("""
                    UPDATE jobs SET
                        status = %s,
                        active_attempt_id = NULL,
                        last_state_change_at = %s
                    WHERE job_id = %s AND status NOT IN (%s, %s, %s)
                    RETURNING job_id
                """, (
                JobStatus.CANCELLED.value,
                now,
                job_id,
                JobStatus.COMPLETED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            ))

            if cur.fetchone() is None:
                raise InvalidTransitionError(
                    f"Cannot cancel job {job_id} (already terminal)"
                )

            # Mark any active attempts as failed
            cur.execute("""
                    UPDATE job_attempts SET
                        status = %s,
                        ended_at = %s,
                        exit_reason = %s
                    WHERE job_id = %s AND status IN (%s, %s, %s)
                """, (
                AttemptStatus.FAILED.value,
                now,
                f"cancelled: {reason}",
                job_id,
                AttemptStatus.CREATED.value,
                AttemptStatus.SUBMITTED.value,
                AttemptStatus.RUNNING.value,
            ))

            conn.commit()
            logger.info(f"Cancelled job {job_id}: {reason}")

    def retry_job(self, job_id: str) -> None:
        """
        Force requeue a job (admin operation).

        Resets retry count and requeues immediately.
        """
        now = datetime.utcnow()

        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                    UPDATE jobs SET
                        status = %s,
                        active_attempt_id = NULL,
                        next_run_at = %s,
                        last_state_change_at = %s
                    WHERE job_id = %s AND status IN (%s, %s)
                    RETURNING job_id
                """, (
                JobStatus.QUEUED.value,
                now,
                now,
                job_id,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            ))

            if cur.fetchone() is None:
                raise InvalidTransitionError(
                    f"Cannot retry job {job_id} (not in terminal status)"
                )

            conn.commit()
            logger.info(f"Force-requeued job {job_id}")

    def update_heartbeat(self, job_id: str) -> None:
        """Update job heartbeat timestamp (for HA detection)."""
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute("""
                    UPDATE jobs SET last_heartbeat_at = NOW()
                    WHERE job_id = %s
                """, (job_id,))
            conn.commit()

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _row_to_job(self, row: dict) -> Job:
        """Convert database row to Job object."""
        return Job(
            job_id=row["job_id"],
            project_id=row["project_id"],
            user_id=row["user_id"],
            job_type=JobType(row["job_type"]),
            spec=row["spec"] if isinstance(row["spec"], dict) else json.loads(row["spec"]),
            idempotency_key=row["idempotency_key"],
            status=JobStatus(row["status"]),
            priority=row["priority"],
            desired_max_attempts=row["desired_max_attempts"],
            restart_policy=RestartPolicy(row["restart_policy"]),
            next_run_at=row["next_run_at"],
            active_attempt_id=row["active_attempt_id"],
            commit_attempt_id=row["commit_attempt_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_state_change_at=row.get("last_state_change_at"),
            last_heartbeat_at=row.get("last_heartbeat_at"),
        )

    def _row_to_attempt(self, row: dict) -> JobAttempt:
        """Convert database row to JobAttempt object."""
        metrics = row.get("metrics", {})
        if isinstance(metrics, str):
            metrics = json.loads(metrics)

        return JobAttempt(
            attempt_id=row["attempt_id"],
            job_id=row["job_id"],
            attempt_no=row["attempt_no"],
            backend=row["backend"],
            backend_ref=row.get("backend_ref"),
            status=AttemptStatus(row["status"]),
            started_at=row.get("started_at"),
            ended_at=row.get("ended_at"),
            exit_reason=row.get("exit_reason"),
            log_uri=row.get("log_uri"),
            metrics=metrics,
            umrs_instance_id=row.get("umrs_instance_id"),
            scheduler_claimed_at=row.get("scheduler_claimed_at"),
            status_detail=row.get("status_detail"),
            ignored_at=row.get("ignored_at"),
            ignore_reason=row.get("ignore_reason"),
            created_at=row["created_at"],
        )
