"""
Scheduler - Job scheduling loop (U-712).

Pulls QUEUED jobs ready to run, acquires advisory lock, creates attempt,
and submits to the appropriate backend (Slurm/K8s/Compose).

The scheduler runs as an async background task.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Protocol

from .models import Job, JobAttempt, JobType
from .store import JobStore, JobStoreError
from umrs.metrics import METRICS

logger = logging.getLogger(__name__)


class OrchestratorBackend(Protocol):
    """Protocol for orchestrator backends."""

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        """
        Submit a job to the backend.

        Returns backend_ref (Slurm job ID, K8s Job name, etc.)
        """
        ...

    @property
    def backend_name(self) -> str:
        """Name of this backend (slurm, kubernetes, compose)."""
        ...


class Scheduler:
    """
    Job scheduler loop.

    Polls for QUEUED jobs and submits them to the appropriate backend.
    Uses advisory locks to prevent concurrent scheduling of the same job.
    """

    def __init__(
        self,
        job_store: JobStore,
        backend_factory: Callable[[JobType], OrchestratorBackend],
        poll_interval: float = 1.0,
        batch_size: int = 10,
    ):
        """
        Initialize Scheduler.

        Args:
            job_store: JobStore for state management
            backend_factory: Factory function to get backend for job type
            poll_interval: Seconds between poll cycles
            batch_size: Max jobs to claim per cycle
        """
        self.job_store = job_store
        self.backend_factory = backend_factory
        self.poll_interval = poll_interval
        self.batch_size = batch_size

        self._running = False
        self._task: asyncio.Task | None = None
        self._stats = {
            "cycles": 0,
            "jobs_scheduled": 0,
            "jobs_failed": 0,
            "last_cycle_at": None,
        }

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started")

    async def stop(self, timeout: float = 10.0) -> None:
        """Stop the scheduler loop."""
        if not self._running:
            return

        self._running = False

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("Scheduler stop timeout, cancelling")
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

        logger.info("Scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        logger.info("Scheduler loop starting")

        while self._running:
            try:
                t0 = time.monotonic()
                await self._poll_cycle()
                METRICS.scheduler_cycle_duration(time.monotonic() - t0)
            except Exception as e:
                logger.error(f"Scheduler cycle error: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

        logger.info("Scheduler loop exited")

    async def _poll_cycle(self) -> None:
        """
        One poll cycle.

        1. Claim ready QUEUED jobs
        2. For each job:
           a. Acquire advisory lock
           b. Create attempt
           c. Submit to backend
           d. Mark attempt submitted
        """
        self._stats["cycles"] += 1
        self._stats["last_cycle_at"] = datetime.utcnow().isoformat()

        # Claim jobs ready to run
        job_ids = self.job_store.claim_next_jobs(limit=self.batch_size)

        if not job_ids:
            return

        METRICS.scheduler_jobs_claimed(len(job_ids))
        logger.debug(f"Claimed {len(job_ids)} jobs for scheduling")

        # Process each job (sequential to respect advisory locks)
        for job_id in job_ids:
            await self._schedule_job(job_id)

    async def _schedule_job(self, job_id: str) -> None:
        """
        Schedule a single job.

        Creates an attempt and submits to backend.
        """
        try:
            # Get job details
            job = self.job_store.get_job(job_id)

            # Record scheduling delay (time from ready to actual schedule)
            if hasattr(job, "next_run_at") and job.next_run_at:
                now = datetime.now(timezone.utc)
                next_run = job.next_run_at
                if next_run.tzinfo is None:
                    next_run = next_run.replace(tzinfo=timezone.utc)
                delay = (now - next_run).total_seconds()
                if delay >= 0:
                    METRICS.scheduler_delay(delay)

            # Get appropriate backend
            try:
                backend = self.backend_factory(job.job_type)
            except Exception as e:
                logger.error(
                    f"backend_factory failed for job {job_id} "
                    f"(type={job.job_type.value}): {e}"
                )
                self._stats["jobs_failed"] += 1
                METRICS.scheduler_cycle("backend_factory_error")
                return

            # Create attempt (this acquires advisory lock)
            attempt_id = self.job_store.create_attempt(
                job_id=job_id,
                backend=backend.backend_name,
            )

            # Get attempt details
            attempt = self.job_store.get_attempt(attempt_id)

            logger.info(
                f"Scheduling job {job_id} (type={job.job_type.value}, "
                f"attempt={attempt_id}, backend={backend.backend_name})"
            )

            # Update status detail
            self.job_store.update_attempt_status_detail(
                attempt_id, "submitting to backend"
            )

            # Submit to backend
            try:
                backend_ref = await backend.submit_job(job, attempt)

                # Mark submitted
                self.job_store.mark_attempt_submitted(attempt_id, backend_ref)

                self._stats["jobs_scheduled"] += 1
                METRICS.job_submitted(backend.backend_name)
                self._consecutive_failures = 0
                METRICS.degradation_level("scheduler", 0)
                logger.info(
                    f"Job {job_id} submitted (attempt={attempt_id}, "
                    f"backend_ref={backend_ref})"
                )

            except Exception as e:
                # Backend submission failed - mark attempt as failed
                logger.error(f"Backend submission failed for {job_id}: {e}")
                self.job_store.mark_attempt_failed(
                    attempt_id,
                    exit_reason=f"backend_submission_failed: {e}",
                    retry_delay=30,  # Retry in 30 seconds
                )
                self._stats["jobs_failed"] += 1
                self._consecutive_failures = getattr(self, "_consecutive_failures", 0) + 1
                if self._consecutive_failures >= 3:
                    METRICS.retry_burst(backend.backend_name)
                    METRICS.degradation_level("scheduler", 1)
                    self._consecutive_failures = 0
                METRICS.scheduler_cycle("submission_error")

        except JobStoreError as e:
            logger.error(f"JobStore error scheduling {job_id}: {e}")
            self._stats["jobs_failed"] += 1
            METRICS.scheduler_cycle("store_error")
        except Exception as e:
            logger.error(f"Unexpected error scheduling {job_id}: {e}", exc_info=True)
            self._stats["jobs_failed"] += 1
            METRICS.scheduler_cycle("unexpected_error")

    @property
    def stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        return self._stats.copy()

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running


class SimpleBackendAdapter:
    """
    Adapter to wrap existing orchestrator backends.

    Bridges the gap between the new job tracking system and
    the existing orchestrator infrastructure.
    """

    def __init__(self, orchestrator, backend_name: str):
        """
        Initialize adapter.

        Args:
            orchestrator: Existing orchestrator backend instance
            backend_name: Name of this backend
        """
        self._orchestrator = orchestrator
        self._backend_name = backend_name

    @property
    def backend_name(self) -> str:
        return self._backend_name

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        """
        Submit job to backend.

        Adapts job spec to orchestrator's submit_job interface.
        """
        # Extract relevant spec fields based on job type
        if job.job_type == JobType.ENDPOINT_PROVISION:
            # Endpoint provisioning uses existing orchestrator
            model_path = job.spec.get("model_path", "/models/default")
            project_id = job.project_id

            # The orchestrator returns a JobInfo with job_id
            extra_kwargs = {}
            if job.spec.get("partition_override"):
                extra_kwargs["partition_override"] = job.spec["partition_override"]
            if job.spec.get("nodelist_hint"):
                extra_kwargs["nodelist_hint"] = job.spec["nodelist_hint"]

            job_info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._orchestrator.submit_job(
                    model_path=model_path,
                    project_id=project_id,
                    env_vars=job.spec.get("env_vars", {}),
                    **extra_kwargs,
                )
            )

            return job_info.job_id

        elif job.job_type == JobType.INDEX_BUILD:
            # Index build - placeholder for future implementation
            raise NotImplementedError("index_build not yet implemented")

        elif job.job_type == JobType.EMBEDDING_BATCH:
            # Embedding batch - placeholder
            raise NotImplementedError("embedding_batch not yet implemented")

        else:
            raise ValueError(f"Unknown job type: {job.job_type}")
