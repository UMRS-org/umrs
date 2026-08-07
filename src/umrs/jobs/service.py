"""
Job Service - Integration layer for durable job tracking.

This module provides the main entry point for using the job tracking system.
It wires together JobStore, Reconciler, Scheduler, and AttemptMonitor.

Usage:
    from umrs.jobs.service import JobService

    # Initialize
    job_service = JobService(db_pool)

    # On boot
    await job_service.startup()

    # Submit jobs
    job_id = job_service.submit_endpoint_provision(project_id, model_path)

    # On shutdown
    await job_service.shutdown()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from .models import (
    Job,
    JobAttempt,
    JobStatus,
    JobType,
    RestartPolicy,
)
from .monitor import AttemptMonitor, CompositeStatusChecker
from .reconciler import ReconcileResult, reconcile_on_startup
from .scheduler import Scheduler
from .store import JobStore

logger = logging.getLogger(__name__)

# Module-level singleton for health/probe access
_job_service_instance: JobService | None = None


def get_job_service() -> JobService | None:
    """Return the current JobService singleton, or None if not started."""
    return _job_service_instance


def set_job_service(service: JobService | None) -> None:
    """Set the module-level JobService singleton."""
    global _job_service_instance
    _job_service_instance = service


@dataclass(frozen=True)
class JobSubmission:
    """Parameters for creating a durable job."""

    project_id: str
    user_id: str
    job_type: JobType
    spec: dict[str, Any]
    priority: int = 0
    max_attempts: int = 3
    restart_policy: RestartPolicy = RestartPolicy.ON_CRASH
    idempotency_key: str | None = None
    enqueue: bool = True


class JobService:
    """
    High-level service for durable job tracking.

    Provides:
    - Boot-time reconciliation
    - Job submission APIs
    - Background scheduler and monitor
    - Integration with existing orchestrators
    """

    def __init__(
        self,
        db_pool,
        scheduler_interval: float = 1.0,
        monitor_interval: float = 5.0,
    ):
        """
        Initialize JobService.

        Args:
            db_pool: Postgres connection pool
            scheduler_interval: Scheduler poll interval in seconds
            monitor_interval: Monitor poll interval in seconds
        """
        self.db_pool = db_pool
        self.scheduler_interval = scheduler_interval
        self.monitor_interval = monitor_interval

        # Generate unique instance ID for this UMRS instance
        self.instance_id = str(uuid.uuid4())

        # Components (initialized during startup)
        self._job_store: JobStore | None = None
        self._scheduler: Scheduler | None = None
        self._monitor: AttemptMonitor | None = None

        # Backend factory (set by caller)
        self._backend_factory = None
        self._status_checker: CompositeStatusChecker | None = None

        self._started = False

    @property
    def job_store(self) -> JobStore:
        """Get the JobStore instance."""
        if self._job_store is None:
            raise RuntimeError("JobService not started")
        return self._job_store

    def set_backend_factory(self, factory) -> None:
        """
        Set the backend factory function.

        Args:
            factory: Callable that takes JobType and returns an OrchestratorBackend
        """
        self._backend_factory = factory

    def set_status_checker(self, checker: CompositeStatusChecker) -> None:
        """Set the status checker for monitoring."""
        self._status_checker = checker

    async def startup(self) -> ReconcileResult:
        """
        Start the job service.

        1. Initialize JobStore
        2. Run boot reconciliation
        3. Start scheduler and monitor loops

        Returns:
            ReconcileResult from boot reconciliation
        """
        if self._started:
            logger.warning("JobService already started")
            return ReconcileResult()

        logger.info(f"Starting JobService (instance={self.instance_id})")

        # Initialize JobStore
        self._job_store = JobStore(self.db_pool, self.instance_id)

        # Wire auto-ingest post-completion hooks (TASK-035).
        try:
            from .adapters import register_auto_ingest_callbacks

            register_auto_ingest_callbacks(self._job_store)
        except Exception as exc:
            logger.warning(
                "Auto-ingest hook registration skipped: %s",
                exc,
            )

        # Run boot reconciliation
        result = reconcile_on_startup(self._job_store)
        logger.info(f"Boot reconciliation: {result}")

        await self._recover_model_download_registry()

        # Start scheduler if backend factory is set
        if self._backend_factory:
            self._scheduler = Scheduler(
                job_store=self._job_store,
                backend_factory=self._backend_factory,
                poll_interval=self.scheduler_interval,
            )
            await self._scheduler.start()
        else:
            logger.warning("No backend factory set - scheduler not started")

        # Start monitor if status checker is set
        if self._status_checker:
            self._monitor = AttemptMonitor(
                job_store=self._job_store,
                status_checker=self._status_checker,
                poll_interval=self.monitor_interval,
            )
            await self._monitor.start()
        else:
            logger.warning("No status checker set - monitor not started")

        self._started = True
        logger.info("JobService started")

        return result

    async def _recover_model_download_registry(self) -> None:
        """Recover in-memory model download task contexts from durable jobs."""
        if self._job_store is None:
            return

        try:
            from .model_download import recover_download_registry_from_job_store

            recovered = await recover_download_registry_from_job_store(self._job_store)
            if recovered:
                logger.info("Recovered %d model download task context(s)", recovered)
        except Exception as exc:
            logger.warning("Model download task registry recovery skipped: %s", exc)

    async def shutdown(self, timeout: float = 10.0) -> None:
        """
        Shutdown the job service.

        Stops scheduler and monitor loops gracefully.
        """
        if not self._started:
            return

        logger.info("Shutting down JobService")

        # Stop scheduler
        if self._scheduler:
            await self._scheduler.stop(timeout)

        # Stop monitor
        if self._monitor:
            await self._monitor.stop(timeout)

        self._started = False
        logger.info("JobService shutdown complete")

    # =========================================================================
    # Job Submission APIs
    # =========================================================================

    def submit_job(
        self,
        request: JobSubmission | str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """
        Submit a new job.

        Args:
            request: Structured job submission. Existing keyword calls are
                accepted and converted into this request object.

        Returns:
            job_id of created job

        Raises:
            DuplicateJobError: If idempotency key already exists
        """
        submission = self._coerce_submission(request, args, kwargs)
        job_id = self.job_store.create_job(
            project_id=submission.project_id,
            user_id=submission.user_id,
            job_type=submission.job_type,
            spec=submission.spec,
            idempotency_key=submission.idempotency_key,
            priority=submission.priority,
            max_attempts=submission.max_attempts,
            restart_policy=submission.restart_policy,
        )

        if submission.enqueue:
            self.job_store.enqueue_job(job_id)

        return job_id

    def _coerce_submission(
        self,
        request: JobSubmission | str | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> JobSubmission:
        """Normalize legacy keyword arguments into a JobSubmission."""
        if isinstance(request, JobSubmission):
            if kwargs:
                raise TypeError("submit_job accepts either request or keyword fields")
            if args:
                raise TypeError(
                    "submit_job request cannot be combined with positional fields"
                )
            return request

        positional_values = () if request is None else (request,)
        positional_values += args
        field_names = tuple(JobSubmission.__dataclass_fields__)
        if len(positional_values) > len(field_names):
            raise TypeError("submit_job received too many positional fields")

        for field_name, value in zip(field_names, positional_values, strict=False):
            if field_name in kwargs:
                raise TypeError(f"submit_job got multiple values for {field_name}")
            kwargs[field_name] = value
        return JobSubmission(**kwargs)

    def submit_endpoint_provision(
        self,
        project_id: str,
        user_id: str,
        model_path: str,
        model_name: str = "rivault-gpt-oss-static",
        ttl_sec: int = 3600,
        env_vars: dict[str, str] | None = None,
        priority: int = 10,  # Higher priority for endpoint jobs
    ) -> str:
        """
        Submit an endpoint provisioning job.

        Convenience method for the common case of provisioning Mini-RiVault.
        """
        spec = {
            "model_path": model_path,
            "model": model_name,
            "ttl_sec": ttl_sec,
            "env_vars": env_vars or {},
        }

        return self.submit_job(
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.ENDPOINT_PROVISION,
            spec=spec,
            priority=priority,
            max_attempts=3,  # Fast retry for endpoints
            restart_policy=RestartPolicy.ON_CRASH,
        )

    def submit_index_build(
        self,
        project_id: str,
        user_id: str,
        dataset: str,
        index_name: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> str:
        """
        Submit an index build job.

        Builds a RAG index from a dataset.
        """
        spec = {
            "dataset": dataset,
            "index_name": index_name,
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }

        return self.submit_job(
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.INDEX_BUILD,
            spec=spec,
            priority=0,
            max_attempts=5,  # More retries for long-running jobs
            restart_policy=RestartPolicy.ON_CRASH,
        )

    def submit_embedding_batch(
        self,
        project_id: str,
        user_id: str,
        input_path: str,
        output_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> str:
        """
        Submit a batch embedding job.

        Raises:
            NotImplementedError: the EMBEDDING_BATCH dispatch backend is not
                wired on this deployment (#216). The REST job-create route
                already rejects this type with 400; this guard keeps service
                callers from creating a durable job that can never run.
        """
        raise NotImplementedError(
            "Embedding-batch dispatch is not implemented (no durable backend). "
            "Use index_build to build embeddings into the RAG index instead."
        )

    # =========================================================================
    # Job Query APIs
    # =========================================================================

    def get_job(self, job_id: str) -> Job:
        """Get job by ID."""
        return self.job_store.get_job(job_id)

    def get_job_with_attempts(self, job_id: str) -> tuple[Job, list[JobAttempt]]:
        """Get job with all its attempts."""
        job = self.job_store.get_job(job_id)
        attempts = self.job_store.get_attempts_for_job(job_id)
        return job, attempts

    def list_jobs(
        self,
        project_id: str | None = None,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs with filters."""
        return self.job_store.list_jobs(
            project_id=project_id,
            status=status,
            job_type=job_type,
            limit=limit,
            offset=offset,
        )

    # =========================================================================
    # Job Control APIs
    # =========================================================================

    def cancel_job(self, job_id: str, reason: str = "user_cancelled") -> None:
        """Cancel a job."""
        self.job_store.cancel_job(job_id, reason)

    def retry_job(self, job_id: str) -> None:
        """Force retry a failed/cancelled job."""
        self.job_store.retry_job(job_id)

    # =========================================================================
    # Stats
    # =========================================================================

    @property
    def stats(self) -> dict[str, Any]:
        """Get service statistics."""
        stats = {
            "instance_id": self.instance_id,
            "started": self._started,
        }

        if self._scheduler:
            stats["scheduler"] = self._scheduler.stats

        if self._monitor:
            stats["monitor"] = self._monitor.stats

        return stats


# Convenience function for creating JobService
def create_job_service(
    db_pool,
    scheduler_interval: float = 1.0,
    monitor_interval: float = 5.0,
) -> JobService:
    """
    Create a new JobService instance.

    Example:
        job_service = create_job_service(db_pool)
        job_service.set_backend_factory(my_backend_factory)
        await job_service.startup()
    """
    return JobService(
        db_pool=db_pool,
        scheduler_interval=scheduler_interval,
        monitor_interval=monitor_interval,
    )
