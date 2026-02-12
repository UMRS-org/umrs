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

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    AttemptStatus,
    Job,
    JobAttempt,
    JobStatus,
    JobType,
    RestartPolicy,
)
from .monitor import AttemptMonitor, CompositeStatusChecker, EndpointProvisionChecker
from .reconciler import ReconcileResult, Reconciler, reconcile_on_startup
from .scheduler import Scheduler, SimpleBackendAdapter
from .store import DuplicateJobError, JobNotFoundError, JobStore

logger = logging.getLogger(__name__)


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
        self._job_store: Optional[JobStore] = None
        self._scheduler: Optional[Scheduler] = None
        self._monitor: Optional[AttemptMonitor] = None

        # Backend factory (set by caller)
        self._backend_factory = None
        self._status_checker: Optional[CompositeStatusChecker] = None

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

        # Run boot reconciliation
        result = reconcile_on_startup(self._job_store)
        logger.info(f"Boot reconciliation: {result}")

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
        project_id: str,
        user_id: str,
        job_type: JobType,
        spec: Dict[str, Any],
        priority: int = 0,
        max_attempts: int = 3,
        restart_policy: RestartPolicy = RestartPolicy.ON_CRASH,
        idempotency_key: Optional[str] = None,
        enqueue: bool = True,
    ) -> str:
        """
        Submit a new job.

        Args:
            project_id: Project/tenant identifier
            user_id: User who submitted the job
            job_type: Type of job
            spec: Job-specific parameters
            priority: Higher = more urgent
            max_attempts: Maximum retry attempts
            restart_policy: When to retry
            idempotency_key: Optional; auto-generated if not provided
            enqueue: Whether to enqueue immediately

        Returns:
            job_id of created job

        Raises:
            DuplicateJobError: If idempotency key already exists
        """
        job_id = self.job_store.create_job(
            project_id=project_id,
            user_id=user_id,
            job_type=job_type,
            spec=spec,
            idempotency_key=idempotency_key,
            priority=priority,
            max_attempts=max_attempts,
            restart_policy=restart_policy,
        )

        if enqueue:
            self.job_store.enqueue_job(job_id)

        return job_id

    def submit_endpoint_provision(
        self,
        project_id: str,
        user_id: str,
        model_path: str,
        model_name: str = "rivault-gpt-oss-static",
        ttl_sec: int = 3600,
        env_vars: Optional[Dict[str, str]] = None,
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
        """
        spec = {
            "input_path": input_path,
            "output_path": output_path,
            "embedding_model": embedding_model,
        }

        return self.submit_job(
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.EMBEDDING_BATCH,
            spec=spec,
            priority=0,
            max_attempts=5,
            restart_policy=RestartPolicy.ON_CRASH,
        )

    # =========================================================================
    # Job Query APIs
    # =========================================================================

    def get_job(self, job_id: str) -> Job:
        """Get job by ID."""
        return self.job_store.get_job(job_id)

    def get_job_with_attempts(self, job_id: str) -> tuple[Job, List[JobAttempt]]:
        """Get job with all its attempts."""
        job = self.job_store.get_job(job_id)
        attempts = self.job_store.get_attempts_for_job(job_id)
        return job, attempts

    def list_jobs(
        self,
        project_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Job]:
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
    def stats(self) -> Dict[str, Any]:
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
