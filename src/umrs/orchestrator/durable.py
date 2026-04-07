"""
Durable Orchestration Service - Integration with Job Tracking.

This module provides a wrapper around OrchestrationService that uses
the durable job tracking system for crash-safe endpoint provisioning.

Usage:
    from umrs.orchestrator.durable import DurableOrchestrationService

    service = DurableOrchestrationService(db_pool)
    await service.startup()

    # Submit durable provision request
    job_id = await service.provision_durable(
        project_id="proj-123",
        model="rivault-gpt-oss-static",
        model_path="/models/rivault-gpt-oss",
    )

    # Job tracking handles:
    # - Crash recovery
    # - Retries
    # - State persistence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..jobs import (
    JobService,
    JobType,
    RestartPolicy,
    create_backend_factory,
    create_status_checker,
    ensure_schema,
)
from .service import OrchestrationService, ProvisionRequest, ProvisionResult

logger = logging.getLogger(__name__)


@dataclass
class DurableProvisionRequest:
    """Request for durable endpoint provisioning."""

    project_id: str
    user_id: str
    model: str
    model_path: str = "/models/rivault-gpt-oss"
    ttl_sec: int = 3600
    gpu_count: int = 1
    max_model_len: int = 4096

    # Job settings
    priority: int = 0
    max_attempts: int = 3
    idempotency_key: Optional[str] = None


class DurableOrchestrationService:
    """
    Orchestration service with durable job tracking.

    Wraps OrchestrationService to provide crash-safe endpoint
    provisioning using the job tracking system.

    Benefits:
    - Survives UMRS restarts
    - Automatic retry on failure
    - Job status queryable via API
    - Audit trail of all provisioning attempts
    """

    def __init__(
        self,
        db_pool,
        orchestration_service: Optional[OrchestrationService] = None,
        scheduler_interval: float = 1.0,
        monitor_interval: float = 5.0,
    ):
        """
        Initialize DurableOrchestrationService.

        Args:
            db_pool: Postgres connection pool
            orchestration_service: Existing service (creates new if None)
            scheduler_interval: Scheduler poll interval
            monitor_interval: Monitor poll interval
        """
        self.db_pool = db_pool
        self._orchestration_service = orchestration_service
        self._job_service: Optional[JobService] = None
        self._scheduler_interval = scheduler_interval
        self._monitor_interval = monitor_interval
        self._started = False

    @property
    def orchestration_service(self) -> OrchestrationService:
        """Get or create the underlying orchestration service."""
        if self._orchestration_service is None:
            self._orchestration_service = OrchestrationService()
        return self._orchestration_service

    @property
    def job_service(self) -> JobService:
        """Get the job service."""
        if self._job_service is None:
            raise RuntimeError("DurableOrchestrationService not started")
        return self._job_service

    async def startup(self) -> None:
        """
        Start the durable orchestration service.

        1. Ensure database schema
        2. Initialize JobService
        3. Wire up adapters
        4. Start scheduler and monitor
        """
        if self._started:
            logger.warning("DurableOrchestrationService already started")
            return

        logger.info("Starting DurableOrchestrationService")

        # Ensure job tracking schema exists
        conn = self.db_pool.getconn()
        try:
            ensure_schema(conn)
            conn.commit()
        finally:
            self.db_pool.putconn(conn)

        # Initialize JobService
        self._job_service = JobService(
            db_pool=self.db_pool,
            scheduler_interval=self._scheduler_interval,
            monitor_interval=self._monitor_interval,
        )

        # Get underlying orchestration components
        orch = self.orchestration_service

        # Create backend factory that wraps the orchestrator
        backend_factory = create_backend_factory(
            orchestrator=orch.backend,
            registry=orch.registry,
            port_manager=orch.port_manager,
        )
        self._job_service.set_backend_factory(backend_factory)

        # Create status checker that uses readiness poller
        status_checker = create_status_checker(
            readiness_poller=orch.readiness_poller,
            registry=orch.registry,
            orchestrator=orch.backend,
        )
        self._job_service.set_status_checker(status_checker)

        # Start JobService (runs reconciliation, scheduler, monitor)
        await self._job_service.startup()

        self._started = True
        logger.info("DurableOrchestrationService started")

    async def shutdown(self, timeout: float = 10.0) -> None:
        """
        Shutdown the durable orchestration service.
        """
        if not self._started:
            return

        logger.info("Shutting down DurableOrchestrationService")

        if self._job_service:
            await self._job_service.shutdown(timeout)

        if self._orchestration_service:
            await self._orchestration_service.close()

        self._started = False
        logger.info("DurableOrchestrationService shutdown complete")

    async def provision_durable(
        self,
        request: DurableProvisionRequest,
    ) -> str:
        """
        Submit a durable endpoint provisioning request.

        Creates a job in QUEUED state. The scheduler will pick it up
        and submit to the backend.

        Args:
            request: Durable provision request

        Returns:
            job_id: Durable job ID for tracking

        Note:
            This returns immediately. Use get_job() to check status.
        """
        if not self._started:
            raise RuntimeError("DurableOrchestrationService not started")

        # Build job spec from request
        spec = {
            "model": request.model,
            "model_path": request.model_path,
            "ttl_sec": request.ttl_sec,
            "gpu_count": request.gpu_count,
            "max_model_len": request.max_model_len,
        }

        # Submit to job service
        job_id = self._job_service.submit_endpoint_provision(
            project_id=request.project_id,
            user_id=request.user_id,
            spec=spec,
            priority=request.priority,
            max_attempts=request.max_attempts,
            idempotency_key=request.idempotency_key,
        )

        logger.info(
            f"Submitted durable provision: job_id={job_id} "
            f"project={request.project_id} model={request.model}"
        )

        return job_id

    def get_job(self, job_id: str):
        """
        Get job by ID.

        Returns job with all attempts.
        """
        return self._job_service.get_job(job_id)

    def get_job_status(self, job_id: str) -> str:
        """Get current job status."""
        job = self._job_service.get_job(job_id)
        if job:
            return job.status.value
        return "not_found"

    def list_jobs(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ):
        """List jobs with optional filters."""
        return self._job_service.list_jobs(
            project_id=project_id,
            status=status,
            limit=limit,
        )

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job."""
        return self._job_service.cancel_job(job_id)

    async def provision_sync(
        self,
        request: DurableProvisionRequest,
        timeout_sec: float = 300.0,
    ) -> ProvisionResult:
        """
        Provision endpoint and wait for result.

        This is a convenience method that submits a durable job
        and polls until completion or timeout.

        Args:
            request: Provision request
            timeout_sec: Maximum wait time

        Returns:
            ProvisionResult with success/failure
        """
        import asyncio
        from ..jobs import JobStatus

        # Submit job
        job_id = await self.provision_durable(request)

        # Poll for completion
        start_time = asyncio.get_event_loop().time()
        poll_interval = 2.0

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > timeout_sec:
                return ProvisionResult(
                    success=False,
                    job_id=job_id,
                    error=f"Timeout after {timeout_sec}s",
                    provisioning_time_sec=elapsed,
                )

            job = self.get_job(job_id)
            if not job:
                return ProvisionResult(
                    success=False,
                    job_id=job_id,
                    error="Job not found",
                    provisioning_time_sec=elapsed,
                )

            if job.status == JobStatus.COMPLETED:
                # Get endpoint from registry
                endpoint = self.orchestration_service.registry.get(
                    project_id=request.project_id,
                    model=request.model,
                    prefer_warm=True,
                )
                return ProvisionResult(
                    success=True,
                    endpoint=endpoint,
                    job_id=job_id,
                    provisioning_time_sec=elapsed,
                )

            elif job.status == JobStatus.FAILED:
                error = "Provisioning failed"
                # Get error from last attempt
                if job.attempts:
                    last_attempt = max(job.attempts, key=lambda a: a.attempt_no)
                    error = last_attempt.error or error

                return ProvisionResult(
                    success=False,
                    job_id=job_id,
                    error=error,
                    provisioning_time_sec=elapsed,
                )

            elif job.status == JobStatus.CANCELLED:
                return ProvisionResult(
                    success=False,
                    job_id=job_id,
                    error="Job cancelled",
                    provisioning_time_sec=elapsed,
                )

            # Still running, wait and check again
            await asyncio.sleep(poll_interval)


# Singleton instance
_durable_service: Optional[DurableOrchestrationService] = None


def get_durable_orchestration_service() -> DurableOrchestrationService:
    """Get the global durable orchestration service singleton."""
    global _durable_service
    if _durable_service is None:
        raise RuntimeError(
            "DurableOrchestrationService not initialized. "
            "Call init_durable_orchestration_service() first."
        )
    return _durable_service


def init_durable_orchestration_service(
    db_pool,
    orchestration_service: Optional[OrchestrationService] = None,
) -> DurableOrchestrationService:
    """
    Initialize the global durable orchestration service.

    Call this during application startup.
    """
    global _durable_service
    _durable_service = DurableOrchestrationService(
        db_pool=db_pool,
        orchestration_service=orchestration_service,
    )
    return _durable_service
