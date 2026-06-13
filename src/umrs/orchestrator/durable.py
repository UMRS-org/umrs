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

from ..jobs import (
    JobService,
    JobType,
    RestartPolicy,
    create_backend_factory,
    create_status_checker,
    ensure_schema,
)
from ..model_registry import ensure_schema as ensure_model_registry_schema
from ..model_registry.store import ModelStore, PathValidator
from .endpoint_mounts import build_mount_environment
from .service import OrchestrationService, ProvisionResult

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
    tier: str = "user"
    mounts: dict[str, object] | None = None
    egress_domains: list[str] | None = None

    # Compute node selection
    partition_override: str | None = None
    nodelist_hint: str | None = None

    # Job settings
    priority: int = 0
    max_attempts: int = 3
    idempotency_key: str | None = None


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
        orchestration_service: OrchestrationService | None = None,
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
        self._job_service: JobService | None = None
        self._model_store_conn = None
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
        from umrs.jobs.service import set_job_service

        set_job_service(self._job_service)

        # Get underlying orchestration components
        orch = self.orchestration_service

        model_store = None
        path_validator = None
        self._model_store_conn = self.db_pool.getconn()
        try:
            ensure_model_registry_schema(self._model_store_conn)
            self._model_store_conn.commit()
            path_validator = PathValidator()
            model_store = ModelStore(
                self._model_store_conn,
                path_validator=path_validator,
            )
        except Exception:
            logger.exception(
                "Failed to initialize model registry store for durable jobs; "
                "model_download jobs will be unavailable"
            )
            if self._model_store_conn is not None:
                self.db_pool.putconn(self._model_store_conn)
                self._model_store_conn = None

        # Create backend factory that wraps the orchestrator
        try:
            from umrs.admin.user_store import UserStore

            user_store = UserStore()
        except Exception:
            logger.debug("UserStore unavailable; per-user Slurm accounts disabled")
            user_store = None

        backend_factory = create_backend_factory(
            orchestrator=orch.backend,
            registry=orch.registry,
            port_manager=orch.port_manager,
            get_job_with_attempts=self._job_service.get_job_with_attempts,
            model_store=model_store,
            path_validator=path_validator,
            user_store=user_store,
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
            from umrs.jobs.service import set_job_service

            set_job_service(None)

        if self._model_store_conn is not None:
            self.db_pool.putconn(self._model_store_conn)
            self._model_store_conn = None

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

        Performs authorization checks before submission:
        - GPU count within site/session limits
        - Project access verified
        - User not exceeding concurrent job limit

        Args:
            request: Durable provision request

        Returns:
            job_id: Durable job ID for tracking

        Raises:
            AuthorizationDenied: If policy check fails.
            RuntimeError: If service not started.

        Note:
            This returns immediately. Use get_job() to check status.
        """
        if not self._started:
            raise RuntimeError("DurableOrchestrationService not started")

        # Policy check: authorize the provisioning request
        from umrs.shared.auth import AuthContext, UserRole
        from umrs.shared.authorization import (
            AuthorizationDenied,
            ResourceRequest,
            authorize,
        )

        auth_ctx = AuthContext(
            user_id=request.user_id,
            username=request.user_id,
            role=UserRole.USER,
            project_id=request.project_id,
        )

        resource = ResourceRequest(
            project_id=request.project_id,
            gpus=request.gpu_count,
            operation="submit_job",
        )

        # authorize() raises AuthorizationDenied on failure
        authorize(auth_ctx, operation="submit_job", resource=resource)

        # Build job spec from request
        spec = {
            "model": request.model,
            "model_path": request.model_path,
            "ttl_sec": request.ttl_sec,
            "gpu_count": request.gpu_count,
            "max_model_len": request.max_model_len,
            "tier": request.tier,
        }
        mount_env, mount_metadata = build_mount_environment(
            tier=request.tier,
            mounts=request.mounts,
            egress_domains=request.egress_domains,
        )
        if mount_env:
            spec["env_vars"] = mount_env
        if request.mounts:
            spec["mounts"] = mount_metadata.get("mounts", request.mounts)
        if request.egress_domains:
            spec["egress_domains"] = list(request.egress_domains)
        if request.partition_override:
            spec["partition_override"] = request.partition_override
        if request.nodelist_hint:
            spec["nodelist_hint"] = request.nodelist_hint

        # Submit to job service
        job_id = self._job_service.submit_job(
            project_id=request.project_id,
            user_id=request.user_id,
            job_type=JobType.ENDPOINT_PROVISION,
            spec=spec,
            priority=request.priority,
            max_attempts=request.max_attempts,
            restart_policy=RestartPolicy.ON_CRASH,
            idempotency_key=request.idempotency_key,
            enqueue=True,
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
        project_id: str | None = None,
        status: str | None = None,
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
_durable_service: DurableOrchestrationService | None = None


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
    orchestration_service: OrchestrationService | None = None,
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
