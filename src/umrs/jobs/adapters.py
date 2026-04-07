"""
Backend Adapters - Bridge between JobService and existing orchestrators.

This module provides adapters that wrap existing orchestration backends
(Compose, Kubernetes, Slurm) to work with the durable job tracking system.

Architecture:
    JobService → Scheduler → BackendAdapter → OrchestrationBackend
                 Monitor → StatusChecker → ReadinessPoller
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from .models import AttemptStatus, Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker
from .scheduler import OrchestratorBackend

if TYPE_CHECKING:
    from umrs.orchestrator.base import OrchestratorBackend as BaseOrchestrator
    from umrs.orchestrator.endpoint_registry import EndpointRegistry
    from umrs.orchestrator.readiness import ReadinessPoller

logger = logging.getLogger(__name__)


class EndpointProvisionBackend(OrchestratorBackend):
    """
    Backend adapter for endpoint provisioning jobs.

    Wraps the existing orchestration backend to submit vLLM/LiteLLM
    endpoint provisioning jobs via the durable job tracking system.
    """

    def __init__(
        self,
        orchestrator: "BaseOrchestrator",
        registry: "EndpointRegistry",
        port_manager,
    ):
        """
        Initialize adapter.

        Args:
            orchestrator: Underlying orchestrator (Compose/K8s/Slurm)
            registry: Endpoint registry for registration
            port_manager: Port allocation manager
        """
        self.orchestrator = orchestrator
        self.registry = registry
        self.port_manager = port_manager

    @property
    def backend_name(self) -> str:
        """Return orchestrator type name."""
        backend_name = getattr(self.orchestrator, "backend_name", None)
        if isinstance(backend_name, str):
            return backend_name

        backend_type = getattr(self.orchestrator, "backend_type", None)
        if hasattr(backend_type, "value"):
            return backend_type.value
        if isinstance(backend_type, str):
            return backend_type

        return "unknown"

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        """
        Submit endpoint provisioning to the orchestrator.

        Extracts parameters from job.spec and calls the underlying
        orchestrator.submit_job().

        Returns:
            Backend reference (container ID, k8s job name, slurm job ID)
        """
        spec = job.spec

        # Extract endpoint parameters from job spec
        model_path = spec.get("model_path")
        project_id = job.project_id
        model_name = spec.get("model", "default")
        gpu_count = spec.get("gpu_count", 1)
        max_model_len = spec.get("max_model_len", 4096)
        ttl_sec = spec.get("ttl_sec", 3600)

        # Determine target host
        target_host = self._determine_target_host()

        # Allocate port using attempt ID for tracking
        port = self.port_manager.allocate(attempt.attempt_id, target_host)
        if port is None:
            raise RuntimeError("Failed to allocate port for endpoint")

        try:
            # Submit to underlying orchestrator
            job_info = await self.orchestrator.submit_job(
                model_path=model_path,
                env={
                    "LITELLM_PORT": str(port),
                    "VLLM_MAX_MODEL_LEN": str(max_model_len),
                    "PROJECT_ID": project_id,
                    "MODEL_NAME": model_name,
                },
                gpu_count=gpu_count,
            )

            if not job_info or not job_info.job_id:
                # Release port on failure
                self.port_manager.release(port, target_host)
                raise RuntimeError("Backend failed to submit job")

            # Get actual host from job info
            actual_host = (
                job_info.endpoint.node
                if job_info.endpoint
                else target_host
            )

            # Register endpoint in registry (status=provisioning)
            # Use attempt_id as endpoint ID for correlation
            self.registry.register(
                project_id=project_id,
                model=model_name,
                job_id=job_info.job_id,
                host=actual_host,
                port=port,
                backend=self.backend_name,
                ttl_sec=ttl_sec,
                metadata={
                    "durable_job_id": job.job_id,
                    "attempt_id": attempt.attempt_id,
                    "attempt_no": attempt.attempt_no,
                },
            )

            logger.info(
                f"Submitted endpoint provision: job={job.job_id} "
                f"attempt={attempt.attempt_id} backend_ref={job_info.job_id}"
            )

            return job_info.job_id

        except Exception as e:
            # Release port on any error
            self.port_manager.release(port, target_host)
            raise

    def _determine_target_host(self) -> str:
        """Determine target host based on orchestrator type."""
        backend_name = self.backend_name.lower()
        if backend_name in ("compose", "docker-compose"):
            return "localhost"
        if backend_name in ("kubernetes", "k8s"):
            return "mini-rivault"  # K8s Service DNS
        if backend_name in ("slurm", "slurm-podman"):
            return "pending"  # Updated after allocation
        return "localhost"


class EndpointProvisionStatusChecker(BackendStatusChecker):
    """
    Status checker for endpoint provisioning jobs.

    Uses ReadinessPoller to check if endpoints are up and healthy.
    """

    def __init__(
        self,
        readiness_poller: "ReadinessPoller",
        registry: "EndpointRegistry",
        orchestrator: Optional["BaseOrchestrator"] = None,
    ):
        """
        Initialize checker.

        Args:
            readiness_poller: Poller for health/readiness checks
            registry: Endpoint registry to lookup endpoints
            orchestrator: Optional backend for status refresh and host discovery
        """
        self.readiness_poller = readiness_poller
        self.registry = registry
        self.orchestrator = orchestrator

    @property
    def job_type(self) -> JobType:
        return JobType.ENDPOINT_PROVISION

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """
        Check status of an endpoint provisioning attempt.

        Looks up the endpoint in registry and checks readiness.
        """
        # Get backend ref (container/k8s/slurm job ID)
        backend_ref = attempt.backend_ref
        if not backend_ref:
            return AttemptCheckResult.failed("No backend reference")

        backend_error = await self._refresh_from_backend(backend_ref)
        if backend_error:
            return AttemptCheckResult.failed(backend_error)

        # Lookup endpoint in registry
        endpoint = self.registry.get_by_job_id(backend_ref)
        if not endpoint:
            return AttemptCheckResult.running("waiting for endpoint registration")

        from umrs.orchestrator.endpoint_registry import EndpointStatus

        if endpoint.status == EndpointStatus.FAILED:
            return AttemptCheckResult.failed(
                endpoint.error_message or endpoint.metadata.get("error", "endpoint failed")
            )

        if endpoint.host in ("", "pending"):
            return AttemptCheckResult.running("waiting for backend allocation")

        # Check if endpoint is ready
        try:
            is_ready = await self.readiness_poller.check_ready(
                endpoint.host,
                endpoint.port,
            )

            if is_ready:
                return AttemptCheckResult.succeeded(
                    metrics={
                        "host": endpoint.host,
                        "port": endpoint.port,
                        "endpoint_status": endpoint.status.value,
                    },
                )

            return AttemptCheckResult.running(
                f"waiting for endpoint readiness ({endpoint.host}:{endpoint.port})"
            )

        except Exception as e:
            logger.warning(f"Readiness check failed for {backend_ref}: {e}")

            # Don't mark as failed on transient errors
            # Only fail if endpoint explicitly reports failure
            if endpoint.status == EndpointStatus.FAILED:
                return AttemptCheckResult.failed(
                    endpoint.error_message or endpoint.metadata.get("error", str(e))
                )

            return AttemptCheckResult.running(
                f"waiting for endpoint readiness ({endpoint.host}:{endpoint.port})"
            )

    async def check_attempt(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """Backward-compatible wrapper for older tests/callers."""
        return await self.check_status(attempt, job)

    async def _refresh_from_backend(self, backend_ref: str) -> Optional[str]:
        """Refresh endpoint coordinates and coarse status from backend state."""
        if self.orchestrator is None:
            return None

        try:
            job_info = await self.orchestrator.get_job_status(backend_ref)
        except Exception as exc:
            logger.warning("Failed to refresh backend status for %s: %s", backend_ref, exc)
            return None

        from umrs.orchestrator.base import JobStatus as OrchestratorJobStatus
        from umrs.orchestrator.endpoint_registry import EndpointStatus

        if job_info.endpoint:
            self.registry.update_location(
                backend_ref,
                host=job_info.endpoint.node,
                port=job_info.endpoint.litellm_port,
                metadata={"backend_status": job_info.status.value},
            )

        if job_info.status == OrchestratorJobStatus.PENDING:
            self.registry.update_status(backend_ref, EndpointStatus.PROVISIONING)
        elif job_info.status in (
            OrchestratorJobStatus.STARTING,
            OrchestratorJobStatus.RUNNING,
            OrchestratorJobStatus.READY,
        ):
            self.registry.update_status(backend_ref, EndpointStatus.STARTING)
        elif job_info.status in (
            OrchestratorJobStatus.FAILED,
            OrchestratorJobStatus.STOPPED,
        ):
            error = job_info.error_message or (
                f"Backend job {backend_ref} {job_info.status.value}"
            )
            self.registry.mark_failed(backend_ref, error)
            return error

        return None


class IndexBuildBackend(OrchestratorBackend):
    """
    Backend adapter for index build jobs.

    Submits vector index build tasks to workers.
    Uses attempt-scoped S3 paths for output isolation.
    """

    def __init__(
        self,
        embedding_worker,
        s3_client,
    ):
        """
        Initialize adapter.

        Args:
            embedding_worker: Worker for embedding/indexing
            s3_client: S3 client for path management
        """
        self.embedding_worker = embedding_worker
        self.s3_client = s3_client

    @property
    def backend_name(self) -> str:
        return "index_builder"

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        """
        Submit index build to embedding worker.

        Writes output to attempt-scoped S3 path.
        """
        spec = job.spec

        # Attempt-scoped output path
        # s3://bucket/jobs/<job_id>/attempt_<n>/index/
        output_path = (
            f"s3://rivault/jobs/{job.job_id}/"
            f"attempt_{attempt.attempt_no}/index/"
        )

        # Submit to worker
        task_id = await self.embedding_worker.submit_index_build(
            project_id=job.project_id,
            user_id=job.user_id,
            collection_name=spec.get("collection_name"),
            input_path=spec.get("input_path"),
            output_path=output_path,
            embedding_model=spec.get("embedding_model", "default"),
        )

        logger.info(
            f"Submitted index build: job={job.job_id} "
            f"attempt={attempt.attempt_id} task_id={task_id}"
        )

        return task_id


class IndexBuildStatusChecker(BackendStatusChecker):
    """
    Status checker for index build jobs.

    Polls worker for task completion status.
    """

    def __init__(self, embedding_worker):
        """
        Initialize checker.

        Args:
            embedding_worker: Worker to query for status
        """
        self.embedding_worker = embedding_worker

    @property
    def job_type(self) -> JobType:
        return JobType.INDEX_BUILD

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """Check status of an index build attempt."""
        task_id = attempt.backend_ref
        if not task_id:
            return AttemptCheckResult.failed("No task ID")

        try:
            task_status = await self.embedding_worker.get_task_status(task_id)

            if task_status.completed:
                if task_status.success:
                    return AttemptCheckResult.succeeded(metrics=task_status.metrics)

                return AttemptCheckResult.failed(task_status.error)

            return AttemptCheckResult.running("running")

        except Exception as e:
            logger.warning(f"Failed to check task status {task_id}: {e}")
            return AttemptCheckResult.running("running")

    async def check_attempt(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """Backward-compatible wrapper for older tests/callers."""
        return await self.check_status(attempt, job)


def create_backend_factory(
    orchestrator: "BaseOrchestrator",
    registry: "EndpointRegistry",
    port_manager,
    embedding_worker=None,
    s3_client=None,
    model_store=None,
    path_validator=None,
):
    """
    Create a backend factory function for JobService.

    Args:
        orchestrator: Underlying orchestrator for endpoint jobs
        registry: Endpoint registry
        port_manager: Port allocation manager
        embedding_worker: Worker for index jobs (optional)
        s3_client: S3 client (optional)
        model_store: Model registry store for download jobs (optional)
        path_validator: Path validator for download jobs (optional)

    Returns:
        Factory function: (JobType) -> OrchestratorBackend
    """
    # Pre-create backends
    endpoint_backend = EndpointProvisionBackend(
        orchestrator=orchestrator,
        registry=registry,
        port_manager=port_manager,
    )

    index_backend = None
    if embedding_worker and s3_client:
        index_backend = IndexBuildBackend(
            embedding_worker=embedding_worker,
            s3_client=s3_client,
        )

    model_download_backend = None
    if model_store and path_validator:
        from .model_download import ModelDownloadBackend

        model_download_backend = ModelDownloadBackend(
            model_store=model_store,
            path_validator=path_validator,
        )

    def factory(job_type: JobType) -> OrchestratorBackend:
        if job_type == JobType.ENDPOINT_PROVISION:
            return endpoint_backend
        elif job_type == JobType.INDEX_BUILD:
            if index_backend is None:
                raise RuntimeError("Index build backend not configured")
            return index_backend
        elif job_type == JobType.MODEL_DOWNLOAD:
            if model_download_backend is None:
                raise RuntimeError(
                    "Model download backend not configured. "
                    "Provide model_store and path_validator."
                )
            return model_download_backend
        else:
            raise ValueError(f"Unsupported job type: {job_type}")

    return factory


def create_status_checker(
    readiness_poller: "ReadinessPoller",
    registry: "EndpointRegistry",
    orchestrator: Optional["BaseOrchestrator"] = None,
    embedding_worker=None,
    include_model_download: bool = True,
):
    """
    Create a composite status checker for JobService.

    Args:
        readiness_poller: For endpoint provisioning checks
        registry: Endpoint registry
        orchestrator: Underlying orchestrator for backend status refresh
        embedding_worker: For index build checks (optional)
        include_model_download: Whether to include model download checker

    Returns:
        CompositeStatusChecker instance
    """
    from .monitor import CompositeStatusChecker

    checker = CompositeStatusChecker()
    checker.register(
        JobType.ENDPOINT_PROVISION,
        EndpointProvisionStatusChecker(
            readiness_poller=readiness_poller,
            registry=registry,
            orchestrator=orchestrator,
        ),
    )

    if embedding_worker:
        checker.register(
            JobType.INDEX_BUILD,
            IndexBuildStatusChecker(embedding_worker=embedding_worker),
        )

    if include_model_download:
        from .model_download import ModelDownloadStatusChecker

        checker.register(JobType.MODEL_DOWNLOAD, ModelDownloadStatusChecker())

    return checker
