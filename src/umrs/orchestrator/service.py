"""
Orchestration Service.

High-level service that coordinates:
- Backend selection (Slurm/K8s/Compose)
- Port allocation
- Endpoint registration
- Readiness polling
- Lifecycle management

This is the main entry point for provisioning Mini-RiVault instances.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional

from .base import JobStatus, OrchestratorBackend
from .detection import BackendType, detect_backend
from .endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    RegisteredEndpoint,
    get_registry,
)
from .factory import get_orchestrator
from .ports import PortManager, get_port_manager
from .readiness import (
    ReadinessPoller,
    get_readiness_poller,
)
from .site_config import load_site_config

logger = logging.getLogger(__name__)


@dataclass
class ProvisionRequest:
    """Request to provision a Mini-RiVault instance."""

    project_id: str
    model: str
    model_path: str = "/models/rivault-gpt-oss"
    ttl_sec: int = 3600  # 1 hour default
    gpu_count: int = 1
    max_model_len: int = 4096
    prefer_warm: bool = True  # Try to reuse existing endpoint
    min_ttl_for_reuse_sec: int = 60  # Min TTL remaining for reuse
    env_vars: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvisionResult:
    """Result of a provision request."""

    success: bool
    endpoint: Optional[RegisteredEndpoint] = None
    job_id: Optional[str] = None
    error: Optional[str] = None
    reused: bool = False  # True if existing endpoint was reused
    provisioning_time_sec: float = 0.0

    @property
    def litellm_url(self) -> Optional[str]:
        """Get LiteLLM URL if endpoint is available."""
        return self.endpoint.litellm_url if self.endpoint else None


class OrchestrationService:
    """
    Main orchestration service.

    Coordinates all orchestration components to provision and manage
    Mini-RiVault inference endpoints.

    Usage:
        service = OrchestrationService()
        result = await service.provision(ProvisionRequest(
            project_id="proj-123",
            model="rivault-gpt-oss-static",
        ))
        if result.success:
            print(f"Endpoint ready: {result.litellm_url}")
    """

    def __init__(
        self,
        registry: Optional[EndpointRegistry] = None,
        port_manager: Optional[PortManager] = None,
        readiness_poller: Optional[ReadinessPoller] = None,
        backend: Optional[OrchestratorBackend] = None,
    ):
        self._registry = registry
        self._port_manager = port_manager
        self._readiness_poller = readiness_poller
        self._backend = backend
        self._backend_type: Optional[BackendType] = None

    @property
    def registry(self) -> EndpointRegistry:
        """Get endpoint registry (lazy init)."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    @property
    def port_manager(self) -> PortManager:
        """Get port manager (lazy init)."""
        if self._port_manager is None:
            self._port_manager = get_port_manager()
        return self._port_manager

    @property
    def readiness_poller(self) -> ReadinessPoller:
        """Get readiness poller (lazy init)."""
        if self._readiness_poller is None:
            self._readiness_poller = get_readiness_poller()
        return self._readiness_poller

    @staticmethod
    def _classify_backend_name(backend_name: str) -> BackendType:
        """Collapse backend names into the coarse BackendType enum."""
        normalized = backend_name.lower()
        if normalized in ("slurm", "slurm-podman"):
            return BackendType.SLURM
        if normalized in ("kubernetes", "k8s"):
            return BackendType.KUBERNETES
        if normalized in ("docker-compose", "compose"):
            return BackendType.COMPOSE
        return BackendType.UNKNOWN

    @staticmethod
    def _configured_backend_name() -> Optional[str]:
        """Return an explicitly configured backend, if one exists."""
        explicit = (os.getenv("RIVAULT_ORCHESTRATOR") or "").strip()
        if explicit:
            return explicit

        try:
            site = load_site_config()
        except FileNotFoundError:
            return None

        configured = (site.scheduler.kind or "").strip()
        return configured or None

    @property
    def backend(self) -> OrchestratorBackend:
        """Get orchestrator backend (lazy init with auto-detection)."""
        if self._backend is None:
            configured_backend = self._configured_backend_name()
            if configured_backend:
                self._backend_type = self._classify_backend_name(configured_backend)
                self._backend = get_orchestrator(configured_backend)
                logger.info(
                    "Using orchestrator backend from configuration: %s",
                    configured_backend,
                )
            else:
                detection = detect_backend()
                self._backend_type = detection.backend
                self._backend = get_orchestrator(detection.backend.value)
                logger.info(f"Using orchestrator backend: {detection}")
        return self._backend

    @property
    def backend_type(self) -> BackendType:
        """Get detected backend type."""
        if self._backend_type is None:
            _ = self.backend  # Trigger detection
        return self._backend_type or BackendType.UNKNOWN

    async def _resolve_endpoint_location(
        self,
        job_id: str,
        host: str,
        port: int,
        deadline: float,
    ) -> tuple[str, int]:
        """Wait for the backend to publish a concrete endpoint host."""
        if host and host != "pending":
            return host, port

        loop = asyncio.get_running_loop()
        poll_interval = max(
            1,
            int(getattr(self.backend.config, "readiness_poll_interval_sec", 5)),
        )

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for backend endpoint location for {job_id}"
                )

            job = await self.backend.get_job_status(job_id)
            if job.status == JobStatus.FAILED:
                raise RuntimeError(
                    f"Backend job {job_id} failed before publishing an endpoint: "
                    f"{job.error_message or 'unknown failure'}"
                )
            if job.status == JobStatus.STOPPED:
                raise RuntimeError(
                    f"Backend job {job_id} stopped before publishing an endpoint"
                )

            if job.endpoint and job.endpoint.node and job.endpoint.node != "pending":
                return job.endpoint.node, job.endpoint.litellm_port or port

            await asyncio.sleep(min(float(poll_interval), remaining))

    async def _wait_for_ready_with_budget(
        self,
        job_id: str,
        host: str,
        port: int,
        deadline: float,
        on_progress: Optional[Callable[[Any], None]] = None,
    ):
        """Run readiness polling within the remaining provision budget."""
        loop = asyncio.get_running_loop()
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for endpoint readiness budget for {job_id}"
            )

        poller = self.readiness_poller
        temp_poller: Optional[ReadinessPoller] = None

        if remaining < (poller.config.timeout_sec - 1):
            temp_poller = ReadinessPoller(
                replace(poller.config, timeout_sec=max(1, int(remaining)))
            )
            poller = temp_poller

        try:
            return await poller.wait_for_ready(
                job_id=job_id,
                host=host,
                port=port,
                on_progress=on_progress,
            )
        finally:
            if temp_poller is not None:
                await temp_poller.close()

    async def provision(
        self,
        request: ProvisionRequest,
        on_progress: Optional[Callable[[str, Any], None]] = None,
    ) -> ProvisionResult:
        """
        Provision a Mini-RiVault endpoint.

        Flow:
        1. Check for existing warm endpoint (if prefer_warm)
        2. Allocate port
        3. Submit job to backend
        4. Register endpoint in registry
        5. Wait for readiness (authoritative)
        6. Update endpoint status

        Args:
            request: Provision request details.
            on_progress: Optional callback for progress updates.

        Returns:
            ProvisionResult with endpoint details or error.
        """
        start_time = asyncio.get_event_loop().time()

        def progress(stage: str, detail: Any = None):
            logger.debug(f"Provision progress: {stage} - {detail}")
            if on_progress:
                on_progress(stage, detail)

        progress("starting", {"project_id": request.project_id, "model": request.model})

        # 1. Check for existing warm endpoint
        if request.prefer_warm:
            existing = self.registry.get(
                request.project_id,
                request.model,
                prefer_warm=True,
                min_ttl_sec=request.min_ttl_for_reuse_sec,
            )
            if existing and existing.is_usable:
                progress("reusing", {"job_id": existing.job_id})
                logger.info(
                    f"Reusing existing endpoint for {request.project_id}/{request.model}: "
                    f"{existing.job_id}"
                )
                return ProvisionResult(
                    success=True,
                    endpoint=existing,
                    job_id=existing.job_id,
                    reused=True,
                    provisioning_time_sec=asyncio.get_event_loop().time() - start_time,
                )

        # 2. Determine target host (backend-dependent)
        # For Slurm: we don't know the host until job starts
        # For K8s: we use Service name
        # For Compose: we use localhost
        target_host = self._determine_target_host()
        progress("host_determined", {"host": target_host})

        # 3. Allocate port
        temp_job_id = f"pending-{request.project_id}-{request.model}-{int(asyncio.get_event_loop().time())}"
        port = self.port_manager.allocate(temp_job_id, target_host)
        if port is None:
            return ProvisionResult(
                success=False,
                error="Failed to allocate port",
            )
        progress("port_allocated", {"port": port})

        job_id: Optional[str] = None
        actual_host = target_host
        actual_port = port
        deadline = start_time + self.readiness_poller.config.timeout_sec

        try:
            # 4. Submit job to backend
            progress("submitting_job", {"backend": self.backend_type.value})

            submit_env = {
                key: str(value)
                for key, value in request.env_vars.items()
                if isinstance(value, (str, int, float))
            }
            submit_env.update(
                {
                    "LITELLM_PORT": str(port),
                    "VLLM_MAX_MODEL_LEN": str(request.max_model_len),
                    "PROJECT_ID": request.project_id,
                    "MODEL_NAME": request.model,
                }
            )

            job_info = await self.backend.submit_job(
                model_path=request.model_path,
                env=submit_env,
                gpu_count=request.gpu_count,
            )

            if not job_info or not job_info.job_id:
                self.port_manager.release(port, target_host)
                return ProvisionResult(
                    success=False,
                    error="Backend failed to submit job",
                )

            job_id = job_info.job_id
            progress("job_submitted", {"job_id": job_id})

            actual_host = job_info.endpoint.node if job_info.endpoint else target_host
            actual_port = (
                job_info.endpoint.litellm_port
                if job_info.endpoint and job_info.endpoint.litellm_port
                else port
            )
            progress("waiting_for_assignment", {"job_id": job_id})
            actual_host, actual_port = await self._resolve_endpoint_location(
                job_id,
                actual_host,
                actual_port,
                deadline,
            )
            progress("assignment_resolved", {"host": actual_host, "port": actual_port})

            endpoint_metadata = dict(request.metadata)
            endpoint_metadata.setdefault("allocated_host", target_host)
            endpoint_metadata.setdefault("allocated_port", port)

            # 5. Register endpoint in registry (status=provisioning)
            endpoint = self.registry.register(
                project_id=request.project_id,
                model=request.model,
                job_id=job_id,
                host=actual_host,
                port=actual_port,
                backend=self.backend_type.value,
                ttl_sec=request.ttl_sec,
                metadata=endpoint_metadata,
            )
            progress("registered", {"endpoint_id": job_id})

            # Update registry status to starting
            self.registry.update_status(job_id, EndpointStatus.STARTING)

            # 6. Wait for readiness (authoritative)
            remaining_timeout = max(1, int(deadline - asyncio.get_running_loop().time()))
            progress("waiting_for_ready", {"timeout": remaining_timeout})

            readiness_result = await self._wait_for_ready_with_budget(
                job_id=job_id,
                host=actual_host,
                port=actual_port,
                deadline=deadline,
                on_progress=lambda r: progress("readiness_check", r.to_dict()),
            )

            # 7. Update endpoint status based on readiness
            if readiness_result.is_ready:
                self.registry.mark_ready(job_id)
                progress("ready", {"models": readiness_result.models})

                # Refresh endpoint from registry
                endpoint = self.registry.get_by_job_id(job_id)

                return ProvisionResult(
                    success=True,
                    endpoint=endpoint,
                    job_id=job_id,
                    reused=False,
                    provisioning_time_sec=asyncio.get_event_loop().time() - start_time,
                )
            else:
                # Readiness failed
                error = readiness_result.error or f"Readiness check failed: {readiness_result.state.value}"
                self.registry.mark_failed(job_id, error)
                progress("failed", {"error": error})

                # Cleanup
                await self._cleanup_failed_job(
                    job_id,
                    actual_host,
                    actual_port,
                    allocated_host=target_host,
                )

                return ProvisionResult(
                    success=False,
                    job_id=job_id,
                    error=error,
                    provisioning_time_sec=asyncio.get_event_loop().time() - start_time,
                )

        except Exception as e:
            logger.exception(f"Provision failed: {e}")
            if job_id is not None:
                try:
                    await self._cleanup_failed_job(
                        job_id,
                        actual_host,
                        actual_port,
                        allocated_host=target_host,
                    )
                except Exception as cleanup_error:
                    logger.warning(
                        "Cleanup failed after provision error for %s: %s",
                        job_id,
                        cleanup_error,
                    )
            else:
                self.port_manager.release(port, target_host)
            return ProvisionResult(
                success=False,
                error=str(e),
            )

    async def deprovision(
        self,
        job_id: str,
        force: bool = False,
    ) -> bool:
        """
        Deprovision (stop) an endpoint.

        Args:
            job_id: Job identifier.
            force: Force cleanup even if endpoint is in use.

        Returns:
            True if successful.
        """
        endpoint = self.registry.get_by_job_id(job_id)
        if not endpoint:
            logger.warning(f"Endpoint not found for deprovision: {job_id}")
            return False

        logger.info(f"Deprovisioning endpoint: {job_id}")

        # Stop the backend job
        try:
            await self.backend.cleanup(job_id)
        except Exception as e:
            logger.error(f"Backend cleanup failed for {job_id}: {e}")
            if not force:
                return False

        # Release port
        self.port_manager.release(endpoint.port, endpoint.host)
        allocated_host = endpoint.metadata.get("allocated_host")
        if allocated_host and allocated_host != endpoint.host:
            self.port_manager.release(endpoint.port, allocated_host)

        # Remove from registry
        self.registry.remove(job_id)

        return True

    async def get_or_provision(
        self,
        request: ProvisionRequest,
    ) -> ProvisionResult:
        """
        Get existing endpoint or provision a new one.

        This is the main entry point for Gateway to get an endpoint.
        """
        return await self.provision(request)

    def get_endpoint(
        self,
        project_id: str,
        model: str,
    ) -> Optional[RegisteredEndpoint]:
        """
        Get an existing ready endpoint.

        Returns:
            Endpoint if found and ready, None otherwise.
        """
        return self.registry.get(project_id, model, prefer_warm=True)

    def list_endpoints(
        self,
        project_id: Optional[str] = None,
    ) -> List[RegisteredEndpoint]:
        """List all endpoints, optionally filtered by project."""
        return self.registry.list_all(project_id=project_id)

    async def check_health(self, job_id: str) -> bool:
        """
        Check if an endpoint is healthy.

        Performs a quick readiness check.
        """
        endpoint = self.registry.get_by_job_id(job_id)
        if not endpoint:
            return False

        return await self.readiness_poller.check_ready(endpoint.host, endpoint.port)

    def _determine_target_host(self) -> str:
        """Determine target host based on backend type."""
        if self.backend_type == BackendType.COMPOSE:
            return "localhost"
        elif self.backend_type == BackendType.KUBERNETES:
            # K8s uses Service DNS name
            return "mini-rivault"  # Will be resolved by K8s DNS
        elif self.backend_type == BackendType.SLURM:
            # For Slurm, we don't know the host until job starts
            # Backend will update this after allocation
            return "pending"
        else:
            return "localhost"

    async def _cleanup_failed_job(
        self,
        job_id: str,
        host: str,
        port: int,
        allocated_host: Optional[str] = None,
    ):
        """Clean up resources after a failed provision."""
        try:
            await self.backend.cleanup(job_id)
        except Exception as e:
            logger.warning(f"Backend cleanup failed: {e}")

        self.port_manager.release(port, host)
        if allocated_host and allocated_host != host:
            self.port_manager.release(port, allocated_host)
        self.registry.remove(job_id)

    async def close(self):
        """Clean up resources."""
        if self._readiness_poller:
            await self._readiness_poller.close()


# Global singleton
_service: Optional[OrchestrationService] = None


def get_orchestration_service() -> OrchestrationService:
    """Get the global orchestration service singleton."""
    global _service
    if _service is None:
        _service = OrchestrationService()
    return _service
