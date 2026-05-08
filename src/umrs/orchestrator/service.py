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
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any

from .base import JobStatus, OrchestratorBackend
from .detection import BackendType, detect_backend
from .endpoint_mounts import build_mount_environment
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


def _friendly_probe_detail(error: str | None, state: str) -> str:
    """Translate raw readiness-probe errors into user-friendly messages.

    Users seeing 'Connection error: ...' tend to think provisioning has
    failed, when in reality the container is still starting up. This
    function maps known transient errors to reassuring progress text.
    """
    if not error:
        return state

    low = error.lower()

    # Container hasn't opened its port yet
    if "connection" in low and ("refused" in low or "failed" in low or "reset" in low):
        return "waiting for service to start"

    # DNS not ready or host unreachable
    if "name or service not known" in low or "nodename nor servname" in low:
        return "waiting for host to become reachable"

    # HTTP 503 — LiteLLM up but vLLM backend not ready
    if "http 503" in low or "service unavailable" in low:
        return "model is loading"

    # HTTP 502/504 — reverse proxy up, backend not ready
    if "http 502" in low or "http 504" in low or "bad gateway" in low:
        return "waiting for backend to respond"

    # Timeout on the probe request itself
    if "timeout" in low:
        return "service not responding yet"

    # vLLM model list empty or expected model missing
    if "no models" in low or "expected model" in low:
        return "model still loading into memory"

    # Chat probe failed — model loaded but warmup incomplete
    if "chat probe" in low:
        return "model warming up"

    # Fallback: keep original but strip noisy details
    short = " ".join(error.split())
    if len(short) > 60:
        short = short[:57] + "..."
    return short


@dataclass
class ProvisionRequest:
    """Request to provision a Mini-RiVault instance."""

    project_id: str
    model: str
    model_path: str = "/models/rivault-gpt-oss"
    ttl_sec: int = 3600  # 1 hour default
    gpu_count: int = 1
    max_model_len: int = 4096
    tier: str = "user"
    mounts: dict[str, Any] = field(default_factory=dict)
    egress_domains: list[str] = field(default_factory=list)
    prefer_warm: bool = True  # Try to reuse existing endpoint
    wait_for_ready: bool = True  # Return only after readiness unless disabled
    min_ttl_for_reuse_sec: int = 60  # Min TTL remaining for reuse
    env_vars: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    partition_override: str | None = None  # Override site partition for this request
    nodelist_hint: str | None = None  # Slurm --nodelist hint (e.g. "ai-h200-brc-00")


@dataclass
class ProvisionResult:
    """Result of a provision request."""

    success: bool
    endpoint: RegisteredEndpoint | None = None
    job_id: str | None = None
    error: str | None = None
    reused: bool = False  # True if existing endpoint was reused
    provisioning_time_sec: float = 0.0

    @property
    def litellm_url(self) -> str | None:
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
        registry: EndpointRegistry | None = None,
        port_manager: PortManager | None = None,
        readiness_poller: ReadinessPoller | None = None,
        backend: OrchestratorBackend | None = None,
    ):
        self._registry = registry
        self._port_manager = port_manager
        self._readiness_poller = readiness_poller
        self._backend = backend
        self._backend_type: BackendType | None = None

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
    def _configured_backend_name() -> str | None:
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

    async def _verify_backend_job_alive(self, job_id: str) -> bool:
        """Return True if the backend reports the job in a non-terminal state.

        Used to guard against stale warm-endpoint reuse: an entry can be
        marked READY in the registry while the underlying Slurm/K8s job
        has already terminated (walltime, manual cancel, crash).  Returns
        False on any verification error so the caller falls through to a
        fresh provisioning path (fail-safe default).
        """
        try:
            info = await self.backend.get_job_status(job_id)
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.warning(
                "Backend get_job_status(%s) failed during warm-endpoint check: %s",
                job_id, exc,
            )
            return False
        return info.status in (
            JobStatus.PENDING,
            JobStatus.STARTING,
            JobStatus.READY,
            JobStatus.RUNNING,
        )

    async def _wait_for_ready_with_budget(
        self,
        job_id: str,
        host: str,
        port: int,
        deadline: float,
        probe_model: str | None = None,
        probe_kind: str = "models",
        on_progress: Callable[[Any], None] | None = None,
    ):
        """Run readiness polling within the remaining provision budget."""
        loop = asyncio.get_running_loop()
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for endpoint readiness budget for {job_id}"
            )

        poller = self.readiness_poller
        temp_poller: ReadinessPoller | None = None
        target_timeout = max(1, int(remaining))

        if target_timeout != poller.config.timeout_sec:
            if is_dataclass(poller.config):
                temp_poller = ReadinessPoller(
                    replace(poller.config, timeout_sec=target_timeout)
                )
                poller = temp_poller
            else:
                poller.config.timeout_sec = target_timeout

        try:
            return await poller.wait_for_ready(
                job_id=job_id,
                host=host,
                port=port,
                probe_model=probe_model,
                probe_kind=probe_kind,
                on_progress=on_progress,
            )
        finally:
            if temp_poller is not None:
                await temp_poller.close()

    async def provision(
        self,
        request: ProvisionRequest,
        on_progress: Callable[[str, Any], None] | None = None,
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
                # Verify the backend job is actually still alive before
                # reusing the cached endpoint.  ``is_usable`` only reflects
                # the registry's cached status; if the Slurm/K8s job has
                # terminated since the last update we would otherwise
                # return a ghost endpoint.
                if await self._verify_backend_job_alive(existing.job_id):
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
                logger.warning(
                    "Stale warm endpoint for %s/%s (job_id=%s) — "
                    "marking failed and provisioning fresh",
                    request.project_id, request.model, existing.job_id,
                )
                self.registry.mark_failed(
                    existing.job_id,
                    "stale registry entry: backend job no longer alive",
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

        job_id: str | None = None
        actual_host = target_host
        actual_port = port
        backend_startup_timeout = int(
            getattr(self.backend.config, "startup_timeout_sec", 0) or 0
        )
        provision_timeout_sec = max(
            self.readiness_poller.config.timeout_sec,
            backend_startup_timeout,
        )
        deadline = start_time + provision_timeout_sec

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
            mount_env, mount_metadata = build_mount_environment(
                tier=request.tier,
                mounts=request.mounts,
                egress_domains=request.egress_domains,
            )
            submit_env.update(mount_env)

            job_info = await self.backend.submit_job(
                model_path=request.model_path,
                env=submit_env,
                gpu_count=request.gpu_count,
                partition_override=request.partition_override,
                nodelist_hint=request.nodelist_hint,
            )

            if not job_info or not job_info.job_id:
                self.port_manager.release(port, target_host)
                return ProvisionResult(
                    success=False,
                    error="Backend failed to submit job",
                )

            job_id = job_info.job_id
            progress("job_submitted", {"job_id": job_id})

            endpoint_metadata = dict(request.metadata)
            endpoint_metadata.update(mount_metadata)
            endpoint_metadata.setdefault("allocated_host", target_host)
            endpoint_metadata.setdefault("allocated_port", port)
            actual_host = job_info.endpoint.node if job_info.endpoint else target_host
            actual_port = (
                job_info.endpoint.litellm_port
                if job_info.endpoint and job_info.endpoint.litellm_port
                else port
            )
            if actual_host and actual_host != "pending":
                endpoint_metadata["detail"] = f"container starting on {actual_host}:{actual_port}"
            else:
                endpoint_metadata["detail"] = "waiting for resource allocation"

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

            if request.wait_for_ready:
                return await self._finalize_provision(
                    request=request,
                    job_id=job_id,
                    target_host=target_host,
                    actual_host=actual_host,
                    actual_port=actual_port,
                    deadline=deadline,
                    progress=progress,
                    start_time=start_time,
                )

            asyncio.create_task(
                self._finalize_provision(
                    request=request,
                    job_id=job_id,
                    target_host=target_host,
                    actual_host=actual_host,
                    actual_port=actual_port,
                    deadline=deadline,
                    progress=None,
                    start_time=start_time,
                )
            )

            return ProvisionResult(
                success=True,
                endpoint=endpoint,
                job_id=job_id,
                reused=False,
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

    async def _finalize_provision(
        self,
        *,
        request: ProvisionRequest,
        job_id: str,
        target_host: str,
        actual_host: str,
        actual_port: int,
        deadline: float,
        progress: Callable[[str, Any], None] | None,
        start_time: float,
    ) -> ProvisionResult:
        """Resolve backend assignment and readiness for a submitted job."""
        try:
            if actual_host == "pending":
                if progress:
                    progress("waiting_for_assignment", {"job_id": job_id})
                actual_host, actual_port = await self._resolve_endpoint_location(
                    job_id,
                    actual_host,
                    actual_port,
                    deadline,
                )
                self.registry.update_location(
                    job_id,
                    host=actual_host,
                    port=actual_port,
                    metadata={"detail": f"container starting on {actual_host}:{actual_port}"},
                )
                if progress:
                    progress("assignment_resolved", {"host": actual_host, "port": actual_port})

            self.registry.update_status(job_id, EndpointStatus.STARTING)
            self.registry.update_metadata(job_id, {
                "detail": "waiting for model to load",
            })

            remaining_timeout = max(1, int(deadline - asyncio.get_running_loop().time()))
            if progress:
                progress("waiting_for_ready", {"timeout": remaining_timeout})
            readiness_probe = str(request.metadata.get("readiness_probe") or "chat")

            def _on_readiness_progress(r) -> None:
                if progress:
                    progress("readiness_check", r.to_dict())
                detail = _friendly_probe_detail(r.error, r.state.value)
                self.registry.update_metadata(job_id, {
                    "detail": f"probe #{r.attempts}: {detail}" if detail else f"probe #{r.attempts}",
                    "readiness_attempts": r.attempts,
                    "readiness_elapsed_sec": round(r.elapsed_sec, 1),
                })

            readiness_result = await self._wait_for_ready_with_budget(
                job_id=job_id,
                host=actual_host,
                port=actual_port,
                deadline=deadline,
                probe_model=request.model,
                probe_kind=readiness_probe,
                on_progress=_on_readiness_progress,
            )

            if readiness_result.is_ready:
                self.registry.mark_ready(job_id)
                if progress:
                    progress("ready", {"models": readiness_result.models})
                endpoint = self.registry.get_by_job_id(job_id)
                return ProvisionResult(
                    success=True,
                    endpoint=endpoint,
                    job_id=job_id,
                    reused=False,
                    provisioning_time_sec=asyncio.get_event_loop().time() - start_time,
                )

            error = readiness_result.error or f"Readiness check failed: {readiness_result.state.value}"
            self.registry.mark_failed(job_id, error)
            if progress:
                progress("failed", {"error": error})
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
            logger.exception(f"Provision finalize failed: {e}")
            self.registry.mark_failed(job_id, str(e))
            await self._cleanup_failed_job(
                job_id,
                actual_host,
                actual_port,
                allocated_host=target_host,
            )
            return ProvisionResult(
                success=False,
                job_id=job_id,
                error=str(e),
                provisioning_time_sec=asyncio.get_event_loop().time() - start_time,
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
            await self.backend.cleanup(job_id, force=force)
        except Exception as e:
            logger.error(f"Backend cleanup failed for {job_id}: {e}")
            return False

        try:
            backend_info = await self.backend.get_job_status(job_id)
        except Exception as e:
            logger.error(f"Backend cleanup verification failed for {job_id}: {e}")
            return False

        if backend_info.status in (
            JobStatus.PENDING,
            JobStatus.STARTING,
            JobStatus.READY,
            JobStatus.RUNNING,
        ):
            logger.error(
                "Backend cleanup for %s returned while job is still active: %s",
                job_id,
                backend_info.status.value,
            )
            self.registry.update_metadata(
                job_id,
                {
                    "detail": "backend cleanup still pending",
                    "cleanup_status": backend_info.status.value,
                },
            )
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
    ) -> RegisteredEndpoint | None:
        """
        Get an existing ready endpoint.

        Returns:
            Endpoint if found and ready, None otherwise.
        """
        return self.registry.get(project_id, model, prefer_warm=True)

    def list_endpoints(
        self,
        project_id: str | None = None,
    ) -> list[RegisteredEndpoint]:
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
        allocated_host: str | None = None,
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
_service: OrchestrationService | None = None


def get_orchestration_service() -> OrchestrationService:
    """Get the global orchestration service singleton."""
    global _service
    if _service is None:
        _service = OrchestrationService()
    return _service
