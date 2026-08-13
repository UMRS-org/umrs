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
from .factory import get_orchestrator, resolve_configured_backend_name
from .ports import PortManager, get_port_manager
from .readiness import (
    ReadinessPoller,
    get_readiness_poller,
)
from .site_config import load_site_config

logger = logging.getLogger(__name__)

# scancel/cleanup is asynchronous: the scheduler can still report a job as
# active for a few seconds after the cancel signal is issued. Poll the backend
# briefly before declaring a deprovision failed.
_DEPROVISION_VERIFY_ATTEMPTS = 6
_DEPROVISION_VERIFY_DELAY_SEC = 1.0


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
    node_count: int = 1  # Slurm --nodes (>1 for multi-node PP)
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
    min_gpu_memory_gb: float | None = (
        None  # Resource hint from model catalog for partition selection
    )
    requesting_user: str | None = (
        None  # Username for per-user Slurm delegation & accounting
    )
    variant: str | None = None  # Deployment variant name (e.g. "128k", "tep-128k")


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
        if normalized in ("slurm", "slurm-podman", "slurm-remote"):
            return BackendType.SLURM
        if normalized in ("kubernetes", "k8s"):
            return BackendType.KUBERNETES
        if normalized in ("docker-compose", "compose"):
            return BackendType.COMPOSE
        return BackendType.UNKNOWN

    @staticmethod
    def _configured_backend_name() -> str | None:
        """Return an explicitly configured backend, if one exists."""
        return resolve_configured_backend_name()

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

        consecutive_errors = 0
        _max_consecutive_errors = 5

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for backend endpoint location for {job_id}"
                )

            try:
                job = await self.backend.get_job_status(job_id)
                consecutive_errors = 0  # reset on success
            except Exception as exc:  # noqa: BLE001
                consecutive_errors += 1
                logger.warning(
                    "Transient backend error polling location for %s (%d/%d): %s",
                    job_id,
                    consecutive_errors,
                    _max_consecutive_errors,
                    exc,
                )
                if consecutive_errors >= _max_consecutive_errors:
                    raise RuntimeError(
                        f"Backend status check for {job_id} failed "
                        f"{consecutive_errors} consecutive times: {exc}"
                    ) from exc
                await asyncio.sleep(min(float(poll_interval), remaining))
                continue

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
                job_id,
                exc,
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
        readiness_probe: dict[str, str | None] | None = None,
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

        probe = readiness_probe or {}
        try:
            return await poller.wait_for_ready(
                job_id=job_id,
                host=host,
                port=port,
                probe_model=probe.get("model"),
                probe_kind=probe.get("kind") or "models",
                probe_path=probe.get("path"),
                on_progress=on_progress,
            )
        finally:
            if temp_poller is not None:
                await temp_poller.close()

    @staticmethod
    def _resolve_slurm_account(username: str | None) -> str | None:
        """Look up the Slurm account for *username* from the user store.

        Returns None if the user store is unavailable, the user does not
        exist, or the user has no account configured.
        """
        if not username:
            return None
        try:
            from umrs.admin.user_store import UserStore

            store = UserStore()
            record = store.get(username)
            site_name: str | None = None
            try:
                site_name = load_site_config().name
            except FileNotFoundError:
                pass
            return record.resolve_slurm_account(site_name)
        except Exception:
            logger.debug(
                "Could not resolve slurm_account for user %s", username, exc_info=True
            )
            return None

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
                variant=request.variant,
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
                        provisioning_time_sec=asyncio.get_event_loop().time()
                        - start_time,
                    )
                logger.warning(
                    "Stale warm endpoint for %s/%s (job_id=%s) — "
                    "marking failed and provisioning fresh",
                    request.project_id,
                    request.model,
                    existing.job_id,
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
            # Single-node tensor parallelism defaults to the requested GPU
            # count (TP == GPUs on the node); honor an explicit override.
            submit_env.setdefault("VLLM_TENSOR_PARALLEL_SIZE", str(request.gpu_count))
            # Forward the LiteLLM master key from the orchestrator process
            # environment so the mini-RiVault wrapper can authenticate the
            # in-job LiteLLM proxy. Never overwrite an explicit request value.
            litellm_master_key = os.environ.get("LITELLM_MASTER_KEY")
            if litellm_master_key:
                submit_env.setdefault("LITELLM_MASTER_KEY", litellm_master_key)
            service_kind = str(request.metadata.get("service_kind") or "openai")
            serving_backend = str(request.metadata.get("serving_backend") or "vllm")
            if service_kind == "bentoml_vision" or serving_backend == "bentoml":
                submit_env["BENTOML_VISION_DOCKER_PORT"] = str(port)
                submit_env["RIVAULT_SERVICE_PORT"] = str(port)
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
                node_count=request.node_count,
                partition_override=request.partition_override,
                nodelist_hint=request.nodelist_hint,
                min_gpu_memory_gb=request.min_gpu_memory_gb,
                requesting_user=request.requesting_user,
                slurm_account=self._resolve_slurm_account(request.requesting_user),
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
            endpoint_metadata.setdefault("max_model_len", request.max_model_len)
            if request.requesting_user:
                endpoint_metadata.setdefault("requesting_user", request.requesting_user)
            actual_host = job_info.endpoint.node if job_info.endpoint else target_host
            actual_port = (
                job_info.endpoint.litellm_port
                if job_info.endpoint and job_info.endpoint.litellm_port
                else port
            )
            if actual_host and actual_host != "pending":
                endpoint_metadata.setdefault(
                    "service_url", f"http://{actual_host}:{actual_port}"
                )
                endpoint_metadata["detail"] = (
                    f"container starting on {actual_host}:{actual_port}"
                )
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
                variant=request.variant,
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
                    metadata={
                        "detail": f"container starting on {actual_host}:{actual_port}",
                        "service_url": f"http://{actual_host}:{actual_port}",
                    },
                )
                if progress:
                    progress(
                        "assignment_resolved",
                        {"host": actual_host, "port": actual_port},
                    )

            self.registry.update_status(job_id, EndpointStatus.STARTING)
            self.registry.update_metadata(
                job_id,
                {
                    "detail": "waiting for model to load",
                },
            )

            remaining_timeout = max(
                1, int(deadline - asyncio.get_running_loop().time())
            )
            if progress:
                progress("waiting_for_ready", {"timeout": remaining_timeout})
            readiness_probe = str(request.metadata.get("readiness_probe") or "chat")
            readiness_path = request.metadata.get("readiness_path")

            def _on_readiness_progress(r) -> None:
                if progress:
                    progress("readiness_check", r.to_dict())
                detail = _friendly_probe_detail(r.error, r.state.value)
                self.registry.update_metadata(
                    job_id,
                    {
                        "detail": f"probe #{r.attempts}: {detail}"
                        if detail
                        else f"probe #{r.attempts}",
                        "readiness_attempts": r.attempts,
                        "readiness_elapsed_sec": round(r.elapsed_sec, 1),
                    },
                )

            readiness_result = await self._wait_for_ready_with_budget(
                job_id=job_id,
                host=actual_host,
                port=actual_port,
                deadline=deadline,
                readiness_probe={
                    "model": request.model,
                    "kind": readiness_probe,
                    "path": str(readiness_path) if readiness_path else None,
                },
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

            error = (
                readiness_result.error
                or f"Readiness check failed: {readiness_result.state.value}"
            )
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

        # Verify the backend job has actually left the active state. scancel is
        # asynchronous, so poll briefly before declaring the cleanup complete.
        # This absorbs the transient window where squeue still reports the job
        # as RUNNING for a few seconds after the cancel signal is issued.
        active_states = (
            JobStatus.PENDING,
            JobStatus.STARTING,
            JobStatus.READY,
            JobStatus.RUNNING,
        )
        backend_status: JobStatus | None = None
        for attempt in range(_DEPROVISION_VERIFY_ATTEMPTS):
            try:
                backend_info = await self.backend.get_job_status(job_id)
            except Exception as e:
                logger.error(f"Backend cleanup verification failed for {job_id}: {e}")
                return False
            backend_status = backend_info.status
            if backend_status not in active_states:
                break
            if attempt < _DEPROVISION_VERIFY_ATTEMPTS - 1:
                await asyncio.sleep(_DEPROVISION_VERIFY_DELAY_SEC)

        if backend_status in active_states:
            # The cancel was issued but the scheduler still reports the job as
            # active (e.g. stuck in COMPLETING). Preserve the registry entry so
            # the allocation is not orphaned; the janitor finalizes removal once
            # the backend job exits. Callers should treat this as "stopping in
            # progress", not a hard failure.
            logger.info(
                "Backend cleanup for %s still settling (status=%s); deferring "
                "registry removal to janitor",
                job_id,
                backend_status.value,
            )
            self.registry.update_metadata(
                job_id,
                {
                    "detail": "backend cleanup still pending",
                    "cleanup_status": backend_status.value,
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

    async def upgrade_endpoint(
        self,
        job_id: str,
        new_model_path: str,
        timeout_sec: int = 900,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> tuple[bool, str, str | None]:
        """Rolling model-version upgrade: provision replacement, flip, retire.

        Contract (Q-13 Phase 2):
          - old endpoint MUST be preserved until new endpoint is READY
            (rollback-on-health-fail: new deprovisioned, old untouched)
          - on success, the registry record for *job_id* is updated so
            callers continue to resolve to the new host:port after the flip;
            the deprecated backend job is then torn down
          - on any failure before the flip, the old endpoint remains running
            and the new job is torn down (no ghost endpoint left behind)
          - concurrency gate: one upgrade job per endpoint_id at a time
            (registry metadata ``upgrade_in_flight`` key)

        Args:
            job_id: Job ID of the endpoint to flip.
            new_model_path: New model path (must be registered in the
                model store; validated upstream).
            timeout_sec: Health-check budget until the new endpoint is
                returned READY.

        Returns:
            (success, new_job_id, error_message) where ``new_job_id`` is the
            fresh backend handle that now owns traffic when success is True,
            otherwise None.
        """
        endpoint = self.registry.get_by_job_id(job_id)
        if endpoint is None:
            return False, None, f"endpoint not found: {job_id}"

        if not new_model_path or not new_model_path.strip():
            return False, None, "new_model_path is required"

        if endpoint.metadata.get("upgrade_in_flight"):
            return False, None, (
                f"upgrade already in flight for {job_id}; "
                "wait for the pending flip to converge"
            )

        request = self._request_from_endpoint(
            endpoint,
            new_model_path=new_model_path,
            timeout_sec=timeout_sec,
        )

        self.registry.update_metadata(
            job_id,
            {
                "upgrade_in_flight": True,
                "upgrade_target_model_path": new_model_path,
                "detail": f"upgrade: provisioning replacement for {new_model_path}",
            },
        )

        try:
            return await self._run_upgrade_flip(
                job_id=job_id,
                endpoint=endpoint,
                request=request,
                new_model_path=new_model_path,
                on_progress=on_progress,
            )
        finally:
            self.registry.update_metadata(job_id, {"upgrade_in_flight": None})

    def _request_from_endpoint(
        self,
        endpoint: RegisteredEndpoint,
        *,
        new_model_path: str,
        timeout_sec: int,
    ) -> ProvisionRequest:
        """Build the replacement provision spec from an existing endpoint.

        Preserves: mounts, egress domains, environment, TTL model scope,
        partition hint, tier, variant knob (suffixed so the pool YAML keeps
        the upgrade rollout observable), and metadata s.t. the flip is
        consistent with the old instance's runtime shape.
        """
        metadata = dict(endpoint.metadata or {})
        # Variant knob — upgrade rollouts side-by-side with the old instance
        # instead of being treated as a fresh pool definition.
        variant = metadata.get("variant") or endpoint.variant or "default"
        metadata["upgrade_source_job_id"] = endpoint.job_id
        metadata["ttl_sec"] = metadata.get("ttl_sec", 3600)

        request = ProvisionRequest(
            project_id=endpoint.project_id,
            model=str(metadata.get("model") or endpoint.model),
            model_path=new_model_path,
            ttl_sec=int(metadata.get("ttl_sec", 3600)),
            gpu_count=int(metadata.get("gpu_count", 1)),
            node_count=int(metadata.get("node_count", 1)),
            max_model_len=int(metadata.get("max_model_len", 4096)),
            tier=str(metadata.get("tier") or "user"),
            mounts=dict(metadata.get("mounts") or {}),
            egress_domains=list(metadata.get("egress_domains") or []),
            prefer_warm=False,  # never reuse — we are launching the flip target
            wait_for_ready=True,
            env_vars=dict(metadata.get("env_vars") or {}),
            metadata=metadata,
            partition_override=metadata.get("partition_override"),
            nodelist_hint=metadata.get("nodelist_hint"),
            requesting_user=metadata.get("requesting_user"),
            variant=f"{variant}-upgrade",
        )
        return request

    async def _run_upgrade_flip(
        self,
        *,
        job_id: str,
        endpoint: RegisteredEndpoint,
        request: ProvisionRequest,
        new_model_path: str,
        on_progress: Callable[[str, Any], None] | None = None,
    ) -> tuple[bool, str, str | None]:
        """Provision replacement → health-check → flip registry → retire old."""
        if on_progress:
            on_progress("provision_replacement", {"job_id": job_id, "model_path": request.model_path})

        result = await self.provision(request, on_progress=on_progress)
        if not result.success or result.endpoint is None:
            # Health-fail pre-flip: report and preserve old endpoint.
            if result.job_id:
                await self._cancel_new_upgrade_attempt(
                    new_job_id=result.job_id,
                    reason="replacement provision failed",
                )
            return False, None, result.error or "replacement provision failed"

        new_endpoint = result.endpoint
        new_job_id = new_endpoint.job_id

        # Sanity: the flip must carry the same project so cross-project
        # promotion never happens through an upgrade hook.
        if new_endpoint.project_id != endpoint.project_id:
            await self._cancel_new_upgrade_attempt(
                new_job_id=new_job_id,
                reason="replacement resolved to a different project",
            )
            return False, None, (
                f"replacement landed in wrong project ({new_endpoint.project_id})"
            )

        if on_progress:
            on_progress("flip_registry", {"job_id": job_id, "new_job_id": new_job_id})

        try:
            self._flip_registry_record(
                old=endpoint,
                new=new_endpoint,
                new_model_path=new_model_path,
            )
        except Exception as exc:
            # Flip is the last irreversible action; on failure the new endpoint
            # is healthy but the registry still points at the old one. Safely
            # deprovision the new one so no zombie endpoint remains.
            logger.exception("Upgrade flip failed for %s: %s", job_id, exc)
            await self._cancel_new_upgrade_attempt(
                new_job_id=new_job_id,
                reason="registry flip failed",
            )
            return False, None, f"registry flip failed: {exc}"

        if on_progress:
            on_progress("retire_old", {"job_id": job_id})

        # Retire the old backend job only after the registry now routes to
        # the new endpoint. A failure here is not fatal: the old job will be
        # picked up by the TTL/janitor cleanup path.
        try:
            await self.bridge_cleanup_with_registry(old=endpoint, new=new_endpoint)
        except Exception:
            logger.warning(
                "Could not cleanly retire old job %s after upgrade; janitor "
                "will collect it",
                job_id,
            )

        return True, new_job_id, None

    async def bridge_cleanup_with_registry(
        self,
        *,
        old: RegisteredEndpoint,
        new: RegisteredEndpoint,
    ) -> None:
        """Generic deferred retirement: teardown old ASYNCHRONOUSLY post-flip."""
        try:
            await self.deprovision(old.job_id, force=False)
        except Exception as exc:
            logger.warning("Deferred retirement of %s failed: %s", old.job_id, exc)

    def _flip_registry_record(
        self,
        *,
        old: RegisteredEndpoint,
        new: RegisteredEndpoint,
        new_model_path: str,
    ) -> None:
        """Point the registry record for ``old.job_id`` at the new backend.

        The public route contract treats ``job_id`` as identity; after the
        flip, callers resolving job_id land on the new host/port/location.
        The new job_id itself stays resolvable to the same coordinates.
        """
        self.registry.update_location(
            old.job_id,
            host=new.host,
            port=new.port,
            metadata={
                **(old.metadata or {}),
                "model_path": new_model_path,
                "upgrade_source_job_id": old.job_id,
                "upgraded_at": new.created_at.isoformat(),
                "detail": f"upgrade: {old.model} → {new_model_path}",
            },
        )
        self.registry.update_metadata(
            new.job_id,
            {
                **(new.metadata or {}),
                "upgrade_source_job_id": old.job_id,
                "model_path": new_model_path,
            },
        )
        if old.metadata.get("served_model_name"):
            self.registry.update_metadata(
                new.job_id,
                {"served_model_name": old.metadata["served_model_name"]},
            )

    async def _cancel_new_upgrade_attempt(
        self,
        *,
        new_job_id: str,
        reason: str,
    ) -> None:
        """Tear down a hot-but-unused replacement job.

        Used when the upgrade fails *before* the registry flip: the new
        endpoint never received traffic, so immediate deprovision is safe.
        """
        try:
            await self.backend.cleanup(new_job_id, force=True)
        except Exception:
            logger.warning(
                "Failed to tear down upgrade-replacement %s (%s)", new_job_id, reason
            )

    async def refresh_endpoint(
        self,
        job_id: str,
        timeout_sec: int = 300,
    ) -> RegisteredEndpoint | None:
        """Re-resolve host/port from the backend and re-probe readiness.

        Recovers endpoints stuck in PROVISIONING/STARTING with host="pending"
        after a ``_finalize_provision`` task died (e.g. UMRS restart).  The
        method mirrors the finalization flow but is invoked on demand.

        Args:
            job_id: The job ID of the endpoint to refresh.
            timeout_sec: Maximum time to wait for readiness.

        Returns:
            The refreshed endpoint, or ``None`` if the endpoint was not
            found in the registry.
        """
        endpoint = self.registry.get_by_job_id(job_id)
        if not endpoint:
            return None

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec

        # 1. If host is "pending", re-resolve from backend.
        if endpoint.host == "pending":
            try:
                actual_host, actual_port = await self._resolve_endpoint_location(
                    job_id,
                    endpoint.host,
                    endpoint.port,
                    deadline,
                )
            except (TimeoutError, RuntimeError):
                # Backend never published a concrete host, or job failed.
                return self.registry.get_by_job_id(job_id)

            self.registry.update_location(
                job_id,
                host=actual_host,
                port=actual_port,
                metadata={
                    "detail": f"container starting on {actual_host}:{actual_port}",
                    "service_url": f"http://{actual_host}:{actual_port}",
                },
            )
            endpoint = self.registry.get_by_job_id(job_id)
            if not endpoint:
                return None

        # 2. Re-probe readiness using the endpoint's configured probe.
        readiness_probe_kind = str(endpoint.metadata.get("readiness_probe") or "chat")
        readiness_path = endpoint.metadata.get("readiness_path")

        try:
            readiness_result = await self._wait_for_ready_with_budget(
                job_id=job_id,
                host=endpoint.host,
                port=endpoint.port,
                deadline=deadline,
                readiness_probe={
                    "model": endpoint.model,
                    "kind": readiness_probe_kind,
                    "path": str(readiness_path) if readiness_path else None,
                },
            )
        except TimeoutError:
            return self.registry.get_by_job_id(job_id)

        if readiness_result.is_ready:
            self.registry.mark_ready(job_id)
        else:
            # If not ready, mark as degraded (if endpoint is reachable)
            # or leave as-is (if endpoint is unreachable).
            pass

        return self.registry.get_by_job_id(job_id)

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

    async def reconcile_stale_endpoints(
        self,
        timeout_sec: int = 300,
    ) -> dict:
        """Re-resolve and re-probe endpoints stuck in PROVISIONING/STARTING.

        Called at boot time after UMRS restart.  When a UMRS process dies
        while a ``_finalize_provision`` task is running (or the task dies),
        the endpoint registry retains rows in PROVISIONING/STARTING with
        ``host="pending"``.  This method iterates those rows, resolves their
        actual host/port from the backend, re-probes readiness, and either
        promotes them to READY or marks them FAILED depending on the result.

        Returns:
            Summary dict with counts::

                {
                    "checked": N,
                    "promoted_to_ready": N,
                    "marked_failed": N,
                    "timed_out": N,
                    "errors": [...],
                }
        """
        result = {
            "checked": 0,
            "promoted_to_ready": 0,
            "marked_failed": 0,
            "timed_out": 0,
            "errors": [],
        }

        stuck_statuses = {EndpointStatus.PROVISIONING, EndpointStatus.STARTING}

        # Gather candidates outside the registry lock so refresh_endpoint
        # (which acquires it) doesn't deadlock.
        with self.registry._lock:
            candidates = [
                ep
                for ep in self.registry._endpoints.values()
                if ep.status in stuck_statuses
            ]

        if not candidates:
            logger.info("No PROVISIONING/STARTING endpoints to reconcile")
            return result

        logger.info(
            "Boot-time reconciler: found %d endpoints in %s — starting reconciliation",
            len(candidates),
            ", ".join(s.value for s in stuck_statuses),
        )

        for ep in candidates:
            result["checked"] += 1
            try:
                refreshed = await self.refresh_endpoint(
                    ep.job_id,
                    timeout_sec=timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)[:200]
                result["errors"].append({"job_id": ep.job_id, "error": error_msg})
                logger.error(
                    "Reconcile error for job %s: %s", ep.job_id, exc, exc_info=True
                )
                self.registry.mark_failed(ep.job_id, f"reconcile_error: {error_msg}")
                result["marked_failed"] += 1
                continue

            if refreshed is None:
                result["timed_out"] += 1
                result["errors"].append(
                    {"job_id": ep.job_id, "error": "endpoint disappeared"}
                )
                continue

            if refreshed.status == EndpointStatus.READY:
                result["promoted_to_ready"] += 1
                logger.info(
                    "Reconciled job %s → READY (host=%s:%s)",
                    ep.job_id,
                    refreshed.host,
                    refreshed.port,
                )
            else:
                result["marked_failed"] += 1
                self.registry.mark_failed(
                    ep.job_id,
                    f"reconcile: not ready after {timeout_sec}s (status={refreshed.status.value})",
                )
                logger.info(
                    "Reconciled job %s → FAILED (status=%s)",
                    ep.job_id,
                    refreshed.status.value,
                )

        logger.info(
            "Boot-time reconciler done: checked=%d ready=%d failed=%d timed_out=%d errors=%d",
            result["checked"],
            result["promoted_to_ready"],
            result["marked_failed"],
            result["timed_out"],
            len(result["errors"]),
        )
        return result

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
