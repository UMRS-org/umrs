"""
HybridOrchestrator Composer.

Combines ExecutionTarget + RuntimeWrapper + ModelRefResolver into a unified
orchestration layer that implements the existing OrchestratorBackend interface
while leveraging the new composable architecture.

Hybrid Slurm profiles are experimental, not for HPC production, until runtime
launch is executed inside the Slurm allocation via srun/SSH or an equivalent
remote execution boundary.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from umrs.model_registry.models import ExecutionAuthority, ModelRef

from ..base import (
    DEFAULT_MINI_RIVAULT_IMAGE,
    JobInfo,
    JobStatus,
    OrchestratorBackend,
    OrchestratorConfig,
)
from ..base import (
    EndpointInfo as BaseEndpointInfo,
)
from .resolvers import (
    ResolverRegistry,
)
from .targets import (
    AllocationInfo,
    AllocationStatus,
    ComposeTarget,
    ExecutionTarget,
    JobSpec,
    K8sTarget,
    SlurmTarget,
)
from .wrappers import (
    ContainerEndpoint,
    DockerWrapper,
    K8sJobWrapper,
    LaunchSpec,
    MountSpec,
    RuntimeWrapper,
    SingularityWrapper,
)

logger = logging.getLogger(__name__)

EXPERIMENTAL_HYBRID_SLURM_ENV = "RIVAULT_ENABLE_EXPERIMENTAL_HYBRID_SLURM"


# =============================================================================
# Deployment Profiles
# =============================================================================


@dataclass
class DeploymentProfile:
    """
    Defines a valid ExecutionTarget + RuntimeWrapper combination.

    Profiles encapsulate tested/supported deployment configurations.
    """

    name: str
    description: str
    execution_authority: ExecutionAuthority
    target_class: type[ExecutionTarget]
    wrapper_class: type[RuntimeWrapper]
    target_kwargs: dict[str, Any] = field(default_factory=dict)
    wrapper_kwargs: dict[str, Any] = field(default_factory=dict)


# Predefined deployment profiles
DEPLOYMENT_PROFILES: dict[str, DeploymentProfile] = {
    "slurm-singularity": DeploymentProfile(
        name="slurm-singularity",
        description="Experimental, not for HPC production: Slurm + Singularity",
        execution_authority=ExecutionAuthority.SLURM,
        target_class=SlurmTarget,
        wrapper_class=SingularityWrapper,
    ),
    "slurm-docker": DeploymentProfile(
        name="slurm-docker",
        description="Experimental, not for HPC production: Slurm + Docker",
        execution_authority=ExecutionAuthority.SLURM,
        target_class=SlurmTarget,
        wrapper_class=DockerWrapper,
    ),
    "k8s-native": DeploymentProfile(
        name="k8s-native",
        description="Kubernetes: K8s Job + K8s containers",
        execution_authority=ExecutionAuthority.K8S,
        target_class=K8sTarget,
        wrapper_class=K8sJobWrapper,
    ),
    "compose-docker": DeploymentProfile(
        name="compose-docker",
        description="Local development: Docker Compose",
        execution_authority=ExecutionAuthority.COMPOSE,
        target_class=ComposeTarget,
        wrapper_class=DockerWrapper,
    ),
}


# =============================================================================
# HybridOrchestrator
# =============================================================================


class HybridOrchestrator(OrchestratorBackend):
    """
    Composable orchestrator using ExecutionTarget + RuntimeWrapper.

    Implements the existing OrchestratorBackend interface for backward
    compatibility while leveraging the new composable architecture.

    Key features:
    - Configurable deployment profiles
    - Automatic model reference resolution
    - Clean separation of concerns
    - Support for multiple target/wrapper combinations

    Slurm profiles are experimental, not for HPC production, until the
    RuntimeWrapper launch runs inside the allocated node.
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        profile: str | None = None,
        target: ExecutionTarget | None = None,
        wrapper: RuntimeWrapper | None = None,
        resolver_registry: ResolverRegistry | None = None,
    ):
        """
        Initialize hybrid orchestrator.

        Args:
            config: Orchestrator configuration.
            profile: Name of deployment profile to use.
            target: Custom ExecutionTarget (overrides profile).
            wrapper: Custom RuntimeWrapper (overrides profile).
            resolver_registry: Custom resolver registry.
        """
        super().__init__(config)

        # Determine deployment profile
        if target and wrapper:
            # Custom composition
            self._target = target
            self._wrapper = wrapper
            self._authority = self._detect_authority(target)
        elif profile:
            # Use named profile
            if profile not in DEPLOYMENT_PROFILES:
                raise ValueError(f"Unknown deployment profile: {profile}")

            dp = DEPLOYMENT_PROFILES[profile]
            self._target = dp.target_class(**dp.target_kwargs)
            self._wrapper = dp.wrapper_class(**dp.wrapper_kwargs)
            self._authority = dp.execution_authority
        else:
            # Auto-detect from environment
            self._authority = self._detect_from_env()
            dp = self._get_default_profile(self._authority)
            self._target = dp.target_class(**dp.target_kwargs)
            self._wrapper = dp.wrapper_class(**dp.wrapper_kwargs)

        self._reject_unapproved_hybrid_slurm()

        # Initialize resolver registry
        self._resolvers = resolver_registry or ResolverRegistry.default()

        # Track active jobs (job_id -> allocation, endpoint)
        self._jobs: dict[str, tuple[AllocationInfo, ContainerEndpoint | None]] = {}

        logger.info(
            f"HybridOrchestrator initialized: "
            f"authority={self._authority.value}, "
            f"target={self._target.__class__.__name__}, "
            f"wrapper={self._wrapper.__class__.__name__}"
        )

    @property
    def backend_name(self) -> str:
        return f"hybrid-{self._authority.value}"

    @property
    def execution_authority(self) -> ExecutionAuthority:
        """Return the execution authority."""
        return self._authority

    def _reject_unapproved_hybrid_slurm(self) -> None:
        if self._authority != ExecutionAuthority.SLURM:
            return
        if os.getenv(EXPERIMENTAL_HYBRID_SLURM_ENV, "").lower() == "true":
            return
        raise RuntimeError(
            "Hybrid Slurm orchestration is experimental, not for HPC production; "
            f"use the slurm-podman/slurm backend or set {EXPERIMENTAL_HYBRID_SLURM_ENV}=true "
            "only for compatibility testing."
        )

    async def submit_job(
        self,
        model_path: str | None = None,
        model_ref: ModelRef | None = None,
        **kwargs: Any,
    ) -> JobInfo:
        """
        Submit a new Mini-RiVault job.

        This implements the two-phase orchestration:
        1. ExecutionTarget.submit() - allocate resources
        2. RuntimeWrapper.launch() - start container

        Args:
            model_path: Legacy path (converted to ModelRef internally).
            model_ref: ModelRef from model registry (preferred).
            **kwargs: Additional options.

        Returns:
            JobInfo with job_id and initial status.
        """
        effective_path, mount = await self._resolve_model_mount(model_path, model_ref)
        job_name = kwargs.get("name", f"rivault-{datetime.now().strftime('%H%M%S')}")
        job_spec = self._build_job_spec(job_name, kwargs)

        logger.info(f"Submitting job {job_name} to {self._authority.value}")
        allocation = await self._target.submit(job_spec)
        allocation = await self._target.wait_for_allocation(
            allocation.job_id,
            timeout_sec=self.config.startup_timeout_sec // 2,
        )
        self._mark_allocation_authority(allocation)

        launch_spec = self._build_launch_spec(effective_path, mount, kwargs)
        logger.info(f"Launching container for job {allocation.job_id}")
        endpoint = await self._wrapper.launch(allocation, launch_spec)
        self._jobs[allocation.job_id] = (allocation, endpoint)
        return self._endpoint_job_info(allocation, endpoint, effective_path)

    async def _resolve_model_mount(
        self,
        model_path: str | None,
        model_ref: ModelRef | None,
    ) -> tuple[str, MountSpec]:
        if model_ref:
            resolved = await self._resolvers.resolve(model_ref, self._authority)
            effective_path = resolved.host_path or self.config.model_path
            mount = resolved.to_mount_spec()
        else:
            effective_path = model_path or self.config.model_path
            mount = MountSpec(
                source=effective_path,
                target="/models",
                mount_type="bind",
                read_only=True,
            )
        if not effective_path:
            raise ValueError("No model path specified and no default configured")
        return effective_path, mount

    def _mark_allocation_authority(self, allocation: AllocationInfo) -> None:
        allocation.metadata.setdefault(
            "execution_authority",
            self._authority.value,
        )

    def _build_job_spec(self, job_name: str, kwargs: dict[str, Any]) -> JobSpec:
        return JobSpec(
            name=job_name,
            execution_authority=self._authority.value,
            gpus=kwargs.get("gpus", self.config.slurm_gpus),
            cpus=kwargs.get("cpus", self.config.slurm_cpus),
            memory_mb=kwargs.get("memory_mb", 65536),
            partition=kwargs.get("partition", self.config.slurm_partition),
            time_limit=kwargs.get("time", self.config.slurm_time),
            labels=kwargs.get("labels", {}),
        )

    def _endpoint_job_info(
        self,
        allocation: AllocationInfo,
        endpoint: ContainerEndpoint,
        effective_path: str,
    ) -> JobInfo:
        return JobInfo(
            job_id=allocation.job_id,
            status=JobStatus.STARTING,
            endpoint=BaseEndpointInfo(
                node=endpoint.node,
                litellm_port=endpoint.port,
            ),
            model_path=effective_path,
            created_at=allocation.allocated_at or datetime.now(),
            backend=self.backend_name,
            metadata={
                "authority": self._authority.value,
                "container_id": endpoint.container_id,
            },
        )

    async def get_job_status(self, job_id: str) -> JobInfo:
        """Get current status of a job."""
        allocation = await self._target.status(job_id)
        endpoint = self._cached_endpoint(job_id)
        status = await self._job_status_from_allocation(allocation, endpoint)

        return JobInfo(
            job_id=job_id,
            status=status,
            endpoint=BaseEndpointInfo(
                node=endpoint.node,
                litellm_port=endpoint.port,
            ) if endpoint else None,
            backend=self.backend_name,
            metadata={
                "allocation_status": allocation.status.value,
                "node": allocation.node,
            },
        )

    def _cached_endpoint(self, job_id: str) -> ContainerEndpoint | None:
        cached = self._jobs.get(job_id)
        return cached[1] if cached else None

    async def _job_status_from_allocation(
        self,
        allocation: AllocationInfo,
        endpoint: ContainerEndpoint | None,
    ) -> JobStatus:
        status_map = {
            AllocationStatus.PENDING: JobStatus.PENDING,
            AllocationStatus.ALLOCATED: JobStatus.STARTING,
            AllocationStatus.COMPLETED: JobStatus.STOPPED,
            AllocationStatus.FAILED: JobStatus.FAILED,
            AllocationStatus.CANCELLED: JobStatus.STOPPED,
        }
        if allocation.status != AllocationStatus.RUNNING:
            return status_map.get(allocation.status, JobStatus.UNKNOWN)
        return await self._running_job_status(endpoint)

    async def _running_job_status(
        self,
        endpoint: ContainerEndpoint | None,
    ) -> JobStatus:
        if not endpoint or not await self._wrapper.is_running(endpoint):
            return JobStatus.RUNNING
        try:
            ready = await self._check_readiness(endpoint)
        except Exception:
            return JobStatus.RUNNING
        return JobStatus.READY if ready else JobStatus.RUNNING

    async def wait_for_ready(
        self,
        job_id: str,
        timeout_sec: int | None = None,
    ) -> BaseEndpointInfo:
        """Wait until job is ready to serve requests."""
        timeout = timeout_sec or self.config.startup_timeout_sec

        cached = self._jobs.get(job_id)
        if not cached or not cached[1]:
            raise RuntimeError(f"No endpoint info for job {job_id}")

        endpoint = cached[1]

        await self._wrapper.wait_for_ready(
            endpoint,
            timeout_sec=timeout,
            poll_interval_sec=self.config.readiness_poll_interval_sec,
            auth_key=self.config.litellm_master_key,
        )

        return BaseEndpointInfo(
            node=endpoint.node,
            litellm_port=endpoint.port,
        )

    async def cleanup(self, job_id: str, force: bool = False) -> None:
        """Stop and cleanup a job."""
        cached = self._jobs.get(job_id)

        if cached and cached[1]:
            # Teardown container first
            await self._wrapper.teardown(cached[1])

        # Cancel the allocation
        await self._target.cancel(job_id)

        # Remove from tracking
        self._jobs.pop(job_id, None)

        logger.info(f"Cleaned up job {job_id}")

    async def list_jobs(
        self,
        status_filter: list[JobStatus] | None = None,
    ) -> list[JobInfo]:
        """List all Mini-RiVault jobs."""
        allocations = await self._target.list_jobs(
            labels={"app.kubernetes.io/name": "rivault"}
        )

        jobs = []
        for alloc in allocations:
            info = await self.get_job_status(alloc.job_id)

            if status_filter and info.status not in status_filter:
                continue

            jobs.append(info)

        return jobs

    async def cleanup_orphans(self) -> int:
        """Find and cleanup orphaned resources."""
        # Get all allocations
        allocations = await self._target.list_jobs()

        cleaned = 0
        for alloc in allocations:
            # Check if allocation is stale
            if alloc.status in (AllocationStatus.COMPLETED, AllocationStatus.FAILED):
                await self.cleanup(alloc.job_id, force=True)
                cleaned += 1
            elif alloc.status == AllocationStatus.RUNNING:
                # Check if container is dead
                cached = self._jobs.get(alloc.job_id)
                if cached and cached[1]:
                    if not await self._wrapper.is_running(cached[1]):
                        await self.cleanup(alloc.job_id, force=True)
                        cleaned += 1

        if cleaned:
            logger.info(f"Cleaned up {cleaned} orphaned jobs")

        return cleaned

    def _build_launch_spec(
        self,
        model_path: str,
        mount: MountSpec,
        kwargs: dict[str, Any],
    ) -> LaunchSpec:
        """Build LaunchSpec from config and kwargs."""
        cfg = self.config

        env = {
            "VLLM_MODEL_PATH": "/models",
            "VLLM_PORT": str(cfg.vllm_port),
            "VLLM_MAX_MODEL_LEN": str(cfg.vllm_max_model_len),
            "VLLM_MAX_NUM_SEQS": str(cfg.vllm_max_num_seqs),
            "VLLM_TENSOR_PARALLEL_SIZE": str(cfg.vllm_tensor_parallel_size),
            "VLLM_GPU_MEMORY_UTILIZATION": str(cfg.vllm_gpu_memory_utilization),
            "LITELLM_PORT": str(cfg.litellm_port),
            "LITELLM_MASTER_KEY": cfg.litellm_master_key,
            "TRANSFORMERS_OFFLINE": "1",  # No egress from inference
            "HF_HUB_OFFLINE": "1",
        }
        if cfg.egress_proxy_url:
            env.update({
                "HTTP_PROXY": cfg.egress_proxy_url,
                "HTTPS_PROXY": os.getenv("RIVAULT_HTTPS_PROXY", cfg.egress_proxy_url),
                "http_proxy": cfg.egress_proxy_url,
                "https_proxy": os.getenv("RIVAULT_HTTPS_PROXY", cfg.egress_proxy_url),
                "NO_PROXY": cfg.egress_no_proxy,
                "no_proxy": cfg.egress_no_proxy,
            })
        env.update(kwargs.get("env", {}))

        # Determine image
        if self._authority == ExecutionAuthority.K8S:
            image = kwargs.get("image", cfg.k8s_image)
        elif self._authority == ExecutionAuthority.COMPOSE:
            image = kwargs.get("image", cfg.k8s_image or DEFAULT_MINI_RIVAULT_IMAGE)
        else:  # Slurm
            image = kwargs.get("image", cfg.singularity_image)

        return LaunchSpec(
            image=image,
            model_path=model_path,
            env=env,
            mounts=[mount],
            ports={cfg.litellm_port: cfg.litellm_port},
            gpus=kwargs.get("gpus", cfg.slurm_gpus),
            cpus=kwargs.get("cpus", cfg.slurm_cpus),
            memory_mb=kwargs.get("memory_mb", 65536),
            labels=kwargs.get("labels", {}),
        )

    async def _check_readiness(self, endpoint: ContainerEndpoint) -> bool:
        """Check if endpoint is ready."""
        import aiohttp

        headers = {}
        if self.config.litellm_master_key:
            headers["Authorization"] = f"Bearer {self.config.litellm_master_key}"

        try:
            async with aiohttp.ClientSession() as session, session.get(
                endpoint.models_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _detect_authority(self, target: ExecutionTarget) -> ExecutionAuthority:
        """Detect authority from target type."""
        authority_map = {
            "slurm": ExecutionAuthority.SLURM,
            "k8s": ExecutionAuthority.K8S,
            "compose": ExecutionAuthority.COMPOSE,
        }
        return authority_map.get(target.authority, ExecutionAuthority.COMPOSE)

    def _detect_from_env(self) -> ExecutionAuthority:
        """Auto-detect execution authority from environment."""
        import shutil

        # Check for Slurm
        if shutil.which("sbatch") and os.getenv("SLURM_CLUSTER_NAME"):
            return ExecutionAuthority.SLURM

        # Check for K8s
        if shutil.which("kubectl") and os.path.exists(
            os.path.expanduser("~/.kube/config")
        ):
            return ExecutionAuthority.K8S

        # Default to Compose
        return ExecutionAuthority.COMPOSE

    def _get_default_profile(self, authority: ExecutionAuthority) -> DeploymentProfile:
        """Get default profile for authority."""
        if authority == ExecutionAuthority.SLURM:
            return DEPLOYMENT_PROFILES["slurm-singularity"]
        elif authority == ExecutionAuthority.K8S:
            return DEPLOYMENT_PROFILES["k8s-native"]
        else:
            return DEPLOYMENT_PROFILES["compose-docker"]
