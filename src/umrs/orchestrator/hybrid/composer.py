"""
HybridOrchestrator Composer.

Combines ExecutionTarget + RuntimeWrapper + ModelRefResolver into a unified
orchestration layer that implements the existing OrchestratorBackend interface
while leveraging the new composable architecture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from ..base import (
    DEFAULT_MINI_RIVAULT_IMAGE,
    EndpointInfo as BaseEndpointInfo,
    JobInfo,
    JobStatus,
    OrchestratorBackend,
    OrchestratorConfig,
)
from umrs.model_registry.models import ModelRef, ExecutionAuthority

from .targets import (
    ExecutionTarget,
    SlurmTarget,
    K8sTarget,
    ComposeTarget,
    AllocationInfo,
    AllocationStatus,
    JobSpec,
)
from .wrappers import (
    RuntimeWrapper,
    SingularityWrapper,
    DockerWrapper,
    K8sJobWrapper,
    LaunchSpec,
    MountSpec,
    ContainerEndpoint,
)
from .resolvers import (
    ResolverRegistry,
)

logger = logging.getLogger(__name__)


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
    target_class: Type[ExecutionTarget]
    wrapper_class: Type[RuntimeWrapper]
    target_kwargs: Dict[str, Any] = field(default_factory=dict)
    wrapper_kwargs: Dict[str, Any] = field(default_factory=dict)


# Predefined deployment profiles
DEPLOYMENT_PROFILES: Dict[str, DeploymentProfile] = {
    "slurm-singularity": DeploymentProfile(
        name="slurm-singularity",
        description="Production HPC: Slurm + Singularity",
        execution_authority=ExecutionAuthority.SLURM,
        target_class=SlurmTarget,
        wrapper_class=SingularityWrapper,
    ),
    "slurm-docker": DeploymentProfile(
        name="slurm-docker",
        description="HPC dev: Slurm + Docker (when Singularity unavailable)",
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
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        profile: Optional[str] = None,
        target: Optional[ExecutionTarget] = None,
        wrapper: Optional[RuntimeWrapper] = None,
        resolver_registry: Optional[ResolverRegistry] = None,
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

        # Initialize resolver registry
        self._resolvers = resolver_registry or ResolverRegistry.default()

        # Track active jobs (job_id -> allocation, endpoint)
        self._jobs: Dict[str, tuple[AllocationInfo, Optional[ContainerEndpoint]]] = {}

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

    async def submit_job(
        self,
        model_path: Optional[str] = None,
        model_ref: Optional[ModelRef] = None,
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
        # Resolve model reference
        if model_ref:
            resolved = await self._resolvers.resolve(model_ref, self._authority)
            effective_path = resolved.host_path or self.config.model_path
            mount = resolved.to_mount_spec()
        elif model_path:
            effective_path = model_path
            mount = MountSpec(
                source=model_path,
                target="/models",
                mount_type="bind",
                read_only=True,
            )
        else:
            effective_path = self.config.model_path
            mount = MountSpec(
                source=self.config.model_path,
                target="/models",
                mount_type="bind",
                read_only=True,
            )

        if not effective_path:
            raise ValueError("No model path specified and no default configured")

        # Build job spec for ExecutionTarget
        job_name = kwargs.get("name", f"rivault-{datetime.now().strftime('%H%M%S')}")
        job_spec = JobSpec(
            name=job_name,
            execution_authority=self._authority.value,
            gpus=kwargs.get("gpus", self.config.slurm_gpus),
            cpus=kwargs.get("cpus", self.config.slurm_cpus),
            memory_mb=kwargs.get("memory_mb", 65536),
            partition=kwargs.get("partition", self.config.slurm_partition),
            time_limit=kwargs.get("time", self.config.slurm_time),
            labels=kwargs.get("labels", {}),
        )

        # Phase 1: Allocate resources
        logger.info(f"Submitting job {job_name} to {self._authority.value}")
        allocation = await self._target.submit(job_spec)

        # Wait for allocation
        allocation = await self._target.wait_for_allocation(
            allocation.job_id,
            timeout_sec=self.config.startup_timeout_sec // 2,
        )

        # Build launch spec for RuntimeWrapper
        launch_spec = self._build_launch_spec(effective_path, mount, kwargs)

        # Phase 2: Launch container
        logger.info(f"Launching container for job {allocation.job_id}")
        endpoint = await self._wrapper.launch(allocation, launch_spec)

        # Track the job
        self._jobs[allocation.job_id] = (allocation, endpoint)

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
        # Get allocation status
        allocation = await self._target.status(job_id)

        # Get cached endpoint info
        cached = self._jobs.get(job_id)
        endpoint = cached[1] if cached else None

        # Determine job status
        if allocation.status == AllocationStatus.PENDING:
            status = JobStatus.PENDING
        elif allocation.status == AllocationStatus.ALLOCATED:
            status = JobStatus.STARTING
        elif allocation.status == AllocationStatus.RUNNING:
            # Check if container is responsive
            if endpoint and await self._wrapper.is_running(endpoint):
                # Check readiness
                try:
                    ready = await self._check_readiness(endpoint)
                    status = JobStatus.READY if ready else JobStatus.RUNNING
                except Exception:
                    status = JobStatus.RUNNING
            else:
                status = JobStatus.RUNNING
        elif allocation.status == AllocationStatus.COMPLETED:
            status = JobStatus.STOPPED
        elif allocation.status == AllocationStatus.FAILED:
            status = JobStatus.FAILED
        elif allocation.status == AllocationStatus.CANCELLED:
            status = JobStatus.STOPPED
        else:
            status = JobStatus.UNKNOWN

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

    async def wait_for_ready(
        self,
        job_id: str,
        timeout_sec: Optional[int] = None,
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
        status_filter: Optional[List[JobStatus]] = None,
    ) -> List[JobInfo]:
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
        kwargs: Dict[str, Any],
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
        }
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
            async with aiohttp.ClientSession() as session:
                async with session.get(
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
        import os
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
