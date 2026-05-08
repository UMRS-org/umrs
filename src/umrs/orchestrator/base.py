"""
Base classes and interfaces for orchestrator backends.

This module defines the abstract interface that all orchestrator backends
(Slurm, Kubernetes, Docker Compose) must implement.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

DEFAULT_MINI_RIVAULT_IMAGE = "mini-rivault:latest"


def _inject_orchestrator_env_defaults() -> None:
    try:
        from .site_config import get_site_config
        get_site_config().inject_env(overwrite=False)
    except Exception:
        pass  # graceful fallback when no site.yaml
    try:
        from .site_profiles import inject_site_profile_env
        inject_site_profile_env(overwrite=False)
    except Exception:
        pass  # graceful fallback when no named profile is selected


def _env_list(name: str, default: str) -> list[str]:
    return os.getenv(name, default).replace(",", " ").split()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}



def _env_nonempty(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return value.strip()

class JobStatus(Enum):
    """Status of a Mini-RiVault job."""

    PENDING = "pending"  # Waiting for resources
    STARTING = "starting"  # Resources allocated, container starting
    READY = "ready"  # LiteLLM endpoint is responding
    RUNNING = "running"  # Actively serving requests
    FAILED = "failed"  # Job failed to start or crashed
    STOPPED = "stopped"  # Intentionally stopped
    UNKNOWN = "unknown"  # Cannot determine status


@dataclass
class EndpointInfo:
    """Information about an active Mini-RiVault endpoint."""

    node: str  # Hostname or IP
    litellm_port: int = 11434
    vllm_port: int = 8000  # Internal, not exposed
    model_name: str = "rivault-gpt-oss-static"

    @property
    def litellm_url(self) -> str:
        """OpenAI-compatible API URL."""
        return f"http://{self.node}:{self.litellm_port}/v1"

    @property
    def health_url(self) -> str:
        """Health check URL."""
        return f"http://{self.node}:{self.litellm_port}/health"

    @property
    def models_url(self) -> str:
        """Models list URL (used for readiness check)."""
        return f"http://{self.node}:{self.litellm_port}/v1/models"


@dataclass
class JobInfo:
    """Information about a Mini-RiVault job."""

    job_id: str  # Backend-specific ID (Slurm job ID, K8s Job name, etc.)
    status: JobStatus
    endpoint: EndpointInfo | None = None
    model_path: str = ""
    created_at: datetime | None = None
    ready_at: datetime | None = None
    stopped_at: datetime | None = None
    error_message: str | None = None
    backend: str = "unknown"  # "slurm", "kubernetes", "docker-compose"
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if job is in an active state."""
        return self.status in (JobStatus.PENDING, JobStatus.STARTING, JobStatus.READY, JobStatus.RUNNING)

    def is_ready(self) -> bool:
        """Check if job is ready to serve requests."""
        return self.status in (JobStatus.READY, JobStatus.RUNNING)


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator backends."""

    # Common settings
    model_path: str = ""
    litellm_master_key: str = ""
    litellm_port: int = 11434
    vllm_port: int = 8000
    vllm_max_model_len: int = 4096
    vllm_max_num_seqs: int = 4
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_enable_prefix_caching: bool = False

    # Timeouts
    startup_timeout_sec: int = 600  # 10 minutes
    idle_timeout_sec: int = 3600  # 1 hour
    readiness_poll_interval_sec: int = 5

    # Slurm-specific
    slurm_partition: str = "gpu"
    slurm_time: str = "02:00:00"
    slurm_gpus: int = 1
    slurm_cpus: int = 8
    slurm_mem: str = "64G"
    slurm_log_dir: str = "/var/log/rivault"
    slurm_job_info_dir: str = "/var/run/rivault"
    slurm_modules: list[str] = field(default_factory=lambda: ["cuda/12.1", "singularity"])
    singularity_image: str = "/opt/singularity/mini-rivault.sif"

    # Kubernetes-specific
    k8s_namespace: str = "rivault-system"
    k8s_image: str = DEFAULT_MINI_RIVAULT_IMAGE
    k8s_gpu_resource: str = "nvidia.com/gpu"
    k8s_storage_class: str = ""
    k8s_model_pvc: str = "rivault-models"

    # Docker Compose-specific (development)
    compose_project_name: str = "mini-rivault"
    compose_file: str = "compose.yaml"
    compose_nvidia_file: str = "compose.nvidia.yaml"

    # EgressGuard proxy settings for mini-RiVault runtimes
    egress_proxy_url: str = "http://egress-guard:8888"
    egress_no_proxy: str = (
        "localhost,127.0.0.1,::1,litellm,vllm,ll-postgres,coredns,nginx,egress-guard"
    )

    @classmethod
    def from_env(cls) -> OrchestratorConfig:
        """Load configuration from environment variables.

        If a ``site.yaml`` is available, its values are injected into
        ``os.environ`` first (without overwriting existing vars), so
        all the ``os.getenv`` calls below pick them up transparently.
        """
        _inject_orchestrator_env_defaults()

        return cls(
            # Common
            model_path=os.getenv("VLLM_MODEL_PATH", ""),
            litellm_master_key=os.getenv("LITELLM_MASTER_KEY", ""),
            litellm_port=int(os.getenv("LITELLM_PORT", "11434")),
            vllm_port=int(os.getenv("VLLM_PORT", "8000")),
            vllm_max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "4096")),
            vllm_max_num_seqs=int(os.getenv("VLLM_MAX_NUM_SEQS", "4")),
            vllm_tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            vllm_gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            vllm_enable_prefix_caching=_env_bool("VLLM_ENABLE_PREFIX_CACHING", False),
            # Timeouts
            startup_timeout_sec=int(os.getenv("RIVAULT_STARTUP_TIMEOUT_SEC", "600")),
            idle_timeout_sec=int(os.getenv("RIVAULT_IDLE_TIMEOUT_SEC", "3600")),
            readiness_poll_interval_sec=int(os.getenv("RIVAULT_READINESS_POLL_SEC", "5")),
            # Slurm
            slurm_partition=os.getenv("SLURM_PARTITION", "gpu"),
            slurm_time=os.getenv("SLURM_TIME", "02:00:00"),
            slurm_gpus=int(os.getenv("SLURM_GPUS", "1")),
            slurm_cpus=int(os.getenv("SLURM_CPUS", "8")),
            slurm_mem=os.getenv("SLURM_MEM", "64G"),
            slurm_log_dir=os.getenv("SLURM_LOG_DIR", "/var/log/rivault"),
            slurm_job_info_dir=os.getenv("SLURM_JOB_INFO_DIR", "/var/run/rivault"),
            slurm_modules=_env_list("SLURM_MODULES", "cuda/12.1 singularity"),
            singularity_image=os.getenv("SINGULARITY_IMAGE", "/opt/singularity/mini-rivault.sif"),
            # Kubernetes
            k8s_namespace=os.getenv("K8S_NAMESPACE", "rivault-system"),
            k8s_image=os.getenv("K8S_MINI_RIVAULT_IMAGE", DEFAULT_MINI_RIVAULT_IMAGE),
            k8s_gpu_resource=os.getenv("K8S_GPU_RESOURCE", "nvidia.com/gpu"),
            k8s_storage_class=os.getenv("K8S_STORAGE_CLASS", ""),
            k8s_model_pvc=os.getenv("K8S_MODEL_PVC", "rivault-models"),
            # Docker Compose
            compose_project_name=os.getenv("COMPOSE_PROJECT_NAME", "mini-rivault"),
            compose_file=os.getenv("COMPOSE_FILE", "compose.yaml"),
            compose_nvidia_file=os.getenv("COMPOSE_NVIDIA_FILE", "compose.nvidia.yaml"),
            # EgressGuard
            egress_proxy_url=_env_nonempty(
                "RIVAULT_HTTP_PROXY",
                _env_nonempty("EGRESS_GUARD_PROXY_URL", "http://egress-guard:8888"),
            ),
            egress_no_proxy=_env_nonempty(
                "RIVAULT_NO_PROXY",
                "localhost,127.0.0.1,::1,litellm,vllm,ll-postgres,coredns,nginx,egress-guard",
            ),
        )


class OrchestratorBackend(ABC):
    """
    Abstract base class for orchestrator backends.

    All backends must implement these methods to manage Mini-RiVault
    inference endpoint lifecycle.
    """

    def __init__(self, config: OrchestratorConfig | None = None):
        self.config = config or OrchestratorConfig.from_env()

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name (e.g., 'slurm', 'kubernetes')."""
        ...

    @abstractmethod
    async def submit_job(
        self,
        model_path: str | None = None,
        **kwargs: Any,
    ) -> JobInfo:
        """
        Submit a new Mini-RiVault job.

        Args:
            model_path: Path to the pre-staged model. Uses config default if None.
            **kwargs: Backend-specific options.

        Returns:
            JobInfo with job_id and initial status.
        """
        ...

    @abstractmethod
    async def get_job_status(self, job_id: str) -> JobInfo:
        """
        Get current status of a job.

        Args:
            job_id: The job identifier returned by submit_job.

        Returns:
            JobInfo with current status and endpoint info if ready.
        """
        ...

    @abstractmethod
    async def wait_for_ready(
        self,
        job_id: str,
        timeout_sec: int | None = None,
    ) -> EndpointInfo:
        """
        Wait until job is ready to serve requests.

        Args:
            job_id: The job identifier.
            timeout_sec: Override default startup timeout.

        Returns:
            EndpointInfo when ready.

        Raises:
            TimeoutError: If job doesn't become ready within timeout.
            RuntimeError: If job fails during startup.
        """
        ...

    @abstractmethod
    async def cleanup(self, job_id: str, force: bool = False) -> None:
        """
        Stop and cleanup a job.

        Args:
            job_id: The job identifier.
            force: Force immediate termination without graceful shutdown.
        """
        ...

    @abstractmethod
    async def list_jobs(
        self,
        status_filter: list[JobStatus] | None = None,
    ) -> list[JobInfo]:
        """
        List all Mini-RiVault jobs.

        Args:
            status_filter: Only return jobs with these statuses.

        Returns:
            List of JobInfo objects.
        """
        ...

    @abstractmethod
    async def cleanup_orphans(self) -> int:
        """
        Find and cleanup orphaned resources (stale jobs, containers, etc.).

        Returns:
            Number of orphaned resources cleaned up.
        """
        ...

    async def check_readiness(self, endpoint: EndpointInfo) -> bool:
        """
        Check if an endpoint is ready to serve requests.

        Default implementation calls GET /v1/models with auth.
        Backends can override for custom checks.
        """
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

    async def health_check(self) -> dict[str, Any]:
        """
        Check health of the orchestrator backend itself.

        Returns:
            Dict with backend-specific health info.
        """
        return {
            "backend": self.backend_name,
            "healthy": True,
            "message": "Backend operational",
        }
