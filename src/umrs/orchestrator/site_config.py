"""
Site Configuration — declarative, per-site deployment settings.

Replaces hardcoded paths, partition names, and GPU parameters with a single
YAML file per deployment site (e.g. ``deploy/rccs-cloud/site.yaml``).

Loading order (first found wins):
  1. ``UMRS_SITE_CONFIG`` environment variable (absolute path)
  2. ``./site.yaml`` (current working directory)
  3. ``~/.config/umrs/site.yaml``

Usage::

    from umrs.orchestrator.site_config import get_site_config

    cfg = get_site_config()
    print(cfg.scheduler.partition)      # "ng-dgx-s"
    print(cfg.storage.model_cache_dir)  # "/lvs0/rccs-dmp/…/umrs_models"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """Cluster job scheduler settings."""

    kind: str = "slurm"                         # slurm | k8s | compose
    partition: str = ""                         # Slurm partition / K8s namespace
    account: str = ""                           # Slurm account
    qos: str = ""                               # Slurm QoS
    time_limit: str = "04:00:00"                # Default wall-clock
    gpus: int = 1
    cpus: int = 8
    memory: str = "64G"
    extra_sbatch: List[str] = field(default_factory=list)  # e.g. ["--constraint=a100"]


@dataclass
class RuntimeConfig:
    """Container runtime settings."""

    kind: str = "podman-compose"                # podman-compose | singularity | docker-compose
    compose_file: str = ""                      # Path to compose YAML
    rivault_dir: str = ""                       # Path to mini-rivault-llm directory
    repo_dir: str = ""                          # Path to UMRS repo root (for builds)
    vision_image: str = "umrs-vision:latest"    # Vision container image name
    profiles: List[str] = field(default_factory=lambda: ["harness", "vllm", "vision-bentoml"])
    gpu_passthrough: str = "auto"               # auto | cdi | device
    cdi_spec: str = "/var/run/cdi/nvidia.yaml"  # CDI spec path (if gpu_passthrough=cdi)
    cgroup_manager: str = "cgroupfs"
    storage_root: str = ""                      # podman graphroot override


@dataclass
class StorageConfig:
    """Filesystem / object-storage paths."""

    model_cache_dir: str = ""                   # HuggingFace / vLLM model cache
    sam3_model_path: str = ""
    hmae_model_path: str = ""
    hmae_config_path: str = ""
    hmae_checkpoint: str = ""
    vision_tarball: str = ""                    # Pre-built vision image tarball
    user_model_path: str = "/tmp"
    s3_tunnel_host: str = ""                    # MinIO tunnel target host
    s3_tunnel_port: int = 9000


@dataclass
class NetworkConfig:
    """Networking and port settings."""

    litellm_port: int = 8001                    # nginx → LiteLLM port inside compose
    litellm_master_key: str = ""
    port_range_min: int = 11434
    port_range_max: int = 11534
    port_strategy: str = "static_range"         # static_range | registry_lock


@dataclass
class VllmConfig:
    """vLLM-specific settings."""

    model: str = "intfloat/e5-base-v2"
    docker_tag: str = "latest"
    gpu_mem_util: float = 0.10
    max_model_len: int = 4096
    max_num_seqs: int = 4
    tensor_parallel_size: int = 1


@dataclass
class DatabaseConfig:
    """PostgreSQL connection settings."""

    host: str = "localhost"
    port: int = 5432
    superuser: str = "postgres"
    superuser_password: str = "postgres"
    user_password: str = ""
    registry_dsn: str = ""                      # UMRS_REGISTRY_POSTGRES_DSN


@dataclass
class SiteConfig:
    """
    Top-level site configuration.

    One instance per UMRS deployment (rccs-cloud, plafrim, laptop, …).
    """

    name: str = "default"
    description: str = ""
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    extra: Dict[str, Any] = field(default_factory=dict)

    # --- helpers -----------------------------------------------------------

    def to_env(self) -> Dict[str, str]:
        """Export SiteConfig as a flat dict of environment variables.

        This allows existing code that reads ``os.environ`` to work with
        values loaded from ``site.yaml`` without refactoring every call-site.
        """
        env: Dict[str, str] = {}

        # Scheduler
        env["SLURM_PARTITION"] = self.scheduler.partition
        env["SLURM_TIME"] = self.scheduler.time_limit
        env["SLURM_GPUS"] = str(self.scheduler.gpus)
        env["SLURM_CPUS"] = str(self.scheduler.cpus)
        env["SLURM_MEM"] = self.scheduler.memory

        # Runtime
        if self.runtime.rivault_dir:
            env["RIVAULT_DIR"] = self.runtime.rivault_dir
        if self.runtime.repo_dir:
            env["UMRS_REPO"] = self.runtime.repo_dir
        if self.runtime.vision_image:
            env["BENTOML_VISION_IMAGE"] = self.runtime.vision_image
        if self.runtime.profiles:
            env["RIVAULT_PROFILES"] = " ".join(self.runtime.profiles)

        # Storage
        if self.storage.model_cache_dir:
            env["VLLM_PATH_TO_HUGGINGFACE"] = self.storage.model_cache_dir
        if self.storage.sam3_model_path:
            env["SAM3_MODEL_PATH"] = self.storage.sam3_model_path
        if self.storage.hmae_model_path:
            env["HMAE_MODEL_PATH"] = self.storage.hmae_model_path
        if self.storage.hmae_config_path:
            env["HMAE_CONFIG_PATH"] = self.storage.hmae_config_path
        if self.storage.hmae_checkpoint:
            env["HMAE_CHECKPOINT"] = self.storage.hmae_checkpoint
        if self.storage.vision_tarball:
            env["VISION_TARBALL"] = self.storage.vision_tarball
        env["USER_MODEL_PATH"] = self.storage.user_model_path
        if self.storage.s3_tunnel_host:
            env["UMRS_S3_TUNNEL_HOST"] = self.storage.s3_tunnel_host
        env["UMRS_S3_TUNNEL_PORT"] = str(self.storage.s3_tunnel_port)

        # Network
        env["NGINX_LITELLM_DOCKER_PORT"] = str(self.network.litellm_port)
        if self.network.litellm_master_key:
            env["LITELLM_MASTER_KEY"] = self.network.litellm_master_key
        env["RIVAULT_PORT_MIN"] = str(self.network.port_range_min)
        env["RIVAULT_PORT_MAX"] = str(self.network.port_range_max)
        env["RIVAULT_PORT_STRATEGY"] = self.network.port_strategy
        env["RIVAULT_REMOTE_PORT"] = str(self.network.litellm_port)

        # vLLM
        env["VLLM_MODEL"] = self.vllm.model
        env["VLLM_DOCKER_TAG"] = self.vllm.docker_tag
        env["VLLM_GPU_MEM_UTIL"] = str(self.vllm.gpu_mem_util)
        env["VLLM_MAX_MODEL_LEN"] = str(self.vllm.max_model_len)
        env["VLLM_MAX_NUM_SEQS"] = str(self.vllm.max_num_seqs)
        env["VLLM_TENSOR_PARALLEL_SIZE"] = str(self.vllm.tensor_parallel_size)
        env["VLLM_GPU_MEMORY_UTILIZATION"] = str(self.vllm.gpu_mem_util)

        # Database
        env["DB_HOST"] = self.database.host
        env["DB_PORT"] = str(self.database.port)
        if self.database.superuser:
            env["DB_SUPERUSER"] = self.database.superuser
            env["POSTGRES_USER"] = self.database.superuser
        if self.database.superuser_password:
            env["DB_SUPERUSER_PASSWORD"] = self.database.superuser_password
            env["POSTGRES_PASSWORD"] = self.database.superuser_password
        if self.database.user_password:
            env["DB_USER_PASSWORD"] = self.database.user_password
        if self.database.registry_dsn:
            env["UMRS_REGISTRY_POSTGRES_DSN"] = self.database.registry_dsn

        return {k: v for k, v in env.items() if v}

    def inject_env(self, overwrite: bool = False) -> None:
        """Write :meth:`to_env` values into ``os.environ``.

        Parameters
        ----------
        overwrite
            If *False* (default), existing env vars take precedence.
        """
        for key, value in self.to_env().items():
            if overwrite or key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_SEARCH_PATHS = [
    lambda: os.environ.get("UMRS_SITE_CONFIG", ""),
    lambda: str(Path.cwd() / "site.yaml"),
    lambda: str(Path.home() / ".config" / "umrs" / "site.yaml"),
]


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _dict_to_dataclass(cls, data: dict):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in field_names})


def load_site_config(path: Optional[str] = None) -> SiteConfig:
    """Load a SiteConfig from a YAML file.

    Parameters
    ----------
    path
        Explicit file path.  When *None*, the standard search paths are tried.

    Raises
    ------
    FileNotFoundError
        If no site.yaml is found.
    """
    import yaml  # deferred to avoid hard dep when unused

    resolved: Optional[str] = None

    if path:
        resolved = path
    else:
        for path_fn in _SEARCH_PATHS:
            candidate = path_fn()
            if candidate and os.path.isfile(candidate):
                resolved = candidate
                break

    if not resolved or not os.path.isfile(resolved):
        raise FileNotFoundError(
            "No site.yaml found.  Set UMRS_SITE_CONFIG or place site.yaml in "
            "the working directory or ~/.config/umrs/."
        )

    with open(resolved, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    logger.info("Loaded site config from %s", resolved)
    return _parse_raw(raw)


def _parse_raw(raw: Dict[str, Any]) -> SiteConfig:
    """Turn a raw YAML dict into a :class:`SiteConfig`."""
    return SiteConfig(
        name=raw.get("name", "default"),
        description=raw.get("description", ""),
        scheduler=_dict_to_dataclass(SchedulerConfig, raw.get("scheduler", {})),
        runtime=_dict_to_dataclass(RuntimeConfig, raw.get("runtime", {})),
        storage=_dict_to_dataclass(StorageConfig, raw.get("storage", {})),
        network=_dict_to_dataclass(NetworkConfig, raw.get("network", {})),
        vllm=_dict_to_dataclass(VllmConfig, raw.get("vllm", {})),
        database=_dict_to_dataclass(DatabaseConfig, raw.get("database", {})),
        extra=raw.get("extra", {}),
    )


def default_site_config() -> SiteConfig:
    """Return a SiteConfig populated from environment variables only.

    This is the fallback when no ``site.yaml`` exists — it ensures that
    existing env-var-based deployments continue to work unchanged.
    """
    cfg = SiteConfig(name="env-fallback")

    # Scheduler
    cfg.scheduler.kind = os.getenv("RIVAULT_ORCHESTRATOR", "slurm")
    cfg.scheduler.partition = os.getenv("SLURM_PARTITION", "")
    cfg.scheduler.time_limit = os.getenv("SLURM_TIME", "04:00:00")
    cfg.scheduler.gpus = int(os.getenv("SLURM_GPUS", "1"))
    cfg.scheduler.cpus = int(os.getenv("SLURM_CPUS", "8"))
    cfg.scheduler.memory = os.getenv("SLURM_MEM", "64G")

    # Runtime
    cfg.runtime.rivault_dir = os.getenv("RIVAULT_DIR", "")
    cfg.runtime.repo_dir = os.getenv("UMRS_REPO", "")
    cfg.runtime.vision_image = os.getenv("BENTOML_VISION_IMAGE", "umrs-vision:latest")
    profiles = os.getenv("RIVAULT_PROFILES", "")
    if profiles:
        cfg.runtime.profiles = profiles.split()

    # Storage
    cfg.storage.model_cache_dir = os.getenv("VLLM_PATH_TO_HUGGINGFACE", "")
    cfg.storage.sam3_model_path = os.getenv("SAM3_MODEL_PATH", "")
    cfg.storage.hmae_model_path = os.getenv("HMAE_MODEL_PATH", "")
    cfg.storage.hmae_config_path = os.getenv("HMAE_CONFIG_PATH", "")
    cfg.storage.hmae_checkpoint = os.getenv("HMAE_CHECKPOINT", "")
    cfg.storage.vision_tarball = os.getenv("VISION_TARBALL", "")
    cfg.storage.s3_tunnel_host = os.getenv("UMRS_S3_TUNNEL_HOST", "")
    cfg.storage.s3_tunnel_port = int(os.getenv("UMRS_S3_TUNNEL_PORT", "9000"))

    # Network
    cfg.network.litellm_port = int(os.getenv("NGINX_LITELLM_DOCKER_PORT",
                                              os.getenv("RIVAULT_REMOTE_PORT", "8001")))
    cfg.network.litellm_master_key = os.getenv("LITELLM_MASTER_KEY", "")
    cfg.network.port_range_min = int(os.getenv("RIVAULT_PORT_MIN", "11434"))
    cfg.network.port_range_max = int(os.getenv("RIVAULT_PORT_MAX", "11534"))
    cfg.network.port_strategy = os.getenv("RIVAULT_PORT_STRATEGY", "static_range")

    # vLLM
    cfg.vllm.model = os.getenv("VLLM_MODEL", "intfloat/e5-base-v2")
    cfg.vllm.docker_tag = os.getenv("VLLM_DOCKER_TAG", "latest")
    cfg.vllm.gpu_mem_util = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.10"))

    # Database
    cfg.database.host = os.getenv("DB_HOST", "localhost")
    cfg.database.port = int(os.getenv("DB_PORT", "5432"))
    cfg.database.superuser = os.getenv("DB_SUPERUSER",
                                       os.getenv("POSTGRES_USER", "postgres"))
    cfg.database.superuser_password = os.getenv("DB_SUPERUSER_PASSWORD",
                                                os.getenv("POSTGRES_PASSWORD", "postgres"))
    cfg.database.registry_dsn = os.getenv("UMRS_REGISTRY_POSTGRES_DSN", "")

    return cfg


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_site_config: Optional[SiteConfig] = None


def get_site_config(path: Optional[str] = None) -> SiteConfig:
    """Return the global site config singleton.

    On first call, tries to load ``site.yaml``; if not found, falls back to
    :func:`default_site_config` (env-var only).  Subsequent calls return the
    cached instance.
    """
    global _site_config
    if _site_config is not None:
        return _site_config

    try:
        _site_config = load_site_config(path)
    except FileNotFoundError:
        logger.info("No site.yaml found — falling back to environment variables")
        _site_config = default_site_config()

    return _site_config


def reset_site_config() -> None:
    """Clear the cached singleton (useful in tests)."""
    global _site_config
    _site_config = None
