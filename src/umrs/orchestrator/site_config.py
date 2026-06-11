"""
Site Configuration — declarative, per-site deployment settings.

Replaces hardcoded paths, partition names, and GPU parameters with a single
YAML file per deployment site (for example ``deploy/rccs-cloud/site.yaml`` or
``deploy/hpbd-atom/site.yaml``).

Loading order (first found wins):
  1. ``UMRS_SITE_CONFIG`` environment variable (absolute path)
  2. ``./site.yaml`` (current working directory)
    3. ``<resolved UMRS config dir>/site.yaml``

Usage::

    from umrs.orchestrator.site_config import get_site_config

    cfg = get_site_config()
    print(cfg.scheduler.partition)      # "gpu-a100"
    print(cfg.storage.model_cache_dir)  # "/shared/umrs/models"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from umrs.shared.config import get_resolved_umrs_subdir

logger = logging.getLogger(__name__)


def _split_partitions(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _expand_env_refs(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_env_refs(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_refs(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SchedulerConfig:
    """Cluster job scheduler settings."""

    kind: str = "slurm"  # slurm | slurm-remote | k8s | compose
    partition: str = ""  # Slurm partition or comma-separated candidates / K8s namespace
    account: str = ""  # Slurm account
    qos: str = ""  # Slurm QoS
    time_limit: str = "04:00:00"  # Default wall-clock
    gpus: int = 1
    cpus: int = 8
    memory: str = "64G"
    nodes: int = 1  # Number of Slurm nodes (>1 for multi-node PP)
    extra_sbatch: list[str] = field(default_factory=list)  # e.g. ["--constraint=a100"]

    # Per-user SSH delegation for Slurm job submission.
    # - service_account: UMRS daemon submits all jobs as itself (default)
    # - per_user_ssh: UMRS SSH's as the requesting user to submit sbatch
    slurm_delegation: str = "service_account"  # service_account | per_user_ssh
    # Path to UMRS service ed25519 public key on shared filesystem.
    # Users enroll by appending this key to their ~/.ssh/authorized_keys.
    service_pubkey_path: str = ""
    # Path to the forced-command proxy script on shared filesystem.
    # Only allows sbatch, squeue, sacct, scancel commands.
    slurm_proxy_path: str = ""

    # When True, endpoint provisioning rejects users who have no per-user
    # Slurm account configured (HTTP 403).  When False (default), jobs
    # fall back to ``account`` (the site master account) so users without
    # login-node accounts can still submit jobs via UMRS.
    require_user_slurm_account: bool = False


@dataclass
class RuntimeConfig:
    """Container runtime settings."""

    kind: str = "podman-compose"  # podman-compose | singularity | docker-compose
    compose_file: str = ""  # Path to compose YAML
    rivault_dir: str = ""  # Path to mini-rivault-llm directory
    repo_dir: str = ""  # Path to UMRS repo root (for builds)
    vision_image: str = "umrs-vision:latest"  # Vision container image name
    profiles: list[str] = field(
        default_factory=lambda: ["harness", "vllm", "vision-bentoml"]
    )
    gpu_passthrough: str = "auto"  # auto | cdi | device
    cdi_spec: str = "/var/run/cdi/nvidia.yaml"  # CDI spec path (if gpu_passthrough=cdi)
    cgroup_manager: str = "cgroupfs"
    storage_root: str = ""  # podman graphroot override


@dataclass
class StorageConfig:
    """Filesystem / object-storage paths."""

    model_cache_dir: str = ""  # HuggingFace / vLLM model cache
    sam3_model_path: str = ""
    hmae_model_path: str = ""
    hmae_config_path: str = ""
    hmae_checkpoint: str = ""
    vision_tarball: str = ""  # Pre-built vision image tarball
    user_model_path: str = "/tmp"
    s3_tunnel_host: str = ""  # MinIO tunnel target host
    s3_tunnel_port: int = 9000


@dataclass
class NetworkConfig:
    """Networking and port settings."""

    litellm_port: int = 8001  # nginx → LiteLLM port inside compose
    litellm_master_key: str = ""
    port_range_min: int = 11434
    port_range_max: int = 11534
    port_strategy: str = "static_range"  # static_range | registry_lock
    egress_proxy_url: str = ""  # Forward proxy URL for mini-rivault egress
    egress_no_proxy: str = ""  # Comma-separated internal hosts bypassing proxy


@dataclass
class VllmConfig:
    """vLLM-specific settings."""

    model: str = "intfloat/e5-base-v2"
    docker_tag: str = "latest"
    gpu_mem_util: float = 0.10
    max_model_len: int = 4096
    max_num_seqs: int = 4
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool = False


@dataclass
class PartitionProfile:
    """Declarative per-partition hardware and runtime profile.

    Eliminates fragile runtime detection of OCI runtime, userns_mode,
    network backend, and GPU vendor.  The wrapper looks up the profile
    for the assigned Slurm partition and applies the values directly.
    """

    arch: str = "x86_64"  # x86_64 | aarch64
    gpu_vendor: str = "nvidia"  # nvidia | amd | none
    gpu_memory_gb: float = 0  # per-GPU VRAM in GiB (0 = unknown)
    oci_runtime: str = "crun"  # crun | runc
    userns_mode: str = "keep-id:uid=0,gid=0"  # keep-id:uid=0,gid=0 | host
    network_backend: str = "netavark"  # netavark | cni
    vllm_docker_tag: str = ""  # override per partition (empty → use site default)
    gpus: int = 1  # Slurm --gpus count
    needs_netavark_switch: bool = True  # force netavark backend file on startup


@dataclass
class DatabaseConfig:
    """PostgreSQL connection settings."""

    host: str = "localhost"
    port: int = 5432
    superuser: str = "postgres"
    superuser_password: str = "postgres"
    user_password: str = ""
    registry_dsn: str = ""  # UMRS_REGISTRY_POSTGRES_DSN


@dataclass
class SiteConfig:
    """
    Top-level site configuration.

    One instance per UMRS deployment site (cluster, workstation, laptop, …).
    """

    name: str = "default"
    description: str = ""
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    vllm: VllmConfig = field(default_factory=VllmConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    partition_profiles: dict[str, PartitionProfile] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # --- helpers -----------------------------------------------------------

    def get_partition_profile(self, partition: str) -> PartitionProfile | None:
        """Look up a partition profile by name.

        Returns ``None`` when no profile is registered for *partition*.
        """
        return self.partition_profiles.get(partition)

    def partition_env(self, partition: str) -> dict[str, str]:
        """Return env vars for a specific partition profile.

        Useful for the wrapper to apply declarative overrides
        (userns_mode, OCI runtime, docker_tag, etc.) without runtime detection.
        Returns an empty dict when no profile is registered.
        """
        pp = self.get_partition_profile(partition)
        if pp is None:
            return {}
        env: dict[str, str] = {
            "RIVAULT_PARTITION_ARCH": pp.arch,
            "RIVAULT_PARTITION_GPU_VENDOR": pp.gpu_vendor,
            "RIVAULT_PARTITION_OCI_RUNTIME": pp.oci_runtime,
            "RIVAULT_USERNS_MODE": pp.userns_mode,
            "RIVAULT_PARTITION_NETWORK_BACKEND": pp.network_backend,
            "RIVAULT_PARTITION_GPUS": str(pp.gpus),
            "RIVAULT_PARTITION_NEEDS_NETAVARK": "1"
            if pp.needs_netavark_switch
            else "0",
        }
        if pp.vllm_docker_tag:
            env["VLLM_DOCKER_TAG"] = pp.vllm_docker_tag
        return env

    def to_env(self) -> dict[str, str]:
        """Export SiteConfig as a flat dict of environment variables.

        This allows existing code that reads ``os.environ`` to work with
        values loaded from ``site.yaml`` without refactoring every call-site.
        """
        env: dict[str, str] = {}
        for layer in (
            _scheduler_env(self.scheduler),
            _runtime_env(self.runtime),
            _storage_env(self.storage),
            _network_env(self.network),
            _vllm_env(self.vllm),
            _database_env(self.database),
        ):
            env.update(layer)
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


def _scheduler_env(scheduler: SchedulerConfig) -> dict[str, str]:
    env: dict[str, str] = {}
    partition_candidates = _split_partitions(scheduler.partition)
    if partition_candidates:
        env["SLURM_PARTITIONS"] = ",".join(partition_candidates)
        env["SLURM_PARTITION"] = partition_candidates[0]
    elif scheduler.partition:
        env["SLURM_PARTITION"] = scheduler.partition
    env["SLURM_TIME"] = scheduler.time_limit
    env["SLURM_GPUS"] = str(scheduler.gpus)
    env["SLURM_CPUS"] = str(scheduler.cpus)
    env["SLURM_MEM"] = scheduler.memory
    env["SLURM_NODES"] = str(scheduler.nodes)
    return env


def _runtime_env(runtime: RuntimeConfig) -> dict[str, str]:
    env: dict[str, str] = {}
    if runtime.rivault_dir:
        env["RIVAULT_DIR"] = runtime.rivault_dir
    if runtime.repo_dir:
        env["UMRS_REPO"] = runtime.repo_dir
    if runtime.vision_image:
        env["BENTOML_VISION_IMAGE"] = runtime.vision_image
    if runtime.profiles:
        env["RIVAULT_PROFILES"] = " ".join(runtime.profiles)
    return env


def _storage_env(storage: StorageConfig) -> dict[str, str]:
    env: dict[str, str] = {}
    if storage.model_cache_dir:
        env["VLLM_PATH_TO_HUGGINGFACE"] = storage.model_cache_dir
    if storage.sam3_model_path:
        env["SAM3_MODEL_PATH"] = storage.sam3_model_path
    if storage.hmae_model_path:
        env["HMAE_MODEL_PATH"] = storage.hmae_model_path
    if storage.hmae_config_path:
        env["HMAE_CONFIG_PATH"] = storage.hmae_config_path
    if storage.hmae_checkpoint:
        env["HMAE_CHECKPOINT"] = storage.hmae_checkpoint
    if storage.vision_tarball:
        env["VISION_TARBALL"] = storage.vision_tarball
    env["USER_MODEL_PATH"] = storage.user_model_path
    if storage.s3_tunnel_host:
        env["UMRS_S3_TUNNEL_HOST"] = storage.s3_tunnel_host
    env["UMRS_S3_TUNNEL_PORT"] = str(storage.s3_tunnel_port)
    return env


def _network_env(network: NetworkConfig) -> dict[str, str]:
    env: dict[str, str] = {
        "NGINX_LITELLM_DOCKER_PORT": str(network.litellm_port),
        "RIVAULT_PORT_MIN": str(network.port_range_min),
        "RIVAULT_PORT_MAX": str(network.port_range_max),
        "RIVAULT_PORT_STRATEGY": network.port_strategy,
        "RIVAULT_REMOTE_PORT": str(network.litellm_port),
    }
    if network.litellm_master_key:
        env["LITELLM_MASTER_KEY"] = network.litellm_master_key
    if network.egress_proxy_url:
        env["RIVAULT_HTTP_PROXY"] = network.egress_proxy_url
        env["RIVAULT_HTTPS_PROXY"] = network.egress_proxy_url
        env["EGRESS_GUARD_PROXY_URL"] = network.egress_proxy_url
    if network.egress_no_proxy:
        env["RIVAULT_NO_PROXY"] = network.egress_no_proxy
    return env


def _vllm_env(vllm: VllmConfig) -> dict[str, str]:
    return {
        "VLLM_MODEL": vllm.model,
        "VLLM_DOCKER_TAG": vllm.docker_tag,
        "VLLM_GPU_MEM_UTIL": str(vllm.gpu_mem_util),
        "VLLM_MAX_MODEL_LEN": str(vllm.max_model_len),
        "VLLM_MAX_NUM_SEQS": str(vllm.max_num_seqs),
        "VLLM_TENSOR_PARALLEL_SIZE": str(vllm.tensor_parallel_size),
        "VLLM_GPU_MEMORY_UTILIZATION": str(vllm.gpu_mem_util),
        "VLLM_ENABLE_PREFIX_CACHING": "true" if vllm.enable_prefix_caching else "false",
    }


def _database_env(database: DatabaseConfig) -> dict[str, str]:
    env: dict[str, str] = {"DB_HOST": database.host, "DB_PORT": str(database.port)}
    if database.superuser:
        env["DB_SUPERUSER"] = database.superuser
        env["POSTGRES_USER"] = database.superuser
    if database.superuser_password:
        env["DB_SUPERUSER_PASSWORD"] = database.superuser_password
        env["POSTGRES_PASSWORD"] = database.superuser_password
    if database.user_password:
        env["DB_USER_PASSWORD"] = database.user_password
    if database.registry_dsn:
        env["UMRS_REGISTRY_BACKEND"] = "postgres"
        env["UMRS_REGISTRY_POSTGRES_DSN"] = database.registry_dsn
    return env


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _default_site_config_path() -> str:
    return get_resolved_umrs_subdir("site.yaml")


_SEARCH_PATHS = [
    lambda: os.environ.get("UMRS_SITE_CONFIG", ""),
    lambda: str(Path.cwd() / "site.yaml"),
    _default_site_config_path,
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


def _apply_env_fallbacks(raw: dict[str, Any]) -> dict[str, Any]:
    """Fill intentionally blank YAML values from environment variables."""
    merged = dict(raw)
    network = dict(merged.get("network") or {})
    if not network.get("litellm_master_key") and os.getenv("LITELLM_MASTER_KEY"):
        network["litellm_master_key"] = os.environ["LITELLM_MASTER_KEY"]
    if network:
        merged["network"] = network
    return merged


def _dict_to_dataclass(cls, data: dict):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in field_names})


def load_site_config(path: str | None = None) -> SiteConfig:
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

    resolved: str | None = None

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
            "the working directory or the resolved UMRS config directory."
        )

    with open(resolved) as fh:
        raw = _expand_env_refs(yaml.safe_load(fh) or {})

    logger.info("Loaded site config from %s", resolved)
    raw = _apply_env_fallbacks(raw)
    return _parse_raw(raw)


def _parse_raw(raw: dict[str, Any]) -> SiteConfig:
    """Turn a raw YAML dict into a :class:`SiteConfig`."""
    pp_raw = raw.get("partition_profiles", {})
    profiles: dict[str, PartitionProfile] = {}
    if isinstance(pp_raw, dict):
        for name, pdata in pp_raw.items():
            if isinstance(pdata, dict):
                profiles[name] = _dict_to_dataclass(PartitionProfile, pdata)

    return SiteConfig(
        name=raw.get("name", "default"),
        description=raw.get("description", ""),
        scheduler=_dict_to_dataclass(SchedulerConfig, raw.get("scheduler", {})),
        runtime=_dict_to_dataclass(RuntimeConfig, raw.get("runtime", {})),
        storage=_dict_to_dataclass(StorageConfig, raw.get("storage", {})),
        network=_dict_to_dataclass(NetworkConfig, raw.get("network", {})),
        vllm=_dict_to_dataclass(VllmConfig, raw.get("vllm", {})),
        database=_dict_to_dataclass(DatabaseConfig, raw.get("database", {})),
        partition_profiles=profiles,
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
    cfg.scheduler.partition = os.getenv(
        "SLURM_PARTITIONS", os.getenv("SLURM_PARTITION", "")
    )
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
    cfg.network.litellm_port = int(
        os.getenv("NGINX_LITELLM_DOCKER_PORT", os.getenv("RIVAULT_REMOTE_PORT", "8001"))
    )
    cfg.network.litellm_master_key = os.getenv("LITELLM_MASTER_KEY", "")
    cfg.network.port_range_min = int(os.getenv("RIVAULT_PORT_MIN", "11434"))
    cfg.network.port_range_max = int(os.getenv("RIVAULT_PORT_MAX", "11534"))
    cfg.network.port_strategy = os.getenv("RIVAULT_PORT_STRATEGY", "static_range")
    cfg.network.egress_proxy_url = os.getenv(
        "RIVAULT_HTTP_PROXY",
        os.getenv("EGRESS_GUARD_PROXY_URL", ""),
    )
    cfg.network.egress_no_proxy = os.getenv("RIVAULT_NO_PROXY", "")

    # vLLM
    cfg.vllm.model = os.getenv("VLLM_MODEL", "intfloat/e5-base-v2")
    cfg.vllm.docker_tag = os.getenv("VLLM_DOCKER_TAG", "v0.20.0")
    cfg.vllm.gpu_mem_util = float(os.getenv("VLLM_GPU_MEM_UTIL", "0.10"))
    cfg.vllm.enable_prefix_caching = _env_bool("VLLM_ENABLE_PREFIX_CACHING", False)

    # Database
    cfg.database.host = os.getenv("DB_HOST", "localhost")
    cfg.database.port = int(os.getenv("DB_PORT", "5432"))
    cfg.database.superuser = os.getenv(
        "DB_SUPERUSER", os.getenv("POSTGRES_USER", "postgres")
    )
    cfg.database.superuser_password = os.getenv(
        "DB_SUPERUSER_PASSWORD", os.getenv("POSTGRES_PASSWORD", "postgres")
    )
    cfg.database.registry_dsn = os.getenv("UMRS_REGISTRY_POSTGRES_DSN", "")

    return cfg


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_site_config: SiteConfig | None = None


def get_site_config(path: str | None = None) -> SiteConfig:
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
