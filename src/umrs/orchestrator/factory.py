"""
Orchestrator factory for selecting and instantiating backends.

The backend is selected via the RIVAULT_ORCHESTRATOR environment variable:
- "slurm" (default on HPC): Uses Slurm + Singularity
- "kubernetes" or "k8s": Uses Kubernetes Jobs
- "docker-compose" or "compose": Uses Docker Compose (development)
- "auto": Auto-detect based on available tools
"""

from __future__ import annotations

import os

from .base import OrchestratorBackend, OrchestratorConfig

# Registry of available backends
_BACKEND_REGISTRY: dict[str, type[OrchestratorBackend]] = {}
_EXPERIMENTAL_HYBRID_BACKENDS = {"hybrid", "hybrid-slurm", "slurm-hybrid"}


def register_backend(name: str, backend_class: type[OrchestratorBackend]) -> None:
    """Register a backend implementation."""
    _BACKEND_REGISTRY[name.lower()] = backend_class


def _lazy_register_builtins() -> None:
    """Lazily register built-in backends to avoid import issues."""
    if "slurm" not in _BACKEND_REGISTRY:
        from .slurm_backend import SlurmBackend
        register_backend("slurm", SlurmBackend)

    if "kubernetes" not in _BACKEND_REGISTRY:
        from .kubernetes_backend import KubernetesBackend
        register_backend("kubernetes", KubernetesBackend)
        register_backend("k8s", KubernetesBackend)

    if "docker-compose" not in _BACKEND_REGISTRY:
        from .compose_backend import DockerComposeBackend
        register_backend("docker-compose", DockerComposeBackend)
        register_backend("compose", DockerComposeBackend)

    if "slurm-podman" not in _BACKEND_REGISTRY:
        from .slurm_podman_backend import SlurmPodmanBackend
        register_backend("slurm-podman", SlurmPodmanBackend)


def _detect_backend() -> str:
    """Auto-detect the best available backend."""
    import shutil

    # Check for Slurm
    if shutil.which("sbatch") and shutil.which("squeue"):
        return "slurm"

    if shutil.which("kubectl") and _command_succeeds("kubectl", "cluster-info"):
        return "kubernetes"

    if shutil.which("docker") and _command_succeeds("docker", "info"):
        return "docker-compose"

    raise RuntimeError(
        "No orchestration backend available. "
        "Install Slurm, Kubernetes (kubectl), or Docker."
    )


def _command_succeeds(*args: str) -> bool:
    import subprocess

    try:
        result = subprocess.run(args, capture_output=True, timeout=5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return result.returncode == 0


def get_orchestrator(
    backend: str | None = None,
    config: OrchestratorConfig | None = None,
) -> OrchestratorBackend:
    """
    Get an orchestrator backend instance.

    Args:
        backend: Backend name. If None, uses RIVAULT_ORCHESTRATOR env var
                 or auto-detects.
        config: Optional configuration. If None, loads from environment.

    Returns:
        An OrchestratorBackend instance.

    Examples:
        # Auto-detect backend
        orchestrator = get_orchestrator()

        # Explicit backend
        orchestrator = get_orchestrator("slurm")

        # With custom config
        config = OrchestratorConfig(model_path="/models/llama")
        orchestrator = get_orchestrator("kubernetes", config)
    """
    _lazy_register_builtins()

    if backend is None:
        backend = os.environ.get("RIVAULT_ORCHESTRATOR", "auto")

    backend = backend.lower()

    if backend == "auto":
        backend = _detect_backend()

    if backend in _EXPERIMENTAL_HYBRID_BACKENDS:
        raise ValueError(
            "Hybrid Slurm orchestration is experimental, not for HPC production. "
            "Use 'slurm-podman' or 'slurm', or instantiate HybridOrchestrator "
            "with RIVAULT_ENABLE_EXPERIMENTAL_HYBRID_SLURM=true for compatibility testing."
        )

    if backend not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown orchestrator backend: {backend}. "
            f"Available: {available}"
        )

    backend_class = _BACKEND_REGISTRY[backend]
    return backend_class(config)


async def get_orchestrator_health() -> dict[str, any]:
    """
    Get health status of all registered backends.

    Returns:
        Dict mapping backend names to their health status.
    """
    _lazy_register_builtins()

    results = {}
    for name, backend_class in _BACKEND_REGISTRY.items():
        # Skip aliases
        if name in ("k8s", "compose"):
            continue

        try:
            backend = backend_class()
            health = await backend.health_check()
            results[name] = health
        except Exception as e:
            results[name] = {
                "backend": name,
                "healthy": False,
                "message": str(e),
            }

    return results
