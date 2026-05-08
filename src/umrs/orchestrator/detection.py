"""
Backend Auto-detection Module.

Detects the appropriate orchestrator backend based on environment:
- Slurm: SLURM_JOB_ID or sbatch available
- Kubernetes: KUBERNETES_SERVICE_HOST or kubeconfig
- Docker Compose: COMPOSE_PROJECT_NAME or docker compose available

Priority: Environment variable > Auto-detection
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Supported orchestrator backends."""

    SLURM = "slurm"
    KUBERNETES = "kubernetes"
    COMPOSE = "compose"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """Result of backend detection."""

    backend: BackendType
    confidence: str  # "explicit", "high", "medium", "low"
    reason: str
    available_backends: list[BackendType]

    def __str__(self) -> str:
        return f"{self.backend.value} ({self.confidence}): {self.reason}"


class BackendDetector:
    """
    Detects the appropriate orchestrator backend.

    Detection order:
    1. Explicit environment variable (RIVAULT_ORCHESTRATOR)
    2. Slurm detection (HPC environment)
    3. Kubernetes detection (K8s cluster)
    4. Docker Compose detection (development)
    """

    # Environment variable to explicitly set backend
    ENV_VAR = "RIVAULT_ORCHESTRATOR"

    def detect(self) -> DetectionResult:
        """
        Detect the appropriate backend.

        Returns:
            DetectionResult with backend type and confidence.
        """
        available = self._detect_available_backends()

        # 1. Check explicit environment variable
        explicit = os.getenv(self.ENV_VAR)
        if explicit:
            try:
                backend = BackendType(explicit.lower())
                return DetectionResult(
                    backend=backend,
                    confidence="explicit",
                    reason=f"{self.ENV_VAR}={explicit}",
                    available_backends=available,
                )
            except ValueError:
                logger.warning(
                    f"Invalid {self.ENV_VAR} value: {explicit}. "
                    f"Valid: {[b.value for b in BackendType if b != BackendType.UNKNOWN]}"
                )

        # 2. Auto-detect based on environment
        if self._is_slurm_environment():
            return DetectionResult(
                backend=BackendType.SLURM,
                confidence="high",
                reason="Slurm environment detected (sbatch available or SLURM vars)",
                available_backends=available,
            )

        if self._is_kubernetes_environment():
            return DetectionResult(
                backend=BackendType.KUBERNETES,
                confidence="high",
                reason="Kubernetes environment detected (in-cluster or kubeconfig)",
                available_backends=available,
            )

        if self._is_compose_environment():
            return DetectionResult(
                backend=BackendType.COMPOSE,
                confidence="medium",
                reason="Docker Compose available (development mode)",
                available_backends=available,
            )

        # 3. Fallback to first available
        if available:
            return DetectionResult(
                backend=available[0],
                confidence="low",
                reason=f"Fallback to first available: {available[0].value}",
                available_backends=available,
            )

        return DetectionResult(
            backend=BackendType.UNKNOWN,
            confidence="low",
            reason="No orchestrator backend detected",
            available_backends=[],
        )

    def _detect_available_backends(self) -> list[BackendType]:
        """Detect all available backends."""
        available = []

        if self._is_slurm_available():
            available.append(BackendType.SLURM)

        if self._is_kubernetes_available():
            available.append(BackendType.KUBERNETES)

        if self._is_compose_available():
            available.append(BackendType.COMPOSE)

        return available

    def _is_slurm_environment(self) -> bool:
        """Check if running in Slurm environment."""
        # Check Slurm environment variables
        if os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_CLUSTER_NAME"):
            return True

        # Check if sbatch is available and we're on a login/submit node
        return self._is_slurm_available()

    def _is_slurm_available(self) -> bool:
        """Check if Slurm commands are available."""
        return shutil.which("sbatch") is not None

    def _is_kubernetes_environment(self) -> bool:
        """Check if running in Kubernetes environment."""
        # In-cluster detection (service account token)
        if os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token"):
            return True

        # Kubernetes service host (set in pods)
        return bool(os.getenv("KUBERNETES_SERVICE_HOST"))

    def _is_kubernetes_available(self) -> bool:
        """Check if Kubernetes is available (in-cluster or kubeconfig)."""
        # In-cluster
        if self._is_kubernetes_environment():
            return True

        # Check kubeconfig
        kubeconfig = os.getenv("KUBECONFIG") or str(
            Path.home() / ".kube" / "config"
        )
        if os.path.exists(kubeconfig):
            return True

        # Check kubectl
        return shutil.which("kubectl") is not None

    def _is_compose_environment(self) -> bool:
        """Check if running in Docker Compose environment."""
        # Check Compose-specific env vars
        return bool(os.getenv("COMPOSE_PROJECT_NAME"))

    def _is_compose_available(self) -> bool:
        """Check if Docker Compose is available."""
        # Check docker compose v2
        if shutil.which("docker"):
            try:
                result = subprocess.run(
                    ["docker", "compose", "version"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                pass

        # Check docker-compose v1
        return shutil.which("docker-compose") is not None


def detect_backend() -> DetectionResult:
    """
    Detect the appropriate orchestrator backend.

    This is the main entry point for backend detection.

    Returns:
        DetectionResult with backend type and confidence.
    """
    detector = BackendDetector()
    result = detector.detect()
    logger.info(f"Backend detection: {result}")
    return result


def get_backend_type() -> BackendType:
    """
    Get the detected backend type.

    Returns:
        BackendType enum value.
    """
    return detect_backend().backend
