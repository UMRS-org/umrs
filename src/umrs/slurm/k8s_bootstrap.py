"""
K8s-on-Slurm Bootstrap Wrapper (U-1133, U-1134).

Provides UMRS integration for running k3s inside Slurm jobs.
This is experimental and intended for:
- Testing Helm charts on HPC
- Development of K8s workloads
- Running lightweight K8s workloads without cluster-wide K8s

NOT recommended for production workloads.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class K8sOnSlurmStatus(Enum):
    """Status of K8s-on-Slurm job."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class K8sOnSlurmConfig:
    """
    Configuration for K8s-on-Slurm job.

    Specifies Slurm resources and k3s settings.
    """

    # Slurm settings
    partition: str = "compute"
    time_limit: str = "02:00:00"
    nodes: int = 1
    cpus_per_task: int = 8
    memory_gb: int = 32
    gpus: int = 0
    gpu_type: str | None = None

    # k3s settings
    k3s_version: str = "v1.29.0+k3s1"
    k3s_port: int = 6443
    disable_services: list[str] = field(
        default_factory=lambda: ["traefik", "servicelb", "metrics-server"]
    )

    # Workload
    manifest_path: str | None = None  # YAML manifest to apply
    helm_chart: str | None = None  # Helm chart to install
    helm_values: str | None = None  # Values file for Helm

    def to_env(self) -> dict[str, str]:
        """Convert to environment variables."""
        env = {
            "K3S_VERSION": self.k3s_version,
            "K3S_PORT": str(self.k3s_port),
            "K3S_DISABLE_SERVICES": ",".join(self.disable_services),
        }

        if self.manifest_path:
            env["K3S_WORKLOAD"] = self.manifest_path
        elif self.helm_chart:
            env["K3S_WORKLOAD"] = f"helm:{self.helm_chart}"

        return env


@dataclass
class K8sOnSlurmJob:
    """
    K8s-on-Slurm job specification.

    Tracks a k3s cluster running inside a Slurm job.
    """

    job_id: str
    slurm_job_id: str | None = None
    config: K8sOnSlurmConfig = field(default_factory=K8sOnSlurmConfig)

    # Status
    status: K8sOnSlurmStatus = K8sOnSlurmStatus.PENDING
    node_name: str | None = None
    kubeconfig_path: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "slurm_job_id": self.slurm_job_id,
            "status": self.status.value,
            "node_name": self.node_name,
            "kubeconfig_path": self.kubeconfig_path,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


class K8sBootstrapWrapper:
    """
    Wrapper for launching and managing k3s inside Slurm jobs.

    This is the main interface for UMRS to interact with K8s-on-Slurm.
    """

    # Path to Slurm scripts
    SLURM_SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "rivault" / "slurm"

    def __init__(self, slurm_scripts_dir: Path | None = None):
        """Initialize wrapper."""
        self.scripts_dir = slurm_scripts_dir or self.SLURM_SCRIPTS_DIR
        self._jobs: dict[str, K8sOnSlurmJob] = {}

    def submit(self, config: K8sOnSlurmConfig) -> K8sOnSlurmJob:
        """
        Submit a K8s-on-Slurm job.

        Args:
            config: Job configuration

        Returns:
            K8sOnSlurmJob instance
        """
        import uuid

        job_id = f"k8s-{uuid.uuid4().hex[:8]}"
        job = K8sOnSlurmJob(job_id=job_id, config=config)

        # Generate sbatch script
        sbatch_script = self._generate_sbatch(config)

        # Submit to Slurm
        try:
            result = subprocess.run(
                ["sbatch", "--parsable"],
                input=sbatch_script,
                capture_output=True,
                text=True,
                check=True,
            )
            slurm_job_id = result.stdout.strip()
            job.slurm_job_id = slurm_job_id
            job.status = K8sOnSlurmStatus.STARTING
            logger.info(
                f"Submitted K8s-on-Slurm job {job_id} as Slurm job {slurm_job_id}"
            )

        except subprocess.CalledProcessError as e:
            job.status = K8sOnSlurmStatus.FAILED
            logger.error(f"Failed to submit job: {e.stderr}")
            raise RuntimeError(f"Failed to submit Slurm job: {e.stderr}") from e

        self._jobs[job_id] = job
        return job

    def get_status(self, job_id: str) -> K8sOnSlurmJob:
        """
        Get job status.

        Args:
            job_id: The job ID

        Returns:
            Updated K8sOnSlurmJob
        """
        if job_id not in self._jobs:
            raise ValueError(f"Unknown job: {job_id}")

        job = self._jobs[job_id]

        if job.slurm_job_id:
            # Query Slurm for status
            try:
                result = subprocess.run(
                    ["squeue", "-j", job.slurm_job_id, "-h", "-o", "%T %N"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split()
                    slurm_state = parts[0]
                    node_name = parts[1] if len(parts) > 1 else None

                    if slurm_state == "RUNNING":
                        job.status = K8sOnSlurmStatus.RUNNING
                        job.node_name = node_name
                        job.started_at = job.started_at or datetime.now()
                        job.kubeconfig_path = (
                            f"/tmp/k3s-{job.slurm_job_id}/kubeconfig.yaml"
                        )
                    elif slurm_state == "PENDING":
                        job.status = K8sOnSlurmStatus.PENDING
                    elif slurm_state in ["CANCELLED", "TIMEOUT", "PREEMPTED"]:
                        job.status = K8sOnSlurmStatus.CANCELLED
                        job.completed_at = datetime.now()
                    elif slurm_state in ["FAILED", "NODE_FAIL"]:
                        job.status = K8sOnSlurmStatus.FAILED
                        job.completed_at = datetime.now()

                else:
                    # Job not in queue - check if completed
                    sacct_result = subprocess.run(
                        ["sacct", "-j", job.slurm_job_id, "-n", "-o", "State", "-X"],
                        capture_output=True,
                        text=True,
                    )
                    if "COMPLETED" in sacct_result.stdout:
                        job.status = K8sOnSlurmStatus.COMPLETED
                        job.completed_at = datetime.now()
                    elif "FAILED" in sacct_result.stdout:
                        job.status = K8sOnSlurmStatus.FAILED
                        job.completed_at = datetime.now()

            except subprocess.CalledProcessError:
                pass  # Keep current status

        return job

    def cancel(self, job_id: str) -> K8sOnSlurmJob:
        """
        Cancel a K8s-on-Slurm job.

        Args:
            job_id: The job ID

        Returns:
            Updated K8sOnSlurmJob
        """
        if job_id not in self._jobs:
            raise ValueError(f"Unknown job: {job_id}")

        job = self._jobs[job_id]

        if job.slurm_job_id:
            try:
                subprocess.run(
                    ["scancel", job.slurm_job_id],
                    check=True,
                )
                job.status = K8sOnSlurmStatus.CANCELLED
                job.completed_at = datetime.now()
                logger.info(f"Cancelled job {job_id} (Slurm {job.slurm_job_id})")

            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to cancel job: {e}")

        return job

    def get_kubeconfig(self, job_id: str) -> str | None:
        """
        Get kubeconfig content for a running job.

        Args:
            job_id: The job ID

        Returns:
            Kubeconfig YAML content or None
        """
        job = self.get_status(job_id)

        if job.status != K8sOnSlurmStatus.RUNNING or not job.kubeconfig_path:
            return None

        # Try to read kubeconfig from compute node
        # This may require SSH or shared filesystem access
        if os.path.exists(job.kubeconfig_path):
            with open(job.kubeconfig_path) as f:
                return f.read()

        return None

    def _generate_sbatch(self, config: K8sOnSlurmConfig) -> str:
        """Generate sbatch script content."""
        lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=k8s-on-slurm",
            f"#SBATCH --time={config.time_limit}",
            f"#SBATCH --nodes={config.nodes}",
            f"#SBATCH --cpus-per-task={config.cpus_per_task}",
            f"#SBATCH --mem={config.memory_gb}G",
            f"#SBATCH --partition={config.partition}",
        ]

        if config.gpus > 0:
            gpu_spec = (
                f"{config.gpu_type}:{config.gpus}"
                if config.gpu_type
                else str(config.gpus)
            )
            lines.append(f"#SBATCH --gres=gpu:{gpu_spec}")

        lines.append("")
        lines.append("set -euo pipefail")
        lines.append("")

        # Export configuration
        for key, value in config.to_env().items():
            lines.append(f'export {key}="{value}"')

        lines.append("")
        lines.append(f'bash "{self.scripts_dir / "k8s-on-slurm.sh"}"')

        return "\n".join(lines)
