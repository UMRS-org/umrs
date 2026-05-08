"""
Kubernetes-on-Slurm Scheduler (U-1131-1134).

Enables Kubernetes workloads to run on HPC clusters via Slurm translation.
This scheduler intercepts Kubernetes API calls and translates them to Slurm jobs.
"""

from __future__ import annotations

import builtins
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class K8sResourceKind(Enum):
    """Supported Kubernetes resource kinds for translation."""

    POD = "Pod"
    JOB = "Job"
    DEPLOYMENT = "Deployment"
    SERVICE = "Service"


@dataclass
class K8sResourceSpec:
    """Kubernetes resource specification."""

    api_version: str
    kind: str
    metadata: dict[str, Any]
    spec: dict[str, Any]


@dataclass
class SlurmJobSpec:
    """Slurm job specification (translated from K8s)."""

    job_name: str
    partition: str
    time: str
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 1
    mem: str = "4G"
    gpus: int = 0
    gpu_type: str | None = None
    account: str | None = None  # #SBATCH --account=<acct>
    container_image: str = ""
    container_runtime: str = "singularity"
    command: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    mounts: list[str] = field(default_factory=list)
    working_dir: str = "/workspace"
    output_file: str = "slurm-%j.out"
    error_file: str = "slurm-%j.err"


@dataclass
class TranslatedJob:
    """Result of K8s to Slurm translation."""

    original_kind: str
    slurm_script: str
    slurm_spec: SlurmJobSpec
    metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class K8sToSlurmTranslator:
    """
    Translates Kubernetes manifests to Slurm job scripts.

    Supports:
    - Pod -> Single Slurm job
    - Job -> Slurm job with completion tracking
    - Deployment -> Slurm job array (approximation)
    """

    def __init__(
        self,
        default_partition: str = "gpu",
        default_time: str = "02:00:00",
        container_runtime: str = "singularity",
        model_mount_path: str = "/models",
        gpu_resource_name: str = "nvidia.com/gpu",
        default_account: str | None = None,
    ):
        """Initialize translator with defaults."""
        self.default_partition = default_partition
        self.default_time = default_time
        self.container_runtime = container_runtime
        self.model_mount_path = model_mount_path
        self.gpu_resource_name = gpu_resource_name
        # Fallback Slurm account used when the K8s manifest does not carry
        # one via the ``rivault.platform/slurm-account`` annotation.
        self.default_account = default_account

    def translate(self, resource: K8sResourceSpec) -> TranslatedJob:
        """
        Translate K8s resource to Slurm job.

        Args:
            resource: Kubernetes resource specification

        Returns:
            Translated Slurm job
        """
        kind = resource.kind

        if kind == K8sResourceKind.POD.value:
            return self._translate_pod(resource)
        elif kind == K8sResourceKind.JOB.value:
            return self._translate_job(resource)
        elif kind == K8sResourceKind.DEPLOYMENT.value:
            return self._translate_deployment(resource)
        else:
            raise ValueError(f"Unsupported resource kind: {kind}")

    def _translate_pod(self, resource: K8sResourceSpec) -> TranslatedJob:
        """Translate Pod to Slurm job."""
        metadata = resource.metadata
        spec = resource.spec
        warnings = []

        # Extract job name
        job_name = metadata.get("name", "k8s-pod")
        job_name = self._sanitize_job_name(job_name)

        # Get container spec (first container)
        containers = spec.get("containers", [])
        if not containers:
            raise ValueError("Pod has no containers")

        container = containers[0]
        if len(containers) > 1:
            warnings.append(
                f"Pod has {len(containers)} containers, only first will be used"
            )

        # Build Slurm spec
        slurm_spec = self._build_slurm_spec(job_name, container, spec)

        # Generate script
        slurm_script = self._generate_script(slurm_spec)

        return TranslatedJob(
            original_kind="Pod",
            slurm_script=slurm_script,
            slurm_spec=slurm_spec,
            metadata={
                "original_name": metadata.get("name"),
                "namespace": metadata.get("namespace", "default"),
                "labels": metadata.get("labels", {}),
            },
            warnings=warnings,
        )

    def _translate_job(self, resource: K8sResourceSpec) -> TranslatedJob:
        """Translate K8s Job to Slurm job."""
        metadata = resource.metadata
        spec = resource.spec
        warnings = []

        job_name = metadata.get("name", "k8s-job")
        job_name = self._sanitize_job_name(job_name)

        # Get template spec
        template = spec.get("template", {})
        pod_spec = template.get("spec", {})

        containers = pod_spec.get("containers", [])
        if not containers:
            raise ValueError("Job template has no containers")

        container = containers[0]

        # Check completions/parallelism
        completions = spec.get("completions", 1)
        parallelism = spec.get("parallelism", 1)

        if completions > 1 or parallelism > 1:
            warnings.append(
                f"Job parallelism ({parallelism}) and completions ({completions}) "
                "will be approximated with Slurm job array"
            )

        slurm_spec = self._build_slurm_spec(job_name, container, pod_spec)

        # Add array for parallelism
        if completions > 1:
            slurm_spec.ntasks = completions

        slurm_script = self._generate_script(slurm_spec)

        return TranslatedJob(
            original_kind="Job",
            slurm_script=slurm_script,
            slurm_spec=slurm_spec,
            metadata={
                "original_name": metadata.get("name"),
                "completions": completions,
                "parallelism": parallelism,
            },
            warnings=warnings,
        )

    def _translate_deployment(self, resource: K8sResourceSpec) -> TranslatedJob:
        """Translate Deployment to Slurm (with warnings)."""
        metadata = resource.metadata
        spec = resource.spec
        warnings = [
            "Deployment translated to single Slurm job. "
            "Replicas and rolling updates are not supported in Slurm."
        ]

        job_name = metadata.get("name", "k8s-deployment")
        job_name = self._sanitize_job_name(job_name)

        # Get template
        template = spec.get("template", {})
        pod_spec = template.get("spec", {})

        containers = pod_spec.get("containers", [])
        if not containers:
            raise ValueError("Deployment template has no containers")

        container = containers[0]
        replicas = spec.get("replicas", 1)

        if replicas > 1:
            warnings.append(f"Replicas ({replicas}) will be ignored")

        slurm_spec = self._build_slurm_spec(job_name, container, pod_spec)
        slurm_script = self._generate_script(slurm_spec)

        return TranslatedJob(
            original_kind="Deployment",
            slurm_script=slurm_script,
            slurm_spec=slurm_spec,
            metadata={
                "original_name": metadata.get("name"),
                "replicas": replicas,
            },
            warnings=warnings,
        )

    def _build_slurm_spec(
        self,
        job_name: str,
        container: dict[str, Any],
        pod_spec: dict[str, Any],
    ) -> SlurmJobSpec:
        """Build Slurm spec from container and pod specs."""
        # Extract resources
        resources = container.get("resources", {})
        requests = resources.get("requests", {})
        limits = resources.get("limits", {})

        # CPU
        cpu_request = requests.get("cpu", "1")
        cpus = self._parse_cpu(cpu_request)

        # Memory
        mem_request = requests.get("memory", "4Gi")
        mem = self._parse_memory(mem_request)

        # GPU
        gpu_limit = limits.get(self.gpu_resource_name, "0")
        gpus = int(gpu_limit)

        # Environment
        env = {}
        for env_var in container.get("env", []):
            name = env_var.get("name")
            value = env_var.get("value", "")
            if name:
                env[name] = value

        # Mounts
        mounts = []
        volume_mounts = container.get("volumeMounts", [])
        volumes = {v["name"]: v for v in pod_spec.get("volumes", [])}

        for vm in volume_mounts:
            vol_name = vm.get("name")
            mount_path = vm.get("mountPath")
            if vol_name in volumes:
                vol = volumes[vol_name]
                # Handle different volume types
                if "hostPath" in vol:
                    host_path = vol["hostPath"]["path"]
                    mounts.append(f"{host_path}:{mount_path}")
                elif "persistentVolumeClaim" in vol:
                    # PVC -> assume NFS mount at same path
                    mounts.append(f"{mount_path}:{mount_path}")

        # Command
        command = container.get("command", [])
        args = container.get("args", [])
        full_command = command + args

        return SlurmJobSpec(
            job_name=job_name,
            partition=self.default_partition,
            time=self.default_time,
            cpus_per_task=cpus,
            mem=mem,
            gpus=gpus,
            account=self.default_account,
            container_image=container.get("image", ""),
            container_runtime=self.container_runtime,
            command=full_command,
            environment=env,
            mounts=mounts,
            working_dir=container.get("workingDir", "/workspace"),
        )

    def _generate_script(self, spec: SlurmJobSpec) -> str:
        """Generate Slurm batch script."""
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={spec.job_name}",
            f"#SBATCH --partition={spec.partition}",
            f"#SBATCH --time={spec.time}",
            f"#SBATCH --nodes={spec.nodes}",
            f"#SBATCH --ntasks={spec.ntasks}",
            f"#SBATCH --cpus-per-task={spec.cpus_per_task}",
            f"#SBATCH --mem={spec.mem}",
            f"#SBATCH --output={spec.output_file}",
            f"#SBATCH --error={spec.error_file}",
        ]

        if spec.gpus > 0:
            gpu_spec = f"{spec.gpus}"
            if spec.gpu_type:
                gpu_spec = f"{spec.gpu_type}:{spec.gpus}"
            lines.append(f"#SBATCH --gres=gpu:{gpu_spec}")

        if spec.account:
            lines.append(f"#SBATCH --account={spec.account}")

        lines.append("")
        lines.append("# Environment setup")
        lines.append("set -euo pipefail")
        lines.append("")

        # Export environment variables
        if spec.environment:
            lines.append("# Environment variables from K8s spec")
            for key, value in spec.environment.items():
                lines.append(f'export {key}="{value}"')
            lines.append("")

        # Container command
        if spec.container_runtime == "singularity":
            lines.append("# Run container with Singularity")
            bind_args = " ".join(f"--bind {m}" for m in spec.mounts)
            cmd = " ".join(spec.command) if spec.command else ""
            lines.append(
                f"singularity exec --nv {bind_args} "
                f"{spec.container_image} {cmd}"
            )
        elif spec.container_runtime == "docker":
            lines.append("# Run container with Docker")
            mount_args = " ".join(f"-v {m}" for m in spec.mounts)
            env_args = " ".join(f'-e {k}="{v}"' for k, v in spec.environment.items())
            cmd = " ".join(spec.command) if spec.command else ""
            gpu_arg = "--gpus all" if spec.gpus > 0 else ""
            lines.append(
                f"docker run --rm {gpu_arg} {mount_args} {env_args} "
                f"{spec.container_image} {cmd}"
            )

        lines.append("")
        lines.append("echo 'Job completed successfully'")

        return "\n".join(lines)

    def _sanitize_job_name(self, name: str) -> str:
        """Sanitize job name for Slurm."""
        # Slurm job names: alphanumeric, underscore, dash
        import re
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
        return sanitized[:128]  # Slurm limit

    def _parse_cpu(self, cpu_str: str) -> int:
        """Parse K8s CPU string (e.g., '500m', '2')."""
        if cpu_str.endswith("m"):
            # millicores to cores
            return max(1, int(cpu_str[:-1]) // 1000)
        return int(float(cpu_str))

    def _parse_memory(self, mem_str: str) -> str:
        """Parse K8s memory string to Slurm format."""
        # K8s: Ki, Mi, Gi, Ti, Pi, Ei (power of 2)
        # K8s: K, M, G, T, P, E (power of 10)
        # Slurm: K, M, G, T
        suffixes = {
            "Ki": "K",
            "Mi": "M",
            "Gi": "G",
            "Ti": "T",
            "K": "K",
            "M": "M",
            "G": "G",
            "T": "T",
        }

        for k8s_suffix, slurm_suffix in suffixes.items():
            if mem_str.endswith(k8s_suffix):
                value = mem_str[:-len(k8s_suffix)]
                return f"{value}{slurm_suffix}"

        # Plain bytes, convert to MB
        try:
            bytes_val = int(mem_str)
            mb = bytes_val // (1024 * 1024)
            return f"{mb}M"
        except ValueError:
            return "4G"  # Default


class K8sOnSlurmScheduler:
    """
    Scheduler that accepts Kubernetes API requests and translates to Slurm.

    This enables Kubernetes-native workflows to run on HPC clusters.
    """

    def __init__(
        self,
        translator: K8sToSlurmTranslator | None = None,
        slurm_submit_cmd: str = "sbatch",
        slurm_status_cmd: str = "squeue",
        slurm_cancel_cmd: str = "scancel",
        script_dir: str = "/tmp/k8s-slurm-scripts",
    ):
        """Initialize scheduler."""
        self.translator = translator or K8sToSlurmTranslator()
        self.slurm_submit_cmd = slurm_submit_cmd
        self.slurm_status_cmd = slurm_status_cmd
        self.slurm_cancel_cmd = slurm_cancel_cmd
        self.script_dir = script_dir

        # Job tracking
        self._jobs: dict[str, dict[str, Any]] = {}

    async def create(
        self,
        resource: K8sResourceSpec,
    ) -> dict[str, Any]:
        """
        Create resource (submit to Slurm).

        Args:
            resource: Kubernetes resource specification

        Returns:
            Created resource status
        """
        import os
        import subprocess

        # Translate to Slurm
        translated = self.translator.translate(resource)

        # Log warnings
        for warning in translated.warnings:
            logger.warning("K8s->Slurm: %s", warning)

        # Write script
        os.makedirs(self.script_dir, exist_ok=True)
        script_path = os.path.join(
            self.script_dir,
            f"{translated.slurm_spec.job_name}.sh",
        )

        with open(script_path, "w") as f:
            f.write(translated.slurm_script)
        os.chmod(script_path, 0o755)

        # Submit to Slurm
        result = subprocess.run(
            [self.slurm_submit_cmd, script_path],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Slurm submit failed: {result.stderr}")

        # Parse job ID from "Submitted batch job XXXX"
        job_id = result.stdout.strip().split()[-1]

        # Track job
        k8s_name = resource.metadata.get("name", job_id)
        self._jobs[k8s_name] = {
            "slurm_job_id": job_id,
            "kind": resource.kind,
            "status": "Pending",
            "created_at": datetime.now().isoformat(),
            "script_path": script_path,
        }

        return {
            "apiVersion": resource.api_version,
            "kind": resource.kind,
            "metadata": {
                "name": k8s_name,
                "uid": job_id,
                "annotations": {
                    "k8s-on-slurm/job-id": job_id,
                },
            },
            "status": {
                "phase": "Pending",
                "conditions": [
                    {
                        "type": "Scheduled",
                        "status": "True",
                        "message": f"Submitted to Slurm as job {job_id}",
                    }
                ],
            },
        }

    async def get(self, name: str) -> dict[str, Any] | None:
        """Get resource status."""
        if name not in self._jobs:
            return None

        job_info = self._jobs[name]
        slurm_status = await self._get_slurm_status(job_info["slurm_job_id"])

        # Map Slurm status to K8s phase
        phase_map = {
            "PENDING": "Pending",
            "RUNNING": "Running",
            "COMPLETED": "Succeeded",
            "FAILED": "Failed",
            "CANCELLED": "Failed",
            "TIMEOUT": "Failed",
        }

        phase = phase_map.get(slurm_status, "Unknown")

        return {
            "metadata": {
                "name": name,
                "uid": job_info["slurm_job_id"],
            },
            "status": {
                "phase": phase,
                "slurmState": slurm_status,
            },
        }

    async def delete(self, name: str) -> bool:
        """Delete resource (cancel Slurm job)."""
        import subprocess

        if name not in self._jobs:
            return False

        job_info = self._jobs[name]

        result = subprocess.run(
            [self.slurm_cancel_cmd, job_info["slurm_job_id"]],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            del self._jobs[name]
            return True

        return False

    async def list(
        self,
        kind: str | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """List resources."""
        items = []
        for name, info in self._jobs.items():
            if kind and info["kind"] != kind:
                continue

            status = await self.get(name)
            if status:
                items.append(status)

        return items

    async def _get_slurm_status(self, job_id: str) -> str:
        """Get Slurm job status."""
        import subprocess

        result = subprocess.run(
            [
                self.slurm_status_cmd,
                "-j", job_id,
                "--noheader",
                "-o", "%T",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Job might have completed
            return "COMPLETED"

        status = result.stdout.strip()
        return status if status else "UNKNOWN"
