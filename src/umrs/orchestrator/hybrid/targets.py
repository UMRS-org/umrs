"""
ExecutionTarget Interface and Implementations (U-1101 to U-1104).

ExecutionTarget handles resource allocation (submit/cancel/status) while
delegating container launch to RuntimeWrapper.

Implementations:
- SlurmTarget: Slurm job scheduling
- K8sTarget: Kubernetes Job scheduling
- ComposeTarget: Docker Compose services
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AllocationStatus(Enum):
    """Status of a resource allocation."""

    PENDING = "pending"      # Waiting for resources
    ALLOCATED = "allocated"  # Resources granted, ready for container
    RUNNING = "running"      # Container launched
    COMPLETED = "completed"  # Job finished successfully
    FAILED = "failed"        # Allocation failed
    CANCELLED = "cancelled"  # User cancelled
    UNKNOWN = "unknown"      # Cannot determine status


@dataclass
class AllocationInfo:
    """
    Information about an allocated execution slot.

    This is returned by ExecutionTarget.submit() and passed to
    RuntimeWrapper.launch() to start the container.
    """

    job_id: str                     # Backend-specific job ID
    node: str                       # Allocated node hostname
    status: AllocationStatus
    allocated_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Resource details (filled by backend)
    gpus: int = 0
    cpus: int = 0
    memory_mb: int = 0

    @property
    def is_allocated(self) -> bool:
        """Check if resources are allocated and ready for container launch."""
        return self.status in (AllocationStatus.ALLOCATED, AllocationStatus.RUNNING)


@dataclass
class JobSpec:
    """
    Specification for a job submission.

    Contains all parameters needed for resource allocation.
    RuntimeWrapper receives a separate LaunchSpec for container details.
    """

    # Required
    name: str
    execution_authority: str  # "slurm" | "k8s" | "compose"

    # Resources
    gpus: int = 1
    cpus: int = 8
    memory_mb: int = 65536  # 64GB

    # Scheduling
    partition: str = ""
    time_limit: str = "02:00:00"
    priority: int = 0

    # Per-submission Slurm account. When set, overrides the SlurmTarget
    # instance default and SLURM_ACCOUNT env. None falls back to instance
    # default / site.yaml. Populated from the submitting user's record so
    # fairshare attributes the job to their own Slurm account.
    slurm_account: str | None = None

    # Attribution: identity of the requesting user for audit/accounting.
    # These fields are injected into the sbatch script as #SBATCH --comment
    # and exported as environment variables inside the job.
    requesting_user: str = ""
    requesting_project: str = ""
    requesting_session: str = ""

    # Labels/annotations
    labels: dict[str, str] = field(default_factory=dict)

    # Backend-specific options
    extra: dict[str, Any] = field(default_factory=dict)


class ExecutionTarget(ABC):
    """
    Abstract base for resource allocation backends (U-1101).

    Responsibilities:
    - Submit jobs to scheduler
    - Cancel running jobs
    - Query job status
    - List active allocations

    Does NOT:
    - Launch containers (RuntimeWrapper's job)
    - Manage model references (ModelRefResolver's job)
    """

    @property
    @abstractmethod
    def authority(self) -> str:
        """Return execution authority name ('slurm', 'k8s', 'compose')."""
        ...

    @abstractmethod
    async def submit(self, spec: JobSpec) -> AllocationInfo:
        """
        Submit a job to the scheduler.

        Args:
            spec: Job specification with resource requirements.

        Returns:
            AllocationInfo with job_id and initial status.

        Raises:
            SubmissionError: If job cannot be submitted.
        """
        ...

    @abstractmethod
    async def cancel(self, job_id: str) -> bool:
        """
        Cancel a running or pending job.

        Args:
            job_id: The job identifier from submit().

        Returns:
            True if cancellation was successful.
        """
        ...

    @abstractmethod
    async def status(self, job_id: str) -> AllocationInfo:
        """
        Get current status of a job.

        Args:
            job_id: The job identifier from submit().

        Returns:
            AllocationInfo with current status and details.
        """
        ...

    @abstractmethod
    async def list_jobs(
        self,
        labels: dict[str, str] | None = None,
    ) -> list[AllocationInfo]:
        """
        List active jobs matching optional label filter.

        Args:
            labels: Optional filter by labels.

        Returns:
            List of AllocationInfo for matching jobs.
        """
        ...

    async def wait_for_allocation(
        self,
        job_id: str,
        timeout_sec: int = 300,
        poll_interval_sec: int = 5,
    ) -> AllocationInfo:
        """
        Wait until job transitions from PENDING to ALLOCATED.

        Args:
            job_id: The job identifier.
            timeout_sec: Maximum wait time.
            poll_interval_sec: Seconds between status checks.

        Returns:
            AllocationInfo when allocated.

        Raises:
            TimeoutError: If allocation takes too long.
            RuntimeError: If job fails during allocation.
        """
        start = datetime.now()
        while True:
            info = await self.status(job_id)

            if info.status == AllocationStatus.ALLOCATED or info.status == AllocationStatus.RUNNING:
                return info
            elif info.status in (AllocationStatus.FAILED, AllocationStatus.CANCELLED):
                raise RuntimeError(f"Job {job_id} failed during allocation: {info.status}")

            elapsed = (datetime.now() - start).total_seconds()
            if elapsed > timeout_sec:
                raise TimeoutError(f"Job {job_id} not allocated after {timeout_sec}s")

            await asyncio.sleep(poll_interval_sec)


class SubmissionError(Exception):
    """Error during job submission."""
    pass


# =============================================================================
# SlurmTarget Implementation (U-1102)
# =============================================================================


class SlurmTarget(ExecutionTarget):
    """
    Slurm-based execution target (U-1102).

    Submits jobs via sbatch and monitors via squeue/sacct.
    This implementation focuses purely on resource allocation —
    container launch is handled by RuntimeWrapper.
    """

    def __init__(
        self,
        partition: str = "",
        account: str = "",
        qos: str = "",
        job_info_dir: str = "/var/run/rivault",
    ):
        """
        Initialize Slurm target.

        Args:
            partition: Default Slurm partition.
            account: Slurm account for billing.
            qos: Quality of service level.
            job_info_dir: Directory for job info files.
        """
        self.default_partition = partition or os.getenv("SLURM_PARTITION", "gpu")
        self.account = account or os.getenv("SLURM_ACCOUNT", "")
        self.qos = qos or os.getenv("SLURM_QOS", "")
        self.job_info_dir = Path(job_info_dir)

    @property
    def authority(self) -> str:
        return "slurm"

    async def submit(self, spec: JobSpec) -> AllocationInfo:
        """Submit job via sbatch."""
        if spec.execution_authority != "slurm":
            raise SubmissionError(
                f"SlurmTarget cannot handle authority '{spec.execution_authority}'"
            )

        # Build minimal sbatch script that allocates resources
        # RuntimeWrapper will start the actual container
        script = self._build_allocation_script(spec)

        try:
            # Write script to temp file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".sh",
                delete=False,
            ) as f:
                f.write(script)
                script_path = f.name

            # Submit via sbatch
            result = await self._run_sbatch(script_path)
            job_id = result.strip()

            logger.info(f"Submitted Slurm job {job_id} for {spec.name}")

            return AllocationInfo(
                job_id=job_id,
                node="",  # Unknown until allocated
                status=AllocationStatus.PENDING,
                allocated_at=None,
                metadata={"script_path": script_path},
                gpus=spec.gpus,
                cpus=spec.cpus,
                memory_mb=spec.memory_mb,
            )

        except subprocess.CalledProcessError as e:
            raise SubmissionError(f"sbatch failed: {e.stderr}") from e
        finally:
            # Clean up temp script
            with contextlib.suppress(Exception):
                os.unlink(script_path)

    async def cancel(self, job_id: str) -> bool:
        """Cancel job via scancel."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "scancel", job_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception as e:
            logger.warning(f"scancel failed for {job_id}: {e}")
            return False

    async def status(self, job_id: str) -> AllocationInfo:
        """Get job status via squeue/sacct."""
        # Try squeue first (for pending/running jobs)
        info = await self._squeue_status(job_id)
        if info:
            return info

        # Fall back to sacct (for completed jobs)
        info = await self._sacct_status(job_id)
        if info:
            return info

        return AllocationInfo(
            job_id=job_id,
            node="",
            status=AllocationStatus.UNKNOWN,
        )

    async def list_jobs(
        self,
        labels: dict[str, str] | None = None,
    ) -> list[AllocationInfo]:
        """List jobs via squeue."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "squeue",
                "--noheader",
                "--format=%i|%j|%T|%N|%C|%m|%b",
                "--me",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            jobs = []
            for line in stdout.decode().strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    job_id, name, state, node = parts[:4]

                    # Filter by labels if specified
                    if labels:
                        # Slurm doesn't have native labels, check job name
                        match = all(v in name for v in labels.values())
                        if not match:
                            continue

                    jobs.append(AllocationInfo(
                        job_id=job_id,
                        node=node,
                        status=self._parse_slurm_state(state),
                    ))

            return jobs

        except Exception as e:
            logger.warning(f"squeue failed: {e}")
            return []

    def _build_allocation_script(self, spec: JobSpec) -> str:
        """Build sbatch script that waits for container signal."""
        partition = spec.partition or self.default_partition

        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={spec.name}",
            f"#SBATCH --partition={partition}",
            f"#SBATCH --time={spec.time_limit}",
            f"#SBATCH --gres=gpu:{spec.gpus}",
            f"#SBATCH --cpus-per-task={spec.cpus}",
            f"#SBATCH --mem={spec.memory_mb}M",
        ]

        # Resolution order: per-submission (user record) > target instance > site default.
        effective_account = spec.slurm_account or self.account
        if effective_account:
            lines.append(f"#SBATCH --account={effective_account}")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")

        # Attribution: embed requesting user identity for sacct auditing.
        # This enables `sacct --format=Comment` to map jobs back to users
        # even when submitted via the UMRS master service account.
        attribution_parts = []
        if spec.requesting_user:
            attribution_parts.append(f"user={spec.requesting_user}")
        if spec.requesting_project:
            attribution_parts.append(f"project={spec.requesting_project}")
        if spec.requesting_session:
            attribution_parts.append(f"session={spec.requesting_session}")
        if attribution_parts:
            comment = ",".join(attribution_parts)
            lines.append(f"#SBATCH --comment={comment}")

        # Output files
        log_dir = spec.extra.get("log_dir", "/var/log/rivault")
        lines.extend([
            f"#SBATCH --output={log_dir}/job-%j.out",
            f"#SBATCH --error={log_dir}/job-%j.err",
            "",
            "# Attribution environment — propagated into container",
        ])

        # Export attribution env vars for container-level auditing
        if spec.requesting_user:
            lines.append(f'export UMRS_USER="{spec.requesting_user}"')
        if spec.requesting_project:
            lines.append(f'export UMRS_PROJECT="{spec.requesting_project}"')
        if spec.requesting_session:
            lines.append(f'export UMRS_SESSION="{spec.requesting_session}"')

        lines.extend([
            "",
            "# Write allocation info for discovery",
            f"INFO_DIR={self.job_info_dir}",
            'mkdir -p "$INFO_DIR"',
            'INFO_FILE="$INFO_DIR/rivault-$SLURM_JOB_ID.json"',
            "",
            "# Signal that allocation is ready",
            'cat > "$INFO_FILE" << EOF',
            '{',
            '  "job_id": "$SLURM_JOB_ID",',
            '  "node": "$SLURMD_NODENAME",',
            '  "status": "allocated",',
            '  "gpus": ' + str(spec.gpus) + ',',
            '  "cpus": ' + str(spec.cpus) + ',',
            '  "user": "' + (spec.requesting_user or "unknown") + '",',
            '  "project": "' + (spec.requesting_project or "") + '",',
            '  "allocated_at": "$(date -Iseconds)"',
            '}',
            'EOF',
            "",
            "# Wait for container to finish (or job timeout)",
            "# RuntimeWrapper will handle actual container launch",
            "while [ -f $INFO_FILE ]; do",
            "    sleep 10",
            "done",
        ])

        return "\n".join(lines)

    async def _run_sbatch(self, script_path: str) -> str:
        """Run sbatch and return job ID."""
        proc = await asyncio.create_subprocess_exec(
            "sbatch", "--parsable", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, "sbatch", stderr=stderr.decode()
            )

        return stdout.decode().strip()

    async def _squeue_status(self, job_id: str) -> AllocationInfo | None:
        """Get status from squeue."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "squeue",
                "--noheader",
                "--format=%i|%T|%N|%C|%m",
                "--job", job_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            line = stdout.decode().strip()
            if not line:
                return None

            parts = line.split("|")
            if len(parts) >= 3:
                _, state, node = parts[:3]
                return AllocationInfo(
                    job_id=job_id,
                    node=node,
                    status=self._parse_slurm_state(state),
                )
            return None

        except Exception:
            return None

    async def _sacct_status(self, job_id: str) -> AllocationInfo | None:
        """Get status from sacct (for completed jobs)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "sacct",
                "--noheader",
                "--format=JobID,State,NodeList",
                "--parsable2",
                "--jobs", job_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            for line in stdout.decode().strip().split("\n"):
                parts = line.split("|")
                if len(parts) >= 3 and parts[0] == job_id:
                    _, state, node = parts[:3]
                    return AllocationInfo(
                        job_id=job_id,
                        node=node,
                        status=self._parse_slurm_state(state),
                    )
            return None

        except Exception:
            return None

    def _parse_slurm_state(self, state: str) -> AllocationStatus:
        """Convert Slurm state to AllocationStatus."""
        state = state.upper()
        mapping = {
            "PENDING": AllocationStatus.PENDING,
            "CONFIGURING": AllocationStatus.PENDING,
            "RUNNING": AllocationStatus.RUNNING,
            "COMPLETING": AllocationStatus.RUNNING,
            "COMPLETED": AllocationStatus.COMPLETED,
            "FAILED": AllocationStatus.FAILED,
            "TIMEOUT": AllocationStatus.FAILED,
            "CANCELLED": AllocationStatus.CANCELLED,
            "PREEMPTED": AllocationStatus.CANCELLED,
        }
        return mapping.get(state, AllocationStatus.UNKNOWN)


# =============================================================================
# K8sTarget Implementation (U-1103)
# =============================================================================


class K8sTarget(ExecutionTarget):
    """
    Kubernetes-based execution target (U-1103).

    Creates Jobs for resource allocation, with RuntimeWrapper
    handling the actual container specification.
    """

    def __init__(
        self,
        namespace: str = "rivault-system",
        gpu_resource: str = "nvidia.com/gpu",
        kubeconfig: str | None = None,
    ):
        """
        Initialize K8s target.

        Args:
            namespace: Kubernetes namespace.
            gpu_resource: GPU resource name.
            kubeconfig: Path to kubeconfig file.
        """
        self.namespace = namespace or os.getenv("K8S_NAMESPACE", "rivault-system")
        self.gpu_resource = gpu_resource or os.getenv("K8S_GPU_RESOURCE", "nvidia.com/gpu")
        self.kubeconfig = kubeconfig or os.getenv("KUBECONFIG")

    @property
    def authority(self) -> str:
        return "k8s"

    async def submit(self, spec: JobSpec) -> AllocationInfo:
        """Create a K8s Job for allocation."""
        if spec.execution_authority != "k8s":
            raise SubmissionError(
                f"K8sTarget cannot handle authority '{spec.execution_authority}'"
            )

        job_name = f"rivault-{spec.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Build Job manifest
        manifest = self._build_job_manifest(job_name, spec)

        try:
            # Apply via kubectl
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "apply", "-f", "-",
                "--namespace", self.namespace,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate(json.dumps(manifest).encode())

            if proc.returncode != 0:
                raise SubmissionError(f"kubectl apply failed: {stderr.decode()}")

            logger.info(f"Created K8s Job {job_name}")

            return AllocationInfo(
                job_id=job_name,
                node="",  # Unknown until scheduled
                status=AllocationStatus.PENDING,
                gpus=spec.gpus,
                cpus=spec.cpus,
                memory_mb=spec.memory_mb,
            )

        except subprocess.CalledProcessError as e:
            raise SubmissionError(f"kubectl failed: {e}") from e

    async def cancel(self, job_id: str) -> bool:
        """Delete K8s Job."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "delete", "job", job_id,
                "--namespace", self.namespace,
                "--ignore-not-found",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to delete job {job_id}: {e}")
            return False

    async def status(self, job_id: str) -> AllocationInfo:
        """Get Job status via kubectl."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "get", "job", job_id,
                "--namespace", self.namespace,
                "-o", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0:
                return AllocationInfo(
                    job_id=job_id,
                    node="",
                    status=AllocationStatus.UNKNOWN,
                )

            data = json.loads(stdout.decode())
            status = self._parse_job_status(data)

            # Get node from pod
            node = await self._get_pod_node(job_id)

            return AllocationInfo(
                job_id=job_id,
                node=node,
                status=status,
            )

        except Exception as e:
            logger.warning(f"Failed to get job status: {e}")
            return AllocationInfo(
                job_id=job_id,
                node="",
                status=AllocationStatus.UNKNOWN,
            )

    async def list_jobs(
        self,
        labels: dict[str, str] | None = None,
    ) -> list[AllocationInfo]:
        """List Jobs with optional label selector."""
        try:
            cmd = [
                "kubectl", "get", "jobs",
                "--namespace", self.namespace,
                "-o", "json",
            ]

            if labels:
                selector = ",".join(f"{k}={v}" for k, v in labels.items())
                cmd.extend(["--selector", selector])

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0:
                return []

            data = json.loads(stdout.decode())
            jobs = []

            for item in data.get("items", []):
                job_id = item["metadata"]["name"]
                status = self._parse_job_status(item)
                jobs.append(AllocationInfo(
                    job_id=job_id,
                    node="",
                    status=status,
                ))

            return jobs

        except Exception as e:
            logger.warning(f"Failed to list jobs: {e}")
            return []

    def _build_job_manifest(self, name: str, spec: JobSpec) -> dict[str, Any]:
        """Build K8s Job manifest."""
        labels = {
            "app.kubernetes.io/name": "rivault",
            "app.kubernetes.io/component": "inference",
            **spec.labels,
        }

        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
                "labels": labels,
            },
            "spec": {
                "completions": 1,
                "parallelism": 1,
                "backoffLimit": 0,
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [{
                            "name": "placeholder",
                            "image": "busybox",
                            "command": ["sleep", "infinity"],
                            "resources": {
                                "requests": {
                                    "cpu": str(spec.cpus),
                                    "memory": f"{spec.memory_mb}Mi",
                                    self.gpu_resource: str(spec.gpus),
                                },
                                "limits": {
                                    "cpu": str(spec.cpus),
                                    "memory": f"{spec.memory_mb}Mi",
                                    self.gpu_resource: str(spec.gpus),
                                },
                            },
                        }],
                    },
                },
            },
        }

    def _parse_job_status(self, job: dict[str, Any]) -> AllocationStatus:
        """Parse Job status from JSON."""
        status = job.get("status", {})

        if status.get("succeeded", 0) > 0:
            return AllocationStatus.COMPLETED
        if status.get("failed", 0) > 0:
            return AllocationStatus.FAILED
        if status.get("active", 0) > 0:
            return AllocationStatus.RUNNING

        # Check conditions
        for condition in status.get("conditions", []):
            if condition.get("type") == "Complete" and condition.get("status") == "True":
                return AllocationStatus.COMPLETED
            if condition.get("type") == "Failed" and condition.get("status") == "True":
                return AllocationStatus.FAILED

        return AllocationStatus.PENDING

    async def _get_pod_node(self, job_name: str) -> str:
        """Get node name from Job's pod."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "get", "pods",
                "--namespace", self.namespace,
                "--selector", f"job-name={job_name}",
                "-o", "jsonpath={.items[0].spec.nodeName}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip()
        except Exception:
            return ""


# =============================================================================
# ComposeTarget Implementation (U-1104)
# =============================================================================


class ComposeTarget(ExecutionTarget):
    """
    Docker Compose-based execution target (U-1104).

    Uses docker-compose for local development/testing.
    Resource "allocation" is immediate (no scheduler).
    """

    def __init__(
        self,
        project_name: str = "mini-rivault",
        compose_file: str = "compose.yaml",
        work_dir: str = "",
    ):
        """
        Initialize Compose target.

        Args:
            project_name: Docker Compose project name.
            compose_file: Path to compose file.
            work_dir: Working directory for compose commands.
        """
        self.project_name = project_name or os.getenv("COMPOSE_PROJECT_NAME", "mini-rivault")
        self.compose_file = compose_file or os.getenv("COMPOSE_FILE", "compose.yaml")
        self.work_dir = work_dir or os.getcwd()

    @property
    def authority(self) -> str:
        return "compose"

    async def submit(self, spec: JobSpec) -> AllocationInfo:
        """Start compose services."""
        if spec.execution_authority != "compose":
            raise SubmissionError(
                f"ComposeTarget cannot handle authority '{spec.execution_authority}'"
            )

        # For compose, "allocation" is immediate
        job_id = f"{self.project_name}-{spec.name}"

        return AllocationInfo(
            job_id=job_id,
            node="localhost",
            status=AllocationStatus.ALLOCATED,
            allocated_at=datetime.now(),
            gpus=spec.gpus,
            cpus=spec.cpus,
            memory_mb=spec.memory_mb,
        )

    async def cancel(self, job_id: str) -> bool:
        """Stop compose project."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "compose",
                "-p", self.project_name,
                "-f", self.compose_file,
                "down",
                cwd=self.work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception as e:
            logger.warning(f"compose down failed: {e}")
            return False

    async def status(self, job_id: str) -> AllocationInfo:
        """Check if compose services are running."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "compose",
                "-p", self.project_name,
                "-f", self.compose_file,
                "ps", "--format", "json",
                cwd=self.work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()

            if proc.returncode != 0:
                return AllocationInfo(
                    job_id=job_id,
                    node="localhost",
                    status=AllocationStatus.UNKNOWN,
                )

            # Parse JSON output
            services = []
            for line in stdout.decode().strip().split("\n"):
                if line:
                    with contextlib.suppress(json.JSONDecodeError):
                        services.append(json.loads(line))

            if not services:
                return AllocationInfo(
                    job_id=job_id,
                    node="localhost",
                    status=AllocationStatus.COMPLETED,  # No services = stopped
                )

            # Check if all services are running
            all_running = all(
                s.get("State") == "running"
                for s in services
            )

            return AllocationInfo(
                job_id=job_id,
                node="localhost",
                status=AllocationStatus.RUNNING if all_running else AllocationStatus.PENDING,
            )

        except Exception as e:
            logger.warning(f"compose ps failed: {e}")
            return AllocationInfo(
                job_id=job_id,
                node="localhost",
                status=AllocationStatus.UNKNOWN,
            )

    async def list_jobs(
        self,
        labels: dict[str, str] | None = None,
    ) -> list[AllocationInfo]:
        """List compose projects (limited to current project)."""
        info = await self.status(self.project_name)
        if info.status in (AllocationStatus.RUNNING, AllocationStatus.PENDING):
            return [info]
        return []
