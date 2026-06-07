"""
Slurm orchestrator backend for Mini-RiVault.

Manages LLM inference endpoints via Slurm job scheduling and Singularity containers.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import (
    EndpointInfo,
    JobInfo,
    JobStatus,
    OrchestratorBackend,
)
from .slurm_script import render_sbatch_script


class SlurmBackend(OrchestratorBackend):
    """
    Slurm-based orchestrator for HPC environments.

    Uses sbatch for job submission and Singularity for containerization.
    Job info files are written to a shared directory for endpoint discovery.
    """

    @property
    def backend_name(self) -> str:
        return "slurm"

    def _job_info_path(self, job_id: str) -> Path:
        """Get path to job info file."""
        return Path(self.config.slurm_job_info_dir) / f"rivault-job-{job_id}.info"

    def _parse_job_info_file(self, path: Path) -> dict[str, str]:
        """Parse a job info file into a dict."""
        info: dict[str, str] = {}
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if "=" in line:
                        key, value = line.split("=", 1)
                        info[key] = value
        return info

    def _status_from_info(self, info: dict[str, str]) -> JobStatus:
        """Convert job info status string to JobStatus enum."""
        status_map = {
            "pending": JobStatus.PENDING,
            "starting": JobStatus.STARTING,
            "ready": JobStatus.READY,
            "running": JobStatus.RUNNING,
            "failed": JobStatus.FAILED,
            "stopped": JobStatus.STOPPED,
        }
        return status_map.get(info.get("RIVAULT_STATUS", "").lower(), JobStatus.UNKNOWN)

    def _build_sbatch_script(self, model_path: str, **kwargs: Any) -> str:
        """Generate sbatch script content."""
        return render_sbatch_script(self.config, model_path, **kwargs)

    async def submit_job(
        self,
        model_path: str | None = None,
        **kwargs: Any,
    ) -> JobInfo:
        """Submit a Slurm job for Mini-RiVault."""
        model = model_path or self.config.model_path
        if not model:
            raise ValueError("model_path is required (set VLLM_MODEL_PATH or pass explicitly)")

        # Generate sbatch script
        script_content = self._build_sbatch_script(model, **kwargs)

        # Submit via sbatch
        proc = await asyncio.create_subprocess_exec(
            "sbatch",
            "--parsable",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate(script_content.encode())

        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {stderr.decode()}")

        job_id = stdout.decode().strip().split(";")[0]  # Handle "JOBID;CLUSTER" format

        return JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_path=model,
            created_at=datetime.now(),
            backend=self.backend_name,
        )

    async def get_job_status(self, job_id: str) -> JobInfo:
        """Get status of a Slurm job."""
        info_path = self._job_info_path(job_id)
        if info_path.exists():
            return self._status_from_job_info(job_id, info_path)

        status = await self._query_slurm_job_status(job_id)
        return JobInfo(job_id=job_id, status=status, backend=self.backend_name)

    def _status_from_job_info(self, job_id: str, info_path: Path) -> JobInfo:
        info = self._parse_job_info_file(info_path)
        status = self._status_from_info(info)
        endpoint = self._endpoint_from_info(status, info)
        return JobInfo(
            job_id=job_id,
            status=status,
            endpoint=endpoint,
            model_path=info.get("RIVAULT_MODEL_PATH", ""),
            backend=self.backend_name,
            metadata=info,
        )

    def _endpoint_from_info(
        self,
        status: JobStatus,
        info: dict[str, str],
    ) -> EndpointInfo | None:
        if status not in (JobStatus.READY, JobStatus.RUNNING):
            return None
        return EndpointInfo(
            node=info.get("RIVAULT_NODE", "unknown"),
            litellm_port=int(info.get("RIVAULT_LITELLM_PORT", "11434")),
            vllm_port=int(info.get("RIVAULT_VLLM_PORT", "8000")),
        )

    async def _query_slurm_job_status(self, job_id: str) -> JobStatus:
        status = await self._query_squeue_status(job_id)
        if status is not None:
            return status
        return await self._query_sacct_status(job_id)

    async def _query_squeue_status(self, job_id: str) -> JobStatus | None:
        proc = await asyncio.create_subprocess_exec(
            "squeue",
            "-j", job_id,
            "-h",
            "-o", "%T",  # State
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        if proc.returncode != 0 or not stdout.strip():
            return None
        return self._status_from_slurm_state(stdout.decode().strip())

    async def _query_sacct_status(self, job_id: str) -> JobStatus:
        proc = await asyncio.create_subprocess_exec(
            "sacct",
            "-j", job_id,
            "-n",
            "-o", "State",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        state = stdout.decode().strip().split("\n")[0].strip() if stdout else ""
        return self._status_from_accounting_state(state)

    def _status_from_slurm_state(self, state: str) -> JobStatus:
        state_map = {
            "PENDING": JobStatus.PENDING,
            "RUNNING": JobStatus.STARTING,
            "COMPLETING": JobStatus.RUNNING,
            "COMPLETED": JobStatus.STOPPED,
            "FAILED": JobStatus.FAILED,
            "CANCELLED": JobStatus.STOPPED,
        }
        return state_map.get(state, JobStatus.UNKNOWN)

    def _status_from_accounting_state(self, state: str) -> JobStatus:
        if "COMPLETED" in state:
            return JobStatus.STOPPED
        if "FAILED" in state or "CANCELLED" in state:
            return JobStatus.FAILED
        return JobStatus.UNKNOWN

    async def wait_for_ready(
        self,
        job_id: str,
        timeout_sec: int | None = None,
    ) -> EndpointInfo:
        """Wait for job to be ready."""
        timeout = timeout_sec or self.config.startup_timeout_sec
        poll_interval = self.config.readiness_poll_interval_sec
        elapsed = 0

        while elapsed < timeout:
            job = await self.get_job_status(job_id)

            if job.status == JobStatus.FAILED:
                raise RuntimeError(f"Job {job_id} failed: {job.error_message}")

            if job.is_ready() and job.endpoint:
                # Double-check with actual HTTP call
                if await self.check_readiness(job.endpoint):
                    return job.endpoint

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Job {job_id} did not become ready within {timeout}s")

    async def cleanup(self, job_id: str, force: bool = False) -> None:
        """Cancel a Slurm job."""
        signal = "KILL" if force else "TERM"
        proc = await asyncio.create_subprocess_exec(
            "scancel",
            f"--signal={signal}",
            job_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Clean up info file
        info_path = self._job_info_path(job_id)
        if info_path.exists():
            info_path.unlink()

    async def list_jobs(
        self,
        status_filter: list[JobStatus] | None = None,
    ) -> list[JobInfo]:
        """List all Mini-RiVault jobs."""
        jobs = await self._jobs_from_info_files(status_filter)
        await self._append_squeue_jobs(jobs, status_filter)
        return jobs

    async def _jobs_from_info_files(
        self,
        status_filter: list[JobStatus] | None,
    ) -> list[JobInfo]:
        jobs: list[JobInfo] = []
        info_dir = Path(self.config.slurm_job_info_dir)
        if info_dir.exists():
            for info_file in info_dir.glob("rivault-job-*.info"):
                match = re.search(r"rivault-job-(\d+)\.info", info_file.name)
                if match:
                    job_id = match.group(1)
                    job = await self.get_job_status(job_id)
                    if status_filter is None or job.status in status_filter:
                        jobs.append(job)
        return jobs

    async def _append_squeue_jobs(
        self,
        jobs: list[JobInfo],
        status_filter: list[JobStatus] | None,
    ) -> None:
        """Append active squeue jobs not already in the list.

        Uses a single squeue call returning job ID + state (%i %T) to
        avoid N per-job subprocess calls.
        """
        user = os.environ.get("USER", "")
        proc = await asyncio.create_subprocess_exec(
            "squeue",
            "-u", user,
            "-h",
            "-o", "%i|%T",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        if not stdout:
            return

        seen = {job.job_id for job in jobs}
        for line in stdout.decode().strip().split("\n"):
            if not line or "|" not in line:
                continue
            job_id, state = line.strip().split("|", 1)
            if not job_id or job_id in seen:
                continue
            status = self._status_from_slurm_state(state)
            if status_filter is None or status in status_filter:
                jobs.append(JobInfo(
                    job_id=job_id,
                    status=status,
                    backend=self.backend_name,
                ))
                seen.add(job_id)

    async def cleanup_orphans(self) -> int:
        """Clean up orphaned job info files whose Slurm jobs are gone.

        Uses a single squeue call for all user jobs instead of N+1
        per-file subprocess calls (O(N) → O(1) subprocesses).
        """
        cleaned = 0
        info_dir = Path(self.config.slurm_job_info_dir)

        if not info_dir.exists():
            return 0

        # Collect all known job IDs from info files.
        known_ids: dict[str, Path] = {}
        for info_file in info_dir.glob("rivault-job-*.info"):
            match = re.search(r"rivault-job-(\d+)\.info", info_file.name)
            if match:
                known_ids[match.group(1)] = info_file

        if not known_ids:
            return 0

        # Batch-query squeue for all user jobs to avoid N subprocess calls.
        user = os.environ.get("USER", "")
        active_ids: set[str] = set()
        proc = await asyncio.create_subprocess_exec(
            "squeue",
            "-u", user,
            "-h",
            "-o", "%i",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if stdout:
            for line in stdout.decode().strip().split("\n"):
                jid = line.strip()
                if jid:
                    active_ids.add(jid)

        # Cross-reference: remove info files for jobs not in squeue.
        for job_id, info_file in known_ids.items():
            if job_id not in active_ids:
                info_file.unlink()
                cleaned += 1

        return cleaned

    async def health_check(self) -> dict[str, Any]:
        """Check if Slurm is accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "sinfo",
                "-h",
                "-p", self.config.slurm_partition,
                "-o", "%P %a %D",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return {
                    "backend": self.backend_name,
                    "healthy": False,
                    "message": f"sinfo failed: {stderr.decode()}",
                }

            return {
                "backend": self.backend_name,
                "healthy": True,
                "message": "Slurm cluster accessible",
                "partition_info": stdout.decode().strip(),
            }
        except FileNotFoundError:
            return {
                "backend": self.backend_name,
                "healthy": False,
                "message": "Slurm commands not found (sinfo)",
            }
