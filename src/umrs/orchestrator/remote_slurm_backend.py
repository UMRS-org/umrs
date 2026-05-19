"""
Remote Slurm Backend — SSH-based Slurm orchestration for front-end deployments.

Extends the standard SlurmBackend to route all Slurm CLI commands (sbatch,
squeue, sacct, scancel) through an SSH ControlMaster connection to the login
node. Used when UMRS runs on a front-end server that lacks direct Slurm access.

Configuration in site.yaml::

    scheduler:
      kind: slurm-remote
      partition: ng-dgx-m0,ng-dgx-m1,ng-dgx-m2,ng-dgx-m3
      time_limit: "04:00:00"
      extra_sbatch: []

    extra:
      ssh_target: rccs-cloud
      ssh_user: ""
      ssh_control_path: ""
      ssh_connect_timeout: 15
      ssh_command_timeout: 30
      ssh_reverse_tunnels:
        - "127.0.0.1:9000:127.0.0.1:9000"
      slurm_job_info_dir: /lvs0/rccs-dmp/masaru.nagaso/ai4s-platform/umrs/deploy/rccs-cloud
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from .base import (
    EndpointInfo,
    JobInfo,
    JobStatus,
)
from .slurm_backend import SlurmBackend
from .ssh_control import SSHControlConfig, SSHControlManager

logger = logging.getLogger(__name__)


class RemoteSlurmBackend(SlurmBackend):
    """
    SSH-based remote Slurm backend for front-end server deployments.

    All Slurm CLI commands are executed remotely via SSH ControlMaster.
    Job info files on the shared filesystem are read via SSH cat.
    """

    def __init__(self, config: Any, ssh_config: Optional[SSHControlConfig] = None) -> None:
        super().__init__(config)
        # Build SSH config from site config extra section or explicit param.
        if ssh_config:
            self._ssh_config = ssh_config
        else:
            extra = getattr(config, "extra", {}) or {}
            if not extra.get("ssh_target"):
                # Fallback: load extra from site config directly
                try:
                    from .site_config import load_site_config
                    site = load_site_config()
                    extra = site.extra or {}
                except Exception:
                    pass
            self._ssh_config = SSHControlConfig.from_site_config(extra)

        self._ssh: Optional[SSHControlManager] = None

    @property
    def backend_name(self) -> str:
        return "slurm-remote"

    async def _get_ssh(self) -> SSHControlManager:
        """Get or create the SSH control manager."""
        if self._ssh is None:
            self._ssh = SSHControlManager(self._ssh_config)
        await self._ssh.ensure_connected()
        return self._ssh

    async def submit_job(
        self,
        model_path: str | None = None,
        **kwargs: Any,
    ) -> JobInfo:
        """Submit a Slurm job via remote SSH sbatch."""
        model = model_path or self.config.model_path
        if not model:
            raise ValueError("model_path is required (set VLLM_MODEL_PATH or pass explicitly)")

        # Generate sbatch script content.
        script_content = self._build_sbatch_script(model, **kwargs)

        # Submit via SSH: pipe script to remote sbatch.
        ssh = await self._get_ssh()
        result = await ssh.run_remote_with_stdin(
            "sbatch --parsable",
            script_content.encode(),
            timeout=60,
        )

        if not result.ok:
            raise RuntimeError(f"Remote sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split(";")[0]  # Handle "JOBID;CLUSTER" format

        return JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_path=model,
            created_at=datetime.now(),
            backend=self.backend_name,
        )

    async def get_job_status(self, job_id: str) -> JobInfo:
        """Get status of a Slurm job via SSH."""
        # Try job info file first (read remotely).
        info_content = await self._read_remote_job_info(job_id)
        if info_content is not None:
            info = self._parse_info_content(info_content)
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

        # Fallback to squeue/sacct.
        status = await self._query_slurm_job_status(job_id)
        return JobInfo(job_id=job_id, status=status, backend=self.backend_name)

    async def _read_remote_job_info(self, job_id: str) -> Optional[str]:
        """Read job info file from remote filesystem via SSH."""
        info_dir = self.config.slurm_job_info_dir
        info_path = f"{info_dir}/rivault-job-{job_id}.info"
        ssh = await self._get_ssh()
        return await ssh.read_remote_file(info_path)

    def _parse_info_content(self, content: str) -> dict[str, str]:
        """Parse job info file content into a dict."""
        info: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                info[key] = value
        return info

    async def _query_slurm_job_status(self, job_id: str) -> JobStatus:
        """Query Slurm job status via remote squeue/sacct."""
        status = await self._query_squeue_status(job_id)
        if status is not None:
            return status
        return await self._query_sacct_status(job_id)

    async def _query_squeue_status(self, job_id: str) -> JobStatus | None:
        """Query squeue for active job status via SSH."""
        ssh = await self._get_ssh()
        result = await ssh.run_remote(f"squeue -j {job_id} -h -o %T")

        if not result.ok or not result.stdout.strip():
            return None
        return self._status_from_slurm_state(result.stdout.strip())

    async def _query_sacct_status(self, job_id: str) -> JobStatus:
        """Query sacct for completed job status via SSH."""
        ssh = await self._get_ssh()
        result = await ssh.run_remote(f"sacct -j {job_id} -n -o State")

        state = result.stdout.strip().split("\n")[0].strip() if result.stdout else ""
        return self._status_from_accounting_state(state)

    async def cleanup(self, job_id: str, force: bool = False) -> None:
        """Cancel a Slurm job via remote scancel."""
        signal = "KILL" if force else "TERM"
        ssh = await self._get_ssh()
        await ssh.run_remote(f"scancel --signal={signal} {job_id}")

        # Clean up remote info file.
        info_dir = self.config.slurm_job_info_dir
        info_path = f"{info_dir}/rivault-job-{job_id}.info"
        await ssh.run_remote(f"rm -f {info_path}")

    async def list_jobs(
        self,
        status_filter: list[JobStatus] | None = None,
    ) -> list[JobInfo]:
        """List all Mini-RiVault jobs via SSH."""
        jobs = await self._remote_jobs_from_info_files(status_filter)
        await self._remote_append_squeue_jobs(jobs, status_filter)
        return jobs

    async def _remote_jobs_from_info_files(
        self,
        status_filter: list[JobStatus] | None,
    ) -> list[JobInfo]:
        """List job info files on remote filesystem."""
        info_dir = self.config.slurm_job_info_dir
        ssh = await self._get_ssh()
        result = await ssh.run_remote(
            f"ls {info_dir}/rivault-job-*.info 2>/dev/null || true"
        )

        jobs: list[JobInfo] = []
        if not result.stdout.strip():
            return jobs

        for line in result.stdout.strip().splitlines():
            match = re.search(r"rivault-job-(\d+)\.info", line)
            if match:
                job_id = match.group(1)
                job = await self.get_job_status(job_id)
                if status_filter is None or job.status in status_filter:
                    jobs.append(job)
        return jobs

    async def _remote_append_squeue_jobs(
        self,
        jobs: list[JobInfo],
        status_filter: list[JobStatus] | None,
    ) -> None:
        """Query squeue for running jobs via SSH."""
        user = os.environ.get("USER", "")
        ssh = await self._get_ssh()
        result = await ssh.run_remote(
            f"squeue -u {user} -n mini-rivault -h -o %i"
        )

        if not result.stdout.strip():
            return

        for job_id in result.stdout.strip().split("\n"):
            job_id = job_id.strip()
            if not job_id or any(j.job_id == job_id for j in jobs):
                continue
            job = await self.get_job_status(job_id)
            if status_filter is None or job.status in status_filter:
                jobs.append(job)

    async def cleanup_orphans(self) -> int:
        """Clean up orphaned job info files via SSH."""
        info_dir = self.config.slurm_job_info_dir
        ssh = await self._get_ssh()

        result = await ssh.run_remote(
            f"ls {info_dir}/rivault-job-*.info 2>/dev/null || true"
        )
        if not result.stdout.strip():
            return 0

        cleaned = 0
        for line in result.stdout.strip().splitlines():
            match = re.search(r"rivault-job-(\d+)\.info", line)
            if not match:
                continue
            job_id = match.group(1)

            # Check if job still exists in Slurm.
            check = await ssh.run_remote(f"squeue -j {job_id} -h")
            if not check.stdout.strip():
                await ssh.run_remote(f"rm -f {info_dir}/rivault-job-{job_id}.info")
                cleaned += 1

        return cleaned

    async def health_check(self) -> dict[str, Any]:
        """Check if Slurm is accessible via SSH."""
        try:
            ssh = await self._get_ssh()
            partition = self.config.slurm_partition
            result = await ssh.run_remote(
                f"sinfo -h -p {partition} -o '%P %a %D'"
            )

            if not result.ok:
                return {
                    "backend": self.backend_name,
                    "healthy": False,
                    "message": f"Remote sinfo failed: {result.stderr}",
                    "ssh_target": self._ssh_config.ssh_target,
                }

            return {
                "backend": self.backend_name,
                "healthy": True,
                "message": "Remote Slurm cluster accessible via SSH",
                "ssh_target": self._ssh_config.ssh_target,
                "partition_info": result.stdout.strip(),
            }
        except ConnectionError as exc:
            return {
                "backend": self.backend_name,
                "healthy": False,
                "message": f"SSH connection failed: {exc}",
                "ssh_target": self._ssh_config.ssh_target,
            }

    async def check_readiness(self, endpoint: EndpointInfo) -> bool:
        """Check endpoint readiness via SSH curl (front-end can't reach compute directly)."""
        ssh = await self._get_ssh()
        url = f"http://{endpoint.node}:{endpoint.litellm_port}/v1/models"
        result = await ssh.run_remote(
            f"curl -sf --max-time 5 {url}",
            timeout=10,
        )
        return result.ok

    async def shutdown(self) -> None:
        """Clean shutdown of SSH connection."""
        if self._ssh:
            await self._ssh.close()
            self._ssh = None
