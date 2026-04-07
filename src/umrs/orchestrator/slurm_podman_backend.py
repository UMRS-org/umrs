"""
Slurm + Podman-Compose orchestrator backend.

Unlike :class:`SlurmBackend` (which generates a Singularity-based sbatch
script), this backend delegates to the site's existing ``serve-model.slurm``
wrapper.  The wrapper handles GPU detection, podman-compose profiles, image
loading, and container lifecycle — all parameterised via environment variables
that :class:`SiteConfig` exports.

Designed for the rccs-cloud deployment pattern:

    login node (UMRS)  ──sbatch──▸  compute node (serve-model.slurm)
                                         │
                                    podman-compose up
                                    (profiles: harness, vllm, vision-bentoml)

Job discovery uses squeue/sacct plus the Slurm output file to extract the
compute-node hostname.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    EndpointInfo,
    JobInfo,
    JobStatus,
    OrchestratorBackend,
    OrchestratorConfig,
)
from .site_config import SiteConfig, get_site_config

logger = logging.getLogger(__name__)


class SlurmPodmanBackend(OrchestratorBackend):
    """
    Orchestrator that submits the site's ``serve-model.slurm`` via sbatch.

    The site script is responsible for the full container lifecycle;
    this backend only manages Slurm job submission, status polling,
    and cleanup.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        site: Optional[SiteConfig] = None,
        slurm_script: Optional[str] = None,
    ):
        super().__init__(config)
        self._site = site
        self._slurm_script = slurm_script
        # output_dir stores sbatch stdout/stderr files
        self._output_dir = Path(
            os.environ.get("RIVAULT_OUTPUT_DIR", "/tmp/umrs-rivault-jobs")
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def site(self) -> SiteConfig:
        if self._site is None:
            self._site = get_site_config()
        return self._site

    @property
    def slurm_script(self) -> str:
        """Resolve the path to the site's serve-model.slurm."""
        if self._slurm_script:
            return self._slurm_script

        site = self.site

        # Prefer deploy/<site-name>/serve-model.slurm when repo_dir is known.
        repo_dir = site.runtime.repo_dir
        if repo_dir and site.name:
            candidate = os.path.join(repo_dir, "deploy", site.name, "serve-model.slurm")
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate):
                return candidate

        # Fallback: infer repo root from runtime.rivault_dir.
        rivault_dir = site.runtime.rivault_dir
        if rivault_dir and site.name:
            candidate = os.path.join(
                os.path.dirname(rivault_dir),
                "..",
                "..",
                "deploy",
                site.name,
                "serve-model.slurm",
            )
            candidate = os.path.normpath(candidate)
            if os.path.isfile(candidate):
                return candidate

        # Fallback to RIVAULT_SLURM_SCRIPT env
        return os.environ.get(
            "RIVAULT_SLURM_SCRIPT",
            str(Path.home() / ".config" / "umrs" / "serve-model.slurm"),
        )

    @property
    def backend_name(self) -> str:
        return "slurm-podman"

    # ------------------------------------------------------------------
    # sbatch environment
    # ------------------------------------------------------------------

    def _sbatch_env(self, **overrides: str) -> Dict[str, str]:
        """Build the environment dict passed to sbatch.

        Merges the current ``os.environ``, site-config exports, and any
        caller-supplied *overrides*.
        """
        env = dict(os.environ)  # inherit caller env
        env.update(self.site.to_env())  # layer site-config on top
        env.update(overrides)  # explicit overrides win
        return env

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    async def submit_job(
        self,
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> JobInfo:
        """Submit the site's serve-model.slurm via sbatch."""
        script = self.slurm_script
        if not os.path.isfile(script):
            raise FileNotFoundError(f"Slurm script not found: {script}")

        site = self.site

        # Build sbatch arguments from site config
        sbatch_args = [
            "sbatch",
            "--parsable",
            f"--output={self._output_dir}/job-%j.out",
            f"--error={self._output_dir}/job-%j.err",
        ]

        if site.scheduler.partition:
            sbatch_args.append(f"--partition={site.scheduler.partition}")
        if site.scheduler.time_limit:
            sbatch_args.append(f"--time={site.scheduler.time_limit}")
        if site.scheduler.gpus:
            sbatch_args.append(f"--gres=gpu:{site.scheduler.gpus}")
        if site.scheduler.cpus:
            sbatch_args.append(f"--cpus-per-task={site.scheduler.cpus}")
        if site.scheduler.memory:
            sbatch_args.append(f"--mem={site.scheduler.memory}")
        if site.scheduler.account:
            sbatch_args.append(f"--account={site.scheduler.account}")
        if site.scheduler.qos:
            sbatch_args.append(f"--qos={site.scheduler.qos}")
        for extra in site.scheduler.extra_sbatch:
            sbatch_args.append(extra)

        sbatch_args.append(script)

        # Build environment
        env_overrides: Dict[str, str] = {}
        if model_path:
            env_overrides["VLLM_MODEL_PATH"] = model_path
        env_overrides.update({k: str(v) for k, v in kwargs.items()
                              if isinstance(v, (str, int, float))})

        env = self._sbatch_env(**env_overrides)

        logger.info("Submitting: %s", " ".join(sbatch_args))

        proc = await asyncio.create_subprocess_exec(
            *sbatch_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed (rc={proc.returncode}): {stderr.decode().strip()}"
            )

        job_id = stdout.decode().strip().split(";")[0]
        logger.info("Submitted Slurm job %s", job_id)

        return JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_path=model_path or site.vllm.model,
            created_at=datetime.now(),
            backend=self.backend_name,
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_job_status(self, job_id: str) -> JobInfo:
        """Poll squeue / sacct for job status + extract node from output."""
        # Try squeue first (live jobs)
        node, slurm_state = await self._squeue_info(job_id)

        if slurm_state:
            status = self._map_slurm_state(slurm_state)
        else:
            # Job left the queue — check sacct
            sacct_state = await self._sacct_state(job_id)
            status = self._map_slurm_state(sacct_state) if sacct_state else JobStatus.UNKNOWN

        # If running, try to resolve the node from the Slurm output
        if status in (JobStatus.STARTING, JobStatus.RUNNING, JobStatus.READY):
            if not node:
                node = await self._node_from_output(job_id)

        endpoint = None
        if node and status in (JobStatus.STARTING, JobStatus.RUNNING, JobStatus.READY):
            endpoint = EndpointInfo(
                node=node,
                litellm_port=self.site.network.litellm_port,
            )

        return JobInfo(
            job_id=job_id,
            status=status,
            endpoint=endpoint,
            backend=self.backend_name,
        )

    async def wait_for_ready(
        self,
        job_id: str,
        timeout_sec: Optional[int] = None,
    ) -> EndpointInfo:
        """Block until endpoint passes readiness check."""
        timeout = timeout_sec or self.config.startup_timeout_sec
        poll = self.config.readiness_poll_interval_sec
        elapsed = 0

        while elapsed < timeout:
            job = await self.get_job_status(job_id)

            if job.status == JobStatus.FAILED:
                raise RuntimeError(f"Job {job_id} failed: {job.error_message}")
            if job.status == JobStatus.STOPPED:
                raise RuntimeError(f"Job {job_id} stopped unexpectedly")

            if job.endpoint:
                if await self.check_readiness(job.endpoint):
                    return job.endpoint

            await asyncio.sleep(poll)
            elapsed += poll

        raise TimeoutError(
            f"Job {job_id} did not become ready within {timeout}s"
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self, job_id: str, force: bool = False) -> None:
        """Cancel a Slurm job (SIGTERM or SIGKILL)."""
        signal = "KILL" if force else "TERM"
        proc = await asyncio.create_subprocess_exec(
            "scancel", f"--signal={signal}", job_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Remove output files
        for suffix in (".out", ".err"):
            p = self._output_dir / f"job-{job_id}{suffix}"
            if p.exists():
                p.unlink(missing_ok=True)

    async def list_jobs(
        self,
        status_filter: Optional[List[JobStatus]] = None,
    ) -> List[JobInfo]:
        """List UMRS-related Slurm jobs."""
        proc = await asyncio.create_subprocess_exec(
            "squeue",
            "-u", os.environ.get("USER", ""),
            "-n", "umrs-rivault",
            "-h",
            "-o", "%i",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        jobs: List[JobInfo] = []
        if stdout:
            for line in stdout.decode().strip().split("\n"):
                jid = line.strip()
                if jid:
                    job = await self.get_job_status(jid)
                    if status_filter is None or job.status in status_filter:
                        jobs.append(job)
        return jobs

    async def cleanup_orphans(self) -> int:
        """Remove output files for jobs that are no longer in Slurm."""
        cleaned = 0
        for out_file in self._output_dir.glob("job-*.out"):
            match = re.search(r"job-(\d+)\.out", out_file.name)
            if not match:
                continue
            jid = match.group(1)
            job = await self.get_job_status(jid)
            if job.status in (JobStatus.UNKNOWN, JobStatus.STOPPED, JobStatus.FAILED):
                for suffix in (".out", ".err"):
                    p = self._output_dir / f"job-{jid}{suffix}"
                    p.unlink(missing_ok=True)
                cleaned += 1
        return cleaned

    async def health_check(self) -> Dict[str, Any]:
        """Verify Slurm accessibility and script existence."""
        # Check sinfo
        try:
            proc = await asyncio.create_subprocess_exec(
                "sinfo", "-h",
                "-p", self.site.scheduler.partition,
                "-o", "%P %a %D",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                return {
                    "backend": self.backend_name,
                    "healthy": False,
                    "message": f"sinfo failed: {stderr.decode().strip()}",
                }
        except FileNotFoundError:
            return {
                "backend": self.backend_name,
                "healthy": False,
                "message": "Slurm commands not found",
            }

        script_ok = os.path.isfile(self.slurm_script)

        return {
            "backend": self.backend_name,
            "healthy": script_ok,
            "message": "OK" if script_ok else f"Script missing: {self.slurm_script}",
            "partition_info": stdout.decode().strip(),
            "slurm_script": self.slurm_script,
            "site": self.site.name,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _squeue_info(self, job_id: str) -> tuple[Optional[str], Optional[str]]:
        """Return (node, state) from squeue, or (None, None) if not found."""
        proc = await asyncio.create_subprocess_exec(
            "squeue", "-j", job_id, "-h", "-o", "%N %T",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        line = stdout.decode().strip()
        if not line:
            return None, None
        parts = line.split(None, 1)
        node = parts[0] if len(parts) >= 1 and parts[0] != "(null)" else None
        state = parts[1] if len(parts) >= 2 else None
        return node, state

    async def _sacct_state(self, job_id: str) -> Optional[str]:
        """Return the first sacct State string, or None."""
        proc = await asyncio.create_subprocess_exec(
            "sacct", "-j", job_id, "-n", "-o", "State",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        lines = stdout.decode().strip().split("\n")
        return lines[0].strip() if lines and lines[0].strip() else None

    async def _node_from_output(self, job_id: str) -> Optional[str]:
        """Try to extract the compute-node hostname from the job output file."""
        out_path = self._output_dir / f"job-{job_id}.out"
        if not out_path.exists():
            return None
        try:
            text = out_path.read_text(errors="replace")
            # serve-model.slurm echoes "Node: <hostname>"
            m = re.search(r"Node:\s+(\S+)", text)
            if m:
                return m.group(1)
        except OSError:
            pass
        return None

    @staticmethod
    def _map_slurm_state(state: str) -> JobStatus:
        """Map a Slurm state string to :class:`JobStatus`."""
        s = state.upper().split()[0] if state else ""
        mapping = {
            "PENDING": JobStatus.PENDING,
            "CONFIGURING": JobStatus.PENDING,
            "RUNNING": JobStatus.STARTING,  # may not be ready yet
            "COMPLETING": JobStatus.RUNNING,
            "COMPLETED": JobStatus.STOPPED,
            "FAILED": JobStatus.FAILED,
            "CANCELLED": JobStatus.STOPPED,
            "TIMEOUT": JobStatus.FAILED,
            "NODE_FAIL": JobStatus.FAILED,
            "PREEMPTED": JobStatus.FAILED,
        }
        return mapping.get(s, JobStatus.UNKNOWN)
