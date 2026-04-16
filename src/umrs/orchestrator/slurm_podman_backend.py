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
from umrs.shared.config import get_default_tmp_subdir, get_default_umrs_cfg_dir

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
        # Keep Slurm output on durable shared storage so wrapper launch context
        # and readiness clues survive normal cleanup.
        self._output_dir = self._resolve_output_dir()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._control_dir = self._output_dir / "control"
        self._control_dir.mkdir(parents=True, exist_ok=True)

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

    @staticmethod
    def _partition_candidates(raw: str) -> List[str]:
        return [candidate.strip() for candidate in raw.split(",") if candidate.strip()]

    # ------------------------------------------------------------------
    # sbatch environment
    # ------------------------------------------------------------------

    def _sbatch_env(self, **overrides: str) -> Dict[str, str]:
        """Build the environment dict passed to sbatch.

        Merges the current ``os.environ``, site-config exports, and any
        caller-supplied *overrides*.
        """
        env = self.site.to_env()
        env.update(os.environ)
        env.update(overrides)  # explicit overrides win
        return env

    def _resolve_output_dir(self) -> Path:
        """Resolve a durable directory for Slurm stdout/stderr capture."""
        explicit_dir = os.environ.get("RIVAULT_OUTPUT_DIR") or os.environ.get(
            "SLURM_JOB_INFO_DIR"
        )
        if explicit_dir:
            return Path(explicit_dir).expanduser()

        site = self.site
        if site.runtime.repo_dir and site.name:
            return (
                Path(site.runtime.repo_dir)
                / "deploy"
                / site.name
                / "logs"
                / "slurm"
            )

        return Path(get_default_umrs_cfg_dir()).expanduser() / get_default_tmp_subdir()

    def _stop_request_path(self, job_id: str) -> Path:
        """Return the shared control file used for graceful stop requests."""
        return self._control_dir / f"stop-{job_id}.request"

    async def _wait_for_job_exit(self, job_id: str, timeout_sec: int) -> bool:
        """Wait briefly for a job to leave the queue after a graceful stop request."""
        elapsed = 0
        poll_interval = 1

        while elapsed < timeout_sec:
            job = await self.get_job_status(job_id)
            if job.status in (JobStatus.STOPPED, JobStatus.FAILED, JobStatus.UNKNOWN):
                return True
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        return False

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

        # Per-request partition override: validate against site candidates
        partition_override = kwargs.pop("partition_override", None)
        nodelist_hint = kwargs.pop("nodelist_hint", None)

        if partition_override:
            allowed = self._partition_candidates(site.scheduler.partition)
            if allowed and partition_override not in allowed:
                raise ValueError(
                    f"Partition '{partition_override}' is not in site-configured "
                    f"candidates: {allowed}"
                )
            selected_partition = partition_override
            logger.info(
                "Using per-request partition override: %s", selected_partition,
            )
        else:
            selected_partition = await self._select_submission_partition()

        # Build sbatch arguments from site config
        sbatch_args = [
            "sbatch",
            "--parsable",
            f"--output={self._output_dir}/slurm-%j.out",
            f"--error={self._output_dir}/slurm-%j.err",
        ]

        if selected_partition:
            sbatch_args.append(f"--partition={selected_partition}")
        if nodelist_hint:
            sbatch_args.append(f"--nodelist={nodelist_hint}")
            logger.info("Using per-request nodelist hint: %s", nodelist_hint)
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
        requested_env = kwargs.pop("env", None)
        if isinstance(requested_env, dict):
            env_overrides.update(
                {
                    key: str(value)
                    for key, value in requested_env.items()
                    if isinstance(value, (str, int, float))
                }
            )
        if model_path:
            env_overrides["VLLM_MODEL_PATH"] = model_path
        if selected_partition:
            env_overrides["SLURM_PARTITION"] = selected_partition
        env_overrides.setdefault("RIVAULT_CONTROL_DIR", str(self._control_dir))
        env_overrides.update(
            {
                key: str(value)
                for key, value in kwargs.items()
                if isinstance(value, (str, int, float))
            }
        )

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
        """Cancel a Slurm job.

        Graceful cleanup must cancel the job itself so Slurm signals the batch
        shell and applies its normal KillWait handling. Forced cleanup must
        still reach the batch shell, not only child steps.
        """
        stop_request = self._stop_request_path(job_id)

        if not force:
            try:
                stop_request.touch()
            except OSError as exc:
                logger.warning(
                    "failed to create graceful stop request for %s: %s", job_id, exc
                )
            else:
                graceful_timeout = int(os.environ.get("RIVAULT_GRACEFUL_STOP_TIMEOUT", "20"))
                logger.info(
                    "requested graceful compute-node stop for %s via %s",
                    job_id,
                    stop_request,
                )
                if await self._wait_for_job_exit(job_id, graceful_timeout):
                    stop_request.unlink(missing_ok=True)
                    return

        cmd = ["scancel", "-Q"]
        if force:
            cmd.extend(["--full", "--signal=KILL"])

        cmd.append(job_id)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(
                "scancel failed for %s (force=%s, rc=%s): %s",
                job_id,
                force,
                proc.returncode,
                stderr.decode().strip(),
            )
        if stop_request.exists():
            stop_request.unlink(missing_ok=True)

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
        """Remove legacy output files for jobs that are no longer in Slurm."""
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

    async def _select_submission_partition(self) -> str:
        candidates = self._partition_candidates(self.site.scheduler.partition)
        if not candidates:
            return ""
        if len(candidates) == 1:
            return candidates[0]

        fallback = candidates[0]
        mixed_candidates: List[str] = []
        for partition in candidates:
            try:
                proc = await asyncio.create_subprocess_exec(
                    "sinfo", "-h", "-p", partition, "-o", "%t",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError:
                logger.warning(
                    "sinfo not available while selecting partition; using fallback %s",
                    fallback,
                )
                return fallback

            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    "Skipping partition %s during selection: %s",
                    partition,
                    stderr.decode().strip(),
                )
                continue

            states = [line.strip().lower() for line in stdout.decode().splitlines() if line.strip()]
            if any(state.startswith("idle") for state in states):
                logger.info("Selected idle Slurm partition %s from %s", partition, candidates)
                return partition
            if any(state.startswith("mix") or state.startswith("alloc") for state in states):
                mixed_candidates.append(partition)

        if mixed_candidates:
            logger.info(
                "Selected busy-but-runnable Slurm partition %s from %s",
                mixed_candidates[0],
                candidates,
            )
            return mixed_candidates[0]

        logger.warning("Falling back to first configured Slurm partition %s from %s", fallback, candidates)
        return fallback

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
        candidate_paths = (
            self._output_dir / f"slurm-{job_id}.out",
            self._output_dir / f"job-{job_id}.out",
        )
        for out_path in candidate_paths:
            if not out_path.exists():
                continue
            try:
                text = out_path.read_text(errors="replace")
            except OSError:
                continue

            for pattern in (
                r"Fully qualified name:\s+(\S+)",
                r"Running on node:\s+(\S+)",
                r"Node:\s+(\S+)",
            ):
                match = re.search(pattern, text)
                if match:
                    return match.group(1)
        return None

    @staticmethod
    def _map_slurm_state(state: str) -> JobStatus:
        """Map a Slurm state string to :class:`JobStatus`."""
        s = re.split(r"[ +]", state.upper(), maxsplit=1)[0] if state else ""
        mapping = {
            "PENDING": JobStatus.PENDING,
            "CONFIGURING": JobStatus.PENDING,
            "RUNNING": JobStatus.STARTING,  # may not be ready yet
            "COMPLETING": JobStatus.STOPPED,
            "COMPLETED": JobStatus.STOPPED,
            "FAILED": JobStatus.FAILED,
            "CANCELLED": JobStatus.STOPPED,
            "TIMEOUT": JobStatus.FAILED,
            "NODE_FAIL": JobStatus.FAILED,
            "PREEMPTED": JobStatus.FAILED,
        }
        return mapping.get(s, JobStatus.UNKNOWN)
