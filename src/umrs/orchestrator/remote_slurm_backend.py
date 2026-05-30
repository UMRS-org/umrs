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

import asyncio
import logging
import os
import re
import shlex
from datetime import datetime
from typing import Any, Optional

from .base import (
    EndpointInfo,
    JobInfo,
    JobStatus,
)
from .slurm_backend import SlurmBackend
from .ssh_control import SSHControlConfig, SSHControlManager
from .tunnel_pool import TunnelPool, TunnelPoolConfig

logger = logging.getLogger(__name__)


class RemoteSlurmBackend(SlurmBackend):
    """
    SSH-based remote Slurm backend for front-end server deployments.

    All Slurm CLI commands are executed remotely via SSH ControlMaster.
    Job info files on the shared filesystem are read via SSH cat.

    When ``slurm_delegation`` is ``per_user_ssh``, job submission and cancellation
    are executed as the requesting user via a per-user SSH connection using the
    UMRS service identity key. The user must have enrolled the service public key
    in their ~/.ssh/authorized_keys (via ``r4 onboard``).
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
        self._tunnel_pool: TunnelPool | None = None

        # Per-user SSH delegation state
        self._delegation_mode: str = "service_account"
        self._service_identity_file: str = ""
        self._user_ssh_pool: dict[str, SSHControlManager] = {}
        try:
            from .site_config import load_site_config
            site = load_site_config()
            self._delegation_mode = site.scheduler.slurm_delegation
            # Derive private key path from pubkey path (strip .pub suffix)
            pubkey_path = site.scheduler.service_pubkey_path
            if pubkey_path.endswith(".pub"):
                self._service_identity_file = pubkey_path[:-4]
        except Exception:
            pass

    @property
    def backend_name(self) -> str:
        return "slurm-remote"

    async def _get_ssh(self) -> SSHControlManager:
        """Get or create the SSH control manager."""
        if self._ssh is None:
            self._ssh = SSHControlManager(self._ssh_config)
        await self._ssh.ensure_connected()
        return self._ssh

    async def _get_ssh_for_user(self, username: str) -> SSHControlManager:
        """Get or create a per-user SSH control manager for delegated submission.

        Uses the UMRS service identity key to connect as the target user.
        The user must have enrolled the service pubkey in their authorized_keys.
        Falls back to the service-account connection if delegation is not
        configured **or** if the per-user connection fails (user has not
        enrolled yet).
        """
        if self._delegation_mode != "per_user_ssh" or not self._service_identity_file:
            return await self._get_ssh()

        if username not in self._user_ssh_pool:
            # Build a user-specific SSH config based on service config
            user_config = SSHControlConfig(
                ssh_target=self._ssh_config.ssh_target,
                ssh_user=username,
                ssh_port=self._ssh_config.ssh_port,
                connect_timeout=self._ssh_config.connect_timeout,
                command_timeout=self._ssh_config.command_timeout,
                control_path="",  # Auto-generated per-user socket
                server_alive_interval=self._ssh_config.server_alive_interval,
                server_alive_count_max=self._ssh_config.server_alive_count_max,
                reverse_tunnels=[],  # No tunnels for per-user connections
                proxy_jump=self._ssh_config.proxy_jump,
                identity_file=self._service_identity_file,
                remote_env=self._ssh_config.remote_env,
            )
            self._user_ssh_pool[username] = SSHControlManager(user_config)

        mgr = self._user_ssh_pool[username]
        try:
            await mgr.ensure_connected()
        except Exception:
            # Per-user SSH failed (user likely hasn't enrolled their key).
            # Fall back to service account so the job can still proceed
            # under the site master Slurm account.
            logger.warning(
                "Per-user SSH connection failed for %s; falling back to "
                "service account.  The user may not have enrolled the UMRS "
                "service key (r4 auth login / r4 onboard).",
                username,
            )
            return await self._get_ssh()
        return mgr

    def _get_wrapper_script(self) -> str | None:
        """Get the remote wrapper script path from site config extra."""
        extra = getattr(self.config, "extra", {}) or {}
        if not extra.get("slurm_wrapper_script"):
            try:
                from .site_config import load_site_config
                site = load_site_config()
                extra = site.extra or {}
            except Exception:
                pass
        return extra.get("slurm_wrapper_script")

    def _job_info_dir(self) -> str:
        """Return the remote shared directory for Slurm job info files."""
        info_dir = getattr(self.config, "slurm_job_info_dir", "/tmp")
        extra = getattr(self.config, "extra", {}) or {}
        if extra.get("slurm_job_info_dir"):
            return str(extra["slurm_job_info_dir"])
        try:
            from .site_config import load_site_config
            site = load_site_config()
            return str(site.extra.get("slurm_job_info_dir", info_dir))
        except Exception:
            return str(info_dir)

    def _control_dir(self) -> str:
        """Return the remote shared directory for graceful stop requests."""
        return f"{self._job_info_dir().rstrip('/')}/control"

    def _stop_request_path(self, job_id: str) -> str:
        """Return the remote shared stop request path for a Slurm job."""
        return f"{self._control_dir()}/stop-{job_id}.request"

    async def submit_job(
        self,
        model_path: str | None = None,
        **kwargs: Any,
    ) -> JobInfo:
        """Submit a Slurm job via remote SSH sbatch.

        When per-user delegation is active and ``requesting_user`` is provided,
        the sbatch is executed as that user via the UMRS service identity key.
        """
        model = model_path or self.config.model_path
        if not model:
            raise ValueError("model_path is required (set VLLM_MODEL_PATH or pass explicitly)")

        # Determine which SSH connection to use for submission
        requesting_user = kwargs.pop("requesting_user", None)

        wrapper_script = self._get_wrapper_script()
        if wrapper_script:
            return await self._submit_wrapper_job(model, wrapper_script, requesting_user=requesting_user, **kwargs)

        # Legacy: generate Singularity-based sbatch script and pipe via stdin.
        script_content = self._build_sbatch_script(model, **kwargs)

        if requesting_user and self._delegation_mode == "per_user_ssh":
            ssh = await self._get_ssh_for_user(requesting_user)
        else:
            ssh = await self._get_ssh()
        result = await ssh.run_remote_with_stdin(
            "sbatch --parsable",
            script_content.encode(),
            timeout=60,
        )

        if not result.ok:
            raise RuntimeError(f"Remote sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split(";")[0]

        return JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_path=model,
            created_at=datetime.now(),
            backend=self.backend_name,
        )

    async def _submit_wrapper_job(
        self,
        model_path: str,
        wrapper_script: str,
        requesting_user: str | None = None,
        **kwargs: Any,
    ) -> JobInfo:
        """Submit a site-provided wrapper script (e.g. serve-model.slurm) via remote sbatch."""
        # Build sbatch CLI args from site config.
        from .site_config import load_site_config
        site = load_site_config()

        sbatch_args = ["sbatch", "--parsable"]

        # Output/error dirs
        output_dir = self._job_info_dir()
        sbatch_args.append(f"--output={output_dir}/slurm-%j.out")
        sbatch_args.append(f"--error={output_dir}/slurm-%j.err")

        # Partition
        partition = kwargs.pop("partition_override", None) or site.scheduler.partition.split(",")[0]
        if partition:
            sbatch_args.append(f"--partition={partition}")

        # Time limit
        if site.scheduler.time_limit:
            sbatch_args.append(f"--time={site.scheduler.time_limit}")

        # GPUs (only if site allows GRES tracking — gpus=0 means no GRES)
        gpu_count = kwargs.pop("gpu_count", None)
        if site.scheduler.gpus != 0:
            effective_gpus = gpu_count if gpu_count is not None else site.scheduler.gpus
            if effective_gpus:
                sbatch_args.append(f"--gres=gpu:{effective_gpus}")

        # CPUs
        if site.scheduler.cpus:
            sbatch_args.append(f"--cpus-per-task={site.scheduler.cpus}")

        # Memory (explicit value or --mem=0 to suppress DefMemPerCPU default)
        if site.scheduler.memory:
            sbatch_args.append(f"--mem={site.scheduler.memory}")
        else:
            sbatch_args.append("--mem=0")

        # Slurm account: per-user override > site config default
        slurm_account = kwargs.pop("slurm_account", None) or site.scheduler.account
        if slurm_account:
            sbatch_args.append(f"--account={slurm_account}")

        # QoS
        if site.scheduler.qos:
            sbatch_args.append(f"--qos={site.scheduler.qos}")

        # Nodelist hint
        nodelist_hint = kwargs.pop("nodelist_hint", None)
        if nodelist_hint:
            sbatch_args.append(f"--nodelist={nodelist_hint}")

        # Node count
        node_count = kwargs.pop("node_count", None)
        if node_count and node_count > 1:
            sbatch_args.append(f"--nodes={node_count}")

        # Extra sbatch args from site config
        for extra in getattr(site.scheduler, "extra_sbatch", []):
            sbatch_args.append(extra)

        # Pop unused kwargs before building env
        kwargs.pop("min_gpu_memory_gb", None)

        # Build environment exports for the job
        env_overrides: dict[str, str] = {}
        requested_env = kwargs.pop("env", None)
        if isinstance(requested_env, dict):
            env_overrides.update(
                {k: str(v) for k, v in requested_env.items() if isinstance(v, (str, int, float))}
            )
        env_overrides["VLLM_MODEL_PATH"] = model_path
        env_overrides.setdefault("VLLM_SERVED_MODEL_NAME", env_overrides.get("MODEL_NAME", model_path))
        env_overrides.setdefault("RIVAULT_WRAPPER_DIR", os.path.dirname(wrapper_script))
        env_overrides.setdefault("JOB_INFO_DIR", output_dir)
        env_overrides.setdefault("RIVAULT_CONTROL_DIR", self._control_dir())

        # Build --export arg: ALL plus per-job overrides
        export_parts = ["ALL"]
        for k, v in env_overrides.items():
            export_parts.append(f"{k}={v}")
        sbatch_args.append(f"--export={','.join(export_parts)}")

        # Append the wrapper script path
        sbatch_args.append(wrapper_script)

        cmd = " ".join(sbatch_args)
        logger.info("Submitting wrapper job: %s", cmd)

        if requesting_user and self._delegation_mode == "per_user_ssh":
            ssh = await self._get_ssh_for_user(requesting_user)
        else:
            ssh = await self._get_ssh()
        result = await ssh.run_remote(cmd, timeout=60)

        if not result.ok:
            raise RuntimeError(f"Remote sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split(";")[0]
        logger.info("Submitted wrapper job %s", job_id)

        return JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_path=model_path,
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
            if endpoint is not None:
                endpoint = await self._endpoint_via_tunnel(endpoint)
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

    async def _endpoint_via_tunnel(self, endpoint: EndpointInfo) -> EndpointInfo:
        """Return a front-end local endpoint backed by an SSH forward tunnel."""
        local_port = await self._get_tunnel_pool().open_tunnel(
            endpoint.node,
            endpoint.litellm_port,
            local_port=endpoint.litellm_port,
        )
        return EndpointInfo(
            node="127.0.0.1",
            litellm_port=local_port,
            vllm_port=endpoint.vllm_port,
        )

    def _get_tunnel_pool(self) -> TunnelPool:
        if self._tunnel_pool is None:
            local_port_min = 12000
            local_port_max = 12999
            try:
                from .site_config import load_site_config
                site = load_site_config()
                local_port_min = site.network.port_range_min
                local_port_max = site.network.port_range_max
            except Exception:
                pass

            self._tunnel_pool = TunnelPool(
                TunnelPoolConfig(
                    ssh_target=self._ssh_config.ssh_target,
                    ssh_user=self._ssh_config.ssh_user,
                    ssh_port=self._ssh_config.ssh_port,
                    local_port_min=local_port_min,
                    local_port_max=local_port_max,
                    control_path=self._ssh_config.control_path,
                )
            )
        return self._tunnel_pool

    async def _read_remote_job_info(self, job_id: str) -> Optional[str]:
        """Read job info file from remote filesystem via SSH."""
        info_dir = self._job_info_dir()
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
        info_content = await self._read_remote_job_info(job_id)
        if info_content is not None and self._tunnel_pool is not None:
            info = self._parse_info_content(info_content)
            endpoint = self._endpoint_from_info(JobStatus.RUNNING, info)
            if endpoint is not None:
                await self._tunnel_pool.close_tunnel(endpoint.node, endpoint.litellm_port)

        ssh = await self._get_ssh()
        info_dir = self._job_info_dir()
        info_path = f"{info_dir}/rivault-job-{job_id}.info"

        if not force:
            stop_request = self._stop_request_path(job_id)
            control_dir = self._control_dir()
            result = await ssh.run_remote(
                "mkdir -p {control_dir} && touch {stop_request}".format(
                    control_dir=shlex.quote(control_dir),
                    stop_request=shlex.quote(stop_request),
                )
            )
            if result.ok:
                graceful_timeout = int(os.environ.get("RIVAULT_GRACEFUL_STOP_TIMEOUT", "20"))
                logger.info(
                    "requested graceful remote compute-node stop for %s via %s",
                    job_id,
                    stop_request,
                )
                if await self._wait_for_remote_job_exit(job_id, graceful_timeout):
                    await ssh.run_remote(f"rm -f {shlex.quote(info_path)}")
                    return
            else:
                logger.warning(
                    "failed to create remote graceful stop request for %s: %s",
                    job_id,
                    result.stderr.strip(),
                )

        cmd = f"scancel -Q {shlex.quote(job_id)}"
        if force:
            cmd = f"scancel -Q --full --signal=KILL {shlex.quote(job_id)}"
        result = await ssh.run_remote(cmd)
        if not result.ok:
            logger.warning(
                "remote scancel failed for %s (force=%s): %s",
                job_id,
                force,
                result.stderr.strip(),
            )

        # Clean up remote info file.
        await ssh.run_remote(f"rm -f {shlex.quote(info_path)}")

    async def _wait_for_remote_job_exit(self, job_id: str, timeout_sec: int) -> bool:
        """Wait briefly for a remote Slurm job to leave the active queue."""
        for _ in range(max(timeout_sec, 0)):
            status = await self._query_slurm_job_status(job_id)
            if status in (JobStatus.STOPPED, JobStatus.FAILED, JobStatus.UNKNOWN):
                return True
            await asyncio.sleep(1)
        return False

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
        info_dir = self._job_info_dir()
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
        info_dir = self._job_info_dir()
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
        if self._tunnel_pool is not None:
            await self._tunnel_pool.shutdown()
            self._tunnel_pool = None
        # Close per-user SSH connections
        for username, mgr in list(self._user_ssh_pool.items()):
            await mgr.close()
        self._user_ssh_pool.clear()
        if self._ssh:
            await self._ssh.close()
            self._ssh = None
