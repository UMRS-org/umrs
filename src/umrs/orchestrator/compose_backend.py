"""
Docker Compose orchestrator backend for development.

Provides a simple local development experience using Docker Compose.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import (
    EndpointInfo,
    JobInfo,
    JobStatus,
    OrchestratorBackend,
    OrchestratorConfig,
)


class DockerComposeBackend(OrchestratorBackend):
    """
    Docker Compose orchestrator for local development.

    Uses docker compose to manage Mini-RiVault containers.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        compose_dir: Optional[str] = None,
    ):
        super().__init__(config)
        self.compose_dir = compose_dir or os.environ.get(
            "MINI_RIVAULT_COMPOSE_DIR",
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "rivault"),
        )

    @property
    def backend_name(self) -> str:
        return "docker-compose"

    def _compose_cmd(self, *args: str, use_gpu: bool = True) -> List[str]:
        """Build docker compose command."""
        cfg = self.config
        cmd = ["docker", "compose"]

        # Add compose files
        compose_file = os.path.join(self.compose_dir, cfg.compose_file)
        cmd.extend(["-f", compose_file])

        if use_gpu:
            nvidia_file = os.path.join(self.compose_dir, cfg.compose_nvidia_file)
            if os.path.exists(nvidia_file):
                cmd.extend(["-f", nvidia_file])

        # Project name
        cmd.extend(["-p", cfg.compose_project_name])

        cmd.extend(args)
        return cmd

    async def _run_compose(self, *args: str, use_gpu: bool = True) -> tuple[int, str, str]:
        """Run docker compose command."""
        cmd = self._compose_cmd(*args, use_gpu=use_gpu)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.compose_dir,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout.decode(), stderr.decode()

    async def submit_job(
        self,
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> JobInfo:
        """Start Mini-RiVault via docker compose."""
        model = model_path or self.config.model_path
        if not model:
            raise ValueError("model_path is required (set VLLM_MODEL_PATH)")

        # Set environment for compose
        env = os.environ.copy()
        env["VLLM_MODEL_PATH"] = model
        env["LITELLM_MASTER_KEY"] = self.config.litellm_master_key or "sk-ll-dev"
        env["LITELLM_PORT"] = str(self.config.litellm_port)
        env["VLLM_PORT"] = str(self.config.vllm_port)
        env["VLLM_MAX_MODEL_LEN"] = str(self.config.vllm_max_model_len)
        env["VLLM_MAX_NUM_SEQS"] = str(self.config.vllm_max_num_seqs)

        use_gpu = kwargs.get("gpu", True)
        detach = kwargs.get("detach", True)

        cmd = self._compose_cmd("up", "-d" if detach else "", use_gpu=use_gpu)
        cmd = [c for c in cmd if c]  # Remove empty strings

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.compose_dir,
            env=env,
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"docker compose up failed: {stderr.decode()}")

        # Use project name as job ID
        job_id = self.config.compose_project_name

        return JobInfo(
            job_id=job_id,
            status=JobStatus.STARTING,
            model_path=model,
            created_at=datetime.now(),
            backend=self.backend_name,
        )

    async def get_job_status(self, job_id: str) -> JobInfo:
        """Get status of docker compose stack."""
        returncode, stdout, stderr = await self._run_compose("ps", "--format", "json")

        if returncode != 0:
            return JobInfo(
                job_id=job_id,
                status=JobStatus.UNKNOWN,
                backend=self.backend_name,
                error_message=stderr,
            )

        # Parse JSON output
        containers = []
        for line in stdout.strip().split("\n"):
            if line:
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        if not containers:
            return JobInfo(
                job_id=job_id,
                status=JobStatus.STOPPED,
                backend=self.backend_name,
            )

        # Check container states
        all_running = True
        any_failed = False

        for container in containers:
            state = container.get("State", "").lower()
            # name is available but not used currently
            # name = container.get("Name", "").lower()

            if state != "running":
                all_running = False
            if state in ("exited", "dead"):
                any_failed = True

        if any_failed:
            status = JobStatus.FAILED
        elif all_running:
            # Check if LiteLLM is ready
            endpoint = EndpointInfo(
                node="localhost",
                litellm_port=self.config.litellm_port,
                vllm_port=self.config.vllm_port,
            )
            if await self.check_readiness(endpoint):
                status = JobStatus.READY
            else:
                status = JobStatus.STARTING
        else:
            status = JobStatus.STARTING

        endpoint = None
        if status == JobStatus.READY:
            endpoint = EndpointInfo(
                node="localhost",
                litellm_port=self.config.litellm_port,
                vllm_port=self.config.vllm_port,
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
        """Wait for docker compose stack to be ready."""
        timeout = timeout_sec or self.config.startup_timeout_sec
        poll_interval = self.config.readiness_poll_interval_sec
        elapsed = 0

        while elapsed < timeout:
            job = await self.get_job_status(job_id)

            if job.status == JobStatus.FAILED:
                raise RuntimeError(f"Stack {job_id} failed")

            if job.is_ready() and job.endpoint:
                return job.endpoint

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Stack {job_id} did not become ready within {timeout}s")

    async def cleanup(self, job_id: str, force: bool = False) -> None:
        """Stop and remove docker compose stack."""
        args = ["down"]
        if force:
            args.extend(["--volumes", "--remove-orphans"])

        returncode, _, stderr = await self._run_compose(*args)

        if returncode != 0:
            raise RuntimeError(f"docker compose down failed: {stderr}")

    async def list_jobs(
        self,
        status_filter: Optional[List[JobStatus]] = None,
    ) -> List[JobInfo]:
        """List docker compose stacks (just the current one)."""
        job = await self.get_job_status(self.config.compose_project_name)

        if status_filter is None or job.status in status_filter:
            return [job]
        return []

    async def cleanup_orphans(self) -> int:
        """Clean up orphaned containers/volumes."""
        # Run docker compose down with orphan removal
        returncode, _, _ = await self._run_compose("down", "--remove-orphans")
        return 1 if returncode == 0 else 0

    async def health_check(self) -> Dict[str, Any]:
        """Check if Docker is accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            if proc.returncode == 0:
                return {
                    "backend": self.backend_name,
                    "healthy": True,
                    "message": "Docker daemon accessible",
                    "compose_dir": self.compose_dir,
                }
            else:
                return {
                    "backend": self.backend_name,
                    "healthy": False,
                    "message": "Docker daemon not running",
                }
        except FileNotFoundError:
            return {
                "backend": self.backend_name,
                "healthy": False,
                "message": "Docker not installed",
            }
