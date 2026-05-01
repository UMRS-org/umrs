"""SingularityWrapper implementation (U-1112)."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import datetime
from typing import List, Optional

from ..targets import AllocationInfo
from .base import ContainerEndpoint, LaunchError, LaunchSpec, RuntimeWrapper

logger = logging.getLogger(__name__)


class SingularityWrapper(RuntimeWrapper):
    """
    Singularity/Apptainer container runtime (U-1112).

    Default runtime for HPC environments.
    """

    def __init__(
        self,
        default_image: str = "",
        bind_paths: Optional[List[str]] = None,
        nvidia_support: bool = True,
    ):
        self.default_image = default_image or os.getenv(
            "SINGULARITY_IMAGE", "/opt/singularity/mini-rivault.sif"
        )
        self.bind_paths = bind_paths or []
        self.nvidia_support = nvidia_support

    @property
    def runtime_name(self) -> str:
        return "singularity"

    async def launch(
        self,
        allocation: AllocationInfo,
        spec: LaunchSpec,
    ) -> ContainerEndpoint:
        """Launch Singularity container."""
        if not allocation.is_allocated:
            raise LaunchError(f"Cannot launch: allocation not ready ({allocation.status})")

        image = spec.image or self.default_image
        if not image:
            raise LaunchError("No Singularity image specified")
        if self._requires_slurm_launch(allocation) and not allocation.node:
            raise LaunchError("Slurm allocation has no node for remote launch")

        cmd = self._build_command(allocation, spec, image)

        logger.info(f"Launching Singularity container on {allocation.node}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            host_port = list(spec.ports.values())[0] if spec.ports else 11434

            return ContainerEndpoint(
                node=allocation.node,
                port=host_port,
                container_id=f"singularity-{allocation.job_id}",
                launched_at=datetime.now(),
                metadata={
                    "pid": proc.pid,
                    "image": image,
                    "job_id": allocation.job_id,
                    "remote_launch": self._requires_slurm_launch(allocation),
                },
            )

        except Exception as e:
            raise LaunchError(f"Failed to launch Singularity container: {e}")

    async def teardown(self, endpoint: ContainerEndpoint) -> bool:
        """Stop Singularity container."""
        pid = endpoint.metadata.get("pid")
        if not pid:
            logger.warning("No PID in endpoint metadata, cannot teardown")
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            await asyncio.sleep(5)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return True
        except Exception as e:
            logger.warning(f"Failed to teardown Singularity container: {e}")
            return False

    async def is_running(self, endpoint: ContainerEndpoint) -> bool:
        """Check if Singularity process is running."""
        if endpoint.metadata.get("remote_launch"):
            return await self._is_slurm_remote_running(endpoint)

        pid = endpoint.metadata.get("pid")
        if not pid:
            return False

        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    async def _is_slurm_remote_running(self, endpoint: ContainerEndpoint) -> bool:
        """Probe the allocated Slurm node for the Singularity process."""
        job_id = endpoint.metadata.get("job_id")
        image = endpoint.metadata.get("image")
        if not job_id or not endpoint.node or not image:
            return False

        proc = await asyncio.create_subprocess_exec(
            "srun",
            f"--jobid={job_id}",
            "--nodes=1",
            "--ntasks=1",
            "--overlap",
            "--nodelist",
            endpoint.node,
            "pgrep",
            "-f",
            image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0

    def _build_command(
        self,
        allocation: AllocationInfo,
        spec: LaunchSpec,
        image: str,
    ) -> List[str]:
        """Build singularity exec command."""
        cmd = ["singularity", "exec"]

        if self.nvidia_support:
            cmd.append("--nv")

        for key, value in spec.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        for mount in spec.mounts:
            bind = f"{mount.source}:{mount.target}"
            if mount.read_only:
                bind += ":ro"
            cmd.extend(["--bind", bind])

        for bind in self.bind_paths:
            cmd.extend(["--bind", bind])

        cmd.append("--network=host")
        cmd.append(image)

        if spec.command:
            cmd.extend(spec.command)
        if spec.args:
            cmd.extend(spec.args)

        return self._with_slurm_launch(allocation, cmd)

    @staticmethod
    def _requires_slurm_launch(allocation: AllocationInfo) -> bool:
        """Return true when the allocation came from Slurm."""
        return allocation.metadata.get("execution_authority") == "slurm"

    def _with_slurm_launch(
        self,
        allocation: AllocationInfo,
        cmd: List[str],
    ) -> List[str]:
        """Wrap a local command in `srun` when launched from Slurm."""
        if not self._requires_slurm_launch(allocation):
            return cmd
        return [
            "srun",
            f"--jobid={allocation.job_id}",
            "--nodes=1",
            "--ntasks=1",
            "--overlap",
            "--nodelist",
            allocation.node,
            *cmd,
        ]
