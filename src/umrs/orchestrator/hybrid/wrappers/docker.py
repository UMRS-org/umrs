"""DockerWrapper implementation (U-1113)."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..targets import AllocationInfo
from .base import ContainerEndpoint, LaunchError, LaunchSpec, MountSpec, RuntimeWrapper

logger = logging.getLogger(__name__)


class DockerWrapper(RuntimeWrapper):
    """
    Docker/Podman container runtime (U-1113).

    Used for development and testing environments.
    """

    def __init__(
        self,
        runtime: str = "docker",
        network: str = "",
        nvidia_support: bool = True,
    ):
        self.runtime = runtime or os.getenv("CONTAINER_RUNTIME", "docker")
        self.network = network or os.getenv("DOCKER_NETWORK", "")
        self.nvidia_support = nvidia_support

    @property
    def runtime_name(self) -> str:
        return "docker"

    async def launch(
        self,
        allocation: AllocationInfo,
        spec: LaunchSpec,
    ) -> ContainerEndpoint:
        """Launch Docker container."""
        if not spec.image:
            raise LaunchError("No Docker image specified")

        container_name = f"rivault-{allocation.job_id}"
        cmd = self._build_command(container_name, spec)

        logger.info(f"Launching Docker container: {container_name}")
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise LaunchError(f"Docker run failed: {stderr.decode()}")

            container_id = stdout.decode().strip()
            host_port = list(spec.ports.values())[0] if spec.ports else 11434

            return ContainerEndpoint(
                node=allocation.node or "localhost",
                port=host_port,
                container_id=container_id,
                launched_at=datetime.now(),
                metadata={
                    "container_name": container_name,
                    "image": spec.image,
                },
            )

        except Exception as e:
            raise LaunchError(f"Failed to launch Docker container: {e}")

    async def teardown(self, endpoint: ContainerEndpoint) -> bool:
        """Stop and remove Docker container."""
        container_id = endpoint.container_id
        if not container_id:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                self.runtime, "stop", "-t", "30", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            proc = await asyncio.create_subprocess_exec(
                self.runtime, "rm", "-f", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            return True

        except Exception as e:
            logger.warning(f"Failed to teardown Docker container: {e}")
            return False

    async def is_running(self, endpoint: ContainerEndpoint) -> bool:
        """Check if Docker container is running."""
        container_id = endpoint.container_id
        if not container_id:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                self.runtime, "inspect", "-f", "{{.State.Running}}", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip().lower() == "true"
        except Exception:
            return False

    def _build_command(self, name: str, spec: LaunchSpec) -> List[str]:
        """Build docker run command."""
        cmd = [self.runtime, "run", "-d", "--name", name]

        if self.nvidia_support and spec.gpus > 0:
            if self.runtime == "docker":
                cmd.extend(["--gpus", f"device={spec.gpus - 1}"])
            else:
                cmd.extend(["--device", "nvidia.com/gpu=all"])

        for key, value in spec.env.items():
            cmd.extend(["-e", f"{key}={value}"])

        for container_port, host_port in spec.ports.items():
            cmd.extend(["-p", f"{host_port}:{container_port}"])

        for mount in spec.mounts:
            if mount.mount_type == "bind":
                vol = f"{mount.source}:{mount.target}"
                if mount.read_only:
                    vol += ":ro"
                cmd.extend(["-v", vol])
            elif mount.mount_type == "volume":
                cmd.extend(["-v", f"{mount.source}:{mount.target}"])

        if self.network:
            cmd.extend(["--network", self.network])

        for key, value in spec.labels.items():
            cmd.extend(["--label", f"{key}={value}"])

        if spec.memory_mb > 0:
            cmd.extend(["--memory", f"{spec.memory_mb}m"])
        if spec.cpus > 0:
            cmd.extend(["--cpus", str(spec.cpus)])

        cmd.append(spec.image)

        if spec.command:
            cmd.extend(spec.command)
        if spec.args:
            cmd.extend(spec.args)

        return cmd
