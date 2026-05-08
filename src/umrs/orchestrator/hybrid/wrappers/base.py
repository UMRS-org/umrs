"""
RuntimeWrapper base classes and shared types (U-1111).

MountSpec, LaunchSpec, ContainerEndpoint, RuntimeWrapper ABC, LaunchError.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..targets import AllocationInfo

logger = logging.getLogger(__name__)


@dataclass
class MountSpec:
    """
    Container mount specification.

    Abstracts different mount types across container runtimes.
    """

    source: str           # Host path, PVC name, or volume name
    target: str           # Container path
    mount_type: str = "bind"  # "bind", "volume", "pvc", "tmpfs"
    read_only: bool = False
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class LaunchSpec:
    """
    Container launch specification.

    This is passed to RuntimeWrapper.launch() along with AllocationInfo.
    Contains everything needed to start the inference container.
    """

    # Container
    image: str
    command: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)

    # Model
    model_path: str = ""  # Resolved path inside container

    # Environment
    env: dict[str, str] = field(default_factory=dict)

    # Mounts
    mounts: list[MountSpec] = field(default_factory=list)

    # Networking
    ports: dict[int, int] = field(default_factory=dict)  # container:host

    # Resources (for runtimes that need explicit limits)
    gpus: int = 1
    cpus: int = 8
    memory_mb: int = 65536

    # Labels/annotations
    labels: dict[str, str] = field(default_factory=dict)

    # Extra options
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerEndpoint:
    """
    Information about a launched container endpoint.

    Returned by RuntimeWrapper.launch() to indicate where the
    inference service is accessible.
    """

    node: str
    port: int
    container_id: str = ""
    model_name: str = "rivault-gpt-oss-static"
    launched_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def litellm_url(self) -> str:
        """OpenAI-compatible API URL."""
        return f"http://{self.node}:{self.port}/v1"

    @property
    def health_url(self) -> str:
        """Health check URL."""
        return f"http://{self.node}:{self.port}/health"

    @property
    def models_url(self) -> str:
        """Models list URL (used for readiness check)."""
        return f"http://{self.node}:{self.port}/v1/models"


class RuntimeWrapper(ABC):
    """
    Abstract base for container launch (U-1111).

    Responsibilities:
    - Launch container inside allocated resources
    - Teardown container
    - Check container health

    Does NOT:
    - Allocate resources (ExecutionTarget's job)
    - Resolve model paths (ModelRefResolver's job)
    """

    @property
    @abstractmethod
    def runtime_name(self) -> str:
        """Return runtime name ('singularity', 'docker', 'k8s')."""
        ...

    @abstractmethod
    async def launch(
        self,
        allocation: AllocationInfo,
        spec: LaunchSpec,
    ) -> ContainerEndpoint:
        """
        Launch container inside allocated resources.

        Args:
            allocation: Resource allocation from ExecutionTarget.
            spec: Container launch specification.

        Returns:
            ContainerEndpoint with access coordinates.

        Raises:
            LaunchError: If container cannot be started.
        """
        ...

    @abstractmethod
    async def teardown(self, endpoint: ContainerEndpoint) -> bool:
        """
        Stop and remove container.

        Args:
            endpoint: Endpoint info from launch().

        Returns:
            True if teardown was successful.
        """
        ...

    @abstractmethod
    async def is_running(self, endpoint: ContainerEndpoint) -> bool:
        """
        Check if container is still running.

        Args:
            endpoint: Endpoint info from launch().

        Returns:
            True if container is running.
        """
        ...

    async def wait_for_ready(
        self,
        endpoint: ContainerEndpoint,
        timeout_sec: int = 300,
        poll_interval_sec: int = 5,
        auth_key: str = "",
    ) -> bool:
        """
        Wait until container is ready to serve requests.

        Default implementation polls /v1/models endpoint.

        Args:
            endpoint: Endpoint info from launch().
            timeout_sec: Maximum wait time.
            poll_interval_sec: Seconds between health checks.
            auth_key: LiteLLM master key for auth.

        Returns:
            True if ready within timeout.

        Raises:
            TimeoutError: If not ready within timeout.
        """
        import aiohttp

        start = datetime.now()
        headers = {}
        if auth_key:
            headers["Authorization"] = f"Bearer {auth_key}"

        while True:
            try:
                async with aiohttp.ClientSession() as session, session.get(
                    endpoint.models_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass

            elapsed = (datetime.now() - start).total_seconds()
            if elapsed > timeout_sec:
                raise TimeoutError(
                    f"Endpoint {endpoint.litellm_url} not ready after {timeout_sec}s"
                )

            await asyncio.sleep(poll_interval_sec)


class LaunchError(Exception):
    """Error during container launch."""
    pass
