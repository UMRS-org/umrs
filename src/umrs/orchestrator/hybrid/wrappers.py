"""
RuntimeWrapper Interface and Implementations (U-1111 to U-1115).

RuntimeWrapper handles container launch inside an allocated resource slot.
It receives AllocationInfo from ExecutionTarget and launches the actual
inference container.

Implementations:
- SingularityWrapper: Singularity/Apptainer containers (HPC default)
- DockerWrapper: Docker/Podman containers (development)
- K8sJobWrapper: Native K8s container spec update
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .targets import AllocationInfo, AllocationStatus

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
    options: Dict[str, str] = field(default_factory=dict)


@dataclass
class LaunchSpec:
    """
    Container launch specification.

    This is passed to RuntimeWrapper.launch() along with AllocationInfo.
    Contains everything needed to start the inference container.
    """

    # Container
    image: str
    command: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)

    # Model
    model_path: str = ""  # Resolved path inside container

    # Environment
    env: Dict[str, str] = field(default_factory=dict)

    # Mounts
    mounts: List[MountSpec] = field(default_factory=list)

    # Networking
    ports: Dict[int, int] = field(default_factory=dict)  # container:host

    # Resources (for runtimes that need explicit limits)
    gpus: int = 1
    cpus: int = 8
    memory_mb: int = 65536

    # Labels/annotations
    labels: Dict[str, str] = field(default_factory=dict)

    # Extra options
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EndpointInfo:
    """
    Information about a launched container endpoint.

    Returned by RuntimeWrapper.launch() to indicate where the
    inference service is accessible.
    """

    node: str
    port: int
    container_id: str = ""
    model_name: str = "rivault-gpt-oss-static"
    launched_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

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
    ) -> EndpointInfo:
        """
        Launch container inside allocated resources.

        Args:
            allocation: Resource allocation from ExecutionTarget.
            spec: Container launch specification.

        Returns:
            EndpointInfo with access coordinates.

        Raises:
            LaunchError: If container cannot be started.
        """
        ...

    @abstractmethod
    async def teardown(self, endpoint: EndpointInfo) -> bool:
        """
        Stop and remove container.

        Args:
            endpoint: Endpoint info from launch().

        Returns:
            True if teardown was successful.
        """
        ...

    @abstractmethod
    async def is_running(self, endpoint: EndpointInfo) -> bool:
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
        endpoint: EndpointInfo,
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
                async with aiohttp.ClientSession() as session:
                    async with session.get(
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


# =============================================================================
# SingularityWrapper Implementation (U-1112)
# =============================================================================


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
        """
        Initialize Singularity wrapper.

        Args:
            default_image: Default SIF image path.
            bind_paths: Default bind mounts.
            nvidia_support: Enable NVIDIA GPU support.
        """
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
    ) -> EndpointInfo:
        """Launch Singularity container."""
        if not allocation.is_allocated:
            raise LaunchError(f"Cannot launch: allocation not ready ({allocation.status})")

        image = spec.image or self.default_image
        if not image:
            raise LaunchError("No Singularity image specified")

        # Build singularity exec command
        cmd = self._build_command(allocation, spec, image)

        logger.info(f"Launching Singularity container on {allocation.node}")
        logger.debug(f"Command: {' '.join(cmd)}")

        # For Slurm, we run inside the job allocation
        # The command is executed in the already-allocated context
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Don't wait - container runs in background
            # Return immediately with endpoint info

            # Determine port
            host_port = list(spec.ports.values())[0] if spec.ports else 11434

            return EndpointInfo(
                node=allocation.node,
                port=host_port,
                container_id=f"singularity-{allocation.job_id}",
                launched_at=datetime.now(),
                metadata={
                    "pid": proc.pid,
                    "image": image,
                    "job_id": allocation.job_id,
                },
            )

        except Exception as e:
            raise LaunchError(f"Failed to launch Singularity container: {e}")

    async def teardown(self, endpoint: EndpointInfo) -> bool:
        """Stop Singularity container."""
        pid = endpoint.metadata.get("pid")
        if not pid:
            logger.warning("No PID in endpoint metadata, cannot teardown")
            return False

        try:
            import signal
            os.kill(pid, signal.SIGTERM)

            # Wait a bit for graceful shutdown
            await asyncio.sleep(5)

            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Already dead

            return True

        except Exception as e:
            logger.warning(f"Failed to teardown Singularity container: {e}")
            return False

    async def is_running(self, endpoint: EndpointInfo) -> bool:
        """Check if Singularity process is running."""
        pid = endpoint.metadata.get("pid")
        if not pid:
            return False

        try:
            os.kill(pid, 0)  # Signal 0 = check existence
            return True
        except ProcessLookupError:
            return False

    def _build_command(
        self,
        allocation: AllocationInfo,
        spec: LaunchSpec,
        image: str,
    ) -> List[str]:
        """Build singularity exec command."""
        cmd = ["singularity", "exec"]

        # NVIDIA GPU support
        if self.nvidia_support:
            cmd.append("--nv")

        # Environment variables
        for key, value in spec.env.items():
            cmd.extend(["--env", f"{key}={value}"])

        # Bind mounts
        for mount in spec.mounts:
            bind = f"{mount.source}:{mount.target}"
            if mount.read_only:
                bind += ":ro"
            cmd.extend(["--bind", bind])

        # Default bind paths
        for bind in self.bind_paths:
            cmd.extend(["--bind", bind])

        # Network (host networking for simplicity)
        cmd.append("--network=host")

        # Image
        cmd.append(image)

        # Command and args
        if spec.command:
            cmd.extend(spec.command)
        if spec.args:
            cmd.extend(spec.args)

        return cmd


# =============================================================================
# DockerWrapper Implementation (U-1113)
# =============================================================================


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
        """
        Initialize Docker wrapper.

        Args:
            runtime: Container runtime ("docker" or "podman").
            network: Docker network to use.
            nvidia_support: Enable NVIDIA GPU support.
        """
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
    ) -> EndpointInfo:
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

            return EndpointInfo(
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

    async def teardown(self, endpoint: EndpointInfo) -> bool:
        """Stop and remove Docker container."""
        container_id = endpoint.container_id
        if not container_id:
            return False

        try:
            # Stop container
            proc = await asyncio.create_subprocess_exec(
                self.runtime, "stop", "-t", "30", container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()

            # Remove container
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

    async def is_running(self, endpoint: EndpointInfo) -> bool:
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

        # GPU support
        if self.nvidia_support and spec.gpus > 0:
            if self.runtime == "docker":
                cmd.extend(["--gpus", f"device={spec.gpus - 1}"])
            else:  # podman
                cmd.extend(["--device", "nvidia.com/gpu=all"])

        # Environment
        for key, value in spec.env.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Ports
        for container_port, host_port in spec.ports.items():
            cmd.extend(["-p", f"{host_port}:{container_port}"])

        # Mounts
        for mount in spec.mounts:
            if mount.mount_type == "bind":
                vol = f"{mount.source}:{mount.target}"
                if mount.read_only:
                    vol += ":ro"
                cmd.extend(["-v", vol])
            elif mount.mount_type == "volume":
                cmd.extend(["-v", f"{mount.source}:{mount.target}"])

        # Network
        if self.network:
            cmd.extend(["--network", self.network])

        # Labels
        for key, value in spec.labels.items():
            cmd.extend(["--label", f"{key}={value}"])

        # Resource limits
        if spec.memory_mb > 0:
            cmd.extend(["--memory", f"{spec.memory_mb}m"])
        if spec.cpus > 0:
            cmd.extend(["--cpus", str(spec.cpus)])

        # Image
        cmd.append(spec.image)

        # Command and args
        if spec.command:
            cmd.extend(spec.command)
        if spec.args:
            cmd.extend(spec.args)

        return cmd


# =============================================================================
# K8sJobWrapper Implementation (U-1114)
# =============================================================================


class K8sJobWrapper(RuntimeWrapper):
    """
    Native Kubernetes container wrapper (U-1114).

    Updates existing K8s Job/Pod spec with container details.
    """

    def __init__(
        self,
        namespace: str = "rivault-system",
        kubeconfig: Optional[str] = None,
    ):
        """
        Initialize K8s wrapper.

        Args:
            namespace: Kubernetes namespace.
            kubeconfig: Path to kubeconfig file.
        """
        self.namespace = namespace or os.getenv("K8S_NAMESPACE", "rivault-system")
        self.kubeconfig = kubeconfig or os.getenv("KUBECONFIG")

    @property
    def runtime_name(self) -> str:
        return "k8s"

    async def launch(
        self,
        allocation: AllocationInfo,
        spec: LaunchSpec,
    ) -> EndpointInfo:
        """Update K8s Job with container spec and create Service."""
        if not spec.image:
            raise LaunchError("No container image specified")

        job_name = allocation.job_id

        # Build patch for the Job
        patch = self._build_job_patch(spec)

        try:
            # Patch the Job to update container
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "patch", "job", job_name,
                "--namespace", self.namespace,
                "--type", "strategic",
                "-p", json.dumps(patch),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                # Job might not support patching, try recreation
                logger.warning(f"Job patch failed: {stderr.decode()}")

            # Create Service for endpoint access
            service_name = f"{job_name}-svc"
            host_port = list(spec.ports.values())[0] if spec.ports else 11434

            service = self._build_service(service_name, job_name, host_port)

            proc = await asyncio.create_subprocess_exec(
                "kubectl", "apply", "-f", "-",
                "--namespace", self.namespace,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate(json.dumps(service).encode())

            # Get service cluster IP or node port
            node = await self._get_service_endpoint(service_name)

            return EndpointInfo(
                node=node or allocation.node,
                port=host_port,
                container_id=job_name,
                launched_at=datetime.now(),
                metadata={
                    "service_name": service_name,
                    "image": spec.image,
                },
            )

        except Exception as e:
            raise LaunchError(f"Failed to update K8s Job: {e}")

    async def teardown(self, endpoint: EndpointInfo) -> bool:
        """Delete K8s Service (Job cleanup is separate)."""
        service_name = endpoint.metadata.get("service_name")
        if not service_name:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "delete", "service", service_name,
                "--namespace", self.namespace,
                "--ignore-not-found",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to delete K8s service: {e}")
            return False

    async def is_running(self, endpoint: EndpointInfo) -> bool:
        """Check if K8s pod is running."""
        job_name = endpoint.container_id
        if not job_name:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "get", "pods",
                "--namespace", self.namespace,
                "--selector", f"job-name={job_name}",
                "-o", "jsonpath={.items[0].status.phase}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip() == "Running"
        except Exception:
            return False

    def _build_job_patch(self, spec: LaunchSpec) -> Dict[str, Any]:
        """Build strategic merge patch for Job."""
        container_spec = {
            "name": "rivault",
            "image": spec.image,
            "env": [
                {"name": k, "value": v}
                for k, v in spec.env.items()
            ],
            "ports": [
                {"containerPort": port}
                for port in spec.ports.keys()
            ],
        }

        if spec.command:
            container_spec["command"] = spec.command
        if spec.args:
            container_spec["args"] = spec.args

        # Volume mounts
        if spec.mounts:
            container_spec["volumeMounts"] = []
            for mount in spec.mounts:
                container_spec["volumeMounts"].append({
                    "name": mount.source.replace("/", "-").strip("-"),
                    "mountPath": mount.target,
                    "readOnly": mount.read_only,
                })

        return {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [container_spec],
                    },
                },
            },
        }

    def _build_service(
        self,
        name: str,
        job_name: str,
        port: int,
    ) -> Dict[str, Any]:
        """Build K8s Service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": name,
                "namespace": self.namespace,
            },
            "spec": {
                "selector": {"job-name": job_name},
                "ports": [{
                    "port": port,
                    "targetPort": port,
                }],
                "type": "ClusterIP",
            },
        }

    async def _get_service_endpoint(self, service_name: str) -> str:
        """Get service cluster IP."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "get", "service", service_name,
                "--namespace", self.namespace,
                "-o", "jsonpath={.spec.clusterIP}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip()
        except Exception:
            return ""


# =============================================================================
# U-1115: K8sBootstrapWrapper (Experimental K8s-on-Slurm)
# =============================================================================


class K8sOnSlurmNotEnabledError(Exception):
    """Raised when K8s-on-Slurm is used without the feature flag."""

    def __init__(self) -> None:
        super().__init__(
            "K8s-on-Slurm is an experimental feature. "
            "Enable with RIVAULT_EXPERIMENTAL_K8S_ON_SLURM=1"
        )


class K8sBootstrapWrapper(RuntimeWrapper):
    """
    Experimental: Boot k3s inside Slurm job (U-1115, U-1131-1134).

    This wrapper enables running K8s workloads on Slurm-managed infrastructure
    by bootstrapping an ephemeral k3s cluster inside a Slurm allocation.

    Use cases:
    - Workflows requiring K8s semantics (operators, CRDs)
    - Testing K8s deployments on HPC hardware
    - Hybrid workloads

    Requirements:
    - Feature flag: RIVAULT_EXPERIMENTAL_K8S_ON_SLURM=1
    - k3s binary available in container or on node
    - Network connectivity from gateway to Slurm compute nodes

    Limitations:
    - Single-node k3s only (no HA)
    - Ephemeral - cluster destroyed when Slurm job ends
    - Network may be limited depending on HPC config
    """

    def __init__(
        self,
        k3s_data_dir: str = "/tmp/k3s-data",
        k3s_timeout_sec: int = 120,
        namespace: str = "rivault",
    ):
        """
        Initialize K8sBootstrapWrapper.

        Args:
            k3s_data_dir: Directory for k3s data (inside allocation).
            k3s_timeout_sec: Timeout for k3s bootstrap.
            namespace: K8s namespace for workloads.
        """
        self._check_feature_flag()
        self.k3s_data_dir = k3s_data_dir
        self.k3s_timeout_sec = k3s_timeout_sec
        self.namespace = namespace
        self._kubeconfig: Optional[str] = None

    def _check_feature_flag(self) -> None:
        """Check that experimental feature is enabled."""
        flag = os.environ.get("RIVAULT_EXPERIMENTAL_K8S_ON_SLURM", "0")
        if flag not in ("1", "true", "yes"):
            raise K8sOnSlurmNotEnabledError()

    @property
    def runtime_name(self) -> str:
        return "k3s-on-slurm"

    async def launch(
        self,
        allocation: AllocationInfo,
        spec: LaunchSpec,
    ) -> EndpointInfo:
        """
        Bootstrap k3s and deploy workload.

        Steps:
        1. Start k3s server on allocated node
        2. Wait for k3s to be ready
        3. Verify network reachability (U-1133)
        4. Deploy workload via kubectl
        5. Return endpoint info
        """
        node = allocation.node
        job_id = allocation.job_id

        logger.info(
            f"[K8sBootstrap] Starting k3s on node={node}, job={job_id}"
        )

        # Generate unique names
        cluster_id = f"rivault-{job_id[:8]}"

        # Start k3s server
        kubeconfig = await self._start_k3s(node, cluster_id)
        self._kubeconfig = kubeconfig

        # Verify network reachability (U-1133)
        if not await self._check_network_reachability(node, 6443):
            await self._stop_k3s(node, cluster_id)
            raise NetworkUnreachableError(
                node=node,
                port=6443,
                message=(
                    "Cannot reach k3s API on Slurm node. "
                    "HPC network may not allow external access. "
                    "Consider using native Slurm backend instead."
                ),
            )

        # Deploy workload
        port = list(spec.ports.values())[0] if spec.ports else 11434
        endpoint = await self._deploy_workload(
            spec=spec,
            kubeconfig=kubeconfig,
            node=node,
            port=port,
        )

        endpoint.metadata["cluster_id"] = cluster_id
        endpoint.metadata["kubeconfig"] = kubeconfig

        return endpoint

    async def teardown(self, endpoint: EndpointInfo) -> bool:
        """Stop k3s and clean up."""
        cluster_id = endpoint.metadata.get("cluster_id", "")
        node = endpoint.node

        if not cluster_id:
            logger.warning("[K8sBootstrap] No cluster_id in endpoint metadata")
            return False

        return await self._stop_k3s(node, cluster_id)

    async def is_running(self, endpoint: EndpointInfo) -> bool:
        """Check if k3s workload is running."""
        kubeconfig = endpoint.metadata.get("kubeconfig")
        if not kubeconfig or not os.path.exists(kubeconfig):
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "kubectl", "--kubeconfig", kubeconfig,
                "get", "pods", "-n", self.namespace,
                "-o", "jsonpath={.items[0].status.phase}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            return stdout.decode().strip() == "Running"
        except Exception:
            return False

    async def _start_k3s(self, node: str, cluster_id: str) -> str:
        """Start k3s server on remote node."""
        data_dir = f"{self.k3s_data_dir}/{cluster_id}"
        kubeconfig = f"{data_dir}/kubeconfig.yaml"

        # Build k3s start command
        k3s_cmd = (
            f"mkdir -p {data_dir} && "
            f"k3s server "
            f"--data-dir {data_dir} "
            f"--write-kubeconfig {kubeconfig} "
            f"--write-kubeconfig-mode 644 "
            f"--disable traefik "
            f"--disable servicelb "
            f"--disable local-storage "
            f"--https-listen-port 6443 "
            f"--tls-san {node} "
            f"&>/dev/null &"
        )

        # Execute via SSH or direct if on same node
        await self._remote_exec(node, k3s_cmd)

        # Wait for k3s to be ready
        ready = await self._wait_for_k3s_ready(node, kubeconfig)
        if not ready:
            raise RuntimeError(
                f"k3s failed to start on {node} within {self.k3s_timeout_sec}s"
            )

        logger.info(f"[K8sBootstrap] k3s ready on {node}, kubeconfig={kubeconfig}")
        return kubeconfig

    async def _wait_for_k3s_ready(
        self,
        node: str,
        kubeconfig: str,
    ) -> bool:
        """Wait for k3s to become ready."""
        import aiohttp

        deadline = asyncio.get_event_loop().time() + self.k3s_timeout_sec

        while asyncio.get_event_loop().time() < deadline:
            try:
                # Check if kubectl can reach the cluster
                proc = await asyncio.create_subprocess_exec(
                    "kubectl", "--kubeconfig", kubeconfig,
                    "get", "nodes", "-o", "name",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0 and stdout:
                    return True
            except Exception:
                pass

            await asyncio.sleep(5)

        return False

    async def _stop_k3s(self, node: str, cluster_id: str) -> bool:
        """Stop k3s server."""
        try:
            stop_cmd = (
                f"pkill -f 'k3s server.*{cluster_id}' || true; "
                f"rm -rf {self.k3s_data_dir}/{cluster_id} || true"
            )
            await self._remote_exec(node, stop_cmd)
            logger.info(f"[K8sBootstrap] Stopped k3s cluster={cluster_id}")
            return True
        except Exception as e:
            logger.error(f"[K8sBootstrap] Failed to stop k3s: {e}")
            return False

    async def _check_network_reachability(
        self,
        node: str,
        port: int,
    ) -> bool:
        """
        Check if we can reach the node from the gateway (U-1133).

        This is critical for HPC environments where compute nodes
        may not be accessible from external networks.
        """
        import socket

        try:
            # Simple TCP connect test
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((node, port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.warning(f"[K8sBootstrap] Network check failed: {e}")
            return False

    async def _deploy_workload(
        self,
        spec: LaunchSpec,
        kubeconfig: str,
        node: str,
        port: int,
    ) -> EndpointInfo:
        """Deploy Mini-RiVault workload to ephemeral k3s."""
        import tempfile
        import yaml

        # Create namespace
        await self._kubectl(kubeconfig, ["create", "namespace", self.namespace])

        # Build deployment manifest
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "rivault-inference",
                "namespace": self.namespace,
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {"app": "rivault-inference"},
                },
                "template": {
                    "metadata": {
                        "labels": {"app": "rivault-inference"},
                    },
                    "spec": {
                        "containers": [{
                            "name": "vllm",
                            "image": spec.image,
                            "command": spec.command if spec.command else None,
                            "args": spec.args if spec.args else None,
                            "env": [
                                {"name": k, "value": v}
                                for k, v in spec.env.items()
                            ],
                            "ports": [{"containerPort": p} for p in spec.ports.keys()],
                            "volumeMounts": [
                                {
                                    "name": f"mount-{i}",
                                    "mountPath": m.target,
                                    "readOnly": m.read_only,
                                }
                                for i, m in enumerate(spec.mounts)
                            ] if spec.mounts else [],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": str(spec.gpus),
                                    "memory": f"{spec.memory_mb}Mi",
                                },
                            },
                        }],
                        "volumes": [
                            {
                                "name": f"mount-{i}",
                                "hostPath": {"path": m.source},
                            }
                            for i, m in enumerate(spec.mounts)
                            if m.mount_type == "bind"
                        ] if spec.mounts else [],
                    },
                },
            },
        }

        # Clean up None values
        container = deployment["spec"]["template"]["spec"]["containers"][0]
        if container["command"] is None:
            del container["command"]
        if container["args"] is None:
            del container["args"]

        # Apply deployment
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.dump(deployment, f)
            manifest_path = f.name

        await self._kubectl(kubeconfig, ["apply", "-f", manifest_path])
        os.unlink(manifest_path)

        # Create service with NodePort for external access
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "rivault-inference",
                "namespace": self.namespace,
            },
            "spec": {
                "type": "NodePort",
                "selector": {"app": "rivault-inference"},
                "ports": [{
                    "port": port,
                    "targetPort": port,
                    "nodePort": port,  # Use same port for simplicity
                }],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.dump(service, f)
            manifest_path = f.name

        await self._kubectl(kubeconfig, ["apply", "-f", manifest_path])
        os.unlink(manifest_path)

        return EndpointInfo(
            node=node,
            port=port,
            container_id=f"k3s:{self.namespace}/rivault-inference",
            launched_at=datetime.now(),
        )

    async def _kubectl(
        self,
        kubeconfig: str,
        args: List[str],
    ) -> str:
        """Execute kubectl command."""
        proc = await asyncio.create_subprocess_exec(
            "kubectl", "--kubeconfig", kubeconfig, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"kubectl {' '.join(args)} failed: {stderr.decode()}"
            )

        return stdout.decode()

    async def _remote_exec(self, node: str, command: str) -> str:
        """Execute command on remote node."""
        import socket

        # If we're on the same node, execute directly
        if socket.gethostname() == node:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Command failed: {stderr.decode()}")
            return stdout.decode()

        # Otherwise use SSH
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            node, command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"SSH command failed: {stderr.decode()}")
        return stdout.decode()


class NetworkUnreachableError(Exception):
    """Raised when network connectivity check fails (U-1134)."""

    def __init__(
        self,
        node: str,
        port: int,
        message: str,
    ):
        self.node = node
        self.port = port
        super().__init__(message)
