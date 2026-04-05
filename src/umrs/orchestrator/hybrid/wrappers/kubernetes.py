"""Kubernetes wrapper implementations (U-1114, U-1115)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..targets import AllocationInfo
from .base import ContainerEndpoint, LaunchError, LaunchSpec, MountSpec, RuntimeWrapper

logger = logging.getLogger(__name__)


# =============================================================================
# K8sJobWrapper (U-1114)
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
        self.namespace = namespace or os.getenv("K8S_NAMESPACE", "rivault-system")
        self.kubeconfig = kubeconfig or os.getenv("KUBECONFIG")

    @property
    def runtime_name(self) -> str:
        return "k8s"

    async def launch(
        self,
        allocation: AllocationInfo,
        spec: LaunchSpec,
    ) -> ContainerEndpoint:
        """Update K8s Job with container spec and create Service."""
        if not spec.image:
            raise LaunchError("No container image specified")

        job_name = allocation.job_id
        patch = self._build_job_patch(spec)

        try:
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
                logger.warning(f"Job patch failed: {stderr.decode()}")

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

            node = await self._get_service_endpoint(service_name)

            return ContainerEndpoint(
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

    async def teardown(self, endpoint: ContainerEndpoint) -> bool:
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

    async def is_running(self, endpoint: ContainerEndpoint) -> bool:
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
        container_spec: Dict[str, Any] = {
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
# K8sBootstrapWrapper — Experimental K8s-on-Slurm (U-1115)
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
    """

    def __init__(
        self,
        k3s_data_dir: str = "/tmp/k3s-data",
        k3s_timeout_sec: int = 120,
        namespace: str = "rivault",
    ):
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
    ) -> ContainerEndpoint:
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

        cluster_id = f"rivault-{job_id[:8]}"

        kubeconfig = await self._start_k3s(node, cluster_id)
        self._kubeconfig = kubeconfig

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

    async def teardown(self, endpoint: ContainerEndpoint) -> bool:
        """Stop k3s and clean up."""
        cluster_id = endpoint.metadata.get("cluster_id", "")
        node = endpoint.node

        if not cluster_id:
            logger.warning("[K8sBootstrap] No cluster_id in endpoint metadata")
            return False

        return await self._stop_k3s(node, cluster_id)

    async def is_running(self, endpoint: ContainerEndpoint) -> bool:
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

        await self._remote_exec(node, k3s_cmd)

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
        deadline = asyncio.get_event_loop().time() + self.k3s_timeout_sec

        while asyncio.get_event_loop().time() < deadline:
            try:
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
        """Check if we can reach the node from the gateway (U-1133)."""
        import socket

        try:
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
    ) -> ContainerEndpoint:
        """Deploy Mini-RiVault workload to ephemeral k3s."""
        import tempfile
        import yaml

        await self._kubectl(kubeconfig, ["create", "namespace", self.namespace])

        deployment: Dict[str, Any] = {
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

        container = deployment["spec"]["template"]["spec"]["containers"][0]
        if container["command"] is None:
            del container["command"]
        if container["args"] is None:
            del container["args"]

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            delete=False,
        ) as f:
            yaml.dump(deployment, f)
            manifest_path = f.name

        await self._kubectl(kubeconfig, ["apply", "-f", manifest_path])
        os.unlink(manifest_path)

        service: Dict[str, Any] = {
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
                    "nodePort": port,
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

        return ContainerEndpoint(
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
