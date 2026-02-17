"""
Kubernetes plugin runner (X-104).

Executes plugins as Kubernetes Jobs with container isolation.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .models import PluginSpec, PluginResult, PluginStatus
from .runner import PluginRunner, DEFAULT_PLUGIN_IMAGE

logger = logging.getLogger(__name__)

# K8s configuration
K8S_NAMESPACE = os.environ.get("PLUGIN_K8S_NAMESPACE", "rivault-plugins")
K8S_SERVICE_ACCOUNT = os.environ.get("PLUGIN_K8S_SA", "plugin-runner")


class K8sPluginRunner(PluginRunner):
    """
    Execute plugins as Kubernetes Jobs (X-104).

    Uses PVC mounts for plugin code and output.
    Supports GPU scheduling via resource requests.
    """

    def __init__(
        self,
        namespace: str = K8S_NAMESPACE,
        service_account: str = K8S_SERVICE_ACCOUNT,
        image: str = DEFAULT_PLUGIN_IMAGE,
        **kwargs,
    ):
        """
        Initialize K8s plugin runner.

        Args:
            namespace: Kubernetes namespace for plugin jobs.
            service_account: Service account for pods.
            image: Container image for plugin execution.
        """
        super().__init__(base_image=image, **kwargs)
        self.namespace = namespace
        self.service_account = service_account

        # Track submitted jobs: plugin_id -> job_name
        self._k8s_jobs: Dict[str, str] = {}

        # Lazy-load kubernetes client
        self._k8s_client = None
        self._batch_api = None

    def _get_k8s_client(self):
        """Get or create Kubernetes client."""
        if self._k8s_client is None:
            try:
                from kubernetes import client, config

                # Try in-cluster config first, then kubeconfig
                try:
                    config.load_incluster_config()
                except config.ConfigException:
                    config.load_kube_config()

                self._k8s_client = client
                self._batch_api = client.BatchV1Api()
                self._core_api = client.CoreV1Api()
            except ImportError:
                raise RuntimeError(
                    "kubernetes package required for K8s plugin runner. "
                    "Install with: pip install kubernetes"
                )
        return self._k8s_client

    async def _execute_backend(
        self,
        plugin_id: str,
        spec: PluginSpec,
    ) -> PluginResult:
        """Execute plugin as Kubernetes Job."""
        client = self._get_k8s_client()

        job_name = f"plugin-{plugin_id[:8]}"
        self._k8s_jobs[plugin_id] = job_name

        try:
            # Create job manifest
            job = self._create_job_manifest(job_name, plugin_id, spec)

            # Submit job
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._batch_api.create_namespaced_job(
                    body=job,
                    namespace=self.namespace,
                ),
            )

            logger.info(f"Created K8s job {job_name} for plugin {plugin_id}")

            # Wait for completion
            result = await self._wait_for_completion(plugin_id, job_name, spec)

            return result

        except Exception as e:
            logger.exception(f"K8s execution failed for {plugin_id}")
            return PluginResult(
                plugin_id=plugin_id,
                status=PluginStatus.FAILED,
                error_message=str(e),
            )

    def _create_job_manifest(
        self,
        job_name: str,
        plugin_id: str,
        spec: PluginSpec,
    ) -> Dict[str, Any]:
        """Create Kubernetes Job manifest."""
        client = self._get_k8s_client()

        # Environment variables
        env_vars = self._generate_container_env(spec)
        env_list = [
            client.V1EnvVar(name=k, value=v)
            for k, v in env_vars.items()
        ]

        # Resource requests and limits
        resources = client.V1ResourceRequirements(
            requests={
                "cpu": spec.cpu_limit,
                "memory": spec.memory_limit,
            },
            limits={
                "cpu": spec.cpu_limit,
                "memory": spec.memory_limit,
            },
        )

        # Add GPU if requested
        if spec.gpu_count > 0:
            resources.limits["nvidia.com/gpu"] = str(spec.gpu_count)
            resources.requests["nvidia.com/gpu"] = str(spec.gpu_count)

        # Volume mounts
        volume_mounts = [
            client.V1VolumeMount(
                name="workspace",
                mount_path="/workspace",
            ),
        ]

        # Volumes
        volumes = [
            client.V1Volume(
                name="workspace",
                empty_dir=client.V1EmptyDirVolumeSource(),
            ),
        ]

        # Add plugin PVC mount if using local path
        if spec.source_type.value == "local_path":
            volume_mounts.append(
                client.V1VolumeMount(
                    name="plugin-source",
                    mount_path="/plugin",
                    read_only=True,
                )
            )
            volumes.append(
                client.V1Volume(
                    name="plugin-source",
                    host_path=client.V1HostPathVolumeSource(
                        path=spec.source_uri,
                        type="Directory",
                    ),
                )
            )

        # Container spec
        container = client.V1Container(
            name="plugin",
            image=self.base_image,
            env=env_list,
            resources=resources,
            volume_mounts=volume_mounts,
            security_context=client.V1SecurityContext(
                run_as_non_root=True,
                read_only_root_filesystem=True,
                allow_privilege_escalation=False,
                capabilities=client.V1Capabilities(drop=["ALL"]),
            ),
        )

        # Pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            volumes=volumes,
            service_account_name=self.service_account,
            restart_policy="Never",
            active_deadline_seconds=spec.timeout_seconds,
            security_context=client.V1PodSecurityContext(
                run_as_non_root=True,
                seccomp_profile=client.V1SeccompProfile(type="RuntimeDefault"),
            ),
        )

        # Job spec
        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels={
                        "app": "plugin-runner",
                        "plugin-id": plugin_id,
                    },
                ),
                spec=pod_spec,
            ),
            backoff_limit=0,  # No retries at K8s level
            ttl_seconds_after_finished=3600,  # Cleanup after 1 hour
        )

        # Job manifest
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels={
                    "app": "plugin-runner",
                    "plugin-id": plugin_id,
                    "project-id": spec.project_id or "unknown",
                },
            ),
            spec=job_spec,
        )

        return job

    async def _wait_for_completion(
        self,
        plugin_id: str,
        job_name: str,
        spec: PluginSpec,
    ) -> PluginResult:
        """Wait for Kubernetes Job completion."""
        poll_interval = 5
        max_wait = spec.timeout_seconds + 300
        elapsed = 0

        while elapsed < max_wait:
            status = await self._get_job_status(job_name)

            if status in (PluginStatus.SUCCEEDED, PluginStatus.FAILED, PluginStatus.CANCELLED):
                break

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Collect results
        return await self._collect_results(plugin_id, job_name, status)

    async def _get_job_status(self, job_name: str) -> PluginStatus:
        """Get Kubernetes Job status."""
        try:
            job = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._batch_api.read_namespaced_job_status(
                    name=job_name,
                    namespace=self.namespace,
                ),
            )

            if job.status.succeeded:
                return PluginStatus.SUCCEEDED
            elif job.status.failed:
                return PluginStatus.FAILED
            elif job.status.active:
                return PluginStatus.RUNNING
            else:
                return PluginStatus.PENDING

        except Exception:
            return PluginStatus.RUNNING

    async def _collect_results(
        self,
        plugin_id: str,
        job_name: str,
        status: PluginStatus,
    ) -> PluginResult:
        """Collect results from completed job."""
        stdout = ""
        stderr = ""
        exit_code = None

        try:
            # Get pod logs
            pods = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._core_api.list_namespaced_pod(
                    namespace=self.namespace,
                    label_selector=f"job-name={job_name}",
                ),
            )

            if pods.items:
                pod = pods.items[0]
                try:
                    logs = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self._core_api.read_namespaced_pod_log(
                            name=pod.metadata.name,
                            namespace=self.namespace,
                            tail_lines=1000,
                        ),
                    )
                    stdout = logs
                except Exception:
                    pass

                # Get exit code from container status
                if pod.status.container_statuses:
                    cs = pod.status.container_statuses[0]
                    if cs.state.terminated:
                        exit_code = cs.state.terminated.exit_code

        except Exception as e:
            logger.warning(f"Error collecting results for {job_name}: {e}")

        return PluginResult(
            plugin_id=plugin_id,
            status=status,
            exit_code=exit_code,
            stdout=stdout[-50000:] if stdout else None,
            stderr=stderr[-50000:] if stderr else None,
            metrics={"k8s_job_name": job_name},
        )

    async def cancel(self, plugin_id: str) -> bool:
        """Cancel a running plugin job."""
        job_name = self._k8s_jobs.get(plugin_id)
        if not job_name:
            return False

        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._batch_api.delete_namespaced_job(
                    name=job_name,
                    namespace=self.namespace,
                    propagation_policy="Background",
                ),
            )
            return True
        except Exception:
            logger.exception(f"Failed to cancel K8s job {job_name}")
            return False

    async def get_status(self, plugin_id: str) -> Optional[PluginStatus]:
        """Get current status of a plugin execution."""
        job_name = self._k8s_jobs.get(plugin_id)
        if not job_name:
            return None

        return await self._get_job_status(job_name)
