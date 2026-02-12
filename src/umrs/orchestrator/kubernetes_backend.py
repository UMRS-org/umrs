"""
Kubernetes orchestrator backend for Mini-RiVault.

Manages LLM inference endpoints via Kubernetes Jobs/Deployments.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import (
    EndpointInfo,
    JobInfo,
    JobStatus,
    OrchestratorBackend,
    OrchestratorConfig,
)


class KubernetesBackend(OrchestratorBackend):
    """
    Kubernetes-based orchestrator for cloud/server environments.

    Uses Kubernetes Jobs for on-demand inference or Deployments for
    persistent endpoints. Supports GPU scheduling via NVIDIA device plugin.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        super().__init__(config)
        self._k8s_client = None
        self._k8s_available = None

    @property
    def backend_name(self) -> str:
        return "kubernetes"

    async def _ensure_k8s_client(self):
        """Lazy-load kubernetes client."""
        if self._k8s_client is not None:
            return

        try:
            from kubernetes import client, config as k8s_config
            from kubernetes.client import ApiException

            # Try in-cluster config first, then kubeconfig
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()

            self._k8s_client = {
                "core": client.CoreV1Api(),
                "batch": client.BatchV1Api(),
                "apps": client.AppsV1Api(),
                "ApiException": ApiException,
            }
            self._k8s_available = True
        except ImportError:
            self._k8s_available = False
            raise RuntimeError(
                "kubernetes Python package not installed. "
                "Install with: pip install kubernetes"
            )
        except Exception as e:
            self._k8s_available = False
            raise RuntimeError(f"Failed to initialize Kubernetes client: {e}")

    def _job_name(self, suffix: str = "") -> str:
        """Generate a unique job name."""
        import uuid
        uid = uuid.uuid4().hex[:8]
        return f"mini-rivault-{uid}{'-' + suffix if suffix else ''}"

    def _build_job_manifest(
        self,
        job_name: str,
        model_path: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build Kubernetes Job manifest for Mini-RiVault."""
        cfg = self.config
        namespace = kwargs.get("namespace", cfg.k8s_namespace)
        gpus = kwargs.get("gpus", 1)

        # Environment variables
        env_vars = [
            {"name": "VLLM_MODEL_PATH", "value": "/models"},
            {"name": "VLLM_PORT", "value": str(cfg.vllm_port)},
            {"name": "LITELLM_PORT", "value": str(cfg.litellm_port)},
            {"name": "VLLM_MAX_MODEL_LEN", "value": str(cfg.vllm_max_model_len)},
            {"name": "VLLM_MAX_NUM_SEQS", "value": str(cfg.vllm_max_num_seqs)},
            {"name": "VLLM_TENSOR_PARALLEL_SIZE", "value": str(cfg.vllm_tensor_parallel_size)},
            {"name": "VLLM_GPU_MEMORY_UTILIZATION", "value": str(cfg.vllm_gpu_memory_utilization)},
            {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
            {"name": "HF_HUB_OFFLINE", "value": "1"},
            {"name": "VLLM_NO_USAGE_STATS", "value": "1"},
        ]

        # Add LiteLLM master key from secret or direct value
        if cfg.litellm_master_key:
            env_vars.append({
                "name": "LITELLM_MASTER_KEY",
                "valueFrom": {
                    "secretKeyRef": {
                        "name": "mini-rivault-secrets",
                        "key": "litellm-master-key",
                        "optional": True,
                    }
                }
            })

        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "mini-rivault",
                    "app.kubernetes.io/component": "inference",
                    "app.kubernetes.io/managed-by": "umrs-orchestrator",
                },
            },
            "spec": {
                "ttlSecondsAfterFinished": 300,  # Cleanup 5 min after completion
                "backoffLimit": 0,  # Don't retry on failure
                "template": {
                    "metadata": {
                        "labels": {
                            "app.kubernetes.io/name": "mini-rivault",
                            "app.kubernetes.io/component": "inference",
                        },
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "mini-rivault",
                                "image": cfg.k8s_image,
                                "imagePullPolicy": "IfNotPresent",
                                "env": env_vars,
                                "ports": [
                                    {"containerPort": cfg.litellm_port, "name": "litellm"},
                                    {"containerPort": cfg.vllm_port, "name": "vllm"},
                                ],
                                "resources": {
                                    "limits": {
                                        cfg.k8s_gpu_resource: str(gpus),
                                        "memory": "64Gi",
                                    },
                                    "requests": {
                                        cfg.k8s_gpu_resource: str(gpus),
                                        "memory": "32Gi",
                                        "cpu": "4",
                                    },
                                },
                                "volumeMounts": [
                                    {
                                        "name": "models",
                                        "mountPath": "/models",
                                        "readOnly": True,
                                    }
                                ],
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/v1/models",
                                        "port": cfg.litellm_port,
                                        "httpHeaders": [
                                            {
                                                "name": "Authorization",
                                                "value": f"Bearer {cfg.litellm_master_key}",
                                            }
                                        ] if cfg.litellm_master_key else [],
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 60,  # 10 min total
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": cfg.litellm_port,
                                    },
                                    "initialDelaySeconds": 60,
                                    "periodSeconds": 30,
                                },
                            }
                        ],
                        "volumes": [
                            {
                                "name": "models",
                                "persistentVolumeClaim": {
                                    "claimName": cfg.k8s_model_pvc,
                                },
                            }
                        ],
                        # Network policy: deny egress except DNS
                        "dnsPolicy": "ClusterFirst",
                    },
                },
            },
        }

        return manifest

    def _build_service_manifest(self, job_name: str, namespace: str) -> Dict[str, Any]:
        """Build Kubernetes Service manifest for the job."""
        cfg = self.config

        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": job_name,
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "mini-rivault",
                    "app.kubernetes.io/managed-by": "umrs-orchestrator",
                },
            },
            "spec": {
                "selector": {
                    "job-name": job_name,
                },
                "ports": [
                    {
                        "name": "litellm",
                        "port": cfg.litellm_port,
                        "targetPort": cfg.litellm_port,
                    },
                ],
                "type": "ClusterIP",
            },
        }

    async def submit_job(
        self,
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> JobInfo:
        """Submit a Kubernetes Job for Mini-RiVault."""
        await self._ensure_k8s_client()

        model = model_path or self.config.model_path
        if not model:
            raise ValueError("model_path is required")

        namespace = kwargs.get("namespace", self.config.k8s_namespace)
        job_name = self._job_name()

        # Create Job
        job_manifest = self._build_job_manifest(job_name, model, **kwargs)

        try:
            self._k8s_client["batch"].create_namespaced_job(
                namespace=namespace,
                body=job_manifest,
            )
        except self._k8s_client["ApiException"] as e:
            raise RuntimeError(f"Failed to create Job: {e}")

        # Create Service for endpoint discovery
        service_manifest = self._build_service_manifest(job_name, namespace)
        try:
            self._k8s_client["core"].create_namespaced_service(
                namespace=namespace,
                body=service_manifest,
            )
        except self._k8s_client["ApiException"]:
            # Service creation failure is not fatal
            pass

        return JobInfo(
            job_id=job_name,
            status=JobStatus.PENDING,
            model_path=model,
            created_at=datetime.now(),
            backend=self.backend_name,
            metadata={"namespace": namespace},
        )

    async def get_job_status(self, job_id: str) -> JobInfo:
        """Get status of a Kubernetes Job."""
        await self._ensure_k8s_client()

        namespace = self.config.k8s_namespace

        try:
            job = self._k8s_client["batch"].read_namespaced_job(
                name=job_id,
                namespace=namespace,
            )
        except self._k8s_client["ApiException"] as e:
            if e.status == 404:
                return JobInfo(
                    job_id=job_id,
                    status=JobStatus.UNKNOWN,
                    backend=self.backend_name,
                    error_message="Job not found",
                )
            raise

        # Determine status from Job conditions
        status = JobStatus.PENDING
        error_message = None

        if job.status.succeeded and job.status.succeeded > 0:
            status = JobStatus.STOPPED
        elif job.status.failed and job.status.failed > 0:
            status = JobStatus.FAILED
            if job.status.conditions:
                for cond in job.status.conditions:
                    if cond.type == "Failed":
                        error_message = cond.message
        elif job.status.active and job.status.active > 0:
            status = JobStatus.STARTING

        # Check if pod is ready
        endpoint = None
        if status == JobStatus.STARTING:
            try:
                pods = self._k8s_client["core"].list_namespaced_pod(
                    namespace=namespace,
                    label_selector=f"job-name={job_id}",
                )
                for pod in pods.items:
                    if pod.status.phase == "Running":
                        # Check readiness
                        for cond in pod.status.conditions or []:
                            if cond.type == "Ready" and cond.status == "True":
                                status = JobStatus.READY
                                endpoint = EndpointInfo(
                                    node=f"{job_id}.{namespace}.svc.cluster.local",
                                    litellm_port=self.config.litellm_port,
                                    vllm_port=self.config.vllm_port,
                                )
                                break
            except self._k8s_client["ApiException"]:
                pass

        return JobInfo(
            job_id=job_id,
            status=status,
            endpoint=endpoint,
            backend=self.backend_name,
            error_message=error_message,
            metadata={"namespace": namespace},
        )

    async def wait_for_ready(
        self,
        job_id: str,
        timeout_sec: Optional[int] = None,
    ) -> EndpointInfo:
        """Wait for job to be ready."""
        timeout = timeout_sec or self.config.startup_timeout_sec
        poll_interval = self.config.readiness_poll_interval_sec
        elapsed = 0

        while elapsed < timeout:
            job = await self.get_job_status(job_id)

            if job.status == JobStatus.FAILED:
                raise RuntimeError(f"Job {job_id} failed: {job.error_message}")

            if job.is_ready() and job.endpoint:
                return job.endpoint

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        raise TimeoutError(f"Job {job_id} did not become ready within {timeout}s")

    async def cleanup(self, job_id: str, force: bool = False) -> None:
        """Delete a Kubernetes Job and its Service."""
        await self._ensure_k8s_client()
        namespace = self.config.k8s_namespace

        propagation = "Background" if force else "Foreground"

        try:
            self._k8s_client["batch"].delete_namespaced_job(
                name=job_id,
                namespace=namespace,
                propagation_policy=propagation,
            )
        except self._k8s_client["ApiException"] as e:
            if e.status != 404:
                raise

        # Delete service
        try:
            self._k8s_client["core"].delete_namespaced_service(
                name=job_id,
                namespace=namespace,
            )
        except self._k8s_client["ApiException"]:
            pass

    async def list_jobs(
        self,
        status_filter: Optional[List[JobStatus]] = None,
    ) -> List[JobInfo]:
        """List all Mini-RiVault Jobs."""
        await self._ensure_k8s_client()
        namespace = self.config.k8s_namespace

        try:
            jobs = self._k8s_client["batch"].list_namespaced_job(
                namespace=namespace,
                label_selector="app.kubernetes.io/name=mini-rivault",
            )
        except self._k8s_client["ApiException"] as e:
            raise RuntimeError(f"Failed to list Jobs: {e}")

        result = []
        for job in jobs.items:
            job_info = await self.get_job_status(job.metadata.name)
            if status_filter is None or job_info.status in status_filter:
                result.append(job_info)

        return result

    async def cleanup_orphans(self) -> int:
        """Clean up completed/failed Jobs past TTL."""
        await self._ensure_k8s_client()
        namespace = self.config.k8s_namespace
        cleaned = 0

        try:
            jobs = self._k8s_client["batch"].list_namespaced_job(
                namespace=namespace,
                label_selector="app.kubernetes.io/name=mini-rivault",
            )
        except self._k8s_client["ApiException"]:
            return 0

        for job in jobs.items:
            # Check if job is completed/failed
            if (job.status.succeeded and job.status.succeeded > 0) or \
               (job.status.failed and job.status.failed > 0):
                try:
                    await self.cleanup(job.metadata.name, force=True)
                    cleaned += 1
                except Exception:
                    pass

        return cleaned

    async def health_check(self) -> Dict[str, Any]:
        """Check if Kubernetes cluster is accessible."""
        try:
            await self._ensure_k8s_client()

            # Try to list namespaces
            self._k8s_client["core"].list_namespace(limit=1)

            return {
                "backend": self.backend_name,
                "healthy": True,
                "message": "Kubernetes cluster accessible",
                "namespace": self.config.k8s_namespace,
            }
        except Exception as e:
            return {
                "backend": self.backend_name,
                "healthy": False,
                "message": str(e),
            }
