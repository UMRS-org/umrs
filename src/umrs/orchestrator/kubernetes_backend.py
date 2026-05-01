"""
Kubernetes orchestrator backend for Mini-RiVault.

Manages LLM inference endpoints via Kubernetes Jobs/Deployments.

Phase 2k robustness changes (R-8 through R-13):
- R-8: Apply a NetworkPolicy denying egress to enforce the offline-inference
  invariant (mini-rivault must not have public network access).
- R-9: Defer the READY verdict to UMRS's authoritative ReadinessPoller; this
  backend reports only PENDING/STARTING/FAILED based on Pod phase.
- R-10: Map the request-supplied ``model_path`` to a per-job volume so that
  ``r4 endpoint up --model X`` actually selects model X on K8s.
- R-11: Capture the last 200 lines of pod logs into ``error_message`` on
  failure so operators do not need cluster shell access to triage.
- R-12: Avoid an N+1 ``read_namespaced_job`` per item in ``list_jobs``;
  build :class:`JobInfo` from the bulk list response. Add the
  ``batch.kubernetes.io/job-name`` selector for K8s >= 1.27.
- R-13: Resource requests/limits come from the provisioning request, not
  from hardcoded constants.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .base import (
    EndpointInfo,
    JobInfo,
    JobStatus,
    OrchestratorBackend,
    OrchestratorConfig,
)

logger = logging.getLogger(__name__)

_DEFAULT_EGRESS_PROXY_URL = "http://egress-guard:8888"
_DEFAULT_EGRESS_NO_PROXY = (
    "localhost,127.0.0.1,::1,litellm,vllm,ll-postgres,coredns,nginx,egress-guard"
)


def _k8s_egress_allowed_cidrs() -> List[str]:
    """Return optional site-specific CIDRs allowed outside EgressGuard."""
    return [
        cidr.strip()
        for cidr in os.environ.get("K8S_EGRESS_ALLOWED_CIDRS", "").split(",")
        if cidr.strip()
    ]


def _proxy_port(proxy_url: str) -> int:
    parsed = urlparse(proxy_url or _DEFAULT_EGRESS_PROXY_URL)
    if parsed.port:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    return 80


def _dns_egress_rule() -> Dict[str, Any]:
    return {
        "to": [
            {
                "namespaceSelector": {},
                "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
            }
        ],
        "ports": [
            {"protocol": "UDP", "port": 53},
            {"protocol": "TCP", "port": 53},
        ],
    }


def _egress_guard_rule(proxy_port: int) -> Dict[str, Any]:
    return {
        "to": [
            {"podSelector": {"matchLabels": {"app.kubernetes.io/name": "egress-guard"}}},
            {"podSelector": {"matchLabels": {"app": "egress-guard"}}},
        ],
        "ports": [{"protocol": "TCP", "port": proxy_port}],
    }


def _cidr_egress_rule(allowed_cidrs: List[str]) -> Dict[str, Any]:
    return {
        "to": [{"ipBlock": {"cidr": cidr}} for cidr in allowed_cidrs],
    }


def _egress_proxy_env_vars(cfg: OrchestratorConfig) -> List[Dict[str, str]]:
    egress_proxy_url = cfg.egress_proxy_url or _DEFAULT_EGRESS_PROXY_URL
    egress_no_proxy = cfg.egress_no_proxy or _DEFAULT_EGRESS_NO_PROXY
    return [
        {"name": "RIVAULT_HTTP_PROXY", "value": egress_proxy_url},
        {"name": "RIVAULT_HTTPS_PROXY", "value": egress_proxy_url},
        {"name": "EGRESS_GUARD_PROXY_URL", "value": egress_proxy_url},
        {"name": "HTTP_PROXY", "value": egress_proxy_url},
        {"name": "HTTPS_PROXY", "value": egress_proxy_url},
        {"name": "http_proxy", "value": egress_proxy_url},
        {"name": "https_proxy", "value": egress_proxy_url},
        {"name": "NO_PROXY", "value": egress_no_proxy},
        {"name": "no_proxy", "value": egress_no_proxy},
    ]


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
                # R-8: NetworkingV1Api needed for egress NetworkPolicy.
                "networking": client.NetworkingV1Api(),
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

    def _build_network_policy(
        self, job_name: str, namespace: str
    ) -> Dict[str, Any]:
        """Build a NetworkPolicy that denies egress except to allowed CIDRs.

        R-8 (Phase 2k): enforces the offline-inference invariant by allowing
        only kube-dns and the configured in-cluster CIDRs (S3/MinIO,
        Gateway). Public internet egress is implicitly denied.
        """
        # 53/UDP+TCP for DNS, plus EgressGuard for all HTTP(S) egress.
        # Site-specific private CIDRs may be added explicitly through
        # K8S_EGRESS_ALLOWED_CIDRS, but the default is no direct CIDR egress.
        egress_rules: List[Dict[str, Any]] = [
            _dns_egress_rule(),
            _egress_guard_rule(_proxy_port(self.config.egress_proxy_url)),
        ]
        allowed_cidrs = _k8s_egress_allowed_cidrs()
        if allowed_cidrs:
            egress_rules.append(_cidr_egress_rule(allowed_cidrs))
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{job_name}-egress",
                "namespace": namespace,
                "labels": {
                    "app.kubernetes.io/name": "mini-rivault",
                    "app.kubernetes.io/managed-by": "umrs-orchestrator",
                    "job-name": job_name,
                },
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app.kubernetes.io/name": "mini-rivault",
                        "job-name": job_name,
                    }
                },
                "policyTypes": ["Egress"],
                "egress": egress_rules,
            },
        }

    def _build_job_manifest(
        self,
        job_name: str,
        model_path: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build Kubernetes Job manifest for Mini-RiVault."""
        cfg = self.config
        namespace = kwargs.get("namespace", cfg.k8s_namespace)
        # R-13 (Phase 2k): resources come from request (gpu_count/memory/cpu)
        # not hardcoded. Falls back to historical defaults when omitted.
        gpus = kwargs.get("gpu_count", kwargs.get("gpus", 1))
        memory_request = kwargs.get("memory_request", "32Gi")
        memory_limit = kwargs.get("memory_limit", "64Gi")
        cpu_request = kwargs.get("cpu_request", "4")
        cpu_limit = kwargs.get("cpu_limit", str(int(cpu_request) * 2) if str(cpu_request).isdigit() else cpu_request)
        # R-10 (Phase 2k): caller-supplied model_path drives an explicit volume
        # subPath, so different requests on the same PVC pick different
        # snapshots. Empty model_path falls back to the legacy whole-PVC mount.
        model_subpath = kwargs.get("model_subpath")
        if model_subpath is None and model_path and not model_path.startswith("/models"):
            # Heuristic: if model_path looks like a relative key (e.g.
            # "Qwen/Qwen3-8B"), use it as a subPath inside the model PVC.
            model_subpath = model_path.lstrip("/")

        # Environment variables
        # R-10: container always sees /models as the mount point; the host
        # subPath selects which model snapshot is exposed.
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
        env_vars.extend(_egress_proxy_env_vars(cfg))
        # Forward any extra env from the request (e.g. VLLM_SERVED_MODEL_NAME).
        for key, value in (kwargs.get("env") or {}).items():
            env_vars.append({"name": key, "value": str(value)})

        # R-10: conditional subPath in the volumeMount.
        model_volume_mount: Dict[str, Any] = {
            "name": "models",
            "mountPath": "/models",
            "readOnly": True,
        }
        if model_subpath:
            model_volume_mount["subPath"] = model_subpath

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
                                        "memory": memory_limit,
                                        "cpu": cpu_limit,
                                    },
                                    "requests": {
                                        cfg.k8s_gpu_resource: str(gpus),
                                        "memory": memory_request,
                                        "cpu": cpu_request,
                                    },
                                },
                                "volumeMounts": [model_volume_mount],
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

        # R-8: apply the egress-deny NetworkPolicy before creating the Job.
        # NetworkPolicy can select pods that do not exist yet, which avoids a
        # startup race where a new pod could egress before the policy lands.
        netpol_manifest = self._build_network_policy(job_name, namespace)
        try:
            self._k8s_client["networking"].create_namespaced_network_policy(
                namespace=namespace,
                body=netpol_manifest,
            )
        except self._k8s_client["ApiException"] as exc:
            raise RuntimeError(
                f"Refusing to start job {job_name} without egress NetworkPolicy: {exc}"
            ) from exc

        job_manifest = self._build_job_manifest(job_name, model, **kwargs)
        try:
            self._k8s_client["batch"].create_namespaced_job(
                namespace=namespace,
                body=job_manifest,
            )
        except self._k8s_client["ApiException"] as exc:
            logger.error(
                "Failed to create Kubernetes Job for %s: %s; removing NetworkPolicy",
                job_name, exc,
            )
            try:
                self._k8s_client["networking"].delete_namespaced_network_policy(
                    name=netpol_manifest["metadata"]["name"],
                    namespace=namespace,
                )
            except self._k8s_client["ApiException"]:
                pass
            raise RuntimeError(f"Failed to create Job: {exc}") from exc

        # Create Service for endpoint discovery
        service_manifest = self._build_service_manifest(job_name, namespace)
        try:
            self._k8s_client["core"].create_namespaced_service(
                namespace=namespace,
                body=service_manifest,
            )
        except self._k8s_client["ApiException"] as exc:
            # Service creation failure is not fatal but degrades discovery;
            # log loudly so operators see it.
            logger.warning("Service creation failed for %s: %s", job_name, exc)

        return JobInfo(
            job_id=job_name,
            status=JobStatus.PENDING,
            model_path=model,
            created_at=datetime.now(),
            backend=self.backend_name,
            metadata={"namespace": namespace, "network_policy": netpol_manifest["metadata"]["name"]},
        )

    # R-12: K8s 1.27 renamed the auto-generated job pod label. Ship both so
    # the selector keeps working across versions.
    _JOB_NAME_SELECTOR_TMPL = "job-name={job},batch.kubernetes.io/job-name={job}"
    _JOB_NAME_SELECTOR_LEGACY = "job-name={job}"

    def _select_job_pods(self, namespace: str, job_id: str):
        """List pods for a job, tolerating both old and new label keys."""
        # Try the union selector first; some K8s versions reject the comma
        # form when one key is absent, so fall back to legacy on failure.
        try:
            return self._k8s_client["core"].list_namespaced_pod(
                namespace=namespace,
                label_selector=self._JOB_NAME_SELECTOR_TMPL.format(job=job_id),
            )
        except self._k8s_client["ApiException"]:
            return self._k8s_client["core"].list_namespaced_pod(
                namespace=namespace,
                label_selector=self._JOB_NAME_SELECTOR_LEGACY.format(job=job_id),
            )

    def _capture_pod_logs(self, namespace: str, job_id: str, tail: int = 200) -> Optional[str]:
        """R-11: pull last N lines of pod logs into the JobInfo error_message."""
        try:
            pods = self._select_job_pods(namespace, job_id)
        except self._k8s_client["ApiException"]:
            return None
        snippets: List[str] = []
        for pod in pods.items:
            try:
                log = self._k8s_client["core"].read_namespaced_pod_log(
                    name=pod.metadata.name,
                    namespace=namespace,
                    tail_lines=tail,
                )
                snippets.append(f"--- pod {pod.metadata.name} ---\n{log}")
            except self._k8s_client["ApiException"]:
                continue
        return "\n".join(snippets) if snippets else None

    def _job_to_info(self, job, namespace: str) -> JobInfo:
        """Build a JobInfo from a V1Job, with R-9 readiness deferral."""
        status = JobStatus.PENDING
        error_message: Optional[str] = None

        if job.status.succeeded and job.status.succeeded > 0:
            status = JobStatus.STOPPED
        elif job.status.failed and job.status.failed > 0:
            status = JobStatus.FAILED
            if job.status.conditions:
                for cond in job.status.conditions:
                    if cond.type == "Failed":
                        error_message = cond.message
            # R-11: enrich with pod logs so the failure message is actionable.
            log_tail = self._capture_pod_logs(namespace, job.metadata.name)
            if log_tail:
                error_message = (error_message or "") + "\n" + log_tail
        elif job.status.active and job.status.active > 0:
            status = JobStatus.STARTING

        # R-9: never promote to READY here. The endpoint coordinates are set
        # once the pod is Running so UMRS's ReadinessPoller can take over;
        # the registry transitions to READY when the poller succeeds.
        endpoint: Optional[EndpointInfo] = None
        if status == JobStatus.STARTING:
            try:
                pods = self._select_job_pods(namespace, job.metadata.name)
                for pod in pods.items:
                    if pod.status.phase == "Running":
                        endpoint = EndpointInfo(
                            node=f"{job.metadata.name}.{namespace}.svc.cluster.local",
                            litellm_port=self.config.litellm_port,
                            vllm_port=self.config.vllm_port,
                        )
                        break
            except self._k8s_client["ApiException"]:
                pass

        return JobInfo(
            job_id=job.metadata.name,
            status=status,
            endpoint=endpoint,
            backend=self.backend_name,
            error_message=error_message,
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

        return self._job_to_info(job, namespace)

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
        """Delete a Kubernetes Job, its Service, and its NetworkPolicy."""
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

        # R-8: delete the egress NetworkPolicy attached to this job.
        try:
            self._k8s_client["networking"].delete_namespaced_network_policy(
                name=f"{job_id}-egress",
                namespace=namespace,
            )
        except self._k8s_client["ApiException"]:
            pass

    async def list_jobs(
        self,
        status_filter: Optional[List[JobStatus]] = None,
    ) -> List[JobInfo]:
        """List all Mini-RiVault Jobs.

        R-12 (Phase 2k): build :class:`JobInfo` directly from the bulk list
        response instead of issuing a per-item ``read_namespaced_job`` call.
        """
        await self._ensure_k8s_client()
        namespace = self.config.k8s_namespace

        try:
            jobs = self._k8s_client["batch"].list_namespaced_job(
                namespace=namespace,
                label_selector="app.kubernetes.io/name=mini-rivault",
            )
        except self._k8s_client["ApiException"] as e:
            raise RuntimeError(f"Failed to list Jobs: {e}")

        result: List[JobInfo] = []
        for job in jobs.items:
            job_info = self._job_to_info(job, namespace)
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
