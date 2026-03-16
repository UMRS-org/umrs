"""
Kubernetes step runner for user-defined workflow steps (U-206c).

Executes UMRSStep instances as Kubernetes Jobs.  Follows the same
synchronous interface as ``SubprocessSandbox`` and ``SlurmStepRunner``
so it can be used interchangeably in ``WorkflowExecutor``.

Design:
  * Creates a K8s Job with the worker script + params JSON as a ConfigMap
  * Mounts user script directory via ``hostPath`` (or PVC, configurable)
  * Polls ``read_namespaced_job_status`` for completion
  * Reads back the result JSON from S3 (the worker uploads it)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from umrs.steps.sandbox import StepAuditLog, _WORKER_SCRIPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class K8sStepConfig:
    """Configuration for the Kubernetes step runner."""

    namespace: str = os.environ.get("UDM_K8S_NAMESPACE", "umrs-steps")
    image: str = os.environ.get(
        "UDM_K8S_IMAGE", "ghcr.io/rivault/umrs-step-runner:latest"
    )
    service_account: str = os.environ.get("UDM_K8S_SA", "umrs-step-runner")
    timeout_seconds: int = 3600
    memory_limit: str = "8Gi"
    cpu_limit: str = "4"
    gpu_count: int = 0
    poll_interval: int = 5
    ttl_after_finished: int = 3600  # auto-cleanup after 1h
    audit_dir: Optional[str] = None
    # Set to a PVC claim name to mount user code from a PVC instead of hostPath
    user_code_pvc: Optional[str] = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class K8sStepRunner:
    """Execute a single UMRSStep as a Kubernetes Job."""

    def __init__(self, config: Optional[K8sStepConfig] = None) -> None:
        self.cfg = config or K8sStepConfig()
        self.audit = StepAuditLog(self.cfg.audit_dir)
        self._k8s_loaded = False

    def _ensure_k8s(self):
        """Lazy-load kubernetes client."""
        if self._k8s_loaded:
            return
        try:
            from kubernetes import client, config as k8s_config

            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                k8s_config.load_kube_config()

            self._client = client
            self._batch_api = client.BatchV1Api()
            self._core_api = client.CoreV1Api()
            self._k8s_loaded = True
        except ImportError:
            raise RuntimeError(
                "kubernetes package required. Install with: "
                "pip install kubernetes"
            )

    # ------------------------------------------------------------------

    def run_step(
        self,
        *,
        entrypoint: str,
        step_config: Dict[str, Any],
        script_dir: str,
        input_bucket: str,
        input_keys: List[str],
        output_bucket: str,
        s3_params: Dict[str, str],
        run_id: str = "",
        shared_metadata: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> "StepResult":
        """Run one step as a Kubernetes Job.

        Same signature as :meth:`SubprocessSandbox.run_step`.
        """
        from umrs.steps.context import StepResult

        self._ensure_k8s()
        client = self._client

        run_id = run_id or os.urandom(6).hex()
        job_name = f"udm-step-{run_id[:12]}"

        # -- Build params dict for the worker script -----------------------
        # The worker writes result.json to /workspace, and we read it back
        # via a shared emptyDir volume.  However, if the pod dies we lose
        # the data, so the worker *also* uploads the result to S3 at a known
        # staging key.
        result_s3_key = f"_staging/{run_id}/_result.json"
        params = {
            "entrypoint": entrypoint,
            "config": step_config,
            "script_dir": "/user_code",
            "umrs_src_dir": "/opt/umrs_src",
            "input_bucket": input_bucket,
            "input_keys": input_keys,
            "output_bucket": output_bucket,
            "s3": s3_params,
            "run_id": run_id,
            "metadata": shared_metadata or {},
            "memory_limit_bytes": 0,
            "env": env or {},
            "result_path": "/workspace/result.json",
        }

        # -- Create ConfigMap with worker script + params ------------------
        cm_name = f"{job_name}-cfg"
        cm = client.V1ConfigMap(
            metadata=client.V1ObjectMeta(
                name=cm_name, namespace=self.cfg.namespace,
            ),
            data={
                "_worker.py": _WORKER_SCRIPT,
                "params.json": json.dumps(params),
            },
        )
        self._core_api.create_namespaced_config_map(
            body=cm, namespace=self.cfg.namespace,
        )

        # -- Build job manifest --------------------------------------------
        job = self._build_job(job_name, cm_name, script_dir, env or {})

        self.audit.log({
            "event": "k8s_step_submit",
            "run_id": run_id,
            "entrypoint": entrypoint,
            "job_name": job_name,
            "namespace": self.cfg.namespace,
        })

        # -- Submit --------------------------------------------------------
        t0 = time.monotonic()
        self._batch_api.create_namespaced_job(
            body=job, namespace=self.cfg.namespace,
        )
        logger.info(
            "K8s job %s created for step '%s' (run=%s)",
            job_name, entrypoint, run_id,
        )

        # -- Poll ----------------------------------------------------------
        try:
            final_ok = self._poll(job_name)
        except TimeoutError:
            self._cleanup(job_name, cm_name)
            elapsed = time.monotonic() - t0
            self.audit.log({
                "event": "k8s_step_timeout",
                "run_id": run_id,
                "job_name": job_name,
                "elapsed_seconds": elapsed,
            })
            raise RuntimeError(
                f"K8s step '{entrypoint}' timed out after "
                f"{self.cfg.timeout_seconds}s (job {job_name})"
            )

        elapsed = time.monotonic() - t0

        if not final_ok:
            logs = self._pod_logs(job_name)
            self._cleanup(job_name, cm_name)
            self.audit.log({
                "event": "k8s_step_failed",
                "run_id": run_id,
                "job_name": job_name,
                "elapsed_seconds": elapsed,
            })
            raise RuntimeError(
                f"K8s step '{entrypoint}' failed "
                f"(job {job_name}):\n{logs[-4000:]}"
            )

        # -- Collect result ------------------------------------------------
        # Try reading from pod logs or the S3 staging key.
        # The worker writes to /workspace/result.json but since emptyDir
        # is ephemeral, we rely on pod logs for the result JSON.
        raw = self._collect_result(job_name, s3_params, output_bucket, result_s3_key)
        self._cleanup(job_name, cm_name)

        embeddings = None
        if "embeddings_b64" in raw:
            import base64
            import io
            import numpy as np
            buf = io.BytesIO(base64.b64decode(raw["embeddings_b64"]))
            embeddings = np.load(buf, allow_pickle=False)

        self.audit.log({
            "event": "k8s_step_succeeded",
            "run_id": run_id,
            "job_name": job_name,
            "output_keys_count": len(raw.get("output_keys", [])),
            "elapsed_seconds": elapsed,
        })

        return StepResult(
            output_keys=raw.get("output_keys", []),
            embeddings=embeddings,
            checksums=raw.get("checksums", []),
            metadata=raw.get("metadata", []),
            extra=raw.get("extra", {}),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_job(
        self,
        job_name: str,
        cm_name: str,
        script_dir: str,
        env: Dict[str, str],
    ) -> Any:
        """Construct the K8s Job manifest."""
        client = self._client

        # Environment
        env_list = [client.V1EnvVar(name=k, value=v) for k, v in env.items()]

        # Resources
        resources = client.V1ResourceRequirements(
            requests={"cpu": self.cfg.cpu_limit, "memory": self.cfg.memory_limit},
            limits={"cpu": self.cfg.cpu_limit, "memory": self.cfg.memory_limit},
        )
        if self.cfg.gpu_count > 0:
            resources.limits["nvidia.com/gpu"] = str(self.cfg.gpu_count)
            resources.requests["nvidia.com/gpu"] = str(self.cfg.gpu_count)

        # Volume mounts
        mounts = [
            client.V1VolumeMount(
                name="workspace", mount_path="/workspace",
            ),
            client.V1VolumeMount(
                name="worker-cfg", mount_path="/opt/worker",
                read_only=True,
            ),
        ]
        volumes = [
            client.V1Volume(
                name="workspace",
                empty_dir=client.V1EmptyDirVolumeSource(),
            ),
            client.V1Volume(
                name="worker-cfg",
                config_map=client.V1ConfigMapVolumeSource(name=cm_name),
            ),
        ]

        # User code mount: PVC or hostPath
        if self.cfg.user_code_pvc:
            mounts.append(client.V1VolumeMount(
                name="user-code", mount_path="/user_code", read_only=True,
            ))
            volumes.append(client.V1Volume(
                name="user-code",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name=self.cfg.user_code_pvc, read_only=True,
                ),
            ))
        else:
            mounts.append(client.V1VolumeMount(
                name="user-code", mount_path="/user_code", read_only=True,
            ))
            volumes.append(client.V1Volume(
                name="user-code",
                host_path=client.V1HostPathVolumeSource(
                    path=script_dir, type="Directory",
                ),
            ))

        container = client.V1Container(
            name="step",
            image=self.cfg.image,
            command=["python", "/opt/worker/_worker.py", "/opt/worker/params.json"],
            env=env_list,
            resources=resources,
            volume_mounts=mounts,
            security_context=client.V1SecurityContext(
                run_as_non_root=True,
                allow_privilege_escalation=False,
                capabilities=client.V1Capabilities(drop=["ALL"]),
            ),
        )

        pod_spec = client.V1PodSpec(
            containers=[container],
            volumes=volumes,
            service_account_name=self.cfg.service_account,
            restart_policy="Never",
            active_deadline_seconds=self.cfg.timeout_seconds,
        )

        return client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name, namespace=self.cfg.namespace,
                labels={"app": "umrs-step-runner", "run-id": job_name},
            ),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": "umrs-step-runner", "run-id": job_name},
                    ),
                    spec=pod_spec,
                ),
                backoff_limit=0,
                ttl_seconds_after_finished=self.cfg.ttl_after_finished,
            ),
        )

    def _poll(self, job_name: str) -> bool:
        """Block until K8s Job finishes.  Returns True on success."""
        deadline = time.monotonic() + self.cfg.timeout_seconds + 300
        while time.monotonic() < deadline:
            job = self._batch_api.read_namespaced_job_status(
                name=job_name, namespace=self.cfg.namespace,
            )
            if job.status.succeeded:
                return True
            if job.status.failed:
                return False
            time.sleep(self.cfg.poll_interval)
        raise TimeoutError(f"K8s job {job_name} did not finish in time")

    def _pod_logs(self, job_name: str) -> str:
        """Fetch pod logs for the given job."""
        try:
            pods = self._core_api.list_namespaced_pod(
                namespace=self.cfg.namespace,
                label_selector=f"job-name={job_name}",
            )
            if pods.items:
                return self._core_api.read_namespaced_pod_log(
                    name=pods.items[0].metadata.name,
                    namespace=self.cfg.namespace,
                    tail_lines=500,
                )
        except Exception:
            logger.debug("Could not fetch pod logs for %s", job_name, exc_info=True)
        return ""

    def _collect_result(
        self,
        job_name: str,
        s3_params: Dict[str, str],
        bucket: str,
        result_key: str,
    ) -> Dict[str, Any]:
        """Read the result JSON.  Tries pod stdout first, then S3."""
        # Option 1: parse the last JSON object from pod logs
        logs = self._pod_logs(job_name)
        # The worker prints nothing to stdout; result is in /workspace/result.json.
        # Since emptyDir survives until pod is deleted, read from pod filesystem
        # via an exec or from S3.  Simplest: read the file from the pod.
        try:
            pods = self._core_api.list_namespaced_pod(
                namespace=self.cfg.namespace,
                label_selector=f"job-name={job_name}",
            )
            if pods.items:
                from kubernetes.stream import stream
                resp = stream(
                    self._core_api.connect_get_namespaced_pod_exec,
                    pods.items[0].metadata.name,
                    self.cfg.namespace,
                    command=["cat", "/workspace/result.json"],
                    stderr=True, stdin=False, stdout=True, tty=False,
                )
                return json.loads(resp)
        except Exception:
            logger.debug("Pod exec read failed, trying S3", exc_info=True)

        # Option 2: read from S3 staging key (if worker uploaded it)
        try:
            import boto3
            import botocore.config
            s3 = boto3.client(
                "s3",
                endpoint_url=s3_params["endpoint_url"],
                aws_access_key_id=s3_params["access_key"],
                aws_secret_access_key=s3_params["secret_key"],
                config=botocore.config.Config(signature_version="s3v4"),
            )
            obj = s3.get_object(Bucket=bucket, Key=result_key)
            return json.loads(obj["Body"].read())
        except Exception:
            pass

        raise RuntimeError(
            f"Could not retrieve result for K8s job {job_name}"
        )

    def _cleanup(self, job_name: str, cm_name: str) -> None:
        """Best-effort cleanup of job + configmap."""
        try:
            self._batch_api.delete_namespaced_job(
                name=job_name, namespace=self.cfg.namespace,
                propagation_policy="Background",
            )
        except Exception:
            pass
        try:
            self._core_api.delete_namespaced_config_map(
                name=cm_name, namespace=self.cfg.namespace,
            )
        except Exception:
            pass
