"""Cluster-venue submission path for research_loop_job (WP-05 round 2).

When a loop job declares ``venue: "cluster:<name>"``, submit the loop's driver
inside a mini-RiVault/Slurm sandbox via the site's Slurm wrapper stack
(same sbatch/SSH plumbing as endpoint provisioning, but with our loop-job
shim), per-job EgressGuard enabled. The venue handshake gateway shared secret
arrives hex-encoded (never in logs/spec/payload bodies).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from umrs.jobs.models import Job, JobAttempt

from .research_loop_backend import REF_PREFIX

logger = logging.getLogger(__name__)

LOOP_WRAPPER_REL = str(Path("deploy") / "rccs" / "loop-job.slurm")


def _hmac_hex(secret: str) -> str:
    return secret.encode().hex()


def _gateway_hmac_secret() -> str:
    """UMRS gateway handshake secret for the loop driver (identity envelope)."""
    from umrs.api._verified_headers import handshake_secret

    secret = handshake_secret()
    if not secret:
        raise RuntimeError("gateway shared secret not configured on this server")
    return secret.strip()


def _cluster_deploy_dir(site_name: str) -> str:
    """Deploy dir for this venue's wrapper stack (matches endpoint provision)."""
    repo = os.environ.get("R4_REPO_ROOT", os.getcwd())
    return str(Path(repo) / "deploy" / "rccs")


class ClusterSlurmSubmitter:
    """Go through RemoteSlurmBackend to stage the loop job wrapper on cluster.

    Testing seam: ``submit_wrapped`` callable receives (model, wrapper_script, kwargs)
    and returns an arbitrary runtime ref; in production that is RemoteSlurmBackend.
    """

    def __init__(
        self,
        submit_wrapped: Callable[..., Awaitable[Any]] | None = None,
        *,
        repo_root: Path | None = None,
    ) -> None:
        self._submit = submit_wrapped
        self._repo_root = repo_root or Path(os.environ.get("R4_REPO_ROOT") or os.getcwd())

    async def submit(self, *, job: Job, attempt: JobAttempt, spec: Mapping[str, Any]) -> str:
        """Submit the loop's driver env-injected into the cluster wrapper."""
        loop_id = str(spec.get("loop_id") or "").strip()
        if not loop_id:
            raise ValueError("cluster-venue loop job requires spec.loop_id")
        wrapper = str((self._repo_root / LOOP_WRAPPER_REL).resolve())
        env = {
            "LOOP_ID": loop_id,
            "R4_GATEWAY_BASE": os.environ.get(
                "R4_GATEWAY_BASE", "https://data0.ai.r-ccs.riken.jp/rccs"
            ),
            "R4_LOOP_JOB_ID": job.job_id,
            "R4_LOOP_ATTEMPT_ID": attempt.attempt_id,
            "R4_LOOP_OWNER": str(job.user_id),
            "R4_LOOP_PROJECT": job.project_id,
            "UMRS_GATEWAY_HMAC_SECRET_HEX": _hmac_hex(_gateway_hmac_secret()),
            "RIVAULT_WRAPPER_DIR": _cluster_deploy_dir("rccs"),
        }
        if self._submit is None:
            from umrs.orchestrator.orchestrator_config import OrchestratorConfig
            from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend

            submitter = RemoteSlurmBackend(OrchestratorConfig())
            result = await submitter.submit_job(
                None,
                env={"SLURM_WRAPPER_SCRIPT": wrapper, **env},
            )
        else:
            result = await self._submit(wrapper, env)
        return _runtime_ref(result, spec)


def _runtime_ref(result: Any, spec: Mapping[str, Any]) -> str:
    slurm_id = ""
    if result is not None:
        slurm_id = str(getattr(result, "job_id", "") or getattr(result, "id", "") or "")
    return f"{REF_PREFIX}loop_id={spec['loop_id']}|driver=slurm|slurm_id={slurm_id}"
