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
LOOP_WRAPPER_WALL_SECONDS = 2 * 3600  # #SBATCH --time=2:00:00 in the wrapper


def _hmac_hex(secret: str) -> str:
    return secret.encode().hex()


def _gateway_hmac_secret() -> str:
    """UMRS gateway handshake secret for the loop driver (identity envelope)."""
    from umrs.shared.handshake import handshake_secret

    secret = handshake_secret()
    if not secret:
        raise RuntimeError("gateway shared secret not configured on this server")
    return secret.strip()


def _cluster_deploy_dir(site_name: str) -> str:
    """Remote deploy dir for this venue's wrapper stack.

    Resolved the same way RemoteSlurmBackend resolves its job-info dir:
    ``site.yaml`` ``extra.slurm_job_info_dir``'s parent — a path that exists
    on the *remote* submit host, not inside the server container. Falls back
    to the local repo path (unit tests / in-proc submit seams).
    """
    try:
        from ..orchestrator.site_config import load_site_config

        info_dir = (load_site_config().extra or {}).get("slurm_job_info_dir")
        if info_dir:
            return str(Path(str(info_dir)).parent)
    except Exception:
        pass
    repo = os.environ.get("R4_REPO_ROOT", os.getcwd())
    return str(Path(repo) / "deploy" / "rccs")


def _job_identity(job: Job) -> Any:
    """The loop owner's USER_WORK identity here (ADR-112 A.3.3).

    The wrapper job charges the site's ``scheduler.account`` (the same rule
    as ``RemoteSlurmBackend._submit_wrapper_job``), so the attempt record
    names it (I-10)."""
    from umrs.shared.hpc_user_work_access import (
        resolve_job_user_work,
        site_slurm_account,
    )

    return resolve_job_user_work(job, slurm_account=site_slurm_account())


def _platform_login_identity(identity: Any) -> Any:
    """*identity* when it is the platform login, else ``HpcAccessRefused``.

    A loop is user work for admission, but its job environment carries the
    operator's gateway secret (``UMRS_GATEWAY_HMAC_SECRET_HEX``), and plan
    invariant I-2 keeps such jobs on the platform login. Wave A resolves user
    work only to ``service_account``, so this never fires today; once wave B
    resolves ``user_account``, a loop must not carry the secret into a user's
    own job (review 2026-11-08). Raised before any SSH, and terminal: the
    scheduler fails the attempt instead of retrying it (plan B1.4 carry-over,
    I-1: never retried under another identity)."""
    from umrs.orchestrator.hpc_identity import SERVICE_ACCOUNT, HpcAccessRefused

    mode = getattr(identity, "mode", "")
    if mode == SERVICE_ACCOUNT:
        return identity
    raise HpcAccessRefused(
        "mode_undeclared",
        str(getattr(identity, "site", "") or ""),
        f"cluster loop refused: its job carries the platform's gateway secret "
        f"and runs only under the platform login, not access mode {mode!r} "
        f"(ADR-112 invariant I-2)",
    )


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
        identity: Callable[[Job], Any] | None = None,
    ) -> None:
        self._submit = submit_wrapped
        self._repo_root = repo_root or Path(
            os.environ.get("R4_REPO_ROOT") or os.getcwd()
        )
        # identity(job) -> HpcIdentity or raises HpcAccessRefused (ADR-112).
        self._identity = identity or _job_identity

    async def submit(
        self, *, job: Job, attempt: JobAttempt, spec: Mapping[str, Any]
    ) -> str:
        """Submit the loop's driver env-injected into the cluster wrapper.

        A loop driver is user work on the site's login (ADR-112 A.3.3): the
        owner's identity is resolved first, a refusal raises before any SSH,
        and the identity is recorded in ``attempt.metrics["hpc_identity"]``.
        """
        loop_id = str(spec.get("loop_id") or "").strip()
        if not loop_id:
            raise ValueError("cluster-venue loop job requires spec.loop_id")
        identity = _platform_login_identity(self._identity(job))
        attempt.metrics["hpc_identity"] = identity.to_metrics()
        # TODO(MS X-5): the deploy dir is still the rccs one for every site;
        # resolve it from the venue's own site (multi-site plan item X-5).
        deploy_dir = _cluster_deploy_dir("rccs")
        wrapper = str(Path(deploy_dir) / Path(LOOP_WRAPPER_REL).name)
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
        # WP-12 round 2: OTel trace-context propagation across the durable-job
        # boundary — only when the server-side boot is active; never fabricate.
        env.update(_otel_job_env())
        unit = _ledger_pending(job, attempt, identity, loop_id)
        if self._submit is None:
            from umrs.orchestrator.base import OrchestratorConfig
            from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend

            submitter = RemoteSlurmBackend(OrchestratorConfig())
            result = await submitter.submit_job(
                None,
                job_name=unit.tag,
                env={"SLURM_WRAPPER_SCRIPT": wrapper, **env},
            )
        else:
            result = await self._submit(wrapper, env)
        ref = _runtime_ref(result, spec)
        _ledger_submitted(unit, ref)
        return ref


def _loop_wall_seconds() -> int:
    """The CLI walltime override, else the wrapper's real two-hour limit."""
    from umrs.orchestrator.site_config import load_site_config

    from .hpc_ledger import _time_seconds

    try:
        configured = str(load_site_config().scheduler.time_limit or "")
    except (OSError, AttributeError, TypeError, ValueError):
        configured = ""
    return _time_seconds(configured) or LOOP_WRAPPER_WALL_SECONDS


def _ledger_pending(job: Job, attempt: JobAttempt, identity: Any, loop_id: str) -> Any:
    """ADR-112 wave D (plan §6 C/D.0 D2): the loop job's pending ledger row
    before ``sbatch`` (binding; a refusal raises before any SSH). The loop
    wrapper's resources are the site's; the estimate names one CPU."""
    from umrs.hpc_accounts.usage_ledger import estimate, write_pending

    from .hpc_ledger import job_unit, recovery_tag

    tag = recovery_tag("umrs-loop", job.project_id, job.user_id, attempt.attempt_id)
    unit = job_unit(job, attempt, identity, lane="cluster_loop", tag=tag)
    write_pending(unit, estimate(1, 0, _loop_wall_seconds()))
    return unit


def _ref_slurm_id(ref: str) -> str:
    return ref.rsplit("slurm_id=", 1)[-1] if "slurm_id=" in ref else ""


def _ledger_submitted(unit: Any, ref: str) -> None:
    from umrs.hpc_accounts.usage_ledger import write_after

    slurm_id = _ref_slurm_id(ref)
    op, args = (
        ("submitted", (slurm_id, None)) if slurm_id else ("not_submitted", ("sacct",))
    )
    write_after(op, unit, *args)


async def settle_cluster_loop(
    attempt: JobAttempt, job: Job, backend_ref: str, ssh_factory: Any = None
) -> None:
    """Best-effort terminal sacct settlement; loop status remains authoritative."""
    slurm_id = _ref_slurm_id(backend_ref)
    if not slurm_id:
        return
    try:
        from .hpc_ledger import (
            job_unit,
            recovery_tag,
            sacct_command,
            settle_job_from_sacct,
        )
        from .hpc_script import _default_ssh_factory

        factory = ssh_factory or _default_ssh_factory
        ssh = await factory()
        identity = dict(attempt.metrics.get("hpc_identity") or {})
        tag = recovery_tag("umrs-loop", job.project_id, job.user_id, attempt.attempt_id)
        unit = job_unit(job, attempt, identity, lane="cluster_loop", tag=tag)
        row = await ssh.run_remote(sacct_command(slurm_id), timeout=30)
        settle_job_from_sacct(unit, slurm_id, row.stdout)
    except Exception:
        logger.warning(
            "cluster loop: ledger of %s not settled", slurm_id, exc_info=True
        )


def _otel_job_env() -> dict[str, str]:
    """Job-facing OTel env, only when the server's boot is active (WP-12 r2).

    Carries the exporter endpoint for the driver's own boot (when the driver
    image has the otel extra it picks it up the same way), plus the current
    W3C traceparent when a span is in flight so the driver's root span links
    to this job's planning span. Inactive/missing stack → empty dict.
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return {}
    env = {"OTEL_EXPORTER_OTLP_ENDPOINT": endpoint}
    try:
        from opentelemetry.propagate import inject

        carrier: dict[str, str] = {}
        inject(carrier)
        if carrier.get("traceparent"):
            env["TRACEPARENT"] = carrier["traceparent"]
    except ImportError:
        pass
    return env


def _runtime_ref(result: Any, spec: Mapping[str, Any]) -> str:
    slurm_id = ""
    if result is not None:
        slurm_id = str(getattr(result, "job_id", "") or getattr(result, "id", "") or "")
    return f"{REF_PREFIX}loop_id={spec['loop_id']}|driver=slurm|slurm_id={slurm_id}"
