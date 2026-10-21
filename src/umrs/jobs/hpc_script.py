"""HPC_SCRIPT durable job kind — the companion's sbatch path (ROADMAP 3.3, D-37).

The design: the client never talks to a scheduler. The companion's
``hpc_submit`` hands the script to UMRS as a durable job; the server
submits it on the login node via the SSH ControlMaster and owns the
lifecycle mapping (squeue/sacct → attempt states).

Ref grammar: ``hpc_script|script=<path>|slurm=<job-id>`` (slurm is the
scheduler-side id; the durable job id remains the caller's handle).
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from typing import Any

from .models import Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker

_SLURM_SUBMIT_RE = re.compile(r"Submitted batch job (\d+)")

# squeue states → running (anything not terminal). sacct states → terminal map.
_SACCT_TERMINAL = {
    "COMPLETED": "succeeded",
    "FAILED": "failed",
    "CANCELLED": "failed",
    "CANCELLED+": "failed",
    "TIMEOUT": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "PREEMPTED": "failed",
}


def _valid_sbatch_args(args: object) -> bool:
    return isinstance(args, list) and all(
        isinstance(a, str) and a.strip() for a in args
    )


def validate_hpc_script_spec(spec: dict) -> dict:
    """spec: {script_path: str (required), sbatch_args: [str] (optional)}."""
    if not isinstance(spec, dict):
        raise TypeError("hpc_script spec must be an object")
    script_path = str(spec.get("script_path") or "").strip()
    if not script_path:
        raise ValueError("hpc_script spec requires script_path")
    args = spec.get("sbatch_args") or []
    if not _valid_sbatch_args(args):
        raise ValueError("hpc_script sbatch_args must be a list of non-empty strings")
    return {"script_path": script_path, "sbatch_args": [str(a) for a in args]}


def _sbatch_command(script_path: str, sbatch_args: list[str]) -> str:
    """Quoted sbatch command — never shell-interpolate raw user text."""
    return " ".join(
        ["sbatch", *[shlex.quote(a) for a in sbatch_args], shlex.quote(script_path)]
    )


class HpcScriptBackend:
    """Submit a user sbatch script via the SSH ControlMaster."""

    def __init__(self, ssh=None) -> None:
        # ssh is an awaitable factory returning a connected SSHControlManager
        # (injected by the adapters registry; None → named degraded ref).
        self._ssh_factory = ssh

    @property
    def backend_name(self) -> str:
        return "hpc_script"

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        spec = validate_hpc_script_spec(dict(job.spec or {}))
        script_path = spec["script_path"]
        if self._ssh_factory is None:
            return f"hpc_script|script={script_path}|degraded=no_ssh"
        ssh = self._ssh_factory
        if not callable(ssh):
            return f"hpc_script|script={script_path}|degraded=no_ssh"
        ssh = await ssh()
        result = await ssh.run_remote(
            _sbatch_command(script_path, spec["sbatch_args"]), timeout=60
        )
        if not result.ok:
            raise RuntimeError(
                f"sbatch failed (rc={result.returncode}): {result.stderr.strip()[:500]}"
            )
        m = _SLURM_SUBMIT_RE.search(result.stdout)
        if not m:
            raise RuntimeError(
                f"sbatch produced no job id: {result.stdout.strip()[:300]}"
            )
        slurm_id = m.group(1)
        attempt.metrics["slurm_job_id"] = slurm_id
        return f"hpc_script|script={script_path}|slurm={slurm_id}"


def _parse_ref(ref: str) -> tuple[str, str]:
    """'hpc_script|script=…|slurm=NNN' → (script, slurm_id)."""
    script, slurm = "", ""
    for part in ref.split("|")[1:]:
        if part.startswith("script="):
            script = part[len("script=") :]
        elif part.startswith("slurm="):
            slurm = part[len("slurm=") :]
    return script, slurm


async def _slurm_state(ssh, slurm_id: str) -> str:
    """squeue → sacct state for one slurm job id; '' when unknown everywhere."""
    q = await ssh.run_remote(
        f"squeue -h -j {shlex.quote(slurm_id)} -o %T 2>/dev/null", timeout=30
    )
    state = q.stdout.strip().splitlines()[0].strip() if q.stdout.strip() else ""
    if state:
        return state
    a = await ssh.run_remote(
        f"sacct -n -j {shlex.quote(slurm_id)} -o State 2>/dev/null | head -1",
        timeout=30,
    )
    return a.stdout.strip().splitlines()[0].strip() if a.stdout.strip() else ""


class HpcScriptStatusChecker(BackendStatusChecker):
    """squeue/sacct → attempt states (the scheduler owns the truth)."""

    def __init__(self, ssh_factory=None) -> None:
        self._ssh_factory = ssh_factory

    @property
    def job_type(self) -> JobType:
        return JobType.HPC_SCRIPT

    async def check_status(self, attempt: JobAttempt, job: Job) -> AttemptCheckResult:
        ref = attempt.backend_ref or ""
        if not ref.startswith("hpc_script|"):
            return AttemptCheckResult.failed("Unexpected backend reference")
        if "degraded=" in ref or self._ssh_factory is None:
            return AttemptCheckResult.failed(
                "hpc_script submitted without an SSH path — unknown scheduler state"
            )
        _, slurm_id = _parse_ref(ref)
        if not slurm_id:
            return AttemptCheckResult.failed("No slurm job id in the backend ref")
        ssh = await self._ssh_factory()
        state = await _slurm_state(ssh, slurm_id)
        if state in ("", "UNKNOWN"):
            return AttemptCheckResult.failed(
                f"Slurm job {slurm_id} not found in squeue/sacct (state unknown)"
            )
        if state == "COMPLETED":
            return AttemptCheckResult.succeeded(
                metrics={"slurm_state": state, "slurm_job_id": slurm_id}
            )
        if state in _SACCT_TERMINAL:
            failure = AttemptCheckResult.failed(f"slurm state {state}")
            failure.metrics = {"slurm_state": state, "slurm_job_id": slurm_id}
            return failure
        return AttemptCheckResult.running(detail=f"slurm state {state}")


async def _default_ssh_factory():
    """Connected SSHControlManager from site config (the login node)."""
    from umrs.orchestrator.base import OrchestratorConfig
    from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend

    backend = RemoteSlurmBackend(OrchestratorConfig())
    return await backend._get_ssh()


def build_hpc_script_services(ssh_factory: Any = None) -> Mapping[str, Any]:
    """Adapter-registry entry: backend + checker sharing one ssh factory."""
    factory = ssh_factory or _default_ssh_factory
    return {
        "backend": HpcScriptBackend(ssh=factory),
        "checker": HpcScriptStatusChecker(ssh_factory=factory),
    }
