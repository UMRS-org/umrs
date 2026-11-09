"""The usage ledger's write points on the service staged-step lane (ADR-112
wave D; plan §6 C/D.0 D2, "service staged step").

A step on the platform login (``service_account``) is user work too: its
job is ledgered pending before the helper's ``submit`` (from the routed
resources; the walltime the job asks), submitted after the job id, and
settled at the existing ``sacct`` measurement (source ``sacct``). A pending
write the ledger refuses (budget, or a database that cannot record) fails
the step by name before anything is submitted; later writes never fail a
step (the sweep recovers them). The runner hands the run's owner and
project in the step metadata (:data:`LEDGER_METADATA_KEY`).
"""

from __future__ import annotations

from typing import Any

from .step_executor import StepExecutionError

#: ``{"person", "project_id"}`` of the run, set by the workflow runner.
LEDGER_METADATA_KEY = "hpc_ledger"


class LedgerRefusedStep(StepExecutionError):
    """A step the ledger refused before submit (``hpcaccess.budget_exceeded``
    or ``hpcaccess.ledger_unavailable``). Final: the runner does not retry
    it and the run stops (K14)."""

    final = True


def _venue() -> str:
    from umrs.shared.hpc_user_work_access import current_site

    return str(getattr(current_site(), "name", "") or "")


def staged_attempt_id(job: Any) -> str:
    """A fresh dispatch's opaque id; the Slurm integer attempt stays unchanged."""
    base = f"{job.run_id}/{job.step}/{job.attempt}"
    dispatch = str(getattr(job, "dispatch_id", "") or "")
    return f"{base}/{dispatch}" if dispatch else base  # legacy reconstructed jobs


def service_unit(job: Any, ctx: Any) -> Any:
    """The ledger unit of one service-lane attempt."""
    from umrs.hpc_accounts.usage_ledger import LedgerUnit

    owner = dict((ctx.metadata or {}).get(LEDGER_METADATA_KEY) or {})
    return LedgerUnit(
        venue=_venue(),
        attempt_id=staged_attempt_id(job),
        person=str(owner.get("person") or ""),
        lane="staged_step",
        tag=job.job_name,
        access_mode="service_account",
        method="platform",
        run_id=job.run_id,
        step=job.step,
        project_id=str(owner.get("project_id") or ""),
        site_login=job.site_login,
        slurm_account=job.slurm_account,
    )


def service_pending(
    job: Any, ctx: Any, cpus: int, gpus: int, seconds: int, *, partition: Any = None
) -> None:
    """The binding write before ``submit``; a refusal sends nothing."""
    from umrs.hpc_accounts.usage_ledger import (
        LedgerRefused,
        estimate,
        requested_partition,
        write_pending,
    )

    priced = requested_partition(partition, cpus=cpus, gpus=gpus)
    try:
        write_pending(
            service_unit(job, ctx),
            estimate(cpus, gpus, seconds, priced),
            estimate_complete=gpus == 0 or int(seconds) > 0,
        )
    except LedgerRefused as exc:
        raise LedgerRefusedStep("setup", f"{exc}; nothing was submitted") from exc


def service_after(op: str, job: Any, ctx: Any, *args: Any) -> None:
    from umrs.hpc_accounts.usage_ledger import write_after, write_job_after

    unit = service_unit(job, ctx)
    if op == "submitted":
        write_job_after(op, unit, str(job.job_id), *args[1:])
    elif op == "settled":
        write_job_after(op, unit, str(job.job_id), *args)
    else:
        write_after(op, unit, *args)


def service_settle(job: Any, ctx: Any, measured: dict[str, Any], cpus: int) -> None:
    """Settle at the lane's ``sacct`` measurement (a COMPLETED job; the
    requested CPUs, the measured GPUs and elapsed seconds)."""
    elapsed = measured.get("elapsed_seconds")
    if elapsed is None:
        return  # unmeasured: the sweep settles it
    usage = {
        "state": "COMPLETED",
        "elapsed_s": int(elapsed),
        "alloc_cpus": int(cpus),
        "alloc_gpus": int(measured.get("gpus") or 0),
        "start": None,
        "end": None,
    }
    service_after("settled", job, ctx, usage, "sacct")


__all__ = [
    "LEDGER_METADATA_KEY",
    "LedgerRefusedStep",
    "service_after",
    "service_pending",
    "service_settle",
    "service_unit",
    "staged_attempt_id",
]
