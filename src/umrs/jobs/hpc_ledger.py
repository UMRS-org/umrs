"""The usage ledger's write points for durable user-work jobs on the
platform login: ``hpc_script`` and cluster loops (ADR-112 wave D; plan §6
C/D.0 D2).

Pending before ``sbatch`` (binding: a budget or ledger refusal raises
before any SSH), submitted after the Slurm job id, settled from the
lifecycle's ``sacct`` row at the terminal state (source ``sacct``). The
estimate of an ``hpc_script`` comes from its own ``sbatch`` arguments
(``--time``, ``--cpus-per-task``, ``--gpus`` / ``--gres=gpu:N``); a value
the arguments do not name counts as one CPU, no GPU, no time.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

_GRES_GPU = re.compile(r"gpu(?::[A-Za-z0-9_-]+)?:(\d+)")
_GPUS = re.compile(r"(?:gpu:)?(?:[A-Za-z0-9_.-]+:)?(\d+)")
SACCT_FIELDS = "JobID,State,Elapsed,AllocTRES,AllocCPUS,Start,End"
_SACCT_KEYS = ("job_id", "state", "elapsed", "alloc_tres", "alloc_cpus", "start", "end")


def _value(args: list[str], flags: tuple[str, ...]) -> str | None:
    from .hpc_script import _flag_value

    found = None
    for index in range(len(args)):
        found = _flag_value(args, index, flags) or found
    return found


def _number_parts(text: str) -> list[int] | None:
    parts = str(text or "").strip().split(":")
    if not 1 <= len(parts) <= 3 or any(not part.isdigit() for part in parts):
        return None
    return [int(part) for part in parts]


def _hms_seconds(parts: list[int] | None) -> int | None:
    if parts is None:
        return None
    hours, minutes, seconds = (parts + [0, 0, 0])[:3]
    if minutes >= 60 or seconds >= 60:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _day_time_seconds(days: str, clock: str) -> int:
    seconds = _hms_seconds(_number_parts(clock))
    if not days.isdigit() or seconds is None:
        return 0
    return int(days) * 86400 + seconds


def _time_seconds(text: str | None) -> int:
    """The full Slurm time grammar: M, M:S, H:M:S and D-H[:M[:S]]."""
    raw = str(text or "").strip()
    if raw.isdigit():
        return int(raw) * 60
    days, separator, clock = raw.partition("-")
    if separator:
        return _day_time_seconds(days, clock)
    parts = _number_parts(raw)
    if parts is None:
        return 0
    if len(parts) == 2:  # M:S (not H:M, which is ambiguous with day syntax)
        minutes, seconds = parts
        return minutes * 60 + seconds if seconds < 60 else 0
    return int(_hms_seconds(parts) or 0)


def _multiplicity(text: str) -> int | None:
    raw = str(text or "1").strip()
    if raw.isdigit():
        return int(raw) if int(raw) > 0 else None
    bounds = raw.split("-", 1)
    if len(bounds) == 2 and all(bound.isdigit() for bound in bounds):
        low, high = (int(bound) for bound in bounds)
        return high if 0 < low <= high else None
    return None


def _multiplied_gpus(value: str, multiplier_text: str) -> tuple[int, bool]:
    match = _GPUS.fullmatch(value.strip())
    multiplier = _multiplicity(multiplier_text)
    if match is None or multiplier is None:
        return 0, False
    return int(match.group(1)) * multiplier, True


def _gpu_resources(args: list[str]) -> tuple[int, bool]:
    """Requested GPU count and whether its total is bounded."""
    direct = _value(args, ("--gpus", "-G"))
    if direct is not None:
        return _multiplied_gpus(direct, "1")
    per_node = _value(args, ("--gpus-per-node",))
    if per_node is not None:
        nodes = _value(args, ("--nodes", "-N")) or "1"
        return _multiplied_gpus(per_node, nodes)
    per_task = _value(args, ("--gpus-per-task",))
    if per_task is not None:
        tasks = _value(args, ("--ntasks", "-n")) or "1"
        return _multiplied_gpus(per_task, tasks)
    gres = _value(args, ("--gres",))
    if gres is None:
        return 0, True
    match = _GRES_GPU.fullmatch(gres.strip())
    return (int(match.group(1)), True) if match else (0, False)


def _script_partition(args: list[str]) -> Any:
    from umrs.orchestrator import partition_choice as pc
    from umrs.shared.hpc_user_work_access import current_site

    site = current_site()
    if site is None:
        return None
    step_executor = getattr(site, "step_executor", None)
    configured = str(getattr(step_executor, "partition", "") or "")
    scheduler = getattr(site, "scheduler", None)
    configured = configured or str(getattr(scheduler, "partition", "") or "")
    name = (
        _value(args, ("--partition", "-p"))
        or (pc.split_partitions(configured) or [configured])[0]
    )
    getter = getattr(site, "get_partition_profile", None)
    if callable(getter):
        return getter(name)
    return (getattr(site, "partition_profiles", {}) or {}).get(name)


def _script_pricing(args: list[str], cpus: int, gpus: int) -> dict[str, Any]:
    from umrs.hpc_accounts.usage_ledger import requested_partition

    return requested_partition(_script_partition(args), cpus=cpus, gpus=gpus)


def script_estimate_complete(sbatch_args: list[str]) -> bool:
    """Whether ``sbatch_args`` bound the GPU-seconds the job may consume."""
    args = [str(a) for a in sbatch_args]
    gpus, complete = _gpu_resources(args)
    if gpus > 0:
        complete = complete and _time_seconds(_value(args, ("--time", "-t"))) > 0
    exclusive = "--exclusive" in args or _script_pricing(args, 1, gpus)["exclusive"]
    if exclusive and gpus > 0:
        complete = complete and _script_pricing(args, 1, gpus).get("max_gpus", 0) > 0
    return complete


def script_estimate(sbatch_args: list[str]) -> Any:
    """Resources an ``hpc_script`` asks for, from its arguments and K4 profile."""
    from umrs.hpc_accounts.usage_ledger import estimate

    args = [str(a) for a in sbatch_args]
    cpus = _value(args, ("--cpus-per-task", "-c")) or "1"
    gpus = _gpu_resources(args)[0]
    seconds = _time_seconds(_value(args, ("--time", "-t")))
    cpu_count = int(cpus) if cpus.isdigit() else 1
    return estimate(cpu_count, gpus, seconds, _script_pricing(args, cpu_count, gpus))


def recovery_tag(prefix: str, *identity: Any) -> str:
    """A bounded Slurm name unique across jobs, attempts, users and projects."""
    material = "\0".join(str(part) for part in identity).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()[:24]
    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", prefix).strip("-") or "umrs-job"
    return f"{safe}-{digest}"


def job_unit(job: Any, attempt: Any, identity: Any, *, lane: str, tag: str) -> Any:
    """The ledger unit of one durable job attempt on the platform login."""
    from umrs.hpc_accounts.usage_ledger import unit_for

    return unit_for(
        identity,
        attempt_id=str(attempt.attempt_id),
        person=str(job.user_id or ""),
        lane=lane,
        tag=tag[:200],
        job_id=str(job.job_id),
        project_id=str(job.project_id or ""),
    )


def script_tag(spec: dict[str, Any]) -> str:
    """The Slurm job name of an ``hpc_script`` (``--job-name``, else the
    script's file name, as ``sbatch`` names it)."""
    args = [str(a) for a in spec.get("sbatch_args") or []]
    return (
        _value(args, ("--job-name", "-J"))
        or os.path.basename(str(spec.get("script_path") or ""))
        or "hpc_script"
    )


def sacct_command(slurm_id: str) -> str:
    import shlex

    return f"sacct -n -X -P -j {shlex.quote(slurm_id)} -o {SACCT_FIELDS} 2>/dev/null | head -1"


def sacct_row(text: str) -> dict[str, str]:
    """One ``sacct -P`` line as the gate's ``usage`` row shape."""
    fields = (text.strip().splitlines() or [""])[0].split("|")
    return dict(zip(_SACCT_KEYS, fields)) if len(fields) == len(_SACCT_KEYS) else {}


def settle_from_sacct(unit: Any, text: str) -> None:
    """Settle *unit* from a ``sacct`` line when it is terminal and measured."""
    from umrs.hpc_accounts.usage_ledger import usage_from_sacct, write_after

    usage = usage_from_sacct(sacct_row(text))
    if usage is not None:
        write_after("settled", unit, usage, "sacct")


def settle_job_from_sacct(unit: Any, slurm_id: str, text: str) -> None:
    """Settle through the original scheduler owner, including adopted jobs."""
    from umrs.hpc_accounts.usage_ledger import usage_from_sacct, write_job_after

    usage = usage_from_sacct(sacct_row(text))
    if usage is not None:
        write_job_after("settled", unit, slurm_id, usage, "sacct")


__all__ = [
    "SACCT_FIELDS",
    "job_unit",
    "recovery_tag",
    "sacct_command",
    "sacct_row",
    "script_estimate",
    "script_estimate_complete",
    "script_tag",
    "settle_from_sacct",
    "settle_job_from_sacct",
]
