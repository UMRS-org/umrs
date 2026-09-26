"""Reaper for the connected-HPC step lane (ADR-074; build spec §8).

A step's Slurm job is otherwise reconciled only when the same step of the
same run is submitted again. A run that ends or is refused first — cancelled
or failed upstream after a restart, the lane switched off or its image
failing verification, a settle thread that died with the process, a server
that died between ``mkwd`` and ``submit`` — would leave uploaded code
running to ``--time`` and its workdir on a filesystem without quota. The
reaper ends those independently of re-submission, and of the lane being
enabled (a kill switch must still kill):

- when a run ends or is refused, each step whose record carries a job not
  yet cleaned up (:func:`reap_run_jobs`);
- at boot, the runs from before this process that nothing will reconcile
  (:func:`orphaned_runs`);
- every hour, the workdirs no queued or running job uses that sat idle past
  the lane's longest queue wait + run + transfers (helper ``sweep``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

SWEEP_INTERVAL_SECONDS = 3600.0

_lock = threading.Lock()
_reaper: Any = None
_built = False


def step_lane_reaper() -> Any:
    """The process-wide reaper (a :class:`RemoteSlurmStepExecutor` that is
    never asked to run a step), or None when this server has no remote step
    lane (no SSH target, workdir root or helper)."""
    global _reaper, _built
    with _lock:
        if not _built:
            _reaper, _built = _build_reaper(), True
        return _reaper


def set_step_lane_reaper(reaper: Any) -> None:
    """Install a reaper (tests) or forget it (None: rebuilt on next use)."""
    global _reaper, _built
    with _lock:
        _reaper, _built = reaper, reaper is not None


def _site_step_executor() -> Any:
    try:
        from umrs.orchestrator.site_config import get_site_config

        site = get_site_config()
        site.inject_env(overwrite=False)  # the lane's clocks and caps (UMRS_SLURM_*)
        return site.step_executor
    except Exception:
        logger.debug("site config unavailable for the step lane reaper", exc_info=True)
        return None


def _lane_path(env_name: str, site_field: str, site: Any) -> str:
    """The env value, else site.yaml's: the converge drops the env lines of a
    lane whose image did not verify, and the reaper must still find the
    helper then."""
    value = os.environ.get(env_name, "").strip()
    value = value or str(getattr(site, site_field, "") or "").strip()
    return value if value.startswith("/") else ""


def _build_reaper() -> Any:
    from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor

    site = _site_step_executor()
    helper = _lane_path("UMRS_STEP_HELPER", "helper_path", site)
    root = _lane_path("UMRS_SLURM_WORKDIR_ROOT", "workdir_root", site)
    if not (helper and root):
        return None
    try:
        return RemoteSlurmStepExecutor(helper_path=helper, workdir_root=root)
    except Exception:  # no SSH target (or no site.yaml): no remote lane here
        logger.info(
            "step lane reaper: no remote step lane on this server", exc_info=True
        )
        return None


#: ``StepProgress.extra`` key of a step's Slurm job facts, one entry per
#: workdir (one per attempt, plus an earlier attempt's it adopted): a newer
#: attempt, or an older attempt's late settle, never overwrites another's.
JOBS_KEY = "slurm_jobs"


def step_jobs(step: Any) -> dict[str, dict[str, Any]]:
    """Workdir -> job facts recorded for *step* (a StepProgress)."""
    entries = (getattr(step, "extra", None) or {}).get(JOBS_KEY)
    if not isinstance(entries, dict):
        return {}
    return {str(k): dict(v) for k, v in entries.items() if isinstance(v, dict)}


def _entry(facts: dict[str, Any]) -> str:
    return str(facts.get("workdir") or facts.get("job_id") or "")


def live_record(record: Any) -> Any:
    """The run's current record. A durable retry re-runs the same execution
    id in the same process and replaces the record; what the earlier run
    left behind (a settle thread, a reaper still walking its steps) must
    write to the live record, never persist its stale snapshot over the
    live row."""
    while getattr(record, "superseded_by", None) is not None:
        record = record.superseded_by
    return record


def record_step_job(record: Any, step: str, facts: dict[str, Any]) -> None:
    """Record one job's facts under its workdir (the step lane's sink)."""
    if _entry(facts):
        record.update_step_entry(step, JOBS_KEY, _entry(facts), dict(facts))


def unreaped_steps(record: Any) -> dict[str, list[dict[str, Any]]]:
    """Step name -> the facts of every job of the step that may have left a
    job or a workdir on the cluster."""
    steps: dict[str, list[dict[str, Any]]] = {}
    for name, step in list(getattr(record, "steps", {}).items()):
        left = [f for f in step_jobs(step).values() if not f.get("cleaned")]
        if left:
            steps[name] = left
    return steps


def carry_step_jobs(prior: Any, record: Any) -> None:
    """A durable re-dispatch replaces the run's record: the jobs the earlier
    process left (a step the new run never reaches, a refused run) stay on
    the new record for the reaper."""
    for name, left in unreaped_steps(prior).items():
        for facts in left:
            record_step_job(record, name, facts)


def reap_record(
    reaper: Any,
    record: Any,
    durable_status: Callable[[str], str | None] | None = None,
) -> None:
    """Reap each step of *record* that left something; a step with nothing
    left is recorded ``cleaned`` (the next boot skips it). Stops once a
    durable retry replaced the record: the retry carries its uncleaned jobs
    (:func:`carry_step_jobs`) and reconciles or reaps them itself."""
    for_retry = _may_retry(record, durable_status)
    for name, left in unreaped_steps(record).items():
        if live_record(record) is not record:
            logger.info(
                "step lane: run %s was dispatched again; its reaping is the "
                "new dispatch's",
                record.execution_id,
            )
            return
        try:
            done = reaper.reap_step(
                str(record.execution_id), name, *left, for_retry=for_retry
            )
        except Exception:
            logger.warning(
                "step lane: reaping step %s of run %s failed",
                name,
                record.execution_id,
                exc_info=True,
            )
            continue
        if done:
            _mark_cleaned(record, name, left)


def _may_retry(
    record: Any, durable_status: Callable[[str], str | None] | None = None
) -> bool:
    """A failed (not cancelled) run with a durable job may be dispatched again,
    and that dispatch's reconcile resumes a live job on the same inputs or
    adopts a COMPLETED one's outputs; --time and the workdir sweep bound them
    if no retry comes. Given *durable_status* (at boot), only while the
    scheduler still dispatches that job: a spent one's jobs are reaped."""
    job = getattr(record, "durable_job_id", None)
    if (
        str(getattr(record, "status", "")) != "failed"
        or getattr(record, "cancel_requested", False)
        or not job
    ):
        return False
    return durable_status is None or _job_status(record, durable_status) in _DISPATCHED


def _mark_cleaned(record: Any, step: str, left: list[dict[str, Any]]) -> None:
    """Record the step's jobs cleaned — on the live record, and there only
    the entries it carries, when a retry replaced *record* meanwhile."""
    live = live_record(record)
    held = step_jobs(live.steps.get(step)) if live is not record else None
    for facts in left:
        if held is None or _entry(facts) in held:
            live.update_step_entry(step, JOBS_KEY, _entry(facts), {"cleaned": True})


def reap_run_jobs(record: Any) -> threading.Thread | None:
    """The run ended or was refused: no attempt of its steps will reconcile
    their jobs any more. They are reaped off the caller's thread."""
    if not unreaped_steps(record):
        return None
    reaper = step_lane_reaper()
    if reaper is None:
        logger.warning(
            "step lane: run %s left step jobs, but this server has no path to "
            "the lane's cluster (site.yaml helper_path / workdir_root / ssh_target)",
            record.execution_id,
        )
        return None
    thread = threading.Thread(
        target=reap_record,
        args=(reaper, record),
        name=f"umrs-step-reap-{record.execution_id}",
        daemon=True,
    )
    thread.start()
    return thread


#: Durable job statuses whose job the scheduler still (re-)dispatches.
_DISPATCHED = frozenset({"planned", "queued", "running"})


def orphaned_runs(
    records: Iterable[Any],
    booted_at: float,
    durable_status: Callable[[str], str | None] | None = None,
) -> list[Any]:
    """Runs from before this process whose steps nothing will reconcile:
    ended ones (their settle threads died with the old process), ones
    without a durable job, and — given *durable_status* (the job store's
    status of a durable job, None when there is no such job; read after the
    boot reconciler ran) — open ones whose durable job will not run again:
    its last attempt was lost in the crash, or it was cancelled before the
    step thread's poll sent ``scancel``. An open run whose durable job is
    queued or running is re-dispatched: its steps reconcile, or its refusal
    reaps them. Without *durable_status* (no job store in this process, so
    the run is another process's to dispatch) an open durable run is left
    alone."""
    return [
        r
        for r in records
        if float(r.submitted_at or 0) < booted_at
        and unreaped_steps(r)
        and not _redispatched(r, durable_status)
    ]


def _redispatched(
    record: Any, durable_status: Callable[[str], str | None] | None
) -> bool:
    if record.is_terminal() or not record.durable_job_id:
        return False
    if durable_status is None:
        return True
    try:
        status = durable_status(str(record.durable_job_id))
    except Exception:  # the store is unreadable: leave the run to its dispatch
        logger.warning(
            "step lane: durable job %s of run %s unreadable",
            record.durable_job_id,
            record.execution_id,
            exc_info=True,
        )
        return True
    return str(status or "").lower() in _DISPATCHED


def close_abandoned_run(
    record: Any, durable_status: Callable[[str], str | None] | None
) -> None:
    """An open run nothing will dispatch again ends here: cancelled when it
    was asked to be (or its durable job was), else failed — the next boot
    then sees an ended run. A record a retry replaced is left as it is: its
    update would persist the stale snapshot over the live row."""
    if record.is_terminal() or live_record(record) is not record:
        return
    job, status = record.durable_job_id, _job_status(record, durable_status)
    why = (
        f"its durable job {job} will not run again (job status: {status or 'unknown'})"
        if job
        else "nothing dispatches it again"
    )
    cancelled = bool(record.cancel_requested) or status == "cancelled"
    record.update(
        status="cancelled" if cancelled else "failed",
        error=f"the server restarted during this run and {why}",
        finished_at=time.time(),
    )


def _job_status(record: Any, durable_status: Callable[[str], str | None] | None) -> str:
    """The run's durable job status, lower case ("" when unknown)."""
    if durable_status is None or not record.durable_job_id:
        return ""
    with contextlib.suppress(Exception):
        return str(durable_status(str(record.durable_job_id)) or "").lower()
    return ""


def _list_records(registry: Any) -> list[Any]:
    try:
        return list(registry.list())
    except Exception:
        logger.warning("step lane: execution records unavailable", exc_info=True)
        return []


def reap_loop(
    reaper: Any,
    registry: Any,
    booted_at: float,
    *,
    interval: float = SWEEP_INTERVAL_SECONDS,
    stop: threading.Event | None = None,
    durable_status: Callable[[str], str | None] | None = None,
) -> None:
    """Reap the runs orphaned before *booted_at* (an open one is then ended:
    :func:`close_abandoned_run`), then sweep idle unused workdirs every
    *interval* seconds until *stop* is set."""
    for record in orphaned_runs(_list_records(registry), booted_at, durable_status):
        reap_record(reaper, record, durable_status)
        close_abandoned_run(record, durable_status)
    stop = stop or threading.Event()
    while True:
        try:
            removed = reaper.sweep_workdirs().get("removed") or []
            if removed:
                logger.warning("step lane: swept idle unused workdirs %s", removed)
        except Exception:
            logger.warning("step lane: workdir sweep failed", exc_info=True)
        if stop.wait(interval):
            return


def start_step_lane_reaper(
    registry: Any = None,
    booted_at: float | None = None,
    *,
    interval: float = SWEEP_INTERVAL_SECONDS,
    durable_status: Callable[[str], str | None] | None = None,
) -> threading.Thread | None:
    """Start :func:`reap_loop` in a daemon thread (None: no remote lane).
    *durable_status* is the job store's status of a durable job, read after
    the boot reconciler ran (:func:`orphaned_runs`)."""
    reaper = step_lane_reaper()
    if reaper is None:
        return None
    if registry is None:
        from umrs.server.workflow_execution_registry import get_execution_registry

        registry = get_execution_registry()
    thread = threading.Thread(
        target=reap_loop,
        args=(reaper, registry, time.time() if booted_at is None else booted_at),
        kwargs={"interval": interval, "durable_status": durable_status},
        name="umrs-step-reaper",
        daemon=True,
    )
    thread.start()
    return thread
