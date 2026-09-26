"""Step-lane smoke — receipt R2 of the rccs step lane (spec §11).

Runs the platform probe step (:mod:`umrs.server.step_lane_probe`) through
the real lane on each named partition, and once with a GPU: helper,
staging, image load and ``.id`` check, hardened pod, JSON reply, collect,
cleanup. A run passes when the step ran as a non-root user on a read-only
root filesystem, its environment held only the four staged ``UMRS_STEP_*``
names beside the image's own, no target connection opened, and the GPU run
saw ``/dev/nvidia*``.

Owner-invoked by the converge (``umrs_front_<site>_step_smoke``) inside the
site's server container. It forces the lane on for this process only; the
server's ``remote_enabled`` is untouched. Prints one JSON line per run and a
summary line; exit 0 when every run passed, 1 otherwise, 2 on bad usage.

    python -m umrs.server.step_lane_smoke --partitions p1,p2 \\
        [--gpu-partition p1] [--connect host:port ...]
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

PROBE_ENTRYPOINT = "umrs.server.step_lane_probe:ProbeStep"

#: The staged job script's ``-e`` (slurm_step_executor ``_STAGED_ENV``).
STAGED_ENV = frozenset(
    {
        "UMRS_STEP_INPUT",
        "UMRS_STEP_OUTPUT_JSON",
        "UMRS_STEP_STAGED",
        "UMRS_STEP_SOURCE_DIR",
    }
)
#: What the step image (python:3.11-slim + Containerfile.step) and podman
#: put in every container's environment.
IMAGE_ENV = frozenset(
    {
        "PATH",
        "LANG",
        "GPG_KEY",
        "PYTHON_VERSION",
        "PYTHON_SHA256",
        "PYTHONUNBUFFERED",
        "PYTHONDONTWRITEBYTECODE",
        "HOME",
        "HOSTNAME",
        "TERM",
        "container",
    }
)


class _RunStore:
    """The run's in-memory object store, as the runner's (no
    ``upload_fileobj``: outputs are buffered)."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def head_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        body = self.objects[(Bucket, Key)]
        return {"ContentLength": len(body), "ETag": '"smoke"'}

    def get_object(self, *, Bucket: str, Key: str, **_: Any) -> dict[str, Any]:
        body = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ContentLength": len(body)}

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **_: Any) -> None:
        self.objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.read()

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str = "", **_: Any
    ) -> dict[str, Any]:
        keys = sorted(
            k for b, k in self.objects if b == Bucket and k.startswith(Prefix)
        )
        return {
            "Contents": [
                {"Key": k, "Size": len(self.objects[(Bucket, k)])} for k in keys
            ]
        }

    def stored_bytes(self) -> int:
        return sum(len(v) for v in self.objects.values())


def check(probe: dict[str, Any], *, gpu: bool, nproc_limit: int = 0) -> list[str]:
    """What is wrong with one run's probe facts ([] when it passed);
    *nproc_limit* is the site's ``--ulimit nproc`` (0: not set)."""
    nproc = probe.get("nproc_soft")
    found = [
        _identity_problem(probe),
        f"RLIMIT_NPROC is {nproc}, not the site's {nproc_limit}"
        if nproc_limit > 0 and nproc != nproc_limit
        else "",
        _env_problem(probe, "unexpected environment names", extra=True),
        _env_problem(probe, "staged names missing", extra=False),
        _connect_problem(probe),
        "the GPU run saw no /dev/nvidia*"
        if gpu and not probe.get("gpu_devices")
        else "",
    ]
    return [p for p in found if p]


def _identity_problem(probe: dict[str, Any]) -> str:
    if probe.get("uid") in (0, None):
        return f"the step ran as uid {probe.get('uid')}"
    if probe.get("rootfs_read_only") is not True:
        return "the root filesystem is writable"
    return ""


def _env_problem(probe: dict[str, Any], what: str, *, extra: bool) -> str:
    keys = set(probe.get("env_keys") or [])
    names = keys - STAGED_ENV - IMAGE_ENV if extra else STAGED_ENV - keys
    return f"{what}: {', '.join(sorted(names))}" if names else ""


def _connect_problem(probe: dict[str, Any]) -> str:
    opened = [
        t for t, outcome in (probe.get("connect") or {}).items() if outcome == "open"
    ]
    return f"connections opened: {', '.join(opened)}" if opened else ""


def lane_executor(partition: str, queue_timeout: str) -> Any:
    """The site's lane (env from site.yaml and the converge), forced on."""
    from .remote_slurm_step_executor import RemoteSlurmStepExecutor

    return RemoteSlurmStepExecutor(
        enabled=True, partition=partition, queue_timeout=queue_timeout
    )


def _probe_ctx(
    run_id: str,
    step: str,
    *,
    targets: list[tuple[str, int]],
    timeout: int,
    gpu: bool = False,
    parents: dict[str, Any] | None = None,
) -> Any:
    from umrs.pipeline_sdk.step import StepContext

    from .remote_slurm_step_executor import STEP_TIMEOUT_METADATA_KEY
    from .step_executor import EXECUTOR_GPU_METADATA_KEY
    from .step_staging import IN_MEMORY_BUCKET, STAGE_REQUEST_METADATA_KEY

    metadata: dict[str, Any] = {
        STAGE_REQUEST_METADATA_KEY: {"owner": "umrs-smoke", "step": step},
        STEP_TIMEOUT_METADATA_KEY: timeout,
    }
    if gpu:
        metadata[EXECUTOR_GPU_METADATA_KEY] = True
    return StepContext(
        s3=_RunStore(),
        input_bucket=IN_MEMORY_BUCKET,
        input_keys=[],
        output_bucket=IN_MEMORY_BUCKET + "-output",
        config={"connect": [[h, p] for h, p in targets]},
        metadata=metadata,
        run_id=run_id,
        parent_results=dict(parents or {}),
    )


def _run_step(executor: Any, ctx: Any) -> tuple[Any, list[str]]:
    """(result, []) or (None, [the failure])."""
    try:
        result = executor.execute(
            entrypoint=PROBE_ENTRYPOINT, config=dict(ctx.config), ctx=ctx
        )
    except Exception as exc:  # the report names it; the smoke goes on
        logger.warning("step-lane smoke step failed", exc_info=True)
        return None, [f"{type(exc).__name__}: {exc}"[:600]]
    return result, []


def _lane(make_executor: Callable[[str], Any], partition: str, report: dict) -> Any:
    """The lane for *partition*, or None with the failure in *report*."""
    try:
        return make_executor(partition)
    except Exception as exc:  # no lane on this server: named, the smoke goes on
        logger.warning("step-lane smoke on %s has no lane", partition, exc_info=True)
        report.update(ok=False, problems=[f"{type(exc).__name__}: {exc}"[:600]])
        return None


def _nproc(executor: Any) -> int:
    return int(getattr(executor, "nproc_limit", 0) or 0)


def _job_id(result: Any) -> str:
    return str((result.extra.get("slurm_job") or {}).get("job_id", ""))


def run_probe(
    partition: str,
    *,
    gpu: bool,
    targets: list[tuple[str, int]],
    timeout: int,
    make_executor: Callable[[str], Any],
) -> dict[str, Any]:
    """One probe step through the lane on *partition*; its report."""
    run_id = f"smoke-{uuid.uuid4().hex[:12]}"
    report: dict[str, Any] = {"partition": partition, "gpu": gpu, "run_id": run_id}
    ctx = _probe_ctx(run_id, "probe", targets=targets, timeout=timeout, gpu=gpu)
    executor = _lane(make_executor, partition, report)
    if executor is None:
        return report
    result, problems = _run_step(executor, ctx)
    if result is None:
        report.update(ok=False, problems=problems)
        return report
    probe = dict(result.extra.get("probe") or {})
    problems = check(probe, gpu=gpu, nproc_limit=_nproc(executor))
    report.update(ok=not problems, problems=problems, job=_job_id(result), probe=probe)
    return report


def run_chain(
    partition: str,
    *,
    targets: list[tuple[str, int]],
    timeout: int,
    make_executor: Callable[[str], Any],
) -> dict[str, Any]:
    """Step a, then step b of the same run with a's result as its parent
    (the staged parent path, live); b's inputs tag must not move when a's
    job facts do, as a durable re-dispatch that re-runs a would change them
    (peer review F1/T1)."""
    run_id = f"smoke-{uuid.uuid4().hex[:12]}"
    report: dict[str, Any] = {"partition": partition, "chain": "a->b", "run_id": run_id}
    executor = _lane(make_executor, partition, report)
    if executor is None:
        return report
    a, problems = _run_step(
        executor, _probe_ctx(run_id, "a", targets=targets, timeout=timeout)
    )
    if a is None:
        report.update(ok=False, problems=[f"step a: {p}" for p in problems])
        return report
    b_ctx = _probe_ctx(run_id, "b", targets=targets, timeout=timeout, parents={"a": a})
    stable = _tag_of(executor, b_ctx) == _tag_of(executor, _rerun_parent(b_ctx, a))
    b, problems = _run_step(executor, b_ctx)
    problems = [f"step b: {p}" for p in problems]
    if b is not None:
        problems += check(
            dict(b.extra.get("probe") or {}), gpu=False, nproc_limit=_nproc(executor)
        )
    if not stable:
        problems.append("b's inputs tag follows a's job facts")
    report.update(
        ok=not problems,
        problems=problems,
        jobs=[_job_id(a), _job_id(b) if b is not None else ""],
        tag_stable=stable,
    )
    return report


def _tag_of(executor: Any, ctx: Any) -> str:
    from .step_staging import build_stage_plan, inputs_tag

    payload = executor._payload(PROBE_ENTRYPOINT, dict(ctx.config), ctx)
    plan = build_stage_plan(ctx, caps=executor.caps)
    return inputs_tag(plan, payload, None, executor.caps, image=executor.image)


def _rerun_parent(ctx: Any, parent: Any) -> Any:
    """*ctx* with its parent as a re-run would hand it on: another job."""
    import dataclasses

    extra = {
        **parent.extra,
        "slurm_job": {"job_id": "0", "workdir": "/rerun"},
        "slurm": {},
    }
    rerun = dataclasses.replace(parent, extra=extra)
    return dataclasses.replace(ctx, parent_results={"a": rerun})


# ── the kill check: scancel still ends the step (peer review 2026-09-26) ────

#: The inspector job: on the node the cancelled step ran on, as the job user,
#: with the step's podman venue — any step container left in podman's store,
#: any process of the container or the worker still alive, the workdir.
_INSPECT_SCRIPT = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --comment={comment}1
#SBATCH --partition={partition}
#SBATCH --account={account}
{nodelist}#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --output={workdir}/log/slurm.out
#SBATCH --error={workdir}/log/slurm.err
set -uo pipefail
if [[ -n "${{XDG_RUNTIME_DIR:-}}" && ! -w "${{XDG_RUNTIME_DIR}}" ]]; then unset XDG_RUNTIME_DIR; fi
source {venue} && rivault_venue_prepare "umrs-inspect-${{SLURM_JOB_ID}}" >/dev/null 2>&1
ctr={ctr}
ps_out="$(podman ps -a --filter "name=^${{ctr}}" --format '{{{{.Names}}}}|{{{{.Status}}}}' 2>&1 | head -n 20)"
procs="$(pgrep -u "$(id -u)" -af -- "${{ctr}}|umrs[.]server[.]step_worker" | head -n 20)"
left=0
[[ -e {killed_workdir} ]] && left=1
python3 -c '{report_py}' "$(hostname)" "$ps_out" "$procs" "$left" > {workdir}/reply/output.json
"""
_INSPECT_REPORT_PY = (
    "import json,sys; h,c,p,w=sys.argv[1:5]; "
    "print(json.dumps(dict(host=h, containers=[x for x in c.splitlines() if x], "
    "processes=[x for x in p.splitlines() if x], workdir_left=w=='1')))"
)
_START_TIMEOUT = 900.0
_INSPECT_TIMEOUT = 1200.0


def inspect_script(
    executor: Any, job: Any, *, killed_job: str, killed_workdir: str, node: str
) -> str:
    import shlex

    return _INSPECT_SCRIPT.format(
        name=job.job_name,
        comment=job.comment,
        partition=executor.partition,
        account=executor.account,
        nodelist=f"#SBATCH --nodelist={node}\n" if node else "",
        workdir=job.workdir,
        venue=shlex.quote(executor.venue_script),
        ctr=shlex.quote(f"umrs-step-{killed_job}"),
        killed_workdir=shlex.quote(killed_workdir),
        report_py=_INSPECT_REPORT_PY,
    )


def run_kill_check(
    partition: str,
    *,
    timeout: int,
    make_executor: Callable[[str], Any],
    poll: float = 5.0,
    start_timeout: float = _START_TIMEOUT,
) -> dict[str, Any]:
    """A sleeping probe step, cancelled once its container runs; then the
    job must settle CANCELLED, and an inspector job on the same node must
    find no step container, no container or worker process and no workdir."""
    run_id = f"smoke-{uuid.uuid4().hex[:12]}"
    report: dict[str, Any] = {
        "partition": partition,
        "kill_check": True,
        "run_id": run_id,
    }
    executor = _lane(make_executor, partition, report)
    if executor is None:
        return report
    facts: dict[str, Any] = {}
    cancel = threading.Event()
    executor.bind_run(
        run_id, cancelled=cancel.is_set, on_job=lambda _s, job: facts.update(job)
    )
    ctx = _probe_ctx(run_id, "sleeper", targets=[], timeout=timeout)
    ctx.config["sleep_seconds"] = timeout * 2  # never ends by itself
    outcome: list[tuple[Any, list[str]]] = []
    worker = threading.Thread(target=lambda: outcome.append(_run_step(executor, ctx)))
    worker.start()
    started = _await_start(executor, facts, worker, poll, start_timeout)
    cancel.set()
    worker.join(timeout=_START_TIMEOUT)
    _join_settlers(executor)
    report.update(_kill_verdict(executor, facts, started, outcome))
    return report


def _await_start(
    executor: Any, facts: dict[str, Any], worker: Any, poll: float, timeout: float
) -> bool:
    """True once the sleeping step's marker is in its job's log."""
    from .step_lane_probe import SLEEP_MARKER

    deadline = time.monotonic() + timeout
    while worker.is_alive() and time.monotonic() < deadline:
        if facts.get("job_id") and SLEEP_MARKER in _log_tail(
            executor, facts.get("workdir", "")
        ):
            return True
        time.sleep(poll)
    return False


def _log_tail(executor: Any, workdir: str) -> str:
    try:
        proc = executor._helper(
            "tail-log", executor.workdir_root, workdir, 8192, phase="worker"
        )
    except Exception:  # unreachable: not started yet, as far as we know
        logger.warning("step-lane smoke: tail-log failed", exc_info=True)
        return ""
    return proc.stdout.decode("utf-8", "replace") if proc.returncode == 0 else ""


def _join_settlers(executor: Any) -> None:
    for thread in list(getattr(executor, "_settlers", [])):
        thread.join(timeout=_START_TIMEOUT)


def _kill_verdict(
    executor: Any,
    facts: dict[str, Any],
    started: bool,
    outcome: list[tuple[Any, list[str]]],
) -> dict[str, Any]:
    job_id = str(facts.get("job_id") or "")
    state = executor._job_state(job_id) if job_id else ""
    inspect, running = _inspected(executor, job_id, str(facts.get("workdir") or ""))
    found = _run_problems(started, outcome, state) + _node_problems(inspect, running)
    problems = [p for p in found if p]
    return {"ok": not problems, "problems": problems, "job": job_id, "state": state,
            "inspect": inspect}  # fmt: skip


def _inspected(
    executor: Any, job_id: str, workdir: str
) -> tuple[dict[str, Any], list[str]]:
    """The inspector's report and the step containers still running. An
    Exited record is no survivor (the job's SIGKILL takes the podman client
    before --rm; the next step job on the node removes it): it is reported
    as ``leftover``."""
    if not job_id:
        return {"error": "no job"}, []
    inspect = _inspect(executor, job_id, workdir)
    containers = list(inspect.get("containers") or [])
    inspect["leftover"] = [c for c in containers if not _running(c)]
    return inspect, [c for c in containers if _running(c)]


def _run_problems(
    started: bool, outcome: list[tuple[Any, list[str]]], state: str
) -> list[str]:
    completed = bool(outcome) and outcome[0][0] is not None
    return [
        "" if started else "the step container never started",
        "the cancelled step completed" if completed else "",
        ""
        if state == "CANCELLED"
        else f"the job settled as {state or 'unknown'}, not CANCELLED",
    ]


def _node_problems(inspect: dict[str, Any], running: list[str]) -> list[str]:
    return [
        f"inspector: {inspect['error']}" if inspect.get("error") else "",
        _left("containers still running", running),
        _left("processes survived", inspect.get("processes")),
        "the workdir survived the cancel" if inspect.get("workdir_left") else "",
    ]


def _running(container: str) -> bool:
    """A ``name|status`` line of a container that runs (podman: "Up ...")."""
    return container.partition("|")[2].strip().lower().startswith("up")


def _left(what: str, items: Any) -> str:
    return f"{what}: {'; '.join(str(i) for i in items)[:400]}" if items else ""


def _inspect(executor: Any, killed_job: str, killed_workdir: str) -> dict[str, Any]:
    """The inspector job's report ({"error": ...} when it could not run)."""
    from .remote_slurm_step_executor import _job_identity
    from .slurm_step_executor import TERMINAL_STATES

    job = _job_identity(f"smoke-{uuid.uuid4().hex[:12]}", "inspect", 1)
    try:
        job.workdir = executor._mkwd(job)
        node = _node_of(executor, killed_job)
        script = inspect_script(executor, job, killed_job=killed_job,
                                killed_workdir=killed_workdir, node=node)  # fmt: skip
        executor._put(job, "job.sbatch", script.encode("utf-8"))
        executor._submit(job)
        deadline = time.monotonic() + _INSPECT_TIMEOUT
        state = executor._job_state(job.job_id)
        while state not in TERMINAL_STATES and time.monotonic() < deadline:
            time.sleep(5.0)
            state = executor._job_state(job.job_id)
        blob = executor._read_reply_at(job.workdir, job)
        return (
            {**json.loads(blob), "node": node}
            if blob
            else {"error": f"no report ({state})"}
        )
    except Exception as exc:  # the report names it
        logger.warning("step-lane smoke: inspector failed", exc_info=True)
        return {"error": f"{type(exc).__name__}: {exc}"[:400]}
    finally:
        if job.workdir:
            executor._rmwd(job.workdir)


def _node_of(executor: Any, job_id: str) -> str:
    """The node the job ran on (sacct NodeList), "" when unreadable."""
    try:
        proc = executor._run_cmd(
            ["sacct", "-n", "-X", "-P", "-j", job_id, "-o", "NodeList"], timeout=30
        )
    except Exception:  # a forced-command key refuses raw sacct
        logger.warning("step-lane smoke: sacct NodeList failed", exc_info=True)
        return ""
    node = (proc.stdout or "").strip().splitlines()
    return (
        node[0].strip()
        if proc.returncode == 0 and node and node[0].strip().isascii()
        else ""
    )


def _target(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(f"{text!r} is not host:port")
    return host, int(port)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m umrs.server.step_lane_smoke")
    parser.add_argument("--partitions", required=True, help="comma-separated")
    parser.add_argument("--gpu-partition", default="", help="one more run, with a GPU")
    parser.add_argument("--connect", type=_target, action="append", default=[])
    parser.add_argument("--timeout", type=int, default=600, help="per step, seconds")
    parser.add_argument(
        "--queue-timeout", default="00:15:00", help="Slurm duration a job may queue"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _plan(args.partitions, args.gpu_partition)
    if not plan:
        return 2
    from umrs.shared.uploaded_code import load_site_env

    load_site_env()
    if args.gpu_partition:
        # This process only: the site's gpu switch stays as the converge set it.
        os.environ["UMRS_STEP_GPU"] = "1"
    reports = run_plan(
        plan,
        targets=args.connect,
        timeout=args.timeout,
        make_executor=lambda p: lane_executor(p, args.queue_timeout),
    )
    for report in reports:
        print(json.dumps(report, sort_keys=True))
    passed = sum(1 for r in reports if r["ok"])
    summary = {"smoke": "step-lane", "passed": passed, "failed": len(reports) - passed}
    print(json.dumps(summary))
    return 0 if passed == len(reports) else 1


def _plan(partitions: str, gpu_partition: str) -> list[tuple[str, str, bool]]:
    """(kind, partition, gpu) runs: a probe per partition, one more with a
    GPU, the two-step chain on the first partition and the kill check on
    the last."""
    names = [p for p in partitions.split(",") if p]
    if not names:
        return []
    plan = [("probe", p, False) for p in names]
    if gpu_partition:
        plan.append(("probe", gpu_partition, True))
    return [*plan, ("chain", names[0], False), ("kill", names[-1], False)]


def run_plan(
    plan: list[tuple[str, str, bool]],
    *,
    targets: list[tuple[str, int]],
    timeout: int,
    make_executor: Callable[[str], Any],
) -> list[dict[str, Any]]:
    """Every run, in parallel: an exclusive node busy with a serve job
    queues its own run only."""

    def one(item: tuple[str, str, bool]) -> dict[str, Any]:
        kind, partition, gpu = item
        if kind == "chain":
            return run_chain(
                partition, targets=targets, timeout=timeout, make_executor=make_executor
            )
        if kind == "kill":
            return run_kill_check(
                partition, timeout=timeout, make_executor=make_executor
            )
        return run_probe(
            partition,
            gpu=gpu,
            targets=targets,
            timeout=timeout,
            make_executor=make_executor,
        )

    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        return list(pool.map(one, plan))


if __name__ == "__main__":
    sys.exit(main())
