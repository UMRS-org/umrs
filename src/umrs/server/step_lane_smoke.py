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


def check(probe: dict[str, Any], *, gpu: bool) -> list[str]:
    """What is wrong with one run's probe facts ([] when it passed)."""
    found = [
        _identity_problem(probe),
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
    problems = check(probe, gpu=gpu)
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
        problems += check(dict(b.extra.get("probe") or {}), gpu=False)
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
    GPU, and the two-step chain on the first partition."""
    plan = [("probe", p, False) for p in partitions.split(",") if p]
    if plan and gpu_partition:
        plan.append(("probe", gpu_partition, True))
    if plan:
        plan.append(("chain", plan[0][1], False))
    return plan


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
