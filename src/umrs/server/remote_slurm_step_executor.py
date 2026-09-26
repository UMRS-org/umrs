"""Remote Slurm step executor — the connected-HPC lane for uploaded step code.

ADR-074 (Amendment 1) sends uploaded step code to the site's connected HPC,
never into the UMRS server. On a front-end deployment (umrs-front,
``scheduler.kind: slurm-remote``) there is no Slurm CLI and no shared
filesystem, so this executor drives the cluster through the login-node helper
``umrs_step_helper.py`` over the site's SSH path — a fixed verb set, the
future forced-command allow-list (build spec
docs/development/rccs-step-lane-spec-2026-09-26.md §3-§8 in the superproject):

1. The stage plan is scope-checked before anything touches the cluster
   (:func:`~umrs.server.step_staging.build_stage_plan`: only the run owner's
   buckets or the run's own store).
2. ``find-jobs umrs-step-<run_id>-<step>`` reconciles earlier attempts: a
   COMPLETED job whose workdir still holds a reply is adopted instead of
   running the step again; a live one is ``scancel``-ed; never two jobs. A
   job whose ``--comment`` names another run is never touched.
3. ``mkwd`` makes ``<root>/<run_id>-<step>-<attempt>-XXXXXXXX`` (0700) with
   ``in out reply src log``; ``stage-in`` extracts one streamed tar of the
   staged objects (and the uploaded source tree) into it; ``put`` writes
   ``input.pkl`` (no platform key, no endpoint) and ``job.sbatch``.
4. ``submit``; then ``status`` polls with separate queue and run clocks, the
   run's cancel flag, the step's own deadline and a newer attempt of the same
   step, any of which ``scancel``-s the job. The job id and workdir go into
   the step record.
5. COMPLETED: ``read-reply`` (JSON only, 4 MiB, never a pickle), then
   ``collect-out`` streams ``out/`` back into the run's output bucket under
   ``runs/<run_id>/<step>/`` and the reply's ``output_keys`` become those
   physical keys. A failure reports a capped ``tail-log``.
6. The workdir is removed only once the job is confirmed terminal.

The step gets files only — no credential, no network, no platform key; the
server keeps every S3 operation. OFF until the site opts in
(``step_executor.remote_enabled``): until then :meth:`execute` refuses every
step before any SSH.

SSH identity/target come from ``site.yaml`` ``extra`` (the same block the
``slurm-remote`` scheduler reads), so there is one SSH path, not two.
Selected with ``UMRS_STEP_EXECUTOR=slurm-remote`` or by a ``slurm``
workflow on a ``slurm-remote`` site.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import pickle
import re
import shlex
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, BinaryIO, ClassVar

from umrs.orchestrator.ssh_control import SSHControlConfig, SSHControlManager
from umrs.pipeline_sdk.step import StepContext, StepResult

from .slurm_step_executor import (
    TERMINAL_FAILURES,
    TERMINAL_STATES,
    UNKNOWN,
    UNREACHABLE,
    UNREACHABLE_BUDGET_SEC,
    WALL_GRACE_SEC,
    PollClock,
    SlurmStepExecutor,
    StagedJob,
    image_archive_stem,
    measure_slurm_accounting,
    render_staged_job_script,
    slurm_duration_seconds,
    stamp_slurm_accounting,
)
from .step_executor import (
    _PAYLOAD_MAX_BYTES,
    EXECUTOR_GPU_METADATA_KEY,
    EXECUTOR_IMAGE_METADATA_KEY,
    EXECUTOR_PAYLOAD_S3_METADATA_KEY,
    EXECUTOR_ROUTED_KEYS_METADATA_KEY,
    EXECUTOR_SOURCE_DIR_METADATA_KEY,
    EXECUTOR_UPSTREAM_METADATA_KEY,
    StepExecutionError,
    _serialise_ctx_payload,
)
from .step_reply import result_from_json
from .step_staging import (
    DEFAULT_CAPS,
    STAGE_REQUEST_METADATA_KEY,
    OutputUploadError,
    StageCaps,
    StagePlan,
    StageStreamError,
    buffered_hint,
    build_stage_plan,
    collect_out,
    inputs_tag,
    out_limits,
    rewrite_output_keys,
    stage_in,
    verify_stage_summary,
)

logger = logging.getLogger(__name__)

#: ``ctx.metadata`` key under which the runner hands over the step's own
#: soft deadline (``timeout_seconds``), so the job is ``scancel``-ed with it.
STEP_TIMEOUT_METADATA_KEY = "__executor_step_timeout__"
#: The reply a step may write (build spec §6); bulk data goes to ``out/``.
REPLY_CAP_BYTES = 4 * 1024 * 1024
LOG_TAIL_BYTES = 64 * 1024
_ERROR_TAIL_CHARS = 1000
_DEFAULT_QUEUE_TIMEOUT = "06:00:00"
_DEFAULT_TIME_LIMIT = 3600.0
#: After ``scancel`` a job may sit in COMPLETING past RCCS KillWait (120 s).
_SETTLE_SECONDS = 300.0
_CONTROL_TIMEOUT = 60
_TRANSFER_TIMEOUT = 300
#: A sweep may remove several workdirs (each up to the helper's 280 s).
_SWEEP_TIMEOUT = 1800
#: Floor for a streamed transfer's timeout: bytes / this rate.
_STREAM_BYTES_PER_SEC = 10 * 1024 * 1024
_MANIFEST_ALLOWANCE = 16 * 1024 * 1024
_SSH_FAILURE_RC = 255
#: The helper's exit code when squeue / sacct failed (helper EXIT_SCHEDULER).
_SCHEDULER_RC = 5
_HELPER_ABSENT = 3
_HELPER_REFUSED = 4
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}")
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]")
_RUN_TAG = re.compile(r"[A-Za-z0-9._-]{1,64}")
_NO_COMMENT = "-"
_COMPLETING = "COMPLETING"
#: Recorded for a job whose end was not confirmed after ``scancel``.
CANCEL_PENDING = "CANCEL_PENDING"
_JOB_ID = re.compile(r"[0-9]{1,20}")
#: What follows a workdir's ``<run>-<step>-`` stem: the attempt, the inputs
#: tag (:func:`~umrs.server.step_staging.inputs_tag`) and mkwd's suffix.
_TAGGED_REST = re.compile(r"[0-9]{1,9}-([0-9a-f]{12})-[a-z0-9_]{8}")
_STATE = re.compile(r"[A-Z_]{1,32}")
#: Executor-internal ``ctx.metadata`` keys: never written into input.pkl.
_INTERNAL_METADATA_KEYS = frozenset(
    {
        EXECUTOR_IMAGE_METADATA_KEY,
        EXECUTOR_PAYLOAD_S3_METADATA_KEY,
        EXECUTOR_ROUTED_KEYS_METADATA_KEY,
        EXECUTOR_GPU_METADATA_KEY,
        EXECUTOR_SOURCE_DIR_METADATA_KEY,
        EXECUTOR_UPSTREAM_METADATA_KEY,
        STAGE_REQUEST_METADATA_KEY,
        STEP_TIMEOUT_METADATA_KEY,
    }
)
#: Platform routing values: useless without a network, never on the cluster.
_PLATFORM_CONFIG_KEYS = ("api_key", "headers", "_routing")
#: ``result.extra`` keys only the executor writes (a step cannot forge them).
_EXECUTOR_EXTRA_KEYS = ("slurm", "slurm_job")
#: A parent result's ``extra`` values that differ between two runs of the
#: same parent on the same inputs (its job, its accounting, a gate's wait).
#: Left out of a step's payload, so a durable re-dispatch that recomputes or
#: re-adopts the parents gives the step the same inputs tag (peer review F1).
_VOLATILE_PARENT_EXTRA = (*_EXECUTOR_EXTRA_KEYS, "waited_seconds")
_VOLATILE_GATE_KEYS = ("decided_at",)
_TRUE = frozenset({"1", "true", "yes", "on"})


def _site_ssh_config() -> SSHControlConfig:
    """SSH config from ``site.yaml`` ``extra`` (the slurm-remote block)."""
    from umrs.orchestrator.site_config import load_site_config

    return SSHControlConfig.from_site_config(load_site_config().extra or {})


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def run_tag(run_id: str) -> str:
    """The run id as the job's ``--comment`` carries it (hashed when it has
    characters an ``#SBATCH`` line should not hold)."""
    if _RUN_TAG.fullmatch(run_id):
        return run_id
    return "h" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:40]


def safe_segment(name: str, fallback: str) -> str:
    """*name* as ``[A-Za-z0-9_-]{1,40}``; a changed name gets a hash suffix,
    so two step names never share a job name (reconciliation cancels by it)."""
    if _SAFE_SEGMENT.fullmatch(name):
        return name
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    safe = _UNSAFE_CHARS.sub("_", name)[:31].lstrip("_-") or fallback
    return f"{safe}-{digest}"


class _Unreachable(StepExecutionError):
    """The login node did not answer (ssh rc 255, a hang)."""


class _SchedulerDown(_Unreachable):
    """The login node answered but Slurm did not (slurmctld / slurmdbd down:
    the helper exits ``_SCHEDULER_RC``) — as transient as an unreachable
    node, never a verdict on a job."""


@dataclass
class _StepJob:
    """One attempt's job: what cleanup and the step record need."""

    run_id: str
    step: str
    attempt: int
    job_name: str
    prefix: str
    comment: str = ""
    workdir: str = ""
    job_id: str = ""
    state: str = ""
    terminal: bool = False
    submit_unknown: bool = False
    keep: bool = False
    #: The workdir is gone and the job terminal: nothing left to reap.
    cleaned: bool = False
    #: The inputs tag this attempt's workdir name carries (empty: reaping).
    tag: str = ""

    @property
    def stem(self) -> str:
        """``<run>-<step>-``: every attempt's workdir name starts with it."""
        return f"{safe_segment(self.run_id, 'run')}-{safe_segment(self.step, 'step')}-"

    def record(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "workdir": self.workdir,
            "job_name": self.job_name,
            "attempt": self.attempt,
            "state": self.state,
            "cleaned": self.cleaned,
        }


@dataclass(frozen=True)
class _FoundJob:
    job_id: str
    state: str
    workdir: str
    #: ``--comment`` as Slurm reports it; empty where sacct stores none.
    comment: str = ""


@dataclass
class _RunBinding:
    cancelled: Callable[[], bool] | None = None
    on_job: Callable[[str, dict[str, Any]], None] | None = None


@dataclass
class _Stream:
    value: Any
    error: StepExecutionError | None
    stdout: bytes
    stderr: bytes
    returncode: int


def _reason(stderr: bytes) -> str:
    """The helper's JSON reason (or the raw tail of stderr)."""
    text = stderr.decode("utf-8", "replace").strip()
    for line in reversed(text.splitlines()):
        try:
            detail = json.loads(line).get("detail")
        except (ValueError, AttributeError):
            continue
        if detail:
            return str(detail)[:500]
    return text[-500:]


@dataclass
class RemoteSlurmStepExecutor(SlurmStepExecutor):
    """Run workflow steps as staged Slurm jobs submitted over SSH."""

    #: The step's ``timeout_seconds`` is enforced on the job's run clock here,
    #: so the runner adds no soft deadline that would count queue wait.
    owns_step_deadline: ClassVar[bool] = True

    ssh_config: SSHControlConfig = field(default_factory=lambda: _site_ssh_config())
    helper_path: str = field(default_factory=lambda: _env("UMRS_STEP_HELPER"))
    venue_script: str = field(default_factory=lambda: _env("UMRS_STEP_VENUE_SCRIPT"))
    image_archive_dir: str = field(
        default_factory=lambda: _env("UMRS_STEP_IMAGE_ARCHIVE_DIR")
    )
    #: Maintainer opt-in (``step_executor.remote_enabled``); off refuses all.
    enabled: bool = field(
        default_factory=lambda: _env("UMRS_STEP_REMOTE_ENABLED").lower() in _TRUE
    )
    queue_timeout: str = field(
        default_factory=lambda: (
            _env("UMRS_SLURM_QUEUE_TIMEOUT") or _DEFAULT_QUEUE_TIMEOUT
        )
    )
    #: Seconds capping ``--time`` only when ``UMRS_STEP_EXECUTOR_TIMEOUT`` is
    #: set: the site's ``time_limit`` and the step's timeout govern otherwise.
    timeout_seconds: int = field(
        default_factory=lambda: int(_env("UMRS_STEP_EXECUTOR_TIMEOUT") or 0)
    )
    #: Empty omits ``--mem`` (partitions such as RCCS ng-dgx-m* reject it); an
    #: empty site value never reaches the environment, so it is the default.
    memory: str = field(default_factory=lambda: _env("UMRS_SLURM_MEM"))
    #: ``--pids-limit`` of the step container; 0 omits it (and its check) for
    #: a venue whose runtime cannot apply it (``step_executor.pids_limit``).
    pids_limit: int = field(
        default_factory=lambda: int(_env("UMRS_STEP_PIDS_LIMIT") or 512)
    )
    #: ``step_executor.cgroups``: "disabled" runs the container without a
    #: cgroup of its own, inside the job's Slurm cgroup (pids_limit must be 0).
    cgroups: str = field(default_factory=lambda: _env("UMRS_STEP_CGROUPS") or "enabled")
    #: ``--ulimit nproc`` of the step container (0 omits it).
    nproc_limit: int = field(
        default_factory=lambda: int(_env("UMRS_STEP_NPROC_LIMIT") or 0)
    )
    helper_python: str = "python3"
    caps: StageCaps = DEFAULT_CAPS
    reply_cap: int = REPLY_CAP_BYTES
    settle_seconds: float = _SETTLE_SECONDS
    unreachable_budget: float = UNREACHABLE_BUDGET_SEC
    #: How long a job whose ``scancel`` has not gone through yet is chased;
    #: None is the site's longest ``--time`` plus the grace.
    cancel_patience: float | None = None
    _ssh: Any = field(default=None, init=False, repr=False)
    _lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _runs: dict[str, _RunBinding] = field(default_factory=dict, init=False, repr=False)
    _attempts: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _settlers: list[threading.Thread] = field(
        default_factory=list, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not self.ssh_config.ssh_target:
            raise RuntimeError(
                "UMRS_STEP_EXECUTOR=slurm-remote needs site.yaml extra.ssh_target "
                "(the login node the slurm-remote scheduler already uses)"
            )
        self._ssh = SSHControlManager(self.ssh_config)

    # ── runner hooks ────────────────────────────────────────────────────────

    def bind_run(
        self,
        run_id: str,
        *,
        cancelled: Callable[[], bool] | None = None,
        on_job: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        """The runner hands over the run's cancel flag and a sink for the
        step record's job facts (job id, workdir, state)."""
        with self._lock:
            self._runs[str(run_id)] = _RunBinding(cancelled, on_job)

    # ── the step ────────────────────────────────────────────────────────────

    def execute(
        self, *, entrypoint: str, config: dict[str, Any], ctx: StepContext
    ) -> StepResult:
        try:
            return self._execute(entrypoint, config, ctx)
        except StepExecutionError:
            raise
        except Exception as exc:  # an S3, SSH or OS failure fails the step by name
            raise StepExecutionError("worker", f"{type(exc).__name__}: {exc}") from exc

    def _execute(
        self, entrypoint: str, config: dict[str, Any], ctx: StepContext
    ) -> StepResult:
        self._check_lane()
        image = self._lane_image(ctx)
        plan = build_stage_plan(ctx, caps=self.caps)
        source_dir = self._resolve_source_dir(ctx)
        payload = self._payload(entrypoint, config, ctx)
        tag = inputs_tag(plan, payload, source_dir or None, self.caps, image=image)
        job = self._begin(ctx, plan.step, tag)
        earlier = self._reconcile(job, ctx, plan)
        if isinstance(earlier, StepResult):
            return earlier
        if earlier is not None:
            return self._resume(job, earlier, ctx, plan)
        job.workdir = self._mkwd(job)
        self._notify(job)  # a crash before submit still leaves the reaper its name
        try:
            return self._run_job(job, payload, ctx, plan, image, source_dir)
        finally:
            self._finish(job)

    def _check_lane(self) -> None:
        required = (
            ("step_executor.remote_enabled", self.enabled),
            ("UMRS_SLURM_WORKDIR_ROOT (step_executor.workdir_root)", self.workdir_root),
            ("UMRS_STEP_HELPER (step_executor.helper_path)", self.helper_path),
            ("UMRS_STEP_VENUE_SCRIPT (step_executor.venue_script)", self.venue_script),
            (
                "UMRS_STEP_IMAGE_ARCHIVE_DIR (step_executor.image_archive_dir)",
                self.image_archive_dir,
            ),
        )
        missing = [
            label
            for label, value in required
            if not value or (isinstance(value, str) and not value.startswith("/"))
        ]
        if missing:
            raise StepExecutionError(
                "setup",
                "the connected-HPC step lane is not configured (needs "
                f"{', '.join(missing)}; paths absolute on the cluster); nothing "
                "was submitted",
            )
        self._check_posture()

    def _check_posture(self) -> None:
        """A container without a cgroup cannot hold a pids limit: say so
        rather than run a step with less than the site asked for."""
        if self.cgroups not in ("enabled", "disabled"):
            raise StepExecutionError(
                "setup",
                f"step_executor.cgroups {self.cgroups!r} is not enabled|disabled",
            )
        if self.cgroups == "disabled" and self.pids_limit > 0:
            raise StepExecutionError(
                "setup",
                "step_executor.cgroups is disabled, so pids_limit "
                f"{self.pids_limit} cannot apply; set pids_limit: 0 (and bound "
                "processes with nproc_limit); nothing was submitted",
            )

    def _lane_image(self, ctx: StepContext) -> str:
        """The step image: the platform image or an allow-listed one (D-7),
        with an archive name the job can load it from."""
        from .step_trust import allowed_step_images

        image = self._resolve_image(ctx)
        if image != self.image and image not in allowed_step_images():
            raise StepExecutionError(
                "setup",
                f"image {image!r} is not an allowed step image on this site "
                "(ADR-074 D-7)",
            )
        image_archive_stem(image)
        return image

    def _begin(self, ctx: StepContext, step: str, tag: str) -> _StepJob:
        key = (str(ctx.run_id), step)
        with self._lock:
            attempt = self._attempts.get(key, 0) + 1
            self._attempts[key] = attempt
        return _job_identity(str(ctx.run_id), step, attempt, tag)

    def _run_job(
        self,
        job: _StepJob,
        payload: bytes,
        ctx: StepContext,
        plan: StagePlan,
        image: str,
        source_dir: str,
    ) -> StepResult:
        self._stage_in(job, ctx, plan, source_dir)
        self._put(job, "input.pkl", payload)
        self._put(job, "job.sbatch", self._job_script(job, ctx, image).encode("utf-8"))
        reason = self._stop_reason(job, ctx, None)
        if reason:
            raise StepExecutionError("run", f"step {job.step} not submitted: {reason}")
        self._submit(job)
        self._notify(job)
        logger.info(
            "RemoteSlurmStepExecutor: job %s (%s) for step %s, workdir %s",
            job.job_id,
            job.job_name,
            job.step,
            job.workdir,
        )
        return self._collect_finished(job, ctx, plan)

    def _collect_finished(
        self, job: _StepJob, ctx: StepContext, plan: StagePlan
    ) -> StepResult:
        state = self._poll(job, ctx)
        if state != "COMPLETED":
            raise self._failure(job, state)
        blob = self._read_reply_at(job.workdir, job)
        if blob is None:
            raise StepExecutionError(
                "worker",
                f"Slurm job {job.job_id} completed but wrote no JSON reply "
                "(reply/output.json); the staged lane reads no other reply",
            )
        return self._result_from(job, ctx, plan, blob)

    def _resume(
        self, job: _StepJob, found: _FoundJob, ctx: StepContext, plan: StagePlan
    ) -> StepResult:
        """An earlier attempt's job on the same inputs still runs (this server
        lost sight of it: an SSH outage, a restart): wait for it instead of
        cancelling it and running the step twice."""
        job.workdir, job.job_id, job.state = found.workdir, found.job_id, found.state
        self._notify(job)
        logger.warning(
            "step lane: resuming job %s of an earlier attempt of %s",
            found.job_id,
            job.step,
        )
        try:
            return self._collect_finished(job, ctx, plan)
        finally:
            self._finish(job)

    # ── payload and job script ──────────────────────────────────────────────

    def _payload(
        self, entrypoint: str, config: dict[str, Any], ctx: StepContext
    ) -> bytes:
        """``input.pkl``: the worker's payload with no executor-internal
        metadata, no platform routing value (key, headers, routing record,
        and an ``api_base`` / ``model`` the router filled in), no
        object-store endpoint (the worker's ``ctx.s3`` is the staged file
        shim), no parent outside the step's dependency closure and no
        volatile parent facts (its inputs tag stays the same across a
        durable re-dispatch)."""
        metadata = {
            k: v for k, v in ctx.metadata.items() if k not in _INTERNAL_METADATA_KEYS
        }
        dropped = (*_PLATFORM_CONFIG_KEYS, *_routed_keys(ctx))
        ctx_payload = _serialise_ctx_payload(ctx, metadata=metadata)
        ctx_payload["config"] = _without(ctx_payload["config"], dropped)
        ctx_payload["parent_results"] = _stable_parents(_upstream_parents(ctx))
        blob = pickle.dumps(
            {
                "entrypoint": entrypoint,
                "config": _without(config, dropped),
                "ctx": ctx_payload,
                "s3_config": {},
            }
        )
        if len(blob) > _PAYLOAD_MAX_BYTES:
            raise StepExecutionError(
                "setup",
                f"payload size {len(blob)} bytes exceeds hard cap "
                f"({_PAYLOAD_MAX_BYTES} bytes); pass large data as staged "
                "objects, not parent_results",
            )
        return blob

    def _job_seconds(self, ctx: StepContext) -> int:
        """``--time``: the least of the site's ``time_limit``, the step's
        ``timeout_seconds`` and ``UMRS_STEP_EXECUTOR_TIMEOUT`` when set."""
        limits = [slurm_duration_seconds(self.time_limit) or _DEFAULT_TIME_LIMIT]
        if self.timeout_seconds > 0:
            limits.append(float(self.timeout_seconds))
        step_timeout = _step_timeout(ctx)
        if step_timeout:
            limits.append(step_timeout)
        return max(1, int(min(limits)))

    def _needs_gpu(self, ctx: StepContext) -> bool:
        """Only a step that asks for a GPU (a real true, not the string
        "false") on a site that enables GPUs (site.yaml ``gpu``) gets one."""
        if _env("UMRS_STEP_GPU", "0") != "1":
            return False
        hint = ctx.metadata.get(EXECUTOR_GPU_METADATA_KEY)
        return hint is True or str(hint).strip().lower() in ("1", "true", "yes")

    def _job_script(self, job: _StepJob, ctx: StepContext, image: str) -> str:
        caps = self.caps
        buffered = out_limits(ctx.s3, caps)[0]
        out_files = caps.max_buffered_files if buffered else caps.max_out_files
        out_bytes = caps.max_buffered_bytes if buffered else caps.max_out_bytes
        seconds = self._job_seconds(ctx)
        return render_staged_job_script(
            StagedJob(
                workdir=job.workdir,
                job_name=job.job_name,
                attempt=job.attempt,
                run_tag=run_tag(job.run_id),
                partition=self.partition,
                account=self.account,
                cpus=self.cpus,
                memory=self.memory,
                time_limit=self._format_time_limit(seconds) or "01:00:00",
                gres_gpus=self.gpus,
                image=image,
                image_archive_dir=self.image_archive_dir,
                venue_script=self.venue_script,
                needs_gpu=self._needs_gpu(ctx),
                pids_limit=self.pids_limit,
                cgroups_disabled=self.cgroups == "disabled",
                nproc_limit=self.nproc_limit,
                max_file_bytes=caps.max_out_file_bytes,
                # Directories count against the file cap too (helper collect-out).
                max_out_entries=2 * out_files + 8,
                max_out_bytes=out_bytes + self.reply_cap,
                # Half the staging floor: below it the lane stops even a step
                # whose outputs the watchdog cannot measure.
                min_free_bytes=caps.min_free_bytes // 2,
                # Slurm drops a job this server no longer chases (a restart
                # lost its record) once it could no longer end in time.
                deadline_seconds=int(self._queue_seconds()) + seconds + WALL_GRACE_SEC,
            )
        )

    # ── transport ───────────────────────────────────────────────────────────

    def _remote_argv(self, command: str) -> list[str]:
        return self._ssh.command_argv(command)

    def _ssh_run(
        self, command: str, *, timeout: int, stdin: bytes | None = None
    ) -> subprocess.CompletedProcess:
        """Run a remote command; bytes in, bytes out."""
        return subprocess.run(
            self._remote_argv(command),
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    def _run_cmd(self, cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess:
        """A raw Slurm CLI call over SSH; only sacct accounting uses it (a
        forced-command key refuses it and the step is then unmeasured)."""
        proc = self._ssh_run(shlex.join(cmd), timeout=timeout)
        return subprocess.CompletedProcess(
            proc.args,
            proc.returncode,
            proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"),
        )

    def _helper_command(self, verb: str, *args: Any) -> str:
        return shlex.join(
            [self.helper_python, "-I", self.helper_path, verb, *map(str, args)]
        )

    def _helper(
        self,
        verb: str,
        *args: Any,
        timeout: int = _CONTROL_TIMEOUT,
        stdin: bytes | None = None,
        phase: str = "setup",
    ) -> subprocess.CompletedProcess:
        try:
            proc = self._ssh_run(
                self._helper_command(verb, *args), timeout=timeout, stdin=stdin
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise _Unreachable(phase, f"helper {verb} did not answer: {exc}") from exc
        if proc.returncode == _SSH_FAILURE_RC:
            raise _Unreachable(phase, f"helper {verb}: the login node is unreachable")
        return proc

    def _checked(self, verb: str, *args: Any, phase: str = "setup", **kw: Any) -> bytes:
        proc = self._helper(verb, *args, phase=phase, **kw)
        if proc.returncode == _SCHEDULER_RC:
            raise _SchedulerDown(
                phase, f"helper {verb}: Slurm did not answer: {_reason(proc.stderr)}"
            )
        if proc.returncode != 0:
            raise StepExecutionError(
                phase,
                f"helper {verb} failed (rc={proc.returncode}): {_reason(proc.stderr)}",
            )
        return proc.stdout

    def _stream(
        self,
        command: str,
        *,
        timeout: float,
        feed: Callable[[BinaryIO], Any] | None = None,
        drain: Callable[[BinaryIO], Any] | None = None,
    ) -> _Stream:
        """One streaming helper call: *feed* writes its stdin, *drain* reads
        its stdout; stderr goes to a file so neither side can block on it. A
        transport that could not even start reports returncode -1."""
        with tempfile.TemporaryFile() as err:
            try:
                proc = subprocess.Popen(
                    self._remote_argv(command),
                    stdin=subprocess.PIPE if feed else subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=err,
                )
            except OSError as exc:
                return _Stream(
                    None, None, b"", f"ssh did not start: {exc}".encode(), -1
                )
            timer = threading.Timer(timeout, proc.kill)
            timer.daemon = True
            timer.start()
            try:
                value, error, stdout = _pump(proc, feed, drain)
            finally:
                # Armed until ssh exits: the timer bounds that wait too.
                returncode = _close(proc)
                timer.cancel()
            err.seek(0)
            return _Stream(value, error, stdout, err.read(64 * 1024), returncode)

    # ── workdir and staging ─────────────────────────────────────────────────

    def _mkwd(self, job: _StepJob) -> str:
        root = self.workdir_root.rstrip("/")
        out = self._checked("mkwd", root, job.prefix).decode("utf-8", "replace").strip()
        if os.path.dirname(out) != root or not os.path.basename(out).startswith(
            job.prefix
        ):
            raise StepExecutionError("setup", f"helper mkwd answered {out[:200]!r}")
        return out

    def _stage_in(
        self, job: _StepJob, ctx: StepContext, plan: StagePlan, source_dir: str
    ) -> None:
        caps = self.caps
        max_files = caps.max_in_objects + caps.max_src_files + 1
        max_bytes = caps.max_in_bytes + caps.max_src_bytes + _MANIFEST_ALLOWANCE
        command = self._helper_command(
            "stage-in",
            self.workdir_root,
            job.workdir,
            "--max-files",
            max_files,
            "--max-bytes",
            max_bytes,
        )
        volume = plan.total_bytes + (caps.max_src_bytes if source_dir else 0)
        result = self._stream(
            command,
            timeout=_TRANSFER_TIMEOUT + volume / _STREAM_BYTES_PER_SEC,
            feed=lambda sink: stage_in(
                plan, ctx.s3, sink, source_dir=source_dir or None, caps=caps
            ),
        )
        _raise_stream_failure(result, "setup", "helper stage-in failed")
        verify_stage_summary(
            result.value, _json_object(result.stdout, "stage-in"), caps
        )

    def _put(self, job: _StepJob, name: str, data: bytes) -> None:
        self._checked(
            "put",
            self.workdir_root,
            job.workdir,
            name,
            stdin=data,
            timeout=_TRANSFER_TIMEOUT,
        )

    def _collect_out(
        self, job: _StepJob, ctx: StepContext, plan: StagePlan
    ) -> dict[str, str]:
        caps = self.caps
        buffered, max_files, limit = out_limits(ctx.s3, caps)
        command = self._helper_command(
            "collect-out",
            self.workdir_root,
            job.workdir,
            "--max-files",
            max_files,
            "--max-bytes",
            limit,
            "--max-file-bytes",
            caps.max_out_file_bytes,
        )
        result = self._stream(
            command,
            timeout=_TRANSFER_TIMEOUT + limit / _STREAM_BYTES_PER_SEC,
            drain=lambda source: collect_out(
                source, ctx.s3, ctx.output_bucket, ctx.run_id, plan.step, caps
            ),
        )
        if _keeps_outputs(result):
            job.keep = True
        _raise_stream_failure(
            result,
            "worker",
            "collecting /work/out failed",
            hint=buffered_hint(caps) if buffered else "",
            note=f"; workdir {job.workdir} is kept" if job.keep else "",
        )
        last_line = (result.stderr.strip().splitlines() or [b""])[-1]
        summary = _json_object(last_line, "collect-out")
        if summary.get("files") != len(result.value):
            raise StepExecutionError(
                "worker",
                f"collect-out sent {summary.get('files')!r} files, "
                f"{len(result.value)} arrived",
            )
        return result.value

    # ── submit and poll ─────────────────────────────────────────────────────

    def _submit(self, job: _StepJob) -> None:
        """``submit``; an outcome this process could not see (ssh failure, a
        hang, Slurm's own timeout) is reconciled by the workdir before
        anything retries — a job found that way is this attempt's."""
        try:
            proc = self._helper(
                "submit", self.workdir_root, job.workdir, timeout=_TRANSFER_TIMEOUT
            )
        except _Unreachable as exc:
            return self._adopt_submitted(job, exc.message)
        job_id = proc.stdout.decode("utf-8", "replace").strip()
        if proc.returncode == 0 and _JOB_ID.fullmatch(job_id):
            job.job_id, job.state = job_id, "SUBMITTED"
            return None
        reason = _reason(proc.stderr)
        if proc.returncode == 0 or "timed out" in reason.lower():
            return self._adopt_submitted(job, reason)
        raise StepExecutionError("worker", f"sbatch refused the step job: {reason}")

    def _adopt_submitted(self, job: _StepJob, why: str) -> None:
        try:
            found = [
                j for j in self._find_jobs(job.job_name) if j.workdir == job.workdir
            ]
        except StepExecutionError as exc:
            job.submit_unknown = True
            raise StepExecutionError(
                "worker",
                f"sbatch outcome unknown ({why}) and find-jobs failed ({exc.message}); "
                f"workdir {job.workdir} is kept — scancel a job running it, if any",
            ) from exc
        if not found:
            raise StepExecutionError(
                "worker", f"sbatch outcome unknown ({why}); no job was queued"
            )
        job.job_id, job.state = found[0].job_id, found[0].state
        logger.warning(
            "RemoteSlurmStepExecutor: sbatch outcome unknown (%s); job %s runs it",
            why,
            job.job_id,
        )

    def _job_state(self, job_id: str) -> str:
        try:
            proc = self._helper("status", job_id, phase="worker")
        except _Unreachable:
            return UNREACHABLE
        state = proc.stdout.decode("utf-8", "replace").strip().upper()
        if proc.returncode != 0:
            return UNREACHABLE
        return state if _STATE.fullmatch(state) else UNKNOWN

    def _poll(self, job: _StepJob, ctx: StepContext) -> str:
        """Poll until a terminal state; any stop reason ends the poll (the
        caller's cleanup ``scancel``-s the job)."""
        clock = PollClock(
            queue_timeout=self._queue_seconds(),
            wall_seconds=float(self._job_seconds(ctx)),
            unreachable_budget=self.unreachable_budget,
        )
        while True:
            job.state = self._job_state(job.job_id)
            clock.observe(job.state)
            if job.state in TERMINAL_STATES:
                job.terminal = True
                return job.state
            reason = clock.expired() or self._stop_reason(job, ctx, clock.running_since)
            if reason:
                # An outage (the state unreadable over SSH) is no decision to
                # stop the job: it keeps running under its own --time, its
                # workdir stays for the next attempt's reconcile or the reaper.
                job.keep = job.keep or job.state == UNREACHABLE
                raise StepExecutionError("run", f"Slurm job {job.job_id} {reason}")
            time.sleep(self.poll_interval)

    def _queue_seconds(self) -> float:
        seconds = slurm_duration_seconds(self.queue_timeout)
        if seconds is None:
            logger.warning(
                "step lane: queue_timeout %r is not a Slurm duration; using %s",
                self.queue_timeout,
                _DEFAULT_QUEUE_TIMEOUT,
            )
            seconds = slurm_duration_seconds(_DEFAULT_QUEUE_TIMEOUT)
        return float(seconds or 0.0)

    def _stop_reason(
        self, job: _StepJob, ctx: StepContext, running_since: float | None
    ) -> str | None:
        """The run was cancelled, a newer attempt took over, or the step's
        own deadline passed — counted from the job's start, so queue wait
        stays bounded by ``queue_timeout`` alone (build spec §8)."""
        binding = self._runs.get(job.run_id)
        if (
            binding is not None
            and binding.cancelled is not None
            and binding.cancelled()
        ):
            return "cancelled: the workflow run was cancelled"
        with self._lock:
            current = self._attempts.get((job.run_id, job.step))
        if current != job.attempt:
            return f"abandoned: attempt {current} of the step took over"
        deadline = _step_timeout(ctx)
        if deadline and running_since is not None:
            if time.monotonic() - running_since > deadline:
                return f"ran past the step's {deadline:g}s deadline (timeout_seconds)"
        return None

    def _failure(self, job: _StepJob, state: str) -> StepExecutionError:
        tail = self._log_tail(job.workdir)
        return StepExecutionError(
            "worker",
            f"Slurm job {job.job_id} failed with status {state}. Error log: {tail}",
        )

    def _log_tail(self, workdir: str) -> str:
        try:
            proc = self._helper(
                "tail-log", self.workdir_root, workdir, LOG_TAIL_BYTES, phase="worker"
            )
        except _Unreachable as exc:
            return f"<unreadable: {exc.message}>"
        if proc.returncode == _HELPER_ABSENT:
            return "<no log>"
        if proc.returncode != 0:
            return f"<refused: {_reason(proc.stderr)}>"
        return proc.stdout[-LOG_TAIL_BYTES:].decode("utf-8", "replace")[
            -_ERROR_TAIL_CHARS:
        ]

    # ── reply and outputs ───────────────────────────────────────────────────

    def _read_reply_at(self, workdir: str, job: _StepJob) -> bytes | None:
        """The JSON reply (capped, no link followed), None when absent. An
        unreachable login node is retried within the budget; past it the
        COMPLETED job's workdir is kept (its output is the only copy) and a
        retry adopts it."""
        started = time.monotonic()
        while True:
            try:
                proc = self._helper(
                    "read-reply",
                    self.workdir_root,
                    workdir,
                    self.reply_cap,
                    phase="worker",
                    timeout=_TRANSFER_TIMEOUT,
                )
                break
            except _Unreachable as exc:
                if time.monotonic() - started >= self.unreachable_budget:
                    job.keep = True
                    raise StepExecutionError(
                        "worker",
                        f"Slurm job {job.job_id} COMPLETED but its reply could not be "
                        f"read ({exc.message}); workdir {workdir} is kept",
                    ) from exc
                time.sleep(self.poll_interval)
        return self._reply_bytes(proc, job)

    def _reply_bytes(
        self, proc: subprocess.CompletedProcess, job: _StepJob
    ) -> bytes | None:
        """Only a refusal (rc 4: a link, the cap) gives up the COMPLETED
        job's workdir; a helper that failed (an I/O error, a checkout being
        rewritten) keeps it for the next attempt to adopt."""
        if proc.returncode == _HELPER_ABSENT:
            return None
        if proc.returncode == _HELPER_REFUSED:
            raise StepExecutionError(
                "worker", f"step reply refused: {_reason(proc.stderr)}"
            )
        if proc.returncode != 0:
            job.keep = True
            raise StepExecutionError(
                "worker",
                f"step reply could not be read (rc={proc.returncode}): "
                f"{_reason(proc.stderr)}; workdir {job.workdir} is kept",
            )
        if len(proc.stdout) > self.reply_cap:
            raise StepExecutionError(
                "worker", f"step reply exceeds {self.reply_cap} bytes"
            )
        return proc.stdout

    def _result_from(
        self, job: _StepJob, ctx: StepContext, plan: StagePlan, blob: bytes
    ) -> StepResult:
        """Decode the reply, collect the outputs, map their keys, stamp the
        measured accounting and the job facts."""
        result = result_from_json(blob)
        mapping = self._collect_out(job, ctx, plan)
        result = rewrite_output_keys(result, mapping, ctx.output_bucket)
        for key in _EXECUTOR_EXTRA_KEYS:
            result.extra.pop(key, None)
        stamp_slurm_accounting(
            result, measure_slurm_accounting(job.job_id, run=self._run_cmd)
        )
        result.extra["slurm_job"] = {"job_id": job.job_id, "workdir": job.workdir}
        return result

    # ── reconciliation ──────────────────────────────────────────────────────

    def _find_jobs(self, job_name: str) -> list[_FoundJob]:
        """This account's jobs named *job_name*; the helper reports a
        workdir whose parent resolves to the root as ``<root>/<name>`` (the
        root may sit behind a symlink, and Slurm records physical paths)."""
        out = self._checked(
            "find-jobs", "--root", self.workdir_root.rstrip("/"), job_name
        ).decode("utf-8", "replace")
        found = []
        for line in out.splitlines():
            fields = line.split(maxsplit=4)
            if (
                len(fields) == 5
                and fields[2] == job_name
                and _JOB_ID.fullmatch(fields[0])
            ):
                comment = "" if fields[3] == _NO_COMMENT else fields[3]
                found.append(
                    _FoundJob(fields[0], fields[1].upper(), fields[4].strip(), comment)
                )
        return found

    def _own_workdir(self, found: _FoundJob, job: _StepJob) -> bool:
        """An earlier attempt of this step of this run: its workdir is this
        lane's and, where Slurm kept the ``--comment`` (squeue; sacct stores
        none without ``AccountingStoreFlags=job_comment``), it names the run."""
        root = self.workdir_root.rstrip("/")
        name = os.path.basename(found.workdir)
        if os.path.dirname(found.workdir) != root or not name.startswith(job.stem):
            return False
        if not found.comment.startswith("umrs-step:"):
            return True  # absent, or rewritten by a site job_submit plugin: the stem decides
        attempt = found.comment[len(job.comment) :]
        return found.comment.startswith(job.comment) and attempt.isdigit()

    def _reconcile(
        self, job: _StepJob, ctx: StepContext, plan: StagePlan
    ) -> StepResult | _FoundJob | None:
        """Earlier attempts of this step (a restart, an abandoned attempt, an
        SSH outage): a COMPLETED one with its reply still on disk is adopted;
        else a live one on the same inputs is returned to resume; other live
        ones are cancelled and finished ones without a reply cleaned up."""
        completed, live = self._earlier_jobs(job, ctx)
        for found in sorted(completed, key=lambda f: int(f.job_id), reverse=True):
            adopted = self._adopt(found, job, ctx, plan)
            if adopted is not None:
                for other in live:
                    self._cancel_earlier(other)
                return adopted
        resume = self._resumable(live, job)
        for other in live:
            if other is not resume:
                self._cancel_earlier(other)
        return resume

    def _earlier_jobs(
        self, job: _StepJob, ctx: StepContext
    ) -> tuple[list[_FoundJob], list[_FoundJob]]:
        """(COMPLETED, live) earlier jobs of this run's step; failed ones'
        workdirs are removed, other runs' jobs left alone."""
        completed: list[_FoundJob] = []
        live: list[_FoundJob] = []
        for found in self._find_jobs_patiently(job, ctx):
            if not self._own_workdir(found, job):
                logger.warning(
                    "step lane: job %s is not this run's (workdir %s, comment %r); "
                    "left alone",
                    found.job_id,
                    found.workdir,
                    found.comment,
                )
                continue
            if found.state == _COMPLETING:
                found = self._after_completing(found)
            if found.state == "COMPLETED":
                completed.append(found)
            elif found.state in TERMINAL_FAILURES:
                self._rmwd(found.workdir)
            else:
                live.append(found)
        return completed, live

    @staticmethod
    def _resumable(live: list[_FoundJob], job: _StepJob) -> _FoundJob | None:
        """The newest live earlier job that runs on this attempt's inputs."""
        same = [f for f in live if _workdir_tag(f.workdir, job) == job.tag]
        return max(same, key=lambda f: int(f.job_id)) if same else None

    def _find_jobs_patiently(self, job: _StepJob, ctx: StepContext) -> list[_FoundJob]:
        """find-jobs, riding out a login-node or Slurm outage for the
        unreachable budget: failing at once burnt a durable run's retries in
        minutes and never adopted a job that finished meanwhile. A cancel of
        the run ends the wait."""
        started = time.monotonic()
        while True:
            try:
                return self._find_jobs(job.job_name)
            except _Unreachable:
                if time.monotonic() - started >= self.unreachable_budget:
                    raise
                reason = self._stop_reason(job, ctx, None)
                if reason:
                    raise StepExecutionError(
                        "run", f"step {job.step} not submitted: {reason}"
                    ) from None
                time.sleep(min(self.poll_interval * 4, 60.0))

    def _after_completing(self, found: _FoundJob) -> _FoundJob:
        """A COMPLETING job has ended but its epilog still runs; scancel-ing it
        would throw away a finished reply. Wait until Slurm settles it."""
        deadline = time.monotonic() + self.settle_seconds
        while time.monotonic() < deadline:
            state = self._job_state(found.job_id)
            if state in TERMINAL_STATES:
                return _FoundJob(found.job_id, state, found.workdir, found.comment)
            time.sleep(self.poll_interval)
        raise StepExecutionError(
            "setup",
            f"Slurm job {found.job_id} of an earlier attempt is still COMPLETING "
            f"after {self.settle_seconds:g}s; refusing to submit the step twice",
        )

    def _cancel_earlier(self, found: _FoundJob) -> None:
        if not self._cancel(found.job_id):
            raise StepExecutionError(
                "setup",
                f"Slurm job {found.job_id} of an earlier attempt still runs and "
                "scancel failed; refusing to submit the step twice",
            )
        logger.warning("step lane: cancelled earlier attempt's job %s", found.job_id)
        self._settle_later(found.job_id, found.workdir, cancelled=True)

    def _adopt(
        self, found: _FoundJob, job: _StepJob, ctx: StepContext, plan: StagePlan
    ) -> StepResult | None:
        earlier = _StepJob(
            run_id=job.run_id,
            step=job.step,
            attempt=job.attempt,
            job_name=job.job_name,
            prefix=job.prefix,
            comment=job.comment,
            workdir=found.workdir,
            job_id=found.job_id,
            state="COMPLETED",
            terminal=True,
        )
        try:
            if _workdir_tag(found.workdir, job) != job.tag:
                logger.warning(
                    "step lane: job %s of an earlier attempt of %s ran on other "
                    "inputs (an upstream step ran again); running the step again",
                    found.job_id,
                    job.step,
                )
                return None
            blob = self._read_reply_at(found.workdir, earlier)
            if blob is None:
                return None
            logger.warning(
                "step lane: job %s of an earlier attempt of %s completed; adopting "
                "its reply instead of running the step again",
                found.job_id,
                job.step,
            )
            return self._result_from(earlier, ctx, plan, blob)
        finally:
            self._finish(earlier)

    # ── settle and clean up ─────────────────────────────────────────────────

    def _cancel(self, job_id: str) -> bool:
        try:
            proc = self._helper("cancel", job_id, phase="worker")
        except _Unreachable:
            return False
        if proc.returncode != 0:
            logger.warning("scancel %s failed: %s", job_id, _reason(proc.stderr))
        return proc.returncode == 0

    def _remove_workdir(self, workdir: str) -> None:
        """The base class's local ``rmtree`` must never see a cluster path."""
        self._rmwd(workdir)

    def _rmwd(self, workdir: str) -> bool:
        """Remove *workdir*; True when it is gone."""
        if self.keep_workdir:
            return False
        try:
            proc = self._helper(
                "rmwd",
                self.workdir_root,
                workdir,
                phase="worker",
                timeout=_TRANSFER_TIMEOUT,  # a step may leave many files
            )
        except _Unreachable as exc:
            logger.warning(
                "step lane: workdir %s not removed: %s", workdir, exc.message
            )
            return False
        if proc.returncode != 0:
            logger.warning(
                "step lane: workdir %s not removed: %s", workdir, _reason(proc.stderr)
            )
        return proc.returncode == 0

    def _finish(self, job: _StepJob) -> None:
        """``scancel`` a job that is not terminal; remove the workdir only
        once the job is confirmed terminal (or was never submitted)."""
        pending = None
        if job.job_id and not job.terminal and not job.keep:
            pending = self._cancel_now(job)
        self._clean_up(job)
        self._notify(job)
        if pending is not None:
            self._settle_later(job.job_id, job.workdir, cancelled=pending, job=job)

    def _clean_up(self, job: _StepJob) -> None:
        if job.submit_unknown or job.keep:
            logger.warning(
                "step lane: workdir %s kept (job %s)", job.workdir, job.job_id or "?"
            )
        elif job.terminal or not job.job_id:
            job.cleaned = self._rmwd(job.workdir)

    def _cancel_now(self, job: _StepJob) -> bool | None:
        """``scancel``; None when the job is terminal now, else whether the
        scancel went through (the job is then recorded CANCEL_PENDING)."""
        cancelled = self._cancel(job.job_id)
        job.state = self._job_state(job.job_id)
        job.terminal = job.state in TERMINAL_STATES
        if job.terminal:
            return None
        job.state = CANCEL_PENDING
        return cancelled

    def _cancel_patience(self) -> float:
        if self.cancel_patience is not None:
            return self.cancel_patience
        limit = slurm_duration_seconds(self.time_limit) or _DEFAULT_TIME_LIMIT
        return limit + WALL_GRACE_SEC

    def _settle_later(
        self, job_id: str, workdir: str, *, cancelled: bool, job: _StepJob | None = None
    ) -> None:
        """Chase a job off the step's thread until it ends, then remove its
        workdir (:meth:`_settle`)."""
        thread = threading.Thread(
            target=self._settle,
            args=(job_id, workdir, cancelled, job),
            name=f"umrs-step-settle-{job_id}",
            daemon=True,
        )
        with self._lock:
            self._settlers.append(thread)
        thread.start()

    def _settle(
        self, job_id: str, workdir: str, cancelled: bool, job: _StepJob | None
    ) -> bool:
        """scancel is re-sent while the job still runs (the first may not have
        reached Slurm: an SSH blip, a login node down) — for up to the job's
        own time limit until one goes through, then ``settle_seconds``. A job
        that does not end stays recorded CANCEL_PENDING for the operator (and
        for the reaper, which chases it again). True once the job ended and
        its workdir is gone."""
        with _SETTLING_LOCK:
            _SETTLING.add(job_id)
        try:
            return self._settle_until_terminal(job_id, workdir, cancelled, job)
        finally:
            with _SETTLING_LOCK:
                _SETTLING.discard(job_id)

    def _settle_until_terminal(
        self, job_id: str, workdir: str, cancelled: bool, job: _StepJob | None
    ) -> bool:
        patience = self.settle_seconds if cancelled else self._cancel_patience()
        deadline = time.monotonic() + patience
        while time.monotonic() < deadline:
            state = self._job_state(job_id)
            if state in TERMINAL_STATES:
                return self._settle_terminal(job, workdir, state, cancelled)
            if state != _COMPLETING and self._cancel(job_id) and not cancelled:
                cancelled = True
                deadline = time.monotonic() + self.settle_seconds
            time.sleep(self.poll_interval)
        logger.warning(
            "step lane: job %s did not end after scancel (%s); workdir %s kept",
            job_id,
            "delivered" if cancelled else "never delivered",
            workdir,
        )
        self._settled(job, CANCEL_PENDING, False)
        return False

    def _settle_terminal(
        self, job: _StepJob | None, workdir: str, state: str, cancelled: bool
    ) -> bool:
        if state == "COMPLETED" and not cancelled:
            # It finished before any scancel of ours reached it: its outputs
            # are intact and a retry may adopt them.
            self._settled(job, state, False)
            return False
        cleaned = self._rmwd(workdir)
        self._settled(job, state, cleaned)
        return cleaned

    def _settled(self, job: _StepJob | None, state: str, cleaned: bool) -> None:
        if job is None:
            return
        job.state, job.terminal, job.cleaned = state, state in TERMINAL_STATES, cleaned
        self._notify(job)

    # ── the reaper: runs that ended without reconciling their jobs ──────────

    def reap_step(
        self,
        run_id: str,
        step: str,
        *facts: dict[str, Any],
        for_retry: bool = False,
    ) -> bool:
        """End what an ended (or refused) run left of one step on the
        cluster: every recorded job of the step still queued or running is
        ``scancel``-ed until terminal, then its workdir removed — a kept
        COMPLETED one too, since no attempt will adopt it now. *for_retry*
        (a durable retry of the run may come) leaves live and COMPLETED jobs
        for that retry to resume or adopt. *facts* are the step's recorded
        jobs (one per workdir). Works with the lane switched off: a kill
        switch must still kill. True when nothing is left."""
        jobs = [_recorded_job(run_id, step, f) for f in facts]
        jobs = jobs or [_job_identity(run_id, step, 1)]
        try:
            targets, complete = self._reap_targets(jobs), True
        except StepExecutionError:  # find-jobs failed: only the recorded jobs
            targets = [
                _FoundJob(j.job_id, self._job_state(j.job_id), j.workdir)
                for j in jobs
                if j.job_id
            ]
            # A workdir without a job id waits for the sweep (a job may use it).
            complete = len(targets) == len(jobs)
        # every one: no short circuit
        done = [self._reap_found(f, for_retry) for f in targets]
        return complete and all(done)

    def _reap_targets(self, jobs: list[_StepJob]) -> list[_FoundJob]:
        """The listed jobs (find-jobs) whose workdir or id the record names,
        plus each recorded workdir no job uses. Never every job of the name:
        a durable retry re-runs the same execution id, and its new jobs carry
        the same name, workdir stem and comment (``mkwd`` records each
        workdir before its job is submitted, so every job of this record is
        named here)."""
        listed = self._find_jobs(jobs[0].job_name)
        workdirs = {j.workdir for j in jobs} - {""}
        found = [
            f
            for f in listed
            if _is_recorded(f, jobs, workdirs) and self._own_workdir(f, jobs[0])
        ]
        # A workdir another run's job names is that job's, whatever we recorded.
        named = {f.workdir for f in listed}
        found += [_FoundJob("", "", w) for w in sorted(workdirs - named)]
        return found

    def _reap_found(self, found: _FoundJob, for_retry: bool = False) -> bool:
        if found.job_id and _settling(found.job_id):
            return False  # this process is chasing it already
        if for_retry and found.job_id and found.state not in TERMINAL_FAILURES:
            # A durable retry of the run resumes a live job on the same inputs
            # or adopts a COMPLETED one; --time and the sweep bound the rest.
            return False
        if not found.job_id or found.state in TERMINAL_STATES:
            return self._rmwd(found.workdir)
        return self._settle(
            found.job_id, found.workdir, self._cancel(found.job_id), None
        )

    def sweep_age(self) -> int:
        """How long a workdir no live job uses must sit idle before the sweep
        removes it: its queue wait, its run and its transfers, with room."""
        caps = self.caps
        transfers = (caps.max_in_bytes + caps.max_out_bytes) / _STREAM_BYTES_PER_SEC
        limit = slurm_duration_seconds(self.time_limit) or _DEFAULT_TIME_LIMIT
        return int(
            self._queue_seconds()
            + limit
            + WALL_GRACE_SEC
            + self.settle_seconds
            + 2 * _TRANSFER_TIMEOUT
            + transfers
        )

    def sweep_workdirs(self) -> dict[str, Any]:
        """``helper sweep``: remove the workdirs no queued or running job
        uses that sat idle past :meth:`sweep_age` (a server that died between
        ``mkwd`` and ``submit``, or before it removed a finished job's)."""
        out = self._checked(
            "sweep",
            self.workdir_root.rstrip("/"),
            "--older-than",
            self.sweep_age(),
            phase="worker",
            timeout=_SWEEP_TIMEOUT,
        )
        return _json_object(out, "sweep")

    def _notify(self, job: _StepJob) -> None:
        binding = self._runs.get(job.run_id)
        if binding is None or binding.on_job is None:
            return
        try:
            binding.on_job(job.step, job.record())
        except Exception:  # the record is advisory; the step goes on
            logger.warning(
                "step lane: recording job %s failed", job.job_id, exc_info=True
            )


def _job_identity(run_id: str, step: str, attempt: int, tag: str = "") -> _StepJob:
    # The whole run id: its first 12 characters keep only 36 random bits.
    run_seg = safe_segment(run_id, "run")
    step_seg = safe_segment(step, "step")
    return _StepJob(
        run_id=run_id,
        step=step,
        attempt=attempt,
        job_name=f"umrs-step-{run_seg}-{step_seg}",
        prefix=f"{run_seg}-{step_seg}-{attempt}-" + (f"{tag}-" if tag else ""),
        comment=f"umrs-step:{run_tag(run_id)}:",
        tag=tag,
    )


def _is_recorded(found: _FoundJob, jobs: list[_StepJob], workdirs: set[str]) -> bool:
    """*found* is one of the recorded *jobs*: by its workdir, or its id."""
    return found.workdir in workdirs or found.job_id in {j.job_id for j in jobs}


def _recorded_job(run_id: str, step: str, facts: dict[str, Any]) -> _StepJob:
    attempt = facts.get("attempt")
    job = _job_identity(
        run_id, step, attempt if isinstance(attempt, int) and attempt > 0 else 1
    )
    job.workdir = str(facts.get("workdir") or "")
    job.job_id = str(facts.get("job_id") or "")
    return job


def _workdir_tag(workdir: str, job: _StepJob) -> str:
    """The inputs tag in an earlier attempt's workdir name ("" when none)."""
    name = os.path.basename(workdir)
    if not name.startswith(job.stem):
        return ""
    match = _TAGGED_REST.fullmatch(name[len(job.stem) :])
    return match.group(1) if match else ""


#: Job ids a settle thread of this process is chasing (the reaper leaves
#: them to it).
_SETTLING: set[str] = set()
_SETTLING_LOCK = threading.Lock()


def _settling(job_id: str) -> bool:
    with _SETTLING_LOCK:
        return job_id in _SETTLING


def _without(config: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: v for k, v in dict(config).items() if k not in keys}


def _routed_keys(ctx: StepContext) -> tuple[str, ...]:
    raw = ctx.metadata.get(EXECUTOR_ROUTED_KEYS_METADATA_KEY) or ()
    return tuple(k for k in raw if isinstance(k, str))


def _upstream_parents(ctx: StepContext) -> dict[str, Any]:
    """The parent results of the step's dependency closure, which the runner
    lists (every earlier step's result reaches ``ctx.parent_results``: an
    unrelated one that varies per run, an LLM step's reply, would re-tag the
    job and a durable re-dispatch would cancel it). Without the list (a
    caller other than the runner) every parent stays."""
    upstream = ctx.metadata.get(EXECUTOR_UPSTREAM_METADATA_KEY)
    parents = dict(ctx.parent_results or {})
    if not isinstance(upstream, (list, tuple)):
        return parents
    return {name: result for name, result in parents.items() if name in upstream}


def _stable_parents(parents: dict[str, Any]) -> dict[str, Any]:
    """The parent results by step name (a parallel wave fills them in
    completion order), without their volatile ``extra`` values."""
    stable: dict[str, Any] = {}
    for name in sorted(parents):
        result = parents[name]
        extra = getattr(result, "extra", None)
        if isinstance(result, StepResult) and isinstance(extra, dict):
            result = dataclasses.replace(result, extra=_stable_extra(extra))
        stable[name] = result
    return stable


def _stable_extra(extra: dict[str, Any]) -> dict[str, Any]:
    """*extra* without the executor's job facts and a gate's timing: a
    re-dispatch re-parks a manual gate, and its new decision carries a new
    ``decided_at`` (peer review T2)."""
    stable = _without(extra, _VOLATILE_PARENT_EXTRA)
    gate = stable.get("gate")
    if isinstance(gate, dict):
        stable["gate"] = _without(gate, _VOLATILE_GATE_KEYS)
    return stable


def _step_timeout(ctx: StepContext) -> float | None:
    raw = ctx.metadata.get(STEP_TIMEOUT_METADATA_KEY)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
        return float(raw)
    return None


def _transport_lost(result: _Stream) -> bool:
    """ssh failed, never started, or the transfer timer killed it."""
    return result.returncode == _SSH_FAILURE_RC or result.returncode < 0


def _keeps_outputs(result: _Stream) -> bool:
    """Whether a failed collect keeps the COMPLETED job's workdir, the only
    copy of its outputs (build spec §5): only a refusal removes it — the
    helper's rc 4 (links, caps, names) or the server's own named refusal. A
    lost transport, a failed store and a helper that failed (EIO on Lustre, a
    checkout rewritten under it) keep it for the next attempt to adopt."""
    if _transport_lost(result) or isinstance(result.error, OutputUploadError):
        return True
    helper_failed = result.returncode not in (0, _HELPER_REFUSED)
    return helper_failed and (
        result.error is None or isinstance(result.error, StageStreamError)
    )


def _raise_stream_failure(
    result: _Stream, phase: str, what: str, *, hint: str = "", note: str = ""
) -> None:
    """The server's own named refusal first: the helper's rc then only says
    its pipe broke. A broken stream or a lost transport is explained by the
    helper's reason (plus *hint*) instead. *note* ends either message."""
    error = result.error
    if result.returncode != 0 and (
        error is None or isinstance(error, StageStreamError) or _transport_lost(result)
    ):
        raise StepExecutionError(
            phase,
            f"{what} (rc={result.returncode}): {_reason(result.stderr)}{hint}{note}",
        ) from error
    if error is not None:
        if note:
            raise StepExecutionError(error.phase, error.message + note) from error
        raise error


def _json_object(raw: bytes, what: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", "replace") or "null")
    except ValueError:
        value = None
    if not isinstance(value, dict):
        raise StepExecutionError("setup", f"helper {what} answered no JSON summary")
    return value


def _pump(
    proc: subprocess.Popen,
    feed: Callable[[BinaryIO], Any] | None,
    drain: Callable[[BinaryIO], Any] | None,
) -> tuple[Any, StepExecutionError | None, bytes]:
    """Run *feed* / *drain* against the process pipes; a StepExecutionError
    is returned (the helper's own reason usually explains it better)."""
    value: Any = None
    try:
        if feed is not None:
            value = feed(proc.stdin)
            proc.stdin.close()
        if drain is not None:
            value = drain(proc.stdout)
        return value, None, proc.stdout.read(64 * 1024)
    except StepExecutionError as exc:
        return value, exc, b""
    except (BrokenPipeError, ConnectionResetError) as exc:
        # The remote side ended first (ssh failed, the helper refused): its
        # exit status and stderr name the cause, not the pipe error.
        return (
            value,
            StageStreamError("setup", f"stream closed by the remote side: {exc}"),
            b"",
        )


def _close(proc: subprocess.Popen) -> int:
    for pipe in (proc.stdin, proc.stdout):
        if pipe is not None and not pipe.closed:
            try:
                pipe.close()
            except OSError:
                pass
    return proc.wait()
