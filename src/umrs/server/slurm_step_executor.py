"""Slurm step executor — runs user workflow steps as Slurm jobs.

This executor follows the same pattern as the existing
:class:`SlurmPodmanBackend` for vLLM/BentoML: it submits a Slurm job
to a GPU compute node, which runs the step inside a podman container.

Flow:
    1. Serialize step entrypoint, config, and StepContext to ``input.pkl``
       using the same nested ``ctx`` payload as PodmanExecutor (includes
       ``parent_results`` / ``input_keys`` for multi-step DAGs)
    2. Generate an sbatch script that:
       - Allocates GPU resources on a compute node
       - Runs ``podman run`` with ``UMRS_STEP_INPUT`` / ``UMRS_STEP_OUTPUT``
         pointing at the bind-mounted pickle files
       - Invokes ``python -m umrs.server.step_worker``
    3. Submit via ``sbatch``
    4. Poll ``squeue`` for completion
    5. Read ``output.pkl`` reply dict and return the :class:`StepResult`

The executor is selected by setting ``UMRS_STEP_EXECUTOR=slurm``.

This module also holds what the front-end lane
(:mod:`umrs.server.remote_slurm_step_executor`, ADR-074) shares with it: the
terminal Slurm states, the queue/run :class:`PollClock` and the staged job
script (:func:`render_staged_job_script`, build spec 2026-09-26 §7) — the
hardened podman argv that mounts only the workdir's ``in``/``out``/``reply``/
``src`` and ``input.pkl``, never the workdir itself, and bounds what the step
can write to the shared filesystem while it runs (a per-file size limit, a
capped log, a watchdog on ``out/`` and ``reply/``).
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from umrs.pipeline_sdk.step import StepContext, StepResult

from .step_executor import (
    _PAYLOAD_MAX_BYTES,
    EXECUTOR_GPU_METADATA_KEY,
    EXECUTOR_IMAGE_METADATA_KEY,
    EXECUTOR_PAYLOAD_S3_METADATA_KEY,
    EXECUTOR_SOURCE_DIR_ENV_VAR,
    EXECUTOR_SOURCE_DIR_METADATA_KEY,
    StepExecutionError,
    StepExecutor,
    _serialise_ctx_payload,
    _serialise_s3_client_config,
)
from .step_reply import (
    check_reply_size,
    json_reply_path,
    reply_max_bytes,
    result_from_json,
    result_from_legacy_pickle,
)

logger = logging.getLogger(__name__)

#: States a job never leaves once Slurm reports them (build spec §8). The
#: lane asks ``--no-requeue``, so NODE_FAIL / PREEMPTED do not come back.
TERMINAL_FAILURES = frozenset(
    {
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "OUT_OF_MEMORY",
        "NODE_FAIL",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
    }
)
TERMINAL_STATES = TERMINAL_FAILURES | {"COMPLETED"}
#: Waiting for a node; a job seen here again after RUNNING was requeued.
QUEUED_STATES = frozenset(
    {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED"}
)
UNKNOWN = "UNKNOWN"
UNREACHABLE = "UNREACHABLE"
#: Past the job's own ``--time`` before the executor gives up on it: above
#: RCCS KillWait (120 s) plus the epilog, so a Slurm TIMEOUT settles first.
WALL_GRACE_SEC = 300
#: How long the job state may stay unreadable over SSH before giving up.
UNREACHABLE_BUDGET_SEC = 600


def measure_slurm_accounting(
    job_id: str,
    run: Any = None,
) -> dict[str, Any]:
    """D-36 leg B: sacct-measured cost for a FINISHED slurm job.

    One ``sacct -o Elapsed,AllocTRES%gres`` read → seconds × GPUs.
    Empty dict when sacct is unavailable or answers garbage — unmeasured,
    never approximated (ADR-067's never-approximate rule).
    """
    from umrs.shared.slurm_accounting import (
        measured_gpu_seconds,
        parse_alloc_tres_gpus,
        parse_slurm_elapsed,
    )

    cmd = [
        "sacct",
        "--job",
        str(job_id),
        "--format=Elapsed,AllocTRES%gres",
        "--noheader",
        "--parsable2",
    ]
    runner = run or _run_local
    try:
        result = runner(cmd, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    parts = result.stdout.strip().splitlines()[0].split("|")
    elapsed = parse_slurm_elapsed(parts[0]) if parts else None
    gpus = parse_alloc_tres_gpus(parts[1]) if len(parts) > 1 else 0
    if elapsed is None:
        return {}
    return {
        "job_id": str(job_id),
        "elapsed_seconds": elapsed,
        "gpus": gpus,
        "gpu_seconds": measured_gpu_seconds(elapsed, gpus),
    }


def _run_local(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    """Local Slurm CLI call (the submit host has sbatch/squeue/sacct)."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def stamp_slurm_accounting(result: Any, accounting: dict[str, Any]) -> Any:
    """Stamp measured accounting onto a StepResult's ``extra`` bag.

    Only stamps when the measurement is real (non-empty) — unmeasured
    steps carry nothing, never an invented zero."""
    if not accounting or not isinstance(result.extra, dict):
        return result
    result.extra["slurm"] = dict(accounting)
    return result


#: An OCI image reference: registry/name[:tag][@digest] characters only. The
#: image lands in a generated shell script, so anything else is refused.
_IMAGE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,511}")


def _checked_image(image: str) -> str:
    """*image* when it is a plain image reference; StepExecutionError otherwise."""
    if not _IMAGE_REF_RE.fullmatch(image):
        raise StepExecutionError(
            "setup",
            f"container image {image[:80]!r} is not a plain image reference "
            "(registry/name[:tag][@digest])",
        )
    return image


def _s3_env_args(s3_config: dict[str, Any]) -> list[str]:
    """S3 endpoint/region env for the worker pod (never a credential)."""
    args: list[str] = []
    endpoint = (
        s3_config.get("endpoint_url")
        or os.environ.get("S3_ENDPOINT_URL")
        or os.environ.get("MINIO_ENDPOINT")
    )
    if endpoint:
        # S3_ENDPOINT_URL is canonical (ADR-055); MINIO_ENDPOINT kept for older workers.
        args.extend(
            [
                "-e",
                shlex.quote(f"S3_ENDPOINT_URL={endpoint}"),
                "-e",
                shlex.quote(f"MINIO_ENDPOINT={endpoint}"),
            ]
        )
    region = s3_config.get("region_name") or os.environ.get("AWS_REGION")
    if region:
        args.extend(["-e", shlex.quote(f"AWS_REGION={region}")])
    # No credential value ever reaches the job script or the pod (ADR-074).
    return args


def _podman_run_argv(
    *,
    network: str,
    workdir: str,
    input_rel: str,
    output_rel: str,
    source_dir: str,
    needs_gpu: bool,
    s3_config: dict[str, Any],
    image: str,
) -> list[str]:
    """The step worker's ``podman run`` argv (env-var file transport —
    argv paths alone are ignored by ``step_worker.main()`` unless
    ``UMRS_STEP_*`` is set; same contract as PodmanExecutor)."""
    parts = [
        "podman",
        "run",
        "--rm",
        f"--network={shlex.quote(network)}",
        "-v",
        f"{shlex.quote(workdir)}:/work",
        "-e",
        f"UMRS_STEP_INPUT=/work/{input_rel}",
        "-e",
        f"UMRS_STEP_OUTPUT=/work/{output_rel}",
        # ADR-074 D-6: the reply the server reads is data (JSON); output.pkl
        # stays for images that predate it.
        "-e",
        f"UMRS_STEP_OUTPUT_JSON=/work/{os.path.splitext(output_rel)[0]}.json",
    ]
    if source_dir:
        # User step modules live outside the image (uploaded bundle):
        # mount the tree read-only at the same absolute path and tell
        # the worker to put it on sys.path. Requires the path to be
        # visible from the compute node (shared filesystem).
        parts.extend(
            [
                "-v",
                f"{shlex.quote(source_dir)}:{shlex.quote(source_dir)}:ro",
                "-e",
                f"{EXECUTOR_SOURCE_DIR_ENV_VAR}={shlex.quote(source_dir)}",
            ]
        )
    if needs_gpu:
        # Match PodmanExecutor device CDI form (site-dependent).
        parts.extend(["--device", "nvidia.com/gpu=all"])
    parts.extend(_s3_env_args(s3_config))
    parts.extend(
        [
            shlex.quote(image),
            "python",
            "-m",
            "umrs.server.step_worker",
        ]
    )
    return parts


def slurm_duration_seconds(text: str) -> float | None:
    """A Slurm duration (``[D-]HH:MM:SS``, ``MM:SS``, bare minutes) in seconds."""
    from umrs.shared.slurm_accounting import parse_slurm_elapsed

    raw = str(text or "").strip()
    if raw.isdigit():  # Slurm reads a bare number as minutes
        return float(int(raw) * 60)
    return parse_slurm_elapsed(raw)


@dataclass
class PollClock:
    """Separate queue and run clocks (build spec §8).

    Queue wait is bounded by ``queue_timeout``; Slurm's ``--time`` counts from
    job start, so the run clock starts at the first state past the queue and
    allows the job's own limit plus :data:`WALL_GRACE_SEC`. A job seen queued
    again after it ran was requeued, and Slurm grants the re-run a fresh
    ``--time``, so both clocks restart. An unreadable state (SSH or the Slurm
    CLI failing) is bounded by ``unreachable_budget`` on its own.
    """

    queue_timeout: float
    wall_seconds: float
    unreachable_budget: float = UNREACHABLE_BUDGET_SEC
    submitted: float = field(default_factory=time.monotonic)
    running_since: float | None = None
    unreachable_since: float | None = None

    def observe(self, state: str) -> None:
        now = time.monotonic()
        if state != UNREACHABLE:
            self.unreachable_since = None
        elif self.unreachable_since is None:
            self.unreachable_since = now
        if state in QUEUED_STATES:
            if self.running_since is not None:
                logger.warning("Slurm step job requeued; its clocks restart")
                self.running_since, self.submitted = None, now
        elif state not in (UNKNOWN, UNREACHABLE) and self.running_since is None:
            self.running_since = now

    def expired(self) -> str | None:
        """Why polling must stop now, or None."""
        now = time.monotonic()
        if self.unreachable_since is not None:
            lasted = now - self.unreachable_since
            if lasted >= self.unreachable_budget:
                return f"state unreadable over SSH for {int(lasted)}s"
        if self.running_since is None:
            if now - self.submitted > self.queue_timeout:
                return f"still queued after {int(self.queue_timeout)}s (queue_timeout)"
            return None
        if now - self.running_since > self.wall_seconds + WALL_GRACE_SEC:
            return (
                f"ran past its {int(self.wall_seconds)}s time limit "
                f"(+{WALL_GRACE_SEC}s grace)"
            )
        return None


#: The step container's contract label (``docker/Containerfile.step``).
STEP_CONTRACT_LABEL = "org.umrs.step-contract"
STEP_CONTRACT_VERSION = "1"
#: Exit codes of the job script: 75 = transient, 78 = configuration.
_EX_TEMPFAIL = 75
_EX_CONFIG = 78
_ARCHIVE_PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RUN_TAG_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")
#: What the step container may write to the job's Slurm log (Lustre): the
#: head live, the tail (where a traceback lands) at the end; the middle of an
#: over-long log is dropped.
LOG_HEAD_BYTES = 63 * 1024 * 1024
LOG_TAIL_BYTES = 1024 * 1024
#: How often the job's watchdog measures ``out/`` and ``reply/``.
WATCH_SECONDS = 15


def image_archive_stem(image: str) -> str:
    """``<name>-<tag>`` of the archive a node loads *image* from: the stager
    saves ``localhost/umrs-step:<sha12>`` as ``umrs-step-<sha12>-<arch>.tar``."""
    leaf = image.rsplit("/", 1)[-1]
    name, _, tag = leaf.partition(":")
    tag = tag or "latest"
    if "@" in leaf or not (
        _ARCHIVE_PART_RE.fullmatch(name) and _ARCHIVE_PART_RE.fullmatch(tag)
    ):
        raise StepExecutionError(
            "setup",
            f"image {image!r} has no archive name on this lane "
            "(expected [registry/]name:tag, no digest)",
        )
    return f"{name}-{tag}"


@dataclass(frozen=True)
class StagedJob:
    """Everything the staged job script is rendered from."""

    workdir: str
    job_name: str
    attempt: int
    #: The run id as ``--comment`` carries it (reconciliation matches it).
    run_tag: str
    partition: str
    account: str
    cpus: int
    memory: str
    time_limit: str
    gres_gpus: int
    image: str
    image_archive_dir: str
    venue_script: str
    needs_gpu: bool
    #: ``--pids-limit`` (0 omits it and its check).
    pids_limit: int = 512
    #: ``--cgroups=disabled``: the venue grants the job no cgroup to create
    #: the container's in; the container stays in the job's Slurm cgroup.
    cgroups_disabled: bool = False
    #: ``--ulimit nproc`` (0 omits it): a process bound without a cgroup.
    nproc_limit: int = 0
    #: ``--ulimit fsize``: no file the step writes grows past this.
    max_file_bytes: int = 8 * 1024**3
    #: The watchdog stops the step once ``out/`` + ``reply/`` pass these.
    max_out_entries: int = 40_008
    max_out_bytes: int = 32 * 1024**3 + 4 * 1024**2
    #: The watchdog stops the step below this much free space on the
    #: workdir's filesystem (0: no floor) — a bound no permission can hide.
    min_free_bytes: int = 0
    #: ``--deadline=now+N``: Slurm itself ends a job that could no longer
    #: finish N s after submission (0: none), even one no server chases.
    deadline_seconds: int = 0
    watch_seconds: int = WATCH_SECONDS
    log_head_bytes: int = LOG_HEAD_BYTES
    log_tail_bytes: int = LOG_TAIL_BYTES


#: The step container's posture (ADR-074 D-4, build spec §7). ``--user``
#: must be explicit: the image's USER 65534 cannot read the 0700 workdir.
#: ``--http-proxy=false``: podman would copy the job's (login session's)
#: ``*_proxy`` variables into the container, a proxy URL may carry a password.
_POD_POSTURE = (
    "--rm",
    "--init",
    "--pull=never",
    "--network=none",
    "--http-proxy=false",
    "--read-only",
    "--tmpfs /tmp:rw,size=4g",
    "--cap-drop=all",
    "--security-opt no-new-privileges",
    "--userns=keep-id",
    '--user "$(id -u):$(id -g)"',
)


def _podman_run(job: StagedJob) -> list[str]:
    """``podman run`` — without a cgroup, with the cgroupfs manager pinned: the
    systemd manager (the containers.conf default on systemd hosts) would put
    conmon in a user@ scope outside the job's Slurm cgroup, and the container
    would outlive scancel (peer review 2026-09-26)."""
    if job.cgroups_disabled:
        return ["podman", "--cgroup-manager=cgroupfs", "run"]
    return ["podman", "run"]


def _pod_posture(job: StagedJob, name: str, *, seccomp: bool = True) -> list[str]:
    """``podman run`` options for container *name* (a shell word); with
    *seccomp*, the node's default profile in which ``fallocate`` fails
    (``${_step_seccomp}``, :func:`_staged_seccomp_lines`)."""
    parts = [*_POD_POSTURE[:2], f"--name {name}", *_POD_POSTURE[2:]]
    if seccomp:
        parts.append('--security-opt seccomp="${_step_seccomp}"')
    if job.cgroups_disabled:
        parts.append("--cgroups=disabled")
    elif job.pids_limit > 0:
        parts.append(f"--pids-limit={int(job.pids_limit)}")
    if job.nproc_limit > 0:
        parts.append(f"--ulimit nproc={int(job.nproc_limit)}")
    return [*parts, f"--ulimit fsize={int(job.max_file_bytes)}"]


#: (workdir entry, container path, mode) — the workdir itself, its job
#: script and its Slurm logs are never mounted.
_STAGED_MOUNTS = (
    ("in", "/work/in", "ro"),
    ("out", "/work/out", "rw"),
    ("reply", "/work/reply", "rw"),
    ("src", "/src", "ro"),
    ("input.pkl", "/work/input.pkl", "ro"),
)
_STAGED_ENV = (
    "UMRS_STEP_INPUT=/work/input.pkl",
    "UMRS_STEP_OUTPUT_JSON=/work/reply/output.json",
    "UMRS_STEP_STAGED=1",
    "UMRS_STEP_SOURCE_DIR=/src",
)


def staged_podman_argv(job: StagedJob) -> list[str]:
    """``podman run`` for a staged step; ``${_step_image}`` is set by the
    image-ensure lines. No other ``-e`` — the step gets files only."""
    parts = [*_podman_run(job), *_pod_posture(job, '"${_step_ctr}"')]
    for entry, target, mode in _STAGED_MOUNTS:
        parts += ["-v", shlex.quote(f"{job.workdir}/{entry}:{target}:{mode}")]
    for assignment in _STAGED_ENV:
        parts += ["-e", assignment]
    if job.needs_gpu:
        parts += ["--device", "nvidia.com/gpu=all"]
    return [*parts, '"${_step_image}"', "python", "-m", "umrs.server.step_worker"]


def _staged_directives(job: StagedJob) -> list[str]:
    if not _RUN_TAG_RE.fullmatch(job.run_tag):
        raise StepExecutionError(
            "setup", f"run tag {job.run_tag[:80]!r} cannot go into --comment"
        )
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job.job_name}",
        f"#SBATCH --comment=umrs-step:{job.run_tag}:{int(job.attempt)}",
        "#SBATCH --no-requeue",
        f"#SBATCH --time={job.time_limit}",
        f"#SBATCH --partition={job.partition}",
    ]
    if job.account:
        lines.append(f"#SBATCH --account={job.account}")
    lines.append(f"#SBATCH --cpus-per-task={int(job.cpus)}")
    # Partitions such as RCCS ng-dgx-m* reject --mem and --gres.
    if job.memory.strip():
        lines.append(f"#SBATCH --mem={job.memory}")
    if job.needs_gpu and job.gres_gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{int(job.gres_gpus)}")
    if job.deadline_seconds > 0:
        lines.append(f"#SBATCH --deadline=now+{int(job.deadline_seconds)}")
    lines += [
        f"#SBATCH --output={job.workdir}/log/slurm.out",
        f"#SBATCH --error={job.workdir}/log/slurm.err",
    ]
    return lines


def _staged_prologue() -> list[str]:
    return [
        "",
        "set -euo pipefail",
        "umask 077",
        '_step_ctr="umrs-step-${SLURM_JOB_ID}"',
        '_step_watch_pid=""',
        "_step_fatal() {",
        "    printf '[umrs-step] FATAL: %s\\n' \"$2\" >&2",
        '    exit "$1"',
        "}",
        "# scancel / TIMEOUT send TERM: exit through the trap so the pod goes too",
        "# (the epilog's pkill backstops it on exclusive nodes).",
        "_step_cleanup() {",
        "    trap '' TERM INT",
        '    if [[ -n "${_step_watch_pid}" ]]; then',
        '        kill "${_step_watch_pid}" 2>/dev/null || true',
        "    fi",
        '    podman kill "${_step_ctr}" >/dev/null 2>&1 || true',
        '    podman rm -f "${_step_ctr}" >/dev/null 2>&1 || true',
        "}",
        "trap _step_cleanup EXIT",
        "trap 'exit 143' TERM",
        "trap 'exit 130' INT",
        "",
    ]


def _or_fatal(command: str, code: int, message: str) -> str:
    """``command || _step_fatal code "message"`` (*message* may expand vars)."""
    return f'{command} || _step_fatal {code} "{message}"'


def _staged_venue_lines(venue_script: str) -> list[str]:
    """Job-private podman configuration (ADR-045 D1: venue.sh)."""
    return [
        "# sbatch over SSH exports the login session's runtime dir, which the",
        "# compute node may not have; venue.sh then picks its own.",
        'if [[ -n "${XDG_RUNTIME_DIR:-}" && ! -w "${XDG_RUNTIME_DIR}" ]]; then',
        "    unset XDG_RUNTIME_DIR",
        "fi",
        f"_step_venue={shlex.quote(venue_script)}",
        _or_fatal(
            '[[ -f "${_step_venue}" ]]',
            _EX_CONFIG,
            "venue script not found: ${_step_venue}",
        ),
        "# shellcheck disable=SC1090",
        'source "${_step_venue}"',
        _or_fatal(
            'rivault_venue_prepare "umrs-step-${SLURM_JOB_ID}"',
            _EX_TEMPFAIL,
            "podman venue could not be prepared on $(hostname)",
        ),
        "",
    ]


def _staged_image_lines(image: str, archive_dir: str) -> list[str]:
    """Image ensure: the node's image must carry the archive's recorded id,
    else it is removed and loaded from the archive; never pulled."""
    stem = image_archive_stem(image)
    archive = shlex.quote(f"{archive_dir.rstrip('/')}/{stem}-")
    label = "'{{ index .Config.Labels \"" + STEP_CONTRACT_LABEL + "\" }}'"
    inspect = "podman image inspect --format"
    image_id = f"$({inspect} '{{{{.Id}}}}' \"${{_step_image}}\" 2>/dev/null || true)"
    contract = f'$({inspect} {label} "${{_step_image}}" 2>/dev/null || true)'
    return [
        f"_step_image={shlex.quote(image)}",
        f'_step_archive={archive}"$(uname -m).tar"',
        _or_fatal(
            '[[ -f "${_step_archive}" && -f "${_step_archive}.id" ]]',
            _EX_CONFIG,
            "no step image archive ${_step_archive} (+ .id)",
        ),
        '_step_want="$(head -c 200 "${_step_archive}.id" | tr -d \'[:space:]\')"',
        f'_step_have="{image_id}"',
        'if [[ -z "${_step_want}" || "${_step_have}" != "${_step_want}" ]]; then',
        '    podman rmi -f "${_step_image}" >/dev/null 2>&1 || true',
        "    "
        + _or_fatal(
            'podman load -i "${_step_archive}" >&2',
            _EX_CONFIG,
            "podman load ${_step_archive} failed",
        ),
        f'    _step_have="{image_id}"',
        "    "
        + _or_fatal(
            '[[ -n "${_step_want}" && "${_step_have}" == "${_step_want}" ]]',
            _EX_CONFIG,
            "image ${_step_image} does not match ${_step_archive}.id",
        ),
        "    echo '[umrs-step] image loaded from archive'",
        "fi",
        f'_step_contract="{contract}"',
        _or_fatal(
            f'[[ "${{_step_contract}}" == "{STEP_CONTRACT_VERSION}" ]]',
            _EX_CONFIG,
            f"image ${{_step_image}} lacks {STEP_CONTRACT_LABEL}={STEP_CONTRACT_VERSION}",
        ),
        "",
    ]


#: Rewrites the node's default seccomp profile (argv[1]) so that
#: ``fallocate`` fails with EOPNOTSUPP: a step could otherwise reserve the
#: shared filesystem's free space in one call the watchdog's size checks
#: never see (``FALLOC_FL_KEEP_SIZE`` keeps the apparent size, and
#: ``RLIMIT_FSIZE`` does not apply to it). glibc's ``posix_fallocate`` then
#: falls back to writing, which the file size limit and the watchdog bound.
#: Runs as ``python -I -c`` inside the step image (no single quote in it).
SECCOMP_FILTER_PY = """import json, sys
profile = json.load(open(sys.argv[1]))
kept = []
for rule in profile.get("syscalls") or []:
    if rule.get("name") == "fallocate":
        continue
    if "names" in rule:
        rule["names"] = [n for n in rule["names"] if n != "fallocate"]
        if not rule["names"]:
            continue
    kept.append(rule)
kept.append({"names": ["fallocate"], "action": "SCMP_ACT_ERRNO", "errnoRet": 95})
profile["syscalls"] = kept
json.dump(profile, sys.stdout)
"""


def _staged_seccomp_lines(job: StagedJob) -> list[str]:
    """``${_step_seccomp}``: the node's default profile plus the fallocate
    rule, written by the step image's Python next to the job script (the
    workdir itself is never mounted). No profile file, no step."""
    info = "podman info --format '{{.Host.Security.SECCOMPProfilePath}}'"
    transform = " ".join(
        [
            *_podman_run(job),
            *_pod_posture(job, '"${_step_ctr}-seccomp"', seccomp=False),
            '-v "${_step_seccomp_base}:/umrs-seccomp-base.json:ro"',
            '"${_step_image}" python -I -c',
            shlex.quote(SECCOMP_FILTER_PY),
            "/umrs-seccomp-base.json",
        ]
    )
    return [
        f'_step_seccomp_base="$({info} 2>/dev/null || true)"',
        _or_fatal(
            '[[ -f "${_step_seccomp_base}" ]]',
            _EX_CONFIG,
            "no default seccomp profile on $(hostname) (${_step_seccomp_base:0:200})",
        ),
        f"_step_seccomp={shlex.quote(job.workdir + '/seccomp.json')}",
        _or_fatal(
            f'{transform} > "${{_step_seccomp}}"',
            _EX_CONFIG,
            "the step seccomp profile could not be written on $(hostname)",
        ),
        "",
    ]


def _staged_pids_lines(job: StagedJob) -> list[str]:
    """``--pids-limit`` must hold before the step runs: rootless podman on a
    cgroupfs node without a delegated cgroup drops it with a warning ("PIDs
    limit discarded"), or its runtime refuses the container. A probe in the
    same posture reads the limit back, so neither passes silently."""
    if job.pids_limit <= 0 or job.cgroups_disabled:
        return []
    limit = int(job.pids_limit)
    probe = " ".join(
        [
            *_podman_run(job),
            *_pod_posture(job, '"${_step_ctr}-pids"'),
            '"${_step_image}" cat /sys/fs/cgroup/pids.max',
        ]
    )
    tail = "$(printf '%s' \"${_step_pids}\" | tail -c 300)"
    return [
        _or_fatal(
            f'_step_pids="$({probe} 2>&1)"',
            _EX_CONFIG,
            f"the step posture was refused on $(hostname): {tail}",
        ),
        '_step_pids_max="$(printf \'%s\\n\' "${_step_pids}" | tail -n 1)"',
        _or_fatal(
            f'[[ "${{_step_pids_max}}" == "{limit}" ]]',
            _EX_CONFIG,
            f"--pids-limit={limit} is not applied on $(hostname) "
            "(pids.max ${_step_pids_max:0:40})",
        ),
        "",
    ]


def _watch_stop(condition: str, why: str) -> list[str]:
    """``if condition`` then name *why* in the log and kill the pod."""
    return [
        f"        if {condition}; then",
        f"            printf '[umrs-step] FATAL: %s\\n' \"{why}; the step was stopped\" >&2",
        '            podman kill "${_step_ctr}" >/dev/null 2>&1',
        "            return",
        "        fi",
    ]


def _watch_floor_lines(job: StagedJob) -> list[str]:
    if job.min_free_bytes <= 0:
        return []
    floor = int(job.min_free_bytes)
    workdir = shlex.quote(job.workdir)
    return [
        (
            f'        avail="$(df -B1 --output=avail -- {workdir} 2>/dev/null'
            ' | tail -n 1 | tr -dc 0-9)"'
        ),
        *_watch_stop(
            f'[[ -n "${{avail}}" ]] && (( avail < {floor} ))',
            f"the workdir filesystem has ${{avail}} bytes free (floor {floor})",
        ),
    ]


#: One bounded pass over ``out/`` and ``reply/``: the entry count (at most
#: the cap + 1: a tree past it is not walked further) and the bytes, each
#: entry counted at the larger of its apparent and its allocated size, so
#: neither a sparse nor a preallocated file hides what it holds.
_WATCH_STATS_AWK = (
    "awk '{n += 1; b = $2 * 512; t += ($1 > b ? $1 : b)} "
    'END {printf "%.0f %.0f\\n", n, t}\''
)


def _staged_watch_lines(job: StagedJob) -> list[str]:
    """A watchdog stops the step once the filesystem drops below its floor
    (checked first on every tick: a tree the step shaped cannot delay it),
    once ``out/`` and ``reply/`` pass their caps (the shared filesystem has
    no quota, and the collect caps apply only after the job), or once the
    step hides a directory there from it (the watchdog has no DAC override:
    ``find`` cannot see into a 0300 directory the step may still fill). The
    pod's output reaches the Slurm log capped."""
    out, reply = (shlex.quote(f"{job.workdir}/{d}") for d in ("out", "reply"))
    hidden = r"-type d \( ! -readable -o ! -executable \) -print -quit"
    cap = int(job.max_out_entries)
    return [
        "_step_watch() {",
        "    set +eo pipefail",
        "    local stats used entries hidden avail",
        # The sleep holds no stdio: killed with the watchdog, it may linger.
        f"    while sleep {int(job.watch_seconds)} </dev/null >/dev/null 2>&1; do",
        *_watch_floor_lines(job),
        (
            f"        stats=\"$(find {out} {reply} -printf '%s %b\\n' 2>/dev/null"
            f' | head -n {cap + 1} | {_WATCH_STATS_AWK})"'
        ),
        '        entries="${stats%% *}"',
        '        used="${stats##* }"',
        *_watch_stop(
            f"(( used > {int(job.max_out_bytes)} || entries > {cap} ))",
            "out/ and reply/ passed their caps (${used} bytes, ${entries} entries)",
        ),
        f'        hidden="$(find {out} {reply} {hidden} 2>/dev/null)"',
        *_watch_stop(
            '[[ -n "${hidden}" ]]',
            "a directory under out/ or reply/ is hidden from the watchdog (not u+rx)",
        ),
        "    done",
        "}",
        "_step_watch &",
        "_step_watch_pid=$!",
        "_step_log() {",
        f"    head -c {int(job.log_head_bytes)}",
        f"    tail -c {int(job.log_tail_bytes)}",
        "}",
        "",
    ]


def _staged_stale_lines() -> list[str]:
    """Step containers a scancel left in the node's store: its SIGKILL takes
    the podman client and conmon before ``--rm`` (peer review 2026-09-26).
    Removed when Slurm no longer lists their job — never when squeue does
    not answer — so their layers do not pile up in the graphroot."""
    listing = (
        "podman ps -a --filter name=^umrs-step- --filter status=exited "
        "--filter status=created --format '{{.Names}}' 2>/dev/null || true"
    )
    return [
        'if _step_live="$(squeue --me -h -o %i 2>/dev/null)"; then',
        "    while read -r _step_old; do",
        '        _step_id="${_step_old#umrs-step-}"',
        '        _step_id="${_step_id%%-*}"',
        '        if [[ -n "${_step_id}" && "${_step_id}" != "${SLURM_JOB_ID}" ]] \\',
        '            && ! grep -qx -- "${_step_id}" <<<"${_step_live}"; then',
        '            podman rm -f "${_step_old}" >/dev/null 2>&1 || true',
        "        fi",
        f"    done < <({listing})",
        "fi",
        "",
    ]


def render_staged_job_script(job: StagedJob) -> str:
    """The staged step's sbatch script (build spec §7). Nothing interpolated
    is echoed and every value from Python is quoted or validated. The pod's
    stdout and stderr go capped to ``log/slurm.err`` (the tail the server
    reads); the job's exit code is the pod's (``pipefail``)."""
    lines = _staged_directives(job) + _staged_prologue()
    lines += _staged_venue_lines(job.venue_script)
    lines += _staged_image_lines(job.image, job.image_archive_dir)
    lines += _staged_seccomp_lines(job)
    lines += _staged_pids_lines(job) + _staged_stale_lines() + _staged_watch_lines(job)
    lines += [" ".join(staged_podman_argv(job)) + " 2>&1 | _step_log >&2", ""]
    return "\n".join(lines)


@dataclass
class SlurmStepExecutor(StepExecutor):
    """Run workflow steps as Slurm jobs on GPU compute nodes.

    This executor is designed for the RCCS deployment pattern where
    UMRS runs on a login node and submits Slurm jobs to GPU compute
    nodes. Each step runs inside a podman container with GPU access.

    Configuration via environment variables:

    * ``UMRS_SLURM_PARTITION`` — Slurm partition (default: ``gpu``)
    * ``UMRS_SLURM_GPUS`` — Number of GPUs to request (default: ``1``);
      ``0`` omits the ``--gres`` directive entirely (required on
      partitions that reject explicit GPU requests, e.g. RCCS
      ``ng-dgx-m*``) while podman GPU passthrough still applies when
      the step wants a GPU
    * ``UMRS_SLURM_CPUS`` — CPUs per task (default: ``8``)
    * ``UMRS_SLURM_MEM`` — Memory per node (default: ``32G``); an
      **empty** value omits the ``--mem`` directive entirely (required
      on partitions that reject explicit memory requests)
    * ``UMRS_SLURM_TIME`` — Time limit (default: ``01:00:00``)
    * ``UMRS_SLURM_ACCOUNT`` — Slurm account (optional)
    * ``UMRS_SLURM_WORKDIR_ROOT`` — Parent directory for the per-step
      working directories (default: system temp). On clusters where
      the submit host's local temp is not visible from compute nodes
      this MUST point at a shared filesystem (e.g. Lustre), otherwise
      the worker never sees ``input.pkl``.
    * ``UMRS_STEP_IMAGE`` — Container image to run the step in
    * ``UMRS_STEP_GPU`` — Whether to attach GPU (default: ``0``)
    """

    partition: str = field(
        default_factory=lambda: os.environ.get("UMRS_SLURM_PARTITION", "gpu")
    )
    gpus: int = field(
        default_factory=lambda: int(os.environ.get("UMRS_SLURM_GPUS", "1"))
    )
    cpus: int = field(
        default_factory=lambda: int(os.environ.get("UMRS_SLURM_CPUS", "8"))
    )
    memory: str = field(default_factory=lambda: os.environ.get("UMRS_SLURM_MEM", "32G"))
    time_limit: str = field(
        default_factory=lambda: os.environ.get("UMRS_SLURM_TIME", "01:00:00")
    )
    account: str = field(
        default_factory=lambda: os.environ.get("UMRS_SLURM_ACCOUNT", "")
    )
    image: str = field(default_factory=lambda: os.environ.get("UMRS_STEP_IMAGE", ""))
    timeout_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("UMRS_STEP_EXECUTOR_TIMEOUT", "3600")
        )
    )
    workdir_root: str = field(
        default_factory=lambda: os.environ.get("UMRS_SLURM_WORKDIR_ROOT", "").strip()
    )
    poll_interval: int = field(default=5)
    keep_workdir: bool = field(default=False)

    def execute(
        self,
        *,
        entrypoint: str,
        config: dict[str, Any],
        ctx: StepContext,
    ) -> StepResult:
        """Execute a step as a Slurm job on a GPU compute node."""
        # Determine the container image
        image = self._resolve_image(ctx)

        # Uploaded workflow source tree (user step modules). The worker
        # runs in a fresh container on a compute node, so the modules
        # must be carried in explicitly — see _generate_sbatch_script.
        source_dir = self._resolve_source_dir(ctx)

        workdir = self._new_workdir()
        logger.info(
            "SlurmStepExecutor: workdir=%s, image=%s, entrypoint=%s, source_dir=%s",
            workdir,
            image,
            entrypoint,
            source_dir or "<none>",
        )

        try:
            # Serialize the step payload — must match step_worker / PodmanExecutor
            # nested ``ctx`` shape (including parent_results) so multi-step
            # DAGs can hand off via input_keys and parent_results.
            input_path = os.path.join(workdir, "input.pkl")
            output_path = os.path.join(workdir, "output.pkl")

            _INTERNAL_KEYS = {
                EXECUTOR_IMAGE_METADATA_KEY,
                EXECUTOR_PAYLOAD_S3_METADATA_KEY,
                EXECUTOR_GPU_METADATA_KEY,
                EXECUTOR_SOURCE_DIR_METADATA_KEY,
            }
            sanitised_metadata = {
                k: v for k, v in ctx.metadata.items() if k not in _INTERNAL_KEYS
            }
            ctx_payload = _serialise_ctx_payload(ctx, metadata=sanitised_metadata)
            payload = {
                "entrypoint": entrypoint,
                "config": dict(config),
                "ctx": ctx_payload,
                "s3_config": self._step_s3_config(ctx),
            }

            payload_blob = pickle.dumps(payload)
            if len(payload_blob) > _PAYLOAD_MAX_BYTES:
                raise StepExecutionError(
                    "setup",
                    f"payload size {len(payload_blob)} bytes exceeds hard cap "
                    f"({_PAYLOAD_MAX_BYTES} bytes); refactor the step to "
                    "pass large data via S3 keys, not parent_results",
                )

            self._write_bytes(input_path, payload_blob)
            source_dir = self._stage_source_dir(source_dir, workdir)

            # Generate the sbatch script
            sbatch_script = self._generate_sbatch_script(
                workdir=workdir,
                image=image,
                input_path=input_path,
                output_path=output_path,
                ctx=ctx,
                source_dir=source_dir,
            )

            # Write the sbatch script
            sbatch_path = os.path.join(workdir, "run_step.sbatch")
            self._write_bytes(sbatch_path, sbatch_script.encode("utf-8"))

            # Submit the Slurm job
            job_id = self._submit_sbatch(sbatch_path, workdir)
            logger.info(
                "SlurmStepExecutor: submitted job %s for step %s",
                job_id,
                entrypoint,
            )

            # Poll for completion
            result = self._poll_completion(
                job_id=job_id,
                output_path=output_path,
                workdir=workdir,
            )
            return result

        finally:
            if not self.keep_workdir:
                self._remove_workdir(workdir)

    # ── I/O hooks (local submit host; RemoteSlurmStepExecutor routes them
    #    over SSH to the cluster login node) ────────────────────────────────

    def _run_cmd(self, cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess:
        return _run_local(cmd, timeout=timeout)

    def _step_s3_config(self, ctx: StepContext) -> dict[str, Any]:
        """S3 connection settings as the compute node must use them."""
        return _serialise_s3_client_config(ctx.s3)

    def _new_workdir(self) -> str:
        """Per-step working directory. When workdir_root is configured it
        must exist and be on a filesystem the compute nodes can see (e.g.
        Lustre) — the whole point of the knob."""
        if not self.workdir_root:
            return tempfile.mkdtemp(prefix="slurm-step-")
        if not os.path.isdir(self.workdir_root):
            raise StepExecutionError(
                "setup",
                f"UMRS_SLURM_WORKDIR_ROOT={self.workdir_root!r} does not "
                "exist; create it (on a shared filesystem visible from "
                "compute nodes) or unset the variable",
            )
        return tempfile.mkdtemp(prefix="slurm-step-", dir=self.workdir_root)

    def _write_bytes(self, path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)

    def _read_bytes(self, path: str, max_bytes: int | None = None) -> bytes | None:
        """File content (at most *max_bytes*), or None when it does not exist."""
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return f.read() if max_bytes is None else f.read(max_bytes)

    def _read_reply(self, job_id: str, output_path: str) -> StepResult:
        """The step's reply as data (ADR-074 D-6): ``output.json`` first; an
        image that predates it leaves only ``output.pkl``, read through the
        restricted loader. Never ``pickle.loads`` on step-written bytes."""
        cap = reply_max_bytes()
        json_path = json_reply_path(output_path)
        blob = self._read_bytes(json_path, max_bytes=cap + 1)
        if blob is not None:
            return result_from_json(check_reply_size(blob, json_path, cap))
        blob = self._read_bytes(output_path, max_bytes=cap + 1)
        if blob is None:
            raise StepExecutionError(
                "worker",
                f"Job {job_id} completed but no reply was written: {json_path}",
            )
        logger.warning(
            "SlurmStepExecutor: job %s wrote only a legacy pickle reply "
            "(image predates JSON replies); read restricted",
            job_id,
        )
        return result_from_legacy_pickle(check_reply_size(blob, output_path, cap))

    def _stage_source_dir(self, source_dir: str, workdir: str) -> str:
        """Where the compute node sees the step source tree (same path locally)."""
        return source_dir

    def _remove_workdir(self, workdir: str) -> None:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    def _resolve_image(self, ctx: StepContext) -> str:
        """Resolve the container image to use for the step."""
        # Check per-step image override
        if EXECUTOR_IMAGE_METADATA_KEY in ctx.metadata:
            override = ctx.metadata[EXECUTOR_IMAGE_METADATA_KEY]
            if not isinstance(override, str):
                raise StepExecutionError(
                    "setup",
                    f"per-step image override must be a string, got "
                    f"{type(override).__name__}",
                )
            if not override.strip():
                raise StepExecutionError(
                    "setup",
                    "per-step image override is empty; remove the hint "
                    "or supply a valid image reference",
                )
            return _checked_image(override)

        # Fall back to deployment-wide default
        if not self.image:
            raise StepExecutionError(
                "setup",
                "SlurmStepExecutor requires a container image. "
                "Set UMRS_STEP_IMAGE or routing_hints.image.",
            )
        return _checked_image(self.image)

    @staticmethod
    def _resolve_source_dir(ctx: StepContext) -> str:
        """Resolve the uploaded workflow source dir (user step modules).

        Returns ``""`` when the runner did not provide one (legacy
        callers, image-baked steps). When provided it must be an
        absolute existing directory — the worker bind-mounts it
        read-only at the same path and puts it on ``sys.path`` via
        ``UMRS_STEP_SOURCE_DIR``; a wrong value would surface as a
        confusing import error deep in the worker, so fail fast here.
        """
        raw = ctx.metadata.get(EXECUTOR_SOURCE_DIR_METADATA_KEY)
        if raw is None:
            return ""
        if not isinstance(raw, str) or not raw.strip():
            raise StepExecutionError(
                "setup",
                f"workflow source dir hint must be a non-empty string, got "
                f"{type(raw).__name__}",
            )
        candidate = raw.strip()
        if not os.path.isabs(candidate):
            raise StepExecutionError(
                "setup",
                f"workflow source dir must be an absolute path, got {candidate!r}",
            )
        if not os.path.isdir(candidate):
            raise StepExecutionError(
                "setup",
                f"workflow source dir {candidate!r} does not exist on the "
                "submit host; re-upload the workflow or fix the registration",
            )
        return candidate

    def _generate_sbatch_script(
        self,
        *,
        workdir: str,
        image: str,
        input_path: str,
        output_path: str,
        ctx: StepContext,
        source_dir: str = "",
    ) -> str:
        """Generate the sbatch script for the Slurm job."""
        # Determine if GPU is needed (metadata hint → env default).
        gpu_hint = ctx.metadata.get(EXECUTOR_GPU_METADATA_KEY)
        if gpu_hint is None:
            needs_gpu = os.environ.get("UMRS_STEP_GPU", "0").strip() == "1"
        else:  # a real true, not the string "false" (as the staged lane)
            needs_gpu = gpu_hint is True or str(gpu_hint).strip().lower() in (
                "1",
                "true",
                "yes",
            )

        time_limit = self._format_time_limit(self.timeout_seconds) or self.time_limit

        # Build sbatch directives
        lines = ["#!/bin/bash"]
        lines.append("#SBATCH --job-name=umrs-step")
        lines.append(f"#SBATCH --partition={self.partition}")
        lines.append(f"#SBATCH --time={time_limit}")
        lines.append(f"#SBATCH --cpus-per-task={self.cpus}")
        # Empty memory means "let the partition default apply" — several
        # partitions (e.g. RCCS ng-dgx-m*) reject explicit --mem requests.
        if self.memory.strip():
            lines.append(f"#SBATCH --mem={self.memory}")
        lines.append(f"#SBATCH --output={workdir}/slurm-%j.out")
        lines.append(f"#SBATCH --error={workdir}/slurm-%j.err")

        if self.account:
            lines.append(f"#SBATCH --account={self.account}")

        # gpus=0 means "no explicit GPU request" — required on partitions
        # that reject --gres (e.g. RCCS ng-dgx-m*). Podman GPU passthrough
        # below is independent: the node may still expose its GPU to the
        # container via CDI when the partition allocates the whole node.
        if needs_gpu and self.gpus > 0:
            lines.append(f"#SBATCH --gres=gpu:{self.gpus}")

        lines.append("")
        lines.append("set -e")
        lines.append('echo "Starting step at $(date)"')
        # Everything interpolated into this script is quoted: a
        # double-quoted echo still expands $(...) (2026-09-25).
        lines.append(f"echo Workdir: {shlex.quote(workdir)}")
        lines.append(f"echo Image: {shlex.quote(image)}")
        lines.append("")

        input_rel = os.path.relpath(input_path, workdir)
        output_rel = os.path.relpath(output_path, workdir)
        network = os.environ.get("UMRS_STEP_PODMAN_NETWORK", "host").strip() or "host"
        s3_config = self._step_s3_config(ctx)

        parts = _podman_run_argv(
            network=network,
            workdir=workdir,
            input_rel=input_rel,
            output_rel=output_rel,
            source_dir=source_dir,
            needs_gpu=needs_gpu,
            s3_config=s3_config,
            image=image,
        )
        podman_cmd = " ".join(parts)

        # Never echo the command: it carries forwarded S3 credentials, and the
        # .out log is readable from the step's own workdir.
        lines.append('echo "Running the step worker container"')
        lines.append(podman_cmd)
        lines.append('echo "Step completed at $(date)"')
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_time_limit(timeout_seconds: float | None) -> str | None:
        """Convert timeout seconds to Slurm ``HH:MM:SS`` when usable."""
        if timeout_seconds is None:
            return None
        try:
            total = int(timeout_seconds)
        except (TypeError, ValueError):
            return None
        if total <= 0:
            return None
        hours, rem = divmod(total, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _submit_sbatch(self, sbatch_path: str, workdir: str) -> str:
        """Submit the sbatch script and return the job ID."""
        cmd = ["sbatch", sbatch_path]
        logger.info("Submitting: %s", " ".join(cmd))

        result = self._run_cmd(cmd, timeout=60)

        if result.returncode != 0:
            raise StepExecutionError(
                "worker",
                f"sbatch submission failed (rc={result.returncode}): "
                f"{result.stderr.strip()}",
            )

        # Parse job ID from "Submitted batch job 12345"
        output = result.stdout.strip()
        if "Submitted batch job" not in output:
            raise StepExecutionError(
                "worker",
                f"Unexpected sbatch output: {output}",
            )

        job_id = output.split()[-1]
        return job_id

    def _poll_completion(
        self,
        *,
        job_id: str,
        output_path: str,
        workdir: str,
    ) -> StepResult:
        """Poll squeue until the job completes, then read the output."""
        start_time = time.monotonic()
        deadline = start_time + self.timeout_seconds

        while time.monotonic() < deadline:
            # Check job status
            status = self._check_job_status(job_id)

            if status == "COMPLETED":
                result = self._read_reply(job_id, output_path)

                logger.info(
                    "SlurmStepExecutor: job %s completed successfully",
                    job_id,
                )
                # D-36 leg B: the only GPU-second number that counts is a
                # measured one — stamp the sacct accounting onto the result
                # so it rides the step record into the execution ledger.
                stamp_slurm_accounting(
                    result, measure_slurm_accounting(job_id, run=self._run_cmd)
                )
                return result

            elif status in TERMINAL_FAILURES:
                # Read the Slurm error log for diagnostics
                error_log = os.path.join(workdir, f"slurm-{job_id}.err")
                raw_log = self._read_bytes(error_log) or b""
                error_msg = raw_log.decode("utf-8", "replace")[-1000:]

                raise StepExecutionError(
                    "worker",
                    f"Slurm job {job_id} failed with status {status}. "
                    f"Error log: {error_msg}",
                )

            # Still running, wait and poll again
            time.sleep(self.poll_interval)

        # Timeout
        raise StepExecutionError(
            "worker",
            f"Slurm job {job_id} timed out after {self.timeout_seconds}s",
        )

    def _check_job_status(self, job_id: str) -> str:
        """Check the status of a Slurm job via squeue/sacct."""
        # Try squeue first (for running/pending jobs)
        cmd = [
            "squeue",
            "--job",
            job_id,
            "--format=%T",
            "--noheader",
        ]
        try:
            result = self._run_cmd(cmd, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                # Job is still in queue. Normalize identically to the sacct
                # branch below: array/het jobs emit one line per element —
                # take the first line, then strip trailing info tokens.
                state = result.stdout.strip().split("\n")[0].split()[0].upper()
                # Map Slurm states to our simplified status
                if state in ("PENDING", "CONFIGURING"):
                    return "PENDING"
                elif state in ("RUNNING", "COMPLETING"):
                    return "RUNNING"
                else:
                    return state
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Job not in squeue, check sacct for final state
        cmd = [
            "sacct",
            "--job",
            job_id,
            "--format=State",
            "--noheader",
            "--parsable2",
        ]
        try:
            result = self._run_cmd(cmd, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                state = result.stdout.strip().split("\n")[0].upper()
                # Remove any extra info like "COMPLETED by 12345"
                state = state.split()[0]
                return state
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Can't determine status, assume running
        return "RUNNING"
