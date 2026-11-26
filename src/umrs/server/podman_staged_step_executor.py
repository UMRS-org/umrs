"""Staged podman step executor — the local lane for uploaded step code.

ADR-091 D-1: a podman executor on the remote-slurm step contract
(:mod:`umrs.server.step_staging`), for a site with no HPC lane that opts in
(``step_executor.uploaded_code_lane: podman``, D-3; the stand-alone hardened
profile). The guarantees are the HPC lane's, on this host:

1. :func:`~umrs.server.step_staging.build_stage_plan` scope-checks the step's
   inputs before anything runs (only the run owner's buckets or the run's
   own store).
2. The server makes ``<root>/<run>-<step>-XXXXXXXX`` (0700, under
   ``UMRS_STEP_WORKDIR``, DEC-064-WD) with ``in out reply src log``, extracts
   the one :func:`~umrs.server.step_staging.stage_in` tar into it (regular
   files only, under the same caps) and writes ``input.pkl`` (no platform
   key, no routing value, no object-store endpoint).
3. One ``podman run`` with ``--network=none``, ``--cap-drop=all``,
   no-new-privileges, a read-only root, ``--unsetenv-all``, pids, memory and
   file-size limits, an image from the site's allow-list (``--pull=never``)
   and the staged mounts only. The podman CLI itself runs with a scrubbed
   environment, so no ``AWS_*`` or other server variable reaches it.
4. The reply is ``reply/output.json`` — JSON of at most 4 MiB, opened without
   following a link; a step that wrote only a pickle fails by name (the
   server never unpickles what a step wrote).
5. ``out/`` is walked without following links (regular, singly-linked files
   only) and streamed through :func:`~umrs.server.step_staging.collect_out`
   into ``runs/<run_id>/<step>/`` of the run's output bucket.
6. The workdir is removed afterwards (``UMRS_STEP_KEEP_WORKDIR=1`` keeps it).

The existing :class:`~umrs.server.step_executor.PodmanExecutor` is not this
lane (D-2): it exchanges pickles, defaults to host networking and hands the
step the object-store endpoint. The executor refuses every step unless the
site opted in, so a direct build without the opt-in runs nothing.
"""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import logging
import os
import pickle
import shutil
import stat
import subprocess
import tarfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, BinaryIO, ClassVar

from umrs.pipeline_sdk.step import StepContext, StepResult

from .remote_slurm_step_executor import (
    _INTERNAL_METADATA_KEYS,
    _PLATFORM_CONFIG_KEYS,
    REPLY_CAP_BYTES,
    _routed_keys,
    _stable_parents,
    _step_timeout,
    _upstream_parents,
    _without,
    safe_segment,
)
from .slurm_step_executor import (
    _POD_BASE_ENV,
    _STAGED_ENV,
    _STAGED_MOUNTS,
    STAGED_GPU_DEVICES,
    SlurmStepExecutor,
)
from .step_executor import (
    _PAYLOAD_MAX_BYTES,
    EXECUTOR_GPU_METADATA_KEY,
    EXECUTOR_IMAGE_METADATA_KEY,
    StepExecutionError,
    _serialise_ctx_payload,
    step_workdir,
)
from .step_reply import result_from_json
from .step_staging import (
    DEFAULT_CAPS,
    StageCaps,
    StagePlan,
    StageStreamError,
    build_stage_plan,
    collect_out,
    rewrite_output_keys,
    stage_in,
    verify_stage_summary,
)
from .step_trust import (
    PODMAN_STAGED_LANE,
    allowed_step_images,
    podman_lane_opted_in,
)

logger = logging.getLogger(__name__)

GiB = 1024**3
#: The local lane's staging caps: the HPC lane's, with a free-space floor a
#: workstation can meet (the HPC default is 200 GiB of a shared filesystem).
LOCAL_CAPS = dataclasses.replace(DEFAULT_CAPS, min_free_bytes=1 * GiB)
#: The workdir's entries; ``log`` holds podman's output and is never mounted.
WORKDIR_DIRS = ("in", "out", "reply", "src", "log")
#: The two tops a stage-in member may name (``in/...`` and ``src/...``).
_STAGE_TOPS = frozenset({"in", "src"})
_DEFAULT_SECONDS = 3600.0
_LOG_TAIL_CHARS = 1000
_REMOVE_SECONDS = 600
_CHUNK = 1024 * 1024
#: Directory levels below ``out/`` walked (one descriptor each), at most.
_MAX_OUT_DEPTH = 64
_TRUE = frozenset({"1", "true", "yes", "on"})
#: The podman CLI's own environment: what it needs to find its store and
#: socket, nothing of the server's (no credential, no endpoint, no proxy).
_PODMAN_ENV_KEEP = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "TMPDIR",
    "XDG_RUNTIME_DIR",
    "XDG_CONFIG_HOME",
    "XDG_DATA_HOME",
    "CONTAINER_HOST",
    "CONTAINER_CONNECTION",
    "CONTAINERS_CONF",
    "CONTAINERS_STORAGE_CONF",
    "CONTAINERS_REGISTRIES_CONF",
    "DBUS_SESSION_BUS_ADDRESS",
)
#: ``result.extra`` keys only an executor writes (a step cannot forge them).
_EXECUTOR_EXTRA_KEYS = ("slurm", "slurm_job", "podman_job")
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_CREATE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_seconds() -> float:
    try:
        return float(_env("UMRS_STEP_EXECUTOR_TIMEOUT") or 0)
    except ValueError:
        return 0.0


def podman_env() -> dict[str, str]:
    """The podman CLI's environment: :data:`_PODMAN_ENV_KEEP` only."""
    return {k: os.environ[k] for k in _PODMAN_ENV_KEEP if k in os.environ}


def staged_payload(entrypoint: str, config: dict[str, Any], ctx: StepContext) -> bytes:
    """``input.pkl`` as the HPC lane writes it: no executor-internal
    metadata, no platform routing value, no object-store endpoint, only the
    parents of the step's dependency closure. The server writes it; the
    step never sends a pickle back."""
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
            f"({_PAYLOAD_MAX_BYTES} bytes); pass large data as staged objects",
        )
    return blob


# ── piping a tar between a producer and a consumer ──────────────────────────


def _piped(
    produce: Callable[[BinaryIO], Any], consume: Callable[[BinaryIO], Any]
) -> tuple[Any, Any]:
    """Run *produce* (writes a stream) in a thread and *consume* (reads it)
    here, over a pipe; ``(produced, consumed)``. A refusal on the producer's
    side wins over the broken stream it leaves the consumer."""
    read_fd, write_fd = os.pipe()
    box: dict[str, Any] = {}
    writer = threading.Thread(
        target=_produce_into, args=(produce, write_fd, box), daemon=True
    )
    writer.start()
    try:
        with os.fdopen(read_fd, "rb") as source:
            consumed = consume(source)
            while source.read(_CHUNK):  # the end-of-archive padding
                pass
    except BaseException as exc:
        writer.join()
        refusal = _first_refusal(box.get("error"), exc)
        if refusal is exc:
            raise
        raise refusal from exc
    writer.join()
    if "error" in box:
        raise box["error"]
    return box.get("value"), consumed


def _produce_into(
    produce: Callable[[BinaryIO], Any], write_fd: int, box: dict[str, Any]
) -> None:
    try:
        with os.fdopen(write_fd, "wb") as sink:
            box["value"] = produce(sink)
    except Exception as exc:  # noqa: BLE001 — re-raised on the consumer's side
        box["error"] = exc


def _first_refusal(produced: BaseException | None, consumed: BaseException) -> Any:
    named = isinstance(produced, StepExecutionError)
    if named and not isinstance(produced, StageStreamError):
        return produced
    return consumed


# ── stage-in: the server's own tar into the fresh workdir ───────────────────


def _member_parts(name: str) -> list[str]:
    parts = name.split("/")
    bad = not parts or parts[0] not in _STAGE_TOPS or len(parts) < 2
    if bad or any(p in ("", ".", "..") or "\0" in p for p in parts):
        raise StepExecutionError("setup", f"stage-in member {name!r} refused")
    return parts


def _write_member(workdir: str, parts: list[str], source: Any) -> int:
    parent = os.path.join(workdir, *parts[:-1])
    os.makedirs(parent, mode=0o700, exist_ok=True)
    fd = os.open(os.path.join(parent, parts[-1]), _CREATE_FLAGS, 0o600)
    written = 0
    with os.fdopen(fd, "wb") as sink:
        while chunk := source.read(_CHUNK):
            sink.write(chunk)
            written += len(chunk)
    return written


def extract_stage(
    source: BinaryIO, workdir: str, caps: StageCaps = LOCAL_CAPS
) -> dict[str, int]:
    """Extract the stage-in tar into ``<workdir>/in`` and ``<workdir>/src``:
    regular files only, never over an existing name; the summary
    :func:`~umrs.server.step_staging.verify_stage_summary` checks."""
    max_files = caps.max_in_objects + caps.max_src_files + 1
    files = total = 0
    with tarfile.open(fileobj=source, mode="r|") as tar:
        for member in tar:
            if not member.isreg():
                raise StepExecutionError(
                    "setup", f"stage-in member {member.name!r} is not a regular file"
                )
            files += 1
            if files > max_files:
                raise StepExecutionError("setup", "stage-in has too many files")
            total += _write_member(
                workdir, _member_parts(member.name), tar.extractfile(member)
            )
    stats = os.statvfs(workdir)
    return {
        "files": files,
        "bytes": total,
        "free_bytes": stats.f_bavail * stats.f_frsize,
    }


# ── collect-out: what the step left in out/, no link followed ───────────────


def _open_dir_at(parent_fd: int, name: str) -> int:
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise StepExecutionError(
                "worker",
                f"/work/out entry {name!r} is a link; only files are collected",
            ) from exc
        raise


def _open_plain_at(dir_fd: int, name: str, rel: str) -> tuple[int, os.stat_result]:
    """A regular, singly-linked file: a link, FIFO, socket or device the
    step left is refused by name, never followed or read."""
    try:
        fd = os.open(name, _READ_FLAGS, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENXIO):
            raise StepExecutionError(
                "worker", f"/work/out entry {rel!r} is a link or a socket"
            ) from exc
        raise
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise StepExecutionError(
            "worker", f"/work/out entry {rel!r} is not a regular, singly-linked file"
        )
    return fd, info


def _add_file(tar: tarfile.TarFile, dir_fd: int, name: str, rel: str) -> None:
    fd, info = _open_plain_at(dir_fd, name, rel)
    member = tarfile.TarInfo(rel)
    member.size, member.mode = info.st_size, 0o600
    with os.fdopen(fd, "rb") as fh:
        tar.addfile(member, fh)


def _add_tree(
    tar: tarfile.TarFile, dir_fd: int, prefix: str, budget: list[int]
) -> None:
    for name in sorted(os.listdir(dir_fd)):
        budget[0] -= 1
        if budget[0] < 0:
            raise StepExecutionError("worker", "/work/out has too many entries")
        rel = f"{prefix}{name}"
        mode = os.stat(name, dir_fd=dir_fd, follow_symlinks=False).st_mode
        if not stat.S_ISDIR(mode):
            _add_file(tar, dir_fd, name, rel)
            continue
        if rel.count("/") >= _MAX_OUT_DEPTH:
            raise StepExecutionError("worker", "/work/out is nested too deep")
        sub = _open_dir_at(dir_fd, name)
        try:
            _add_tree(tar, sub, f"{rel}/", budget)
        finally:
            os.close(sub)


def out_tar(workdir: str, sink: BinaryIO, max_entries: int) -> int:
    """Write ``<workdir>/out`` as a tar of its regular files to *sink*."""
    wd_fd = os.open(workdir, _DIR_FLAGS)
    try:
        out_fd = _open_dir_at(wd_fd, "out")
    finally:
        os.close(wd_fd)
    budget = [max_entries]
    try:
        with tarfile.open(fileobj=sink, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            _add_tree(tar, out_fd, "", budget)
    finally:
        os.close(out_fd)
    return max_entries - budget[0]


def read_reply(workdir: str, cap: int = REPLY_CAP_BYTES) -> bytes | None:
    """``reply/output.json`` (no link followed, at most *cap* bytes), None
    when the step wrote none."""
    try:
        reply_fd = os.open(os.path.join(workdir, "reply"), _DIR_FLAGS)
    except FileNotFoundError:
        return None
    try:
        fd, info = _open_plain_at(reply_fd, "output.json", "reply/output.json")
    except FileNotFoundError:
        return None
    finally:
        os.close(reply_fd)
    with os.fdopen(fd, "rb") as fh:
        if info.st_size > cap:
            raise StepExecutionError("worker", f"step reply exceeds {cap} bytes")
        blob = fh.read(cap + 1)
    if len(blob) > cap:
        raise StepExecutionError("worker", f"step reply exceeds {cap} bytes")
    return blob


def _log_tail(workdir: str) -> str:
    try:
        with open(os.path.join(workdir, "log", "podman.log"), "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 64 * 1024))
            return fh.read().decode("utf-8", "replace")[-_LOG_TAIL_CHARS:]
    except OSError:
        return "<no log>"


# ── removing the workdir the step shaped ────────────────────────────────────


def remove_workdir(workdir: str) -> None:
    """Remove *workdir* whatever shape the step left in ``out/`` or
    ``reply/``: a directory it made read-only or unreadable for its owner, a
    tree deeper than Python recurses (as the HPC lane's helper ``rmwd``
    does). Never raises: a failure here must not replace the step's own
    outcome; a workdir that stays is logged by name."""
    try:
        shutil.rmtree(workdir)
    except FileNotFoundError:
        return
    except Exception:  # noqa: BLE001 — RecursionError too (one frame per level)
        _force_remove(workdir)


def _force_remove(workdir: str) -> None:
    """``chmod -R u+rwX`` then ``rm -rf``: neither follows a link it meets,
    neither recurses on the Python stack (POSIX tools, Linux and macOS)."""
    for argv in (["chmod", "-R", "u+rwX", "--", workdir], ["rm", "-rf", "--", workdir]):
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=_REMOVE_SECONDS,
                env=podman_env(),
                check=False,
            )
    if os.path.lexists(workdir):
        logger.warning("podman-staged: step workdir %s could not be removed", workdir)


@dataclass
class _Binding:
    cancelled: Callable[[], bool] | None = None
    gated: Any = None


@dataclass
class PodmanStagedStepExecutor:
    """Run uploaded step code in one hardened, staged podman container."""

    name: ClassVar[str] = PODMAN_STAGED_LANE
    #: The step's ``timeout_seconds`` is enforced on the container here.
    owns_step_deadline: ClassVar[bool] = True
    #: No routed endpoint reaches this lane (``--network=none``, no forwarder).
    endpoint_socket: ClassVar[bool] = False

    #: The platform step image (``step_executor.image``).
    image: str = field(default_factory=lambda: _env("UMRS_STEP_IMAGE"))
    podman_bin: str = field(
        default_factory=lambda: _env("UMRS_STEP_PODMAN_BIN") or "podman"
    )
    #: The workdir parent; None reads ``UMRS_STEP_WORKDIR`` (DEC-064-WD).
    workdir_root: str | None = None
    #: Caps ``timeout_seconds`` when > 0 (``UMRS_STEP_EXECUTOR_TIMEOUT``).
    timeout_seconds: float = field(default_factory=_env_seconds)
    memory: str = field(default_factory=lambda: _env("UMRS_STEP_MEMORY") or "8g")
    cpus: str = field(default_factory=lambda: _env("UMRS_STEP_CPUS"))
    pids_limit: int = field(
        default_factory=lambda: _env_int("UMRS_STEP_PIDS_LIMIT", 512)
    )
    gpu_form: str = field(default_factory=lambda: _env("UMRS_STEP_GPU_FORM") or "cdi")
    keep_workdir: bool = field(
        default_factory=lambda: _env("UMRS_STEP_KEEP_WORKDIR").lower() in _TRUE
    )
    caps: StageCaps = LOCAL_CAPS
    reply_cap: int = REPLY_CAP_BYTES
    poll_interval: float = 1.0
    _runs: dict[str, _Binding] = field(default_factory=dict, init=False, repr=False)

    # ── runner hooks ────────────────────────────────────────────────────────

    def bind_run(
        self,
        run_id: str,
        *,
        cancelled: Callable[[], bool] | None = None,
        on_job: Callable[[str, dict[str, Any]], None] | None = None,
        gated: Any = None,
    ) -> None:
        """The run's cancel flag; a gated run (ADR-112 B2) is recorded only
        to be refused: this lane runs no step under a person's HPC account."""
        del on_job  # no scheduler job: nothing for the reaper to chase
        self._runs[str(run_id)] = _Binding(cancelled, gated)

    # ── the step ────────────────────────────────────────────────────────────

    def execute(
        self, *, entrypoint: str, config: dict[str, Any], ctx: StepContext
    ) -> StepResult:
        try:
            return self._execute(entrypoint, config, ctx)
        except StepExecutionError:
            raise
        except Exception as exc:  # an S3, podman or OS failure fails the step
            raise StepExecutionError("worker", f"{type(exc).__name__}: {exc}") from exc

    def _execute(
        self, entrypoint: str, config: dict[str, Any], ctx: StepContext
    ) -> StepResult:
        self._check_lane(ctx)
        image = self._lane_image(ctx)
        gpu = self._gpu_devices(ctx)
        plan = build_stage_plan(ctx, caps=self.caps)
        source_dir = SlurmStepExecutor._resolve_source_dir(ctx)
        payload = staged_payload(entrypoint, config, ctx)
        workdir = self._make_workdir(ctx.run_id, plan.step)
        try:
            self._stage(workdir, ctx, plan, source_dir, payload)
            container = self._run(workdir, ctx, self.podman_argv(workdir, image, gpu))
            return self._result(workdir, ctx, plan, container)
        finally:
            self._remove(workdir)

    def _check_lane(self, ctx: StepContext) -> None:
        if not podman_lane_opted_in():
            raise StepExecutionError(
                "setup",
                "the local podman lane is off on this site "
                "(step_executor.uploaded_code_lane is not podman, ADR-091 D-3); "
                "nothing was run",
            )
        binding = self._runs.get(str(ctx.run_id))
        if binding is not None and binding.gated is not None:
            raise StepExecutionError(
                "setup",
                "this run carries your signed intent, but the local podman lane "
                "has no gated path: nothing was run (hpcaccess.lane_not_gated)",
            )

    def _lane_image(self, ctx: StepContext) -> str:
        """The platform step image or an allow-listed one (ADR-074 D-7)."""
        if not self.image:
            raise StepExecutionError(
                "setup",
                "the local podman lane has no platform step image "
                "(step_executor.image / UMRS_STEP_IMAGE); nothing was run",
            )
        hint = ctx.metadata.get(EXECUTOR_IMAGE_METADATA_KEY)
        image = hint.strip() if isinstance(hint, str) and hint.strip() else self.image
        if image != self.image and image not in allowed_step_images():
            raise StepExecutionError(
                "setup",
                f"image {image!r} is not an allowed step image on this site "
                "(ADR-074 D-7)",
            )
        return image

    def _gpu_devices(self, ctx: StepContext) -> tuple[str, ...]:
        """The devices of a step that asks for a GPU on a site that enables
        GPUs (the HPC lane's rule); CDI or ROCm only on this lane."""
        hint = ctx.metadata.get(EXECUTOR_GPU_METADATA_KEY)
        asked = hint is True or str(hint).strip().lower() in _TRUE
        if not (asked and _env("UMRS_STEP_GPU", "0") == "1"):
            return ()
        if self.gpu_form not in ("cdi", "rocm"):
            raise StepExecutionError(
                "setup",
                f"step_executor.gpu_form {self.gpu_form!r} is not cdi|rocm on the "
                "local podman lane",
            )
        return STAGED_GPU_DEVICES[self.gpu_form]

    # ── workdir and staging ─────────────────────────────────────────────────

    def _make_workdir(self, run_id: str, step: str) -> str:
        prefix = f"{safe_segment(run_id, 'run')}-{safe_segment(step, 'step')}-"
        workdir = step_workdir(prefix, self.workdir_root)
        if ":" in workdir or "," in workdir:
            shutil.rmtree(workdir, ignore_errors=True)
            raise StepExecutionError(
                "setup",
                f"step workdir {workdir!r} holds ':' or ',', which a podman "
                "volume cannot name; choose another step_executor.workdir_root",
            )
        os.chmod(workdir, 0o700)
        for sub in WORKDIR_DIRS:
            os.mkdir(os.path.join(workdir, sub), 0o700)
        return workdir

    def _stage(
        self,
        workdir: str,
        ctx: StepContext,
        plan: StagePlan,
        source_dir: str,
        payload: bytes,
    ) -> None:
        sent, summary = _piped(
            lambda sink: stage_in(
                plan, ctx.s3, sink, source_dir=source_dir or None, caps=self.caps
            ),
            lambda source: extract_stage(source, workdir, self.caps),
        )
        verify_stage_summary(sent, summary, self.caps)
        fd = os.open(os.path.join(workdir, "input.pkl"), _CREATE_FLAGS, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)

    def _remove(self, workdir: str) -> None:
        if self.keep_workdir:
            logger.info("podman-staged: keeping workdir %s", workdir)
            return
        remove_workdir(workdir)

    # ── the container ───────────────────────────────────────────────────────

    def podman_argv(
        self, workdir: str, image: str, gpu: tuple[str, ...] = (), name: str = ""
    ) -> list[str]:
        """``podman run`` of one staged step (ADR-091 D-1): no network, every
        capability dropped, no new privileges, a read-only root, the
        container's environment the base set and the staged names only."""
        argv = [
            self.podman_bin,
            "run",
            "--rm",
            "--init",
            "--pull=never",
            "--network=none",
            "--http-proxy=false",
            "--unsetenv-all",
            "--image-volume=ignore",
            "--workdir=/tmp",
            "--read-only",
            "--tmpfs=/tmp:rw,size=4g",
            "--cap-drop=all",
            "--security-opt=no-new-privileges",
            "--userns=keep-id",
            f"--user={os.getuid()}:{os.getgid()}",
            f"--name={name or self._container_name(workdir)}",
            *self._limits(),
        ]
        for assignment in (*_POD_BASE_ENV, *_STAGED_ENV):
            argv += ["-e", assignment]
        for entry, target, mode in _STAGED_MOUNTS:
            argv += ["-v", f"{workdir}/{entry}:{target}:{mode}"]
        return [
            *argv,
            *gpu,
            "--entrypoint",
            "python",
            image,
            "-m",
            "umrs.server.step_worker",
        ]

    def _limits(self) -> list[str]:
        limits = [f"--ulimit=fsize={int(self.caps.max_out_file_bytes)}"]
        if self.pids_limit > 0:
            limits.append(f"--pids-limit={int(self.pids_limit)}")
        if self.memory:
            limits.append(f"--memory={self.memory}")
        if self.cpus:
            limits.append(f"--cpus={self.cpus}")
        return limits

    @staticmethod
    def _container_name(workdir: str) -> str:
        return f"umrs-step-{os.path.basename(workdir)}-{uuid.uuid4().hex[:8]}"

    def _step_seconds(self, ctx: StepContext) -> float:
        limits = [_step_timeout(ctx) or _DEFAULT_SECONDS]
        if self.timeout_seconds > 0:
            limits.append(float(self.timeout_seconds))
        return min(limits)

    def _run(self, workdir: str, ctx: StepContext, argv: list[str]) -> str:
        """Run the container to its end, or stop it (cancel, deadline)."""
        container = next(a for a in argv if a.startswith("--name="))[len("--name=") :]
        deadline = time.monotonic() + self._step_seconds(ctx)
        with open(os.path.join(workdir, "log", "podman.log"), "wb") as log:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=podman_env(),
            )
            rc = self._wait(proc, ctx, deadline, container)
        if rc != 0:
            raise StepExecutionError(
                "worker",
                f"step container {container} exited with status {rc}. "
                f"Error log: {_log_tail(workdir)}",
            )
        return container

    def _wait(
        self, proc: subprocess.Popen, ctx: StepContext, deadline: float, container: str
    ) -> int:
        while True:
            try:
                return proc.wait(timeout=self.poll_interval)
            except subprocess.TimeoutExpired:
                reason = self._stop_reason(ctx, deadline)
            if reason:
                self._stop(proc, container)
                raise StepExecutionError("run", f"step container {container} {reason}")

    def _stop_reason(self, ctx: StepContext, deadline: float) -> str | None:
        binding = self._runs.get(str(ctx.run_id))
        if (
            binding is not None
            and binding.cancelled is not None
            and binding.cancelled()
        ):
            return "stopped: the workflow run was cancelled"
        if time.monotonic() > deadline:
            return f"stopped: ran past the step's {self._step_seconds(ctx):g}s deadline"
        return None

    def _stop(self, proc: subprocess.Popen, container: str) -> None:
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                [self.podman_bin, "rm", "-f", "-t", "0", container],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=60,
                env=podman_env(),
                check=False,
            )
        proc.kill()
        proc.wait()

    # ── reply and outputs ───────────────────────────────────────────────────

    def _result(
        self, workdir: str, ctx: StepContext, plan: StagePlan, container: str
    ) -> StepResult:
        blob = read_reply(workdir, self.reply_cap)
        if blob is None:
            raise StepExecutionError(
                "worker",
                f"step container {container} wrote no JSON reply "
                "(reply/output.json); the staged lane reads no other reply",
            )
        result = result_from_json(blob)
        mapping = self._collect(workdir, ctx, plan)
        result = rewrite_output_keys(result, mapping, ctx.output_bucket)
        for key in _EXECUTOR_EXTRA_KEYS:
            result.extra.pop(key, None)
        result.extra["podman_job"] = {"container": container}
        return result

    def _collect(
        self, workdir: str, ctx: StepContext, plan: StagePlan
    ) -> dict[str, str]:
        max_entries = 2 * self.caps.max_out_files + 8
        _count, mapping = _piped(
            lambda sink: out_tar(workdir, sink, max_entries),
            lambda source: collect_out(
                source, ctx.s3, ctx.output_bucket, ctx.run_id, plan.step, self.caps
            ),
        )
        return mapping


__all__ = [
    "LOCAL_CAPS",
    "PodmanStagedStepExecutor",
    "extract_stage",
    "out_tar",
    "podman_env",
    "read_reply",
    "remove_workdir",
    "staged_payload",
]
