"""Pluggable step execution boundary.

Slice 8 (Distributed step execution boundary) introduces a thin
abstraction over ``Step.setup() → Step.run() → Step.teardown()`` so
the workflow runner no longer hard-codes "execute in this process".

The two executors that ship today are:

* :class:`InProcessExecutor` — current behaviour. Instantiates the
  step class in the runner process and calls its lifecycle methods
  directly. Default for development and for legacy workflows.

* :class:`SubprocessExecutor` — spawns a fresh Python subprocess via
  ``python -m umrs.server.step_worker`` for each step. The step's
  ``setup/run/teardown`` runs in isolation, which keeps GPU memory,
  CUDA contexts, and any global state contained per step. Crashes in
  user code cannot bring down the runner; OOMs in the subprocess do
  not poison the parent.

Both executors satisfy the same protocol so the runner is agnostic to
which one is wired in. ``build_step_executor()`` is a factory that
respects ``UMRS_STEP_EXECUTOR=inproc|subprocess`` (default
``inproc``). Future executors (Singularity exec, k8s Job, gRPC to a
GPU worker pool) can be added without touching the runner.

Design constraints (from the development instructions):

* UMRS owns workflow execution; this module lives inside UMRS.
* Mini-RiVault is the security boundary; the subprocess executor is a
  process-level boundary, not a network boundary. Network-level
  isolation is delegated to the existing UMRS Singularity / k8s
  hybrid-orchestration layer.
* Public APIs (``Step`` ABC, ``StepContext``, ``StepResult``,
  ``workflow.json``) are unchanged.
"""

from __future__ import annotations

import importlib
import logging
import os
import pickle
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from umrs.core.zmq_security import ZMQSecurityConfig
from umrs.pipeline_sdk.step import Step, StepContext, StepResult

logger = logging.getLogger(__name__)


# Metadata key used by the runner to pass a per-step container image
# override to :class:`PodmanExecutor`. Marked with dunder underscores
# to signal "internal contract; not part of the user-step API". User
# steps should never read or write this key directly.
EXECUTOR_IMAGE_METADATA_KEY = "__executor_image__"

# Metadata key used by the runner to pass object-store payload config
# to :class:`PodmanExecutor` (Slice 11). When present, the executor
# uploads ``input.pkl`` to S3 and passes an S3 reference to the
# worker instead of relying on a host bind-mount.
# Shape: ``{"bucket": str, "prefix": str}``
EXECUTOR_PAYLOAD_S3_METADATA_KEY = "__payload_s3__"

# Metadata key for per-step GPU override (Slice 12). The runner
# lifts ``routing_hints.gpu`` into this key so :class:`PodmanExecutor`
# can decide per-step whether to attach a GPU device. Value is a
# ``bool``; non-bool triggers a setup-phase error.
EXECUTOR_GPU_METADATA_KEY = "__executor_gpu__"

# Default threshold: payloads below this size stay on local file even
# when S3 mode is enabled. Above it, the executor uploads to S3.
_PAYLOAD_THRESHOLD_BYTES = int(
    os.environ.get("UMRS_STEP_PAYLOAD_THRESHOLD_BYTES", "1048576")
)
# Hard cap: payloads above this size are rejected as a setup error to
# protect the object store and network from accidental massive pickles.
_PAYLOAD_MAX_BYTES = int(os.environ.get("UMRS_STEP_PAYLOAD_MAX_BYTES", "52428800"))


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------


class StepExecutionError(RuntimeError):
    """Raised when a step's execution fails inside an executor.

    ``phase`` is one of ``"setup"``, ``"run"``, or ``"worker"`` (the
    last one is reserved for transport-level failures such as the
    subprocess exiting before returning a payload).
    """

    def __init__(self, phase: str, message: str) -> None:
        super().__init__(f"{phase}() raised: {message}")
        self.phase = phase
        self.message = message


# ----------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------


class StepExecutor(Protocol):
    """Run a single step end-to-end and return its :class:`StepResult`.

    Implementations MUST drive the full ``setup → run → teardown``
    lifecycle and translate failures into :class:`StepExecutionError`
    so the runner has uniform exception-handling regardless of the
    backend.
    """

    def execute(
        self,
        *,
        entrypoint: str,
        config: dict[str, Any],
        ctx: StepContext,
    ) -> StepResult: ...


# ----------------------------------------------------------------------
# Helpers shared by all executors
# ----------------------------------------------------------------------


def _load_step_class(entrypoint: str) -> type[Step]:
    """Resolve ``"package.module:Class"`` to a Step subclass.

    Mirrors the loader the runner uses; kept here so the subprocess
    worker can use the same logic without importing the runner module
    (which pulls in heavy dependencies it doesn't need).
    """
    if ":" not in entrypoint:
        raise StepExecutionError(
            "setup", f"invalid entrypoint {entrypoint!r}; expected 'pkg.mod:Cls'"
        )
    module_path, _, class_name = entrypoint.partition(":")
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        raise StepExecutionError(
            "setup", f"could not import {module_path!r}: {exc}"
        ) from exc
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise StepExecutionError(
            "setup",
            f"module {module_path!r} has no attribute {class_name!r}",
        ) from exc
    if not (isinstance(cls, type) and issubclass(cls, Step)):
        raise StepExecutionError(
            "setup",
            f"{entrypoint!r} does not resolve to a Step subclass",
        )
    return cls


def _drive_lifecycle(
    cls: type[Step],
    config: dict[str, Any],
    ctx: StepContext,
) -> StepResult:
    """Run ``setup → run → teardown`` and translate failures."""
    instance = cls()
    try:
        instance.setup(config)
    except Exception as exc:  # noqa: BLE001
        raise StepExecutionError("setup", str(exc)) from exc
    try:
        result = instance.run(ctx)
    except Exception as exc:  # noqa: BLE001
        # Always attempt teardown so the step can release resources
        # even on failure — but never let teardown errors mask the
        # original run() error.
        try:
            instance.teardown()
        except Exception as t_exc:  # noqa: BLE001
            logger.warning("teardown after run() failure raised: %s", t_exc)
        raise StepExecutionError("run", str(exc)) from exc
    try:
        instance.teardown()
    except Exception as exc:  # noqa: BLE001
        # Teardown errors are logged but do not fail the step — the
        # result is already in hand.
        logger.warning("teardown raised: %s", exc)
    if not isinstance(result, StepResult):
        raise StepExecutionError(
            "run",
            f"step returned {type(result).__name__!r}, expected StepResult",
        )
    return result


def _payload_s3_key(
    prefix: str,
    run_id: str,
    step_name: str,
    filename: str,
    attempt: int = 0,
) -> str:
    """Build a deterministic S3 key for a step payload file.

    Layout::

        <prefix>/<run_id>/<step_name>/<attempt>/input.pkl
        <prefix>/<run_id>/<step_name>/<attempt>/output.pkl

    The ``attempt`` counter is always ``0`` today (the executor has no
    retry semantics), but the key layout reserves space for it so a
    future retry slice doesn't need a migration.
    """
    return f"{prefix}/{run_id}/{step_name}/{attempt}/{filename}"


# ----------------------------------------------------------------------
# In-process executor
# ----------------------------------------------------------------------


@dataclass
class InProcessExecutor:
    """Run the step in the runner's own process.

    This is the default — it preserves today's semantics exactly and
    is the only option that supports steps relying on shared in-memory
    state across calls (which is rare and discouraged, but exists).
    """

    name: str = "inproc"

    def execute(
        self,
        *,
        entrypoint: str,
        config: dict[str, Any],
        ctx: StepContext,
    ) -> StepResult:
        if EXECUTOR_IMAGE_METADATA_KEY in ctx.metadata:
            logger.debug(
                "image hint %r ignored (executor=%s)",
                ctx.metadata[EXECUTOR_IMAGE_METADATA_KEY],
                self.name,
            )
        if EXECUTOR_GPU_METADATA_KEY in ctx.metadata:
            logger.debug(
                "gpu hint %r ignored (executor=%s)",
                ctx.metadata[EXECUTOR_GPU_METADATA_KEY],
                self.name,
            )
        cls = _load_step_class(entrypoint)
        return _drive_lifecycle(cls, config, ctx)


# ----------------------------------------------------------------------
# Subprocess executor
# ----------------------------------------------------------------------


@dataclass
class SubprocessExecutor:
    """Run the step in a fresh Python subprocess.

    Sends a pickle payload to ``python -m umrs.server.step_worker`` on
    stdin and reads back a pickle reply on stdout. The boto3 S3 client
    on :class:`StepContext` is NOT picklable; the executor strips it
    and the worker reconstructs an equivalent client from the
    ``s3_config`` carried in the payload (or via env vars set on the
    child process, matching the parent's MinIO/S3 configuration).

    Slice 8 ships subprocess isolation as a feature flag
    (``UMRS_STEP_EXECUTOR=subprocess``); the default remains
    :class:`InProcessExecutor` so existing tests and workflows are
    untouched.
    """

    name: str = "subprocess"
    timeout_seconds: float | None = None
    python_executable: str | None = None

    def execute(
        self,
        *,
        entrypoint: str,
        config: dict[str, Any],
        ctx: StepContext,
    ) -> StepResult:
        if EXECUTOR_IMAGE_METADATA_KEY in ctx.metadata:
            logger.debug(
                "image hint %r ignored (executor=%s)",
                ctx.metadata[EXECUTOR_IMAGE_METADATA_KEY],
                self.name,
            )
        if EXECUTOR_GPU_METADATA_KEY in ctx.metadata:
            logger.debug(
                "gpu hint %r ignored (executor=%s)",
                ctx.metadata[EXECUTOR_GPU_METADATA_KEY],
                self.name,
            )
        # Strip the unpicklable boto3 client; the worker rebuilds one
        # from the s3_config metadata so the child has its own
        # connection pool. ``parent_results`` may contain numpy
        # arrays — pickle handles those natively.
        s3_config = _serialise_s3_client_config(ctx.s3)
        ctx_payload = _serialise_ctx_payload(ctx)
        payload = {
            "entrypoint": entrypoint,
            "config": dict(config),
            "ctx": ctx_payload,
            "s3_config": s3_config,
        }

        cmd = [
            self.python_executable or sys.executable,
            "-m",
            "umrs.server.step_worker",
        ]
        try:
            completed = subprocess.run(
                cmd,
                input=pickle.dumps(payload),
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise StepExecutionError(
                "worker", f"step worker timed out after {exc.timeout}s"
            ) from exc
        except FileNotFoundError as exc:
            raise StepExecutionError(
                "worker", f"could not launch worker: {exc}"
            ) from exc

        if completed.returncode != 0 and not completed.stdout:
            stderr = (completed.stderr or b"").decode(errors="replace")
            raise StepExecutionError(
                "worker",
                f"worker exited {completed.returncode} with no payload: "
                f"{stderr.strip()[:400]}",
            )

        try:
            reply = pickle.loads(completed.stdout)
        except Exception as exc:  # noqa: BLE001
            stderr = (completed.stderr or b"").decode(errors="replace")
            raise StepExecutionError(
                "worker",
                f"could not decode worker payload: {exc}; "
                f"stderr={stderr.strip()[:200]}",
            ) from exc

        if not isinstance(reply, dict) or "status" not in reply:
            raise StepExecutionError("worker", f"malformed worker reply: {reply!r}")
        if reply["status"] == "ok":
            result = reply.get("result")
            if not isinstance(result, StepResult):
                raise StepExecutionError(
                    "worker",
                    f"worker returned {type(result).__name__!r}, expected StepResult",
                )
            return result
        if reply["status"] == "error":
            phase = str(reply.get("phase") or "run")
            message = str(reply.get("message") or "<no message>")
            raise StepExecutionError(phase, message)
        raise StepExecutionError("worker", f"unknown reply status: {reply['status']!r}")


def _serialise_s3_client_config(s3: Any) -> dict[str, Any]:
    """Best-effort capture of a boto3 client's connection settings.

    The worker uses these to rebuild an equivalent client without
    needing to inherit any open connections from the parent. Falls
    back to env vars when the client object doesn't expose its
    configuration cleanly (which is the common case for stub clients
    used in tests — those tests use the in-process executor anyway).
    """
    if s3 is None:
        return {}
    out: dict[str, Any] = {}
    meta = getattr(s3, "meta", None)
    if meta is not None:
        endpoint = getattr(meta, "endpoint_url", None)
        if endpoint:
            out["endpoint_url"] = endpoint
        region = getattr(meta, "region_name", None)
        if region:
            out["region_name"] = region
    return out


def _serialise_ctx_payload(
    ctx: StepContext,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the nested ``ctx`` dict shared by all container/subprocess executors."""
    return {
        "input_bucket": ctx.input_bucket,
        "input_keys": list(ctx.input_keys),
        "output_bucket": ctx.output_bucket,
        "parent_output_bucket": getattr(ctx, "parent_output_bucket", "") or "",
        "config": dict(ctx.config),
        "metadata": dict(metadata if metadata is not None else ctx.metadata),
        "run_id": ctx.run_id,
        "parent_results": dict(ctx.parent_results),
    }


# ----------------------------------------------------------------------
# Podman executor
# ----------------------------------------------------------------------


@dataclass
class PodmanExecutor:
    """Run the step in a fresh Podman container.

    Builds on the same payload schema as :class:`SubprocessExecutor`,
    but transports it via a host tmpdir mounted into the container at
    ``/umrs-step``. The worker reads ``input.pkl`` from the env-var
    path and writes ``output.pkl`` next to it; stdin/stdout are kept
    free for diagnostic logs.

    Designed for the "podman on Slurm compute node" topology:
    the runner is already running inside a podman container with the
    podman socket mounted so it can spawn *sibling* step containers
    inside the existing Slurm allocation. Cluster-level isolation
    (Slurm, k8s, Singularity) remains UMRS hybrid-orchestration's
    responsibility — :class:`PodmanExecutor` is a per-step worker
    launcher, not an inference-server lifecycle manager.

    Required configuration:

    * ``image`` — container image carrying the umrs Python package
      and the step's dependencies. Required; a clear error is raised
      at :func:`build_step_executor` time if ``UMRS_STEP_IMAGE`` is
      unset.

    Optional knobs (all env-driven via :func:`build_step_executor`):

    * ``UMRS_STEP_PODMAN_BIN`` — defaults to ``"podman"``.
    * ``UMRS_STEP_PODMAN_NETWORK`` — defaults to ``"host"`` so the
      worker can reach MinIO/S3. Set to ``"none"`` for steps that
      operate purely on ``parent_results``.
    * ``UMRS_STEP_PODMAN_EXTRA_ARGS`` — whitespace-split escape hatch
      injected before the image name (e.g. ``--pull=missing``).
    * ``UMRS_STEP_GPU=1`` — adds ``--device nvidia.com/gpu=all``.
    * ``UMRS_STEP_KEEP_WORKDIR=1`` — preserves the host tmpdir on
      exit for post-mortem inspection.
    * ``UMRS_STEP_EXECUTOR_TIMEOUT`` — seconds (shared with
      :class:`SubprocessExecutor`).
    """

    image: str
    name: str = "podman"
    timeout_seconds: float | None = None
    podman_bin: str = "podman"
    network: str = "host"
    gpu: bool = False
    extra_args: list[str] = field(default_factory=list)
    keep_workdir: bool = False

    def execute(
        self,
        *,
        entrypoint: str,
        config: dict[str, Any],
        ctx: StepContext,
    ) -> StepResult:
        # Slice 10: per-step image override.
        image = self._resolve_image(ctx)

        # Slice 12: per-step GPU override.
        gpu = self._resolve_gpu(ctx)

        # Slice 11: decide payload transport (local file vs S3).
        payload_s3_cfg = self._resolve_payload_s3(ctx)

        # Strip executor-internal keys from the metadata that crosses
        # into the worker — user steps never need to see them.
        _INTERNAL_KEYS = {
            EXECUTOR_IMAGE_METADATA_KEY,
            EXECUTOR_PAYLOAD_S3_METADATA_KEY,
            EXECUTOR_GPU_METADATA_KEY,
        }
        sanitised_metadata = {
            k: v for k, v in ctx.metadata.items() if k not in _INTERNAL_KEYS
        }

        s3_config = _serialise_s3_client_config(ctx.s3)
        ctx_payload = _serialise_ctx_payload(ctx, metadata=sanitised_metadata)
        payload = {
            "entrypoint": entrypoint,
            "config": dict(config),
            "ctx": ctx_payload,
            "s3_config": s3_config,
        }

        # Serialize once — reused for size checks and upload/write.
        payload_blob = pickle.dumps(payload)
        payload_size = len(payload_blob)

        # Hard cap: reject oversized payloads before touching storage.
        if payload_size > _PAYLOAD_MAX_BYTES:
            raise StepExecutionError(
                "setup",
                f"payload size {payload_size} bytes exceeds hard cap "
                f"({_PAYLOAD_MAX_BYTES} bytes); refactor the step to "
                "pass large data via S3 keys, not parent_results",
            )

        workdir = tempfile.mkdtemp(prefix="umrs-step-")
        input_path = os.path.join(workdir, "input.pkl")
        output_path = os.path.join(workdir, "output.pkl")
        container_name = f"umrs-step-{uuid.uuid4().hex[:12]}"

        # Decide whether this invocation actually uses S3 transport.
        use_s3 = payload_s3_cfg is not None and payload_size >= _PAYLOAD_THRESHOLD_BYTES
        input_s3_key: str | None = None
        output_s3_key: str | None = None

        try:
            if use_s3:
                assert payload_s3_cfg is not None  # type narrowing
                bucket = payload_s3_cfg["bucket"]
                prefix = payload_s3_cfg["prefix"]
                step_name = config.get("step_name", "unknown")
                run_id = ctx.run_id or "unknown"
                input_s3_key = _payload_s3_key(prefix, run_id, step_name, "input.pkl")
                output_s3_key = _payload_s3_key(prefix, run_id, step_name, "output.pkl")
                try:
                    ctx.s3.put_object(
                        Bucket=bucket,
                        Key=input_s3_key,
                        Body=payload_blob,
                    )
                except Exception as exc:  # noqa: BLE001
                    raise StepExecutionError(
                        "setup",
                        f"failed to upload payload to "
                        f"s3://{bucket}/{input_s3_key}: {exc}",
                    ) from exc
                logger.debug(
                    "uploaded %d-byte payload to s3://%s/%s",
                    payload_size,
                    bucket,
                    input_s3_key,
                )
            else:
                # Local file transport (existing behaviour).
                with open(input_path, "wb") as fh:
                    fh.write(payload_blob)

            cmd = self._build_command(
                container_name=container_name,
                workdir=workdir,
                s3_config=s3_config,
                image=image,
                gpu=gpu,
                input_s3_key=input_s3_key,
                output_s3_key=output_s3_key,
                payload_s3_bucket=(payload_s3_cfg["bucket"] if use_s3 else None),
            )
            try:
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                self._best_effort_remove(container_name)
                raise StepExecutionError(
                    "worker",
                    f"podman step worker timed out after {exc.timeout}s",
                ) from exc
            except FileNotFoundError as exc:
                raise StepExecutionError(
                    "worker", f"could not launch podman: {exc}"
                ) from exc

            # ---- Read the reply (S3 or local) -------------------------
            if use_s3:
                assert payload_s3_cfg is not None
                assert output_s3_key is not None
                reply = self._download_s3_reply(
                    ctx.s3,
                    payload_s3_cfg["bucket"],
                    output_s3_key,
                    completed,
                )
            else:
                reply = self._read_local_reply(output_path, completed)

            return self._interpret_reply(reply)
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_image(self, ctx: StepContext) -> str:
        """Return the image to run for this step.

        Slice 10 contract:
        * ``ctx.metadata[EXECUTOR_IMAGE_METADATA_KEY]`` absent → use
          ``self.image`` (the deployment-wide default from
          ``UMRS_STEP_IMAGE``).
        * Present and a non-empty string → use it.
        * Present but empty or not a string → ``setup``-phase error.
          Treated as a misconfiguration (templating bug) rather than
          silently falling back to the default.
        """
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
            return override
        if not self.image:
            raise StepExecutionError(
                "setup",
                "PodmanExecutor requires a non-empty image (set "
                "UMRS_STEP_IMAGE or routing_hints.image)",
            )
        return self.image

    def _resolve_payload_s3(self, ctx: StepContext) -> dict[str, str] | None:
        """Return S3 payload config if the runner enabled it, else None.

        Expected shape in ``ctx.metadata``::

            {"bucket": "my-bucket", "prefix": "umrs"}

        Non-dict or missing ``bucket`` → silently ignored (local mode).
        """
        raw = ctx.metadata.get(EXECUTOR_PAYLOAD_S3_METADATA_KEY)
        if raw is None:
            return None
        if not isinstance(raw, dict) or "bucket" not in raw:
            logger.debug("payload S3 hint ignored (bad shape): %r", raw)
            return None
        return {
            "bucket": str(raw["bucket"]),
            "prefix": str(raw.get("prefix", "umrs")),
        }

    def _resolve_gpu(self, ctx: StepContext) -> bool:
        """Return whether this step should attach a GPU device.

        Slice 12 tri-state contract:
        * ``ctx.metadata[EXECUTOR_GPU_METADATA_KEY]`` absent → use
          ``self.gpu`` (the deployment-wide default from
          ``UMRS_STEP_GPU``).
        * Present and ``True``  → GPU on.
        * Present and ``False`` → GPU off (overrides global default).
        * Present but non-bool → ``setup``-phase error.
        """
        if EXECUTOR_GPU_METADATA_KEY not in ctx.metadata:
            return self.gpu
        override = ctx.metadata[EXECUTOR_GPU_METADATA_KEY]
        if not isinstance(override, bool):
            raise StepExecutionError(
                "setup",
                f"per-step gpu override must be a bool, got {type(override).__name__}",
            )
        return override

    # ---- Reply helpers (shared between local + S3 paths) ----

    @staticmethod
    def _read_local_reply(output_path: str, completed: Any) -> dict[str, Any]:
        if not os.path.isfile(output_path):
            stderr = (completed.stderr or b"").decode(errors="replace")
            raise StepExecutionError(
                "worker",
                f"podman worker exited {completed.returncode} "
                f"without writing {output_path}: "
                f"{stderr.strip()[:400]}",
            )
        try:
            with open(output_path, "rb") as fh:
                return pickle.loads(fh.read())
        except Exception as exc:  # noqa: BLE001
            stderr = (completed.stderr or b"").decode(errors="replace")
            raise StepExecutionError(
                "worker",
                f"could not decode podman worker output: {exc}; "
                f"stderr={stderr.strip()[:200]}",
            ) from exc

    @staticmethod
    def _download_s3_reply(
        s3: Any,
        bucket: str,
        key: str,
        completed: Any,
    ) -> dict[str, Any]:
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            return pickle.loads(obj["Body"].read())
        except Exception as exc:  # noqa: BLE001
            stderr = (completed.stderr or b"").decode(errors="replace")
            raise StepExecutionError(
                "worker",
                f"could not download reply from s3://{bucket}/{key}: "
                f"{exc}; podman exit={completed.returncode}; "
                f"stderr={stderr.strip()[:200]}",
            ) from exc

    @staticmethod
    def _interpret_reply(reply: dict[str, Any]) -> StepResult:
        if not isinstance(reply, dict) or "status" not in reply:
            raise StepExecutionError(
                "worker", f"malformed podman worker reply: {reply!r}"
            )
        if reply["status"] == "ok":
            result = reply.get("result")
            if not isinstance(result, StepResult):
                raise StepExecutionError(
                    "worker",
                    f"podman worker returned {type(result).__name__!r}, "
                    "expected StepResult",
                )
            return result
        if reply["status"] == "error":
            phase = str(reply.get("phase") or "run")
            message = str(reply.get("message") or "<no message>")
            raise StepExecutionError(phase, message)
        raise StepExecutionError(
            "worker", f"unknown podman reply status: {reply['status']!r}"
        )

    def _build_command(
        self,
        *,
        container_name: str,
        workdir: str,
        s3_config: dict[str, Any],
        image: str,
        gpu: bool,
        input_s3_key: str | None = None,
        output_s3_key: str | None = None,
        payload_s3_bucket: str | None = None,
    ) -> list[str]:
        """Construct the ``podman run`` invocation for the step worker."""
        cmd: list[str] = [
            self.podman_bin,
            "run",
            "--rm",
            "--name",
            container_name,
            f"--network={self.network}",
            "--security-opt=no-new-privileges",
            "-v",
            f"{workdir}:/umrs-step:Z",
        ]

        # Payload transport env vars.
        if input_s3_key and output_s3_key and payload_s3_bucket:
            # S3 mode: worker reads/writes via S3 instead of local files.
            cmd.extend(
                [
                    "-e",
                    f"UMRS_STEP_INPUT_S3={payload_s3_bucket}/{input_s3_key}",
                    "-e",
                    f"UMRS_STEP_OUTPUT_S3={payload_s3_bucket}/{output_s3_key}",
                ]
            )
        else:
            # Local file mode (Slice 9 original behaviour).
            cmd.extend(
                [
                    "-e",
                    "UMRS_STEP_INPUT=/umrs-step/input.pkl",
                    "-e",
                    "UMRS_STEP_OUTPUT=/umrs-step/output.pkl",
                ]
            )

        # Forward S3 connection settings so the in-container worker can
        # rebuild a boto3 client matching the parent's MinIO endpoint.
        endpoint = s3_config.get("endpoint_url") or os.environ.get("MINIO_ENDPOINT")
        if endpoint:
            cmd.extend(["-e", f"MINIO_ENDPOINT={endpoint}"])
        region = s3_config.get("region_name") or os.environ.get("AWS_REGION")
        if region:
            cmd.extend(["-e", f"AWS_REGION={region}"])

        # Forward AWS/MinIO credentials so the worker can authenticate.
        # boto3 reads these from env; we never embed them in the payload.
        for cred_var in (
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        ):
            val = os.environ.get(cred_var)
            if val:
                cmd.extend(["-e", f"{cred_var}={val}"])

        if gpu:
            cmd.extend(["--device", "nvidia.com/gpu=all"])

        if self.extra_args:
            cmd.extend(self.extra_args)

        cmd.extend(
            [
                image,
                "python",
                "-m",
                "umrs.server.step_worker",
            ]
        )
        return cmd

    def _best_effort_remove(self, container_name: str) -> None:
        """Force-remove a step container after a timeout. Errors are logged."""
        try:
            subprocess.run(
                [self.podman_bin, "rm", "-f", container_name],
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("podman rm -f %s failed: %s", container_name, exc)


# ----------------------------------------------------------------------
# Pool worker (persistent container)
# ----------------------------------------------------------------------


class _PooledWorker:
    """A persistent Podman worker communicating via length-prefixed pickle.

    Wraps a ``podman run -i`` :class:`subprocess.Popen` handle. The
    container runs ``python -m umrs.server.step_worker --pool`` which
    loops over stdin/stdout: ``[4-byte big-endian length][pickle]``.

    The caller must hold exclusive access while ``run_task`` is
    active. The pool executor manages concurrency by treating each
    ``_PooledWorker`` as a single-threaded resource — idle workers
    sit in a ``deque``, and busy workers are held by their thread.
    """

    def __init__(
        self,
        *,
        proc: subprocess.Popen,
        container_name: str,
        image: str,
        gpu: bool,
        podman_bin: str = "podman",
    ) -> None:
        self.proc = proc
        self.container_name = container_name
        self.image = image
        self.gpu = gpu
        self._podman_bin = podman_bin
        self.last_used_at: float = time.monotonic()

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def run_task(
        self, payload_blob: bytes, timeout: float | None = None
    ) -> dict[str, Any]:
        """Send a task and block until the reply arrives.

        Raises :class:`StepExecutionError` on protocol or worker
        failure.
        """
        if not self.is_alive():
            raise StepExecutionError("worker", "pool worker exited unexpectedly")
        stdin = self.proc.stdin
        stdout = self.proc.stdout
        assert stdin is not None
        assert stdout is not None

        try:
            stdin.write(struct.pack(">I", len(payload_blob)))
            stdin.write(payload_blob)
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise StepExecutionError(
                "worker", f"could not send to pool worker: {exc}"
            ) from exc

        try:
            header = stdout.read(4)
            if not header or len(header) < 4:
                stderr_bytes = self.proc.stderr or b""
                if hasattr(stderr_bytes, "read"):
                    stderr_bytes = stderr_bytes.read()
                stderr_text = (stderr_bytes or b"").decode(errors="replace")
                raise StepExecutionError(
                    "worker",
                    f"pool worker closed stdout without reply; "
                    f"stderr={stderr_text.strip()[:400]}",
                )
            length = struct.unpack(">I", header)[0]
            data = stdout.read(length)
            if len(data) < length:
                raise StepExecutionError(
                    "worker",
                    f"pool worker reply truncated: expected {length} bytes, "
                    f"got {len(data)}",
                )
        except StepExecutionError:
            raise
        except (BrokenPipeError, OSError) as exc:
            raise StepExecutionError(
                "worker", f"could not read from pool worker: {exc}"
            ) from exc

        try:
            return pickle.loads(data)
        except Exception as exc:  # noqa: BLE001
            raise StepExecutionError(
                "worker", f"could not decode pool worker reply: {exc}"
            ) from exc

    def stop(self) -> None:
        """Shut down the worker by closing stdin and waiting."""
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception as exc:  # noqa: BLE001
            logger.warning("pool worker stop failed: %s", exc)
        finally:
            # Best-effort container removal.
            try:
                subprocess.run(
                    [self._podman_bin, "rm", "-f", self.container_name],
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except Exception:  # noqa: BLE001
                pass


@dataclass
class WorkerPoolExecutor:
    """Run steps in persistent Podman containers, reusing across tasks.

    Workers are keyed by ``(image, gpu)`` — steps that share both
    properties reuse the same container, amortising container startup
    and Python import overhead.

    **Slice 14 pool controls:**

    * ``max_workers_per_key`` — cap on live workers per ``(image, gpu)``
      key. When the cap is hit, callers block up to
      ``acquire_timeout`` seconds waiting for a release.
    * ``acquire_timeout`` — seconds to wait for a worker when the key
      is at capacity. ``None`` means wait forever.
    * ``idle_timeout`` — seconds after which an idle worker is
      eligible for eviction. At least one warm worker per key is
      always kept. Eviction happens lazily on the next acquire.
    * ``prewarm`` — if ``True``, the first :meth:`execute` call for a
      key spawns one extra idle worker proactively.

    Communication uses the same length-prefixed pickle protocol as
    ``step_worker.pool_main()`` over ``podman run -i`` stdin/stdout.
    """

    image: str
    name: str = "pool"
    timeout_seconds: float | None = None
    podman_bin: str = "podman"
    network: str = "host"
    gpu: bool = False
    extra_args: list[str] = field(default_factory=list)

    # -- Slice 14 pool controls --
    max_workers_per_key: int = 4
    acquire_timeout: float | None = 30.0
    idle_timeout: float | None = 300.0
    prewarm: bool = False

    # -- internal state --
    _pool: dict[tuple[str, bool], deque[_PooledWorker]] = field(
        default_factory=dict, init=False, repr=False
    )
    _all_workers: list[_PooledWorker] = field(
        default_factory=list, init=False, repr=False
    )
    _active_count: dict[tuple[str, bool], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _conds: dict[tuple[str, bool], threading.Condition] = field(
        default_factory=dict, init=False, repr=False
    )

    def execute(
        self,
        *,
        entrypoint: str,
        config: dict[str, Any],
        ctx: StepContext,
    ) -> StepResult:
        # Resolve per-step overrides (same pattern as PodmanExecutor).
        image = self._resolve_image(ctx)
        gpu = self._resolve_gpu(ctx)

        # Strip executor-internal keys from worker-visible metadata.
        _INTERNAL_KEYS = {
            EXECUTOR_IMAGE_METADATA_KEY,
            EXECUTOR_PAYLOAD_S3_METADATA_KEY,
            EXECUTOR_GPU_METADATA_KEY,
        }
        sanitised_metadata = {
            k: v for k, v in ctx.metadata.items() if k not in _INTERNAL_KEYS
        }

        s3_config = _serialise_s3_client_config(ctx.s3)
        ctx_payload = _serialise_ctx_payload(ctx, metadata=sanitised_metadata)
        payload = {
            "entrypoint": entrypoint,
            "config": dict(config),
            "ctx": ctx_payload,
            "s3_config": s3_config,
        }

        payload_blob = pickle.dumps(payload)
        payload_size = len(payload_blob)
        if payload_size > _PAYLOAD_MAX_BYTES:
            raise StepExecutionError(
                "setup",
                f"payload size {payload_size} bytes exceeds hard cap "
                f"({_PAYLOAD_MAX_BYTES} bytes); refactor the step to "
                "pass large data via S3 keys, not parent_results",
            )

        key = (image, gpu)
        t0 = time.monotonic()
        worker = self._acquire_worker(key, image, gpu)
        wait_ms = (time.monotonic() - t0) * 1000
        pool_hit = wait_ms < 5  # heuristic: < 5ms ≈ idle-pool hit
        logger.debug(
            "pool acquire key=%s wait_ms=%.1f hit=%s",
            key,
            wait_ms,
            pool_hit,
        )
        try:
            reply = worker.run_task(payload_blob)
        except StepExecutionError:
            # Worker is broken — discard it rather than returning to pool.
            self._discard_worker(key, worker)
            raise
        else:
            self._release_worker(key, worker)

        return self._interpret_reply(reply)

    # ------------------------------------------------------------------
    # Pool management (Slice 14: condition-variable blocking)
    # ------------------------------------------------------------------

    def _cond_for(self, key: tuple[str, bool]) -> threading.Condition:
        """Return (or lazily create) the condition variable for *key*."""
        cond = self._conds.get(key)
        if cond is None:
            cond = threading.Condition(self._lock)
            self._conds[key] = cond
        return cond

    def _acquire_worker(
        self, key: tuple[str, bool], image: str, gpu: bool
    ) -> _PooledWorker:
        deadline = (
            time.monotonic() + self.acquire_timeout
            if self.acquire_timeout is not None
            else None
        )
        cond = self._cond_for(key)
        first_use = False
        with cond:
            while True:
                # 1) Try idle pool (evicting stale workers along the way).
                worker = self._pop_idle(key)
                if worker is not None:
                    return worker

                # 2) Below cap — spawn a new worker (outside the lock).
                count = self._active_count.get(key, 0)
                if count < self.max_workers_per_key:
                    self._active_count[key] = count + 1
                    first_use = count == 0
                    break

                # 3) At cap — block until a release or timeout.
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StepExecutionError(
                            "setup",
                            f"pool acquire timeout ({self.acquire_timeout}s) "
                            f"for key={key}; all {self.max_workers_per_key} "
                            "workers busy",
                        )
                    cond.wait(remaining)
                else:
                    cond.wait()

        # Spawn outside the lock to avoid holding it during container start.
        try:
            worker = self._start_worker(image, gpu)
        except Exception:
            # Rollback count so waiters aren't permanently stuck.
            with cond:
                self._active_count[key] = self._active_count.get(key, 1) - 1
                cond.notify()
            raise

        # Prewarm: on first use of a key, spawn one extra idle worker.
        if self.prewarm and first_use:
            try:
                self._prewarm_key(key, image, gpu)
            except Exception:  # noqa: BLE001
                logger.debug("prewarm failed for key=%s (non-fatal)", key)

        return worker

    def _pop_idle(self, key: tuple[str, bool]) -> _PooledWorker | None:
        """Pop a live idle worker, evicting stale / dead ones.

        Must be called with ``_lock`` held.
        """
        idle = self._pool.get(key)
        if not idle:
            return None

        now = time.monotonic()
        while idle:
            w = idle.popleft()
            if not w.is_alive():
                self._active_count[key] = max(self._active_count.get(key, 1) - 1, 0)
                logger.debug("discarding dead idle worker %s", w.container_name)
                continue
            # Evict if past idle timeout, but keep at least one warm.
            if (
                self.idle_timeout is not None
                and (now - w.last_used_at) > self.idle_timeout
                and (len(idle) + 1) > 1  # +1 because we already popped w
            ):
                self._active_count[key] = max(self._active_count.get(key, 1) - 1, 0)
                logger.debug(
                    "evicting idle worker %s (idle %.0fs)",
                    w.container_name,
                    now - w.last_used_at,
                )
                # Stop in background to avoid blocking acquire.
                threading.Thread(target=w.stop, daemon=True, name="pool-evict").start()
                continue
            return w
        return None

    def _release_worker(self, key: tuple[str, bool], worker: _PooledWorker) -> None:
        cond = self._cond_for(key)
        with cond:
            if worker.is_alive():
                worker.last_used_at = time.monotonic()
                self._pool.setdefault(key, deque()).append(worker)
            else:
                self._active_count[key] = max(self._active_count.get(key, 1) - 1, 0)
                logger.debug(
                    "pool worker %s died during task, discarding",
                    worker.container_name,
                )
            cond.notify()

    def _discard_worker(self, key: tuple[str, bool], worker: _PooledWorker) -> None:
        """Stop a failed worker without returning it to the pool."""
        cond = self._cond_for(key)
        with cond:
            self._active_count[key] = max(self._active_count.get(key, 1) - 1, 0)
            cond.notify()
        try:
            worker.stop()
        except Exception:  # noqa: BLE001
            pass

    def _prewarm_key(self, key: tuple[str, bool], image: str, gpu: bool) -> None:
        """Spawn one extra idle worker for *key* if under cap."""
        cond = self._cond_for(key)
        with cond:
            count = self._active_count.get(key, 0)
            if count >= self.max_workers_per_key:
                return
            self._active_count[key] = count + 1

        try:
            w = self._start_worker(image, gpu)
        except Exception:
            with cond:
                self._active_count[key] = max(self._active_count.get(key, 1) - 1, 0)
            raise
        with cond:
            w.last_used_at = time.monotonic()
            self._pool.setdefault(key, deque()).append(w)

    def _start_worker(self, image: str, gpu: bool) -> _PooledWorker:
        """Launch a persistent Podman container in pool mode."""
        container_name = f"umrs-pool-{uuid.uuid4().hex[:12]}"
        cmd: list[str] = [
            self.podman_bin,
            "run",
            "-i",  # keep stdin open
            "--rm",
            "--name",
            container_name,
            f"--network={self.network}",
            "--security-opt=no-new-privileges",
        ]

        # Forward S3 credentials so the worker can build a boto3 client.
        for env_var in (
            "MINIO_ENDPOINT",
            "AWS_REGION",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
        ):
            val = os.environ.get(env_var)
            if val:
                cmd.extend(["-e", f"{env_var}={val}"])

        # Force unbuffered Python so framed protocol works.
        cmd.extend(["-e", "PYTHONUNBUFFERED=1"])

        if gpu:
            cmd.extend(["--device", "nvidia.com/gpu=all"])

        if self.extra_args:
            cmd.extend(self.extra_args)

        cmd.extend(
            [
                image,
                "python",
                "-u",
                "-m",
                "umrs.server.step_worker",
                "--pool",
            ]
        )

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise StepExecutionError(
                "worker", f"could not launch podman: {exc}"
            ) from exc

        worker = _PooledWorker(
            proc=proc,
            container_name=container_name,
            image=image,
            gpu=gpu,
            podman_bin=self.podman_bin,
        )
        with self._lock:
            self._all_workers.append(worker)
        return worker

    def shutdown(self) -> None:
        """Stop all pooled workers and remove their containers."""
        with self._lock:
            workers = list(self._all_workers)
            self._all_workers.clear()
            self._pool.clear()
            self._active_count.clear()
            self._conds.clear()
        for w in workers:
            try:
                w.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pool shutdown: could not stop %s: %s",
                    w.container_name,
                    exc,
                )

    # ------------------------------------------------------------------
    # Resolution helpers (same logic as PodmanExecutor)
    # ------------------------------------------------------------------

    def _resolve_image(self, ctx: StepContext) -> str:
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
            return override
        if not self.image:
            raise StepExecutionError(
                "setup",
                "WorkerPoolExecutor requires a non-empty image (set "
                "UMRS_STEP_IMAGE or routing_hints.image)",
            )
        return self.image

    def _resolve_gpu(self, ctx: StepContext) -> bool:
        if EXECUTOR_GPU_METADATA_KEY not in ctx.metadata:
            return self.gpu
        override = ctx.metadata[EXECUTOR_GPU_METADATA_KEY]
        if not isinstance(override, bool):
            raise StepExecutionError(
                "setup",
                f"per-step gpu override must be a bool, got {type(override).__name__}",
            )
        return override

    @staticmethod
    def _interpret_reply(reply: dict[str, Any]) -> StepResult:
        if not isinstance(reply, dict) or "status" not in reply:
            raise StepExecutionError(
                "worker", f"malformed pool worker reply: {reply!r}"
            )
        if reply["status"] == "ok":
            result = reply.get("result")
            if not isinstance(result, StepResult):
                raise StepExecutionError(
                    "worker",
                    f"pool worker returned {type(result).__name__!r}, "
                    "expected StepResult",
                )
            return result
        if reply["status"] == "error":
            phase = str(reply.get("phase") or "run")
            message = str(reply.get("message") or "<no message>")
            raise StepExecutionError(phase, message)
        raise StepExecutionError(
            "worker", f"unknown pool reply status: {reply['status']!r}"
        )


# ----------------------------------------------------------------------
# Remote worker (ZMQ REQ) — Slice 16
# ----------------------------------------------------------------------


_REMOTE_WORKER_COOLDOWN = float(os.environ.get("UMRS_REMOTE_WORKER_COOLDOWN", "10"))

_WORKER_CAPABILITY_TTL = float(os.environ.get("UMRS_WORKER_CAPABILITY_TTL", "30"))


class _RemoteWorker:
    """ZMQ-based remote worker with same behavioural contract as _PooledWorker.

    Connects a ZMQ REQ socket to a pre-started worker running
    ``step_worker.py --zmq-pool``. The worker is externally managed
    (started by Slurm/k8s/manually) — this class only manages the
    socket lifecycle.

    Hardening (Slice 16b):
    - ZMQ SNDTIMEO/RCVTIMEO for clean timeouts (no manual poll).
    - Socket recreation after timeout — REQ/REP deadlock trap fix.
    - Cooldown backoff after failure (``UMRS_REMOTE_WORKER_COOLDOWN``).
    """

    def __init__(
        self,
        *,
        address: str,
        zmq_context: Any,
        timeout: float | None = None,
        cooldown_sec: float = _REMOTE_WORKER_COOLDOWN,
    ) -> None:
        import zmq  # type: ignore

        self.address = address
        self._zmq = zmq
        self._zmq_ctx = zmq_context
        self._timeout = timeout
        self._cooldown_sec = cooldown_sec
        self._alive = True
        self._last_failure_at: float | None = None
        self.last_used_at: float = time.monotonic()
        self._sock = self._make_socket()
        # Slice 17: capability cache (no RPC in router; cached here).
        self.capabilities: dict[str, Any] = {}
        self._last_capability_refresh: float = 0
        self._capability_ttl = _WORKER_CAPABILITY_TTL
        # Slice 18a/19: dynamic capacity (slots). Starts at env or 1;
        # updated by capability refresh when GPU info is available.
        self.capacity: int = int(os.environ.get("UMRS_REMOTE_WORKER_SLOTS", "1"))
        # Slice 19: previous slots for smoothing (shrink fast, grow slow).
        self._prev_slots: int | None = None

    def _make_socket(self) -> Any:
        sock = self._zmq_ctx.socket(self._zmq.REQ)
        timeout_ms = int(self._timeout * 1000) if self._timeout else -1
        sock.setsockopt(self._zmq.SNDTIMEO, timeout_ms)
        sock.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
        ZMQSecurityConfig.from_env().apply_to_socket(
            sock,
            server=False,
            context=self._zmq_ctx,
        )
        sock.connect(self.address)
        return sock

    def _recreate_socket(self) -> None:
        try:
            self._sock.close(linger=0)
        except Exception:  # noqa: BLE001
            pass
        self._sock = self._make_socket()

    def is_alive(self) -> bool:
        if not self._alive:
            # Check if cooldown has expired → auto-reset.
            if (
                self._last_failure_at is not None
                and (time.monotonic() - self._last_failure_at) >= self._cooldown_sec
            ):
                self._alive = True
                self._last_failure_at = None
                self._recreate_socket()
                return True
            return False
        return True

    def _mark_failed(self) -> None:
        self._alive = False
        self._last_failure_at = time.monotonic()

    def refresh_capabilities(self) -> dict[str, Any]:
        """Lazy-fetch capabilities from the remote worker via ZMQ ping.

        Returns cached capabilities if the TTL has not expired.
        Never blocks the router — failures return last-known state.
        """
        now = time.monotonic()
        if (now - self._last_capability_refresh) < self._capability_ttl:
            return self.capabilities
        if not self.is_alive():
            return self.capabilities
        try:
            ping_blob = pickle.dumps({"type": "ping"})
            self._sock.send(ping_blob)
            data = self._sock.recv()
            reply = pickle.loads(data)
            if isinstance(reply, dict) and reply.get("type") == "pong":
                caps = reply.get("capabilities", {})
                if isinstance(caps, dict):
                    self.capabilities = caps
                    # Slice 19: update capacity with smoothing.
                    raw_slots = caps.get("slots")
                    if isinstance(raw_slots, int) and raw_slots >= 1:
                        self._update_capacity(raw_slots)
            self._last_capability_refresh = now
        except Exception:  # noqa: BLE001
            # On failure: keep last-known, don't break socket.
            self._recreate_socket()
        return self.capabilities

    def _update_capacity(self, new_slots: int) -> None:
        """Smooth capacity changes: shrink immediately, grow slowly."""
        if self._prev_slots is None:
            self._prev_slots = new_slots
            self.capacity = new_slots
            return
        if new_slots <= self._prev_slots:
            # Shrink immediately — avoid OOM.
            self.capacity = new_slots
        else:
            # Grow slowly: move halfway toward new.
            self.capacity = self._prev_slots + max(
                1, (new_slots - self._prev_slots) // 2
            )
        self._prev_slots = self.capacity

    def run_task(
        self, payload_blob: bytes, timeout: float | None = None
    ) -> dict[str, Any]:
        """Send a task and block until the reply arrives.

        Raises :class:`StepExecutionError` on transport failure.
        """
        if not self.is_alive():
            raise StepExecutionError(
                "worker", f"remote worker {self.address} marked dead"
            )

        try:
            self._sock.send(payload_blob)
        except self._zmq.Again:
            self._mark_failed()
            self._recreate_socket()
            raise StepExecutionError(
                "worker",
                f"remote worker {self.address} send timeout",
            ) from None
        except Exception as exc:  # noqa: BLE001
            self._mark_failed()
            self._recreate_socket()
            raise StepExecutionError(
                "worker", f"could not send to remote worker {self.address}: {exc}"
            ) from exc

        try:
            data = self._sock.recv()
        except self._zmq.Again:
            self._mark_failed()
            self._recreate_socket()
            raise StepExecutionError(
                "worker",
                f"remote worker {self.address} timeout after {self._timeout}s",
            ) from None
        except Exception as exc:  # noqa: BLE001
            self._mark_failed()
            self._recreate_socket()
            raise StepExecutionError(
                "worker", f"could not recv from remote worker {self.address}: {exc}"
            ) from exc

        self.last_used_at = time.monotonic()
        try:
            return pickle.loads(data)
        except Exception as exc:  # noqa: BLE001
            raise StepExecutionError(
                "worker", f"could not decode remote worker reply: {exc}"
            ) from exc

    def stop(self) -> None:
        """Close the ZMQ socket. Does NOT stop the remote process."""
        self._alive = False
        try:
            self._sock.close(linger=0)
        except Exception:  # noqa: BLE001
            pass


@dataclass
class RemotePoolExecutor:
    """Run steps on pre-started remote workers via ZMQ REQ/REP (Slice 16).

    Workers are assumed already running (started by Slurm, k8s, or
    manually via ``step_worker.py --zmq-pool``). This executor does
    NOT manage worker lifecycle.

    Config via env:
        ``UMRS_REMOTE_WORKERS`` — comma-separated ZMQ addresses
        ``UMRS_WORKER_ACQUIRE_TIMEOUT`` — seconds to wait for a free worker
        ``UMRS_STEP_EXECUTOR_TIMEOUT`` — per-task timeout on the REQ/REP round-trip
    """

    workers: list[str] = field(default_factory=list)
    name: str = "remote"
    timeout_seconds: float | None = None
    acquire_timeout: float | None = 30.0

    # -- internal state --
    _zmq_ctx: Any = field(default=None, init=False, repr=False)
    _pool: dict[str, deque[_RemoteWorker]] = field(
        default_factory=dict, init=False, repr=False
    )
    _active_count: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _conds: dict[str, threading.Condition] = field(
        default_factory=dict, init=False, repr=False
    )
    _rr_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        import zmq  # type: ignore

        self._zmq_ctx = zmq.Context()

    def execute(
        self,
        *,
        entrypoint: str,
        config: dict[str, Any],
        ctx: StepContext,
    ) -> StepResult:
        _INTERNAL_KEYS = {
            EXECUTOR_IMAGE_METADATA_KEY,
            EXECUTOR_PAYLOAD_S3_METADATA_KEY,
            EXECUTOR_GPU_METADATA_KEY,
        }
        sanitised_metadata = {
            k: v for k, v in ctx.metadata.items() if k not in _INTERNAL_KEYS
        }

        s3_config = _serialise_s3_client_config(ctx.s3)
        ctx_payload = _serialise_ctx_payload(ctx, metadata=sanitised_metadata)
        payload = {
            "entrypoint": entrypoint,
            "config": dict(config),
            "ctx": ctx_payload,
            "s3_config": s3_config,
        }

        payload_blob = pickle.dumps(payload)
        payload_size = len(payload_blob)
        if payload_size > _PAYLOAD_MAX_BYTES:
            raise StepExecutionError(
                "setup",
                f"payload size {payload_size} bytes exceeds hard cap "
                f"({_PAYLOAD_MAX_BYTES} bytes); refactor the step to "
                "pass large data via S3 keys, not parent_results",
            )

        worker = self._acquire_worker()
        address = worker.address
        logger.debug(
            "remote execute worker=%s entrypoint=%s",
            address,
            entrypoint,
        )
        try:
            reply = worker.run_task(payload_blob)
        except StepExecutionError:
            self._discard_worker(address, worker)
            raise
        else:
            self._release_worker(address, worker)

        return WorkerPoolExecutor._interpret_reply(reply)

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def _cond_for(self, key: str) -> threading.Condition:
        cond = self._conds.get(key)
        if cond is None:
            cond = threading.Condition(self._lock)
            self._conds[key] = cond
        return cond

    def _acquire_worker(self) -> _RemoteWorker:
        if not self.workers:
            raise StepExecutionError(
                "setup",
                "RemotePoolExecutor has no workers configured "
                "(set UMRS_REMOTE_WORKERS)",
            )
        deadline = (
            time.monotonic() + self.acquire_timeout
            if self.acquire_timeout is not None
            else None
        )

        n = len(self.workers)
        with self._lock:
            while True:
                # Round-robin: try each address for an idle worker.
                for i in range(n):
                    idx = (self._rr_index + i) % n
                    addr = self.workers[idx]
                    idle = self._pool.get(addr)
                    if idle:
                        w = idle.popleft()
                        if w.is_alive():
                            self._rr_index = (idx + 1) % n
                            return w
                        self._active_count[addr] = max(
                            self._active_count.get(addr, 1) - 1, 0
                        )

                # Each address → at most `capacity` concurrent requests (Slice 18a).
                for i in range(n):
                    idx = (self._rr_index + i) % n
                    addr = self.workers[idx]
                    count = self._active_count.get(addr, 0)
                    # Get capacity from any existing worker, else env default.
                    cap = self._get_capacity(addr)
                    if count < cap:
                        self._active_count[addr] = count + 1
                        self._rr_index = (idx + 1) % n
                        # Create worker outside lock? No — socket creation is fast.
                        worker = _RemoteWorker(
                            address=addr,
                            zmq_context=self._zmq_ctx,
                            timeout=self.timeout_seconds,
                        )
                        return worker

                # All workers busy — block.
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StepExecutionError(
                            "setup",
                            f"remote pool acquire timeout "
                            f"({self.acquire_timeout}s); all "
                            f"{len(self.workers)} workers busy",
                        )
                    # Wait on any condition (use first address).
                    cond = self._cond_for(self.workers[0])
                    cond.wait(remaining)
                else:
                    cond = self._cond_for(self.workers[0])
                    cond.wait()

    def _release_worker(self, address: str, worker: _RemoteWorker) -> None:
        with self._lock:
            if worker.is_alive():
                worker.last_used_at = time.monotonic()
                self._pool.setdefault(address, deque()).append(worker)
            else:
                self._active_count[address] = max(
                    self._active_count.get(address, 1) - 1, 0
                )
            # Notify all conditions since any address might be free.
            for cond in self._conds.values():
                cond.notify()

    def _discard_worker(self, address: str, worker: _RemoteWorker) -> None:
        with self._lock:
            self._active_count[address] = max(self._active_count.get(address, 1) - 1, 0)
            for cond in self._conds.values():
                cond.notify()
        try:
            worker.stop()
        except Exception:  # noqa: BLE001
            pass

    def _get_capacity(self, address: str) -> int:
        """Return the concurrency limit for *address*.

        Checks any pooled worker's capacity first (updated by
        capability refresh). Falls back to env default.
        """
        idle = self._pool.get(address)
        if idle:
            for w in idle:
                if w.capacity > 0:
                    return w.capacity
        return int(os.environ.get("UMRS_REMOTE_WORKER_SLOTS", "1"))

    def shutdown(self) -> None:
        """Close all sockets and terminate the ZMQ context."""
        with self._lock:
            for idle_deque in self._pool.values():
                for w in idle_deque:
                    try:
                        w.stop()
                    except Exception:  # noqa: BLE001
                        pass
            self._pool.clear()
            self._active_count.clear()
            self._conds.clear()
        if self._zmq_ctx is not None:
            try:
                self._zmq_ctx.term()
            except Exception:  # noqa: BLE001
                pass


# ----------------------------------------------------------------------
# ROUTER-based executor (Slice 18b)
# ----------------------------------------------------------------------


@dataclass
class RouterPoolExecutor:
    """Run steps on DEALER workers via a ZMQ ROUTER socket (Slice 18b).

    Unlike :class:`RemotePoolExecutor` (REQ/REP per worker), this
    executor binds a single ROUTER socket.  Workers connect with DEALER
    sockets.  Requests carry a ``correlation_id`` so the executor can
    match asynchronous replies to outstanding requests, enabling
    multiple in-flight requests per worker without head-of-line
    blocking.

    Config via env:
        ``UMRS_ROUTER_BIND`` — ZMQ bind address for the ROUTER socket
        ``UMRS_STEP_EXECUTOR_TIMEOUT`` — per-task timeout
    """

    bind_address: str = "tcp://0.0.0.0:5560"
    name: str = "router"
    timeout_seconds: float | None = None
    session_curve_provider: Any = None

    # -- internal state --
    _zmq_ctx: Any = field(default=None, init=False, repr=False)
    _sock: Any = field(default=None, init=False, repr=False)
    _pending: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _recv_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _shutdown_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _zmq_authenticator: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        import zmq  # type: ignore

        self._zmq = zmq
        self._zmq_ctx = zmq.Context()
        self._sock = self._zmq_ctx.socket(zmq.ROUTER)

        zmq_sec = ZMQSecurityConfig.from_env()
        # Inject session provider when using ephemeral session CURVE mode
        if zmq_sec.is_session_mode and self.session_curve_provider is not None:
            if zmq_sec.credentials_provider is not self.session_curve_provider:
                from dataclasses import replace as _dc_replace

                zmq_sec = _dc_replace(
                    zmq_sec, credentials_provider=self.session_curve_provider
                )

        self._zmq_authenticator = zmq_sec.apply_to_socket(
            self._sock,
            server=True,
            context=self._zmq_ctx,
        )
        self._sock.bind(self.bind_address)
        logger.info("RouterPoolExecutor listening on %s", self.bind_address)
        # Background thread to receive replies.
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="router-recv"
        )
        self._recv_thread.start()

    def _recv_loop(self) -> None:
        """Background loop: receive ROUTER frames and resolve pending futures."""
        while not self._shutdown_event.is_set():
            try:
                if self._sock.poll(timeout=500, flags=self._zmq.POLLIN) == 0:
                    continue
                frames = self._sock.recv_multipart()
                # ROUTER frames: [identity, empty, payload]
                if len(frames) < 2:
                    continue
                data = frames[-1]
                reply = pickle.loads(data)
                cid = reply.get("correlation_id") if isinstance(reply, dict) else None
                if cid is None:
                    continue
                with self._lock:
                    future = self._pending.pop(cid, None)
                if future is not None:
                    future["reply"] = reply
                    future["event"].set()
            except Exception:  # noqa: BLE001
                if self._shutdown_event.is_set():
                    break

    def execute(
        self,
        *,
        entrypoint: str,
        config: dict[str, Any],
        ctx: StepContext,
    ) -> StepResult:
        _INTERNAL_KEYS = {
            EXECUTOR_IMAGE_METADATA_KEY,
            EXECUTOR_PAYLOAD_S3_METADATA_KEY,
            EXECUTOR_GPU_METADATA_KEY,
        }
        sanitised_metadata = {
            k: v for k, v in ctx.metadata.items() if k not in _INTERNAL_KEYS
        }

        s3_config = _serialise_s3_client_config(ctx.s3)
        ctx_payload = _serialise_ctx_payload(ctx, metadata=sanitised_metadata)

        correlation_id = str(uuid.uuid4())
        payload = {
            "entrypoint": entrypoint,
            "config": dict(config),
            "ctx": ctx_payload,
            "s3_config": s3_config,
            "correlation_id": correlation_id,
        }

        payload_blob = pickle.dumps(payload)
        if len(payload_blob) > _PAYLOAD_MAX_BYTES:
            raise StepExecutionError(
                "setup",
                f"payload size {len(payload_blob)} exceeds cap "
                f"({_PAYLOAD_MAX_BYTES}); use S3 for large data",
            )

        future: dict[str, Any] = {
            "event": threading.Event(),
            "reply": None,
        }
        with self._lock:
            self._pending[correlation_id] = future

        # Send to any available DEALER worker via ROUTER.
        # ROUTER needs [identity, empty, payload] but we can broadcast
        # to the first connected peer by sending without identity prefix
        # — ZMQ ROUTER requires explicit identity.  For simplicity we
        # keep track of connected workers via their pong identities.
        # However, in a minimal implementation we send to all connected
        # peers and the first to reply wins via correlation_id.
        #
        # In practice, the ROUTER socket dispatches round-robin to
        # connected DEALER peers when the identity prefix is omitted.
        # We send [empty_frame, payload] as the envelope.
        try:
            self._sock.send_multipart([b"", payload_blob])
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._pending.pop(correlation_id, None)
            raise StepExecutionError(
                "worker",
                f"router send failed: {exc}",
            ) from exc

        timeout = self.timeout_seconds
        if not future["event"].wait(timeout=timeout):
            with self._lock:
                self._pending.pop(correlation_id, None)
            raise StepExecutionError(
                "worker",
                f"router task timeout after {timeout}s "
                f"(correlation_id={correlation_id})",
            )

        return WorkerPoolExecutor._interpret_reply(future["reply"])

    def shutdown(self) -> None:
        """Stop the receiver thread and close the ROUTER socket."""
        self._shutdown_event.set()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=5)
        if self._sock is not None:
            try:
                self._sock.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
        if self._zmq_authenticator is not None:
            try:
                self._zmq_authenticator.stop()
            except Exception:  # noqa: BLE001
                pass
            self._zmq_authenticator = None
        if self._zmq_ctx is not None:
            try:
                self._zmq_ctx.term()
            except Exception:  # noqa: BLE001
                pass


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def build_step_executor(
    name: str | None = None,
) -> StepExecutor:
    """Return the executor named by ``UMRS_STEP_EXECUTOR`` (or ``name``).

    Values:
        ``inproc`` (default) â†’ :class:`InProcessExecutor`
        ``subprocess``       â†’ :class:`SubprocessExecutor`
        ``podman``           â†’ :class:`PodmanExecutor`
        ``pool``             â†’ :class:`WorkerPoolExecutor`
        ``remote``           â†’ :class:`RemotePoolExecutor`
        ``router``           â†’ :class:`RouterPoolExecutor`

    Unknown values fall back to ``inproc`` with a warning so a typo
    in deployment config doesn't crash the workflow runner.
    """
    chosen = (name or os.environ.get("UMRS_STEP_EXECUTOR") or "inproc").lower()
    if chosen == "inproc":
        return InProcessExecutor()
    if chosen == "slurm":
        from .slurm_step_executor import SlurmStepExecutor

        return SlurmStepExecutor()
    if chosen == "subprocess":
        return SubprocessExecutor(timeout_seconds=_executor_timeout())
    if chosen == "podman":
        image = os.environ.get("UMRS_STEP_IMAGE", "").strip()
        if not image:
            raise RuntimeError(
                "UMRS_STEP_EXECUTOR=podman requires UMRS_STEP_IMAGE to be set"
            )
        extra_raw = os.environ.get("UMRS_STEP_PODMAN_EXTRA_ARGS", "").strip()
        extra_args = shlex.split(extra_raw) if extra_raw else []
        return PodmanExecutor(
            image=image,
            timeout_seconds=_executor_timeout(),
            podman_bin=os.environ.get("UMRS_STEP_PODMAN_BIN", "podman"),
            network=os.environ.get("UMRS_STEP_PODMAN_NETWORK", "host"),
            gpu=os.environ.get("UMRS_STEP_GPU", "0").strip() == "1",
            extra_args=extra_args,
            keep_workdir=os.environ.get("UMRS_STEP_KEEP_WORKDIR", "0").strip() == "1",
        )
    if chosen == "pool":
        image = os.environ.get("UMRS_STEP_IMAGE", "").strip()
        if not image:
            raise RuntimeError(
                "UMRS_STEP_EXECUTOR=pool requires UMRS_STEP_IMAGE to be set"
            )
        extra_raw = os.environ.get("UMRS_STEP_PODMAN_EXTRA_ARGS", "").strip()
        extra_args = shlex.split(extra_raw) if extra_raw else []
        return WorkerPoolExecutor(
            image=image,
            timeout_seconds=_executor_timeout(),
            podman_bin=os.environ.get("UMRS_STEP_PODMAN_BIN", "podman"),
            network=os.environ.get("UMRS_STEP_PODMAN_NETWORK", "host"),
            gpu=os.environ.get("UMRS_STEP_GPU", "0").strip() == "1",
            extra_args=extra_args,
            max_workers_per_key=_parse_int_env("UMRS_WORKER_MAX_PER_KEY", 4),
            acquire_timeout=_parse_float_env("UMRS_WORKER_ACQUIRE_TIMEOUT", 30.0),
            idle_timeout=_parse_float_env("UMRS_WORKER_IDLE_TIMEOUT", 300.0),
            prewarm=os.environ.get("UMRS_WORKER_PREWARM", "0").strip() == "1",
        )
    if chosen == "remote":
        raw_workers = os.environ.get("UMRS_REMOTE_WORKERS", "").strip()
        if not raw_workers:
            raise RuntimeError(
                "UMRS_STEP_EXECUTOR=remote requires UMRS_REMOTE_WORKERS "
                "to be set (comma-separated ZMQ addresses)"
            )
        workers = [w.strip() for w in raw_workers.split(",") if w.strip()]
        return RemotePoolExecutor(
            workers=workers,
            timeout_seconds=_executor_timeout(),
            acquire_timeout=_parse_float_env("UMRS_WORKER_ACQUIRE_TIMEOUT", 30.0),
        )
    if chosen == "router":
        bind = os.environ.get("UMRS_ROUTER_BIND", "tcp://0.0.0.0:5560").strip()
        return RouterPoolExecutor(
            bind_address=bind,
            timeout_seconds=_executor_timeout(),
        )
    logger.warning("unknown UMRS_STEP_EXECUTOR=%r; falling back to inproc", chosen)
    return InProcessExecutor()


def _executor_timeout() -> float | None:
    """Parse ``UMRS_STEP_EXECUTOR_TIMEOUT`` shared by subprocess + podman."""
    raw = os.environ.get("UMRS_STEP_EXECUTOR_TIMEOUT")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid %s=%r, using default %d", name, raw, default)
        return default


def _parse_float_env(name: str, default: float | None) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("invalid %s=%r, using default %s", name, raw, default)
        return default
