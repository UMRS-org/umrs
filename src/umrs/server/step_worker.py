"""Subprocess / pool worker entry point for step executors.

**One-shot mode** (default — ``SubprocessExecutor`` / ``PodmanExecutor``):
Reads a pickled payload from stdin (or file / S3), runs
``setup → run → teardown``, and writes a pickled reply.

**Pool mode** (``--pool`` flag — ``WorkerPoolExecutor``):
Loops over a length-prefixed pickle protocol on stdin/stdout:
``[4-byte big-endian length][pickle bytes]``.  Each task is processed
independently; the process stays alive between tasks so that container
startup and Python import overhead are amortised across many steps.

**JSON mode** (``UMRS_STEP_INPUT_JSON`` set — the gated HPC lane, ADR-112
wave B2, plan §5 B2.0 item 9): reads the gate-written ``input.json``
(data only, never unpickled), requires its ``entrypoint`` to equal
``UMRS_STEP_ENTRYPOINT`` before any import, rebuilds the parent results
with :func:`umrs.server.step_reply.checked_result` and writes only the JSON
reply (``UMRS_STEP_OUTPUT_JSON``).

Stdout is binary (pickle / framed protocol); all logging goes to
stderr.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import pickle
import shutil
import signal
import struct
import sys
import tempfile
import traceback
from typing import Any

from umrs.core.zmq_security import ZMQSecurityConfig
from umrs.pipeline_sdk.staged_s3 import StagedS3Client, StagedS3Error, is_staged
from umrs.pipeline_sdk.step import StepContext

# Logging to stderr keeps stdout clean for the pickle reply.
logging.basicConfig(level=os.environ.get("UMRS_STEP_WORKER_LOG", "WARNING"))
logger = logging.getLogger("umrs.step_worker")


def _build_s3_client(s3_config: dict[str, Any]) -> Any:
    """Reconstruct a boto3-compatible S3 client inside the worker.

    Honours ``s3_config['endpoint_url']`` / ``region_name`` if present
    so the child connects to the same MinIO endpoint as the parent.
    Returns ``None`` when boto3 is unavailable; the step itself is
    responsible for handling a missing client (most steps do not need
    one if they only operate on ``parent_results``).

    On the staged HPC lane (``UMRS_STEP_STAGED=1``) the client is the
    file-backed :class:`StagedS3Client` over ``/work/in`` and ``/work/out``:
    no credential, no endpoint, no boto3 (ADR-074 D-5).
    """
    if is_staged():
        return StagedS3Client()
    try:
        import boto3  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.warning("boto3 unavailable in worker: %s", exc)
        return None
    kwargs: dict[str, Any] = {}
    endpoint = (
        s3_config.get("endpoint_url")
        or os.environ.get("S3_ENDPOINT_URL")
        or os.environ.get("MINIO_ENDPOINT")  # pre-ADR-055 name
    )
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    region = s3_config.get("region_name") or os.environ.get("AWS_REGION")
    if region:
        kwargs["region_name"] = region
    try:
        return boto3.client("s3", **kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not build worker S3 client: %s", exc)
        return None


def _emit(reply: dict[str, Any]) -> None:
    """Write the pickle reply.

    Priority order (Slice 11):
    1. ``UMRS_STEP_OUTPUT_S3`` — upload to S3 (format: ``bucket/key``).
    2. ``UMRS_STEP_OUTPUT`` — write to a local file (PodmanExecutor
       bind-mount).
    3. stdout — binary pipe for :class:`SubprocessExecutor`.

    ``UMRS_STEP_OUTPUT_JSON`` additionally names a data-only JSON reply
    (ADR-074 D-6): servers that set it read that file and never unpickle
    what the step wrote; ``output.pkl`` stays for older servers. The staged
    HPC lane reads only the JSON reply, so a staged worker writes no pickle.
    """
    json_path = os.environ.get("UMRS_STEP_OUTPUT_JSON")
    if json_path:
        _write_json_reply(json_path, reply)
    if _json_only():
        return
    blob = pickle.dumps(reply)

    s3_ref = os.environ.get("UMRS_STEP_OUTPUT_S3")
    if s3_ref:
        bucket, _, key = s3_ref.partition("/")
        if bucket and key:
            s3 = _build_s3_client({})
            if s3 is not None:
                try:
                    s3.put_object(Bucket=bucket, Key=key, Body=blob)
                    logger.info(
                        "wrote %d-byte reply to s3://%s/%s", len(blob), bucket, key
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("S3 reply upload failed, falling back: %s", exc)
            else:
                logger.warning("S3 client unavailable for reply upload, falling back")

    out_path = os.environ.get("UMRS_STEP_OUTPUT")
    if out_path:
        with open(out_path, "wb") as fh:
            fh.write(blob)
        return
    sys.stdout.buffer.write(blob)
    sys.stdout.buffer.flush()


def _json_only() -> bool:
    """The staged lane and JSON mode read only the JSON reply: no pickle."""
    return is_staged() or INPUT_JSON_ENV in os.environ


def _write_json_reply(path: str, reply: dict[str, Any]) -> None:
    """Write the JSON reply by atomic rename: the server never sees half."""
    from umrs.server.step_reply import reply_to_json

    blob = reply_to_json(reply)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".output-")
    # mkstemp makes 0600; keep the mode open() gave (a server outside the
    # container's uid mapping may need to read it).
    mask = os.umask(0)
    os.umask(mask)
    try:
        os.fchmod(fd, 0o666 & ~mask)
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _read_payload() -> bytes:
    """Read the pickle payload.

    Priority order (Slice 11):
    1. ``UMRS_STEP_INPUT_S3`` — download from S3 (format: ``bucket/key``).
    2. ``UMRS_STEP_INPUT`` — read from a local file.
    3. stdin — binary pipe for :class:`SubprocessExecutor`.
    """
    s3_ref = os.environ.get("UMRS_STEP_INPUT_S3")
    if s3_ref:
        bucket, _, key = s3_ref.partition("/")
        if bucket and key:
            s3 = _build_s3_client({})
            if s3 is not None:
                try:
                    obj = s3.get_object(Bucket=bucket, Key=key)
                    data = obj["Body"].read()
                    logger.info(
                        "read %d-byte payload from s3://%s/%s", len(data), bucket, key
                    )
                    return data
                except Exception as exc:  # noqa: BLE001
                    logger.warning("S3 payload download failed: %s", exc)
                    return b""
            else:
                logger.warning("S3 client unavailable for payload download")
                return b""

    in_path = os.environ.get("UMRS_STEP_INPUT")
    if in_path:
        with open(in_path, "rb") as fh:
            return fh.read()
    return sys.stdin.buffer.read()


_WORKER_SOURCE_DIRS: list[str] = []


def _drop_modules_under(stale_root: str) -> None:
    """Forget modules imported from the previous task's source dir."""
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and os.path.realpath(str(path)).startswith(stale_root):
            del sys.modules[name]


def _prepare_worker_source_dir() -> None:
    """Put the uploaded workflow source dir on ``sys.path``.

    Container/remote executors (e.g. :class:`SlurmStepExecutor`) export
    ``UMRS_STEP_SOURCE_DIR`` and bind-mount the tree read-only at the
    same absolute path; without it the user step modules are not
    importable here. One-shot workers get a fresh process per step, so
    this is a plain insert. Pool workers reuse the process across
    tasks: when the dir changes between tasks, modules imported from
    the previous dir are dropped so same-named packages (two workflows
    both shipping ``steps``) don't go stale.
    """
    from umrs.server.step_executor import EXECUTOR_SOURCE_DIR_ENV_VAR

    source_dir = os.environ.get(EXECUTOR_SOURCE_DIR_ENV_VAR, "").strip()
    if not source_dir or not os.path.isdir(source_dir):
        return
    if _WORKER_SOURCE_DIRS and _WORKER_SOURCE_DIRS[-1] == source_dir:
        return  # already prepared for this dir
    if _WORKER_SOURCE_DIRS:
        _drop_modules_under(_WORKER_SOURCE_DIRS[-1])
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    _WORKER_SOURCE_DIRS.append(source_dir)


def _process_task(payload: dict[str, Any]) -> dict[str, Any]:
    """Run a single task and return the reply dict.

    Shared by both one-shot ``main()`` and ``pool_main()``.
    """
    _prepare_worker_source_dir()

    try:
        entrypoint = payload["entrypoint"]
        config = payload["config"]
        ctx_data = payload["ctx"]
        s3_config = payload.get("s3_config") or {}
        s3 = _build_s3_client(s3_config)
        ctx = StepContext(
            s3=s3,
            input_bucket=ctx_data["input_bucket"],
            input_keys=ctx_data["input_keys"],
            output_bucket=ctx_data["output_bucket"],
            parent_output_bucket=ctx_data.get("parent_output_bucket") or "",
            config=ctx_data["config"],
            metadata=ctx_data.get("metadata") or {},
            run_id=ctx_data.get("run_id") or "",
            parent_results=ctx_data.get("parent_results") or {},
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "phase": "worker",
            "message": f"malformed payload: {exc}",
        }

    return _run_step(entrypoint, config, ctx)


def _run_step(
    entrypoint: str, config: dict[str, Any], ctx: StepContext
) -> dict[str, Any]:
    """Load the step class, drive ``setup → run → teardown``, return the reply."""
    from umrs.server.step_executor import (
        StepExecutionError,
        _drive_lifecycle,
        _load_step_class,
    )

    try:
        cls = _load_step_class(entrypoint)
        result = _drive_lifecycle(cls, config, ctx)
    except StepExecutionError as exc:
        return {"status": "error", "phase": exc.phase, "message": exc.message}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "phase": "run",
            "message": f"{exc}\n{traceback.format_exc()}",
        }

    return {"status": "ok", "result": result}


# -- Length-prefixed framing helpers (pool mode) ---------------------------


def _read_framed(stream) -> bytes:
    """Read a ``[4-byte length][payload]`` message from *stream*.

    Returns empty bytes when the stream is closed or truncated.
    """
    header = stream.read(4)
    if not header or len(header) < 4:
        return b""
    length = struct.unpack(">I", header)[0]
    data = stream.read(length)
    if len(data) < length:
        return b""
    return data


def _write_framed(stream, data: bytes) -> None:
    """Write a ``[4-byte length][payload]`` message to *stream*."""
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


def pool_main() -> int:
    """Persistent worker loop for :class:`WorkerPoolExecutor`.

    Protocol:
    * stdin/stdout carry length-prefixed pickle messages.
    * Each request is an independent task (same payload shape as
      one-shot mode).
    * The worker replies with the result dict and loops.
    * Closing stdin exits the loop cleanly.
    """
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        raw = _read_framed(stdin)
        if not raw:
            break  # stdin closed — clean exit
        try:
            payload = pickle.loads(raw)
        except Exception as exc:  # noqa: BLE001
            reply: dict[str, Any] = {
                "status": "error",
                "phase": "worker",
                "message": f"could not decode payload: {exc}",
            }
            _write_framed(stdout, pickle.dumps(reply))
            continue

        reply = _process_task(payload)
        _write_framed(stdout, pickle.dumps(reply))

    return 0


# ------------------------------------------------------------------
# Capability introspection (Slice 17 / 19)
# ------------------------------------------------------------------


def _gpu_mem_info() -> dict[str, int]:
    """Return GPU memory info via pynvml, or empty dict on failure."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {"total": int(info.total), "free": int(info.free)}
    except Exception:  # noqa: BLE001
        return {}


def _compute_slots(
    free_mem_bytes: int,
    mem_per_task_mb: int,
    safety: float,
) -> int:
    """Return the number of concurrent task slots based on free GPU memory.

    Always returns at least 1 to keep liveness. Capped at
    ``UMRS_MAX_SLOTS_CAP`` (default 16) to prevent runaway parallelism.
    """
    if free_mem_bytes <= 0 or mem_per_task_mb <= 0:
        return 1
    effective_bytes = int(free_mem_bytes * safety)
    mem_per_task_bytes = mem_per_task_mb * 1024 * 1024
    slots = effective_bytes // mem_per_task_bytes
    max_cap = int(os.environ.get("UMRS_MAX_SLOTS_CAP", "16"))
    return max(1, min(slots, max_cap))


def _get_capabilities() -> dict[str, Any]:
    """Collect worker capabilities for Slice 17 ping responses.

    Returns a dict of capability tags. Currently:
    - ``gpu``: True if ``nvidia-smi`` is on PATH
    - GPU memory fields + ``slots`` when pynvml is available (Slice 19)
    """
    caps: dict[str, Any] = {}
    caps["gpu"] = shutil.which("nvidia-smi") is not None

    gpu_info = _gpu_mem_info()
    if gpu_info:
        mem_per_task_mb = int(os.environ.get("UMRS_GPU_MEM_PER_TASK", "4096"))
        safety = float(os.environ.get("UMRS_GPU_MEM_SAFETY_FACTOR", "0.9"))
        caps["gpu_mem_total"] = gpu_info["total"]
        caps["gpu_mem_free"] = gpu_info["free"]
        caps["mem_per_task"] = mem_per_task_mb
        caps["slots"] = _compute_slots(gpu_info["free"], mem_per_task_mb, safety)

    return caps


def zmq_pool_main(bind_address: str) -> int:
    """Persistent worker loop over ZMQ REP (Slice 16).

    Binds a ZMQ REP socket on *bind_address* and loops:
    ``recv → pickle.loads → _process_task → pickle.dumps → send``.

    Capability pings (Slice 17): payloads with ``{"type": "ping"}``
    are answered with ``{"type": "pong", "capabilities": {...}}``
    without dispatching to ``_process_task``.

    When ``UMRS_TUNNEL_MODE`` is set, the bind address is forced to
    ``127.0.0.1`` regardless of what was passed.
    """
    import zmq  # type: ignore

    # Enforce tunnel-mode bind restriction.
    if os.environ.get("UMRS_TUNNEL_MODE"):
        from urllib.parse import urlparse

        parsed = urlparse(bind_address)
        if parsed.hostname and parsed.hostname != "127.0.0.1":
            bind_address = f"{parsed.scheme}://127.0.0.1:{parsed.port}"
            logger.info("UMRS_TUNNEL_MODE: forced bind to %s", bind_address)

    zmq_ctx = zmq.Context()
    sock = zmq_ctx.socket(zmq.REP)
    authenticator = ZMQSecurityConfig.from_env().apply_to_socket(
        sock,
        server=True,
        context=zmq_ctx,
    )
    sock.bind(bind_address)
    logger.info("zmq_pool_main listening on %s", bind_address)

    max_payload_bytes = int(os.environ.get("UMRS_ZMQ_MAX_PAYLOAD_BYTES", "52428800"))

    _shutdown = False

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal _shutdown
        _shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not _shutdown:
            # Use a poller so we can check _shutdown periodically.
            if sock.poll(timeout=1000, flags=zmq.POLLIN) == 0:
                continue
            raw = sock.recv()
            if len(raw) > max_payload_bytes:
                reply: dict[str, Any] = {
                    "status": "error",
                    "phase": "worker",
                    "message": (
                        f"payload too large: {len(raw)} bytes "
                        f"exceeds limit {max_payload_bytes}"
                    ),
                }
                sock.send(pickle.dumps(reply))
                continue
            try:
                payload = pickle.loads(raw)
            except Exception as exc:  # noqa: BLE001
                reply: dict[str, Any] = {
                    "status": "error",
                    "phase": "worker",
                    "message": f"could not decode payload: {exc}",
                }
                sock.send(pickle.dumps(reply))
                continue

            # Slice 17: capability ping — respond without running a task.
            if isinstance(payload, dict) and payload.get("type") == "ping":
                pong: dict[str, Any] = {
                    "type": "pong",
                    "capabilities": _get_capabilities(),
                }
                sock.send(pickle.dumps(pong))
                continue

            reply = _process_task(payload)
            sock.send(pickle.dumps(reply))
    finally:
        sock.close(linger=0)
        if authenticator is not None:
            authenticator.stop()
        zmq_ctx.term()

    return 0


def zmq_dealer_main(connect_address: str) -> int:
    """Persistent worker loop over ZMQ DEALER (Slice 18b).

    Connects a ZMQ DEALER socket to a ROUTER on the orchestrator side.
    Unlike REP mode the DEALER can receive pipelined requests:
    each request carries a ``correlation_id`` that the worker echoes
    back so the orchestrator can match replies to requests.

    Capability pings (Slice 17): payloads with ``{"type": "ping"}``
    are answered with ``{"type": "pong", ...}`` just like REP mode.
    """
    import zmq  # type: ignore

    worker_id = os.environ.get("UMRS_WORKER_ID", "")

    zmq_ctx = zmq.Context()
    sock = zmq_ctx.socket(zmq.DEALER)
    if worker_id:
        sock.setsockopt_string(zmq.IDENTITY, worker_id)
    ZMQSecurityConfig.from_env().apply_to_socket(
        sock,
        server=False,
        context=zmq_ctx,
    )
    sock.connect(connect_address)
    logger.info(
        "zmq_dealer_main connected to %s (identity=%s)",
        connect_address,
        worker_id or "<auto>",
    )

    max_payload_bytes = int(os.environ.get("UMRS_ZMQ_MAX_PAYLOAD_BYTES", "52428800"))

    _shutdown = False

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal _shutdown
        _shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        while not _shutdown:
            if sock.poll(timeout=1000, flags=zmq.POLLIN) == 0:
                continue
            # DEALER receives [empty_frame, payload] or just [payload]
            # depending on peer socket type.  We accept both forms.
            frames = sock.recv_multipart()
            if not frames:
                continue
            raw = frames[-1]
            if len(raw) > max_payload_bytes:
                reply: dict[str, Any] = {
                    "status": "error",
                    "phase": "worker",
                    "message": (
                        f"payload too large: {len(raw)} bytes "
                        f"exceeds limit {max_payload_bytes}"
                    ),
                }
                sock.send_multipart(frames[:-1] + [pickle.dumps(reply)])
                continue
            try:
                payload = pickle.loads(raw)
            except Exception as exc:  # noqa: BLE001
                reply = {
                    "status": "error",
                    "phase": "worker",
                    "message": f"could not decode payload: {exc}",
                }
                sock.send_multipart(frames[:-1] + [pickle.dumps(reply)])
                continue

            # Echo correlation_id for request-reply matching.
            correlation_id = None
            if isinstance(payload, dict):
                correlation_id = payload.get("correlation_id")

            # Capability ping.
            if isinstance(payload, dict) and payload.get("type") == "ping":
                pong: dict[str, Any] = {
                    "type": "pong",
                    "capabilities": _get_capabilities(),
                }
                if correlation_id is not None:
                    pong["correlation_id"] = correlation_id
                sock.send_multipart(frames[:-1] + [pickle.dumps(pong)])
                continue

            result = _process_task(payload)
            if correlation_id is not None:
                result["correlation_id"] = correlation_id
            sock.send_multipart(frames[:-1] + [pickle.dumps(result)])
    finally:
        sock.close(linger=0)
        zmq_ctx.term()

    return 0


# ------------------------------------------------------------------
# JSON mode (ADR-112 wave B2; plan §5 B2.0 items 8 and 9)
# ------------------------------------------------------------------

#: The gate-written ``input.json`` (read only, data only).
INPUT_JSON_ENV = "UMRS_STEP_INPUT_JSON"
#: The signed entrypoint the gate's job script exports.
ENTRYPOINT_ENV = "UMRS_STEP_ENTRYPOINT"
#: The read cap; the gate writes at most 48 MiB (``INPUT_JSON_MAX_BYTES``).
INPUT_JSON_READ_MAX_BYTES = 64 * 1024 * 1024

_PICKLE_IO_ENVS = (
    "UMRS_STEP_INPUT",
    "UMRS_STEP_INPUT_S3",
    "UMRS_STEP_OUTPUT",
    "UMRS_STEP_OUTPUT_S3",
)
_POOL_FLAGS = ("--pool", "--zmq-pool", "--zmq-dealer")
_INPUT_JSON_KEYS = frozenset({"v", "entrypoint", "config", "parent_results", "ctx"})
_INPUT_JSON_CTX_TYPES: dict[str, type] = {
    "run_tag": str,
    "step": str,
    "attempt": int,
    "input_bucket": str,
    "output_bucket": str,
    "parent_output_bucket": str,
    "input_keys": list,
}


class InputRefused(Exception):
    """A JSON-mode input check failed: ``<name>: <detail>`` in the reply."""

    def __init__(self, name: str, detail: str) -> None:
        super().__init__(f"{name}: {detail}")
        self.name = name


def _check_json_mode_env(argv: list[str]) -> None:
    """Step 1: JSON mode admits no pickle channel and no pool loop."""
    mixed = [name for name in _PICKLE_IO_ENVS if name in os.environ]
    flags = [flag for flag in _POOL_FLAGS if flag in argv]
    if mixed or flags:
        raise InputRefused(
            "mixed_input_modes", f"{INPUT_JSON_ENV} with {', '.join(mixed + flags)}"
        )
    if not os.environ.get("UMRS_STEP_OUTPUT_JSON"):
        raise InputRefused("mixed_input_modes", "UMRS_STEP_OUTPUT_JSON is not set")


def _read_input_json(path: str) -> bytes:
    """Step 2: the file's bytes, at most ``INPUT_JSON_READ_MAX_BYTES``."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(INPUT_JSON_READ_MAX_BYTES + 1)
    except (OSError, ValueError) as exc:
        raise InputRefused("input_unreadable", str(exc)) from exc
    if len(data) > INPUT_JSON_READ_MAX_BYTES:
        raise InputRefused(
            "input_unreadable", f"larger than {INPUT_JSON_READ_MAX_BYTES} bytes"
        )
    return data


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite number")
    return value


def _parse_input_json(data: bytes) -> Any:
    """Step 3: strict JSON (no duplicate keys, NaN or Infinity; floats and
    ``null`` allowed). Never ``pickle.loads``."""
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_no_constant,
            parse_float=_finite_float,
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise InputRefused("input_not_json", str(exc)) from exc


def _typed(value: Any, kind: type) -> bool:
    """``type(v) is kind`` (``true`` is not an int); list entries are strings."""
    if type(value) is not kind:
        return False
    return not isinstance(value, list) or all(type(item) is str for item in value)


def _check_input_ctx(ctx: Any) -> None:
    if type(ctx) is not dict or set(ctx) != set(_INPUT_JSON_CTX_TYPES):
        raise InputRefused(
            "input_schema", f"ctx keys must be {sorted(_INPUT_JSON_CTX_TYPES)}"
        )
    for key, kind in _INPUT_JSON_CTX_TYPES.items():
        if not _typed(ctx[key], kind):
            raise InputRefused("input_schema", f"ctx.{key} must be a {kind.__name__}")


def _check_input_schema(body: Any) -> None:
    """Step 4: exactly the v1 keys, each typed."""
    if type(body) is not dict or set(body) != _INPUT_JSON_KEYS:
        raise InputRefused("input_schema", f"keys must be {sorted(_INPUT_JSON_KEYS)}")
    if type(body["v"]) is not int or body["v"] != 1:
        raise InputRefused("input_schema", "v must be 1")
    if type(body["entrypoint"]) is not str or not body["entrypoint"]:
        raise InputRefused("input_schema", "entrypoint must be a non-empty string")
    if type(body["config"]) is not dict:
        raise InputRefused("input_schema", "config must be an object")
    if type(body["parent_results"]) is not dict:
        raise InputRefused("input_schema", "parent_results must be an object")
    _check_input_ctx(body["ctx"])


def _check_entrypoint(entrypoint: str) -> None:
    """Step 5: the signed entrypoint, before anything is imported."""
    expected = os.environ.get(ENTRYPOINT_ENV, "")
    if not expected or entrypoint != expected:
        raise InputRefused(
            "entrypoint_mismatch",
            f"input.json names {entrypoint!r}, {ENTRYPOINT_ENV} is {expected!r}",
        )


def _parent_results(raw: dict[str, Any]) -> dict[str, Any]:
    """Step 6: each parent rebuilt with ``step_reply.checked_result``."""
    from umrs.server.step_executor import StepExecutionError
    from umrs.server.step_reply import checked_result

    parents: dict[str, Any] = {}
    for name, fields in raw.items():
        if type(fields) is not dict:
            raise InputRefused("parent_result_invalid", f"{name}: not an object")
        try:
            parents[name] = checked_result(fields)
        except StepExecutionError as exc:
            raise InputRefused(
                "parent_result_invalid", f"{name}: {exc.message}"
            ) from exc
    return parents


def _staged_client() -> Any:
    """The file-backed ``ctx.s3`` over the gate-written ``in/manifest.json``."""
    try:
        return StagedS3Client()
    except StagedS3Error as exc:
        raise InputRefused("input_unreadable", str(exc)) from exc


def _json_step_input(argv: list[str]) -> tuple[str, dict[str, Any], StepContext]:
    """The entrypoint, config and context from ``input.json``, every check
    in the contract's order."""
    _check_json_mode_env(argv)
    body = _parse_input_json(_read_input_json(os.environ[INPUT_JSON_ENV]))
    _check_input_schema(body)
    _check_entrypoint(body["entrypoint"])
    parents = _parent_results(body["parent_results"])
    ctx_data = body["ctx"]
    ctx = StepContext(
        s3=_staged_client(),
        input_bucket=ctx_data["input_bucket"],
        input_keys=list(ctx_data["input_keys"]),
        output_bucket=ctx_data["output_bucket"],
        parent_output_bucket=ctx_data["parent_output_bucket"],
        config=body["config"],
        metadata={},
        run_id=ctx_data["run_tag"],
        parent_results=parents,
    )
    return body["entrypoint"], body["config"], ctx


def json_main(argv: list[str] | None = None) -> int:
    """Run one step from the gate-written ``input.json``; reply JSON only.

    A refused input is answered with ``{"status": "error", "phase":
    "worker", "message": "<name>: <detail>"}`` and exit 0, so the reply is
    the one record of what happened. Only when there is nowhere to write
    that reply (``UMRS_STEP_OUTPUT_JSON`` unset) does the worker exit 1.
    """
    args = sys.argv[1:] if argv is None else argv
    try:
        entrypoint, config, ctx = _json_step_input(args)
    except InputRefused as exc:
        _emit({"status": "error", "phase": "worker", "message": str(exc)})
        if not os.environ.get("UMRS_STEP_OUTPUT_JSON"):
            logger.error("step worker JSON mode refused: %s", exc)
            return 1
        return 0
    _prepare_worker_source_dir()
    _emit(_run_step(entrypoint, config, ctx))
    return 0


def main() -> int:
    """One-shot entry: JSON mode when ``UMRS_STEP_INPUT_JSON`` is set
    (checked first, never falling back to pickle), else the pickle payload."""
    if INPUT_JSON_ENV in os.environ:
        return json_main()
    return _pickle_main()


def _pickle_main() -> int:
    raw = _read_payload()
    if not raw:
        _emit({"status": "error", "phase": "worker", "message": "empty payload"})
        return 1
    try:
        payload = pickle.loads(raw)
    except Exception as exc:  # noqa: BLE001
        _emit(
            {
                "status": "error",
                "phase": "worker",
                "message": f"could not decode payload: {exc}",
            }
        )
        return 1

    reply = _process_task(payload)
    _emit(reply)
    return 0 if reply["status"] == "ok" else (0 if "phase" in reply else 1)


def _address_arg(argv: list[str], flag: str) -> str | None:
    idx = argv.index(flag)
    return argv[idx + 1] if idx + 1 < len(argv) else None


def entry(argv: list[str]) -> int:
    """``python -m umrs.server.step_worker``: JSON mode first (it refuses
    the pool flags), then the pickle pool loops, then one-shot pickle."""
    if INPUT_JSON_ENV in os.environ:
        return json_main(argv)
    if "--pool" in argv:
        return pool_main()
    for flag, loop in (
        ("--zmq-pool", zmq_pool_main),
        ("--zmq-dealer", zmq_dealer_main),
    ):
        if flag in argv:
            address = _address_arg(argv, flag)
            if address is None:
                logger.error("%s requires an address", flag)
                return 1
            return loop(address)
    return _pickle_main()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(entry(sys.argv[1:]))
