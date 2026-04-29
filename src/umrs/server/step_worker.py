"""Subprocess / pool worker entry point for step executors.

**One-shot mode** (default — ``SubprocessExecutor`` / ``PodmanExecutor``):
Reads a pickled payload from stdin (or file / S3), runs
``setup → run → teardown``, and writes a pickled reply.

**Pool mode** (``--pool`` flag — ``WorkerPoolExecutor``):
Loops over a length-prefixed pickle protocol on stdin/stdout:
``[4-byte big-endian length][pickle bytes]``.  Each task is processed
independently; the process stays alive between tasks so that container
startup and Python import overhead are amortised across many steps.

Stdout is binary (pickle / framed protocol); all logging goes to
stderr.
"""

from __future__ import annotations

import logging
import os
import pickle
import signal
import struct
import sys
import traceback
from typing import Any, Dict

from umrs.pipeline_sdk.step import StepContext

# Logging to stderr keeps stdout clean for the pickle reply.
logging.basicConfig(level=os.environ.get("UMRS_STEP_WORKER_LOG", "WARNING"))
logger = logging.getLogger("umrs.step_worker")


def _build_s3_client(s3_config: Dict[str, Any]) -> Any:
    """Reconstruct a boto3-compatible S3 client inside the worker.

    Honours ``s3_config['endpoint_url']`` / ``region_name`` if present
    so the child connects to the same MinIO endpoint as the parent.
    Returns ``None`` when boto3 is unavailable; the step itself is
    responsible for handling a missing client (most steps do not need
    one if they only operate on ``parent_results``).
    """
    try:
        import boto3  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.warning("boto3 unavailable in worker: %s", exc)
        return None
    kwargs: Dict[str, Any] = {}
    endpoint = s3_config.get("endpoint_url") or os.environ.get("MINIO_ENDPOINT")
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


def _emit(reply: Dict[str, Any]) -> None:
    """Write the pickle reply.

    Priority order (Slice 11):
    1. ``UMRS_STEP_OUTPUT_S3`` — upload to S3 (format: ``bucket/key``).
    2. ``UMRS_STEP_OUTPUT`` — write to a local file (PodmanExecutor
       bind-mount).
    3. stdout — binary pipe for :class:`SubprocessExecutor`.
    """
    blob = pickle.dumps(reply)

    s3_ref = os.environ.get("UMRS_STEP_OUTPUT_S3")
    if s3_ref:
        bucket, _, key = s3_ref.partition("/")
        if bucket and key:
            s3 = _build_s3_client({})
            if s3 is not None:
                try:
                    s3.put_object(Bucket=bucket, Key=key, Body=blob)
                    logger.info("wrote %d-byte reply to s3://%s/%s", len(blob), bucket, key)
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
                    logger.info("read %d-byte payload from s3://%s/%s", len(data), bucket, key)
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


def _process_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single task and return the reply dict.

    Shared by both one-shot ``main()`` and ``pool_main()``.
    """
    from umrs.server.step_executor import (
        StepExecutionError,
        _drive_lifecycle,
        _load_step_class,
    )

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
            reply: Dict[str, Any] = {
                "status": "error",
                "phase": "worker",
                "message": f"could not decode payload: {exc}",
            }
            _write_framed(stdout, pickle.dumps(reply))
            continue

        reply = _process_task(payload)
        _write_framed(stdout, pickle.dumps(reply))

    return 0


def zmq_pool_main(bind_address: str) -> int:
    """Persistent worker loop over ZMQ REP (Slice 16).

    Binds a ZMQ REP socket on *bind_address* and loops:
    ``recv → pickle.loads → _process_task → pickle.dumps → send``.

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
    sock.bind(bind_address)
    logger.info("zmq_pool_main listening on %s", bind_address)

    max_payload_bytes = int(
        os.environ.get("UMRS_ZMQ_MAX_PAYLOAD_BYTES", "52428800")
    )

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
                reply: Dict[str, Any] = {
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
                reply: Dict[str, Any] = {
                    "status": "error",
                    "phase": "worker",
                    "message": f"could not decode payload: {exc}",
                }
                sock.send(pickle.dumps(reply))
                continue

            reply = _process_task(payload)
            sock.send(pickle.dumps(reply))
    finally:
        sock.close(linger=0)
        zmq_ctx.term()

    return 0


def main() -> int:
    raw = _read_payload()
    if not raw:
        _emit({"status": "error", "phase": "worker", "message": "empty payload"})
        return 1
    try:
        payload = pickle.loads(raw)
    except Exception as exc:  # noqa: BLE001
        _emit({
            "status": "error",
            "phase": "worker",
            "message": f"could not decode payload: {exc}",
        })
        return 1

    reply = _process_task(payload)
    _emit(reply)
    return 0 if reply["status"] == "ok" else (0 if "phase" in reply else 1)


if __name__ == "__main__":  # pragma: no cover
    if "--pool" in sys.argv:
        sys.exit(pool_main())
    if "--zmq-pool" in sys.argv:
        idx = sys.argv.index("--zmq-pool")
        if idx + 1 >= len(sys.argv):
            print("--zmq-pool requires a bind address", file=sys.stderr)
            sys.exit(1)
        sys.exit(zmq_pool_main(sys.argv[idx + 1]))
    sys.exit(main())
