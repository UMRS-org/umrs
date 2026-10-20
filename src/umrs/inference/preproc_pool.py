"""Shared process pool that fetches S3 objects and runs model CPU preprocessing (UMRS_PREPROC_WORKERS > 0)."""

import contextlib
import io
import logging
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

_pool = None
_pool_lock = threading.Lock()


def configured_workers() -> int:
    return int(os.getenv("UMRS_PREPROC_WORKERS", "0"))


def s3_connect_args(s3):
    """Arguments connect_to_s3 used to build *s3*, or None for clients built elsewhere."""
    return getattr(s3, "_umrs_connect_args", None)


def get_pool() -> ProcessPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            workers = configured_workers()
            method = os.getenv("UMRS_PREPROC_START_METHOD", "forkserver")
            ctx = mp.get_context(method)
            if method == "forkserver":
                # import server modules and torch once in the forkserver, not in every worker
                ctx.set_forkserver_preload([
                    "__main__", "torch", "torchvision.transforms", "umrs.shared.storage.s3",
                    "umrs.inference.resnet_50", __name__,
                ])
            _pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker)
            logger.info("[preproc-pool] created: workers=%d start_method=%s", workers, method)
        return _pool


def warm_up():
    """Start every worker now, so that the first request does not pay for it."""
    t0 = time.perf_counter()
    list(get_pool().map(_warm_worker, range(2 * configured_workers())))
    logger.info("[preproc-pool] %d workers ready in %.1fs", configured_workers(), time.perf_counter() - t0)


def reset_pool():
    """Drop the pool (e.g. after a worker died) so the next call creates a new one."""
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------

_clients = {}


def _init_worker():
    import signal

    import torch

    # Ctrl+C reaches the whole process group; the server handles it and shuts the pool down
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)


def _warm_worker(_):
    time.sleep(0.1)  # keep this worker busy so the other tasks reach the other workers


def _client(connect_args):
    s3 = _clients.get(connect_args)
    if s3 is None:
        from umrs.shared.storage.s3 import connect_to_s3

        config_file, endpoint_url_override = connect_args
        with contextlib.redirect_stdout(io.StringIO()):
            s3, _ = connect_to_s3(config_file, endpoint_url_override=endpoint_url_override)
        _clients[connect_args] = s3
    return s3


def load_and_preprocess(task):
    """Fetch one object and preprocess it. Returns (data, etag, metadata, size)."""
    connect_args, bucket, key, preprocess = task
    obj = _client(connect_args).get_object(Bucket=bucket, Key=key)
    data = preprocess(obj["Body"].read())
    return data, obj["ETag"].strip('"'), obj.get("Metadata", {}), obj["ContentLength"]
