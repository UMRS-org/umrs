#!/usr/bin/env python3
"""End-to-end latency benchmark for UMRS KNN search.

Each virtual user gets its own ZMQ socket. The query image is uploaded to S3
once (setup phase) under the reserved ``.umrs-knn/<uuid>-<name>`` key, never
over a bucket object, and removed at the end (``umrs.client.knn``, shared with
``umrs knn``). The embedding cache is warmed before timing starts, so
the benchmark measures the retrieval path only (DB kNN query + network round-
trip), not model inference time. Every request is sent as ``--username``.
Only ``{"status": "ok"}`` replies are timed; error, overloaded, unauthorized
and timed-out requests are counted in ``errors``.

Usage
-----
# 1 user, 50 requests, k=5
python bench_knn.py --image ../data/img_test/cat.jpeg \\
    --bucket user1-1.images-and-texts.bucket-3 \\
    --username user1_1 --s3cfg ~/.s3cfg_user1_1

# Sweep N = 1, 5, 10, 25 with 50 requests each
python bench_knn.py --image ../data/img_test/cat.jpeg \\
    --bucket user1-1.images-and-texts.bucket-3 \\
    --username user1_1 --s3cfg ~/.s3cfg_user1_1 \\
    --sweep 1,5,10,25

# Sweep k values (number of neighbours; the server caps k at UMRS_KNN_MAX_K)
python bench_knn.py --image ../data/img_test/cat.jpeg \\
    --bucket user1-1.images-and-texts.bucket-3 \\
    --username user1_1 --s3cfg ~/.s3cfg_user1_1 \\
    --sweep-k 1,5,10,50,100

# Save results to CSV
python bench_knn.py ... --csv log/bench_knn.csv --label "N=10,k=5"
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_SRC = _BENCH.parents[1]
for _path in (_SRC, _BENCH):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _bench_common import append_csv, fmt_ms, stats

from umrs.client.control_api import ControlReplyError, UMRSControlAPI
from umrs.client.knn import uploaded_query
from umrs.shared.connection_profiles import resolve_connection


def _connect_s3(s3cfg_path: str, endpoint_override: str | None = None):
    from umrs.shared.storage.s3 import connect_to_s3

    s3, _ = connect_to_s3(os.path.expanduser(s3cfg_path), endpoint_url_override=endpoint_override)
    return s3


def _make_mq(server_ip: str, control_port: int, tunnel_mode: bool):
    from umrs.core.MessageQueue import MessageQueue
    return MessageQueue(
        role="client",
        queue_type="control",
        server_ip=server_ip,
        port=control_port,
        tunnel_mode=tunnel_mode,
        prefer_env_server_host=False,
    )


def _knn(mq, username: str, k: int, s3_key: str, bucket: str) -> dict:
    """One KNN request through the shared control-reply decoder
    (``UMRSControlAPI.knn``); a non-JSON reply is an error object."""
    try:
        return UMRSControlAPI(mq, username).knn(k, s3_key, bucket)
    except ControlReplyError as exc:
        return {"status": "error", "message": str(exc)}


class _LostReply(Exception):
    """The request got no reply in time; its reply may still arrive later."""


def _timed_request(mq, username: str, k: int, s3_key: str, bucket: str) -> float | None:
    """Seconds one KNN request took, or None for an error reply.

    Raises :class:`_LostReply` on a timeout or send failure.
    """
    t0 = time.perf_counter()
    try:
        reply = _knn(mq, username, k, s3_key, bucket)
    except (TimeoutError, RuntimeError) as exc:
        raise _LostReply(str(exc)) from exc
    elapsed = time.perf_counter() - t0
    return elapsed if reply.get("status") == "ok" else None


def _request_or_reconnect(mq, connect, username, k, s3_key, bucket):
    """``(seconds or None, mq to use next)``.

    A DEALER socket does not pair replies with requests: after a timeout the
    late reply would answer the next request. So the timed-out socket is
    closed and the next request goes out on a fresh one (review U1).
    """
    try:
        return _timed_request(mq, username, k, s3_key, bucket), mq
    except _LostReply:
        with contextlib.suppress(Exception):
            mq.close()
        return None, connect()


def _worker(
    server_ip: str,
    control_port: int,
    tunnel_mode: bool,
    n_requests: int,
    warmup: int,
    username: str,
    k: int,
    s3_key: str,
    bucket: str,
) -> tuple[list[float], int]:
    """``(latencies of successful timed requests, failed timed requests)``."""

    def connect():
        return _make_mq(server_ip, control_port, tunnel_mode)

    mq = connect()
    latencies: list[float] = []
    errors = 0
    try:
        for i in range(warmup + n_requests):
            elapsed, mq = _request_or_reconnect(mq, connect, username, k, s3_key, bucket)
            if i < warmup:
                continue
            if elapsed is None:
                errors += 1
            else:
                latencies.append(elapsed)
    finally:
        with contextlib.suppress(Exception):
            mq.close()
    return latencies, errors


def run_bench(
    N: int,
    K: int,
    server_ip: str,
    control_port: int,
    tunnel_mode: bool,
    warmup: int,
    username: str,
    k: int,
    s3_key: str,
    bucket: str,
) -> dict:
    wall_start = time.perf_counter()
    all_latencies: list[float] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=N) as pool:
        futures = [
            pool.submit(_worker, server_ip, control_port, tunnel_mode,
                        K, warmup, username, k, s3_key, bucket)
            for _ in range(N)
        ]
        for f in as_completed(futures):
            try:
                latencies, failed = f.result()
            except Exception as e:  # noqa: BLE001 — a failed worker is counted, not fatal
                errors += K
                print(f"  [worker error] {e}", file=sys.stderr)
                continue
            all_latencies.extend(latencies)
            errors += failed

    wall_elapsed = time.perf_counter() - wall_start
    total_req = len(all_latencies)
    throughput = total_req / wall_elapsed if wall_elapsed > 0 else 0.0
    s = stats([x * 1000 for x in all_latencies])
    s.update({"N": N, "K": K, "k": k, "errors": errors,
               "throughput_rps": throughput, "wall_s": wall_elapsed})
    return s


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UMRS KNN latency benchmark")
    parser.add_argument("--image", "-i", required=True, help="Local query image path")
    parser.add_argument("--bucket", required=True, help="S3 path_to_bucket (full name)")
    parser.add_argument("--username", required=True,
                        help="UMRS username every request is sent as")
    parser.add_argument("--s3cfg", required=True, help="Path to .s3cfg file")
    parser.add_argument("-N", "--concurrency", type=int, default=1,
                        help="Number of concurrent virtual users")
    parser.add_argument("-K", "--requests", type=int, default=50,
                        help="Requests per user (excluding warmup)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup requests per user (not counted)")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbours")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Sweep concurrency N values, e.g. 1,5,10,25")
    parser.add_argument("--sweep-k", type=str, default=None,
                        help="Sweep k (neighbours) values, e.g. 1,5,10,50,100")
    parser.add_argument("--server-ip", type=str, default=None)
    parser.add_argument("--control-port", type=int, default=None)
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Override S3 endpoint URL")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    return parser


def _sweep_values(args) -> tuple[list[int], list[int]]:
    """Return (N_values, k_values): sweep either N or k, not both."""
    if args.sweep_k:
        return [args.concurrency], [int(x) for x in args.sweep_k.split(",")]
    n_values = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.concurrency]
    return n_values, [args.k]


def _warm_cache(connection, args, s3_key: str) -> None:
    """One KNN call; exits when it does not answer ``ok`` (nothing to time)."""
    print("Warming up embedding cache (1 KNN call)...")
    mq = _make_mq(connection.server_host, connection.control_port, connection.tunnel_mode)
    try:
        reply = _knn(mq, args.username, args.k, s3_key, args.bucket)
    finally:
        mq.close()
    if reply.get("status") != "ok":
        raise SystemExit(
            f"KNN warm-up failed, nothing to benchmark: {reply.get('message', reply)}")
    print("  Cache warmed.")


def _print_row(s: dict) -> None:
    def ms(key):
        return fmt_ms(s, key)

    print(
        f"{s['N']:>4}  {s['k']:>4}  {ms('mean_ms'):>8}  {ms('p50_ms'):>8}  "
        f"{ms('p95_ms'):>8}  {ms('p99_ms'):>8}  "
        f"{ms('min_ms'):>8}  {ms('max_ms'):>8}  "
        f"{s['throughput_rps']:>8.1f}  {s['errors']:>4}"
    )


def _run_sweep(args, connection, s3_key: str, label: str) -> list[dict]:
    """One row per (N, k); the CSV names the query by its file name."""
    n_values, k_values = _sweep_values(args)
    rows = []
    for N in n_values:
        for k in k_values:
            s = run_bench(
                N=N, K=args.requests,
                server_ip=connection.server_host,
                control_port=connection.control_port,
                tunnel_mode=connection.tunnel_mode,
                warmup=args.warmup,
                username=args.username, k=k, s3_key=s3_key, bucket=args.bucket,
            )
            _print_row(s)
            if label:
                s["label"] = label
            s["query"] = Path(args.image).name
            rows.append(s)
    return rows


def main():
    args = _build_parser().parse_args()
    connection = resolve_connection(
        explicit_server_ip=args.server_ip,
        explicit_control_port=args.control_port,
    )

    print(f"Uploading query image '{args.image}' to S3 bucket '{args.bucket}'...")
    s3 = _connect_s3(args.s3cfg, args.endpoint)
    with uploaded_query(s3, args.bucket, str(args.image), None) as s3_key:
        print(f"  s3_key = {s3_key}")
        rows = _benchmark(args, connection, s3_key)
    if args.csv:
        append_csv(args.csv, rows)


def _benchmark(args, connection, s3_key: str) -> list[dict]:
    """Warm the cache, print the header, run the sweep."""
    _warm_cache(connection, args, s3_key)
    label = args.label or ""
    print(
        f"\nUMRS KNN latency benchmark"
        f"  server={connection.server_host}:{connection.control_port}"
        f"  query={s3_key}  bucket={args.bucket}"
        f"  K={args.requests}  warmup={args.warmup}"
        + (f"  label={label}" if label else "")
    )
    header = f"{'N':>4}  {'k':>4}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'min':>8}  {'max':>8}  {'rps':>8}  {'err':>4}"
    print(header)
    print("-" * len(header))
    return _run_sweep(args, connection, s3_key, label)


if __name__ == "__main__":
    main()
