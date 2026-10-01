#!/usr/bin/env python3
"""End-to-end latency benchmark for UMRS KNN search.

Each virtual user gets its own ZMQ socket. The query image is uploaded to S3
once (setup phase) and the embedding cache is warmed before timing starts, so
the benchmark measures the retrieval path only (DB kNN query + network round-
trip), not model inference time.

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

# Sweep k values (number of neighbours)
python bench_knn.py --image ../data/img_test/cat.jpeg \\
    --bucket user1-1.images-and-texts.bucket-3 \\
    --username user1_1 --s3cfg ~/.s3cfg_user1_1 \\
    --sweep-k 1,5,10,50,100

# Save results to CSV
python bench_knn.py ... --csv log/bench_knn.csv --label "N=10,k=5"
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from umrs.shared.connection_profiles import resolve_connection


def _upload_query_image(s3cfg_path: str, bucket: str, image_path: str, endpoint_override: str | None = None) -> str:
    """Upload *image_path* to S3 and return the s3_key (basename)."""
    from umrs.shared.storage.s3 import connect_to_s3
    s3, _ = connect_to_s3(os.path.expanduser(s3cfg_path), endpoint_url_override=endpoint_override)
    s3_key = Path(image_path).name
    s3.upload_file(str(image_path), bucket, s3_key)
    return s3_key


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


def _worker(
    worker_id: int,
    client_id: str,
    server_ip: str,
    control_port: int,
    tunnel_mode: bool,
    n_requests: int,
    warmup: int,
    username: str,
    k: int,
    s3_key: str,
    bucket: str,
) -> list[float]:
    mq = _make_mq(server_ip, control_port, tunnel_mode)
    latencies: list[float] = []
    try:
        for i in range(warmup + n_requests):
            t0 = time.perf_counter()
            mq.request_knn(username, k, s3_key, bucket)
            elapsed = time.perf_counter() - t0
            if i >= warmup:
                latencies.append(elapsed)
    finally:
        try:
            mq.close()
        except Exception:
            pass
    return latencies


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    data_sorted = sorted(data)
    idx = max(0, int(len(data_sorted) * p / 100) - 1)
    return data_sorted[min(idx, len(data_sorted) - 1)]


def _stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {}
    return {
        "n": len(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": _percentile(latencies_ms, 95),
        "p99_ms": _percentile(latencies_ms, 99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


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
    client_id_prefix: str,
) -> dict:
    futures_map: dict = {}
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N) as pool:
        for i in range(N):
            cid = f"{client_id_prefix}_{i + 1}" if N > 1 else client_id_prefix
            f = pool.submit(
                _worker,
                i, cid,
                server_ip, control_port, tunnel_mode,
                K, warmup,
                username, k, s3_key, bucket,
            )
            futures_map[f] = cid

        all_latencies: list[float] = []
        errors = 0
        for f in as_completed(futures_map):
            try:
                all_latencies.extend(f.result())
            except Exception as e:
                errors += 1
                print(f"  [worker error] {e}", file=sys.stderr)

    wall_elapsed = time.perf_counter() - wall_start
    total_req = len(all_latencies)
    throughput = total_req / wall_elapsed if wall_elapsed > 0 else 0.0
    ms = [x * 1000 for x in all_latencies]
    s = _stats(ms)
    s.update({"N": N, "K": K, "k": k, "errors": errors,
               "throughput_rps": throughput, "wall_s": wall_elapsed})
    return s


def main():
    parser = argparse.ArgumentParser(description="UMRS KNN latency benchmark")
    parser.add_argument("--image", "-i", required=True, help="Local query image path")
    parser.add_argument("--bucket", required=True, help="S3 path_to_bucket (full name)")
    parser.add_argument("--username", required=True, help="UMRS username")
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
    parser.add_argument("--client-id", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    connection = resolve_connection(
        explicit_server_ip=args.server_ip,
        explicit_control_port=args.control_port,
    )
    server_ip = connection.server_host
    control_port = connection.control_port
    tunnel_mode = connection.tunnel_mode

    print(f"Uploading query image '{args.image}' to S3 bucket '{args.bucket}'...")
    s3_key = _upload_query_image(args.s3cfg, args.bucket, args.image, args.endpoint)
    print(f"  s3_key = {s3_key}")

    print("Warming up embedding cache (1 KNN call)...")
    mq = _make_mq(server_ip, control_port, tunnel_mode)
    try:
        mq.request_knn(args.username, args.k, s3_key, args.bucket)
    finally:
        mq.close()
    print("  Cache warmed.")

    client_id_prefix = (
        args.client_id
        or os.environ.get("UMRS_CLIENT_ID")
        or os.environ.get("USER", "bench_user")
    )
    label = args.label or ""

    # Build sweep list: either N or k, not both simultaneously
    if args.sweep_k:
        k_values = [int(x) for x in args.sweep_k.split(",")]
        N_values = [args.concurrency]
        sweep_dim = "k"
    else:
        k_values = [args.k]
        N_values = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.concurrency]
        sweep_dim = "N"

    print(
        f"\nUMRS KNN latency benchmark"
        f"  server={server_ip}:{control_port}"
        f"  query={s3_key}  bucket={args.bucket}"
        f"  K={args.requests}  warmup={args.warmup}"
        + (f"  label={label}" if label else "")
    )
    header = f"{'N':>4}  {'k':>4}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  {'min':>8}  {'max':>8}  {'rps':>8}  {'err':>4}"
    print(header)
    print("-" * len(header))

    rows = []
    for N in N_values:
        for k in k_values:
            s = run_bench(
                N=N, K=args.requests,
                server_ip=server_ip, control_port=control_port, tunnel_mode=tunnel_mode,
                warmup=args.warmup,
                username=args.username, k=k, s3_key=s3_key, bucket=args.bucket,
                client_id_prefix=client_id_prefix,
            )
            def ms(key):
                return f"{s.get(key, float('nan')):.1f}"
            print(
                f"{s['N']:>4}  {s['k']:>4}  {ms('mean_ms'):>8}  {ms('p50_ms'):>8}  "
                f"{ms('p95_ms'):>8}  {ms('p99_ms'):>8}  "
                f"{ms('min_ms'):>8}  {ms('max_ms'):>8}  "
                f"{s['throughput_rps']:>8.1f}  {s['errors']:>4}"
            )
            if label:
                s["label"] = label
            s["query"] = s3_key
            rows.append(s)

    if args.csv:
        fieldnames = list(rows[0].keys())
        file_exists = os.path.isfile(args.csv)
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        with open(args.csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults {'appended to' if file_exists else 'saved to'} {args.csv}")


if __name__ == "__main__":
    main()
