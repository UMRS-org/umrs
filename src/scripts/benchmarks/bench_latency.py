#!/usr/bin/env python3
"""End-to-end latency benchmark for UMRS ZMQ status queries.

Each virtual user gets its own ZMQ REQ socket (one MessageQueue per thread)
so the server sees truly concurrent requests, avoiding subprocess fork overhead.

Usage
-----
# 1 concurrent user, 50 requests
python bench_latency.py -N 1

# Sweep N = 1, 5, 10, 25, 50 with 100 requests each
python bench_latency.py --sweep 1,5,10,25,50 -K 100

# Use a specific server / port
python bench_latency.py --server-ip 127.0.0.1 --control-port 5555 -N 10

# Measure a different command (status for a specific user)
python bench_latency.py -N 10 --cmd status --target process_objects
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make sure the src/ directory is on the path when running from scripts/benchmarks/
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from umrs.shared.connection_profiles import resolve_connection
from umrs.client.control_api import UMRSControlAPI


def _lazy_mq():
    """Import MessageQueue lazily (requires umrs[client-full])."""
    from umrs.core.MessageQueue import MessageQueue
    return MessageQueue


# --- per-thread worker -------------------------------------------------------

def _worker(
    worker_id: int,
    client_id: str,
    server_ip: str,
    control_port: int,
    tunnel_mode: bool,
    n_requests: int,
    cmd: str,
    target: str,
    warmup: int,
) -> list[float]:
    """Run *n_requests* timed ZMQ calls in this thread.

    Returns the list of per-request wall-clock times in seconds
    (warm measurements only — warmup requests are excluded).
    """
    MessageQueue = _lazy_mq()
    mq = MessageQueue(
        role="client",
        queue_type="control",
        server_ip=server_ip,
        port=control_port,
        tunnel_mode=tunnel_mode,
        prefer_env_server_host=False,
    )
    api = UMRSControlAPI(mq, client_id)
    latencies: list[float] = []
    try:
        for i in range(warmup + n_requests):
            t0 = time.perf_counter()
            if cmd == "status-all":
                api.request_status_all()
            elif cmd == "status":
                api.request_status(target)
            elif cmd == "ping":
                api.ping()
            else:
                raise ValueError(f"Unknown cmd: {cmd!r}")
            elapsed = time.perf_counter() - t0
            if i >= warmup:
                latencies.append(elapsed)
    finally:
        try:
            mq.close()
        except Exception:
            pass
    return latencies


# --- stats helpers -----------------------------------------------------------

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


# --- main benchmark run ------------------------------------------------------

def run_bench(
    N: int,
    K: int,
    server_ip: str,
    control_port: int,
    tunnel_mode: bool,
    cmd: str,
    target: str,
    client_id_prefix: str,
    warmup: int,
) -> dict:
    """Launch N threads, each doing K requests. Returns aggregated stats."""
    futures_map: dict = {}
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=N) as pool:
        for i in range(N):
            cid = f"{client_id_prefix}_{i + 1}" if N > 1 else client_id_prefix
            f = pool.submit(
                _worker,
                i,
                cid,
                server_ip,
                control_port,
                tunnel_mode,
                K,
                cmd,
                target,
                warmup,
            )
            futures_map[f] = cid

        all_latencies: list[float] = []
        errors = 0
        for f in as_completed(futures_map):
            try:
                lats = f.result()
                all_latencies.extend(lats)
            except Exception as e:
                errors += 1
                print(f"  [worker error] {e}", file=sys.stderr)

    wall_elapsed = time.perf_counter() - wall_start
    total_req = len(all_latencies)
    throughput = total_req / wall_elapsed if wall_elapsed > 0 else 0.0

    ms = [x * 1000 for x in all_latencies]
    s = _stats(ms)
    s["N"] = N
    s["K"] = K
    s["errors"] = errors
    s["throughput_rps"] = throughput
    s["wall_s"] = wall_elapsed
    return s


# --- CLI ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UMRS ZMQ latency benchmark")
    parser.add_argument("-N", "--concurrency", type=int, default=1,
                        help="Number of concurrent virtual users")
    parser.add_argument("-K", "--requests", type=int, default=50,
                        help="Requests per user (excluding warmup)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup requests per user (not counted)")
    parser.add_argument("--sweep", type=str, default=None,
                        help="Comma-separated list of N values to sweep, e.g. 1,5,10,50")
    parser.add_argument("--server-ip", type=str, default=None)
    parser.add_argument("--control-port", type=int, default=None)
    parser.add_argument("--cmd", type=str, default="status-all",
                        choices=["status-all", "status", "ping"],
                        help="Command to benchmark")
    parser.add_argument("--target", type=str, default="process_objects",
                        help="Target for --cmd status (e.g. process_objects, all)")
    parser.add_argument("--client-id", type=str, default=None,
                        help="Client ID prefix (default: $UMRS_CLIENT_ID or $USER)")
    parser.add_argument("--label", type=str, default=None,
                        help="Annotation written into CSV (e.g. 'cache=on' or 'cache=off')")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to a CSV file (appends if exists)")
    args = parser.parse_args()

    connection = resolve_connection(
        explicit_server_ip=args.server_ip,
        explicit_control_port=args.control_port,
    )
    server_ip = connection.server_host
    control_port = connection.control_port
    tunnel_mode = connection.tunnel_mode

    client_id_prefix = (
        args.client_id
        or os.environ.get("UMRS_CLIENT_ID")
        or os.environ.get("USER", "bench_user")
    )

    Ns = (
        [int(x) for x in args.sweep.split(",")]
        if args.sweep
        else [args.concurrency]
    )

    label = args.label or ""
    print(
        f"UMRS latency benchmark  cmd={args.cmd}"
        f"  server={server_ip}:{control_port}"
        f"  K={args.requests}  warmup={args.warmup}"
        + (f"  label={label}" if label else "")
    )
    print(f"{'N':>4}  {'mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}  "
          f"{'min':>8}  {'max':>8}  {'rps':>8}  {'err':>4}")
    print("-" * 78)

    rows = []
    for N in Ns:
        s = run_bench(
            N=N,
            K=args.requests,
            server_ip=server_ip,
            control_port=control_port,
            tunnel_mode=tunnel_mode,
            cmd=args.cmd,
            target=args.target,
            client_id_prefix=client_id_prefix,
            warmup=args.warmup,
        )
        def ms(key):
            return f"{s.get(key, float('nan')):.1f}"

        print(
            f"{s['N']:>4}  {ms('mean_ms'):>8}  {ms('p50_ms'):>8}  "
            f"{ms('p95_ms'):>8}  {ms('p99_ms'):>8}  "
            f"{ms('min_ms'):>8}  {ms('max_ms'):>8}  "
            f"{s['throughput_rps']:>8.1f}  {s['errors']:>4}"
        )
        if label:
            s["label"] = label
        rows.append(s)

    if args.csv:
        import os as _os
        fieldnames = list(rows[0].keys())
        file_exists = _os.path.isfile(args.csv)
        with open(args.csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults {'appended to' if file_exists else 'saved to'} {args.csv}")


if __name__ == "__main__":
    main()
