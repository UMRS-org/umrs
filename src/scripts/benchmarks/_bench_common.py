"""Latency statistics and CSV output shared by bench_latency.py and bench_knn.py."""

from __future__ import annotations

import csv
import os
import statistics


def fmt_ms(stats: dict, key: str) -> str:
    return f"{stats.get(key, float('nan')):.1f}"


def percentile(data: list[float], p: float) -> float:
    if not data:
        return float("nan")
    data_sorted = sorted(data)
    idx = max(0, int(len(data_sorted) * p / 100) - 1)
    return data_sorted[min(idx, len(data_sorted) - 1)]


def stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {}
    return {
        "n": len(latencies_ms),
        "mean_ms": statistics.mean(latencies_ms),
        "p50_ms": statistics.median(latencies_ms),
        "p95_ms": percentile(latencies_ms, 95),
        "p99_ms": percentile(latencies_ms, 99),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }


def client_id_prefix(explicit: str | None) -> str:
    return explicit or os.environ.get("UMRS_CLIENT_ID") or os.environ.get("USER", "bench_user")


def append_csv(path: str, rows: list[dict]) -> None:
    """Append *rows* to *path* (header on first write), creating its directory."""
    file_exists = os.path.isfile(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults {'appended to' if file_exists else 'saved to'} {path}")
