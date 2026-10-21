#!/usr/bin/env python3
"""Ingestion sweep: upload_dir --blocking of <images-root>/<N> for each N, each run to a new prefix.

Every run uses a fresh S3 prefix, so no object is skipped as already indexed.
Client-side times go to a CSV; per-step server times come from the server log:
  python parse_server_log.py <server.log> --csv --timing > log/sweep.csv

Usage (from src/, with umrs_env):
  python scripts/benchmarks/sweep_ingest.py -c ~/.config/umrs/n_users/users/user1_1.json \
      --sizes 10,100,1000,2000,3000,4000,5000,6000,7000 --reps 3
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
CLIENT_FIELDS = ("scan_s", "upload_s", "notify_s")


def client_timing(out_file):
    """key=value fields of the last [CLIENT_TIMING] line printed by client_umrs.py upload_dir."""
    fields = {}
    for line in out_file.read_text(errors="replace").splitlines():
        if "[CLIENT_TIMING]" in line:
            fields = dict(kv.split("=", 1) for kv in line.split("[CLIENT_TIMING]", 1)[1].split() if "=" in kv)
    return fields


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True, help="UMRS user JSON")
    p.add_argument("--sizes", default="10,100,1000,2000,3000,4000,5000,6000,7000")
    p.add_argument("--images-root", default="/mnt/storage/ejeannot/images",
                   help="directory holding one sub-directory per size: <root>/<N>")
    p.add_argument("--bucket", default="bucket-3", help="bucket alias")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--pause", type=float, default=60.0, help="seconds between runs")
    p.add_argument("--tag", default=datetime.now().strftime("%Y%m%d-%H%M%S"),
                   help="part of the S3 prefix, keeps runs unique")
    p.add_argument("--client", default=str(SRC / "client_umrs.py"))
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = p.parse_args()

    sizes = [int(x) for x in args.sizes.split(",")]
    for n in sizes:
        d = Path(args.images_root) / str(n)
        n_files = sum(1 for f in d.iterdir() if f.is_file()) if d.is_dir() else 0
        if n_files != n:
            print(f"WARNING: {d} has {n_files} files, expected {n}", file=sys.stderr)

    log_dir = Path(__file__).resolve().parent / "log" / f"sweep_{args.tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "client.csv"
    print(f"tag={args.tag}  sizes={sizes}  reps={args.reps}  logs={log_dir}")

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tag", "rep", "n", "prefix", "start", "end", "client_wall(s)", "returncode",
                         "scan(s)", "upload(s)", "notify(s)"])
        for rep in range(1, args.reps + 1):
            for n in sizes:
                prefix = f"{args.bucket}/sweep-{args.tag}-n{n}-r{rep}"
                cmd = [sys.executable, args.client, "-c", os.path.expanduser(args.config),
                       "--output", "text", "upload_dir", "--blocking",
                       str(Path(args.images_root) / str(n)), prefix]
                if args.dry_run:
                    print(" ".join(cmd))
                    continue

                out_file = log_dir / f"n{n}_r{rep}.out"
                start = datetime.now()
                t0 = time.perf_counter()
                with open(out_file, "w") as out:
                    rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=SRC).returncode
                wall = time.perf_counter() - t0
                end = datetime.now()
                ct = client_timing(out_file)

                writer.writerow([args.tag, rep, n, prefix, start.isoformat(timespec="milliseconds"),
                                 end.isoformat(timespec="milliseconds"), f"{wall:.3f}", rc]
                                + [ct.get(k, "NA") for k in CLIENT_FIELDS])
                fh.flush()
                print(f"rep={rep} n={n:>5}  wall={wall:8.2f}s  upload={ct.get('upload_s', 'NA')}s  "
                      f"notify={ct.get('notify_s', 'NA')}s  rc={rc}  {start:%H:%M:%S}->{end:%H:%M:%S}",
                      flush=True)
                time.sleep(args.pause)

    print(f"\nClient times: {csv_path}")
    print("Server steps: python scripts/benchmarks/parse_server_log.py <server.log> --csv --timing")


if __name__ == "__main__":
    main()
