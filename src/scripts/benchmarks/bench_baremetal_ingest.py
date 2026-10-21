#!/usr/bin/env python3
"""Bare-metal ingestion sweep with a PyTorch DataLoader: same steps and images as sweep_ingest.py, no UMRS server.

Per run, in one long-lived process: upload to S3 (new prefix) -> optional already-indexed check
(S3 HEAD + SQL) -> DataLoader workers (S3 GET + decode + transforms) -> ResNet-50 forward ->
store into a dedicated PostgreSQL database (same table schema as UMRS).
Writes log/baremetal_<tag>/baremetal.csv; plot it with: Rscript plot_breakdown.R <that csv>

Usage (from src/, with umrs_env):
  python scripts/benchmarks/bench_baremetal_ingest.py -c ~/.config/umrs/n_users/users/user1_1.json \
      --workers 16 --reps 3        # bucket: the ResNet modality's alias in the JSON, or --bucket
  python scripts/benchmarks/bench_baremetal_ingest.py --drop-db      # delete the benchmark database
"""
import argparse
import atexit
import contextlib
import csv
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info  # noqa: E402
from torchvision import transforms  # noqa: E402

# Same preprocessing as umrs.inference.resnet_50.preprocess_images_batch
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# The steps of the breakdown figure (plot_breakdown.R), with the same meaning as for UMRS
STEP_COLUMNS = ["client_startup", "client_upload", "index_check", "read_decode",
                "forward", "store", "server_other"]
# Finer measurements the steps are made of
DETAIL_COLUMNS = ["scan", "upload", "index_check_head", "index_check_sql"]


# ---------------------------------------------------------------------------
# DataLoader side (runs in worker processes)
# ---------------------------------------------------------------------------

def _connect_s3(s3_cfg):
    from umrs.shared.storage.s3 import connect_to_s3

    with contextlib.redirect_stdout(io.StringIO()):
        s3, _ = connect_to_s3(s3_cfg)
    return s3


class S3Images(Dataset):
    """Every key of the sweep; one item = (preprocessed tensor, index, ETag)."""

    def __init__(self, s3_cfg, bucket, keys):
        self.s3_cfg = s3_cfg
        self.bucket = bucket
        self.keys = keys
        self.s3 = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["s3"] = None  # boto3 clients cannot be sent to other processes
        return state

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        if self.s3 is None:
            self.s3 = _connect_s3(self.s3_cfg)
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.keys[i])
        img = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
        return TRANSFORM(img), i, obj["ETag"].strip('"')


def _worker_init(_worker_id):
    ds = get_worker_info().dataset
    ds.s3 = _connect_s3(ds.s3_cfg)


def _shutdown_workers(loader):
    it = getattr(loader, "_iterator", None)
    if it is not None:
        it._shutdown_workers()


class RunSampler(Sampler):
    """Indices of the current run, set before each run so the persistent workers are reused."""

    def __init__(self):
        self.indices = []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# ---------------------------------------------------------------------------
# PostgreSQL side
# ---------------------------------------------------------------------------

def db_params():
    from umrs.server.config import get_db_config

    cfg = get_db_config()
    return {"host": cfg["host"], "port": int(cfg["port"]),
            "user": cfg["superuser"], "password": cfg["superuser_password"]}


def admin_execute(params, statement):
    import psycopg2

    conn = psycopg2.connect(dbname="postgres", **params)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(statement)
            return cur.fetchall() if cur.description else None
    finally:
        conn.close()


def create_db(params, db_name):
    from psycopg2 import sql

    exists = admin_execute(params, sql.SQL("SELECT 1 FROM pg_database WHERE datname = {}")
                           .format(sql.Literal(db_name)))
    if not exists:
        admin_execute(params, sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))


def drop_db(params, db_name):
    from psycopg2 import sql

    admin_execute(params, sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))


def setup_table(conn, table, vector_size, meta_columns, hnsw):
    """Fresh table with the same schema and indexes as UMRS (umrs.server.sql.configure_table)."""
    from psycopg2 import sql

    cols = [sql.SQL("id SERIAL PRIMARY KEY"), sql.SQL("s3_object TEXT"), sql.SQL("checksum TEXT"),
            sql.SQL("date DATE"), sql.SQL("time TIME"),
            sql.SQL("embedding VECTOR({})").format(sql.Literal(vector_size)), sql.SQL("model TEXT")]
    cols += [sql.SQL("{} " + c["type"]).format(sql.Identifier(c["name"])) for c in meta_columns]
    cols.append(sql.SQL("UNIQUE(s3_object)"))
    t = sql.Identifier(table)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(t))
        cur.execute(sql.SQL("CREATE TABLE {} ({})").format(t, sql.SQL(", ").join(cols)))
        cur.execute(sql.SQL("CREATE INDEX ON {} (s3_object)").format(t))
        cur.execute(sql.SQL("CREATE INDEX ON {} (checksum)").format(t))
        if hnsw:
            cur.execute(sql.SQL("CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops) "
                                "WITH (m = 16, ef_construction = 64)").format(t))
    conn.commit()


def insert_sql(table, values_clause):
    from psycopg2 import sql

    return sql.SQL(
        "INSERT INTO {} (s3_object, embedding, date, time, checksum, model) VALUES " + values_clause +
        " ON CONFLICT (s3_object) DO UPDATE SET embedding = EXCLUDED.embedding, date = EXCLUDED.date,"
        " time = EXCLUDED.time, checksum = EXCLUDED.checksum, model = EXCLUDED.model"
    ).format(sql.Identifier(table))


def store(conn, table, mode, rows):
    """rows: (s3_object, embedding list, date, time, checksum, model)."""
    from psycopg2.extras import execute_values

    with conn.cursor() as cur:
        if mode == "batch":  # one statement and one commit per batch
            execute_values(cur, insert_sql(table, "%s").as_string(conn), rows, page_size=len(rows))
            conn.commit()
        else:  # per_row: one statement and one commit per object, like UMRS
            q = insert_sql(table, "(%s, %s, %s, %s, %s, %s)").as_string(conn) + " RETURNING id"
            for row in rows:
                cur.execute(q, row)
                cur.fetchone()
                conn.commit()


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def check_bucket_writable(s3, bucket, tag):
    """Fail in seconds, before the model and the DataLoader workers start."""
    from botocore.exceptions import ClientError

    key = f"baremetal-{tag}-write-test"
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=b"")
        s3.delete_object(Bucket=bucket, Key=key)
    except ClientError as e:
        sys.exit(f"Cannot write to bucket '{bucket}': {e}")
    print(f"bucket={bucket}")


def upload(s3, bucket, prefix, files, threads):
    def put(f):
        s3.put_object(Bucket=bucket, Key=f"{prefix}/{f.name}", Body=f.read_bytes())

    if threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(put, files))
    else:
        for f in files:
            put(f)


def index_check(s3, bucket, keys, conn, table):
    """Same work as UMRS filter_already_indexed: HEAD every object, then one SQL query."""
    from psycopg2 import sql

    def head(k):
        return k, s3.head_object(Bucket=bucket, Key=k)["ETag"].strip('"')

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=10) as pool:
        dict(pool.map(head, keys))
    head_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT s3_object, checksum FROM {} WHERE s3_object = ANY(%s)")
                    .format(sql.Identifier(table)), ([f"{bucket}/{k}" for k in keys],))
        cur.fetchall()
    return head_s, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", help="UMRS user JSON (model, S3 config, table schema)")
    p.add_argument("--bucket", help="bucket alias (e.g. bucket-3) or full name "
                                    "(user1-1.images-and-texts.bucket-3); default: the ResNet modality's alias")
    p.add_argument("--sizes", default="10,100,1000,2000,3000,4000,5000,6000,7000")
    p.add_argument("--images-root", default="/mnt/storage/ejeannot/images")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tag", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--workers", type=int, default=16, help="DataLoader num_workers")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--upload-threads", type=int, default=1, help="1 = sequential, like the UMRS client")
    p.add_argument("--index-check", action="store_true", help="also run the already-indexed check")
    p.add_argument("--store-mode", choices=["batch", "per_row"], default="batch")
    p.add_argument("--no-hnsw", action="store_true", help="table without the HNSW embedding index")
    p.add_argument("--pause", type=float, default=0.0, help="seconds between runs")
    p.add_argument("--db-name", default="umrs_baremetal_bench")
    p.add_argument("--drop-db-at-end", action="store_true")
    p.add_argument("--drop-db", action="store_true", help="only delete the benchmark database and exit")
    args = p.parse_args()

    params = db_params()
    if args.drop_db:
        drop_db(params, args.db_name)
        print(f"Database {args.db_name} dropped")
        return
    if not args.config:
        p.error("-c/--config is required")

    import psycopg2

    from umrs.inference.loading import load_model
    from umrs.shared.storage.s3 import make_path_to_bucket_string

    cfg = json.load(open(os.path.expanduser(args.config)))
    xp, mod = next((xp, m) for xp in cfg["experiences"] for m in xp["modalities"]
                   if m["model"]["model_type"] == "Resnet-50")
    alias = args.bucket or mod["s3"]["bucket_alias"]
    args.bucket = alias if "." in alias else make_path_to_bucket_string(xp["xp_name"], alias, cfg["username"])
    s3_cfg = cfg["s3_configuration_file"]
    s3 = _connect_s3(s3_cfg)
    check_bucket_writable(s3, args.bucket, args.tag)

    em = load_model(mod["model"], mod["modality_name"])
    net, device = em.model, em.device
    table = mod["sql"]["table_name"].lower()

    sizes = [int(x) for x in args.sizes.split(",")]
    runs, all_keys = [], []
    for rep in range(1, args.reps + 1):
        for n in sizes:
            d = Path(args.images_root) / str(n)
            files = sorted(f for f in d.iterdir() if f.is_file())
            if len(files) != n:
                print(f"WARNING: {d} has {len(files)} files, expected {n}", file=sys.stderr)
            prefix = f"baremetal-{args.tag}-n{n}-r{rep}"
            start = len(all_keys)
            all_keys += [f"{prefix}/{f.name}" for f in files]
            runs.append((rep, n, prefix, range(start, len(all_keys))))

    create_db(params, args.db_name)
    conn = psycopg2.connect(dbname=args.db_name, **params)
    setup_table(conn, table, int(mod["model"]["vector_size"]),
                mod["sql"].get("meta_data", []), hnsw=not args.no_hnsw)

    sampler = RunSampler()
    loader = DataLoader(
        S3Images(s3_cfg, args.bucket, all_keys), batch_size=args.batch_size, sampler=sampler,
        num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True,
        prefetch_factor=2 if args.workers else None,
        multiprocessing_context="spawn" if args.workers else None,
        worker_init_fn=_worker_init if args.workers else None,
    )
    t0 = time.perf_counter()
    list(loader)  # empty run: starts the workers and their S3 clients before timing
    # registered after torch's own exit hooks, so it runs first and stops the persistent workers
    atexit.register(_shutdown_workers, loader)
    with torch.no_grad():
        net(torch.zeros(args.batch_size, 3, 224, 224, device=device)).cpu()
    print(f"tag={args.tag}  workers={args.workers}  store={args.store_mode}  "
          f"hnsw={not args.no_hnsw}  index_check={args.index_check}  device={device}  "
          f"start-up={time.perf_counter() - t0:.1f}s")

    log_dir = Path(__file__).resolve().parent / "log" / f"baremetal_{args.tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "baremetal.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tag", "rep", "n", "prefix", "workers", "batch_size", "store_mode", "hnsw",
                         "with_index_check", "upload_threads"]
                        + STEP_COLUMNS + ["total"] + DETAIL_COLUMNS)
        for rep, n, prefix, idx in runs:
            t_run = time.perf_counter()
            # client part, as in UMRS: list the directory, then upload (no server to notify)
            t0 = time.perf_counter()
            files = sorted(f for f in (Path(args.images_root) / str(n)).iterdir() if f.is_file())
            scan_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            upload(s3, args.bucket, prefix, files, args.upload_threads)
            upload_s = time.perf_counter() - t0

            head_s = sql_s = read_s = fwd_s = store_s = 0.0
            if args.index_check:
                head_s, sql_s = index_check(s3, args.bucket, [all_keys[i] for i in idx], conn, table)

            sampler.indices = list(idx)
            batches = iter(loader)
            while True:
                t0 = time.perf_counter()
                try:
                    x, ids, etags = next(batches)
                except StopIteration:
                    break
                read_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                with torch.no_grad():
                    emb = net(x.to(device, non_blocking=True)).cpu().numpy()
                fwd_s += time.perf_counter() - t0

                t0 = time.perf_counter()
                now = datetime.now()
                rows = [(f"{args.bucket}/{all_keys[i]}", e.tolist(), now.strftime("%Y-%m-%d"),
                         now.strftime("%H:%M:%S"), etag, em.model_name)
                        for i, e, etag in zip(ids.tolist(), emb, etags)]
                store(conn, table, args.store_mode, rows)
                store_s += time.perf_counter() - t0

            total = time.perf_counter() - t_run
            steps = {
                "client_startup": 0.0,                 # persistent process: nothing per request
                "client_upload": scan_s + upload_s,    # UMRS: scan + upload + notify
                "index_check": head_s + sql_s,         # UMRS: s3_events + HEAD + SQL
                "read_decode": read_s,
                "forward": fwd_s,
                "store": store_s,
            }
            steps["server_other"] = total - sum(steps.values())
            details = {"scan": scan_s, "upload": upload_s,
                       "index_check_head": head_s, "index_check_sql": sql_s}
            writer.writerow([args.tag, rep, n, prefix, args.workers, args.batch_size, args.store_mode,
                             not args.no_hnsw, args.index_check, args.upload_threads]
                            + [f"{steps[c]:.3f}" for c in STEP_COLUMNS] + [f"{total:.3f}"]
                            + [f"{details[c]:.3f}" for c in DETAIL_COLUMNS])
            fh.flush()
            print(f"rep={rep} n={n:>5}  total={total:9.3f}s  upload={upload_s:8.3f}  "
                  f"check={head_s + sql_s:7.3f}  read={read_s:7.3f}  fwd={fwd_s:7.3f}  store={store_s:8.3f}",
                  flush=True)
            time.sleep(args.pause)

    conn.close()
    if args.drop_db_at_end:
        drop_db(params, args.db_name)
        print(f"Database {args.db_name} dropped")
    print(f"\n{csv_path}\nPlot: Rscript scripts/benchmarks/plot_breakdown.R {csv_path}")


if __name__ == "__main__":
    main()
