#!/usr/bin/env python3
"""
Populate MinIO buckets for mass upload benchmark.
1. Upload data/images/5000/ → 5000aa/ for each user
2. Server-side copy 5000aa/ → 5000ab/, 5000ac/, ... for remaining rounds

No UMRS server involvement — boto3 with UMRS's S3 helpers: transient errors
are retried by botocore (standard mode, UMRS_S3_MAX_ATTEMPTS), unchanged files
are not re-uploaded (UMRS_S3_UPLOAD_SKIP_UNCHANGED). TLS certificates are
verified unless the s3cfg says check_ssl_certificate = False or --insecure is
given. Resumable: finished steps are recorded in the state file and skipped
on the next run.

Example:
  python populate.py --user 14:20000 --s3cfg-template '~/.s3cfg_user1_{z}'
"""

import argparse
import configparser
import itertools
import json
import logging
import math
import os
import sys
from pathlib import Path

import boto3
from tqdm import tqdm

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parents[2]
if str(_BENCH_DIR.parents[1]) not in sys.path:  # src/, for umrs
    sys.path.insert(0, str(_BENCH_DIR.parents[1]))

from umrs.shared.storage.s3_upload import s3_client_config, upload_local_file

# ── Defaults (all overridable on the command line) ───────────────────────────
DATA_DIR        = str(_REPO_ROOT / "data" / "images" / "5000")
S3CFG_TMPL      = "~/.s3cfg_user1_{z}"
LOG_DIR         = str(_BENCH_DIR / "log")
FILES_PER_ROUND = 5000

# Upstream's last sweep; the larger tiers (z=7:10000, 9:50000, 10:100000,
# 11:200000, 12:500000, 13:1000000) were commented out there.
DEFAULT_USERS = [
    {"z": 14, "target": 20_000},
]

log = logging.getLogger("populate")


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/populate_buckets.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log.addHandler(logging.StreamHandler(sys.stdout))


# ── Helpers ──────────────────────────────────────────────────────────────────
def rounds_needed(target):
    return math.ceil(target / FILES_PER_ROUND)


def all_prefixes():
    for a, b in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=2):
        yield f"5000{a}{b}"


def make_client(s3cfg_path, insecure=False):
    """boto3 client for the s3cfg; verifies TLS like ``connect_to_s3`` does."""
    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser(s3cfg_path))
    d = cfg["default"]
    protocol = "https" if d.getboolean("use_https", fallback=False) else "http"
    verify = not insecure and d.getboolean("check_ssl_certificate", fallback=True)
    return boto3.client(
        "s3",
        endpoint_url=f"{protocol}://{d['host_base']}",
        aws_access_key_id=d["access_key"],
        aws_secret_access_key=d["secret_key"],
        verify=verify,
        # 10 s instead of the 60 s default; botocore is the one retry layer.
        config=s3_client_config(connect_timeout=5, read_timeout=10),
    )


def bucket_name(z):
    return f"user1-{z}.images-and-texts.bucket-3"


def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(state_file, state):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def upload_initial(client, z, bucket, prefix, state, data_dir, state_file):
    key = f"{z}_upload"
    if state.get(key) == "done":
        log.info(f"user1_{z} | initial upload already done, skipping")
        return

    files = sorted(Path(data_dir).iterdir())
    total = len(files)
    log.info(f"user1_{z} | uploading {total} files → {bucket}/{prefix}/")

    for i, f in enumerate(files, 1):
        s3_key = f"{prefix}/{f.name}"
        upload_local_file(client, bucket, s3_key, str(f))
        if i % 500 == 0 or i == total:
            pct = round(100 * i / total, 1)
            log.info(f"user1_{z} | upload {i}/{total} ({pct}%)")

    state[key] = "done"
    save_state(state_file, state)
    log.info(f"user1_{z} | initial upload complete")


# ── Server-side copy ──────────────────────────────────────────────────────────
def server_copy(client, bucket, src_prefix, dst_prefix, z, round_idx, total_rounds):
    paginator = client.get_paginator("list_objects_v2")

    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{src_prefix}/"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    pct = round(100 * round_idx / total_rounds, 1)
    desc = f"user1_{z} | round {round_idx}/{total_rounds} ({pct}%) {src_prefix}→{dst_prefix}"

    with tqdm(total=len(keys), desc=desc, unit="file") as bar:
        for src_key in keys:
            dst_key = dst_prefix + "/" + src_key[len(src_prefix) + 1:]
            client.copy_object(
                Bucket=bucket, CopySource={"Bucket": bucket, "Key": src_key}, Key=dst_key,
            )
            bar.update(1)

    log.info(
        f"user1_{z} | copy {src_prefix} → {dst_prefix} | "
        f"{len(keys)} files | round {round_idx}/{total_rounds} ({pct}%)"
    )


def populate_user(user, state, args):
    z = user["z"]
    total = rounds_needed(user["target"])
    prefixes = list(itertools.islice(all_prefixes(), total))
    bucket = bucket_name(z)
    client = make_client(args.s3cfg_template.format(z=z), insecure=args.insecure)

    log.info(f"=== user1_{z} | {total} rounds | bucket={bucket} ===")

    # Step 1: initial upload → first prefix (5000aa)
    upload_initial(client, z, bucket, prefixes[0], state, args.data_dir, args.state_file)

    # Step 2: server-side copy of the first prefix into the others
    for i, dst_prefix in enumerate(prefixes[1:], 2):
        copy_key = f"{z}_copy_{dst_prefix}"
        if state.get(copy_key) == "done":
            log.info(f"user1_{z} | {dst_prefix} already copied, skipping")
            continue
        server_copy(client, bucket, prefixes[0], dst_prefix, z, i, total)
        state[copy_key] = "done"
        save_state(args.state_file, state)

    log.info(f"user1_{z} | DONE — {total} prefixes ready in {bucket}")


def _user_spec(text):
    z, _, target = text.partition(":")
    try:
        return {"z": int(z), "target": int(target)}
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected Z:TARGET, got {text!r}") from None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", dest="users", action="append", type=_user_spec, metavar="Z:TARGET",
                   help="user index z (user1_<z>) and object target; repeatable "
                        "(default: 14:20000)")
    p.add_argument("--data-dir", default=DATA_DIR,
                   help="directory of the 5000 source images (default: <repo>/data/images/5000)")
    p.add_argument("--s3cfg-template", default=S3CFG_TMPL,
                   help="s3cmd config per user, {z} is replaced (default: %(default)s)")
    p.add_argument("--log-dir", default=LOG_DIR, help="log + state directory (default: benchmarks/log)")
    p.add_argument("--state-file", default=None,
                   help="resume state JSON (default: <log-dir>/populate_state.json)")
    p.add_argument("--insecure", action="store_true",
                   help="do not verify the S3 endpoint's TLS certificate (e.g. a "
                        "self-signed test MinIO); verified by default")
    args = p.parse_args(argv)
    args.users = args.users or DEFAULT_USERS
    args.state_file = args.state_file or os.path.join(args.log_dir, "populate_state.json")
    return args


# ── Main ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_dir)
    state = load_state(args.state_file)

    for user in args.users:
        populate_user(user, state, args)

    log.info("All users complete.")


if __name__ == "__main__":
    main()
