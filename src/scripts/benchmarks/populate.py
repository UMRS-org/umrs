#!/usr/bin/env python3
"""
Populate MinIO buckets for mass upload benchmark.
1. Upload data/images/7000/ → 7000aa/ for each user (with retry/backoff)
2. Server-side copy 7000aa/ → 7000ab/, 7000ac/, ... for remaining rounds

No UMRS involvement — pure boto3.
"""

import boto3
import configparser
import itertools
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from botocore.exceptions import ClientError
import urllib3
urllib3.disable_warnings()

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR     = "/home/emmanuel.jeannot/Unified_Multimodal_Retriever_Service/data/images/7000/"
S3CFG_TMPL   = "/home/emmanuel.jeannot/.s3cfg_user1_{z}"
LOG_DIR      = "/mnt/ejeannot/log"
STATE_FILE   = "/mnt/data/ejeannot/log/populate_state.json"
FILES_PER_ROUND = 7000
LOG_DIR      = "/mnt/storage/ejeannot/log"
STATE_FILE   = "/mnt/storage/ejeannot/log/mass_upload_state.json"
RETRY_DELAYS = [1, 2, 5, 5, 10]

USERS = [
    {"z": 10, "target": 100_000},
    {"z": 11, "target": 200_000},
    {"z": 12, "target": 500_000},
    {"z": 13, "target": 1_000_000},
]

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=f"{LOG_DIR}/populate_buckets.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))


# ── Helpers ──────────────────────────────────────────────────────────────────
def rounds_needed(target):
    return math.ceil(target / FILES_PER_ROUND)


def all_prefixes():
    for a, b in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=2):
        yield f"7000{a}{b}"


def make_s3_client(z):
    cfg = configparser.ConfigParser()
    cfg.read(S3CFG_TMPL.format(z=z))
    d = cfg["default"]
    protocol = "https" if d.get("use_https", "False").strip() == "True" else "http"
    return boto3.client(
        "s3",
        endpoint_url=f"{protocol}://{d['host_base']}",
        aws_access_key_id=d["access_key"],
        aws_secret_access_key=d["secret_key"],
        verify=False,
    )


def bucket_name(z):
    return f"user1-{z}.images-and-texts.bucket-3"


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ── Upload with retry ────────────────────────────────────────────────────────
def upload_file_with_retry(s3, local_path, bucket, s3_key):
    for attempt, delay in enumerate(RETRY_DELAYS + [None]):
        try:
            s3.upload_file(str(local_path), bucket, s3_key, ExtraArgs={"ChecksumAlgorithm": "SHA256"} if False else {})
            return
        except Exception as e:
            if delay is None:
                raise RuntimeError(f"Failed after all retries: {local_path} → {s3_key}: {e}")
            log.warning(f"Upload failed ({e}), retry in {delay}s...")
            time.sleep(delay)


def upload_initial(s3, z, bucket, prefix, state):
    key = f"{z}_upload"
    if state.get(key) == "done":
        log.info(f"user1_{z} | initial upload already done, skipping")
        return

    files = sorted(Path(DATA_DIR).iterdir())
    total = len(files)
    log.info(f"user1_{z} | uploading {total} files → {bucket}/{prefix}/")

    for i, f in enumerate(files, 1):
        s3_key = f"{prefix}/{f.name}"
        upload_file_with_retry(s3, f, bucket, s3_key)
        if i % 500 == 0 or i == total:
            pct = round(100 * i / total, 1)
            log.info(f"user1_{z} | upload {i}/{total} ({pct}%)")

    state[key] = "done"
    save_state(state)
    log.info(f"user1_{z} | initial upload complete")


# ── Server-side copy ─────────────────────────────────────────────────────────
def server_copy(s3, bucket, src_prefix, dst_prefix, z, round_idx, total_rounds):
    paginator = s3.get_paginator("list_objects_v2")
    copied = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{src_prefix}/"):
        for obj in page.get("Contents", []):
            src_key = obj["Key"]
            dst_key = dst_prefix + "/" + src_key[len(src_prefix) + 1:]
            s3.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": src_key},
                Key=dst_key,
            )
            copied += 1
    pct = round(100 * round_idx / total_rounds, 1)
    log.info(f"user1_{z} | copy {src_prefix} → {dst_prefix} | {copied} files | round {round_idx}/{total_rounds} ({pct}%)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    state = load_state()

    for user in USERS:
        z = user["z"]
        total = rounds_needed(user["target"])
        prefixes = list(itertools.islice(all_prefixes(), total))
        bucket = bucket_name(z)
        s3 = make_s3_client(z)

        log.info(f"=== user1_{z} | {total} rounds | bucket={bucket} ===")

        # Step 1: initial upload → first prefix (7000aa)
        upload_initial(s3, z, bucket, prefixes[0], state)

        # Step 2: server-side copies
        for i, dst_prefix in enumerate(prefixes[1:], 2):
            copy_key = f"{z}_copy_{dst_prefix}"
            if state.get(copy_key) == "done":
                log.info(f"user1_{z} | {dst_prefix} already copied, skipping")
                continue
            server_copy(s3, bucket, prefixes[0], dst_prefix, z, i, total)
            state[copy_key] = "done"
            save_state(state)

        log.info(f"user1_{z} | DONE — {total} prefixes ready in {bucket}")

    log.info("All users complete.")


if __name__ == "__main__":
    main()
