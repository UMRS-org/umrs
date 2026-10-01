#!/usr/bin/env python3
"""
Mass upload script — sequential, resumable.
Uploads 7000-image batches into multiple subdirs for multiple users.
"""

import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
CLIENT       = "/home/emmanuel.jeannot/Unified_Multimodal_Retriever_Service/src/client_umrs.py"
CONFIG_DIR   = "/home/emmanuel.jeannot/.config/umrs/n_users/users"
DATA_DIR     = "/home/emmanuel.jeannot/Unified_Multimodal_Retriever_Service/data/images/7000/"
BUCKET_BASE  = "bucket-3"
SLEEP_S      = 600
LOG_DIR      = "/mnt/data/ejeannot/log"
STATE_FILE   = "/mnt/data/ejeannot/log/mass_upload_state.json"

USERS = [
    {"z": 10, "target": 100_000},
    {"z": 11, "target": 200_000},
    {"z": 12, "target": 500_000},
    {"z": 13, "target": 1_000_000},
]

FILES_PER_ROUND = 7000

# ── Logging ──────────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=f"{LOG_DIR}/mass_upload.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger()
# Also print to stdout
log.addHandler(logging.StreamHandler(sys.stdout))


def all_prefixes():
    """Generate aa, ab, ..., zz"""
    for a, b in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=2):
        yield f"{a}{b}"


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def rounds_needed(target):
    import math
    return math.ceil(target / FILES_PER_ROUND)


def run_upload(z, prefix, round_idx, total_rounds):
    config = f"{CONFIG_DIR}/user1_{z}.json"
    bucket = f"{BUCKET_BASE}/7000{prefix}"
    pct = round(100 * round_idx / total_rounds, 1)
    log.info(f"user1_{z} | {bucket} | round {round_idx}/{total_rounds} ({pct}%)")
    subprocess.run([
        "python", CLIENT,
        "-c", config,
        "--output", "text",
        "upload_dir", DATA_DIR, bucket,
    ], check=True)


def main():
    state = load_state()

    for user in USERS:
        z = user["z"]
        total = rounds_needed(user["target"])
        prefixes = list(itertools.islice(all_prefixes(), total))
        key = str(z)

        # Find resume point
        done = state.get(key, 0)
        if done >= total:
            log.info(f"user1_{z} | already complete ({total} rounds)")
            continue

        for i, prefix in enumerate(prefixes):
            round_idx = i + 1
            if round_idx <= done:
                continue  # skip already done rounds

            run_upload(z, prefix, round_idx, total)
            state[key] = round_idx
            save_state(state)

            if round_idx < total:
                log.info(f"user1_{z} | sleeping {SLEEP_S}s...")
                time.sleep(SLEEP_S)

        log.info(f"user1_{z} | DONE ({total} rounds, ~{total * FILES_PER_ROUND} files)")

    log.info("All users complete.")


if __name__ == "__main__":
    main()
