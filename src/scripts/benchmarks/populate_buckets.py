#!/usr/bin/env python3
"""
Mass upload script — sequential, resumable.
Uploads 7000-image batches into multiple subdirs for multiple users,
through the UMRS client (client_umrs.py upload_dir).

Example:
  python populate_buckets.py --user 10:100000 --config-dir ~/.config/umrs/n_users/users
"""

import argparse
import itertools
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_SRC_DIR = _BENCH_DIR.parents[1]

# ── Defaults (all overridable on the command line) ───────────────────────────
CLIENT       = str(_SRC_DIR / "client_umrs.py")
CONFIG_DIR   = "~/.config/umrs/n_users/users"
DATA_DIR     = str(_SRC_DIR.parent / "data" / "images" / "7000")
BUCKET_BASE  = "bucket-3"
SLEEP_S      = 300
LOG_DIR      = str(_BENCH_DIR / "log")

DEFAULT_USERS = [
    {"z": 10, "target": 100_000},
    {"z": 11, "target": 200_000},
    {"z": 12, "target": 500_000},
    {"z": 13, "target": 1_000_000},
]

FILES_PER_ROUND = 7000

log = logging.getLogger("populate_buckets")


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/mass_upload.log",
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Also print to stdout
    log.addHandler(logging.StreamHandler(sys.stdout))


def all_prefixes():
    """Generate aa, ab, ..., zz"""
    for a, b in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=2):
        yield f"{a}{b}"


def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file) as f:
            return json.load(f)
    return {}


def save_state(state_file, state):
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def rounds_needed(target):
    return math.ceil(target / FILES_PER_ROUND)


def run_upload(z, prefix, round_idx, total_rounds, args):
    config = os.path.join(os.path.expanduser(args.config_dir), f"user1_{z}.json")
    bucket = f"{args.bucket_base}/7000{prefix}"
    pct = round(100 * round_idx / total_rounds, 1)
    log.info(f"user1_{z} | {bucket} | round {round_idx}/{total_rounds} ({pct}%)")
    subprocess.run([
        args.python, args.client,
        "-c", config,
        "--output", "text",
        "upload_dir", args.data_dir, bucket,
    ], check=True)


def populate_user(user, state, args):
    z = user["z"]
    total = rounds_needed(user["target"])
    prefixes = list(itertools.islice(all_prefixes(), total))
    key = str(z)

    # Find resume point
    done = state.get(key, 0)
    if done >= total:
        log.info(f"user1_{z} | already complete ({total} rounds)")
        return

    for round_idx, prefix in enumerate(prefixes[done:], done + 1):
        run_upload(z, prefix, round_idx, total, args)
        state[key] = round_idx
        save_state(args.state_file, state)

        if round_idx < total:
            log.info(f"user1_{z} | sleeping {args.sleep}s...")
            time.sleep(args.sleep)

    log.info(f"user1_{z} | DONE ({total} rounds, ~{total * FILES_PER_ROUND} files)")


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
                        "(default: 10:100000 11:200000 12:500000 13:1000000)")
    p.add_argument("--client", default=CLIENT, help="client_umrs.py path (default: src/client_umrs.py)")
    p.add_argument("--python", default="python", help="interpreter for the client (default: %(default)s)")
    p.add_argument("--config-dir", default=CONFIG_DIR,
                   help="directory of user1_<z>.json client configs (default: %(default)s)")
    p.add_argument("--data-dir", default=DATA_DIR,
                   help="directory of the 7000 source images (default: <repo>/data/images/7000)")
    p.add_argument("--bucket-base", default=BUCKET_BASE, help="bucket alias (default: %(default)s)")
    p.add_argument("--sleep", type=float, default=SLEEP_S,
                   help="seconds between rounds (default: %(default)s)")
    p.add_argument("--log-dir", default=LOG_DIR, help="log + state directory (default: benchmarks/log)")
    p.add_argument("--state-file", default=None,
                   help="resume state JSON (default: <log-dir>/mass_upload_state.json)")
    args = p.parse_args(argv)
    args.users = args.users or DEFAULT_USERS
    args.state_file = args.state_file or os.path.join(args.log_dir, "mass_upload_state.json")
    return args


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log_dir)
    state = load_state(args.state_file)

    for user in args.users:
        populate_user(user, state, args)

    log.info("All users complete.")


if __name__ == "__main__":
    main()
