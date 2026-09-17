#!/usr/bin/env python3
"""
Monitor UMRS users processing status.
Usage: ./monitor_users.py <N> [--config-dir DIR] [--interval SEC] [--client PATH]
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime

from tqdm import tqdm

DEFAULT_CONFIG_DIR = "~/.config/umrs/n_users/users"
DEFAULT_CLIENT     = "./client_umrs.py"
DEFAULT_INTERVAL   = 5.0
ITEM_WIDTH         = 14

ICON_PENDING = "⚪"
ICON_RUNNING = "🔴"
ICON_DONE    = "🟢"


def query_user(user: str, config_dir: str, client: str, debug: bool = False) -> tuple:
    """
    `entries` accumulates one item per run (Done stays forever, a new run
    appends a new thread_id). The only entry that reflects the CURRENT run
    is the most recent one (max updated_at) — summing statuses across all
    entries mixes past and present runs.

    Returns (is_running, latest_status, latest_ts)
    """
    config_file = os.path.expanduser(f"{config_dir}/{user}.json")
    cmd = [client, "-c", config_file, "--output", "json", "status", "--username", user]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)

                if debug:
                    print(f"----- raw JSON received for {user} -----", flush=True)
                    print(json.dumps(data, indent=2, ensure_ascii=False), flush=True)
                    print("-" * 42, flush=True)

                summaries = data.get("summaries", {})
                entries = (summaries.get("process_objects") or
                           summaries.get("PROCESS_OBJECTS") or {}).get("data", [])

                dated = [e for e in entries if e.get("updated_at")]
                if not dated:
                    return None, None, None

                latest = max(dated, key=lambda e: e["updated_at"])
                latest_status = latest.get("status")
                latest_ts = latest["updated_at"]
                is_running = latest_status == "Running"

                return is_running, latest_status, latest_ts
            except json.JSONDecodeError:
                continue

    if debug:
        print(f"----- no parsable JSON for {user} -----", flush=True)
        print(result.stdout, flush=True)
        print("-" * 42, flush=True)

    return None, None, None


def fmt_ts(iso_ts):
    if not iso_ts:
        return ""
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_ts


def render(users, state, last_done_ts, start_time, poll_count):
    cols, _ = shutil.get_terminal_size(fallback=(80, 24))
    items_per_row = max(1, cols // ITEM_WIDTH)

    lines = []
    elapsed = int(time.time() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    n_done    = sum(1 for u in users if state[u] == "done")
    n_running = sum(1 for u in users if state[u] == "running")
    n_pending = sum(1 for u in users if state[u] == "pending")

    poll_str = f"Poll #{poll_count}" if poll_count > 0 else "Poll #0 (first scan in progress...)"
    header = (f" Done: {n_done}/{len(users)}  "
              f"🔴 Running: {n_running}  "
              f"⚪ Pending: {n_pending}  "
              f"⏱ {h:02d}:{m:02d}:{s:02d}  "
              f"🔄 {poll_str}")
    lines.append(header)
    lines.append("─" * min(len(header), cols))

    icon_map = {"pending": ICON_PENDING, "running": ICON_RUNNING, "done": ICON_DONE}
    row_buf = []
    for i, u in enumerate(users):
        cell = f"{u} {icon_map[state[u]]}".ljust(ITEM_WIDTH)
        row_buf.append(cell)
        if len(row_buf) == items_per_row or i == len(users) - 1:
            lines.append("".join(row_buf))
            row_buf = []

    lines.append("")
    if n_done == len(users) and last_done_ts:
        lines.append(f" ✅ All done! Last finished: {fmt_ts(last_done_ts)}")
    else:
        lines.append(" Press Ctrl+C to quit.")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Monitor UMRS users processing status in real time."
    )
    parser.add_argument("N", type=int, help="Number of users (user1_1 … user1_N)")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--client", default=DEFAULT_CLIENT)
    parser.add_argument("--log-file", default=None,
                        help="Append 'Last finished: <date>' to this file when all users are done")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-user raw query results and internal state each poll")
    args = parser.parse_args()

    users        = [f"user1_{i}" for i in range(1, args.N + 1)]
    state        = {u: "pending" for u in users}
    prev_done_ts = {u: None for u in users}  # None = no valid reading yet
    done         = set()
    last_done_ts = None
    start        = time.time()
    poll_count   = 0

    # Erase log file at startup if it exists
    if args.log_file and os.path.exists(args.log_file):
        os.remove(args.log_file)

    # --- Bootstrap phase: poll each user until a valid timestamp is obtained ---
    # Sequential and blocking: a user stuck on repeated query failures will
    # delay the start of monitoring for every other user.
    bootstrap_bar = None
    if not args.debug:
        bootstrap_bar = tqdm(total=len(users), desc="Initializing", unit="user")

    for user in users:
        while prev_done_ts[user] is None:
            is_running, latest_status, latest_ts = query_user(
                user, args.config_dir, args.client, debug=args.debug
            )
            if args.debug:
                print(f"[bootstrap] {user}: is_running={is_running} "
                      f"latest_status={latest_status} latest_ts={latest_ts}", flush=True)
            if is_running is True:
                state[user] = "running"
            if latest_ts is not None:
                prev_done_ts[user] = latest_ts
                if bootstrap_bar is not None:
                    bootstrap_bar.update(1)
            else:
                time.sleep(args.interval)

    if bootstrap_bar is not None:
        bootstrap_bar.close()


    if args.log_file:
        with open(args.log_file, "a"):
            os.utime(args.log_file, None)

    if not args.debug:
        # Show the initial board (poll #0) so the user sees everyone pending
        # (grey), except users already caught running during bootstrap.
        lines = render(users, state, last_done_ts, start, poll_count)
        print("\033[2J\033[H", end="", flush=True)
        out = ""
        for line in lines:
            out += line + "\033[K\n"
        print(out, end="", flush=True)
        time.sleep(1)

    try:
        while True:
            to_scan = [u for u in users if u not in done]
            cur_snapshot = {}

            for user in to_scan:
                prev_state = state[user]

                is_running, latest_status, latest_ts = query_user(
                    user, args.config_dir, args.client, debug=args.debug
                )
                cur_snapshot[user] = (is_running, latest_status, latest_ts)

                if is_running is True:
                    state[user] = "running"

                elif is_running is False:
                    went_done = state[user] == "running"

                    # Case 2: the latest entry's timestamp moved forward since
                    # last poll -> a new run completed, even if we never saw
                    # it as Running (full cycle happened between two polls).
                    if (not went_done
                            and prev_done_ts[user] is not None
                            and latest_ts is not None
                            and latest_ts > prev_done_ts[user]):
                        went_done = True

                    if went_done:
                        state[user] = "done"
                        done.add(user)
                        if latest_ts and (last_done_ts is None or latest_ts > last_done_ts):
                            last_done_ts = latest_ts

                if latest_ts is not None:
                    prev_done_ts[user] = latest_ts

                if state[user] != prev_state and not args.debug:
                    lines = render(users, state, last_done_ts, start, poll_count + 1)
                    out = "\033[H"
                    for line in lines:
                        out += line + "\033[K\n"
                    print(out, end="", flush=True)

            poll_count += 1

            if not args.debug:
                lines = render(users, state, last_done_ts, start, poll_count)
                out = "\033[H"
                for line in lines:
                    out += line + "\033[K\n"
                print(out, end="", flush=True)

            if args.debug:
                for user in to_scan:
                    ir, ls, ts = cur_snapshot[user]
                    print(f"[poll {poll_count}] {user}: state={state[user]} "
                          f"is_running={ir} latest_status={ls} "
                          f"prev_done_ts={prev_done_ts[user]} latest_ts={ts}",
                          flush=True)

            if len(done) == len(users):
                if args.log_file and last_done_ts:
                    with open(args.log_file, "a") as f:
                        f.write(f"Last finished: {fmt_ts(last_done_ts)}\n")
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nInterrupted.", flush=True)


if __name__ == "__main__":
    main()
