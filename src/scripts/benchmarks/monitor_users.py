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

DEFAULT_CONFIG_DIR = "~/.config/umrs/n_users/users"
DEFAULT_CLIENT     = "./client_umrs.py"
DEFAULT_INTERVAL   = 5.0
ITEM_WIDTH         = 14

ICON_PENDING = "⚪"
ICON_RUNNING = "🔴"
ICON_DONE    = "🟢"


def query_user(user: str, config_dir: str, client: str) -> tuple:
    """
    Returns (is_running, done_count, last_updated_at)
    """
    config_file = os.path.expanduser(f"{config_dir}/{user}.json")
    cmd = [client, "-c", config_file, "--output", "json", "status", "--username", user]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                summaries = data.get("summaries", {})
                entries = (summaries.get("process_objects") or
                           summaries.get("PROCESS_OBJECTS") or {}).get("data", [])

                is_running = any(e.get("status") == "Running" for e in entries)
                done_count = sum(1 for e in entries if e.get("status") == "Done")
                timestamps = [e.get("updated_at") for e in entries if e.get("updated_at")]
                last_ts = max(timestamps) if timestamps else None

                return is_running, done_count, last_ts
            except json.JSONDecodeError:
                continue
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
    args = parser.parse_args()

    users           = [f"user1_{i}" for i in range(1, args.N + 1)]
    state           = {u: "pending" for u in users}
    prev_done_count = {u: None for u in users}  # None = first poll not yet done
    done            = set()
    last_done_ts    = None
    start           = time.time()
    poll_count      = 0

    # Erase log file at startup if it exists
    if args.log_file and os.path.exists(args.log_file):
        os.remove(args.log_file)

    print("\033[2J\033[H", end="", flush=True)

    try:
        while True:
            to_scan = [u for u in users if u not in done]

            for user in to_scan:
                is_running, done_count, last_ts = query_user(
                    user, args.config_dir, args.client
                )

                if is_running is True:
                    state[user] = "running"

                elif is_running is False:
                    went_done = state[user] == "running"

                    # Case 2: done_count increased since last poll (skip first poll)
                    if (not went_done
                            and prev_done_count[user] is not None
                            and done_count is not None
                            and done_count > prev_done_count[user]):
                        went_done = True

                    if went_done:
                        state[user] = "done"
                        done.add(user)
                        if last_ts and (last_done_ts is None or last_ts > last_done_ts):
                            last_done_ts = last_ts

                if done_count is not None:
                    prev_done_count[user] = done_count

            poll_count += 1

            # Touch log file after first poll to signal script is ready
            if poll_count == 1 and args.log_file:
                with open(args.log_file, "a"):
                    os.utime(args.log_file, None)

            lines = render(users, state, last_done_ts, start, poll_count)
            out = "\033[H"
            for line in lines:
                out += line + "\033[K\n"
            print(out, end="", flush=True)

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
