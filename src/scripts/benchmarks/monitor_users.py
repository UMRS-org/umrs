#!/usr/bin/env python3
"""
Monitor UMRS users processing status.
Usage: ./monitor_users.py <N> [--config-dir DIR] [--interval SEC] [--client PATH]

Each poll cycle issues a single STATUS_ALL request to the server instead of
N per-user requests, reducing monitoring overhead from O(N) to O(1) round-trips.
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
ITEM_WIDTH         = 17

ICON_PENDING = "⚪"
ICON_RUNNING = "🔴"
ICON_DONE    = "🟢"

# Sentinel returned by query_user when the server has no record of the client.
NOT_FOUND = "NOT_FOUND"


def _count_done(entries: list) -> int:
    """Done entries excluding the server-startup snapshot row (round-0 marker)."""
    return sum(
        1
        for e in entries
        if e.get("status") == "Done" and e.get("thread_id") != "snapshot"
    )


def _resolve_latest(dated: list, process_objects: dict, n_done: int) -> tuple:
    """Latest-entry verdict from dated rows (or the Idle/generated_at fallback)."""
    if not dated:
        generated_at = process_objects.get("generated_at")
        if generated_at:
            return False, "Idle", generated_at, n_done
        return None, None, None, 0
    latest = max(dated, key=lambda e: e["updated_at"])
    latest_status = latest.get("status")
    latest_ts = latest["updated_at"]
    return latest_status == "Running", latest_status, latest_ts, n_done


def _parse_client_entry(entry: "dict | None") -> tuple:
    """Parse one client entry from a STATUS or STATUS_ALL response.

    Returns ``(is_running, latest_status, latest_ts, n_done)``:
    - NOT_FOUND  when entry is None (server has no record of the client)
    - True/False when a run is known
    - None       on a transient / incomplete response

    ``n_done`` is the total count of Done entries in ``data`` — used to
    detect completed rounds regardless of polling interval.
    """
    if entry is None:
        return NOT_FOUND, None, None, 0

    summaries = entry.get("summaries", {})
    po = summaries.get("process_objects") or summaries.get("PROCESS_OBJECTS") or {}
    entries = po.get("data", [])
    n_done = _count_done(entries)
    dated = [e for e in entries if e.get("updated_at")]
    return _resolve_latest(dated, po, n_done)


def query_user(user: str, config_dir: str, client: str, debug: bool = False) -> tuple:
    """Query the server for *user*'s processing status (single-user fallback).

    Prefer ``query_all_users_once`` for bulk monitoring; this function is kept
    for diagnostics or scenarios where only one user is needed.
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

                if data.get("status") == "error":
                    return NOT_FOUND, None, None, 0

                return _parse_client_entry(data)
            except json.JSONDecodeError:
                continue

    if debug:
        print(f"----- no parsable JSON for {user} -----", flush=True)
        print(result.stdout, flush=True)
        print("-" * 42, flush=True)

    return None, None, None, 0


def _parse_status_all_line(line: str, users: list, debug: bool) -> "dict | None":
    """One STATUS_ALL stdout line → per-user verdicts, or None to keep scanning."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if debug:
        print("----- STATUS_ALL raw JSON -----", flush=True)
        print(json.dumps(data, indent=2, ensure_ascii=False), flush=True)
        print("-" * 42, flush=True)
    if data.get("status") != "ok":
        return {u: (None, None, None, 0) for u in users}
    clients = data.get("clients", {})
    return {u: _parse_client_entry(clients.get(u)) for u in users}


def query_all_users_once(
    users: list,
    config_dir: str,
    client: str,
    debug: bool = False,
) -> dict:
    """One STATUS_ALL round-trip → status for every user in *users*.

    Returns ``dict[user -> (is_running, latest_status, latest_ts)]``.
    A user absent from the server response maps to ``(NOT_FOUND, None, None)``.
    A transient error returns ``(None, None, None)`` for every user.
    """
    config_file = os.path.expanduser(f"{config_dir}/{users[0]}.json")
    cmd = [client, "-c", config_file, "--output", "json", "status-all"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for line in result.stdout.splitlines():
        parsed = _parse_status_all_line(line, users, debug)
        if parsed is not None:
            return parsed

    if debug:
        print("----- no parsable STATUS_ALL JSON -----", flush=True)
        print(result.stdout, flush=True)
        print("-" * 42, flush=True)

    return {u: (None, None, None, 0) for u in users}


def fmt_ts(iso_ts):
    if not iso_ts:
        return ""
    try:
        dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except Exception:
        return iso_ts


def _state_counts(users, state):
    """(done, running, pending) tallies."""
    n_done = sum(1 for u in users if state[u] == "done")
    n_running = sum(1 for u in users if state[u] == "running")
    n_pending = sum(1 for u in users if state[u] == "pending")
    return n_done, n_running, n_pending


def _poll_text(poll_count: int) -> str:
    return f"Poll #{poll_count}" if poll_count > 0 else "Poll #0 (first scan in progress...)"


def _rounds_text(users, completions, rounds: int) -> str:
    if rounds <= 1:
        return ""
    total_completions = sum(completions[u] for u in users)
    return f"  🔵 {total_completions}/{len(users) * rounds} tasks"


def _render_header(users, state, completions, rounds, start_time, poll_count, pid, cols):
    """Header + rule lines for the monitor board."""
    elapsed = int(time.time() - start_time)
    h, m, sec = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    n_done, n_running, n_pending = _state_counts(users, state)
    pid_str = f"PID {pid}  " if pid is not None else ""
    header = (f"{pid_str}"
              f" Done: {n_done}/{len(users)}  "
              f"🔴 Running: {n_running}  "
              f"⚪ Pending: {n_pending}"
              f"{_rounds_text(users, completions, rounds)}  "
              f"⏱ {h:02d}:{m:02d}:{sec:02d}  "
              f"🔄 {_poll_text(poll_count)}")
    return [header, "─" * min(len(header), cols)]


def _render_user_cells(users, state, items_per_row):
    """Icon cells, wrapped to the terminal width."""
    icon_map = {"pending": ICON_PENDING, "running": ICON_RUNNING, "done": ICON_DONE}
    lines = []
    row_buf = []
    for i, u in enumerate(users):
        cell = f"{u} {icon_map[state[u]]}".ljust(ITEM_WIDTH)
        row_buf.append(cell)
        if len(row_buf) == items_per_row or i == len(users) - 1:
            lines.append("".join(row_buf))
            row_buf = []
    return lines


def render(users, state, completions, rounds, last_done_ts, start_time, poll_count, pid=None):
    cols, _ = shutil.get_terminal_size(fallback=(80, 24))
    items_per_row = max(1, cols // ITEM_WIDTH)

    lines = _render_header(users, state, completions, rounds, start_time, poll_count, pid, cols)
    lines += _render_user_cells(users, state, items_per_row)

    n_done = sum(1 for u in users if state[u] == "done")
    lines.append("")
    if n_done == len(users) and last_done_ts:
        lines.append(f" ✅ All done! Last finished: {fmt_ts(last_done_ts)}")
    else:
        lines.append(" Press Ctrl+C to quit.")
    return lines


def _parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(
        description=(
            "Monitor UMRS users processing status in real time. "
            "Uses a single STATUS_ALL request per poll cycle (O(1) round-trips "
            "regardless of N)."
        )
    )
    parser.add_argument("N", type=int, help="Number of users (user1_1 … user1_N)")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--client", default=DEFAULT_CLIENT)
    parser.add_argument("--log-file", default=None,
                        help="Append 'Last finished: <date>' to this file when all users are done")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of sequential upload rounds per user to wait for (default: 1)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-user raw query results and internal state each poll")
    return parser.parse_args()


def _resolve_bootstrap_user(snapshot_entry, user, state, base_done, bootstrap_bar, debug):
    """One user's bootstrap verdict; advances the bar when resolvable."""
    is_running, latest_status, latest_ts, n_done = snapshot_entry
    if debug:
        print(f"[bootstrap] {user}: is_running={is_running} "
              f"latest_status={latest_status} latest_ts={latest_ts} "
              f"n_done={n_done}", flush=True)
    if is_running is NOT_FOUND:
        if bootstrap_bar is not None:
            bootstrap_bar.close()
        print(
            f"\n[Error] {user}: client unknown to server — aborting monitor.",
            flush=True,
        )
        raise SystemExit(1)
    if is_running is True:
        state[user] = "running"
    if latest_ts is None:
        return False
    # Record baseline: pre-existing Done entries don't count as completed
    # rounds for this benchmark run.
    base_done[user] = n_done
    if bootstrap_bar is not None:
        bootstrap_bar.update(1)
    return True


def _bootstrap(users, args, state, base_done):
    """One STATUS_ALL per retry until every user has a timestamp."""
    bootstrap_bar = None if args.debug else tqdm(
        total=len(users), desc="Initializing", unit="user"
    )
    remaining = set(users)
    while remaining:
        snapshot = query_all_users_once(
            users, args.config_dir, args.client, debug=args.debug
        )
        for user in list(remaining):
            resolved = _resolve_bootstrap_user(
                snapshot.get(user, (None, None, None, 0)),
                user, state, base_done, bootstrap_bar, args.debug,
            )
            if resolved:
                remaining.discard(user)
        if remaining:
            if args.debug:
                print(
                    f"[bootstrap] {len(remaining)} user(s) still without timestamp, retrying…",
                    flush=True,
                )
            time.sleep(args.interval)
    if bootstrap_bar is not None:
        bootstrap_bar.close()


def _render_board(users, state, completions, rounds, last_done_ts, start, poll_count, *, clear):
    out = "\033[2J\033[H" if clear else "\033[H"
    lines = render(users, state, completions, rounds, last_done_ts, start, poll_count, pid=os.getpid())
    for line in lines:
        out += line + "\033[K\n"
    print(out, end="", flush=True)


def _apply_new_completions(user, is_running, latest_ts, state, completions, rounds,
                           done, last_done_ts, new_completions):
    """Advance one user's completion counter; returns the winning latest ts."""
    completions[user] = new_completions
    if latest_ts and (last_done_ts is None or latest_ts > last_done_ts):
        last_done_ts = latest_ts
    if completions[user] >= rounds:
        state[user] = "done"
        done.add(user)
    else:
        # More rounds remain; set pending unless already running
        state[user] = "running" if is_running is True else "pending"
    return last_done_ts


def _update_user_state(user, entry, state, completions, base_done, rounds, done, last_done_ts):
    """One bulk-scan entry → state/completions transition; returns the winning ts."""
    is_running, latest_status, latest_ts, n_done = entry
    prev_state = state[user]
    new_completions = max(0, n_done - base_done[user])
    if new_completions > completions[user]:
        last_done_ts = _apply_new_completions(
            user, is_running, latest_ts, state, completions, rounds, done,
            last_done_ts, new_completions,
        )
    elif is_running is True:
        state[user] = "running"
    elif is_running is False and prev_state == "running":
        state[user] = "pending"
    # is_running is None → transient error, keep current state
    return last_done_ts, prev_state


def _append_finish_log(log_file, last_done_ts):
    client_start_ts = None
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                if line.startswith("Started:"):
                    client_start_ts = line.split(":", 1)[1].strip()
                    break
    with open(log_file, "a") as f:
        f.write(f"Last finished: {fmt_ts(last_done_ts)}\n")
        if not client_start_ts:
            return
    _print_duration_line(log_file, client_start_ts, last_done_ts)


def _print_duration_line(log_file, client_start_ts, last_done_ts):
    try:
        t0 = datetime.fromisoformat(client_start_ts)
        t1 = datetime.fromisoformat(fmt_ts(last_done_ts))
        elapsed = (t1 - t0).total_seconds()
        h = int(elapsed) // 3600
        m = (int(elapsed) % 3600) // 60
        sec = int(elapsed) % 60
        ms = int((elapsed % 1) * 1000)
        duration_str = f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"
        with open(log_file, "a") as f:
            f.write(f"Duration: {duration_str}\n")
        print(f"\n Total duration: {duration_str}\033[K", flush=True)
    except Exception:
        pass


def _debug_dump(to_scan, cur_snapshot, state, completions, base_done, rounds, poll_count):
    for user in to_scan:
        ir, ls, ts, nd = cur_snapshot[user]
        print(f"[poll {poll_count}] {user}: state={state[user]} "
              f"completions={completions[user]}/{rounds} "
              f"is_running={ir} latest_status={ls} "
              f"n_done={nd} base_done={base_done[user]} latest_ts={ts}",
              flush=True)


def _poll_once(users, args, state, completions, base_done, done, last_done_ts, poll_count, start_time):
    """One poll cycle; returns the (possibly updated) last_done_ts."""
    to_scan = [u for u in users if u not in done]
    # One STATUS_ALL round-trip covers all pending users.
    bulk = query_all_users_once(
        to_scan, args.config_dir, args.client, debug=args.debug
    )
    cur_snapshot = {}

    poll_count += 1
    for user in to_scan:
        entry = bulk.get(user, (None, None, None, 0))
        cur_snapshot[user] = entry
        last_done_ts, prev_state = _update_user_state(
            user, entry, state, completions, base_done, args.rounds, done, last_done_ts
        )
        if state[user] != prev_state and not args.debug:
            _render_board(users, state, completions, args.rounds,
                          last_done_ts, start_time, poll_count, clear=False)

    if not args.debug:
        _render_board(users, state, completions, args.rounds,
                      last_done_ts, start_time, poll_count, clear=False)

    if args.debug:
        _debug_dump(to_scan, cur_snapshot, state, completions, base_done, args.rounds, poll_count)
    return last_done_ts, poll_count


def _finish_when_all_done(done, users, args, last_done_ts) -> bool:
    """Terminal condition: every user completed their rounds (log appended once)."""
    if len(done) != len(users):
        return False
    if args.log_file and last_done_ts:
        _append_finish_log(args.log_file, last_done_ts)
    return True


def _erase_log_on_startup(log_file: "str | None") -> None:
    if log_file and os.path.exists(log_file):
        os.remove(log_file)


def _touch_log(log_file: "str | None") -> None:
    if log_file:
        with open(log_file, "a"):
            os.utime(log_file, None)


def main():
    args = _parse_args()

    users = [f"user1_{i}" for i in range(1, args.N + 1)]
    state = {u: "pending" for u in users}
    base_done = {u: 0 for u in users}
    completions = {u: 0 for u in users}
    done = set()
    last_done_ts = None
    start_time = time.time()
    poll_count = 0

    _erase_log_on_startup(args.log_file)

    _bootstrap(users, args, state, base_done)

    _touch_log(args.log_file)

    if not args.debug:
        # Show the initial board (poll #0) so the user sees everyone pending
        # (grey), except users already caught running during bootstrap.
        _render_board(users, state, completions, args.rounds,
                      last_done_ts, start_time, poll_count, clear=True)
        time.sleep(1)

    try:
        while True:
            last_done_ts, poll_count = _poll_once(
                users, args, state, completions, base_done, done, last_done_ts, poll_count,
                start_time,
            )

            if _finish_when_all_done(done, users, args, last_done_ts):
                break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nInterrupted.", flush=True)


if __name__ == "__main__":
    main()
