#!/usr/bin/env python3
"""
Check if a UMRS user is still processing.
Usage: ./check_user_status.py <user> [--config-dir DIR]
Exit code: 0 = idle, 1 = still running
"""

import argparse
import json
import subprocess
import sys

CLIENT = "./client_umrs.py"
DEFAULT_CONFIG_DIR = "~/.config/umrs/n_users/users"


def parse_json_from_output(output: str) -> dict:
    """Extract the JSON line from mixed output (ignores non-JSON lines)."""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise ValueError("No valid JSON found in output")


def is_user_running(user: str, config_dir: str) -> bool:
    config_file = f"{config_dir}/{user}.json"
    cmd = [
        CLIENT,
        "-c", config_file,
        "--output", "json",
        "status",
        "--username", user,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        data = parse_json_from_output(result.stdout)
    except ValueError as e:
        print(f"[{user}] ERROR: {e}", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        return False

    summaries = data.get("summaries", {})
    # Check both possible keys
    process_objects = summaries.get("process_objects") or summaries.get("PROCESS_OBJECTS") or {}
    entries = process_objects.get("data", [])

    # A user is running if ANY thread has status "Running"
    return any(entry.get("status") == "Running" for entry in entries)


def main():
    parser = argparse.ArgumentParser(
        description="Check if a UMRS user is still processing."
    )
    parser.add_argument("user", help="Username to check (e.g. user1_1)")
    parser.add_argument(
        "--config-dir",
        default=DEFAULT_CONFIG_DIR,
        help=f"Directory containing user JSON configs (default: {DEFAULT_CONFIG_DIR})",
    )
    args = parser.parse_args()

    running = is_user_running(args.user, args.config_dir)
    if running:
        print(f"[{args.user}] RUNNING")
        sys.exit(1)
    else:
        print(f"[{args.user}] IDLE")
        sys.exit(0)


if __name__ == "__main__":
    main()
