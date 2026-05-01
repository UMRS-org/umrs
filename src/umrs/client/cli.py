"""Canonical client CLI entrypoint for UMRS."""

from __future__ import annotations

import json
import sys


ADMIN_DENIED_MESSAGE = (
    "umrs-client admin is disabled. Use 'umrs-server admin' for offline "
    "operator tasks."
)
HELP_CROSS_REFERENCE = (
    "CLI map:\n"
    "  Use r4 for Gateway-mediated research workflows.\n"
    "  Use umrs-client only for UMRS diagnostics/migration helpers.\n"
    "  Use umrs-server admin for offline operator administration.\n"
)

_OPTION_WITH_VALUE = {
    "-c",
    "--conf",
    "--server-conf",
    "--s3cfg",
    "--profile",
    "-k",
    "--server-ip",
    "--control-port",
    "--output",
    "--admin-token",
}


def _requested_action(argv: list[str]) -> str | None:
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--":
            return None
        if arg.startswith("--") and "=" in arg:
            index += 1
            continue
        if arg in _OPTION_WITH_VALUE:
            index += 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        return arg
    return None


def _wants_json(argv: list[str]) -> bool:
    for index, arg in enumerate(argv):
        if arg == "--output" and index + 1 < len(argv):
            return argv[index + 1] == "json"
        if arg == "--output=json":
            return True
    return False


def _requested_help(argv: list[str]) -> bool:
    return any(arg in {"-h", "--help"} for arg in argv[1:])


def _deny_legacy_admin(argv: list[str]) -> int:
    if _wants_json(argv):
        print(json.dumps({"status": "error", "message": ADMIN_DENIED_MESSAGE}))
    else:
        print(f"Error: {ADMIN_DENIED_MESSAGE}", file=sys.stderr)
    return 2


def main():
    if _requested_help(sys.argv):
        print(HELP_CROSS_REFERENCE)

    if _requested_action(sys.argv) == "admin":
        return _deny_legacy_admin(sys.argv)

    from client_umrs import main as legacy_main

    return legacy_main()