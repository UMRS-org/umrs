"""``umrs-server admin usage show`` (ADR-099 D-4/D-5, RS-1 lane Q).

Calls ``GET /umrs/admin/usage`` on the running UMRS with the admin JWT of
``http_admin`` (``$UMRS_ADMIN_JWT`` or ``--admin-jwt``) and prints the
remote-session token usage per subject and day, with each subject's daily
token limit from the session policy (``-`` = no limit).
"""

from __future__ import annotations

import argparse
import json
from typing import Any
from urllib.parse import urlencode

from .http_admin import _print_table, umrs_api, umrs_base_url

_USAGE_COLS = (
    "SCOPE",
    "SUBJECT",
    "DAY",
    "PROMPT",
    "COMPLETION",
    "TOTAL",
    "REQUESTS",
    "LIMIT",
)


def _usage_row(r: dict[str, Any]) -> tuple[str, ...]:
    limit = r.get("limit")
    return (
        str(r.get("scope", "")),
        str(r.get("subject", "")),
        str(r.get("day", "")),
        f"{int(r.get('prompt_tokens') or 0):,}",
        f"{int(r.get('completion_tokens') or 0):,}",
        f"{int(r.get('total_tokens') or 0):,}",
        str(int(r.get("requests") or 0)),
        "-" if limit is None else f"{int(limit):,}",
    )


def _usage_url(args: argparse.Namespace) -> str:
    query = {
        k: v
        for k, v in (
            ("user", args.user),
            ("project", args.project),
            ("venue", args.venue),
            ("days", args.days),
        )
        if v
    }
    return f"{umrs_base_url(args)}/umrs/admin/usage?{urlencode(query)}"


def _cmd_usage_show(args: argparse.Namespace, _store: Any) -> int:
    """Remote-session token usage per user (or project) and day."""
    data = umrs_api("GET", _usage_url(args), args) or {}
    if args.json:
        print(json.dumps(data, indent=2, default=str))
        return 0
    rows = data.get("rows", [])
    if not rows:
        print(f"(no inference-proxy usage at {data.get('venue') or 'this site'})")
        return 0
    _print_table(_USAGE_COLS, [_usage_row(r) for r in rows])
    if data.get("limits") == "unavailable":
        print("\nnote: the session policy was unreadable; limits not shown")
    return 0


def add_usage_parsers(usage_sub: argparse._SubParsersAction) -> None:
    show_p = usage_sub.add_parser(
        "show",
        help="Token usage of remote sessions per user (or project) and day",
        description=(
            "Remote-session token usage recorded by the inference proxy "
            "(llm_usage_log, origin=infer_proxy) at one site, per user and "
            "day, with each user's daily token limit (ADR-099). --project "
            "alone groups by project."
        ),
    )
    show_p.add_argument("--user", default="", help="Only this user")
    show_p.add_argument("--project", default="", help="Only this project")
    show_p.add_argument(
        "--venue", default="", help="Site name (default: the server's own site)"
    )
    show_p.add_argument(
        "--days", type=int, default=1, help="Calendar days back, today included (1-90)"
    )
    show_p.add_argument("--json", action="store_true")
    show_p.set_defaults(func=_cmd_usage_show)


__all__ = ["add_usage_parsers"]
