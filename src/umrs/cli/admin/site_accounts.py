"""``umrs-server admin site-accounts list|show`` (ADR-112 wave B1).

Reads the ``site_accounts`` / ``project_site_allocations`` rows of one venue
(default: this deployment's site) straight from the site database
(``cli_db_pool``), like the account cascade. Refuses with a clear message when
no database is configured. Prints the public account view only: key ids and
public fingerprints, never a key (the private halves are files the database
does not hold).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .http_admin import _print_table

_COLS = (
    "USER",
    "LOGIN",
    "STATUS",
    "PROBE",
    "GATE",
    "KEY",
    "ALLOCATIONS",
    "LINKS",
    "VERIFIED",
)


def _store(args: argparse.Namespace) -> Any:
    """This venue's store, or None after printing why there is none."""
    from umrs.admin.user_store import site_venue
    from umrs.cli.admin.cascade import cli_db_pool
    from umrs.hpc_accounts.store import SiteAccountStore

    venue = (args.venue or site_venue()).strip()
    pool = cli_db_pool()
    if pool is None or not venue:
        why = (
            "no site database is configured"
            if pool is None
            else "no venue (pass --venue)"
        )
        print(f"error: site accounts unavailable: {why}", file=sys.stderr)
        return None
    return SiteAccountStore(pool, venue)


def _row(account: dict[str, Any], links: int) -> tuple[str, ...]:
    allocations = sorted(
        {str(a.get("account", "")) for a in account["verified_assocs"]}
    )
    return (
        account["username"],
        account["site_login"] or "-",
        account["status"],
        account["gate_probe"],
        account["gate_version"] or "-",
        account["transport_key_id"] or "-",
        ",".join(allocations) or "-",
        str(links),
        str(account["verified_at"] or "-"),
    )


def _cmd_list(args: argparse.Namespace, _store_unused: Any) -> int:
    store = _store(args)
    if store is None:
        return 2
    accounts, links = store.list_accounts(), store.list_links()
    if args.json:
        print(
            json.dumps(
                {"venue": store.venue, "accounts": accounts, "links": links}, indent=2
            )
        )
        return 0
    if not accounts:
        print(f"(no HPC site accounts at {store.venue})")
        return 0
    counts = {
        a["username"]: sum(x["username"] == a["username"] for x in links)
        for a in accounts
    }
    _print_table(_COLS, [_row(a, counts[a["username"]]) for a in accounts])
    return 0


def _cmd_show(args: argparse.Namespace, _store_unused: Any) -> int:
    store = _store(args)
    if store is None:
        return 2
    account = store.get_account(args.username)
    if account is None:
        print(
            f"error: no site account for {args.username} at {store.venue}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {"account": account, "links": store.list_links(args.username)}, indent=2
        )
    )
    return 0


def add_site_accounts_parser(sub: argparse._SubParsersAction) -> None:
    """``site-accounts`` with ``list`` and ``show`` (called by build_parser)."""
    group = sub.add_parser(
        "site-accounts",
        help="List users' own HPC site accounts at a venue (ADR-112; reads the site database)",
        description=(
            "User-registered HPC accounts (site_accounts) and their project "
            "links (project_site_allocations) at one venue, read from the site "
            "database. Shows key ids and public fingerprints, never a key."
        ),
    )
    actions = group.add_subparsers(dest="action", required=True)
    list_p = actions.add_parser("list", help="Every site account at the venue")
    show_p = actions.add_parser("show", help="One user's account and links")
    show_p.add_argument("username")
    for parser, func in ((list_p, _cmd_list), (show_p, _cmd_show)):
        parser.add_argument(
            "--venue", default="", help="Site name (default: this site)"
        )
        parser.add_argument("--json", action="store_true")
        parser.set_defaults(func=func)


__all__ = ["add_site_accounts_parser"]
