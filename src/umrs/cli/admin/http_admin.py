"""HTTP admin commands: the credential they send and ``admin session``.

``umrs-server admin endpoint …`` and ``umrs-server admin session …`` call
the running UMRS API (``--umrs-url`` / ``$UMRS_BASE_URL``). Until RS-0 they
sent no credential, so in strict mode every admin read was refused (or, for
the endpoint registry, scoped to nobody). They now send an admin JWT as
``Authorization: Bearer``: the credential the endpoint registry's admin
routes (``endpoints._caller_is_admin``) and the admin session routes
(``api/admin_sessions.py``) verify against the server's JWT secret.

The JWT comes from ``--admin-jwt`` or ``$UMRS_ADMIN_JWT`` (prefer the env:
argv is visible in ``ps``). Mint one where UMRS runs:

    umrs-server admin token generate-jwt --user ADMIN --project PROJECT \\
        --roles admin --days 1

No credential, or an opaque ``token generate`` token (that one is the ZMQ
control plane's), is refused before any request is sent. The token is never
printed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, NoReturn
from urllib.parse import quote, urlencode

ADMIN_JWT_ENV = "UMRS_ADMIN_JWT"
_MINT_HINT = (
    "mint one where UMRS runs: umrs-server admin token generate-jwt "
    "--user ADMIN --project PROJECT --roles admin --days 1"
)


def _fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def admin_jwt(args: argparse.Namespace) -> str:
    """The admin JWT to send, or exit 2 with a named reason."""
    token = (
        getattr(args, "admin_jwt", None) or os.environ.get(ADMIN_JWT_ENV, "")
    ).strip()
    if not token:
        _fail(
            f"no admin credential: set ${ADMIN_JWT_ENV} (or pass --admin-jwt) "
            f"to an admin JWT; {_MINT_HINT}"
        )
    if token.count(".") != 2:
        _fail(
            f"${ADMIN_JWT_ENV} is not a JWT (an opaque `token generate` token "
            f"is for the ZMQ control plane, not the HTTP API); {_MINT_HINT}"
        )
    return token


def umrs_base_url(args: argparse.Namespace) -> str:
    """The UMRS HTTP base URL (``--umrs-url``, ``$UMRS_BASE_URL``, localhost)."""
    url = getattr(args, "umrs_url", None) or os.environ.get(
        "UMRS_BASE_URL", "http://localhost:8000"
    )
    return url.rstrip("/")


def umrs_api(
    method: str, url: str, args: argparse.Namespace, data: dict | None = None
) -> Any:
    """One authenticated call to the UMRS API; parsed JSON (None when empty)."""
    import urllib.error
    import urllib.request

    token = admin_jwt(args)  # refuse before anything leaves the host
    payload = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Accept", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode() if exc.fp else ""
        _fail(f"UMRS API returned {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        _fail(
            f"cannot reach UMRS at {url}: {exc.reason}\n"
            "hint: is the UMRS server running? Set --umrs-url or $UMRS_BASE_URL."
        )
    return json.loads(body) if body.strip() else None


def add_http_admin_args(parser: argparse.ArgumentParser) -> None:
    """``--umrs-url`` and ``--admin-jwt`` for a group of HTTP admin commands."""
    parser.add_argument(
        "--umrs-url",
        dest="umrs_url",
        default=None,
        help="UMRS server URL (default: $UMRS_BASE_URL or http://localhost:8000)",
    )
    parser.add_argument(
        "--admin-jwt",
        dest="admin_jwt",
        default=None,
        help=f"Admin JWT to authenticate with (default: ${ADMIN_JWT_ENV}; prefer the env)",
    )


# ── admin session (C-1, remote-session plan RS-0) ──────────────────────────

_SESSION_COLS = ("SESSION", "USER", "PROJECT", "STATE", "HOST", "STARTED", "REASON")


def _session_row(s: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(s.get("session_id", "")),
        str(s.get("user_id", "")),
        str(s.get("project", "") or "-"),
        str(s.get("state", "")),
        str(s.get("host", "")),
        str(s.get("started_at", "") or "-")[:19],
        str(s.get("reason", "") or "")[:40],
    )


def _print_table(cols: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [max([len(c)] + [len(r[i]) for r in rows]) for i, c in enumerate(cols)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def _cmd_session_list(args: argparse.Namespace, _store: Any) -> int:
    """Every user's remote sessions (admin view)."""
    query = {
        k: v
        for k, v in (
            ("user", args.user),
            ("project", args.project),
            ("state", args.state),
            ("host", args.host),
            ("limit", args.limit),
        )
        if v
    }
    url = f"{umrs_base_url(args)}/umrs/admin/sessions"
    data = umrs_api("GET", f"{url}?{urlencode(query)}" if query else url, args) or {}
    sessions = data.get("sessions", [])
    if args.json:
        print(json.dumps(sessions, indent=2, default=str))
        return 0
    if not sessions:
        print("(no sessions)")
        return 0
    _print_table(_SESSION_COLS, [_session_row(s) for s in sessions])
    print(f"\nTotal: {len(sessions)} session(s)")
    return 0


def _session_url(args: argparse.Namespace, suffix: str = "") -> str:
    sid = quote(args.session_id, safe="")
    return f"{umrs_base_url(args)}/umrs/admin/sessions/{sid}{suffix}"


def _cmd_session_show(args: argparse.Namespace, _store: Any) -> int:
    """One session of any owner (job, lease, owner, project)."""
    print(json.dumps(umrs_api("GET", _session_url(args), args), indent=2, default=str))
    return 0


def _cmd_session_stop(args: argparse.Namespace, _store: Any) -> int:
    """Stop any user's session (teardown, lease revoked, audited)."""
    from .main import _confirm

    if not _confirm(f"Stop session {args.session_id}?", args.yes):
        print("aborted", file=sys.stderr)
        return 1
    result = (
        umrs_api("POST", _session_url(args, "/stop"), args, {"note": args.note}) or {}
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"session {args.session_id}: {result.get('state', 'unknown')}")
    return 0


def add_session_parsers(session_sub: argparse._SubParsersAction) -> None:
    list_p = session_sub.add_parser("list", help="List every user's remote sessions")
    list_p.add_argument("--user", default="", help="Only this owner")
    list_p.add_argument("--project", default="", help="Only this project")
    list_p.add_argument(
        "--state",
        default="",
        choices=["", "provisioning", "running", "paused", "failed", "stopped"],
        help="Only this state",
    )
    list_p.add_argument(
        "--host", default="", help="Only this session host (today: local)"
    )
    list_p.add_argument(
        "--limit", type=int, default=100, help="At most N rows (1-1000)"
    )
    list_p.add_argument("--json", action="store_true")
    list_p.set_defaults(func=_cmd_session_list)

    show_p = session_sub.add_parser("show", help="Show one session of any owner")
    show_p.add_argument("session_id")
    show_p.set_defaults(func=_cmd_session_show)

    stop_p = session_sub.add_parser(
        "stop",
        help="Stop any user's session (teardown + lease revoke)",
        description=(
            "Stop a remote session whoever owns it: the sandbox is torn down, "
            "its lease revoked and its job settled. Audited as "
            "venue.session.stop with cause=admin."
        ),
    )
    stop_p.add_argument("session_id")
    stop_p.add_argument("--note", default="", help="Why (kept in the session's reason)")
    stop_p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    stop_p.add_argument("--json", action="store_true")
    stop_p.set_defaults(func=_cmd_session_stop)
