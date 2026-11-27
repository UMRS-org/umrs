"""``umrs-server admin host list|drain|undrain|remove|enroll|unenroll``
(ADR-098 D-4 SH-H1a; D-5 SH-H1b).

The hosts registry of one site (default: this deployment's), read and
written straight against the site database (``cli_db_pool``), like the
account cascade and ``site-accounts``. Placement reads the registry live
from slice SH-H2 on; until then a drain is recorded but sessions still
start on the front. A drain outlives the host going down and coming back
(it returns ``draining``); only ``undrain`` ends it. A transition that
races the janitor or another admin is refused (``hosts.conflict``).

    list      every host: state, mode, roles, last heartbeat
    drain     no new placement; the host keeps its sessions
    undrain   a draining host becomes placeable again
    remove    forget a draining or down host (an active one is refused:
              drain it first); a draining host that still runs sessions is
              refused too (they end first, or the host goes down). Its
              enrolled key goes with it: an agent host enrolls again.
    enroll    print a one-time enrollment token for an agent host (default
              valid one hour); the host enrolls its ed25519 public key with
              it. Printed once, stored only as a hash, never logged or
              audited. An enrolled host is refused unless --rekey.
    unenroll  forget a host's key and its unused tokens: its host tokens
              are refused from the next request on

An inline host removed by an admin registers itself again when its UMRS
process restarts; it never enrolls (its identity is the UMRS process).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .http_admin import _print_table

_COLS = ("HOST", "STATE", "MODE", "ROLES", "LAST HEARTBEAT", "REASON")


def _fail(code: str, message: str) -> int:
    print(f"error: {code}: {message}", file=sys.stderr)
    return 1


def _context(args: argparse.Namespace) -> tuple[Any, Any] | None:
    """(registry, pool) of the site, or None after printing why not."""
    from umrs.admin.user_store import site_venue
    from umrs.cli.admin.cascade import cli_db_pool
    from umrs.hosts.registry import HostRegistry

    pool = cli_db_pool()
    if pool is None:
        print(
            "error: hosts unavailable: no site database is configured", file=sys.stderr
        )
        return None
    venue = args.venue if args.venue is not None else site_venue()
    return HostRegistry(pool, venue=venue.strip()), pool


def _key_store(registry: Any, pool: Any) -> Any:
    from umrs.hosts.identity import HostKeyStore

    return HostKeyStore(pool, venue=registry.venue)


def _row(host: Any) -> tuple[str, ...]:
    beat = host.last_heartbeat_at
    return (
        host.host_id,
        host.state,
        host.launch_mode,
        ",".join(host.roles) or "-",
        beat.isoformat(timespec="seconds") if beat else "-",
        host.reason or "-",
    )


def _cmd_list(args: argparse.Namespace, _store: Any) -> int:
    found = _context(args)
    if found is None:
        return 2
    registry, _pool = found
    hosts = registry.hosts()
    if args.json:
        body = {"venue": registry.venue, "hosts": [h.to_dict() for h in hosts]}
        print(json.dumps(body, indent=2))
        return 0
    if not hosts:
        print(f"(no hosts at site {registry.venue!r})")
        return 0
    _print_table(_COLS, [_row(h) for h in hosts])
    return 0


def live_sessions(pool: Any, venue: str, host_id: str) -> int:
    """Live sessions the enabled extensions run on the host."""
    from umrs.hosts.janitor import extension_host_sessions

    return sum(
        len(provider.sessions_on_host(venue, host_id))
        for provider in extension_host_sessions(pool)
    )


def _transition(args: argparse.Namespace, verb: str) -> int:
    from umrs.hosts.registry import HostRegistryError

    found = _context(args)
    if found is None:
        return 2
    registry, pool = found
    try:
        host = _apply(registry, pool, args, verb)
    except HostRegistryError as exc:
        return _fail(exc.code, str(exc))
    if verb == "remove":
        _key_store(registry, pool).unenroll(host.host_id)
    _audit(verb, host, args.reason)
    print(f"host {host.host_id!r} at site {registry.venue!r}: {verb} ({host.state})")
    return 0


def _apply(registry: Any, pool: Any, args: argparse.Namespace, verb: str) -> Any:
    from umrs.hosts.registry import HostRegistryError

    if verb == "drain":
        return registry.drain(args.host_id, args.reason)
    if verb == "undrain":
        return registry.undrain(args.host_id)
    host = registry.get(args.host_id)
    if host is not None and host.state == "draining":
        busy = live_sessions(pool, registry.venue, args.host_id)
        if busy:
            raise HostRegistryError(
                "hosts.host_busy",
                f"host {args.host_id!r} still runs {busy} session(s): "
                "wait for them to end",
            )
    return registry.remove(args.host_id)


def _audit(verb: str, host: Any, reason: str) -> None:
    _audit_host(verb, host.host_id, host.venue, reason)


def _audit_host(verb: str, host_id: str, venue: str, detail: str) -> None:
    """Audit an admin host verb; ``detail`` never carries a token."""
    from umrs.cli.admin.cascade import _actor
    from umrs.venues.session_audit import audit_venue

    audit_venue(
        f"host.{verb}",
        actor=_actor(),
        session_id="",
        scope="admin",
        host_id=host_id,
        venue=venue,
        detail=(detail or "")[:200],
    )


def _cmd_enroll(args: argparse.Namespace, _store: Any) -> int:
    """Print a one-time enrollment token (stdout, once; nowhere else)."""
    from umrs.cli.admin.cascade import _actor
    from umrs.hosts.enroll import EnrollmentError, issue_enrollment

    found = _context(args)
    if found is None:
        return 2
    registry, pool = found
    try:
        issued = issue_enrollment(
            _key_store(registry, pool),
            args.host_id,
            ttl_s=args.ttl,
            created_by=_actor(),
            rekey=args.rekey,
        )
    except EnrollmentError as exc:
        return _fail(exc.code, str(exc).split(": ", 1)[-1])
    expires = issued.expires_at.isoformat(timespec="seconds")
    _audit_host("enroll", issued.host_id, issued.venue, f"expires {expires}")
    print(
        f"enrollment token for host {issued.host_id!r} at site {issued.venue!r}"
        f" (one use, expires {expires}; shown once, not stored):",
        file=sys.stderr,
    )
    print(issued.token)
    return 0


def _cmd_unenroll(args: argparse.Namespace, _store: Any) -> int:
    found = _context(args)
    if found is None:
        return 2
    registry, pool = found
    if not _key_store(registry, pool).unenroll(args.host_id):
        return _fail(
            "hosts.not_enrolled",
            f"host {args.host_id!r} has no key or token at site {registry.venue!r}",
        )
    _audit_host("unenroll", args.host_id, registry.venue, args.reason)
    print(f"host {args.host_id!r} at site {registry.venue!r}: unenrolled")
    return 0


def _cmd_drain(args: argparse.Namespace, _store: Any) -> int:
    return _transition(args, "drain")


def _cmd_undrain(args: argparse.Namespace, _store: Any) -> int:
    return _transition(args, "undrain")


def _cmd_remove(args: argparse.Namespace, _store: Any) -> int:
    return _transition(args, "remove")


def _add_venue(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--venue", default=None, help="Site name (default: this site)")


def add_hosts_parser(sub: argparse._SubParsersAction) -> None:
    """``host`` with ``list``, ``drain``, ``undrain``, ``remove``."""
    group = sub.add_parser(
        "host",
        help=(
            "List, drain, remove and enroll session/MCP hosts "
            "(ADR-098; reads the site database)"
        ),
        description=(
            "The hosts registry of one site: the hosts UMRS may place "
            "sessions on, their state (active, draining, down) and last "
            "heartbeat. A down host's sessions were ended host_lost."
        ),
    )
    actions = group.add_subparsers(dest="action", required=True)
    list_p = actions.add_parser("list", help="Every host at the site")
    list_p.add_argument("--json", action="store_true")
    list_p.set_defaults(func=_cmd_list)
    _add_venue(list_p)
    verbs = (
        ("drain", _cmd_drain, "Stop new placement on a host (keeps its sessions)"),
        ("undrain", _cmd_undrain, "Make a draining host placeable again"),
        ("remove", _cmd_remove, "Forget a draining or down host"),
    )
    for name, func, text in verbs:
        parser = actions.add_parser(name, help=text)
        parser.add_argument("host_id")
        parser.add_argument("--reason", default="", help="Recorded with the change")
        _add_venue(parser)
        parser.set_defaults(func=func)
    _add_identity_parsers(actions)


def _add_identity_parsers(actions: argparse._SubParsersAction) -> None:
    """``enroll`` and ``unenroll`` (ADR-098 D-5, SH-H1b)."""
    from umrs.hosts.enroll import DEFAULT_ENROLL_TTL_S

    enroll = actions.add_parser(
        "enroll", help="Print a one-time enrollment token for an agent host"
    )
    enroll.add_argument("host_id")
    enroll.add_argument(
        "--ttl", type=int, default=DEFAULT_ENROLL_TTL_S, help="Token lifetime (s)"
    )
    enroll.add_argument(
        "--rekey", action="store_true", help="Replace an enrolled host's key"
    )
    _add_venue(enroll)
    enroll.set_defaults(func=_cmd_enroll)
    unenroll = actions.add_parser(
        "unenroll", help="Forget a host's key and its unused enrollment tokens"
    )
    unenroll.add_argument("host_id")
    unenroll.add_argument("--reason", default="", help="Recorded with the change")
    _add_venue(unenroll)
    unenroll.set_defaults(func=_cmd_unenroll)


__all__ = ["add_hosts_parser", "live_sessions"]
