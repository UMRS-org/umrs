"""Admin CLI legs of the account cascade (RS-0 S-2).

``user disable`` / ``delete`` / ``remove-from-project`` change the account
record and then end what the account still holds on this site through
``umrs.admin.account_cascade.run_cascade``, straight against the database the
server reads live. The CLI may run in a process of its own, so it never
relies on signalling the server: leases are read per request, and the
server's standing sweep re-derives the rest from the record every 30 s.
"""

from __future__ import annotations

import getpass
import sys
from typing import Any

from umrs.admin.account_cascade import (
    ACCOUNT_DISABLED,
    MEMBERSHIP_REVOKED,
    CascadeResult,
    live_project_ids,
    project_exempt,
    run_cascade,
)
from umrs.admin.user_store import UserRecord, UserStore, UserStoreError


class _ConnPerCallPool:
    """psycopg2 pool shape over one DSN: a fresh connection per lease."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def getconn(self) -> Any:
        import psycopg2

        from umrs.shared.pg_settings import pg_connect_kwargs

        return psycopg2.connect(self._dsn, **pg_connect_kwargs("admin-cli"))

    def putconn(self, conn: Any) -> None:
        conn.close()


def cli_db_pool() -> Any:
    """The site database the server reads, or None when none is configured
    (or ``UMRS_ADMIN_CLI_DB=off``: JSON files only, the pre-RS-0 behaviour)."""
    import os

    if os.environ.get("UMRS_ADMIN_CLI_DB", "").strip().lower() == "off":
        return None
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn

    dsn, _ = _resolve_knowledge_dsn()
    return _ConnPerCallPool(dsn) if dsn else None


def _actor() -> str:
    try:
        return f"admin-cli:{getpass.getuser()}"
    except Exception:  # noqa: BLE001 — no passwd entry inside some containers
        return "admin-cli"


def mirror_to_pg(pool: Any, store: UserStore, username: str) -> None:
    """Copy the record just written to the ``users`` row the server reads."""
    if pool is None:
        return
    from umrs.admin.user_store import mirror_record_to_pg, site_venue

    try:
        mirror_record_to_pg(pool, store.get(username), venue=site_venue())
    except Exception as exc:  # noqa: BLE001 — named in output; the sweep reads JSON too
        print(f"warning: users row not updated ({exc})", file=sys.stderr)


def is_last_enabled_admin(store: UserStore, username: str) -> bool:
    """True when ``username`` is the only enabled admin of this site."""
    admins = [r.username for r in store.list() if r.role == "admin" and not r.disabled]
    return admins == [username]


def cascade_account(pool: Any, username: str, *, reason: str = ACCOUNT_DISABLED) -> int:
    """The whole account is gone (disable, soft or hard delete)."""
    return _run(pool, username, None, reason)


def cascade_project_removal(
    pool: Any, store: UserStore, username: str, project: str
) -> int:
    """``username`` left ``project``: end its leases and jobs there, unless
    it may still act in it (an admin or service account per ADR-097 D-3, or
    the project's owner)."""
    if pool is None:
        return _run(None, username, project, MEMBERSHIP_REVOKED)
    try:
        record = store.get(username)
    except UserStoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    keep = _still_admitted(pool, record, project)
    if keep:
        print(f"cascade: skipped — {username} still has access to {project} ({keep})")
        return 0
    return _run(pool, username, project, MEMBERSHIP_REVOKED)


def _still_admitted(pool: Any, record: UserRecord, project: str) -> str:
    if project_exempt(record):
        return f"role {record.role}"
    conn = pool.getconn()
    try:
        return "live member" if project in live_project_ids(conn, record) else ""
    finally:
        pool.putconn(conn)


def _run(pool: Any, username: str, project: str | None, reason: str) -> int:
    if pool is None:
        print(
            "cascade: no database configured here — the server's account sweep "
            "ends this account's sessions and leases within a minute"
        )
        return 0
    from umrs.jobs.secret_broker import SecretBroker
    from umrs.jobs.store import JobStore

    try:
        result = run_cascade(
            username,
            project=project,
            reason=reason,
            actor=_actor(),
            trigger="admin-cli",
            broker=SecretBroker(db_pool=pool),
            job_store=JobStore(pool),
            audit_pool=pool,
        )
    except Exception as exc:  # noqa: BLE001 — the record change stands; say what did not
        print(
            f"error: cascade failed ({exc}); the record change stands and the "
            "server's account sweep ends the sessions within a minute",
            file=sys.stderr,
        )
        return 1
    _print_result(result)
    return 0


def _print_result(result: CascadeResult) -> None:
    scope = f" in project {result.project}" if result.project else ""
    print(
        f"cascade{scope}: revoked {result.leases_revoked} lease(s); "
        f"{result.sessions_ended} remote session(s) end at their next status "
        f"check; cancelled {result.jobs_cancelled} job(s)"
    )
    if not result.project:
        print(
            "cascade: CURVE/ZMQ sessions end at the server's next keystore sweep (60 s)"
        )
    if not result.audited:
        print("warning: the audit_events row was not written", file=sys.stderr)


__all__ = [
    "cascade_account",
    "cascade_project_removal",
    "cli_db_pool",
    "is_last_enabled_admin",
    "mirror_to_pg",
]
