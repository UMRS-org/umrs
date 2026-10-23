"""Account and project cascade (RS-0 S-2, remote-session platform plan).

Disabling an account, removing it from a project, and (ADR-097 ``enforce``)
losing a project membership all end what the account still runs on this
site, as ONE audited event (``audit_events.event = 'account.cascade'``):

- **leases** — ``SecretBroker.revoke_by_owner`` (scoped to the project for a
  removal). The inference proxy and the rail read leases live per request,
  so they refuse at once.
- **remote sessions** (``frontend_session`` jobs) — end through the lease:
  the session's status checker sees the revoked lease on its next cycle
  (5 s) and runs the venue backend's teardown. The cascade never cancels a
  session job itself: that flips the durable record without the teardown,
  which only the server process (the podman socket's owner) can run.
- **other jobs** — a durable cancel of every planned/queued/running job the
  account owns on this site (the project's only, for a removal); in the
  server process the in-process cancel bus is told too. A running Slurm job
  is not killed (the platform's cancel never does that).
- **CURVE/ZMQ sessions** — ``revoke_user`` on the server's provider when the
  whole account is lost. The admin CLI runs in another process; there the
  provider's own keystore sweep ends a disabled account's sessions (D-03,
  every 60 s). A project removal leaves them: the ZMQ project check (S-1)
  reads live memberships per command.

Where it runs: the admin CLI runs the cascade against the database at once;
``AccountStandingSweep`` runs inside the server every 30 s (at most 60 s) and
re-derives everything from live state — the account record and the live
leases — so nothing depends on signalling between processes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

CASCADE_EVENT = "account.cascade"
ACCOUNT_DISABLED = "account_disabled"
MEMBERSHIP_REVOKED = "project_membership_revoked"
# ADR-097 D-3 admits admins to any project; service accounts are platform
# identities. Neither loses a project by not being listed in it.
_PROJECT_EXEMPT_ROLES = frozenset({"admin", "service"})
_SWEEP_DEFAULT_SEC = 30.0
_SWEEP_MIN_SEC = 5.0
_SWEEP_MAX_SEC = 60.0  # the plan's exit criterion: sessions end within a minute


@dataclass
class CascadeResult:
    """What one cascade ended; the audit event's details."""

    user: str
    project: str  # "" = the whole account
    reason: str
    trigger: str  # admin-cli | sweep
    venue: str = ""
    leases_revoked: int = 0
    sessions_ended: int = 0  # sessions whose lease was revoked
    jobs_cancelled: int = 0
    curve_sessions_revoked: int = 0
    audited: bool = False

    def details(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("audited")
        return data


# ── standing: does the account still have access? ────────────────────────


def membership_enforced() -> bool:
    """ADR-097 D-4 on the UMRS side: ``UMRS_PROJECT_MEMBERSHIP=enforce``
    (set it with the gateway's ``GATEWAY_PROJECT_MEMBERSHIP``). Below
    ``enforce`` the gateway still admits a token's own project without a
    live membership, so the sweep must not end its sessions either."""
    raw = os.environ.get("UMRS_PROJECT_MEMBERSHIP", "token")
    return raw.strip().lower() == "enforce"


def live_project_ids(conn: Any, record: Any) -> set[str]:
    """ADR-097 D-1 memberships (``umrs.api.projects.live_memberships``,
    restated below the api layer): the record's projects minus retired rows,
    plus the active projects the user owns."""
    listed = [p.strip() for p in record.projects if isinstance(p, str) and p.strip()]
    cur = conn.cursor()
    cur.execute(
        "SELECT project_id, owner_id, status FROM projects"
        " WHERE owner_id = %s OR project_id = ANY(%s)",
        (record.username, listed),
    )
    rows = cur.fetchall()
    retired = {r[0] for r in rows if r[2] == "retired"}
    owned = {r[0] for r in rows if r[1] == record.username and r[2] != "retired"}
    return (set(listed) - retired) | owned


def project_exempt(record: Any) -> bool:
    return str(getattr(record, "role", "") or "") in _PROJECT_EXEMPT_ROLES


def standing_loss(
    record: Any, project: str, *, member_of: Callable[[], set[str]], enforce: bool
) -> str | None:
    """Why ``record`` may no longer act in ``project``; None when it may.

    No record → None: the sweep acts on evidence only, never on an absent
    or unreadable account (a lookup error must not end every session).
    """
    if record is None:
        return None
    if record.disabled:
        return ACCOUNT_DISABLED
    if not enforce or not project or project_exempt(record):
        return None
    return None if project in member_of() else MEMBERSHIP_REVOKED


def read_record(username: str) -> Any:
    """The live account record (the ``users`` row first, then JSON); None
    when there is no such account. Raises when a lane cannot be read, so a
    caller can tell "unknown" from "unreadable". A disable in either lane
    wins: the JSON file is the admin CLI's first write, the row its mirror."""
    from .user_store import UserStore, UserStoreError

    store = UserStore()
    try:
        record = store.get(username)
    except UserStoreError:
        return None
    if not record.disabled and store.json_disabled(username):
        record.disabled = True
    return record


def load_record(username: str) -> Any:
    """``read_record``, with an unreadable record read as None."""
    from .user_store import UserStoreError

    try:
        return read_record(username)
    except (UserStoreError, OSError, ValueError):
        return None


def account_disabled(username: str) -> bool:
    """True only on evidence that ``username``'s account is disabled.

    The admin-credential check (2026-10-23, ADR-097 Amendment D-8): an admin
    JWT outlived its owner's disable. An absent or unreadable record is not
    evidence, so a user-store outage never locks the operator out of the
    admin routes (the sweep's rule: act on evidence only).
    """
    if not (username or "").strip():
        return False
    try:
        record = load_record(username.strip())
    except Exception:  # e.g. a Postgres error: no evidence
        logger.warning("standing of %s unreadable", username, exc_info=True)
        return False
    return bool(record is not None and record.disabled)


def refusal_for(owner: str, project: str, *, pool: Any) -> str | None:
    """Standing check for one (owner, project) — the session provisioning
    guard. Fails open on lookup errors, like the sweep."""
    try:
        record = load_record(owner)
        return standing_loss(
            record,
            project,
            member_of=lambda: _member_of(pool, record),
            enforce=membership_enforced(),
        )
    except Exception:
        logger.warning("standing check failed for %s", owner, exc_info=True)
        return None


def _member_of(pool: Any, record: Any) -> set[str]:
    conn = pool.getconn()
    try:
        return live_project_ids(conn, record)
    finally:
        pool.putconn(conn)


# ── the cascade ───────────────────────────────────────────────────────────


def run_cascade(
    user: str,
    *,
    project: str | None,
    reason: str,
    actor: str,
    trigger: str,
    broker: Any,
    job_store: Any = None,
    curve_provider: Any = None,
    audit_pool: Any = None,
    in_process: bool = False,
) -> CascadeResult:
    """End what ``user`` runs on this site (``project`` set: only there)
    and record it as one ``audit_events`` row. Lease revocation raises on a
    database error (it is the cascade's core); the other legs are
    best-effort and their counts say what happened."""
    result = CascadeResult(
        user=user,
        project=project or "",
        reason=reason,
        trigger=trigger,
        venue=str(getattr(broker, "venue", "") or ""),
    )
    revoked = broker.revoke_by_owner(user, project=project, actor=actor, reason=reason)
    result.leases_revoked = len(revoked)
    result.sessions_ended = len(
        {lease.session_id for lease in revoked if lease.session_id}
    )
    result.jobs_cancelled = cancel_owner_jobs(
        job_store, user, project, reason=f"{reason} ({actor})", broadcast=in_process
    )
    if project is None and curve_provider is not None:
        result.curve_sessions_revoked = int(curve_provider.revoke_user(user) or 0)
    result.audited = write_cascade_audit(audit_pool, result, actor)
    logger.info("account cascade: %s", result.details())
    return result


def cancel_owner_jobs(
    job_store: Any, user: str, project: str | None, *, reason: str, broadcast: bool
) -> int:
    """Durable cancel of the user's open jobs on this site, session jobs
    excepted (they end through their lease)."""
    if job_store is None:
        return 0
    try:
        jobs = _open_jobs(job_store, user, project)
    except Exception:
        logger.warning("cascade: jobs of %s not listed", user, exc_info=True)
        return 0
    cancelled = 0
    for job in jobs:
        cancelled += int(_cancel_job(job_store, job, reason, broadcast))
    return cancelled


def _open_jobs(job_store: Any, user: str, project: str | None) -> list[Any]:
    from umrs.jobs.models import JobStatus, JobType

    venue = str(getattr(job_store, "site_name", "") or "")
    jobs: list[Any] = []
    for status in (JobStatus.PLANNED, JobStatus.QUEUED, JobStatus.RUNNING):
        jobs += job_store.list_jobs(
            user_id=user, project_id=project, status=status, limit=1000
        )
    return [
        job
        for job in jobs
        if job.job_type != JobType.FRONTEND_SESSION and job.venue in (venue, "")
    ]


def _cancel_job(job_store: Any, job: Any, reason: str, broadcast: bool) -> bool:
    from umrs.jobs.store import InvalidTransitionError

    try:
        job_store.cancel_job(job.job_id, reason=reason)
    except InvalidTransitionError:
        return False  # ended meanwhile
    except Exception:
        logger.warning("cascade: job %s not cancelled", job.job_id, exc_info=True)
        return False
    if broadcast:
        _notify_in_process(job)
    return True


def _notify_in_process(job: Any) -> None:
    """Server process only: the cancel bus (downloads, fine-tunes, ...) and
    the in-process workflow registry."""
    from umrs.jobs.download_safety import broadcast_cancel
    from umrs.jobs.models import JobType
    from umrs.jobs.workflow_run import cancel_workflow_execution_for_job

    broadcast_cancel(job.job_id)
    if job.job_type != JobType.WORKFLOW_RUN:
        return
    try:
        cancel_workflow_execution_for_job(job)
    except Exception:
        logger.info("cascade: run of job %s not cancelled", job.job_id, exc_info=True)


def write_cascade_audit(pool: Any, result: CascadeResult, actor: str) -> bool:
    """One ``audit_events`` row for the cascade; False when not written."""
    if pool is None:
        return False
    row = (
        str(uuid.uuid4()),
        "1",
        time.time(),
        CASCADE_EVENT,
        actor[:128],
        result.project[:128],
        "",
        "success",
        "account",
        json.dumps(result.details()),
    )
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_events (event_id, schema_version, ts, event,"
                " actor, project_id, request_id, outcome, scope, details)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                row,
            )
        conn.commit()
        return True
    except Exception:
        logger.warning("account cascade audit not written", exc_info=True)
        conn.rollback()
        return False
    finally:
        pool.putconn(conn)


# ── the server-side sweep ─────────────────────────────────────────────────


def sweep_interval_seconds() -> float:
    raw = os.environ.get("UMRS_ACCOUNT_SWEEP_SEC", str(_SWEEP_DEFAULT_SEC))
    try:
        value = float(raw)
    except ValueError:
        value = _SWEEP_DEFAULT_SEC
    return min(_SWEEP_MAX_SEC, max(_SWEEP_MIN_SEC, value))


class AccountStandingSweep:
    """Ends what disabled accounts and (``enforce``) ex-members still hold.

    Every tick reads the live leases of this site, groups them by owner and
    checks each owner's live record; a loss runs ``run_cascade`` in-process
    (CURVE provider and cancel bus included). The sessions themselves end on
    their status checker's next cycle, through the revoked lease.
    """

    def __init__(
        self,
        *,
        pool: Any,
        broker: Any = None,
        job_store: Any = None,
        job_store_factory: Callable[[], Any] | None = None,
        record_lookup: Callable[[str], Any] = load_record,
        curve_provider: Callable[[], Any] | None = None,
        interval: float | None = None,
    ) -> None:
        self._pool = pool
        self._broker = broker
        self._job_store = job_store
        self._job_store_factory = job_store_factory
        self._lookup = record_lookup
        self._curve_provider = curve_provider or _shared_curve_provider
        self.interval = interval if interval is not None else sweep_interval_seconds()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def broker(self) -> Any:
        if self._broker is None:
            from umrs.jobs.secret_broker import SecretBroker

            self._broker = SecretBroker(db_pool=self._pool)
        return self._broker

    @property
    def job_store(self) -> Any:
        """Built on first use (it needs the database); a failure skips the
        jobs leg for this tick, never the lease revocation."""
        if self._job_store is None and self._job_store_factory is not None:
            try:
                self._job_store = self._job_store_factory()
            except Exception:
                logger.warning("account sweep: job store unavailable", exc_info=True)
        return self._job_store

    def run_once(self) -> list[CascadeResult]:
        """One sweep; returns the cascades it ran."""
        results = []
        for (user, project), reason in self.losses().items():
            results.append(
                run_cascade(
                    user,
                    project=project,
                    reason=reason,
                    actor="umrs:account-sweep",
                    trigger="sweep",
                    broker=self.broker,
                    job_store=self.job_store,
                    curve_provider=self._curve_provider() if project is None else None,
                    audit_pool=self._pool,
                    in_process=True,
                )
            )
        return results

    def losses(self) -> dict[tuple[str, str | None], str]:
        """``{(user, project or None for the whole account): reason}``."""
        enforce = membership_enforced()
        found: dict[tuple[str, str | None], str] = {}
        for user, projects in _by_owner(self.broker.active_owner_projects()).items():
            try:
                found.update(self._user_losses(user, projects, enforce))
            except Exception:
                logger.warning("account sweep: %s skipped", user, exc_info=True)
        return found

    def _user_losses(
        self, user: str, projects: Iterable[str], enforce: bool
    ) -> dict[tuple[str, str | None], str]:
        record = self._lookup(user)
        if record is not None and record.disabled:
            return {(user, None): ACCOUNT_DISABLED}
        members = _Memo(lambda: _member_of(self._pool, record))
        out: dict[tuple[str, str | None], str] = {}
        for project in projects:
            reason = standing_loss(
                record, project, member_of=members.get, enforce=enforce
            )
            if reason:
                out[(user, project)] = reason
        return out

    # ── thread ──

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="umrs-account-sweep", daemon=True
        )
        self._thread.start()
        logger.info("account standing sweep started (every %.0fs)", self.interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.run_once()
            except Exception:
                logger.warning("account standing sweep tick failed", exc_info=True)


class _Memo:
    """Compute once, on first use (one projects query per account)."""

    def __init__(self, fn: Callable[[], set[str]]) -> None:
        self._fn = fn
        self._value: set[str] | None = None

    def get(self) -> set[str]:
        if self._value is None:
            self._value = self._fn()
        return self._value


def _by_owner(pairs: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for owner, project in pairs:
        grouped.setdefault(owner, set()).add(project)
    return grouped


def _shared_curve_provider() -> Any:
    from umrs.core.session_curve_provider import get_shared_session_provider

    return get_shared_session_provider()


__all__ = [
    "ACCOUNT_DISABLED",
    "CASCADE_EVENT",
    "MEMBERSHIP_REVOKED",
    "AccountStandingSweep",
    "CascadeResult",
    "account_disabled",
    "cancel_owner_jobs",
    "live_project_ids",
    "load_record",
    "membership_enforced",
    "project_exempt",
    "read_record",
    "refusal_for",
    "run_cascade",
    "standing_loss",
    "write_cascade_audit",
]
