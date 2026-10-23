"""Concurrent remote sessions per user, and optionally per project (ADR-099 D-6).

At venue start (chat sessions and plan-run workers alike) UMRS counts the
user's live ``frontend_session`` jobs at this venue (``planned``, ``queued``,
``running``) and refuses when the count is >= ``sessions.concurrent_user``
of the start's resolved policy (site default 12, from
``UMRS_SESSION_CONCURRENT_PER_USER``, parsed by ``session_policy.site_defaults``
only), and the project's likewise when
``sessions.concurrent_project`` is set: ``quota.concurrent_sessions_exceeded``
(429).

Only remote sessions count: the cap protects shared server capacity, and a
local session is neither on the server nor observable by it.

Starts at one venue are serialised from the count to the job insert
(:func:`start_lock`: a Postgres advisory lock, so across uvicorn workers and
containers on one database), so parallel starts never overshoot the cap
(ADR-099 D-6, Amendment 2; before, N racing starts could overshoot by N-1).
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from umrs.api.agent_quota import LIVE_JOB_STATUSES
from umrs.api.typed_errors import PlatformError

if TYPE_CHECKING:  # lane P's start hook (umrs.venues.start_checks)
    from umrs.venues.start_checks import StartCheck

SESSION_JOB_TYPE = "frontend_session"

#: Advisory-lock class (two-key form, its own key space) of venue starts.
START_LOCK_CLASS = 0x5343
START_LOCK_WAIT_S = 15.0
_START_LOCK_POLL_S = 0.05


class StartLockTimeout(RuntimeError):
    """Another start at this venue held the start lock past the wait."""


class StartInFlight(RuntimeError):
    """A start of this session is already in progress in this process."""


_SCOPES = (
    ("user", "concurrent_user", "owner", "user_id"),
    ("project", "concurrent_project", "project", "project_id"),
)


def session_limits(policy: Mapping[str, Any] | None) -> dict[str, int | None]:
    """``sessions`` of a resolved policy; a missing key is the site default.

    The site default and every value are read the one way the policy reads
    them (``session_policy``: ``site_defaults`` parses
    ``UMRS_SESSION_CONCURRENT_PER_USER``, ``limit_value`` reads a count:
    null = no limit, a positive integer = the cap)."""
    from umrs.session_policy import limit_value, site_defaults

    sessions = dict((policy or {}).get("sessions") or {})
    if "concurrent_user" not in sessions:
        sessions["concurrent_user"] = site_defaults()["sessions"]["concurrent_user"]
    return {
        "concurrent_user": limit_value(sessions.get("concurrent_user")),
        "concurrent_project": limit_value(sessions.get("concurrent_project")),
    }


def count_live_sessions(
    pool: Any, *, venue: str, user_id: str = "", project_id: str = ""
) -> int:
    """Live ``frontend_session`` jobs at ``venue`` (of one user and/or project)."""
    if pool is None:
        return 0
    clauses = ["job_type = %s", "status = ANY(%s)", "venue = %s"]
    params: list[Any] = [SESSION_JOB_TYPE, list(LIVE_JOB_STATUSES), venue]
    for column, value in (("user_id", user_id), ("project_id", project_id)):
        if value:
            clauses.append(f"{column} = %s")
            params.append(value)
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM jobs WHERE " + " AND ".join(clauses),
                    tuple(params),
                )
                row = cur.fetchone()
    finally:
        pool.putconn(conn)
    return int(row[0]) if row else 0


def _refusal(
    scope: str, subject: str, venue: str, limit: int, used: int
) -> HTTPException:
    where = venue or "this site"
    message = (
        f"concurrent session limit reached for {scope} {subject} at {where}: "
        f"{used} of {limit} live sessions; stop one (r4 session stop <id>) "
        "or ask a platform admin to raise the limit"
    )
    error = PlatformError.quota(
        "concurrent_sessions_exceeded",
        message,
        scope=scope,
        subject=subject,
        limit=limit,
        used=used,
    )
    return HTTPException(**error.to_http_parts())


def check_concurrent_sessions(pool: Any, check: StartCheck) -> None:
    """Raise HTTPException(quota.concurrent_sessions_exceeded) at the cap."""
    limits = session_limits(check.policy)
    for scope, key, attr, column in _SCOPES:
        limit = limits[key]
        subject = str(getattr(check, attr, "") or "")
        if limit is None or not subject:
            continue
        used = count_live_sessions(pool, venue=check.venue, **{column: subject})
        if used >= limit:
            raise _refusal(scope, subject, check.venue, limit, used)


def start_lock_key(venue: str) -> int:
    """The venue's 31-bit key inside :data:`START_LOCK_CLASS`."""
    digest = hashlib.sha256(f"venue-start:{venue}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big") & 0x7FFFFFFF


def _try_lock(conn: Any, key: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s, %s)", (START_LOCK_CLASS, key))
        row = cur.fetchone()
    conn.commit()  # a session lock: no transaction stays open while held
    return bool(row and row[0])


def _acquire(conn: Any, key: int, wait_s: float) -> None:
    deadline = time.monotonic() + wait_s
    while not _try_lock(conn, key):
        if time.monotonic() >= deadline:
            raise StartLockTimeout(f"venue start lock busy for {wait_s:g}s")
        time.sleep(_START_LOCK_POLL_S)


def _release(conn: Any, key: int) -> bool:
    """Unlock; False when the connection must not go back to the pool."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s, %s)", (START_LOCK_CLASS, key))
        conn.commit()
        return True
    except Exception:  # noqa: BLE001 — a broken connection drops its locks when closed
        return False


@contextmanager
def start_lock(
    pool: Any, venue: str, *, wait_s: float = START_LOCK_WAIT_S
) -> Iterator[None]:
    """Hold the venue's start lock (count → job insert, ADR-099 D-6).

    A session-level advisory lock on its own pooled connection, polled with
    ``pg_try_advisory_lock`` up to ``wait_s`` (then :class:`StartLockTimeout`).
    The job store commits the new job on its own connection before this
    unlocks, so the next start's count sees it.

    The process's own lock is taken first, so at most one connection per
    process waits on the database lock: N parallel starts no longer pin N
    pooled connections and starve the holder's own queries (Amendment 3)."""
    deadline = time.monotonic() + wait_s
    with local_start_lock(venue, wait_s=wait_s):
        key = start_lock_key(venue)
        conn = pool.getconn()
        reusable = False
        try:
            _acquire(conn, key, max(0.0, deadline - time.monotonic()))
            try:
                yield
            finally:
                reusable = _release(conn, key)
        finally:
            _put_back(pool, conn, reusable)


def _put_back(pool: Any, conn: Any, reusable: bool) -> None:
    if reusable:
        pool.putconn(conn)
        return
    try:
        pool.putconn(conn, close=True)
    except TypeError:  # a pool without ``close``
        conn.close()
        pool.putconn(conn)


_LOCAL_LOCKS: dict[str, threading.Lock] = {}
_LOCAL_GUARD = threading.Lock()


@contextmanager
def local_start_lock(
    venue: str, *, wait_s: float = START_LOCK_WAIT_S
) -> Iterator[None]:
    """The in-process start lock (alone without a database pool: tests, one
    process; before :func:`start_lock`'s connection with one). A wait past
    ``wait_s`` is :class:`StartLockTimeout`."""
    with _LOCAL_GUARD:
        lock = _LOCAL_LOCKS.setdefault(venue, threading.Lock())
    if not lock.acquire(timeout=wait_s):
        raise StartLockTimeout(f"venue start lock busy for {wait_s:g}s")
    try:
        yield
    finally:
        lock.release()


_IN_FLIGHT: set[tuple[str, str]] = set()


@contextmanager
def start_in_flight(venue: str, session_id: str) -> Iterator[None]:
    """Mark a start of ``session_id`` at ``venue`` in progress in this
    process; a second one is :class:`StartInFlight` at once, never another
    waiter on the start lock (Amendment 3)."""
    key = (venue, session_id)
    with _LOCAL_GUARD:
        if key in _IN_FLIGHT:
            raise StartInFlight(f"a start of session {session_id} is in progress")
        _IN_FLIGHT.add(key)
    try:
        yield
    finally:
        with _LOCAL_GUARD:
            _IN_FLIGHT.discard(key)


__all__ = [
    "SESSION_JOB_TYPE",
    "START_LOCK_CLASS",
    "StartInFlight",
    "StartLockTimeout",
    "check_concurrent_sessions",
    "count_live_sessions",
    "local_start_lock",
    "session_limits",
    "start_in_flight",
    "start_lock",
    "start_lock_key",
]
