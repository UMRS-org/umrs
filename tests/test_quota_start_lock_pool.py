"""ADR-099 D-6, Amendment 3 (RS-1 round-3 review, finding N2): waiting for
the venue start lock never uses up the front's database pool.

- ``start_lock`` takes the process's own lock before a pooled connection,
  so only the holder's connection is pinned while others wait: N parallel
  starts against an N-connection pool all succeed; before, every waiter
  polled on its own connection and the holder's count query raised
  ``PoolError: connection pool exhausted`` (the start failed with 503, and
  every other route on the pool with it);
- the in-process wait is bounded (``StartLockTimeout``), like the database's;
- a second start of a session already starting in this process is refused
  at once (``already_running``), never another waiter on the lock.
"""

from __future__ import annotations

import threading
import time

import pytest
from psycopg2.pool import PoolError
from tests.test_session_policy_start import _harness
from umrs.quotas.session_cap import (
    StartInFlight,
    StartLockTimeout,
    local_start_lock,
    start_in_flight,
    start_lock,
)
from umrs.venues.frontend_service import VenueError

_HELD: dict = {}
_GUARD = threading.Lock()


class _Cursor:
    """``pg_try_advisory_lock`` / ``pg_advisory_unlock`` on one shared table."""

    def __init__(self, conn):
        self.conn, self.row = conn, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        with _GUARD:
            if "try_advisory_lock" in sql:
                ok = _HELD.get(params) in (None, self.conn)
                if ok:
                    _HELD[params] = self.conn
                self.row = (ok,)
            elif "advisory_unlock" in sql:
                if _HELD.get(params) is self.conn:
                    del _HELD[params]
                self.row = (True,)

    def fetchone(self):
        return self.row


class _Conn:
    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    def close(self):
        pass


class _Pool:
    """psycopg2 ``ThreadedConnectionPool`` semantics: raise when exhausted."""

    def __init__(self, maxconn):
        self.free, self.lock, self.peak = maxconn, threading.Lock(), 0
        self.maxconn = maxconn

    def getconn(self):
        with self.lock:
            if self.free == 0:
                raise PoolError("connection pool exhausted")
            self.free -= 1
            self.peak = max(self.peak, self.maxconn - self.free)
            return _Conn()

    def putconn(self, _conn, **_kw):
        with self.lock:
            self.free += 1


def _parallel(n, fn):
    barrier = threading.Barrier(n)
    results: list = []

    def one(i):
        barrier.wait()
        try:
            fn(i)
            results.append("ok")
        except Exception as exc:  # noqa: BLE001 — the outcome is the assertion
            results.append(type(exc).__name__)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return results


class TestThePoolIsNotUsedUp:
    def test_parallel_starts_on_a_pool_their_size_all_succeed(self):
        pool = _Pool(4)

        def start(_i):
            with start_lock(pool, "rccs-pool", wait_s=5):
                time.sleep(0.1)  # the others are waiting now
                conn = pool.getconn()  # the holder's count / job insert
                pool.putconn(conn)

        assert _parallel(4, start) == ["ok"] * 4
        assert pool.peak == 2  # the holder's lock connection + its query
        assert _HELD == {}


class TestBoundedLocalWait:
    def test_a_local_wait_past_the_limit_times_out(self):
        with local_start_lock("rccs-local"):
            started = time.monotonic()
            with pytest.raises(StartLockTimeout), local_start_lock("rccs-local", wait_s=0.1):
                pass
            assert time.monotonic() - started < 2
        with local_start_lock("rccs-local", wait_s=0.1):
            pass  # released

    def test_a_start_behind_the_holder_times_out_without_a_connection(self):
        pool = _Pool(4)

        def behind(_i):
            with start_lock(pool, "rccs-deadline", wait_s=0.2):
                pass

        with start_lock(pool, "rccs-deadline"):
            started = time.monotonic()
            assert _parallel(1, behind) == ["StartLockTimeout"]
            assert time.monotonic() - started < 2
            assert pool.peak == 1  # the waiter never took a connection


class TestInFlight:
    def test_a_second_start_of_the_same_session_is_refused_at_once(self):
        with start_in_flight("rccs", "sess-x"):
            with pytest.raises(StartInFlight), start_in_flight("rccs", "sess-x"):
                pass
            with start_in_flight("rccs", "sess-y"), start_in_flight("octopus", "sess-x"):
                pass
        with start_in_flight("rccs", "sess-x"):
            pass  # released

    def test_the_service_refuses_a_duplicate_start_without_waiting(self, monkeypatch):
        h = _harness(monkeypatch)
        entered, release = threading.Event(), threading.Event()
        real = h.service._reserve_and_submit

        def slow(**kw):
            entered.set()
            release.wait(5)
            return real(**kw)

        h.service._reserve_and_submit = slow

        def start():
            h.service.start_plan_run(
                owner="alice",
                session_id="sess-1",
                run_id="run-1",
                run_job_id="job-run-1",
                source_refs=["openai-main"],
                web_policy="none",
            )

        first = threading.Thread(target=start)
        first.start()
        assert entered.wait(5)
        began = time.monotonic()
        with pytest.raises(VenueError) as err:
            start()
        assert err.value.code == "already_running"
        assert time.monotonic() - began < 1
        release.set()
        first.join(10)
        assert len(h.jobs.created) == 1
