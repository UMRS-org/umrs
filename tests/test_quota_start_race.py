"""ADR-099 D-6, Amendment 2 (RS-1 round-2 review, finding 2): parallel
starts never overshoot the concurrent-session cap.

- starts at one venue run the cap's count, the subnet, the job and the row
  under one start lock: eight parallel starts against a cap of three leave
  three jobs on distinct subnets; before, every start counted the same
  number and all eight were created;
- the routes' lock is a Postgres advisory lock (one database, every worker
  and container): a second holder waits, another venue does not, a wait
  past the limit is ``StartLockTimeout``, nothing stays locked, and eight
  parallel count-then-insert starts on the real job store leave exactly the
  cap (CI-shape Postgres: ``POSTGRES_*`` or ``UMRS_TEST_DB_URL``).
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import pytest
from fastapi import HTTPException
from tests.test_session_policy_start import _harness
from umrs.api.companion_venue import _start_lock
from umrs.quotas.session_cap import (
    START_LOCK_CLASS,
    StartLockTimeout,
    count_live_sessions,
    local_start_lock,
    start_lock,
)
from umrs.venues.frontend_service import VenueError

CAP = 3
STARTS = 8


def _capped(jobs, cap=CAP):
    """A start check that counts the live session jobs, slowly (the race window)."""

    def check(_check):
        live = sum(1 for j in jobs.jobs.values() if str(j.status) != "cancelled")
        time.sleep(0.05)
        if live >= cap:
            raise HTTPException(status_code=429, detail="cap")

    return check


def _start_all(service, n=STARTS):
    barrier = threading.Barrier(n)
    outcomes: list = []

    def one(i):
        barrier.wait()
        try:
            service.start_plan_run(
                owner="alice",
                session_id=f"sess-{i}",
                run_id=f"run-{i}",
                run_job_id=f"job-run-{i}",
                source_refs=["openai-main"],
                web_policy="none",
            )
            outcomes.append("ok")
        except HTTPException as exc:
            outcomes.append(exc.status_code)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return outcomes


class TestParallelStarts:
    def test_parallel_starts_stop_at_the_cap(self, monkeypatch):
        h = _harness(monkeypatch)
        h.service._quota_check = _capped(h.jobs)
        outcomes = _start_all(h.service)
        assert len(h.jobs.created) == CAP
        assert sorted(outcomes, key=str) == sorted(["ok"] * CAP + [429] * (STARTS - CAP), key=str)
        subnets = [c["spec"]["subnet_hint"] for c in h.jobs.created]
        assert len(set(subnets)) == CAP

    def test_a_busy_lock_is_a_typed_unavailable(self, monkeypatch):
        h = _harness(monkeypatch)

        def busy(_venue):
            raise StartLockTimeout("venue start lock busy for 15s")

        h.service._start_lock = busy
        with pytest.raises(VenueError) as err:
            h.service.start_plan_run(
                owner="alice",
                session_id="sess-1",
                run_id="run-1",
                run_job_id="job-run-1",
                source_refs=["openai-main"],
                web_policy="none",
            )
        assert err.value.code == "unavailable"
        assert h.jobs.created == []

    def test_the_routes_use_the_advisory_lock_with_a_pool(self):
        assert _start_lock(None) is local_start_lock
        assert _start_lock(object()) is not local_start_lock


def _db_url() -> str | None:
    """CI-shape Postgres (the job's ``POSTGRES_*``) or ``UMRS_TEST_DB_URL``."""
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


needs_pg = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


@pytest.fixture
def pg_pool():
    from psycopg2.pool import ThreadedConnectionPool

    pool = ThreadedConnectionPool(1, 2 * STARTS + 2, _db_url())
    yield pool
    pool.closeall()


def _held(pool) -> int:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM pg_locks WHERE locktype = 'advisory' AND classid = %s",
                (START_LOCK_CLASS,),
            )
            (n,) = cur.fetchone()
        conn.rollback()
        return int(n)
    finally:
        pool.putconn(conn)


@needs_pg
class TestAdvisoryLock:
    def test_a_second_start_waits_for_the_first(self, pg_pool):
        order: list[str] = []
        entered = threading.Event()

        def first():
            with start_lock(pg_pool, "rccs"):
                entered.set()
                time.sleep(0.3)
                order.append("first-out")

        t = threading.Thread(target=first)
        t.start()
        entered.wait(5)
        with start_lock(pg_pool, "rccs"):
            order.append("second-in")
        t.join(5)
        assert order == ["first-out", "second-in"]
        assert _held(pg_pool) == 0

    def test_another_venue_does_not_wait_and_a_long_wait_times_out(self, pg_pool):
        with start_lock(pg_pool, "rccs"):
            with start_lock(pg_pool, "octopus", wait_s=0.2):
                pass
            with pytest.raises(StartLockTimeout), start_lock(pg_pool, "rccs", wait_s=0.2):
                pass
        assert _held(pg_pool) == 0

    def test_parallel_starts_against_the_real_job_store_stop_at_the_cap(self, pg_pool):
        from umrs.jobs.models import JobType, RestartPolicy
        from umrs.jobs.store import JobStore

        venue, user = f"race-{uuid.uuid4().hex[:8]}", f"u-{uuid.uuid4().hex[:8]}"
        store = JobStore(pg_pool)
        barrier = threading.Barrier(STARTS)

        def one(i):
            barrier.wait()
            with start_lock(pg_pool, venue):
                live = count_live_sessions(pg_pool, venue=venue, user_id=user)
                time.sleep(0.05)
                if live < CAP:
                    store.create_job(
                        project_id="race",
                        user_id=user,
                        job_type=JobType.FRONTEND_SESSION,
                        spec={"venue": venue, "session_id": f"s-{i}"},
                        idempotency_key=f"race-{venue}-{i}",
                        max_attempts=1,
                        restart_policy=RestartPolicy.NEVER,
                    )

        threads = [threading.Thread(target=one, args=(i,)) for i in range(STARTS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert count_live_sessions(pg_pool, venue=venue, user_id=user) == CAP
        assert _held(pg_pool) == 0
