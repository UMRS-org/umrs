"""WP-14 venue claim scoping (WP-14/6b-3 theft fix).

End-to-end against the shared env-driven PG: site-bound jobs must only be
claimed by their own site's schedulers; pre-venue rows (venue='') stay
claimable anywhere; spec-declared venue wins over the site's default.
"""

from __future__ import annotations

from typing import Any

import pytest
from umrs.jobs.models import JobType


def _env_pool_or_skip():
    pytest.importorskip("psycopg2")
    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("TRUNCATE jobs CASCADE")
            cur.execute("DELETE FROM job_attempts")
            cur.execute("DELETE FROM jobs")
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture
def pool(request):
    p = _env_pool_or_skip()
    request.addfinalizer(p.closeall)
    return p


def _make_store(pool: Any, site_name: str):
    from umrs.jobs.store import JobStore

    store = JobStore(pool, "test-instance")
    store.site_name = site_name
    return store


def _create(store, spec: dict[str, Any] | None = None) -> str:
    use_spec = dict(spec or {})
    return store.create_job(
        project_id="proj-x",
        user_id="alice",
        job_type=JobType.RESEARCH_LOOP,
        spec={"loop_id": "l1", **use_spec},
        idempotency_key=None,
    )


class TestVenueClaimScoping:
    def test_mine_only_my_site_when_row_is_bound(self, pool):
        rccs = _make_store(pool, "rccs-front")
        bdrt = _make_store(pool, "bdrt")
        job = _create(rccs)
        # Externalize to QUEUED
        rccs.enqueue_job(job)

        assert rccs.site_name in ("rccs-front",)
        assert rccs.claim_next_jobs(limit=10) == [job]
        assert bdrt.claim_next_jobs(limit=10) == []

    def test_empty_venue_claimable_everywhere(self, pool):
        # Simulate a pre-venue row with venue='' (global).
        store_a = _make_store(pool, "rccs-front")
        store_b = _make_store(pool, "bdrt")
        job = _create(store_a)
        store_a.enqueue_job(job)
        conn = pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("UPDATE jobs SET venue = '' WHERE job_id = %s", (job,))
        finally:
            pool.putconn(conn)
        assert job in store_a.claim_next_jobs(limit=10)
        assert job in store_b.claim_next_jobs(limit=10)

    def test_spec_declared_venue_outranks_site_default(self, pool):
        rccs = _make_store(pool, "rccs-front")
        bdrt = _make_store(pool, "bdrt")
        job = _create(bdrt, spec={"venue": "rccs-front"})
        bdrt.enqueue_job(job)  # enqueue is lifecycle; claim is venue-scoped
        assert rccs.claim_next_jobs(limit=10) == [job]
        assert bdrt.claim_next_jobs(limit=10) == []

    def test_cluster_prefix_normalizes_to_site_name_at_create(self, pool):
        store = _make_store(pool, "bdrt")
        job = store.create_job(
            project_id="proj-x",
            user_id="alice",
            job_type=JobType.RESEARCH_LOOP,
            spec={"venue": "cluster:rccs-front", "loop_id": "l"},
            idempotency_key=None,
        )
        rccs = _make_store(pool, "rccs-front")
        bdrt = _make_store(pool, "bdrt")
        store.enqueue_job(job)
        assert rccs.claim_next_jobs(limit=10) == [job]
        assert bdrt.claim_next_jobs(limit=10) == []


class TestEnsureSchema:
    def test_ensure_schema_adds_venue_column_for_existing_dbs(self, pool):
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'jobs' AND column_name = 'venue'"
                )
                assert cur.fetchone() is not None
        finally:
            pool.putconn(conn)
