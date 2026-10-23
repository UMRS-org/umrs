"""S1a (N-3): a store never writes a bare venue no server will claim.

Before S1a ``JobStore._resolve_venue`` stored any bare venue as written: a
typo such as ``bdrt`` (the site is ``bdrt-front``) became a row no server
claims, so the job sat QUEUED forever. With a site name configured, a bare
venue other than ``""``, ``frontend`` or the store's own name now raises
``unknown_venue`` and nothing is inserted; with no site name the venue is
stored as written (unchanged). ``cluster:<site>`` keeps resolving.

Runs against the env-driven Postgres like ``test_job_venue_scoping.py``.
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
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture
def pool(request):
    p = _env_pool_or_skip()
    request.addfinalizer(p.closeall)
    return p


def _store(pool: Any, site_name: str):
    from umrs.jobs.store import JobStore

    store = JobStore(pool, "test-instance")
    store.site_name = site_name
    return store


def _create(store, venue: str) -> str:
    return store.create_job(
        project_id="proj-v",
        user_id="alice",
        job_type=JobType.RESEARCH_LOOP,
        spec={"loop_id": "l1", "venue": venue},
        idempotency_key=None,
    )


def _venue_of(pool: Any, job_id: str) -> str:
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT venue FROM jobs WHERE job_id = %s", (job_id,))
            return cur.fetchone()[0]
    finally:
        pool.putconn(conn)


def _count(pool: Any) -> int:
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM jobs")
            return int(cur.fetchone()[0])
    finally:
        pool.putconn(conn)


@pytest.mark.parametrize("venue", ["bdrt", "rccs-front"])
def test_unknown_bare_venue_is_refused_and_nothing_is_stored(pool, venue):
    store = _store(pool, "bdrt-front")
    with pytest.raises(ValueError, match="unknown_venue"):
        _create(store, venue)
    assert _count(pool) == 0


def test_unknown_venue_is_a_value_error_for_the_client_seams():
    from umrs.jobs.store import UnknownVenueError

    assert issubclass(UnknownVenueError, ValueError)


@pytest.mark.parametrize("venue", ["", "frontend", "bdrt-front"])
def test_own_site_bare_venues_resolve_to_the_site(pool, venue):
    store = _store(pool, "bdrt-front")
    job = _create(store, venue)
    assert _venue_of(pool, job) == "bdrt-front"
    store.enqueue_job(job)
    assert store.claim_next_jobs(limit=10) == [job]


def test_cluster_venue_still_resolves_to_its_site(pool):
    store = _store(pool, "bdrt-front")
    job = _create(store, "cluster:rccs-front")
    assert _venue_of(pool, job) == "rccs-front"


def test_no_site_name_stores_the_bare_venue_unchanged(pool):
    store = _store(pool, "")
    job = _create(store, "bdrt")
    assert _venue_of(pool, job) == "bdrt"


def _fan_out(store, child_spec: dict) -> str:
    return store.create_fan_out_graph(
        project_id="proj-v",
        user_id="alice",
        parent_job_type=JobType.CLEANUP,
        parent_spec={},
        children=[{"name": "c", "job_type": "cleanup", "spec": child_spec}],
    )


def _children(pool: Any, parent_id: str) -> list[tuple[str, str]]:
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, venue FROM jobs WHERE spec->'_fan_out'->>'parent_job_id' = %s",
                (parent_id,),
            )
            return [(row[0], row[1]) for row in cur.fetchall()]
    finally:
        pool.putconn(conn)


def test_fan_out_rows_carry_their_venue_and_no_other_site_claims_them(pool):
    """N-1: parent and child rows were inserted without a venue (''), so any
    umrs-front sharing the jobs table claimed a QUEUED child and ran it with
    no --account under its own service connection."""
    rccs, bdrt = _store(pool, "rccs-front"), _store(pool, "bdrt-front")
    parent = _fan_out(rccs, {})
    [(child, venue)] = _children(pool, parent)
    assert venue == "rccs-front"
    assert _venue_of(pool, parent) == "rccs-front"
    assert bdrt.claim_next_jobs(limit=10) == []
    assert rccs.claim_next_jobs(limit=10) == [child]


def test_fan_out_child_keeps_a_cluster_venue_and_refuses_a_typo(pool):
    rccs = _store(pool, "rccs-front")
    parent = _fan_out(rccs, {"venue": "cluster:rccs-front"})
    assert [v for _id, v in _children(pool, parent)] == ["rccs-front"]
    with pytest.raises(ValueError, match="unknown_venue"):
        _fan_out(rccs, {"venue": "bdrt"})
