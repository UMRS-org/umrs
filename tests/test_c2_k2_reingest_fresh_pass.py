"""Cycle-2 umrs-follow lane (N3, 2026-10-21): re-ingest after the first ingest ended.

Found by the umrs-api lane: a KNOWLEDGE_INGEST job's idempotency key was
the bare spec hash and the unique index on (project_id, idempotency_key) is
not partial on status, so the same sources (a git repo, a URL) could never
be ingested again: every later submit was a 409 naming the first, long
finished, job. Re-ingesting to pick up new content is a core user need.

Decision (INTEGRATION-NOTES, N3): deduplicate ONLY while an identical job
is in flight (double-click protection: DuplicateJobError -> 409 +
existing_job_id); once that job is terminal the same spec is a fresh pass.
The key is now the fresh-pass key (spec hash + nonce, as for
KNOWLEDGE_REINDEX) and ``JobStore.create_job`` refuses, under an advisory
lock in the insert's transaction, while a job of the same spec hash is live.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any

import pytest
from umrs.jobs.models import (
    FRESH_PASS_JOB_TYPES,
    INFLIGHT_DEDUPE_JOB_TYPES,
    JobType,
    compute_idempotency_key,
)
from umrs.jobs.store import DuplicateJobError, JobStore

SPEC = {"sources": [{"kind": "git", "uri": "https://example.org/repo.git"}], "graph_id": "g"}


# ── hermetic: the SQL create_job runs ──────────────────────────────────────


class _Cursor:
    def __init__(self, live_job: str | None) -> None:
        self.live_job = live_job
        self.sql: list[str] = []
        self.params: list[tuple] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.sql.append(" ".join(sql.split()))
        self.params.append(tuple(params))

    def fetchone(self) -> tuple | None:
        last = self.sql[-1]
        if last.startswith("SELECT job_id FROM jobs WHERE project_id = %s AND (idempotency_key"):
            return (self.live_job,) if self.live_job else None
        if last.startswith("INSERT INTO jobs"):
            return ("job-new",)
        return None


class _Conn:
    def __init__(self, cur: _Cursor) -> None:
        self.cur = cur
        self.commits = 0
        self.rollbacks = 0

    def cursor(self) -> _Cursor:
        return self.cur

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    def getconn(self) -> _Conn:
        return self.conn

    def putconn(self, conn: _Conn) -> None:
        pass


def _fake_store(monkeypatch: pytest.MonkeyPatch, live_job: str | None) -> tuple[JobStore, _Conn]:
    import umrs.jobs.store as store_mod

    monkeypatch.setattr(store_mod, "ensure_schema", lambda _conn: None)
    conn = _Conn(_Cursor(live_job))
    return JobStore(_Pool(conn)), conn


def test_ingest_is_an_inflight_dedupe_fresh_pass_type() -> None:
    assert JobType.KNOWLEDGE_INGEST in INFLIGHT_DEDUPE_JOB_TYPES
    assert JobType.KNOWLEDGE_INGEST in FRESH_PASS_JOB_TYPES
    assert JobType.KNOWLEDGE_REINDEX not in INFLIGHT_DEDUPE_JOB_TYPES


def test_live_identical_ingest_is_refused_before_any_insert(monkeypatch) -> None:
    store, conn = _fake_store(monkeypatch, live_job="job-live")
    with pytest.raises(DuplicateJobError) as exc:
        store.create_job("pa", "alice", JobType.KNOWLEDGE_INGEST, SPEC)
    assert exc.value.existing_job_id == "job-live"
    assert not [s for s in conn.cur.sql if s.startswith("INSERT")]
    assert conn.rollbacks == 1  # the transaction (and its advisory lock) ends
    assert conn.cur.sql[0].startswith("SELECT pg_advisory_xact_lock")


def test_without_a_live_twin_the_ingest_gets_a_fresh_pass_key(monkeypatch) -> None:
    store, conn = _fake_store(monkeypatch, live_job=None)
    assert store.create_job("pa", "alice", JobType.KNOWLEDGE_INGEST, SPEC).startswith("job-")
    base = compute_idempotency_key("pa", "alice", JobType.KNOWLEDGE_INGEST, SPEC)
    check = conn.cur.params[1]
    assert check[:3] == ("pa", base, f"{base}:%")
    assert set(check[3]) == {"completed", "all_complete", "failed", "cancelled"}
    insert_key = conn.cur.params[2][5]
    assert insert_key.startswith(f"{base}:") and len(insert_key) <= 64


def test_a_callers_own_key_and_other_types_skip_the_inflight_check(monkeypatch) -> None:
    store, conn = _fake_store(monkeypatch, live_job="job-live")
    store.create_job("pa", "alice", JobType.KNOWLEDGE_INGEST, SPEC, idempotency_key="mine")
    store.create_job("pa", "alice", JobType.INDEX_BUILD, SPEC)
    assert not [s for s in conn.cur.sql if "pg_advisory_xact_lock" in s]


# ── real Postgres (CI-shape POSTGRES_* env) ────────────────────────────────


def _db_url() -> str | None:
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


@pytest.fixture()
def pg_store():
    from psycopg2.pool import ThreadedConnectionPool

    pool = ThreadedConnectionPool(1, 6, _db_url())
    project = f"c2-n3-{uuid.uuid4().hex[:8]}"
    store = JobStore(pool)
    yield store, pool, project
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM jobs WHERE project_id = %s", (project,))
        conn.commit()
    finally:
        pool.putconn(conn)
    pool.closeall()


def _set_status(pool: Any, job_id: str, status: str) -> None:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE jobs SET status = %s WHERE job_id = %s", (status, job_id))
        conn.commit()
    finally:
        pool.putconn(conn)


@needs_pg
@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_reingest_after_the_first_ended_is_a_fresh_job(pg_store, terminal: str) -> None:
    store, pool, project = pg_store
    first = store.create_job(project, "alice", JobType.KNOWLEDGE_INGEST, SPEC)
    with pytest.raises(DuplicateJobError) as dup:
        store.create_job(project, "alice", JobType.KNOWLEDGE_INGEST, SPEC)
    assert dup.value.existing_job_id == first

    _set_status(pool, first, terminal)
    second = store.create_job(project, "alice", JobType.KNOWLEDGE_INGEST, SPEC)
    assert second != first
    with pytest.raises(DuplicateJobError) as dup2:
        store.create_job(project, "alice", JobType.KNOWLEDGE_INGEST, SPEC)
    assert dup2.value.existing_job_id == second


@needs_pg
def test_a_live_legacy_bare_hash_row_still_dedupes(pg_store) -> None:
    """Rows written before the fresh-pass key carry the bare spec hash."""
    store, _pool, project = pg_store
    base = compute_idempotency_key(project, "alice", JobType.KNOWLEDGE_INGEST, SPEC)
    legacy = store.create_job(
        project, "alice", JobType.KNOWLEDGE_INGEST, SPEC, idempotency_key=base
    )
    with pytest.raises(DuplicateJobError) as dup:
        store.create_job(project, "alice", JobType.KNOWLEDGE_INGEST, SPEC)
    assert dup.value.existing_job_id == legacy


@needs_pg
def test_racing_identical_submits_yield_one_job(pg_store) -> None:
    store, _pool, project = pg_store
    barrier = threading.Barrier(4)
    created: list[str] = []
    refused: list[str] = []

    def submit() -> None:
        barrier.wait()
        try:
            created.append(store.create_job(project, "alice", JobType.KNOWLEDGE_INGEST, SPEC))
        except DuplicateJobError as exc:
            refused.append(exc.existing_job_id)

    threads = [threading.Thread(target=submit) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert len(created) == 1, (created, refused)
    assert refused == created * 3


@needs_pg
def test_other_users_and_specs_are_not_deduped_against(pg_store) -> None:
    store, _pool, project = pg_store
    store.create_job(project, "alice", JobType.KNOWLEDGE_INGEST, SPEC)
    store.create_job(project, "bob", JobType.KNOWLEDGE_INGEST, SPEC)
    store.create_job(project, "alice", JobType.KNOWLEDGE_INGEST, {**SPEC, "graph_id": "h"})
