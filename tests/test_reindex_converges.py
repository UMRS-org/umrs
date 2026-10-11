"""KNOWLEDGE_REINDEX converges (ADR-093 D-11 amendment, wave F, 2026-10-11).

S7's first live run could not converge a 5.5k-artifact project:

- one reindex swept at most ``batch_limit`` artifacts and nothing continued
  the pass;
- an identical repeat was refused for ever: the store hashed the spec into
  the idempotency key, the unique index on (project_id, idempotency_key) is
  not partial on status, so the REST path answered 409 and the ZMQ path
  silently returned the old completed job;
- a second pass re-embedded what unfinished embed jobs already held.

One job now makes one keyset-paged pass, skips held artifacts, takes a
per-project lock, and a reindex submitted without a key gets a fresh one.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.jobs import reindex
from umrs.jobs.models import (
    FRESH_PASS_JOB_TYPES,
    JobType,
    compute_idempotency_key,
    default_idempotency_key,
)
from umrs.jobs.reindex import KnowledgeReindexBackend, ReindexBusy
from umrs.jobs.store import JobStore
from umrs.knowledge.embedding_metadata import HeldBy, select_stale_artifacts

SPEC = {
    "target_model": "e5-base-v2",
    "index_name": "unused",
    "database": "umrs_db",
    "table": "knowledge_node_embeddings",
}


class _Cur:
    """``answers`` reply to successive fetchall calls; ``lock`` to fetchone."""

    def __init__(self, answers: list[list[tuple]], lock: bool = True) -> None:
        self.statements: list[tuple[str, Any]] = []
        self._answers = [list(a) for a in answers]
        self._lock = lock

    def execute(self, sql: str, params: Any = None) -> None:
        self.statements.append((" ".join(str(sql).split()), params))

    def fetchall(self) -> list[tuple]:
        return self._answers.pop(0) if self._answers else []

    def fetchone(self) -> tuple:
        return (self._lock,)

    def sweeps(self) -> list[tuple[str, Any]]:
        return [(q, p) for q, p in self.statements if q.startswith("SELECT artifact_id FROM")]


def _rows(*ids: str) -> list[tuple]:
    return [(a, "p1", f"k-{a}", "document", {}) for a in ids]


def _submit(monkeypatch, cur: _Cur, **spec: Any) -> tuple[str, list[Any]]:
    @contextlib.contextmanager
    def scope():
        yield SimpleNamespace(cursor=lambda: cur)

    submitted: list[Any] = []

    class _Svc:
        def submit_job(self, submission):
            submitted.append(submission)
            return f"job-embed-{len(submitted)}"

    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "/srv/models/e5-base-v2")
    monkeypatch.delenv("UMRS_KNOWLEDGE_CHUNK_SCHEME", raising=False)
    monkeypatch.setattr(reindex, "_artifact_conn", scope)
    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: _Svc())
    job = SimpleNamespace(
        job_id="j-1", spec={**SPEC, **spec}, user_id="u", priority=0, project_id="p1"
    )
    ref = asyncio.run(KnowledgeReindexBackend().submit_job(job, SimpleNamespace(attempt_id="a")))
    return ref, submitted


# ── one pass, in keyset pages ────────────────────────────────────────────────


def test_one_reindex_pages_past_batch_limit(monkeypatch):
    """3 stale, batch_limit=2: the probe of 2026-10-10 got ['a1', 'a2'] and
    the job ended. One pass now covers a1..a3 in two embed jobs."""
    cur = _Cur([[("a1",), ("a2",)], _rows("a1", "a2"), [], [("a3",)], _rows("a3"), []])
    ref, submitted = _submit(monkeypatch, cur, batch_limit=2)
    sources = [x["artifact_id"] for s in submitted for x in s.spec["sources"]]
    assert sources == ["a1", "a2", "a3"]
    assert "swept=3|reembed_jobs=2|pages=2" in ref
    assert all(len(s.spec["sources"]) <= 2 for s in submitted)  # batch_limit per embed job


def test_the_next_page_starts_after_the_last_artifact(monkeypatch):
    cur = _Cur([[("a1",), ("a2",)], _rows("a1", "a2"), [], [("a3",)], _rows("a3"), []])
    _submit(monkeypatch, cur, batch_limit=2)
    (first_sql, first), (second_sql, second) = cur.sweeps()
    assert "artifact_id > %s" not in first_sql
    assert "artifact_id > %s" in second_sql and "a2" in second
    assert second_sql.endswith("ORDER BY artifact_id LIMIT %s")


def test_an_artifact_that_turns_stale_again_is_not_visited_twice(monkeypatch):
    """a1 fails and is stale again while the pass runs: the keyset cursor
    (artifact_id > 'a2') never returns it in this pass, so the pass ends."""
    cur = _Cur([[("a1",), ("a2",)], _rows("a1", "a2"), [], [], [], []])
    ref, submitted = _submit(monkeypatch, cur, batch_limit=2)
    assert [x["artifact_id"] for s in submitted for x in s.spec["sources"]] == ["a1", "a2"]
    assert "pages=1" in ref and len(cur.sweeps()) == 2


def test_a_pass_stops_at_the_first_short_page(monkeypatch):
    cur = _Cur([[("a1",)], _rows("a1"), []])
    _submit(monkeypatch, cur, batch_limit=2)
    assert len(cur.sweeps()) == 1


def test_the_checker_reports_the_pages():
    attempt = SimpleNamespace(
        backend_ref="knowledge_reindex|database=d|table=t|swept=3|reembed_jobs=2|pages=2"
    )
    result = asyncio.run(
        reindex.KnowledgeReindexStatusChecker().check_status(attempt, SimpleNamespace())
    )
    assert result.metrics["pages"] == "2"


# ── held artifacts and the per-project lock ──────────────────────────────────


def test_the_sweep_skips_artifacts_an_unfinished_embed_or_understand_job_holds():
    cur = _Cur([[]])
    held = HeldBy(("knowledge_embed", "knowledge_understand"), ("queued", "running"))
    select_stale_artifacts(SimpleNamespace(cursor=lambda: cur), "m", project_id="p1", held_by=held)
    ((sql, params),) = cur.statements
    assert "NOT EXISTS (SELECT 1 FROM jobs j CROSS JOIN LATERAL jsonb_array_elements(" in sql
    assert "jsonb_typeof(j.spec->'sources') = 'array'" in sql
    assert "j.project_id = knowledge_artifacts.project_id" in sql
    assert "NOT IN" not in sql
    assert params[-3:] == (
        ["knowledge_embed", "knowledge_understand"],
        ["queued", "running"],
        500,
    )


def test_a_pass_holds_out_queued_and_running_embed_and_understand_jobs():
    assert set(reindex.HELD_BY.job_types) == {"knowledge_embed", "knowledge_understand"}
    assert set(reindex.HELD_BY.statuses) == {"queued", "running"}  # never planned


def test_a_pass_while_another_holds_the_project_fails_transiently_before_any_sweep(
    monkeypatch,
):
    cur = _Cur([[("a1",)]], lock=False)
    with pytest.raises(ReindexBusy, match="'p1'"):
        _submit(monkeypatch, cur, batch_limit=2)
    ((lock_sql, lock_params),) = cur.statements
    assert lock_sql == "SELECT pg_try_advisory_xact_lock(%s, hashtext(%s))"
    assert lock_params == (reindex._PASS_LOCK_CLASS, "knowledge_reindex:p1")


# ── a fresh key for each pass ────────────────────────────────────────────────


def _key(job_type: JobType) -> str:
    return default_idempotency_key("p1", "u", job_type, {"table": "t"})


def test_an_identical_reindex_spec_is_a_new_pass():
    first, second = _key(JobType.KNOWLEDGE_REINDEX), _key(JobType.KNOWLEDGE_REINDEX)
    assert first != second and len(first) <= 64
    stable = compute_idempotency_key("p1", "u", JobType.KNOWLEDGE_REINDEX, {"table": "t"})
    assert first.startswith(f"{stable}:")


def test_other_job_types_keep_their_spec_hash():
    assert JobType.INDEX_BUILD not in FRESH_PASS_JOB_TYPES
    assert _key(JobType.INDEX_BUILD) == _key(JobType.INDEX_BUILD)
    assert _key(JobType.INDEX_BUILD) == compute_idempotency_key(
        "p1", "u", JobType.INDEX_BUILD, {"table": "t"}
    )


class _StoreCur:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __enter__(self) -> _StoreCur:
        return self

    def __exit__(self, *_a: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        if "INSERT INTO jobs" in sql:
            self.keys.append(params[5])


class _StoreConn:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def cursor(self) -> _StoreCur:
        return _StoreCur(self.keys)

    def commit(self) -> None:
        return None


class _StorePool:
    def __init__(self) -> None:
        self.keys: list[str] = []

    def getconn(self) -> _StoreConn:
        return _StoreConn(self.keys)

    def putconn(self, _conn: Any) -> None:
        return None


def _store() -> tuple[JobStore, _StorePool]:
    pool = _StorePool()
    store = JobStore.__new__(JobStore)
    store.db_pool, store.site_name = pool, ""
    return store, pool


def test_the_store_gives_two_identical_reindex_jobs_different_keys():
    """Both REST and ZMQ go through store.create_job: an identical repeat is
    inserted as new work, not a 409 / the old job."""
    store, pool = _store()
    spec = {**SPEC, "batch_limit": 6000}
    store.create_job("p1", "u", JobType.KNOWLEDGE_REINDEX, spec)
    store.create_job("p1", "u", JobType.KNOWLEDGE_REINDEX, spec)
    assert len(pool.keys) == 2 and pool.keys[0] != pool.keys[1]


def test_a_callers_key_still_deduplicates():
    store, pool = _store()
    store.create_job("p1", "u", JobType.KNOWLEDGE_REINDEX, dict(SPEC), idempotency_key="k-1")
    store.create_job("p1", "u", JobType.INDEX_BUILD, dict(SPEC))
    store.create_job("p1", "u", JobType.INDEX_BUILD, dict(SPEC))
    assert pool.keys[0] == "k-1" and pool.keys[1] == pool.keys[2]


# ── integration: a real Postgres (skipped without a database URL) ────────────

_IT = "reindex-held-it"


def _seed(cur: Any) -> None:
    for artifact in ("held-1", "free-2", "free-3"):
        cur.execute(
            "INSERT INTO knowledge_artifacts (artifact_id, project_id, content_hash, kind, "
            "embedding_model, embedding_status) VALUES (%s, %s, %s, 'document', 'm', 'stale')",
            (f"{_IT}-{artifact}", _IT, f"h-{artifact}"),
        )
    specs = [
        ("queued", {"sources": [{"artifact_id": f"{_IT}-held-1"}, {"source_ref": "no-id"}]}),
        ("running", {"sources": "not-an-array"}),
        ("completed", {"sources": [{"artifact_id": f"{_IT}-free-2"}]}),
    ]
    for i, (status, spec) in enumerate(specs):
        cur.execute(
            "INSERT INTO jobs (job_id, project_id, user_id, job_type, spec, idempotency_key, "
            "status) VALUES (%s, %s, 'u', 'knowledge_embed', %s::jsonb, %s, %s)",
            (f"{_IT}-job-{i}", _IT, json.dumps(spec), f"{_IT}-key-{i}", status),
        )


@pytest.mark.integration
def test_the_sweep_returns_only_the_artifacts_no_unfinished_job_holds():
    """Pins NOT EXISTS over NOT IN (a source without artifact_id) and the
    jsonb_typeof guard (a spec whose sources is not an array)."""
    url = os.environ.get("UMRS_TEST_DB_URL")
    if not url:
        pytest.skip("Need UMRS_TEST_DB_URL (a migrated test database)")
    psycopg2 = pytest.importorskip("psycopg2")
    conn = psycopg2.connect(url)
    try:
        _seed(conn.cursor())
        found = select_stale_artifacts(conn, "m", project_id=_IT, held_by=reindex.HELD_BY)
    finally:
        conn.rollback()
        conn.close()
    assert found == [f"{_IT}-free-2", f"{_IT}-free-3"]
