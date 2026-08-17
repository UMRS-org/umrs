"""KNOWLEDGE_REINDEX dispatcher wire-up tests (S10 follow-up)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from umrs.jobs import reindex
from umrs.jobs.models import JobType
from umrs.jobs.reindex import (
    KnowledgeReindexBackend,
    KnowledgeReindexStatusChecker,
    validate_knowledge_reindex_spec,
)
from umrs.knowledge.embedding_metadata import EmbeddingMetadataError

SPEC = {
    "target_model": "bge-m3",
    "index_name": "knowledge_items_v1_hnsw",
    "database": "umrs_db",
    "table": "knowledge_artifacts_embeddings",
    "embedding_dim": 1024,
    "batch_limit": 250,
}


# ── payload contract ──────────────────────────────────────────────────────


def test_spec_valid_full_zoo():
    out = validate_knowledge_reindex_spec(dict(SPEC))
    assert out["database"] == "umrs_db"
    assert out["target_model"] == "bge-m3"
    assert out["embedding_dim"] == 1024
    assert out["batch_limit"] == 250


@pytest.mark.parametrize(
    ("delta", "match"),
    [
        ({"database": ""}, "spec.database"),
        ({"table": " "}, "spec.table"),
        ({"target_model": " "}, "target_model"),
        ({"index_name": ""}, "index_name"),
        ({"embedding_dim": 0}, "embedding_dim"),
        ({"embedding_dim": -1}, "embedding_dim"),
        ({"batch_limit": 0}, "batch_limit"),
    ],
)
def test_spec_rejects_bad_fields(delta, match):
    bad = {**SPEC, **delta}
    with pytest.raises((EmbeddingMetadataError, Exception), match=match):
        validate_knowledge_reindex_spec(bad)


def test_spec_defaults_applied():
    out = validate_knowledge_reindex_spec(
        {"target_model": "m", "index_name": "i", "database": "d", "table": "t"}
    )
    assert out["batch_limit"] == 500
    assert out["embedding_dim"] is None


# ── backend execution ─────────────────────────────────────────────────────


class _Cur:
    def __init__(self, rows=(), rowcount=0):
        self.statements: list[tuple] = []
        self._rows = list(rows)
        self._rowcount = rowcount

    def execute(self, q, p=None):
        self.statements.append((" ".join(str(q).split()), p))

    def fetchall(self):
        return list(self._rows)

    @property
    def rowcount(self):
        return self._rowcount


class _Conn:
    def __init__(self, cur):
        self._cur = cur

    def cursor(self):
        return self._cur


@contextlib.contextmanager
def _conn_scope(cur):
    yield _Conn(cur)


async def _run_backend(backend, spec):
    job = SimpleNamespace(job_id="j-1", spec=spec)
    attempt = SimpleNamespace(attempt_id="a-1")
    return await backend.submit_job(job, attempt)


def _stub_compute(monkeypatch) -> dict:
    called = {}

    def _maintain(database, table, rebuild=False, index_name=None):
        called.update(database=database, table=table, rebuild=rebuild, index_name=index_name)
        return {"action": "rebuild", "index_name": "knowledge_items_v1_hnsw"}

    import umrs.server.sql as server_sql

    monkeypatch.setattr(server_sql, "maintain_pgvector_index", _maintain)
    return called


def test_submit_sweep_maintain_stamp(monkeypatch):
    cur = _Cur(rows=[("art-1",), ("art-2",)], rowcount=2)
    monkeypatch.setattr(reindex, "_artifact_conn", lambda: _conn_scope(cur))
    called = _stub_compute(monkeypatch)

    ref = _sync(_run_backend(KnowledgeReindexBackend(), dict(SPEC)))
    assert called["database"] == "umrs_db"
    assert called["rebuild"] is True
    assert "swept=2" in ref and "stamped=2" in ref

    (select_first, _), (update_second, update_params) = cur.statements
    assert select_first.startswith("SELECT artifact_id")
    assert update_second.startswith("UPDATE knowledge_artifacts")
    assert update_params[0] == SPEC["target_model"]
    assert update_params[-1] == ["art-1", "art-2"]


def test_submit_no_stale_ref_and_no_compute(monkeypatch):
    cur = _Cur(rows=[], rowcount=0)
    monkeypatch.setattr(reindex, "_artifact_conn", lambda: _conn_scope(cur))
    import umrs.server.sql as server_sql

    def _boom(*_a, **_k):  # pragma: no cover - must never be called here
        raise AssertionError("maintain must not run with zero stale rows")

    monkeypatch.setattr(server_sql, "maintain_pgvector_index", _boom)
    ref = _sync(_run_backend(KnowledgeReindexBackend(), dict(SPEC)))
    assert "swept=0" in ref and "stamped=0" in ref


def _sync(coro):
    import asyncio

    return asyncio.run(coro)


# ── status checker ────────────────────────────────────────────────────────


def _make_attempt(ref):
    return (
        SimpleNamespace(attempt_id="a-1", backend_ref=ref),
        SimpleNamespace(job_id="j-1", spec={}),
    )


def test_checker_succeeds_on_valid_ref():
    attempt, job = _make_attempt(
        "knowledge_reindex|database=umrs_db|table=kae|index=ix|model=bge|swept=4|stamped=4"
    )
    result = _sync(KnowledgeReindexStatusChecker().check_status(attempt, job))
    assert result.status == "succeeded"
    assert result.metrics["swept"] == "4"


def test_checker_fails_on_missing_or_malformed_ref():
    attempt, job = _make_attempt(None)
    assert _sync(KnowledgeReindexStatusChecker().check_status(attempt, job)).status == "failed"
    attempt, job = _make_attempt("custom|foo=bar")
    assert _sync(KnowledgeReindexStatusChecker().check_status(attempt, job)).status == "failed"


# ── wiring ────────────────────────────────────────────────────────────────


def test_spec_validator_registered():
    from umrs.server.job_tracking_handler import _JOB_SPEC_VALIDATORS

    assert JobType.KNOWLEDGE_REINDEX in _JOB_SPEC_VALIDATORS


def test_factory_and_checker_registered():
    src = (
        Path(__file__).resolve().parents[1] / "src" / "umrs" / "jobs" / "adapters.py"
    ).read_text()
    assert "KnowledgeReindexBackend" in src
    assert "JobType.KNOWLEDGE_REINDEX: {\n        return knowledge_reindex_backend" in src or (
        "return knowledge_reindex_backend" in src
    )
    assert "KnowledgeReindexStatusChecker" in src
