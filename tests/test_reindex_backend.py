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
    """``rows`` answers every fetchall; ``batches`` answers successive ones."""

    def __init__(self, rows=(), rowcount=0, batches=None):
        self.statements: list[tuple] = []
        self._rows = list(rows)
        self._batches = [list(b) for b in batches] if batches else None
        self._rowcount = rowcount

    def execute(self, q, p=None):
        self.statements.append((" ".join(str(q).split()), p))

    def fetchall(self):
        if self._batches is not None:
            return self._batches.pop(0) if self._batches else []
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
    job = SimpleNamespace(job_id="j-1", spec=spec, user_id="u-1", priority=0)
    attempt = SimpleNamespace(attempt_id="a-1")
    return await backend.submit_job(job, attempt)


def _stub_compute(monkeypatch) -> dict:
    called = {}

    def _maintain(database, table, rebuild=False, index_name=None):
        called.update(database=database, table=table, rebuild=rebuild, index_name=index_name)
        return {"action": "rebuild", "index_name": "knowledge_items_v1_hnsw"}

    import umrs.shared.pgvector_index as server_sql  # canonical home post-ADR-053

    monkeypatch.setattr(server_sql, "maintain_pgvector_index", _maintain)
    return called


def _stub_job_service(monkeypatch) -> list:
    submitted: list = []

    class _Svc:
        def submit_job(self, submission):
            submitted.append(submission)
            return f"job-embed-{len(submitted)}"

    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: _Svc())
    return submitted


def _run_reembed(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "/srv/models/bge-m3")
    cur = _Cur(
        batches=[
            [("art-1",), ("art-2",), ("art-3",)],
            [("art-1", "p1", "k1"), ("art-2", "p1", "k2"), ("art-3", "p2", "k3")],
        ]
    )
    monkeypatch.setattr(reindex, "_artifact_conn", lambda: _conn_scope(cur))
    called = _stub_compute(monkeypatch)
    submitted = _stub_job_service(monkeypatch)
    ref = _sync(_run_backend(KnowledgeReindexBackend(), dict(SPEC)))
    return ref, cur, called, submitted


def test_submit_sweeps_reembeds_and_never_stamps(monkeypatch):
    """Until 2026-09-24 this job stamped swept artifacts current with the
    target model WITHOUT recomputing a vector. It now enqueues KNOWLEDGE_EMBED
    per project (the embed job stamps after writing) and issues no UPDATE."""
    ref, cur, called, _ = _run_reembed(monkeypatch)
    assert ("swept=3" in ref, "reembed_jobs=2" in ref, called["rebuild"]) == (True, True, True)
    assert "model=/srv/models/bge-m3" in ref  # resolved to the site embedder
    assert not [q for q, _ in cur.statements if q.startswith("UPDATE")]
    # sweep compares against the site model string the embed job stamps
    assert cur.statements[0][1][0] == "/srv/models/bge-m3"


def test_reembed_jobs_are_per_project_with_their_sources(monkeypatch):
    _, _, _, submitted = _run_reembed(monkeypatch)
    by_project = {s.project_id: s for s in submitted}
    assert set(by_project) == {"p1", "p2"}
    assert {s.job_type for s in submitted} == {JobType.KNOWLEDGE_EMBED}
    assert by_project["p1"].spec["sources"] == [
        {"artifact_id": "art-1", "source_ref": "k1"},
        {"artifact_id": "art-2", "source_ref": "k2"},
    ]
    assert by_project["p1"].spec["reason"] == "knowledge_reindex"


@pytest.mark.parametrize("site", ["", "/srv/models/e5-base-v2"])
def test_target_model_must_be_the_site_embedder(monkeypatch, site):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", site)
    cur = _Cur(rows=[("art-1",)])
    monkeypatch.setattr(reindex, "_artifact_conn", lambda: _conn_scope(cur))
    with pytest.raises(EmbeddingMetadataError, match="not this site's embedder"):
        _sync(_run_backend(KnowledgeReindexBackend(), dict(SPEC)))
    assert cur.statements == []  # refused before any sweep


def test_submit_no_stale_ref_and_no_compute(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "/srv/models/bge-m3")
    cur = _Cur(rows=[], rowcount=0)
    monkeypatch.setattr(reindex, "_artifact_conn", lambda: _conn_scope(cur))
    import umrs.shared.pgvector_index as server_sql  # canonical home post-ADR-053

    def _boom(*_a, **_k):  # pragma: no cover - must never be called here
        raise AssertionError("maintain must not run with zero stale rows")

    monkeypatch.setattr(server_sql, "maintain_pgvector_index", _boom)
    ref = _sync(_run_backend(KnowledgeReindexBackend(), dict(SPEC)))
    assert "swept=0" in ref and "reembed_jobs=0" in ref


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
        "knowledge_reindex|database=umrs_db|table=kae|index=ix|model=bge|swept=4|reembed_jobs=2"
    )
    result = _sync(KnowledgeReindexStatusChecker().check_status(attempt, job))
    assert result.status == "succeeded"
    assert (result.metrics["swept"], result.metrics["reembed_jobs"]) == ("4", "2")


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
