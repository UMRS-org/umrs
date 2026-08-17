"""Embedding-metadata helper tests (stability S10)."""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.jobs.models import DEFAULT_BACKOFFS, JobType
from umrs.knowledge.embedding_metadata import (
    EmbeddingMetadataError,
    mark_embeddings_stale,
    select_stale_artifacts,
    stamp_embeddings_current,
    validate_reindex_payload,
)


class _FakeCursor:
    def __init__(self, rows: list = (), rowcount: int = 0):
        self.statements: list[tuple] = []
        self._rows = list(rows)
        self._rowcount = rowcount
        self._last = ""

    def execute(self, query, params=None):
        self.statements.append((query, params))
        self._last = query

    def fetchall(self):
        return list(self._rows)

    @property
    def rowcount(self):
        return self._rowcount


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


# ── payload contract ──────────────────────────────────────────────────────


def test_payload_contract_requires_both_keys():
    assert validate_reindex_payload({"target_model": "m", "index_name": "idx"}) == {
        "target_model": "m",
        "index_name": "idx",
    }
    with pytest.raises(EmbeddingMetadataError, match="missing"):
        validate_reindex_payload({"target_model": "m"})
    with pytest.raises(EmbeddingMetadataError, match="target_model"):
        validate_reindex_payload({"target_model": "  ", "index_name": "i"})
    with pytest.raises(EmbeddingMetadataError, match="index_name"):
        validate_reindex_payload({"target_model": "m", "index_name": ""})
    with pytest.raises(EmbeddingMetadataError, match="dict"):
        validate_reindex_payload("not-a-dict")


def test_job_type_registered_with_conservative_backoff():
    assert JobType.KNOWLEDGE_REINDEX.value == "knowledge_reindex"
    backoff = DEFAULT_BACKOFFS[JobType.KNOWLEDGE_REINDEX]
    assert backoff["max_attempts"] == 3
    assert len(backoff["delays"]) == 3


# ── sweep ─────────────────────────────────────────────────────────────────


def test_select_stale_targets_model_drift_or_uncurrent():
    conn = _FakeConn(_FakeCursor(rows=[("art-1",), ("art-2",)]))
    stale = select_stale_artifacts(conn, "bge-m3")
    assert stale == ["art-1", "art-2"]
    query, params = conn.cursor().statements[0]
    assert "embedding_status <> 'current'" in query
    assert "embedding_model <> %s" in query
    assert "status = 'active'" in query
    assert params == ("bge-m3", 500)


def test_select_rejects_empty_model():
    with pytest.raises(EmbeddingMetadataError):
        select_stale_artifacts(_FakeConn(_FakeCursor()), "  ")


# ── stamp ─────────────────────────────────────────────────────────────────


def test_stamp_records_model_dim_time_and_status():
    cur = _FakeCursor(rowcount=3)
    conn = _FakeConn(cur)
    done = stamp_embeddings_current(
        conn, ["a1", "a2", "a3"], model="bge-m3", dim=1024, embedded_at="2026-08-17T00:00:00Z"
    )
    assert done == 3
    query, params = cur.statements[0]
    assert "embedding_status = 'current'" in query
    assert params == (
        "bge-m3",
        1024,
        "2026-08-17T00:00:00Z",
        "2026-08-17T00:00:00Z",
        ["a1", "a2", "a3"],
    )


def test_stamp_noop_on_empty_and_validates():
    assert stamp_embeddings_current(_FakeConn(_FakeCursor()), [], model="m", dim=8) == 0
    with pytest.raises(EmbeddingMetadataError, match="dim"):
        stamp_embeddings_current(_FakeConn(_FakeCursor()), ["a"], model="m", dim=0)
    with pytest.raises(EmbeddingMetadataError, match="model"):
        stamp_embeddings_current(_FakeConn(_FakeCursor()), ["a"], model=" ", dim=8)


# ── stale marking after model change ───────────────────────────────────────


def test_mark_stale_scoped_to_other_models():
    cur = _FakeCursor(rowcount=7)
    conn = _FakeConn(cur)
    assert mark_embeddings_stale(conn, previous_model="bge-m3") == 7
    query, params = cur.statements[0]
    assert "embedding_model <> %s" in query
    assert params[1] == "bge-m3"


def test_mark_stale_full_reindex():
    cur = _FakeCursor(rowcount=12)
    conn = _FakeConn(cur)
    assert mark_embeddings_stale(conn) == 12
    query, _ = cur.statements[0]
    assert "embedding_status = 'current'" in query


# ── migration 0073 ────────────────────────────────────────────────────────


def test_migration_0073_columns_and_index():
    sql = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "umrs"
        / "migrations"
        / "0073_artifact_embedding_metadata.sql"
    ).read_text()
    for fragment in (
        "ADD COLUMN IF NOT EXISTS embedding_model",
        "ADD COLUMN IF NOT EXISTS embedding_dim",
        "ADD COLUMN IF NOT EXISTS embedded_at",
        "ADD COLUMN IF NOT EXISTS embedding_status",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_artifacts_embedding_status",
    ):
        assert fragment in sql, f"0073 must add {fragment} idempotently"
