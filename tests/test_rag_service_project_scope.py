"""Fail-closed project→index resolution in the RAG service (ADR-027 P0).

An unmapped project must never silently fall back to another project's
embedding table: 404 by default, legacy fallback only behind
UMRS_RAG_LEGACY_TABLE_FALLBACK.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from umrs.api import rag_service


@pytest.fixture
def fake_store(monkeypatch):
    store = {
        "/a": {"xp_name": "spring8", "store_embeddings_flag": True},
        "/b": {"xp_name": "other", "store_embeddings_flag": True},
        "/c": {"xp_name": "nocfg", "store_embeddings_flag": False},
    }
    monkeypatch.setattr(rag_service, "DB_MODEL_STORE", store)
    monkeypatch.delenv("UMRS_RAG_LEGACY_TABLE_FALLBACK", raising=False)
    return store


class TestResolveXpForProject:
    def test_matching_xp_returns_name(self, fake_store):
        assert rag_service._resolve_xp_for_project("spring8") == "spring8"

    def test_empty_project_returns_none(self, fake_store):
        assert rag_service._resolve_xp_for_project("") is None

    def test_unmapped_project_raises_404(self, fake_store):
        with pytest.raises(HTTPException) as exc_info:
            rag_service._resolve_xp_for_project("spring9")
        assert exc_info.value.status_code == 404
        assert "No RAG index for project 'spring9'" in exc_info.value.detail

    def test_embedding_disabled_entry_is_no_match(self, fake_store):
        with pytest.raises(HTTPException) as exc_info:
            rag_service._resolve_xp_for_project("nocfg")
        assert exc_info.value.status_code == 404

    def test_legacy_fallback_returns_none(self, fake_store, monkeypatch):
        monkeypatch.setenv("UMRS_RAG_LEGACY_TABLE_FALLBACK", "true")
        assert rag_service._resolve_xp_for_project("spring9") is None


class TestResolveEntryForUpload:
    def test_matching_xp_returns_entry(self, fake_store):
        entry = rag_service._resolve_entry_for_upload("spring8")
        assert entry is fake_store["/a"]

    def test_unmapped_project_returns_none(self, fake_store):
        assert rag_service._resolve_entry_for_upload("spring9") is None

    def test_embedding_disabled_entry_is_no_match(self, fake_store):
        assert rag_service._resolve_entry_for_upload("nocfg") is None


class _FakeCursor:
    def __init__(self):
        self.executed: list = []

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def close(self):
        pass


class _FakeConn:
    def __init__(self):
        self.cur = _FakeCursor()
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        pass


class _FakeDataBase:
    def __init__(self, conn, has_content=True, has_embedding=True):
        self._conn = conn
        self.db_pool = _FakePool(conn)
        self._has_content = has_content
        self._has_embedding = has_embedding

    def has_column(self, table, column):
        if column == "_content":
            return self._has_content
        if column == "embedding":
            return self._has_embedding
        return False


class TestPersistChunksReturn:
    """ADR-027 P1a: _persist_chunks returns inserted chunk rows."""

    def _entry(self, conn, **kwargs):
        return {"data_base": _FakeDataBase(conn, **kwargs), "table_name": "t"}

    def _call(self, entry):
        return rag_service._persist_chunks(
            entry,
            s3_base="uploads/notes.txt",
            chunks=[
                {
                    "text": "alpha beta gamma",
                    "start": 0,
                    "end": 16,
                    "md5": "m0",
                }
            ],
            embeddings=[[0.1, 0.2]],
            file_checksum="cs",
            date_str="2026-08-22",
            time_str="00:00:00",
            model_name="m",
        )

    def test_inserted_rows_have_chunk_detail_fields(self):
        conn = _FakeConn()
        inserted = self._call(self._entry(conn))
        assert len(inserted) == 1
        row = inserted[0]
        assert row["s3_object"] == "uploads/notes.txt#chunk0"
        assert row["chunk_start"] == 0
        assert row["chunk_end"] == 16
        assert row["chunk_md5"] == "m0"
        assert row["excerpt"] == "alpha beta gamma"
        assert conn.committed

    def test_table_without_embedding_inserts_nothing(self):
        conn = _FakeConn()
        inserted = self._call(self._entry(conn, has_embedding=False))
        assert inserted == []
        assert conn.committed
