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
