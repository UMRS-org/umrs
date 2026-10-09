"""The RAG service says which embedder it loaded, and that RAG_EMBEDDING_MODEL is ignored.

Defect 3 of defect-wave-2026-10-08: every deployment artifact sets
``RAG_EMBEDDING_MODEL``, which nothing reads (the service reads
``UMRS_RAG_EMBEDDING_MODEL``), and ``/rag/stats`` reported a label —
``intfloat/e5-base-v2`` unless the right variable was set — instead of the model
actually loaded. Which name is canonical is a maintainer decision (renaming would
make the dead variable live on deploy); this makes the state visible.
"""

from __future__ import annotations

import logging

import pytest
from umrs.api import rag_service

_STORE = {
    "/a": {
        "modality_name": "text",
        "store_embeddings_flag": True,
        "model": {"name": "/models/e5-base-v2"},
    }
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("UMRS_RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
    # No knowledge embedder staged unless a test stages one.
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", str(tmp_path / "absent"))
    monkeypatch.setattr(rag_service, "_model_name_from_entry", lambda m: (m or {}).get("name"))


def test_the_staged_knowledge_embedder_comes_before_the_user_config(monkeypatch, tmp_path):
    """The query encoder must be the model the spine's vectors were written with."""
    staged = tmp_path / "e5-base-v2"
    staged.mkdir()
    (staged / "config.json").write_text("{}")
    (staged / "model.safetensors").write_bytes(b"x")
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", str(staged))
    assert rag_service._resolve_embedding_model(_STORE) == (
        str(staged),
        "knowledge text embedder",
    )


def test_explicit_variable_wins_and_is_named():
    import os

    os.environ["UMRS_RAG_EMBEDDING_MODEL"] = "/models/me5b"
    try:
        assert rag_service._resolve_embedding_model(_STORE) == (
            "/models/me5b",
            "UMRS_RAG_EMBEDDING_MODEL",
        )
    finally:
        del os.environ["UMRS_RAG_EMBEDDING_MODEL"]


def test_user_config_is_the_next_source():
    assert rag_service._resolve_embedding_model(_STORE) == (
        "/models/e5-base-v2",
        "user config (first text modality)",
    )


def test_built_in_default_is_named_as_such():
    assert rag_service._resolve_embedding_model({})[1] == "built-in default"


def test_the_dead_variable_is_reported_not_silently_ignored(monkeypatch, caplog):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "intfloat/e5-base-v2")
    with caplog.at_level(logging.WARNING, logger=rag_service.logger.name):
        model, source = rag_service._resolve_embedding_model(_STORE)
    assert (model, source) == ("/models/e5-base-v2", "user config (first text modality)")
    assert "RAG_EMBEDDING_MODEL" in caplog.text and "not read" in caplog.text


class _Loaded:
    model_name = "/models/e5-base-v2"

    class client:  # noqa: N801 — mirrors SentenceTransformerEmbeddings.client
        @staticmethod
        def get_sentence_embedding_dimension():
            return 768


def test_stats_report_the_loaded_model_and_where_it_came_from(monkeypatch):
    monkeypatch.setattr(rag_service, "EMBEDDINGS", _Loaded())
    monkeypatch.setattr(rag_service, "EMBEDDINGS_SOURCE", "user config (first text modality)")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "intfloat/e5-base-v2")
    stats = rag_service._embedder_stats()
    assert stats["embedding_model"] == "/models/e5-base-v2"
    assert stats["embedding_model_source"] == "user config (first text modality)"
    assert stats["embedding_dimensions"] == 768
    assert stats["ignored_settings"] == {"RAG_EMBEDDING_MODEL": "intfloat/e5-base-v2"}


def test_stats_without_a_loaded_model_say_so(monkeypatch):
    monkeypatch.setattr(rag_service, "EMBEDDINGS", None)
    stats = rag_service._embedder_stats()
    assert stats["embedding_model"] is None
    assert "ignored_settings" not in stats
