"""Cycle-3 memfix #3 (2026-10-22): a cut text whose prefix makes one or two
chunks still writes its first chunk and stamps the truncation.

Found by the mem review (``.stab-cycle3/memreview/memreview-v2-1``): the
embed job reads a text's first ``_embed_read_bytes()`` and drops the last two
chunks of a cut text. A single-line 1.2 MB FASTA is one BERT word ([UNK]),
so its 512 KiB prefix made one chunk, ``chunks[:-2]`` left none, and
``_write_embedded_chunks`` returned before the stamp with nothing in
``failed_items`` or ``skipped``: the job reported success and the artifact
had no vector and no stamp. The base wrote the one whole-text chunk.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from tests.test_knowledge_reindex_rechunk import _Store
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge import embed_worker as ew

MODEL = "/srv/models/e5-base-v2"


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", MODEL)
    for name in (
        "UMRS_KNOWLEDGE_EMBED_MAX_CHUNKS",
        "UMRS_KNOWLEDGE_CHUNK_SCHEME",
        "UMRS_EMBEDDING_SANDBOX",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(("count", "kept"), [(1, 1), (2, 1), (3, 1), (4, 2), (10, 8)])
def test_a_cut_text_keeps_its_first_chunk_and_reports_the_truncation(count, kept):
    chunks = list(range(count))
    pairs, total = ka._kept_chunks(chunks, chunks, cap=128, cut=True)
    assert [c for c, _ in pairs] == list(range(kept))
    assert total > len(pairs)


def test_an_empty_cut_text_still_writes_nothing():
    assert ka._kept_chunks([], [], cap=128, cut=True) == ([], 1)


class _OneWordEmbedder:
    """No tokenizer: the no-NLP chunker cuts one chunk per text, as BERT's
    WordPiece does for a single-line sequence (one [UNK] word)."""

    client = type("C", (), {"tokenizer": None})()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]


def test_a_single_line_fasta_cut_at_the_budget_is_written_and_stamped(monkeypatch):
    stamps: list[Any] = []

    async def stamp(_store: Any, artifact_id: str, _model: str, _dim: int, chunks: Any = None):
        stamps.append((artifact_id, chunks))

    monkeypatch.setattr(ka, "_stamp_embedded", stamp)
    prefix = ">chr1 synthetic\n" + "ACGTTGCA" * (ka._embed_read_bytes() // 8)
    (embedded,) = ew.chunk_and_embed(
        _OneWordEmbedder(), [prefix], prefix_model_name=MODEL, max_tokens=256
    )
    assert len(embedded["chunks"]) == 1
    store, result = _Store(), ka.KnowledgeEmbedResult(cut_texts={"art-1"})
    asyncio.run(ka._write_embedded_chunks(store, "art-1", "p1", "", embedded, result))
    assert result.nodes_embedded == 1 and len(store.vectors) == 1
    ((artifact_id, chunk_stamp),) = stamps
    assert artifact_id == "art-1"
    assert chunk_stamp.total > chunk_stamp.indexed == 1
    assert result.chunks_truncated == 1
