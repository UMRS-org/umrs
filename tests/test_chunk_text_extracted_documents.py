"""Raw-slice readers decode an artifact as the embed job chunked it (ADR-093
D-12, review chunk-data-12).

The embed job chunks a PDF — and a PPTX recognised by its source name —
through the catalog's text extractor, and drops NUL. Hydrate's artifact
slice (the export's and the import's fallback too) decoded the raw bytes,
so a PDF chunk never verified and only its 500-character excerpt was served.
"""

from __future__ import annotations

from _extracted_documents import chunk_content, embed_text, pdf_bytes, pptx_bytes
from umrs.knowledge.retrieval.hydrate import artifact_slice, chunked_text, passage_text

SENTENCES = "Interleukin six rises after exercise. It falls during rest periods."


def _first_sentence(document: str) -> tuple[int, int]:
    end = document.index(".") + 1
    return 0, end


def test_a_pdf_chunk_verifies_against_the_extracted_text():
    raw = pdf_bytes(SENTENCES)
    document = embed_text(raw, "paper.pdf")
    content = chunk_content(document, *_first_sentence(document))
    piece = artifact_slice(content, "k", {}, lambda _ref: raw, "paper.pdf")
    assert piece == "Interleukin six rises after exercise."


def test_a_pdf_passage_is_the_artifact_slice_not_the_excerpt():
    raw = pdf_bytes(SENTENCES)
    document = embed_text(raw, "paper.pdf")
    content = chunk_content(document, 0, len(document))
    text, source = passage_text(content, "k", {}, lambda _ref: raw, None, "paper.pdf")
    assert (text, source) == (document, "artifact")


def test_a_pptx_is_extracted_by_its_source_name(monkeypatch):
    """The name decides: ``_is_binary_document_bytes`` takes a zip for a
    deck only under a ``.pptx`` name. (The catalog's own PPTX extractor is
    stood in for here; ``test_the_catalog_extractor_reads_a_deck`` runs it.)"""

    def extract(_name: str, raw: bytes) -> bytes:
        return SENTENCES.encode() if raw.startswith(b"PK") else raw

    monkeypatch.setattr("umrs.knowledge.extract.resolve_extraction_content", extract)
    raw = pptx_bytes(SENTENCES)
    document = embed_text(raw, "talk.pptx")
    content = chunk_content(document, *_first_sentence(document))
    assert artifact_slice(content, "k", {}, lambda _ref: raw, "talk.pptx") is not None
    # without its name a deck is a zip the embed job refuses: nothing verifies
    assert artifact_slice(content, "k", {}, lambda _ref: raw, "") is None


def test_the_catalog_extractor_reads_a_deck(monkeypatch):
    """python-pptx 1.0 refuses ``presentation.slides[:n]``: every deck read
    as "extract_failed" and its embed failed by name (found 2026-10-10)."""
    from umrs.data_catalog.extract import _extract_pptx_detailed

    monkeypatch.setenv("UMRS_DATA_CATALOG_MAX_DOCUMENT_PAGES", "1")
    text, flags = _extract_pptx_detailed(pptx_bytes(SENTENCES))
    assert (text, flags) == (SENTENCES, [])
    assert embed_text(pptx_bytes(SENTENCES), "talk.pptx") == SENTENCES


def test_text_bytes_decode_as_the_embed_job_reads_them():
    raw = b"  Line one.\x00 Line two.\xff\n"
    assert chunked_text(raw, "notes.txt") == embed_text(raw, "notes.txt") == "Line one. Line two."


def test_a_pdf_without_extractable_text_has_no_slice():
    assert chunked_text(b"%PDF-1.4\n", "scan.pdf") is None
