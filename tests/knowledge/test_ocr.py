"""Tests for the optional OCR fallback (Phase 2).

All tests use a fake backend or monkeypatched environment — no network calls
and no PyMuPDF dependency are required.
"""

from __future__ import annotations

import pytest
from umrs.knowledge import ocr as ocr_module
from umrs.knowledge.ocr import (
    LiteLLMOCRBackend,
    get_ocr_backend,
    resolve_content_with_ocr,
)

PDF_BYTES = b"%PDF-1.4 scanned-no-text-layer"
TEXT_BYTES = b"plain research notes"


class _FakeOCRBackend:
    name = "fake-ocr"

    def __init__(self, text: str):
        self._text = text
        self.calls: list[str] = []

    def ocr(self, filename: str, _content: bytes) -> str:
        self.calls.append(filename)
        return self._text


class TestGetOCRBackend:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("KNOWLEDGE_OCR_ENABLED", raising=False)
        assert get_ocr_backend() is None

    def test_enabled_requires_url_and_model(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_OCR_ENABLED", "true")
        monkeypatch.delenv("KNOWLEDGE_OCR_BASE_URL", raising=False)
        monkeypatch.delenv("KNOWLEDGE_OCR_MODEL", raising=False)
        assert get_ocr_backend() is None

    def test_enabled_returns_litellm_backend(self, monkeypatch):
        monkeypatch.setenv("KNOWLEDGE_OCR_ENABLED", "1")
        monkeypatch.setenv("KNOWLEDGE_OCR_BASE_URL", "http://litellm:11434/v1")
        monkeypatch.setenv("KNOWLEDGE_OCR_MODEL", "glm-ocr")
        backend = get_ocr_backend()
        assert isinstance(backend, LiteLLMOCRBackend)
        assert backend.name == "litellm:glm-ocr"


class TestResolveContentWithOCR:
    def test_plain_text_passes_through_without_ocr(self):
        backend = _FakeOCRBackend("should not be used")
        resolved, engine = resolve_content_with_ocr("notes.txt", TEXT_BYTES, backend)
        assert resolved == TEXT_BYTES
        assert engine is None
        assert backend.calls == []

    def test_no_backend_returns_original(self):
        resolved, engine = resolve_content_with_ocr("scan.pdf", PDF_BYTES, None)
        assert resolved == PDF_BYTES
        assert engine is None

    def test_scanned_pdf_falls_back_to_ocr(self, monkeypatch):
        # Force the embedded-text extractor to yield nothing so the OCR branch runs.
        monkeypatch.setattr(
            ocr_module,
            "resolve_extraction_content",
            lambda _filename, content: content,
        )
        backend = _FakeOCRBackend("transcribed page text")
        resolved, engine = resolve_content_with_ocr("scan.pdf", PDF_BYTES, backend)
        assert resolved == b"transcribed page text"
        assert engine == "fake-ocr"
        assert backend.calls == ["scan.pdf"]

    def test_embedded_text_wins_over_ocr(self, monkeypatch):
        monkeypatch.setattr(
            ocr_module,
            "resolve_extraction_content",
            lambda _filename, _content: b"embedded layer text",
        )
        backend = _FakeOCRBackend("ocr text")
        resolved, engine = resolve_content_with_ocr("doc.pdf", PDF_BYTES, backend)
        assert resolved == b"embedded layer text"
        assert engine is None
        assert backend.calls == []

    def test_empty_ocr_result_keeps_original(self, monkeypatch):
        monkeypatch.setattr(
            ocr_module,
            "resolve_extraction_content",
            lambda _filename, content: content,
        )
        backend = _FakeOCRBackend("   ")
        resolved, engine = resolve_content_with_ocr("scan.pdf", PDF_BYTES, backend)
        assert resolved == PDF_BYTES
        assert engine is None


class TestLiteLLMBackend:
    def test_non_pdf_input_returns_empty(self):
        backend = LiteLLMOCRBackend("http://litellm/v1", "glm-ocr")
        assert backend.ocr("image.png", b"\x89PNG not a pdf") == ""

    def test_pdf_without_rasterizer_returns_empty(self, monkeypatch):
        # Simulate PyMuPDF being unavailable.
        monkeypatch.setattr(ocr_module, "_pdf_to_page_images", lambda _content: [])
        backend = LiteLLMOCRBackend("http://litellm/v1", "glm-ocr")
        assert backend.ocr("scan.pdf", PDF_BYTES) == ""

    def test_transcribes_pages(self, monkeypatch):
        monkeypatch.setattr(
            ocr_module, "_pdf_to_page_images", lambda _content: [b"img1", b"img2"]
        )
        backend = LiteLLMOCRBackend("http://litellm/v1", "glm-ocr")
        monkeypatch.setattr(
            backend, "_transcribe_image", lambda image: f"page:{len(image)}"
        )
        result = backend.ocr("scan.pdf", PDF_BYTES)
        assert result == "page:4\n\npage:4"

    def test_transcription_error_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            ocr_module, "_pdf_to_page_images", lambda _content: [b"img1"]
        )
        backend = LiteLLMOCRBackend("http://litellm/v1", "glm-ocr")

        def _boom(_image):
            raise TimeoutError("slow model")

        monkeypatch.setattr(backend, "_transcribe_image", _boom)
        assert backend.ocr("scan.pdf", PDF_BYTES) == ""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
