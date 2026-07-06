"""Optional OCR fallback for scanned binary documents (Phase 2).

The knowledge extraction pipeline (:func:`extract.resolve_extraction_content`)
uses ``pypdf`` / ``python-pptx`` to pull embedded text out of PDF/PPTX bytes.
A *scanned* document has no embedded text layer, so that path yields an empty
string and the artifact contributes nothing to the knowledge graph. This module
provides an **opt-in** OCR fallback for exactly that case.

Invariants
----------
* OCR is **disabled by default**. It only activates when
  ``KNOWLEDGE_OCR_ENABLED`` is truthy in the environment.
* When enabled, OCR routes page images to a vision-capable model served
  *inside* the platform through LiteLLM (a platform-internal service). It never
  reaches the public internet, so the offline inference invariant
  (``TRANSFORMERS_OFFLINE=1``, no egress from inference runtimes) is preserved.
* The fallback is best-effort: any failure (missing rasterizer, network error,
  disabled backend) leaves the original bytes untouched — extraction degrades
  gracefully to "no text" rather than raising.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Protocol, runtime_checkable

from .extract import _is_binary_document_bytes, resolve_extraction_content

logger = logging.getLogger(__name__)

DEFAULT_OCR_PROMPT = (
    "Transcribe all text visible in this document page exactly. "
    "Preserve reading order. Output only the transcribed text."
)


@runtime_checkable
class OCRBackend(Protocol):
    """Renders scanned document bytes to plain text."""

    name: str

    def ocr(self, filename: str, content: bytes) -> str:
        """Return transcribed text for *content*, or ``""`` on failure."""
        ...


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def _pdf_to_page_images(content: bytes) -> list[bytes]:
    """Rasterize PDF pages to PNG bytes using PyMuPDF, if available.

    Returns an empty list when PyMuPDF is not installed or rasterization
    fails — the caller then skips OCR gracefully.
    """
    try:
        import fitz  # type: ignore  # PyMuPDF
    except ImportError:
        logger.debug("PyMuPDF not installed; OCR rasterization unavailable")
        return []

    images: list[bytes] = []
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        try:
            for page in doc:
                pixmap = page.get_pixmap(dpi=150)
                images.append(pixmap.tobytes("png"))
        finally:
            doc.close()
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - depends on fitz
        logger.warning("PDF rasterization failed: %s", exc)
        return []
    return images


class LiteLLMOCRBackend:
    """OCR backend that calls a vision model over a platform-internal LiteLLM.

    The endpoint is OpenAI chat-completions compatible. Only PDF input is
    supported (rasterized to page images); other binary types return ``""``.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        prompt: str = DEFAULT_OCR_PROMPT,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._prompt = prompt
        self._timeout = timeout
        self.name = f"litellm:{model}"

    def _transcribe_image(self, image_png: bytes) -> str:
        data_uri = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(  # noqa: S310 - internal LiteLLM URL only
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"] or ""

    def ocr(self, filename: str, content: bytes) -> str:
        if not content.startswith(b"%PDF"):
            logger.debug("OCR backend only supports PDF input; skipping %s", filename)
            return ""
        images = _pdf_to_page_images(content)
        if not images:
            return ""
        pages: list[str] = []
        for image in images:
            try:
                pages.append(self._transcribe_image(image))
            except (urllib.error.URLError, KeyError, ValueError, TimeoutError) as exc:
                logger.warning("OCR transcription failed for %s: %s", filename, exc)
                return ""
        return "\n\n".join(p for p in pages if p.strip())


def get_ocr_backend() -> OCRBackend | None:
    """Return a configured OCR backend, or ``None`` when OCR is disabled.

    OCR stays off unless ``KNOWLEDGE_OCR_ENABLED`` is truthy. When enabled, the
    LiteLLM base URL and model are read from the environment; both must be set.
    """
    if not _truthy(os.environ.get("KNOWLEDGE_OCR_ENABLED")):
        return None
    base_url = os.environ.get("KNOWLEDGE_OCR_BASE_URL", "").strip()
    model = os.environ.get("KNOWLEDGE_OCR_MODEL", "").strip()
    if not base_url or not model:
        logger.warning(
            "KNOWLEDGE_OCR_ENABLED is set but base URL/model are missing; OCR disabled"
        )
        return None
    return LiteLLMOCRBackend(
        base_url=base_url,
        model=model,
        api_key=os.environ.get("KNOWLEDGE_OCR_API_KEY") or None,
    )


def resolve_content_with_ocr(
    filename: str,
    content: bytes,
    backend: OCRBackend | None,
) -> tuple[bytes, str | None]:
    """Resolve document bytes to text, using OCR only when normal extraction fails.

    Returns ``(resolved_bytes, ocr_engine)`` where ``ocr_engine`` is the backend
    name when OCR produced the text, else ``None``. Non-binary content and
    documents with an embedded text layer never trigger OCR.
    """
    resolved = resolve_extraction_content(filename, content)
    if resolved is not content:
        # Embedded text layer was successfully extracted.
        return resolved, None
    if backend is None or not _is_binary_document_bytes(filename, content):
        return content, None
    text = backend.ocr(filename, content)
    if text.strip():
        return text.encode("utf-8"), backend.name
    return content, None
