"""Text and metadata extraction for catalog ingestion (MVP, no GPU)."""

from __future__ import annotations

import csv
import io
import json
import os
from typing import Any

from .modality import detect_modality


def extract_content(
    filename: str,
    content: bytes,
) -> tuple[str, str, list[str]]:
    """
    Extract searchable text and a short summary from object bytes.

    Returns:
        (summary, extracted_text, tags)
    """
    modality = detect_modality(filename, content)
    if modality == "text":
        text = _decode_text(content)
        return _summarize(text), text, []
    if modality == "structured":
        text, tags = _extract_structured(filename, content)
        return _summarize(text), text, tags
    if modality == "document":
        text, tags = _extract_document_detailed(filename, content)
        return _summarize(text), text, tags
    if modality == "image":
        text = _extract_image_metadata(content)
        summary = text or f"Image file {filename}"
        return summary[:500], text, []
    if modality == "notebook":
        from ..knowledge.parse_notebook import parse_notebook

        result = parse_notebook(filename, content)
        return result.summary, result.extracted_text, result.tags
    if modality == "code":
        from ..knowledge.parse_code import parse_code

        result = parse_code(filename, content)
        return result.summary, result.extracted_text, result.tags
    return f"{modality} file {filename}", "", []


def _decode_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return ""


def _extract_structured(filename: str, content: bytes) -> tuple[str, list[str]]:
    text = _decode_text(content)
    tags: list[str] = []
    lower = filename.lower()
    if lower.endswith(".json"):
        try:
            payload = json.loads(text)
            tags.extend(_json_keys(payload, max_keys=12))
            text = json.dumps(payload, indent=2)[:50_000]
        except json.JSONDecodeError:
            pass
    elif lower.endswith((".csv", ".tsv")):
        delimiter = "\t" if lower.endswith(".tsv") else ","
        try:
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
            if rows:
                header = rows[0][:12]
                tags.extend(header)
                for col_idx in range(min(3, len(header))):
                    seen: set[str] = set()
                    for row in rows[1:201]:
                        if col_idx >= len(row):
                            continue
                        value = row[col_idx].strip()
                        if not value or value in seen:
                            continue
                        seen.add(value)
                        tags.append(value)
                        if len(seen) >= 5:
                            break
                text = "\n".join(",".join(row) for row in rows[:200])
        except csv.Error:
            pass
    return text, [t for t in tags if t][:24]


def _json_keys(payload: Any, *, max_keys: int, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.append(f"{prefix}{key}" if prefix else str(key))
            if len(keys) >= max_keys:
                break
            keys.extend(_json_keys(value, max_keys=max_keys - len(keys), prefix=f"{key}."))
    elif isinstance(payload, list) and payload and isinstance(payload[0], dict):
        keys.extend(_json_keys(payload[0], max_keys=max_keys))
    return keys[:max_keys]


def _extract_document_detailed(filename: str, content: bytes) -> tuple[str, list[str]]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return _extract_pdf_detailed(content)
    if lower.endswith((".pptx", ".ppt")):
        return _extract_pptx_detailed(content)
    return "", ["extract_unsupported"]


def _extract_pdf_detailed(content: bytes) -> tuple[str, list[str]]:
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except ImportError:
        return "", ["extract_dependency_missing"]

    try:
        reader = PdfReader(io.BytesIO(content))
        parts = []
        max_pages = _max_document_pages()
        for page in reader.pages[:max_pages]:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts)[: _max_document_text_chars()]
    except Exception:
        return "", ["extract_failed"]

    if not text.strip():
        return "", ["extract_empty"]
    return text, []


def _extract_pptx_detailed(content: bytes) -> tuple[str, list[str]]:
    try:
        from pptx import Presentation  # type: ignore[import-untyped]
    except ImportError:
        return "", ["extract_dependency_missing"]

    try:
        presentation = Presentation(io.BytesIO(content))
        parts = []
        max_slides = _max_document_pages()
        for slide in presentation.slides[:max_slides]:
            for shape in slide.shapes:
                text = getattr(shape, "text", None)
                if text:
                    parts.append(text)
        text = "\n".join(parts)[: _max_document_text_chars()]
    except Exception:
        return "", ["extract_failed"]

    if not text.strip():
        return "", ["extract_empty"]
    return text, []


def _extract_image_metadata(content: bytes) -> str:
    try:
        from PIL import Image
    except ImportError:
        return ""
    try:
        image = Image.open(io.BytesIO(content))
        parts = [f"format={image.format}", f"size={image.size}"]
        exif = image.getexif()
        if exif:
            for tag_id, value in list(exif.items())[:20]:
                parts.append(f"exif:{tag_id}={value}")
        return "; ".join(str(p) for p in parts)
    except Exception:
        return ""


def _summarize(text: str, limit: int = 500) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _max_document_pages() -> int:
    raw = os.getenv("UMRS_DATA_CATALOG_MAX_DOCUMENT_PAGES", "200")
    try:
        return max(1, int(raw))
    except ValueError:
        return 200


def _max_document_text_chars() -> int:
    raw = os.getenv("UMRS_DATA_CATALOG_MAX_DOCUMENT_TEXT_CHARS", "500000")
    try:
        return max(1_000, int(raw))
    except ValueError:
        return 500_000
