"""Document parser seam for the understand stage (ADR-116 D-1).

``parse_for_extraction`` gives the text that semantic extraction reads. It
never changes the text that chunking and embedding use: retrieval hydration
recomputes chunk offsets and md5s from the existing text path
(``knowledge/extract.resolve_extraction_content``), so the parser output
feeds extraction only.

``UMRS_KNOWLEDGE_DOC_PARSER=pypdf|docling`` (default ``pypdf``). Docling runs
only for PDF, DOCX, PPTX and HTML, with OCR off and its layout models read
from a pre-staged directory (``UMRS_KNOWLEDGE_DOCLING_ARTIFACTS``, default
``/models/docling``) with the Hugging Face hub offline. With
``UMRS_EMBEDDING_SANDBOX=podman`` it runs in the sandbox worker
(``python -m umrs.knowledge.ingest_worker``, op ``parse``); otherwise in the
calling thread (the understand job calls this from a worker thread). Any
failure returns the existing text with the note
``doc_parser_fallback:<reason>``.
"""

from __future__ import annotations

import base64
import io
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from . import ingest_worker

PARSER_ENV = "UMRS_KNOWLEDGE_DOC_PARSER"
ARTIFACTS_ENV = "UMRS_KNOWLEDGE_DOCLING_ARTIFACTS"
# The path the sandbox host sees for the artifacts directory, when the
# (containerised) server sees it under another path; unset = the same path.
ARTIFACTS_HOST_ENV = "UMRS_KNOWLEDGE_DOCLING_ARTIFACTS_HOST_DIR"
# The host directory the server's /models is mounted from (the umrs-front
# converge writes it): a model path under /models maps to the same place
# under it, for the sandbox mount, when no per-model *_HOST_DIR is set.
MODELS_HOST_ENV = "UMRS_KNOWLEDGE_MODELS_HOST_DIR"
SERVER_MODELS_ROOT = "/models"
DEFAULT_ARTIFACTS = "/models/docling"
SANDBOX_ARTIFACTS_DIR = "/models-ingest/docling"
# The sandbox runs the worker as ``python -m`` this name (imported, not
# spelled, so a rename cannot leave the sandbox calling a missing module).
WORKER_MODULE = ingest_worker.__name__

_DOCLING_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".html", ".htm"})
_NOT_PROSE_KINDS = frozenset({"code", "cpp", "cuda", "schema"})
# Original-bytes check: a PDF past READ_CAP reaches understand as text the
# spool path already extracted, which Docling cannot read.
_MAGIC = {".pdf": b"%PDF", ".docx": b"PK", ".pptx": b"PK"}


@dataclass(frozen=True)
class ParsedText:
    """Extraction text, the parser that produced it, and notes."""

    text: str
    parser: str  # "docling@<version>" | "pypdf" | "python-pptx" | "raw"
    notes: tuple[str, ...] = ()


class ParserUnavailable(RuntimeError):
    """Docling cannot run here; the message is the short reason."""


def configured_parser() -> str:
    """``docling`` or ``pypdf`` (anything else reads as the default)."""
    value = os.environ.get(PARSER_ENV, "").strip().lower()
    return "docling" if value == "docling" else "pypdf"


def artifacts_dir() -> str:
    return os.environ.get(ARTIFACTS_ENV, "").strip() or DEFAULT_ARTIFACTS


def sandbox_host_path(server_path: str, explicit_env: str) -> str:
    """The host path the sandbox mounts for a model the server sees at
    ``server_path``: ``explicit_env`` when set, else the path re-rooted from
    ``/models`` onto ``UMRS_KNOWLEDGE_MODELS_HOST_DIR``, else the path itself
    (server and host see the same path)."""
    explicit = os.environ.get(explicit_env, "").strip()
    if explicit:
        return explicit
    root = os.environ.get(MODELS_HOST_ENV, "").strip()
    path = PurePosixPath(server_path)
    if root and path.is_relative_to(SERVER_MODELS_ROOT):
        return str(PurePosixPath(root) / path.relative_to(SERVER_MODELS_ROOT))
    return server_path


def wants_docling(filename: str, kind: str) -> bool:
    """Docling is configured and the file is a PDF, DOCX, PPTX or HTML document."""
    if configured_parser() != "docling" or kind in _NOT_PROSE_KINDS:
        return False
    return Path(filename).suffix.lower() in _DOCLING_SUFFIXES


def parse_for_extraction(
    data: bytes,
    filename: str,
    kind: str,
    *,
    fallback_text: str | None = None,
    timeout_s: float | None = None,
) -> ParsedText:
    """The text semantic extraction reads; never raises.

    ``fallback_text`` is the text the existing path already produced (the
    understand job passes it so a PDF is not extracted twice); without it
    the existing path runs here. ``timeout_s`` bounds the Docling parse (the
    understand job passes what is left of its budget); a parse that does not
    finish in time falls back with ``doc_parser_fallback:timeout``.
    """
    if not wants_docling(filename, kind):
        return _fallback(data, filename, fallback_text)
    try:
        text, version = _docling_text(data, filename, timeout_s)
    except Exception as exc:  # noqa: BLE001 — every failure falls back, named
        return _fallback(
            data, filename, fallback_text, f"doc_parser_fallback:{_reason(exc)}"
        )
    if not text.strip():
        return _fallback(
            data, filename, fallback_text, "doc_parser_fallback:empty_text"
        )
    return ParsedText(text=text, parser=f"docling@{version}")


def _reason(exc: BaseException) -> str:
    if isinstance(exc, ParserUnavailable):
        return str(exc)
    if isinstance(exc, ImportError):
        return "docling_not_installed"
    from .embed_sandbox import EmbeddingSandboxError

    if isinstance(exc, EmbeddingSandboxError):
        return "sandbox_failed"
    return type(exc).__name__


def _fallback(
    data: bytes, filename: str, fallback_text: str | None, note: str = ""
) -> ParsedText:
    """The existing text path (pypdf / python-pptx / the bytes as UTF-8)."""
    from .extract import _is_binary_document_bytes, resolve_extraction_content

    notes = (note,) if note else ()
    binary = _is_binary_document_bytes(filename, data)
    parser = _fallback_parser(filename) if binary else "raw"
    if fallback_text is not None:
        return ParsedText(text=fallback_text, parser=parser, notes=notes)
    raw = resolve_extraction_content(filename, data) if binary else data
    return ParsedText(
        text=raw.decode("utf-8", errors="replace"), parser=parser, notes=notes
    )


def _fallback_parser(filename: str) -> str:
    return "python-pptx" if Path(filename).suffix.lower() == ".pptx" else "pypdf"


def _docling_text(
    data: bytes, filename: str, timeout_s: float | None = None
) -> tuple[str, str]:
    suffix = Path(filename).suffix.lower()
    magic = _MAGIC.get(suffix)
    if magic and not data.startswith(magic):
        raise ParserUnavailable("not_original_bytes")
    if timeout_s is not None and timeout_s <= 0:
        raise ParserUnavailable("budget_exhausted")
    artifacts = artifacts_dir()
    if not os.path.isdir(artifacts):
        raise ParserUnavailable("artifacts_missing")
    from .embed_sandbox import worker_sandbox_config

    config = worker_sandbox_config()
    if config is None:
        return _convert_bounded(data, filename, artifacts, timeout_s)
    if timeout_s is not None:
        config = replace(config, timeout=min(config.timeout, timeout_s))
    return _docling_in_sandbox(config, data, filename, artifacts)


def _convert_bounded(
    data: bytes, filename: str, artifacts: str, timeout_s: float | None
) -> tuple[str, str]:
    """``docling_convert`` in a daemon thread joined for ``timeout_s``.

    A parse still running at the timeout raises ``ParserUnavailable("timeout")``
    and is left to finish on its own thread (a thread cannot be stopped); the
    caller moves on with the fallback text.
    """
    if timeout_s is None:
        return docling_convert(data, filename, artifacts)
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = docling_convert(data, filename, artifacts)
        except BaseException as exc:  # noqa: BLE001 — re-raised in the caller
            box["error"] = exc

    worker = threading.Thread(target=work, name="kg-docling", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise ParserUnavailable("timeout")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _docling_in_sandbox(
    config: Any, data: bytes, filename: str, artifacts: str
) -> tuple[str, str]:
    from .embed_sandbox import run_worker

    host_dir = sandbox_host_path(artifacts, ARTIFACTS_HOST_ENV)
    reply = run_worker(
        config,
        {
            "op": "parse",
            "filename": Path(filename).name,
            "data_b64": base64.b64encode(data).decode("ascii"),
            "artifacts_path": SANDBOX_ARTIFACTS_DIR,
        },
        module=WORKER_MODULE,
        mounts=[(host_dir, SANDBOX_ARTIFACTS_DIR)],
    )
    return str(reply.get("text") or ""), str(reply.get("version") or "unknown")


def set_offline_env() -> None:
    """No model download at runtime (ADR-015, CONSTITUTION offline rule)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_CONVERTERS: dict[str, Any] = {}
_CONVERTER_LOCK = threading.Lock()


def _converter(artifacts_path: str) -> Any:
    """One Docling converter per artifacts directory (loading is slow)."""
    with _CONVERTER_LOCK:
        cached = _CONVERTERS.get(artifacts_path)
        if cached is None:
            cached = _CONVERTERS[artifacts_path] = _new_converter(artifacts_path)
        return cached


def _new_converter(artifacts_path: str) -> Any:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(artifacts_path=artifacts_path, do_ocr=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def docling_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("docling")
    except PackageNotFoundError:
        return "unknown"


def docling_convert(data: bytes, filename: str, artifacts_path: str) -> tuple[str, str]:
    """``(markdown, docling version)`` of one document, OCR off, offline."""
    set_offline_env()
    from docling.datamodel.base_models import DocumentStream

    stream = DocumentStream(name=Path(filename).name, stream=io.BytesIO(data))
    result = _converter(artifacts_path).convert(stream)
    return str(result.document.export_to_markdown()), docling_version()


__all__ = [
    "ARTIFACTS_ENV",
    "PARSER_ENV",
    "ParsedText",
    "ParserUnavailable",
    "configured_parser",
    "docling_convert",
    "parse_for_extraction",
    "sandbox_host_path",
    "wants_docling",
]
