"""Document parser seam for the understand stage (ADR-116 D-1).

``parse_for_extraction`` gives the text that semantic extraction reads. It
never changes the text that chunking and embedding use: retrieval hydration
recomputes chunk offsets and md5s from the existing text path
(``knowledge/extract.resolve_extraction_content``), so the parser output
feeds extraction only.

``UMRS_KNOWLEDGE_DOC_PARSER=docling|pypdf`` (default ``docling`` since
2026-11-13; the settings and their defaults live in ``ingest_models``).
Docling runs only for PDF, DOCX, PPTX and HTML, with OCR off, its layout
models read from a staged directory (``UMRS_KNOWLEDGE_DOCLING_ARTIFACTS``,
default the bundled ``/opt/umrs/models/docling``) with the Hugging Face hub
offline; a document over ``UMRS_KNOWLEDGE_DOCLING_MAX_BYTES`` or a PDF over
``UMRS_KNOWLEDGE_DOCLING_MAX_PAGES`` is left to the existing path. For one
document the parse tries, in order:

1. the shared ``umrs-docmodels`` service when ``UMRS_KNOWLEDGE_DOCMODELS_URL``
   is set. A refusal about the document itself (``ingest_models.
   SERVICE_FINAL_REASONS``: too large, too many pages, a parse that failed or
   a transfer that died after the document was sent) ends the chain at 3
   with ``doc_parser_fallback:<reason>``: the same bytes are never parsed
   again in this process. Any other failure adds
   ``docmodels_unavailable:<reason>`` and goes to 2 only as far as
   ``UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK`` allows (default ``sandbox``:
   the sandbox worker only, else ``doc_parser_fallback:no_local_fallback``);
2. the local path when the directory is staged: the sandbox worker with
   ``UMRS_EMBEDDING_SANDBOX=podman`` (``python -m
   umrs.knowledge.ingest_worker``, op ``parse``), otherwise a bounded thread
   in this process (the understand job calls this from a worker thread);
   without a service URL this always runs;
3. the existing text, with ``doc_parser_fallback:<reason>``.

The whole chain shares one per-document budget (``timeout_s``, by default
``UMRS_KNOWLEDGE_DOCMODELS_TIMEOUT_S``). It never raises.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

from . import ingest_models, ingest_worker

PARSER_ENV = ingest_models.DOC_PARSER_ENV
ARTIFACTS_ENV = ingest_models.DOCLING_ARTIFACTS_ENV
# The path the sandbox host sees for the artifacts directory, when the
# (containerised) server sees it under another path; unset = the same path.
ARTIFACTS_HOST_ENV = "UMRS_KNOWLEDGE_DOCLING_ARTIFACTS_HOST_DIR"
# The host directory the server's /models is mounted from (the umrs-front
# converge writes it): a model path under /models maps to the same place
# under it, for the sandbox mount, when no per-model *_HOST_DIR is set.
MODELS_HOST_ENV = "UMRS_KNOWLEDGE_MODELS_HOST_DIR"
SERVER_MODELS_ROOT = "/models"
SANDBOX_ARTIFACTS_DIR = "/models-ingest/docling"
# The sandbox runs the worker as ``python -m`` this name (imported, not
# spelled, so a rename cannot leave the sandbox calling a missing module).
WORKER_MODULE = ingest_worker.__name__

_DOCLING_SUFFIXES = frozenset({".pdf", ".docx", ".pptx", ".html", ".htm"})
_NOT_PROSE_KINDS = frozenset({"code", "cpp", "cuda", "schema"})
# Original-bytes check: a PDF past READ_CAP reaches understand as text the
# spool path already extracted, which Docling cannot read.
_MAGIC = {".pdf": b"%PDF", ".docx": b"PK", ".pptx": b"PK"}

_T = TypeVar("_T")


@dataclass(frozen=True)
class ParsedText:
    """Extraction text, the parser that produced it, and notes."""

    text: str
    parser: str  # "docling@<version>" | "pypdf" | "python-pptx" | "raw"
    notes: tuple[str, ...] = ()


class ParserUnavailable(RuntimeError):
    """Docling cannot run here; the message is the short reason."""


class BoundedTimeout(RuntimeError):
    """A bounded call was still running at its timeout (left to finish)."""


def configured_parser() -> str:
    """``docling`` or ``pypdf`` (``ingest_models.configured_doc_parser``)."""
    return ingest_models.configured_doc_parser()


def artifacts_dir() -> str:
    return ingest_models.configured_docling_artifacts()


def sandbox_host_path(server_path: str, explicit_env: str) -> str:
    """The host path the sandbox mounts for a model the server sees at
    ``server_path``: ``explicit_env`` when set; ``""`` (mount nothing) for a
    model bundled in the image, which the sandbox runs too; else the path
    re-rooted from ``/models`` onto ``UMRS_KNOWLEDGE_MODELS_HOST_DIR``; else
    the path itself (server and host see the same path)."""
    explicit = os.environ.get(explicit_env, "").strip()
    if explicit:
        return explicit
    path = PurePosixPath(server_path)
    if path.is_relative_to(ingest_models.BUNDLED_MODELS_ROOT):
        return ""
    root = os.environ.get(MODELS_HOST_ENV, "").strip()
    if root and path.is_relative_to(SERVER_MODELS_ROOT):
        return str(PurePosixPath(root) / path.relative_to(SERVER_MODELS_ROOT))
    return server_path


def sandbox_mounts(
    server_path: str, explicit_env: str, container_dir: str
) -> tuple[list[tuple[str, str]], str]:
    """``(read-only mounts, the path the worker reads)``: a bundled model is
    read where the image has it, with no mount (as ``embed_sandbox`` does
    for e5); any other is mounted at ``container_dir``."""
    host = sandbox_host_path(server_path, explicit_env)
    if not host:
        return [], server_path
    return [(host, container_dir)], container_dir


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
    the existing path runs here. ``timeout_s`` bounds the whole Docling
    chain for this document (service, then local); None means
    ``UMRS_KNOWLEDGE_DOCMODELS_TIMEOUT_S``. A parse that does not finish in
    time falls back with ``doc_parser_fallback:timeout``.
    """
    if not wants_docling(filename, kind):
        return _fallback(data, filename, fallback_text)
    notes: list[str] = []
    if timeout_s is None:
        timeout_s = ingest_models.configured_docmodels_timeout()
    try:
        text, version = _docling_text(data, filename, timeout_s, notes)
    except Exception as exc:  # noqa: BLE001 — every failure falls back, named
        notes.append(f"doc_parser_fallback:{_reason(exc)}")
        return _fallback(data, filename, fallback_text, notes)
    if not text.strip():
        notes.append("doc_parser_fallback:empty_text")
        return _fallback(data, filename, fallback_text, notes)
    return ParsedText(text=text, parser=f"docling@{version}", notes=tuple(notes))


def _reason(exc: BaseException) -> str:
    if isinstance(exc, ParserUnavailable):
        return str(exc)
    if isinstance(exc, BoundedTimeout):
        return "timeout"
    if isinstance(exc, ImportError):
        return "docling_not_installed"
    from .embed_sandbox import EmbeddingSandboxError

    if isinstance(exc, EmbeddingSandboxError):
        return "sandbox_failed"
    return type(exc).__name__


def _fallback(
    data: bytes, filename: str, fallback_text: str | None, notes: Sequence[str] = ()
) -> ParsedText:
    """The existing text path (pypdf / python-pptx / the bytes as UTF-8)."""
    from .extract import _is_binary_document_bytes, resolve_extraction_content

    binary = _is_binary_document_bytes(filename, data)
    parser = _fallback_parser(filename) if binary else "raw"
    if fallback_text is not None:
        return ParsedText(text=fallback_text, parser=parser, notes=tuple(notes))
    raw = resolve_extraction_content(filename, data) if binary else data
    return ParsedText(
        text=raw.decode("utf-8", errors="replace"), parser=parser, notes=tuple(notes)
    )


def _fallback_parser(filename: str) -> str:
    return "python-pptx" if Path(filename).suffix.lower() == ".pptx" else "pypdf"


def pdf_page_count(data: bytes, filename: str) -> int | None:
    """A PDF's page count (pypdf, no rendering); None for another format or
    a PDF pypdf cannot read (Docling then decides)."""
    if Path(filename).suffix.lower() != ".pdf" or not data.startswith(b"%PDF"):
        return None
    try:
        from pypdf import PdfReader

        return len(PdfReader(io.BytesIO(data)).pages)
    except Exception:  # noqa: BLE001 — unreadable here is not a parse failure
        return None


def _check_document(data: bytes, filename: str, timeout_s: float) -> None:
    """Refuse, by name, what Docling should not be given at all."""
    magic = _MAGIC.get(Path(filename).suffix.lower())
    if magic and not data.startswith(magic):
        raise ParserUnavailable("not_original_bytes")
    if timeout_s <= 0:
        raise ParserUnavailable("budget_exhausted")
    if len(data) > ingest_models.configured_docling_max_bytes():
        raise ParserUnavailable("too_large")
    pages = pdf_page_count(data, filename)
    if pages is not None and pages > ingest_models.configured_docling_max_pages():
        raise ParserUnavailable("too_many_pages")


def _docling_text(
    data: bytes, filename: str, timeout_s: float, notes: list[str]
) -> tuple[str, str]:
    """``(markdown, docling version)``: the service first, then the local
    path (as far as the service's answer allows) with what is left of
    ``timeout_s``."""
    _check_document(data, filename, timeout_s)
    stop = time.monotonic() + timeout_s
    if ingest_models.configured_docmodels_url():
        served = _docling_via_service(data, filename, timeout_s, notes)
        if served is not None:
            return served
        refusal = local_fallback_refusal()
        if refusal:
            raise ParserUnavailable(refusal)
    return _docling_local(data, filename, stop - time.monotonic())


def local_fallback_refusal() -> str:
    """After the service could not answer: ``""`` when the local path may run
    (``UMRS_KNOWLEDGE_DOCMODELS_LOCAL_FALLBACK``), else the note reason.

    ``sandbox`` (the default) allows only the one-shot ingest sandbox, which
    frees its memory; ``inprocess`` also allows this server process.
    """
    mode = ingest_models.configured_local_fallback()
    if mode == "inprocess":
        return ""
    if mode == "sandbox":
        from .embed_sandbox import worker_sandbox_config

        if worker_sandbox_config() is not None:
            return ""
    return "no_local_fallback"


def _docling_via_service(
    data: bytes, filename: str, timeout_s: float, notes: list[str]
) -> tuple[str, str] | None:
    """The service's parse, or None (unavailable: noted). A refusal about the
    document itself raises :class:`ParserUnavailable` (no local retry)."""
    from . import docmodels_client

    try:
        reply = docmodels_client.parse(
            data,
            Path(filename).name,
            timeout_s,
            max_pages=ingest_models.configured_docling_max_pages(),
            artifacts=artifacts_dir(),
            base_url=ingest_models.configured_docmodels_url(),
        )
    except docmodels_client.DocmodelsUnavailable as exc:
        if exc.reason in ingest_models.SERVICE_FINAL_REASONS:
            raise ParserUnavailable(exc.reason) from exc
        notes.append(f"docmodels_unavailable:{exc.reason}")
        return None
    return reply.text, reply.version


def _docling_local(data: bytes, filename: str, timeout_s: float) -> tuple[str, str]:
    if timeout_s <= 0:
        raise ParserUnavailable("budget_exhausted")
    artifacts = artifacts_dir()
    if not ingest_models.is_docling_staged(artifacts):
        raise ParserUnavailable("artifacts_missing")
    from .embed_sandbox import worker_sandbox_config

    config = worker_sandbox_config()
    if config is None:
        return _convert_bounded(data, filename, artifacts, timeout_s)
    config = replace(config, timeout=min(config.timeout, timeout_s))
    return _docling_in_sandbox(config, data, filename, artifacts)


def bounded_call(fn: Callable[[], _T], timeout_s: float | None, name: str) -> _T:
    """``fn()`` in a daemon thread joined for ``timeout_s`` (None = unbounded).

    A call still running at the timeout raises :class:`BoundedTimeout` and is
    left to finish on its own thread (a thread cannot be stopped); the
    caller moves on with its fallback.
    """
    if timeout_s is None:
        return fn()
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised in the caller
            box["error"] = exc

    worker = threading.Thread(target=work, name=name, daemon=True)
    worker.start()
    worker.join(max(0.0, timeout_s))
    if worker.is_alive():
        raise BoundedTimeout(name)
    if "error" in box:
        raise box["error"]
    return box["value"]


def _convert_bounded(
    data: bytes, filename: str, artifacts: str, timeout_s: float | None
) -> tuple[str, str]:
    """``docling_convert`` bounded by ``timeout_s`` (see :func:`bounded_call`)."""
    cap = ingest_models.configured_docling_max_pages()
    return bounded_call(
        lambda: docling_convert(data, filename, artifacts, cap), timeout_s, "kg-docling"
    )


def _docling_in_sandbox(
    config: Any, data: bytes, filename: str, artifacts: str
) -> tuple[str, str]:
    from .embed_sandbox import run_worker

    mounts, artifacts_path = sandbox_mounts(
        artifacts, ARTIFACTS_HOST_ENV, SANDBOX_ARTIFACTS_DIR
    )
    reply = run_worker(
        config,
        {
            "op": "parse",
            "filename": Path(filename).name,
            "data_b64": base64.b64encode(data).decode("ascii"),
            "artifacts_path": artifacts_path,
            "max_pages": ingest_models.configured_docling_max_pages(),
        },
        module=WORKER_MODULE,
        mounts=mounts,
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


def warm_converter(artifacts_path: str) -> Any:
    """Build (and cache) the converter now and load its PDF pipeline (the
    layout and TableFormer models), offline: the docmodels service's warm
    load. ``DocumentConverter()`` alone loads no model, so a broken weight
    would surface only at the first parse, with ``/readyz`` already 200."""
    set_offline_env()
    converter = _converter(artifacts_path)
    converter.initialize_pipeline(_pdf_format())
    return converter


def _pdf_format() -> Any:
    from docling.datamodel.base_models import InputFormat

    return InputFormat.PDF


def _document_stream(filename: str, data: bytes) -> Any:
    from docling.datamodel.base_models import DocumentStream

    return DocumentStream(name=Path(filename).name, stream=io.BytesIO(data))


def docling_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("docling")
    except PackageNotFoundError:
        return "unknown"


def _page_total(document: Any) -> int:
    try:
        return int(document.num_pages())
    except Exception:  # noqa: BLE001 — a count is informational
        return 0


def docling_parse(
    data: bytes, filename: str, artifacts_path: str, max_pages: int | None = None
) -> tuple[str, str, int]:
    """``(markdown, docling version, pages)`` of one document, OCR off,
    offline. ``max_pages`` is Docling's page limit for a PDF only: Docling
    also counts PPTX slides as pages and refuses a deck over the limit, which
    had no cap before 2026-11-13."""
    set_offline_env()
    stream = _document_stream(filename, data)
    is_pdf = Path(filename).suffix.lower() == ".pdf"
    limit = {"max_num_pages": int(max_pages)} if max_pages and is_pdf else {}
    document = _converter(artifacts_path).convert(stream, **limit).document
    return str(document.export_to_markdown()), docling_version(), _page_total(document)


def docling_convert(
    data: bytes, filename: str, artifacts_path: str, max_pages: int | None = None
) -> tuple[str, str]:
    """``(markdown, docling version)`` of one document, OCR off, offline."""
    text, version, _pages = docling_parse(data, filename, artifacts_path, max_pages)
    return text, version


__all__ = [
    "ARTIFACTS_ENV",
    "PARSER_ENV",
    "BoundedTimeout",
    "ParsedText",
    "ParserUnavailable",
    "bounded_call",
    "configured_parser",
    "docling_convert",
    "docling_parse",
    "local_fallback_refusal",
    "parse_for_extraction",
    "pdf_page_count",
    "sandbox_host_path",
    "sandbox_mounts",
    "wants_docling",
]
