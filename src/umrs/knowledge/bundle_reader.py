"""ExportBundleWriter — writes r4-kb v1.x bundle directory.

Produces the flat-file format described in INV-13 / TASK-041:

    <project_id>/
    ├── manifest.json
    ├── artifacts.jsonl
    ├── documents.jsonl
    ├── nodes.jsonl
    ├── edges.jsonl
    ├── embeddings.jsonl
    ├── catalog.jsonl
    └── signature.sig

r4-kb v1.3 (ADR-093 D-12, export ``chunk_text`` opt-in): a chunk row of
nodes.jsonl may carry ``text``, the chunk's full body (md5 =
``content.chunk_md5``, at most :data:`CHUNK_TEXT_MAX_CHARS`; a longer body
is cut and flagged ``text_truncated``) and, for a chunk with a header (C9),
``text_header`` (the first ``content.header_chars`` characters of its
``search_text``). :func:`chunk_passage` reads a chunk row preferring that
text; :func:`chunk_search_text` gives the ``search_text`` an import writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .chunking import search_text_of
from .retrieval.hydrate import chunk_span

logger = logging.getLogger(__name__)

#: The largest chunk body a v1.3 node row carries (ADR-093 D-12).
CHUNK_TEXT_MAX_CHARS = 4096

#: A chunk row -> its md5-verified slice of a raw object carried in the
#: bundle, or None.
RawSlice = Callable[[dict[str, Any]], "str | None"]


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _content(row: dict[str, Any]) -> dict[str, Any]:
    content = row.get("content")
    return content if isinstance(content, dict) else {}


def is_chunk_row(row: dict[str, Any]) -> bool:
    """A chunk node (``content.kind == 'chunk'``) of nodes.jsonl."""
    return _content(row).get("kind") == "chunk"


def node_text(row: dict[str, Any]) -> str | None:
    """The v1.3 node ``text`` of a chunk row, only when its md5 matches."""
    text = row.get("text")
    span = chunk_span(_content(row))
    if not isinstance(text, str) or not text or span is None:
        return None
    return text if _md5(text) == span[2] else None


def search_text_for(row: dict[str, Any], body: str) -> str | None:
    """``search_text`` for a verified chunk body: the body, or header + body
    when the chunk records ``header_chars`` (None when the header is unknown).

    The header is the row's ``text_header`` (the exported prefix itself)
    when its length is ``header_chars``, else the embed writer's rule over
    ``content.header`` (``chunking.search_text_of``: header, newline) — a
    chunk exported through the artifact slice, or from a v1.2 bundle,
    carries no ``text_header``.
    """
    content = _content(row)
    header_chars = content.get("header_chars")
    if not isinstance(header_chars, int) or header_chars <= 0:
        return body
    prefix = row.get("text_header")
    if isinstance(prefix, str) and len(prefix) == header_chars:
        return prefix + body
    return search_text_of(content, body)


def chunk_search_text(
    row: dict[str, Any], raw_slice: RawSlice | None = None
) -> tuple[str | None, str]:
    """``(search_text, source)`` an import writes for one node row.

    ``source`` is ``node`` (the md5-verified v1.3 text), ``raw`` (the
    md5-verified slice of a raw object in the bundle), ``pending`` (a chunk
    with a span but no verified text: a backfill is needed) or ``""`` (not a
    chunk with a span — nothing to fill). A text holding a NUL is pending
    too: Postgres refuses it, and the embed job's own text never has one.
    """
    if not is_chunk_row(row) or chunk_span(_content(row)) is None:
        return None, ""
    body = node_text(row)
    source = "node"
    if body is None and raw_slice is not None:
        body, source = raw_slice(row), "raw"
    text = search_text_for(row, body) if body else None
    if text is None or "\x00" in text:
        return None, "pending"
    return text, source


def chunk_passage(
    row: dict[str, Any], raw_slice: RawSlice | None = None
) -> tuple[str, str]:
    """``(text, source)`` of a bundle chunk row, preferring the node text.

    Order: the md5-verified v1.3 ``text`` (``node``), the md5-verified raw
    slice (``raw``), a cut v1.3 text (``node_truncated``), the 500-char
    excerpt (``excerpt``); ``("", "")`` when the row has none.
    """
    text = node_text(row)
    if text:
        return text, "node"
    text = raw_slice(row) if raw_slice is not None else None
    if text:
        return text, "raw"
    return _unverified_text(row)


def _unverified_text(row: dict[str, Any]) -> tuple[str, str]:
    """A cut v1.3 text, else the excerpt, else ``("", "")``."""
    candidates = (
        (row.get("text"), "node_truncated"),
        (_content(row).get("excerpt"), "excerpt"),
    )
    for value, source in candidates:
        if isinstance(value, str) and value.strip():
            return value, source
    return "", ""


class ExportBundleWriter:
    """Writes knowledge tables → flat r4-kb v1.x bundle directory."""

    def __init__(self, output_dir: Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """The bundle directory being written."""
        return self._dir

    async def write_artifacts(self, rows: list[dict[str, Any]]) -> None:
        """Write artifacts.jsonl."""
        path = self._dir / "artifacts.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d artifact rows to %s", len(rows), path)

    async def write_documents(self, rows: list[dict[str, Any]]) -> None:
        """Write documents.jsonl (IR content)."""
        path = self._dir / "documents.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d document rows to %s", len(rows), path)

    async def write_nodes(self, rows: list[dict[str, Any]]) -> None:
        """Write nodes.jsonl."""
        path = self._dir / "nodes.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d node rows to %s", len(rows), path)

    async def write_edges(self, rows: list[dict[str, Any]]) -> None:
        """Write edges.jsonl."""
        path = self._dir / "edges.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d edge rows to %s", len(rows), path)

    async def write_embeddings(self, rows: list[dict[str, Any]]) -> None:
        """Write embeddings.jsonl."""
        path = self._dir / "embeddings.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d embedding rows to %s", len(rows), path)

    async def write_node_embeddings(self, rows: list[dict[str, Any]]) -> None:
        """Write node_embeddings.jsonl (canonical node vectors, 0075)."""
        path = self._dir / "node_embeddings.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d node-embedding rows to %s", len(rows), path)

    async def write_catalog(self, rows: list[dict[str, Any]]) -> None:
        """Write catalog.jsonl."""
        path = self._dir / "catalog.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d catalog rows to %s", len(rows), path)

    async def write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write manifest.json."""
        path = self._dir / "manifest.json"
        content = json.dumps(manifest, sort_keys=True, indent=2)
        path.write_text(content, encoding="utf-8")
        logger.debug("Wrote manifest to %s", path)

    async def write_a2l_links(self, rows: list[dict[str, Any]]) -> None:
        """Write a2l_links.jsonl (knowledge artifact → A2L mapping)."""
        path = self._dir / "a2l_links.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d A2L link rows to %s", len(rows), path)

    async def write_a2l_refs(self, rows: list[dict[str, Any]]) -> None:
        """Write a2l_refs.jsonl (A2L artifact identity mappings)."""
        path = self._dir / "a2l_refs.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d A2L ref rows to %s", len(rows), path)

    async def write_signature(self, signature: str) -> None:
        """Write signature.sig."""
        path = self._dir / "signature.sig"
        path.write_text(signature, encoding="utf-8")
        logger.debug("Wrote signature to %s", path)


async def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as newline-delimited JSON."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
