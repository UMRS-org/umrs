"""Give retrieved chunks their full text and trusted labels (ADR-093 D-2 step 6).

A chunk node stores offsets, an md5 and a 500-character excerpt; its full
text is, in order of preference:

1. ``knowledge_nodes.search_text`` (migration 0120, writer-maintained) when
   its md5 equals ``content.chunk_md5`` (after the ``header_chars`` a
   sentence-aware header adds, C9);
2. the slice of the artifact the embed job chunked, md5-verified — at most
   one object-store GET per artifact per request, decoded as the embed job
   decoded it (:func:`chunked_text`: PDF/PPTX extracted, NUL removed);
3. the text a legacy canonized node stores itself, else its excerpt.

Labels come from trusted artifact metadata (``source_name``, the staged
upload key, the title) of the node's own project, never from passage text.

Moved here from ``api/rag_spine.py`` (``_passage_text``, ``_artifact_slice``,
``_chunk_span``) so every consumer hydrates the same way. Composition-layer
code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import PurePosixPath
from typing import Any

from .dense import as_content, query_rows
from .types import Candidate, FetchBytes

# The artifact must be the node's own project's: ``content.artifact_id`` is
# node data, and a node imported under a chosen id must not borrow another
# project's label or object (review security-3, 2026-10-10). A node whose
# artifact lives elsewhere falls back to its stored text or excerpt.
_NODE_SQL = (
    "SELECT n.node_id, n.graph_id, n.content::text, a.source_ref, "
    "a.metadata->>'source_name', a.metadata->>'staged_from', a.title{search_text} "
    "FROM knowledge_nodes n "
    "LEFT JOIN knowledge_artifacts a ON a.artifact_id = n.content->>'artifact_id' "
    "AND a.project_id = n.project_id "
    "WHERE n.node_id = ANY(%s)"
)
_COUNT_SQL = (
    "SELECT n.graph_id, n.content->>'artifact_id', count(*) FROM knowledge_nodes n "
    "WHERE n.graph_id = ANY(%s) AND n.content->>'artifact_id' = ANY(%s) "
    "AND n.content->>'kind' = 'chunk' GROUP BY 1, 2"
)
_PROBE_SQL = (
    "SELECT 1 FROM information_schema.columns "
    "WHERE table_name = 'knowledge_nodes' AND column_name = 'search_text'"
)

_SEARCH_TEXT: bool | None = None  # probed once per process
_PROBE_LOCK = threading.Lock()


def has_search_text(pool: Any) -> bool:
    """Whether migration 0120's column exists (probed once per process)."""
    global _SEARCH_TEXT
    with _PROBE_LOCK:
        if _SEARCH_TEXT is None:
            try:
                _SEARCH_TEXT = bool(query_rows(pool, _PROBE_SQL, ()))
            except Exception:  # noqa: BLE001 — no column is the safe answer
                return False
        return _SEARCH_TEXT


# ── the text of one chunk ───────────────────────────────────────────────────


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def chunk_span(content: dict[str, Any]) -> tuple[int, int, str] | None:
    """``(start, end, md5)`` when the node records where its text sits."""
    start, end, md5 = (
        content.get("chunk_start"),
        content.get("chunk_end"),
        content.get("chunk_md5"),
    )
    if isinstance(start, int) and isinstance(end, int) and md5:
        return start, end, str(md5)
    return None


def stored_text(content: dict[str, Any], search_text: str | None) -> str | None:
    """The chunk body from ``search_text``, only when its md5 matches."""
    span = chunk_span(content)
    if not search_text or span is None:
        return None
    header = content.get("header_chars")
    body = (
        search_text[header:] if isinstance(header, int) and header > 0 else search_text
    )
    return body if _md5(body) == span[2] else None


def decode_chunked(raw: bytes) -> str:
    """Text bytes as the embed job chunks them: UTF-8 with undecodable bytes
    dropped, NUL removed (Postgres TEXT and JSONB refuse it, so a chunk body
    holding one could never be stored), outer whitespace stripped. Chunk
    offsets index this string."""
    return raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()


def chunked_text(raw: bytes | None, source_name: str = "") -> str | None:
    """The text the embed job chunked from an artifact's bytes
    (``knowledge_adapters._artifact_text``): a PDF — or a PPTX, recognised by
    its ``metadata.source_name`` — through the same catalog extractor, then
    :func:`decode_chunked`. ``None`` when there is no text (a scanned PDF)."""
    if not raw:
        return None
    from umrs.knowledge.extract import (
        _is_binary_document_bytes,
        resolve_extraction_content,
    )

    if _is_binary_document_bytes(source_name, raw):
        raw = resolve_extraction_content(source_name, raw)
        if _is_binary_document_bytes(source_name, raw):
            return None
    return decode_chunked(raw) or None


def artifact_slice(
    content: dict[str, Any],
    source_ref: str,
    documents: dict[str, str | None],
    fetch_bytes: FetchBytes | None,
    source_name: str = "",
) -> str | None:
    """The chunk's full text from its artifact, only when its md5 matches.

    ``source_name`` is the artifact's ``metadata.source_name``, the name the
    embed job decoded it under (a PPTX is extracted only by its name)."""
    span = chunk_span(content)
    if span is None or not source_ref or fetch_bytes is None:
        return None
    if source_ref not in documents:
        documents[source_ref] = chunked_text(fetch_bytes(source_ref), source_name)
    document = documents[source_ref]
    if document is None:
        return None
    start, end, md5 = span
    piece = document[start:end]
    return piece if _md5(piece) == md5 else None


def passage_text(
    content: dict[str, Any],
    source_ref: str,
    documents: dict[str, str | None],
    fetch_bytes: FetchBytes | None,
    search_text: str | None = None,
    source_name: str = "",
) -> tuple[str, str]:
    """``(text, source)``: stored body, verified artifact slice, the node's
    own text, else its excerpt; ``("", "")`` when the node has none."""
    text = stored_text(content, search_text)
    if text:
        return text, "search_text"
    text = artifact_slice(content, source_ref, documents, fetch_bytes, source_name)
    if text:
        return text, "artifact"
    for key, source in (("text", "node"), ("excerpt", "excerpt")):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value, source
    return "", ""


# ── hydrating candidates ────────────────────────────────────────────────────


def _basename(uri: str | None) -> str:
    return PurePosixPath(uri).name if uri else ""


def _rows(pool: Any, node_ids: list[str]) -> dict[str, tuple]:
    column = ", n.search_text" if has_search_text(pool) else ""
    rows = query_rows(pool, _NODE_SQL.format(search_text=column), (node_ids,))
    return {str(row[0]): row for row in rows}


def _chunk_counts(pool: Any, keys: set[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """Chunks per (graph, artifact): one grouped read for every candidate."""
    if not keys:
        return {}
    graphs = sorted({g for g, _a in keys})
    artifacts = sorted({a for _g, a in keys})
    rows = query_rows(pool, _COUNT_SQL, (graphs, artifacts))
    return {(str(g), str(a)): int(n) for g, a, n in rows}


def _label(
    name: str | None, staged: str | None, title: str | None, fallback: str
) -> str:
    return name or _basename(staged) or title or fallback


def _fill_identity(candidate: Candidate, content_text: Any) -> dict[str, Any]:
    content = candidate.content or as_content(content_text)
    candidate.content = content
    candidate.artifact_id = candidate.artifact_id or str(
        content.get("artifact_id") or ""
    )
    index = content.get("chunk_index")
    candidate.chunk_index = index if isinstance(index, int) else candidate.chunk_index
    candidate.section = str(content.get("section") or "")
    span = chunk_span(content)
    candidate.span = (span[0], span[1]) if span else None
    return content


def _fill(
    candidate: Candidate,
    row: tuple,
    documents: dict[str, str | None],
    fetch_bytes: FetchBytes | None,
) -> None:
    _node, _graph, content_text, ref, name, staged, title = row[:7]
    content = _fill_identity(candidate, content_text)
    candidate.source_ref = ref or ""
    candidate.source_name = _label(
        name, staged, title, candidate.artifact_id or candidate.node_id
    )
    search_text = row[7] if len(row) > 7 else None
    candidate.text, candidate.text_source = passage_text(
        content, candidate.source_ref, documents, fetch_bytes, search_text, name or ""
    )


def _graph_of(row: tuple) -> str:
    return str(row[1] or "")


def _fill_all(
    pool: Any,
    todo: list[Candidate],
    documents: dict[str, str | None],
    fetch_bytes: FetchBytes | None,
) -> dict[str, tuple]:
    rows = _rows(pool, [c.node_id for c in todo])
    for candidate in todo:
        row = rows.get(candidate.node_id)
        if row is not None:
            _fill(candidate, row, documents, fetch_bytes)
    return rows


def _fill_counts(pool: Any, todo: list[Candidate], rows: dict[str, tuple]) -> None:
    graphs = {c.node_id: _graph_of(rows[c.node_id]) for c in todo if c.node_id in rows}
    keys = {
        (graphs[c.node_id], c.artifact_id)
        for c in todo
        if c.node_id in graphs and c.artifact_id
    }
    counts = _chunk_counts(pool, keys)
    for candidate in todo:
        key = (graphs.get(candidate.node_id, ""), candidate.artifact_id)
        candidate.chunk_count = counts.get(key, candidate.chunk_count)


def hydrate(
    pool: Any,
    candidates: list[Candidate],
    fetch_bytes: FetchBytes | None,
    *,
    documents: dict[str, str | None] | None = None,
) -> None:
    """Fill text, labels, span and chunk count in place.

    Candidates already hydrated (``text_source`` set) are skipped; a
    candidate whose node has no text keeps ``text == ""`` and the caller
    drops it. ``documents`` is the per-request artifact cache, shared with the
    neighbour merge so an artifact is fetched at most once.
    """
    todo = [c for c in candidates if not c.text_source]
    if not todo:
        return
    cache = documents if documents is not None else {}
    rows = _fill_all(pool, todo, cache, fetch_bytes)
    _fill_counts(pool, todo, rows)
