"""Fill ``knowledge_nodes.search_text`` for existing chunks, without a re-embed
(ADR-093 D-6, component C6).

Usage::

    python -m umrs.knowledge.retrieval.backfill --project P [--graph G] [--dry-run] [--verify]

For every chunk node of project ``P`` (optionally one graph, an id or a name;
legacy ``graph_id ''`` rows count for the project's default graph) whose
``search_text`` is NULL, the text written is the first of:

1. the node's own stored text (legacy canonized corpora: ``content.text``);
2. the artifact slice ``text[chunk_start:chunk_end]`` of the text the embed
   job chunked, fetched once per artifact;

that hashes to the node's ``content.chunk_md5``. A sentence-aware chunk
(C9, ``content.header_chars``) stores its header, a newline and that body,
the embed writer's own rule (``chunking.search_text_of``), so hydrate's
``search_text[header_chars:]`` verifies it. Nothing else is ever written: a
candidate text whose md5 differs counts as ``mismatch``, a chunk with no text
to try counts as ``unavailable``. The UPDATE re-checks that the row is still
NULL and still has that md5.

``--verify`` also re-checks every chunk that already HAS ``search_text``
against the column contract (the body after ``content.header_chars`` hashes
to ``content.chunk_md5``, and the whole is what ``search_text_of`` builds).
A writer that replaced a chunk's content under the same node id without
touching the column leaves text the lexical lanes would still rank: such a
row counts as ``stale`` and is rewritten with the verified text
(``repaired``) or, when none verifies, set back to NULL (``cleared``). That
UPDATE re-checks the text it read and the md5. ``--dry-run`` computes the
same counts and writes nothing. Prints one JSON object of counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from umrs.knowledge.chunking import search_text_of
from umrs.knowledge.graph_registry import default_graph_id

logger = logging.getLogger(__name__)

#: artifact row (artifact_id, source_ref, kind, metadata) -> the text the
#: embed job chunked, or None when it cannot be read.
DocumentLoader = Callable[[dict[str, Any]], "str | None"]

OUTCOMES: tuple[str, ...] = ("from_node", "from_artifact", "mismatch", "unavailable")

# An id, else a name in the project, else a name of a project-less graph.
_GRAPH_SQL = (
    "/* backfill:graph */ SELECT graph_id FROM knowledge_graphs "
    "WHERE graph_id = %s OR (name = %s AND coalesce(project_id, '') IN (%s, '')) "
    "ORDER BY (graph_id = %s) DESC, (coalesce(project_id, '') = %s) DESC, graph_id "
    "LIMIT 1"
)

_UPDATE_SQL = (
    "/* backfill:write */ UPDATE knowledge_nodes SET search_text = %s "
    "WHERE node_id = %s AND search_text IS NULL AND content->>'chunk_md5' = %s"
)

# --verify: (new text or NULL, node_id, the text read, the md5 read).
_RESET_SQL = (
    "/* backfill:reset */ UPDATE knowledge_nodes SET search_text = %s "
    "WHERE node_id = %s AND search_text = %s "
    "AND content->>'chunk_md5' IS NOT DISTINCT FROM %s"
)

_ROWS_SQL = (
    "SELECT n.node_id, n.content, a.artifact_id, a.source_ref, a.kind, a.metadata, "
    "n.search_text FROM knowledge_nodes n "
    "LEFT JOIN knowledge_artifacts a ON a.artifact_id = n.content->>'artifact_id' "
    "AND a.project_id = n.project_id "  # never another project's object (security-3)
    "WHERE {scope} AND n.search_text IS {null} "
    "ORDER BY a.artifact_id NULLS LAST, n.node_id"
)


@dataclass
class _Chunk:
    node_id: str
    content: dict[str, Any]
    artifact: dict[str, Any]
    search_text: str | None = None

    @property
    def md5(self) -> str:
        return str(self.content.get("chunk_md5") or "")


@dataclass
class BackfillReport:
    """What one run found and did."""

    project_id: str
    graph_id: str | None = None
    dry_run: bool = False
    chunks: int = 0  # chunk rows in scope
    already: int = 0  # rows that had search_text before this run
    outcomes: dict[str, int] = field(default_factory=lambda: dict.fromkeys(OUTCOMES, 0))
    artifacts_read: int = 0
    written: int = 0
    # --verify: filled rows the column contract disowns, and what became of them
    verify: bool = False
    stale: int = 0
    repaired: int = 0
    cleared: int = 0
    reset: int = 0  # rows the verify pass wrote

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "dry_run": self.dry_run,
            "chunks": self.chunks,
            "already": self.already,
            **self.outcomes,
            "artifacts_read": self.artifacts_read,
            "written": self.written,
            "verify": self.verify,
            "stale": self.stale,
            "repaired": self.repaired,
            "cleared": self.cleared,
            "reset": self.reset,
        }


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8"), usedforsecurity=False).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


# ── reading ──────────────────────────────────────────────────────────────────


def _scope(project_id: str, graph_id: str | None) -> tuple[str, list[Any]]:
    """The project's chunk rows; with a graph, ``KnowledgeStore.list_nodes``'
    rule (legacy ``''`` rows belong to the project's default graph)."""
    sql = "n.project_id = %s AND n.content->>'kind' = 'chunk'"
    params: list[Any] = [project_id]
    if graph_id is not None:
        sql += " AND (n.graph_id = %s OR (n.graph_id = '' AND %s))"
        params += [graph_id, graph_id == default_graph_id(project_id)]
    return sql, params


def _resolve_graph(cur: Any, project_id: str, graph: str | None) -> str | None:
    """A graph id from an id or a name in the project; the token itself when unknown."""
    if not graph:
        return None
    cur.execute(_GRAPH_SQL, (graph, graph, project_id, graph, project_id))
    row = cur.fetchone()
    return str(row[0]) if row else graph


def _counts(cur: Any, scope: str, params: list[Any]) -> tuple[int, int]:
    """(chunk rows in scope, rows that already have search_text)."""
    cur.execute(
        "/* backfill:counts */ SELECT count(*), count(n.search_text) "
        f"FROM knowledge_nodes n WHERE {scope}",
        params,
    )
    row = cur.fetchone() or (0, 0)
    return int(row[0]), int(row[1])


def _chunk(row: tuple) -> _Chunk:
    node_id, content, artifact_id, source_ref, kind, metadata, *text = row
    return _Chunk(
        node_id=str(node_id),
        content=_as_dict(content),
        artifact={
            "artifact_id": artifact_id or "",
            "source_ref": source_ref or "",
            "kind": kind or "",
            "metadata": _as_dict(metadata),
        },
        search_text=text[0] if text else None,
    )


def _pending(cur: Any, scope: str, params: list[Any]) -> list[_Chunk]:
    """Chunk rows still without search_text, grouped by artifact."""
    cur.execute(
        "/* backfill:rows */ " + _ROWS_SQL.format(scope=scope, null="NULL"), params
    )
    return [_chunk(row) for row in cur.fetchall()]


def _filled(cur: Any, scope: str, params: list[Any]) -> list[_Chunk]:
    """Chunk rows that have search_text (``--verify``), grouped by artifact."""
    cur.execute(
        "/* backfill:filled */ " + _ROWS_SQL.format(scope=scope, null="NOT NULL"),
        params,
    )
    return [_chunk(row) for row in cur.fetchall()]


# ── verifying ────────────────────────────────────────────────────────────────


def _span(content: dict[str, Any]) -> tuple[int, int] | None:
    start, end = content.get("chunk_start"), content.get("chunk_end")
    if isinstance(start, int) and isinstance(end, int) and 0 <= start < end:
        return start, end
    return None


class _Documents:
    """The current artifact's text. Rows arrive grouped by artifact, so one
    entry means one read per artifact with one document held in memory."""

    def __init__(self, load: DocumentLoader) -> None:
        self._load = load
        self._artifact_id: str | None = None
        self._text: str | None = None
        self.reads = 0

    def text(self, artifact: dict[str, Any]) -> str | None:
        if artifact["artifact_id"] != self._artifact_id:
            self._artifact_id = artifact["artifact_id"]
            self._text = self._load(artifact)
            self.reads += 1
        return self._text


def _artifact_piece(chunk: _Chunk, documents: _Documents) -> str | None:
    """The chunk's span of its artifact's text."""
    span = _span(chunk.content)
    if (
        span is None
        or not chunk.artifact["artifact_id"]
        or not chunk.artifact["source_ref"]
    ):
        return None
    document = documents.text(chunk.artifact)
    return document[span[0] : span[1]] if document else None


def _candidate_texts(
    chunk: _Chunk, documents: _Documents
) -> Iterator[tuple[str, str | None]]:
    """``(outcome, text)`` in preference order; the artifact is read lazily."""
    node_text = chunk.content.get("text")
    yield "from_node", node_text if isinstance(node_text, str) and node_text else None
    yield "from_artifact", _artifact_piece(chunk, documents)


def verified_text(chunk: _Chunk, documents: _Documents) -> tuple[str | None, str]:
    """``(text, outcome)``: the first candidate whose md5 is the chunk's, else
    ``(None, "mismatch")`` when some text was found, ``(None, "unavailable")``."""
    if not chunk.md5:
        return None, "unavailable"
    found = False
    for outcome, text in _candidate_texts(chunk, documents):
        if text is None:
            continue
        if _md5(text) == chunk.md5:
            return text, outcome
        found = True
    return None, "mismatch" if found else "unavailable"


def _search_text(chunk: _Chunk, documents: _Documents) -> tuple[str | None, str]:
    """``(search_text, outcome)``: the verified body as the column stores it
    (header + newline + body for a headed chunk). A headed chunk whose
    header disagrees with its ``header_chars`` is a ``mismatch``."""
    body, outcome = verified_text(chunk, documents)
    if body is None:
        return None, outcome
    text = search_text_of(chunk.content, body)
    return (text, outcome) if text is not None else (None, "mismatch")


def honours_contract(content: dict[str, Any], search_text: str) -> bool:
    """The column contract (0120): the body after ``content.header_chars``
    hashes to ``content.chunk_md5``, and the whole is exactly what
    ``chunking.search_text_of`` builds for that body (so a headed chunk's
    header is the node's own)."""
    md5 = str(content.get("chunk_md5") or "")
    header_chars = content.get("header_chars")
    headed = isinstance(header_chars, int) and header_chars > 0
    body = search_text[header_chars:] if headed else search_text
    return (
        bool(md5) and _md5(body) == md5 and search_text_of(content, body) == search_text
    )


def _verify(
    chunks: list[_Chunk], documents: _Documents, report: BackfillReport
) -> list[tuple[str | None, str, str, str | None]]:
    """``_RESET_SQL`` rows for the filled chunks the contract disowns: the
    verified text when one exists (``repaired``), else NULL (``cleared``)."""
    resets: list[tuple[str | None, str, str, str | None]] = []
    for chunk in chunks:
        held = chunk.search_text
        if held is None or honours_contract(chunk.content, held):
            continue
        report.stale += 1
        text, _outcome = _search_text(chunk, documents)
        if text is None:
            report.cleared += 1
        else:
            report.repaired += 1
        resets.append((text, chunk.node_id, held, chunk.md5 or None))
    return resets


# ── writing ──────────────────────────────────────────────────────────────────


def _write(pool: Any, sql: str, updates: list[tuple], batch_size: int) -> int:
    """Apply ``updates`` with ``sql``, one commit per batch."""
    written = 0
    for start in range(0, len(updates), max(batch_size, 1)):
        conn = pool.getconn()
        try:
            with conn:
                with conn.cursor() as cur:
                    for row in updates[start : start + batch_size]:
                        cur.execute(sql, row)
                        written += max(cur.rowcount, 0)
        finally:
            pool.putconn(conn)
    return written


def _read(
    pool: Any, report: BackfillReport, graph: str | None
) -> tuple[list[_Chunk], list[_Chunk]]:
    """``(rows without search_text, rows with it)``; the second only for
    ``--verify``."""
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                report.graph_id = _resolve_graph(cur, report.project_id, graph)
                scope, params = _scope(report.project_id, report.graph_id)
                report.chunks, report.already = _counts(cur, scope, params)
                filled = _filled(cur, scope, params) if report.verify else []
                return _pending(cur, scope, params), filled
    finally:
        pool.putconn(conn)


def run_backfill(
    pool: Any,
    project_id: str,
    *,
    load_document: DocumentLoader,
    graph: str | None = None,
    dry_run: bool = False,
    batch_size: int = 500,
    verify: bool = False,
) -> BackfillReport:
    """Fill ``search_text`` for the scope's chunks from md5-verified text;
    with ``verify``, first repair or clear filled rows the contract disowns."""
    report = BackfillReport(project_id=project_id, dry_run=dry_run, verify=verify)
    documents = _Documents(load_document)
    pending, filled = _read(pool, report, graph)
    resets = _verify(filled, documents, report)
    updates: list[tuple[str, str, str]] = []
    for chunk in pending:
        text, outcome = _search_text(chunk, documents)
        report.outcomes[outcome] += 1
        if text is not None:
            updates.append((text, chunk.node_id, chunk.md5))
    report.artifacts_read = documents.reads
    if not dry_run:
        report.reset = _write(pool, _RESET_SQL, resets, batch_size)
        report.written = _write(pool, _UPDATE_SQL, updates, batch_size)
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────


def _s3_document_loader() -> DocumentLoader:
    """The embed job's own text derivation (fetch, PDF/PPTX extraction, decode,
    strip), so spans index the same string the chunker saw."""
    from umrs.jobs.knowledge_adapters import (
        KnowledgeEmbedResult,
        _artifact_text,
        _build_s3_client,
    )

    client, bucket = _build_s3_client({})

    def load(artifact: dict[str, Any]) -> str | None:
        return _artifact_text(artifact, client, bucket, KnowledgeEmbedResult())

    return load


def _build_pool() -> Any:
    from psycopg2.pool import ThreadedConnectionPool

    from umrs.jobs.knowledge_adapters import _resolve_knowledge_dsn
    from umrs.shared.pg_settings import pg_connect_kwargs

    dsn = _resolve_knowledge_dsn()
    if not dsn:
        raise SystemExit(
            "no knowledge database configured (DATABASE_URL or DB_* variables)"
        )
    return ThreadedConnectionPool(
        0, 2, dsn, **pg_connect_kwargs("search-text-backfill")
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m umrs.knowledge.retrieval.backfill",
        description="Fill knowledge_nodes.search_text from md5-verified chunk text (no re-embed).",
    )
    parser.add_argument("--project", required=True, help="project id")
    parser.add_argument(
        "--graph", default=None, help="graph id or name (default: whole project)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="count only, write nothing"
    )
    parser.add_argument("--batch-size", type=int, default=500, help="rows per commit")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="also re-check filled rows against chunk_md5; repair or clear stale ones",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    pool = _build_pool()
    try:
        report = run_backfill(
            pool,
            args.project,
            load_document=_s3_document_loader(),
            graph=args.graph,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            verify=args.verify,
        )
    finally:
        pool.closeall()
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI path
    sys.exit(main())
