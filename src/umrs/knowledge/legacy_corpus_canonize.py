"""Canonize legacy Gen1 rag corpora into the canonical knowledge spine
(ADR-027 G5 — 2026-08-23 live-survey wave).

Legacy corpora (surveyed on data0: ``default_demo_user_demo_user.text`` ×200,
``default_mnagaso3_mnagaso3.text`` ×158, both ``intfloat/e5-base-v2``) were
never minted into canonical nodes, so the md5-join backfill
(``node_embedding_backfill``) has no targets. This runner is the missing
first step: derive canonical artifact + document node + chunk nodes +
part_of edges + migration-0075 embeddings DIRECTLY from the legacy table.

Document identity: Gen1 tables carry no document_id, and their
``s3_object`` is PER-CHUNK (either ``path/name.py#chunk<i>`` or an
md5-named object per chunk — both shapes surveyed live 2026-08-23).
The universal per-document key is ``checksum`` (fallback: fragment-
stripped s3_object); chunk order comes from ``chunk_start`` (fallback:
row ``id``). All writes reuse the executor mint SQL shapes
(``rag_upload_mint`` helpers) with origin ``rag.legacy-canonize`` for
provenance, and are upsert-shaped, so the runner is idempotent and
safe to re-run.

Usage (dry-run first, then --apply)::

    python -m umrs.knowledge.legacy_corpus_canonize \
        --source-dsn postgresql://umrs:***@127.0.0.1:5432/default_demo_user_demo_user \
        --dsn postgresql://umrs:***@127.0.0.1:5432/umrs_db \
        --project-id default --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from typing import Any

from umrs.knowledge.graph_registry import default_graph_id
from umrs.knowledge.rag_upload_mint import (
    _upsert_artifact,
    _upsert_embedding,
    _upsert_node,
    _upsert_part_of_edge,
)

logger = logging.getLogger("umrs.knowledge.legacy_corpus_canonize")

ORIGIN = "rag.legacy-canonize"

_SELECT_ROWS_SQL = """
SELECT s3_object, checksum, chunk_start, chunk_end, chunk_md5, model, embedding, _content
FROM {table}
WHERE s3_object IS NOT NULL AND chunk_md5 IS NOT NULL AND chunk_md5 <> ''
  AND embedding IS NOT NULL AND model IS NOT NULL
ORDER BY checksum, chunk_start NULLS LAST, id
"""


def _strip_fragment(s3_object: str) -> str:
    """Per-chunk uris carry `#chunk<i>` — strip to the document path."""
    return s3_object.split("#", 1)[0]


def _legacy_document_id(doc_key: str) -> str:
    """doc_key = document checksum (or fragment-stripped s3_object fallback)."""
    return hashlib.md5(doc_key.encode("utf-8")).hexdigest()


def _doc_label(doc_uri: str) -> str:
    return doc_uri.rsplit("/", 1)[-1] or doc_uri


def _as_float_list(raw: Any) -> list[float]:
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",") if x.strip()]
    return [float(v) for v in raw]


def read_corpus(source_pool: Any, table: str) -> dict[str, dict[str, Any]]:
    """Group legacy rows per document (checksum key), chunk-ordered.

    Returns {doc_key: {"uri": doc uri, "rows": [chunk dicts, ordered]}}.
    """
    docs: dict[str, dict[str, Any]] = {}
    conn = source_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(_SELECT_ROWS_SQL.format(table=table))
            for (
                s3_object,
                checksum,
                start,
                end,
                md5,
                model,
                emb,
                content,
            ) in cur.fetchall():
                doc_uri = _strip_fragment(s3_object)
                key = checksum or doc_uri
                entry = docs.setdefault(key, {"uri": doc_uri, "rows": []})
                entry["rows"].append(
                    {
                        "chunk_start": start,
                        "chunk_end": end,
                        "chunk_md5": md5,
                        "model": model,
                        "embedding": _as_float_list(emb),
                        "content": content,
                        "s3_chunk": s3_object,
                    }
                )
        return docs
    finally:
        source_pool.putconn(conn)


def canonize_document(
    canonical_conn: Any,
    *,
    doc_key: str,
    doc_uri: str,
    chunk_rows: list[dict[str, Any]],
    project_id: str,
    max_per_doc: int,
) -> int:
    """Mint one legacy document; returns chunk-node count."""
    document_id = f"legacy-{_legacy_document_id(doc_key)}"
    doc_node_id = f"ragdoc-{document_id}"
    gid = default_graph_id(project_id)
    filename = _doc_label(doc_uri)
    content_hash = hashlib.sha256(f"{document_id}:{filename}".encode()).hexdigest()[:16]
    artifact_id = f"artifact-rag-{document_id}"
    with canonical_conn.cursor() as cur:
        _upsert_artifact(
            cur,
            artifact_id=artifact_id,
            content_hash=content_hash,
            project_id=project_id,
            document_id=document_id,
            filename=filename,
            mime_type="application/octet-stream",
            origin=ORIGIN,
            extra_node_content={"s3_object": doc_uri},
        )
        _upsert_node(
            cur,
            node_id=doc_node_id,
            project_id=project_id,
            label=filename,
            node_type="file",
            content={
                "kind": "document",
                "document_id": document_id,
                "filename": filename,
                "origin": ORIGIN,
                "artifact_id": artifact_id,
                "s3_object": doc_uri,
            },
            content_hash=content_hash,
            graph_id=gid,
        )
        minted = 0
        for i, row in enumerate(chunk_rows[:max_per_doc]):
            chunk_id = f"{doc_node_id}-chunk{i}"
            _upsert_node(
                cur,
                node_id=chunk_id,
                project_id=project_id,
                label=f"chunk {i} of {filename}",
                node_type="entity",
                content={
                    "kind": "chunk",
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_start": row["chunk_start"],
                    "chunk_end": row["chunk_end"],
                    "chunk_md5": row["chunk_md5"],
                    "s3_chunk": row["s3_chunk"],
                    # The passage itself: a legacy chunk has no artifact to slice
                    # it from, and spine answers drop a text-less chunk.
                    "text": row.get("content") or "",
                    "origin": ORIGIN,
                },
                content_hash=row["chunk_md5"],
                graph_id=gid,
            )
            _upsert_part_of_edge(
                cur,
                chunk_id=chunk_id,
                doc_node_id=doc_node_id,
                project_id=project_id,
                graph_id=gid,
                origin=ORIGIN,
            )
            _upsert_embedding(
                cur,
                chunk_id=chunk_id,
                model_name=row["model"],
                embedding=row["embedding"],
                content_hash=row["chunk_md5"],
            )
            minted += 1
    return minted


def _build_pool(dsn: str):
    from psycopg2 import pool as pg_pool

    from umrs.shared.pg_settings import pg_connect_kwargs

    return pg_pool.ThreadedConnectionPool(
        minconn=1, maxconn=2, dsn=dsn, **pg_connect_kwargs("corpus-canonize")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    ap.add_argument("--source-dsn", required=True, help="Legacy corpus database DSN")
    ap.add_argument("--dsn", required=True, help="Canonical umrs_db DSN")
    ap.add_argument(
        "--table", default="text", help="Legacy chunk table (default: text)"
    )
    ap.add_argument(
        "--project-id", required=True, help="Canonical project for minted nodes"
    )
    ap.add_argument(
        "--max-per-doc", type=int, default=500, help="Sanity cap per document"
    )
    ap.add_argument("--apply", action="store_true", help="Write (default: dry-run)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    source_pool = _build_pool(args.source_dsn)
    canonical_pool = _build_pool(args.dsn)
    try:
        docs = read_corpus(source_pool, args.table)
        total_chunks = sum(len(entry["rows"]) for entry in docs.values())
        print(
            f"plan: {len(docs)} documents, {total_chunks} chunks (project={args.project_id})"
        )
        if args.verbose:
            for entry in docs.values():
                print(f"  {entry['uri']}: {len(entry['rows'])} chunks")
        if not args.apply:
            print("dry-run: pass --apply to write")
            return 0
        written_docs = 0
        written_chunks = 0
        for doc_key, entry in docs.items():
            conn = canonical_pool.getconn()
            try:
                written_chunks += canonize_document(
                    conn,
                    doc_key=doc_key,
                    doc_uri=entry["uri"],
                    chunk_rows=entry["rows"],
                    project_id=args.project_id,
                    max_per_doc=args.max_per_doc,
                )
                conn.commit()
                written_docs += 1
            except Exception:
                conn.rollback()
                raise
            finally:
                canonical_pool.putconn(conn)
        print(
            f"applied: {written_docs} documents, {written_chunks} chunk nodes+vectors"
        )
        return 0
    finally:
        source_pool.closeall()
        canonical_pool.closeall()


if __name__ == "__main__":  # pragma: no cover — CLI path
    raise SystemExit(main())
