"""Catalogue→canonical link hydration (ADR-031 plane hierarchy).

Canonical anchor direction (never upward): a catalogue row resolves to a
canonical *document* by way of `document_id`, never through raw corpus
identifiers or S3-path hacks. Two trustworthy link findings exist today:

1. **s3_object lane** — legacy chunk nodes minted with
   `content->>'s3_object'` exactly matching the artefact key. Most direct.
2. **filename lane** — upload-minted doc nodes with
   `content->>'filename'` equal to the artefact basename. Filename-only
   (upload mux tickets no key) — reported as such.

Counts are per-plane (chunk nodes / vector rows), "server-side read
projections" — NOT a new plane, never persisted in catalogue state.
Failing any of pool, row access, or match resolution → `None`,
fail-open per the ADR (links absent and NOT an error).
"""

from __future__ import annotations

import posixpath
from typing import Any

_DOC_BY_S3_OBJECT_SQL = """
SELECT DISTINCT content->>'document_id'
  FROM knowledge_nodes
 WHERE project_id = %s
   AND content->>'kind' = 'document'
   AND content->>'s3_object' = %s
 LIMIT 1
"""

_DOC_BY_FILENAME_SQL = """
SELECT content->>'document_id' AS document_id, count(*) AS n
  FROM knowledge_nodes
 WHERE project_id = %s
   AND (content->>'kind' = 'document' OR node_type = 'file')
   AND content->>'filename' = %s
 GROUP BY content->>'document_id'
"""

_CHUNK_COUNT_SQL = """
SELECT count(*)
  FROM knowledge_nodes
 WHERE project_id = %s
   AND content->>'kind' = 'chunk'
   AND content->>'document_id' = %s
"""

_VECTOR_STATS_SQL = """
SELECT e.model_name, e.dimension, count(*)
  FROM knowledge_node_embeddings e
 WHERE e.node_id LIKE %s
 GROUP BY e.model_name, e.dimension
"""


def _vector_like(doc_id: str) -> str:
    # Chunk ids mirror the executor mint shape: ragdoc-<document_id>-chunk<i>.
    return f"ragdoc-{doc_id}-chunk%"


def _resolve_document_id(pool: Any, project_id: str, s3_key: str) -> tuple[str | None, str | None]:
    """document_id + link_method, or (None, None) when no match."""
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(_DOC_BY_S3_OBJECT_SQL, (project_id, s3_key))
            row = cur.fetchone()
            if row and row[0]:
                return str(row[0]), "s3_object"

            filename = posixpath.basename(s3_key.rstrip("/")) or s3_key
            cur.execute(_DOC_BY_FILENAME_SQL, (project_id, filename))
            rows = cur.fetchall()
            if len(rows) == 1 and rows[0][0]:
                return str(rows[0][0]), "filename"
            return None, None
    finally:
        pool.putconn(conn)


def hydrate_links(pool: Any, *, project_id: str, s3_key: str) -> dict[str, Any] | None:
    """Server-side document→node→vector links for one catalogue row.

    Returns None when no canonical match exists or the canonical pool is
    absent — never raises (fail-open, see ADR-031 §Consequences.1).
    """
    if pool is None or not s3_key:
        return None
    try:
        doc_id, link_method = _resolve_document_id(pool, project_id, s3_key)
        if not doc_id:
            return None
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_CHUNK_COUNT_SQL, (project_id, doc_id))
                chunk_count = int(cur.fetchone()[0] or 0)
                cur.execute(_VECTOR_STATS_SQL, (_vector_like(doc_id),))
                vectors = [
                    {"model": model, "dimension": dim, "rows": count}
                    for model, dim, count in cur.fetchall()
                ]
        finally:
            pool.putconn(conn)
        return {
            "document_id": doc_id,
            "link_method": link_method,
            "chunk_nodes": chunk_count,
            "vector_rows": vectors,
        }
    except Exception:  # noqa: BLE001 — fail-open by contract
        return None
