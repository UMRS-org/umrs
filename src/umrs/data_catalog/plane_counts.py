"""Per-plane row counts for the catalogue stats projection.

ADR-031 §Consequences 6: catalogue stats answers "what physically exists"
*alongside* the downstream planes so the hierarchy fills up as mint/backfill
converges:

    catalogue objects      (this store's own truth — CatalogMetaStore)
    canonical documents    (0055 knowledge_nodes kind='document')
    chunk nodes            (0055 knowledge_nodes kind='chunk')
    vector rows            (0075 knowledge_node_embeddings, project-joined)

These are counts of *other planes' rows*, never a consistency verdict —
classification stays with the server-side plane reconciler (Invariant 4).
Read-only; the caller decides the honest-absence policy on pool absence
(fail-open to ``None`` at the route, never a fabricated zero).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_DOCS_SQL = """
SELECT COUNT(*)
  FROM knowledge_nodes
 WHERE project_id = %s
   AND content->>'kind' = 'document'
"""

_CHUNKS_SQL = """
SELECT COUNT(*)
  FROM knowledge_nodes
 WHERE project_id = %s
   AND content->>'kind' = 'chunk'
"""

_VECTORS_SQL = """
SELECT COUNT(*)
  FROM knowledge_node_embeddings e
  JOIN knowledge_nodes n USING (node_id)
 WHERE n.project_id = %s
"""


@dataclass(frozen=True)
class PlaneCounts:
    """Downstream-plane row counts for one project (catalogue excluded)."""

    canonical_documents: int  # knowledge_nodes kind='document'
    chunk_nodes: int  # knowledge_nodes kind='chunk'
    vector_rows: int  # knowledge_node_embeddings (0075), project-joined

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _count(pool: Any, sql: str, project_id: str) -> int:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (project_id,))
            row = cur.fetchone()
    finally:
        pool.putconn(conn)
    return int(row[0]) if row else 0


def compute_plane_counts(pool: Any, project_id: str) -> PlaneCounts:
    """Count canonical + vector plane rows for ``project_id``.

    Read-only. Raises on pool/query failure — the route decides whether the
    stats projection renders these fields or omits them (omission is honest;
    a fabricated zero is not).
    """
    if pool is None:
        raise RuntimeError("plane counts require the shared umrs_db pool")
    return PlaneCounts(
        canonical_documents=_count(pool, _DOCS_SQL, project_id),
        chunk_nodes=_count(pool, _CHUNKS_SQL, project_id),
        vector_rows=_count(pool, _VECTORS_SQL, project_id),
    )
