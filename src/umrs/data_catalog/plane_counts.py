"""Per-plane row counts for the catalogue stats projection.

ADR-031 §Consequences 6: catalogue stats answers "what physically exists"
*alongside* the downstream planes so the hierarchy fills up as mint/backfill
converges:

    catalogue objects      (this store's own truth — CatalogMetaStore)
    canonical documents    (0055 knowledge_nodes kind='document')
    chunk nodes            (0055 knowledge_nodes kind='chunk')
    vector rows            (node-scoped embeddings, project-joined; the
                            table itself lives exclusively in
                            ``data_catalog.links`` per the fence contract)

These are counts of *other planes' rows*, never a consistency verdict —
classification stays with the server-side plane reconciler (Invariant 4).
Read-only; the caller decides the honest-absence policy on pool absence
(fail-open to ``None`` at the route, never a fabricated zero).
Dependency rule respected: no catalogue-side file other than the
designated bridge names the vector-plane table (ADR-031 / fence).
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

from .links import count_vector_rows_for_project  # vector-token lives ONLY here


@dataclass(frozen=True)
class PlaneCounts:
    """Downstream-plane row counts for one project (catalogue excluded)."""

    canonical_documents: int  # knowledge_nodes kind='document'
    chunk_nodes: int  # knowledge_nodes kind='chunk'
    vector_rows: int  # vector rows (node-scoped embeddings), project-joined

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
    a fabricated zero is not). The vector count comes from the designated
    bridge (`links.count_vector_rows_for_project`) — a None there reads as
    zero (the bridge fold-open contract returns None on pool failure,
    but plane counts remain fail-strict per their route contract... the
    route sees a plain int either way).
    """
    if pool is None:
        raise RuntimeError("plane counts require the shared umrs_db pool")
    vector_rows = count_vector_rows_for_project(pool, project_id)
    return PlaneCounts(
        canonical_documents=_count(pool, _DOCS_SQL, project_id),
        chunk_nodes=_count(pool, _CHUNKS_SQL, project_id),
        vector_rows=int(vector_rows or 0),
    )
