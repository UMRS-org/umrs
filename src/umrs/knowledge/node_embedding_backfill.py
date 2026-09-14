"""Backfill migration-0075 node embeddings from existing RAG corpora
(ADR-027 G5 / P2 wave 2).

Existing corpora predate the executor mint: their rag chunk rows hold live
vectors but the canonical `knowledge_nodes` (once minted by the proxy /
executor path) have no 0075 embeddings. Derivation is keyed on
``chunk_md5`` — the only field present in BOTH the canonical chunk node's
content JSON and the rag chunk row (Gen1 tables carry no document_id).

For every canonical chunk node (``content->>'kind' = 'chunk'``):

    node.chunk_md5  ==  rag_chunk.chunk_md5  →  (node_id, model, embedding)

Upsert into ``knowledge_node_embeddings``. Idempotent by PK design.
Dry-run by default at the CLI; ``--apply`` writes.

Usage::

    python -m umrs.knowledge.node_embedding_backfill \
        --dsn postgresql://umrs:***@127.0.0.1:5432/umrs_db \
        --tables vector_xp_text_1 --dry-run --verbose
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

logger = logging.getLogger("umrs.knowledge.node_embedding_backfill")

_SELECT_NODES_SQL = """
SELECT node_id, project_id, content->>'chunk_md5' AS chunk_md5
  FROM knowledge_nodes
  WHERE node_type = 'entity'
    AND content->>'kind' = 'chunk'
    AND content->>'origin' = 'rag.upload'
    AND content->>'chunk_md5' IS NOT NULL
    AND content->>'chunk_md5' <> ''
"""

_UPSERT_SQL = """
INSERT INTO knowledge_node_embeddings
    (node_id, model_name, dimension, embedding, content_hash)
VALUES (%(node_id)s, %(model_name)s, %(dimension)s, %(embedding)s::vector, %(md5)s)
ON CONFLICT (node_id, model_name) DO UPDATE SET
    dimension    = EXCLUDED.dimension,
    embedding    = EXCLUDED.embedding,
    content_hash = EXCLUDED.content_hash,
    updated_at   = now()
"""


def _rag_chunk_lookup(
    cur: Any, tables: list[str]
) -> dict[str, tuple[str, list[float]]]:
    """md5 → (model_name, embedding_list) from one or more rag tables."""
    out: dict[str, tuple[str, list[float]]] = {}
    for table in tables:
        cur.execute(
            "SELECT chunk_md5, model, embedding "
            "FROM " + table + " WHERE chunk_md5 IS NOT NULL AND embedding IS NOT NULL"
        )
        for md5, model, emb in cur.fetchall():
            if md5 and model and emb:
                out.setdefault(md5, (model, _as_float_list(emb)))
    return out


def _as_float_list(raw: Any) -> list[float]:
    """pgvector rows come back as '[...]' text or real lists depending on driver."""
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",") if x.strip()]
    return [float(v) for v in raw]


def _vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in embedding) + "]"


def plan_backfill(
    pool: Any,
    tables: list[str],
    *,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """Compute the upsert set without writing.

    Returns rows of {node_id, model_name, dimension, md5} ready to apply —
    nodes whose chunk_md5 resolves to a rag chunk vector. Nodes with no
    matching rag chunk are skipped (they were never embedded server-side).
    """
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            sql = _SELECT_NODES_SQL
            params: tuple = ()
            if project_id:
                sql += " AND project_id = %s"
                params = (project_id,)
            cur.execute(sql, params)
            node_rows = cur.fetchall()

            lookup = _rag_chunk_lookup(cur, tables)

        plan: list[dict[str, Any]] = []
        for node_id, proj, md5 in node_rows:
            hit = lookup.get(md5)
            if hit is None:
                continue
            model, emb = hit
            plan.append(
                {
                    "node_id": node_id,
                    "project_id": proj,
                    "model_name": model,
                    "dimension": len(emb),
                    "embedding": _vector_literal(emb),
                    "md5": md5,
                }
            )
        return plan
    finally:
        pool.putconn(conn)


def apply_backfill(
    pool: Any, plan: list[dict[str, Any]], *, batch_size: int = 500
) -> int:
    """Write the planned upserts; returns row count."""
    if not plan:
        return 0
    written = 0
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            for row in plan:
                cur.execute(_UPSERT_SQL, row)
                written += 1
                if written % batch_size == 0:
                    conn.commit()
            conn.commit()
        return written
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def _build_pool(dsn: str):
    from psycopg2 import pool as pg_pool

    from umrs.shared.pg_settings import pg_connect_kwargs

    return pg_pool.ThreadedConnectionPool(
        minconn=1, maxconn=2, dsn=dsn, **pg_connect_kwargs("embedding-backfill")
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    ap.add_argument(
        "--dsn", required=True, help="PostgreSQL DSN against the central umrs_db"
    )
    ap.add_argument(
        "--tables",
        nargs="+",
        required=True,
        help="RAG chunk tables to read vectors from (e.g. vector_xp_<name>)",
    )
    ap.add_argument("--project-id", default=None, help="Limit backfill to one project")
    ap.add_argument("--apply", action="store_true", help="Write (default: dry-run)")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    pool = _build_pool(args.dsn)
    try:
        plan = plan_backfill(pool, args.tables, project_id=args.project_id)
        models = {(r["model_name"], r["dimension"]) for r in plan}
        print(f"plan: {len(plan)} rows | models/dims: {sorted(models) or '-'}")
        if not args.apply:
            print("dry-run: pass --apply to write")
            return 0
        written = apply_backfill(pool, plan, batch_size=args.batch_size)
        print(f"applied: {written} rows")
        return 0
    finally:
        pool.closeall()


if __name__ == "__main__":  # pragma: no cover — CLI path
    raise SystemExit(main())
