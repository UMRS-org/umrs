"""pgvector index maintenance — jobs-tier-safe home (ADR-053).

Rehomed from `umrs/server/sql.py`. Behavior byte-identical; the server
module re-exports for its existing callers.
"""

from __future__ import annotations

import psycopg2
from psycopg2 import sql

from umrs.shared.pg_config import get_db_config
from umrs.shared.pg_settings import pg_connect_kwargs


def ensure_pgvector_enabled(conn):
    """Ensure pgvector extension is installed."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()


def maintain_pgvector_index(
    db_name: str,
    table_name: str,
    *,
    rebuild: bool = False,
    index_name: str | None = None,
) -> dict:
    """
    Ensure the standard pgvector HNSW index exists for a table.

    Args:
        db_name: Database containing the target embedding table.
        table_name: Table whose ``embedding`` column should be indexed.
        rebuild: If true and the index already exists, run ``REINDEX INDEX``.
        index_name: Optional explicit index name. Defaults to the UMRS naming
            convention ``idx_<table>_embedding_hnsw``.

    Returns:
        Metrics describing the operation performed.
    """
    cfg = get_db_config()
    target_index = index_name or f"idx_{table_name}_embedding_hnsw"
    metrics = {
        "database": db_name,
        "table": table_name,
        "index_name": target_index,
        "rebuild_requested": rebuild,
    }

    conn = psycopg2.connect(
        dbname=db_name,
        user=cfg["superuser"],
        password=cfg["superuser_password"],
        host=cfg["host"],
        port=cfg["port"],
        **pg_connect_kwargs("maintenance"),
    )
    try:
        ensure_pgvector_enabled(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """,
                (table_name,),
            )
            if not cur.fetchone()[0]:
                raise ValueError(
                    f"Table '{table_name}' not found in database '{db_name}'"
                )

            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = %s
                      AND column_name = 'embedding'
                )
                """,
                (table_name,),
            )
            if not cur.fetchone()[0]:
                raise ValueError(
                    f"Table '{table_name}' has no 'embedding' column in '{db_name}'"
                )

            cur.execute(
                "SELECT 1 FROM pg_indexes WHERE tablename = %s AND indexname = %s",
                (table_name, target_index),
            )
            index_exists = cur.fetchone() is not None

            if not index_exists:
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw "
                        "(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
                    ).format(
                        sql.Identifier(target_index),
                        sql.Identifier(table_name),
                    )
                )
                metrics["action"] = "created"
            elif rebuild:
                cur.execute(
                    sql.SQL("REINDEX INDEX {}").format(sql.Identifier(target_index))
                )
                metrics["action"] = "reindexed"
            else:
                metrics["action"] = "verified"

            cur.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(table_name)))

        conn.commit()
        return metrics
    finally:
        conn.close()
