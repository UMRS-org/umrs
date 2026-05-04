"""pgvector connector — query and verify embedding tables."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PgVectorConnector:
    """Query pgvector databases for embedding verification.

    Args:
        host: PostgreSQL host.
        port: PostgreSQL port.
        user: Database superuser.
        password: Database password.
    """

    def __init__(self, host: str = "localhost", port: int = 5432,
                 user: str = "umrs", password: str = ""):
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    def _connect(self, dbname: str = "postgres"):
        import psycopg2
        return psycopg2.connect(
            host=self.host, port=self.port,
            user=self.user, password=self.password,
            dbname=dbname,
        )

    def list_user_databases(self) -> list[str]:
        """Return names of non-system databases."""
        system_dbs = {"postgres", "template0", "template1", "umrs"}
        conn = self._connect("postgres")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT datname FROM pg_database")
                return [row[0] for row in cur.fetchall()
                        if row[0] not in system_dbs]
        finally:
            conn.close()

    def count_embeddings(self, dbname: str,
                         table: str | None = None) -> dict[str, Any]:
        """Count total rows and non-null embeddings in *table*.

        If *table* is ``None``, scans all tables with an ``embedding`` column.

        Returns::

            {"tables": [
                {"name": "text", "total": 42, "with_embedding": 42, "nulls": 0},
                ...
            ]}
        """
        conn = self._connect(dbname)
        try:
            tables = [table] if table else self._find_embedding_tables(conn)
            result = []
            for tbl in tables:
                with conn.cursor() as cur:
                    cur.execute(
                        f'SELECT count(*), '
                        f'count(*) FILTER (WHERE embedding IS NOT NULL), '
                        f'count(*) FILTER (WHERE embedding IS NULL) '
                        f'FROM "{tbl}"'
                    )
                    total, with_emb, nulls = cur.fetchone()
                    result.append({
                        "name": tbl,
                        "total": total,
                        "with_embedding": with_emb,
                        "nulls": nulls,
                    })
            return {"tables": result}
        finally:
            conn.close()

    def verify_all(self) -> tuple[bool, str]:
        """Check all user databases for complete embeddings.

        Returns ``(ok, summary_text)``.
        """
        lines = []
        ok = True
        for db in self.list_user_databases():
            info = self.count_embeddings(db)
            for t in info["tables"]:
                tag = "OK" if t["nulls"] == 0 and t["total"] > 0 else "WARN"
                if tag == "WARN":
                    ok = False
                lines.append(
                    f"  {db}.{t['name']}: {t['with_embedding']}/{t['total']} "
                    f"embeddings ({t['nulls']} null) [{tag}]"
                )
        summary = "\n".join(lines) if lines else "  (no embedding tables found)"
        return ok, summary

    @staticmethod
    def _find_embedding_tables(conn) -> list[str]:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.columns "
                "WHERE column_name='embedding' AND table_schema='public'"
            )
            return [row[0] for row in cur.fetchall()]
