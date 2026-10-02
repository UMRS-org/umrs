"""ADR-060 U-1 rollout helper: seed the hub endpoint registry once.

The four per-cluster ``front_*`` databases each held their own
``endpoint_registry`` table. With migration 0117 the shared ``umrs_db``
owns one venue-scoped table; this copies each old table's rows into it,
stamping the site's venue, ``ON CONFLICT (job_id) DO NOTHING`` — fully
idempotent, safe to run at every converge. Rows are read with the
positional 16-column contract (ADR-058) and written with ``venue`` as the
17th column.

CLI (run inside an umrs-server container, DSNs via env for secrecy)::

    SRC_DSN=postgres://... DST_DSN=postgres://... \
        python -m umrs.migrations.hub_endpoint_registry_copy --venue rccs-front
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

#: The 16 positional columns of the per-cluster registry tables (ADR-058).
SOURCE_COLUMNS: tuple[str, ...] = (
    "job_id",
    "project_id",
    "model",
    "host",
    "port",
    "status",
    "backend",
    "variant",
    "created_at",
    "ready_at",
    "last_used_at",
    "expires_at",
    "error_message",
    "consecutive_failures",
    "metadata",
    "backend_job_id",
)
_INSERT = (
    "INSERT INTO endpoint_registry ("
    + ", ".join(SOURCE_COLUMNS)
    + ", venue) VALUES ("
    + ", ".join(["%s"] * (len(SOURCE_COLUMNS) + 1))
    + ") ON CONFLICT (job_id) DO NOTHING"
)
_SELECT = "SELECT " + ", ".join(SOURCE_COLUMNS) + " FROM endpoint_registry"


def _table_exists(conn: Any, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        row = cur.fetchone()
    return bool(row and row[0])


def copy_endpoints(src_conn: Any, dst_conn: Any, *, venue: str) -> int:
    """Copy source rows into the hub table with *venue* stamped.

    Returns rows actually inserted (conflicts skipped). Missing source
    table (a cluster that never had one) is an honest 0.
    """
    if not _table_exists(src_conn, "endpoint_registry"):
        logger.info("source has no endpoint_registry table — nothing to copy")
        return 0
    with src_conn.cursor() as cur:
        cur.execute(_SELECT)
        rows = cur.fetchall()
    inserted = 0
    with dst_conn.cursor() as cur:
        for row in rows:
            cur.execute(_INSERT, tuple(row) + (venue,))
            inserted += cur.rowcount
    dst_conn.commit()
    logger.info(
        "hub endpoint registry copy: venue=%s source_rows=%d inserted=%d",
        venue,
        len(rows),
        inserted,
    )
    return inserted


def main(argv: list[str] | None = None) -> int:
    """CLI entry: copy one per-cluster table into the hub registry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", required=True, help="site.yaml name to stamp")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    src_dsn = os.environ.get("SRC_DSN", "").strip()
    dst_dsn = os.environ.get("DST_DSN", "").strip()
    if not src_dsn or not dst_dsn:
        parser.error("SRC_DSN and DST_DSN environment variables are required")

    import psycopg2

    from umrs.shared.pg_settings import pg_connect_kwargs

    kwargs = pg_connect_kwargs("hub-registry-copy")
    with psycopg2.connect(src_dsn, **kwargs) as src, psycopg2.connect(
        dst_dsn, **kwargs
    ) as dst:
        copy_endpoints(src, dst, venue=args.venue)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
