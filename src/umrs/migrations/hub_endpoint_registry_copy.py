"""ADR-060 U-1 rollout helper: seed the hub endpoint registry ONCE per venue.

The four per-cluster ``front_*`` databases each held their own
``endpoint_registry`` table. With migration 0117 the shared ``umrs_db``
owns one venue-scoped table; this copies one old table's rows into it,
stamping the site's venue. Rows are read with the positional 16-column
contract (ADR-058) and written with ``venue`` as the 17th column.

One-shot (U-1 follow-up 2026-10-03, migration 0118). Since the U-1 flip
the servers write only ``umrs_db`` and delete rows on remove()/replace, so
the earlier copy-at-every-converge would resurrect removed endpoints with
stale host/port/status. One destination transaction now claims the venue's
``hub_registry_copy_markers`` row, inserts the rows and records the counts.
A venue that already has a marker is skipped before its source is opened
(its ``front_*`` database may be retired). Any error rolls the venue back
(no marker, no rows), prints one masked ``status=error`` line on stderr and
exits 1, and the converge fails on it.

CLI (run inside an umrs-server container, DSNs via env for secrecy)::

    SRC_DSN=postgres://... DST_DSN=postgres://... \
        python -m umrs.migrations.hub_endpoint_registry_copy --venue rccs-front

stdout carries one line per run:
``hub-copy venue=<v> status=copied|skipped|no-source-table source_rows=N
inserted=M [marker_at=<ts>]``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
_METADATA_INDEX = SOURCE_COLUMNS.index("metadata")
_INSERT = (
    "INSERT INTO endpoint_registry ("
    + ", ".join(SOURCE_COLUMNS)
    + ", venue) VALUES ("
    + ", ".join(["%s"] * (len(SOURCE_COLUMNS) + 1))
    + ") ON CONFLICT (job_id) DO NOTHING"
)
_SELECT = "SELECT " + ", ".join(SOURCE_COLUMNS) + " FROM endpoint_registry"

#: Claim the venue's marker inside the copy transaction. A concurrent run
#: blocks on the primary key until the first commits, then sees its marker.
_CLAIM = (
    "INSERT INTO hub_registry_copy_markers (venue, note) VALUES (%s, %s) "
    "ON CONFLICT (venue) DO NOTHING RETURNING venue"
)
_MARKER_AT = "SELECT copied_at FROM hub_registry_copy_markers WHERE venue = %s"
_RECORD = (
    "UPDATE hub_registry_copy_markers SET source_rows = %s, inserted = %s "
    "WHERE venue = %s"
)
_NOTE = "copied by umrs.migrations.hub_endpoint_registry_copy"
#: Greedy up to the LAST '@' of the URL token: the converge builds the DSN
#: with an unencoded password, which may itself contain '@'.
_DSN_SECRET = re.compile(r"postgres(?:ql)?://\S*@")


@dataclass(frozen=True)
class CopyResult:
    """What one venue's run did, printed as one machine-readable line."""

    venue: str
    status: str  # copied | skipped | no-source-table
    source_rows: int = 0
    inserted: int = 0
    marker_at: str = ""

    def line(self) -> str:
        """``hub-copy venue=… status=… source_rows=… inserted=…``."""
        text = (
            f"hub-copy venue={self.venue} status={self.status} "
            f"source_rows={self.source_rows} inserted={self.inserted}"
        )
        return f"{text} marker_at={self.marker_at}" if self.marker_at else text


def mask_dsn(text: str) -> str:
    """Replace the credentials of any Postgres URL in *text*."""
    return _DSN_SECRET.sub("postgres://***@", text)


def _table_exists(conn: Any, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table,))
        row = cur.fetchone()
    return bool(row and row[0])


def _source_rows(src_conn: Any) -> list[tuple[Any, ...]] | None:
    """The per-cluster table's rows; ``None`` when the table does not exist."""
    if not _table_exists(src_conn, "endpoint_registry"):
        return None
    with src_conn.cursor() as cur:
        cur.execute(_SELECT)
        return [tuple(row) for row in cur.fetchall()]


def _hub_params(row: Sequence[Any], venue: str) -> tuple[Any, ...]:
    """Source row → INSERT params, venue appended.

    psycopg2 reads JSONB ``metadata`` back as a dict and cannot adapt a
    dict as a parameter ("can't adapt type 'dict'"), so it is re-serialised.
    The pre-marker helper passed it through and raised on the first row of
    any non-empty source.
    """
    values = list(row)
    metadata = values[_METADATA_INDEX]
    if isinstance(metadata, (dict, list)):
        values[_METADATA_INDEX] = json.dumps(metadata)
    return (*values, venue)


def _claim_marker(dst_conn: Any, venue: str) -> str | None:
    """Claim *venue*'s marker in the open transaction.

    Returns ``None`` when this run claimed it, else the existing marker's
    timestamp.
    """
    with dst_conn.cursor() as cur:
        cur.execute(_CLAIM, (venue, _NOTE))
        if cur.fetchone():
            return None
        cur.execute(_MARKER_AT, (venue,))
        row = cur.fetchone()
    stamp = row[0] if row else None
    return stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp)


def _insert_and_record(dst_conn: Any, rows: list[tuple[Any, ...]], venue: str) -> int:
    """Insert *rows* venue-stamped and record the counts on the marker."""
    inserted = 0
    with dst_conn.cursor() as cur:
        for row in rows:
            cur.execute(_INSERT, _hub_params(row, venue))
            inserted += max(cur.rowcount, 0)
        cur.execute(_RECORD, (len(rows), inserted, venue))
    return inserted


def _copy_once(dst_conn: Any, venue: str, open_source: Callable[[], Any]) -> CopyResult:
    marker_at = _claim_marker(dst_conn, venue)
    if marker_at is not None:
        dst_conn.rollback()  # nothing was written; end the transaction
        return CopyResult(venue, "skipped", marker_at=marker_at)
    with open_source() as src_conn:
        rows = _source_rows(src_conn)
    status = "no-source-table" if rows is None else "copied"
    rows = rows or []
    inserted = _insert_and_record(dst_conn, rows, venue)
    dst_conn.commit()
    return CopyResult(venue, status, source_rows=len(rows), inserted=inserted)


def copy_endpoints(
    dst_conn: Any, *, venue: str, open_source: Callable[[], Any]
) -> CopyResult:
    """Copy *venue*'s per-cluster rows into the hub table at most once.

    *open_source* returns a context manager yielding the source connection;
    it is called only when this run claimed the marker. The marker, the
    rows and the counts commit together; on any error the transaction is
    rolled back and the error propagates.
    """
    try:
        result = _copy_once(dst_conn, venue, open_source)
    except Exception:
        dst_conn.rollback()
        raise
    logger.info("%s", result.line())
    return result


def _run(src_dsn: str, dst_dsn: str, venue: str) -> CopyResult:
    import psycopg2

    from umrs.shared.pg_settings import pg_connect_kwargs

    kwargs = pg_connect_kwargs("hub-registry-copy")

    def open_source() -> Any:
        conn = psycopg2.connect(src_dsn, **kwargs)
        conn.set_session(readonly=True)  # the frozen source is never written
        return contextlib.closing(conn)

    with contextlib.closing(psycopg2.connect(dst_dsn, **kwargs)) as dst:
        return copy_endpoints(dst, venue=venue, open_source=open_source)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: copy one per-cluster table into the hub registry, once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venue", required=True, help="site.yaml name to stamp")
    args = parser.parse_args(argv)

    src_dsn = os.environ.get("SRC_DSN", "").strip()
    dst_dsn = os.environ.get("DST_DSN", "").strip()
    if not src_dsn or not dst_dsn:
        parser.error("SRC_DSN and DST_DSN environment variables are required")

    try:
        result = _run(src_dsn, dst_dsn, args.venue)
    except Exception as exc:  # noqa: BLE001 — reported (masked) and exit 1
        detail = mask_dsn(f"{type(exc).__name__}: {exc}").replace("\n", " | ")
        print(
            f"hub-copy venue={args.venue} status=error error={detail}",
            file=sys.stderr,
        )
        return 1
    print(result.line())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
