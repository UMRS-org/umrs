"""Durable connector sync state store (P2).

Postgres-backed persistence for connector checkpoints, stats, dead letters,
and content-hash dedup records. Mirrors the satellite conventions:
``build_default_store()`` resolves PG when the environment provides a
database, else an in-memory reference implementation (test double).

Tables: ``connector_sync_state`` + ``connector_seen_hashes`` (migration 0063).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SyncStateRecord:
    """Durable view of one connector instance's sync state."""

    name: str
    connector_type: str
    cursor: str | None = None
    last_sync_at: str | None = None
    last_status: str = "never"  # never | ok | error
    docs_synced: int = 0
    errors_count: int = 0
    dlq: list[dict[str, str]] = field(default_factory=list)
    checkpoint_json: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InMemorySyncStateStore:
    """Reference implementation; test double matching the PG interface."""

    def __init__(self) -> None:
        self._rows: dict[str, SyncStateRecord] = {}
        self._hashes: dict[str, set[str]] = {}

    def get(self, name: str) -> SyncStateRecord | None:
        return self._rows.get(name)

    def put(self, record: SyncStateRecord) -> None:
        record.updated_at = _now()
        self._rows[record.name] = record

    def list(self) -> list[SyncStateRecord]:
        return sorted(self._rows.values(), key=lambda r: r.name)

    def seen(self, name: str, content_hash: str) -> bool:
        return content_hash in self._hashes.get(name, set())

    def mark_seen(self, name: str, content_hash: str) -> None:
        self._hashes.setdefault(name, set()).add(content_hash)

    def clear_seen(self, name: str) -> None:
        self._hashes.pop(name, None)


class PgSyncStateStore:
    """Durable sync-state store backed by Postgres (migration 0063)."""

    def __init__(self, db_pool: Any) -> None:
        self._pool = db_pool

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            with conn:
                yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _cursor(self):
        from psycopg2.extras import RealDictCursor

        with self._conn() as conn:
            yield conn.cursor(cursor_factory=RealDictCursor)

    # ── state rows ────────────────────────────────────────────────────────

    def get(self, name: str) -> SyncStateRecord | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM connector_sync_state WHERE name = %s",
                (name,),
            )
            row = cur.fetchone()
        return _row_to_record(row) if row else None

    def put(self, record: SyncStateRecord) -> None:
        record.updated_at = _now()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO connector_sync_state "
                "(name, connector_type, cursor, last_sync_at, last_status, "
                " docs_synced, errors_count, dlq, checkpoint_json, last_error, "
                " updated_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, "
                "%s, %s) "
                "ON CONFLICT (name) DO UPDATE SET "
                "connector_type = excluded.connector_type, "
                "cursor = excluded.cursor, "
                "last_sync_at = excluded.last_sync_at, "
                "last_status = excluded.last_status, "
                "docs_synced = excluded.docs_synced, "
                "errors_count = excluded.errors_count, "
                "dlq = excluded.dlq, "
                "checkpoint_json = excluded.checkpoint_json, "
                "last_error = excluded.last_error, "
                "updated_at = excluded.updated_at",
                (
                    record.name,
                    record.connector_type,
                    record.cursor,
                    record.last_sync_at,
                    record.last_status,
                    record.docs_synced,
                    record.errors_count,
                    json.dumps(record.dlq),
                    json.dumps(record.checkpoint_json),
                    record.last_error,
                    record.updated_at,
                ),
            )

    def list(self) -> list[SyncStateRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM connector_sync_state ORDER BY name ASC")
            return [_row_to_record(row) for row in cur.fetchall()]

    # ── dedup hashes ──────────────────────────────────────────────────────

    def seen(self, name: str, content_hash: str) -> bool:
        with self._cursor() as cur:
            cur.execute(
                "SELECT 1 FROM connector_seen_hashes "
                "WHERE name = %s AND content_hash = %s",
                (name, content_hash),
            )
            return cur.fetchone() is not None

    def mark_seen(self, name: str, content_hash: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO connector_seen_hashes (name, content_hash, seen_at) "
                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (name, content_hash, _now()),
            )

    def clear_seen(self, name: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM connector_seen_hashes WHERE name = %s", (name,))


def _row_to_record(row: dict[str, Any]) -> SyncStateRecord:
    def _json(key: str, default: Any) -> Any:
        value = row.get(key)
        if isinstance(value, str):
            return json.loads(value)
        return default if value is None else value

    return SyncStateRecord(
        name=str(row["name"]),
        connector_type=str(row["connector_type"]),
        cursor=row.get("cursor"),
        last_sync_at=row.get("last_sync_at"),
        last_status=str(row.get("last_status") or "never"),
        docs_synced=int(row.get("docs_synced") or 0),
        errors_count=int(row.get("errors_count") or 0),
        dlq=_json("dlq", []),
        checkpoint_json=_json("checkpoint_json", {}),
        last_error=row.get("last_error"),
        updated_at=str(row.get("updated_at") or ""),
    )


def build_default_store(db_pool: Any = None) -> Any:
    """Build the default connector sync-state store backend.

    Selection via ``UMRS_CONNECTOR_SYNC_STATE_BACKEND`` (default ``auto``):

      - ``pg``     — force Postgres (raises if unavailable)
      - ``memory`` — force the in-memory reference store
      - ``auto``   — Postgres when a pool is provided or env configured,
                     otherwise in-memory
    """
    from ..shared.pg import resolve_satellite_pool

    backend = os.environ.get("UMRS_CONNECTOR_SYNC_STATE_BACKEND", "auto").lower()
    if backend == "memory":
        return InMemorySyncStateStore()
    if backend not in ("auto", "pg"):
        logger.warning("Unknown store backend %r; using auto", backend)
        backend = "auto"

    pool = resolve_satellite_pool(backend, db_pool, "Connector sync state")
    if pool is None:
        if backend == "pg":
            raise RuntimeError(
                "UMRS_CONNECTOR_SYNC_STATE_BACKEND=pg but no db_pool resolvable"
            )
        return InMemorySyncStateStore()
    return PgSyncStateStore(pool)


__all__ = [
    "SyncStateRecord",
    "InMemorySyncStateStore",
    "PgSyncStateStore",
    "build_default_store",
]
