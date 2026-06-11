"""Persistent audit event store backed by Postgres.

Provides durable storage for audit events with indexed queries by
request_id, event, actor, project_id, and time range. The in-memory
ring buffer in audit_events.py remains as a fast-path for recent events;
this store is the durable persistence layer.

Schema:
    audit_events table — one row per audit event, indexed for trace queries.
    audit_schema_version table — tracks schema migrations.

Schema evolution:
    Each version bump is a forward-only migration. The ensure_audit_schema()
    function checks the current stored version and applies migrations in
    sequence. Downgrades are not supported — rollback by restoring from backup.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

AUDIT_STORE_SCHEMA_VERSION = 1

CREATE_AUDIT_EVENTS_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id VARCHAR(64) PRIMARY KEY,
    schema_version VARCHAR(8) NOT NULL DEFAULT '1',
    ts DOUBLE PRECISION NOT NULL,
    event VARCHAR(256) NOT NULL,
    actor VARCHAR(128) NOT NULL DEFAULT '',
    project_id VARCHAR(128) NOT NULL DEFAULT '',
    request_id VARCHAR(128) NOT NULL DEFAULT '',
    outcome VARCHAR(32) NOT NULL DEFAULT 'success',
    scope VARCHAR(32) NOT NULL DEFAULT '',
    details JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

CREATE_AUDIT_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_audit_events_ts ON audit_events(ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_events_request_id ON audit_events(request_id)
    WHERE request_id != '';
CREATE INDEX IF NOT EXISTS idx_audit_events_event ON audit_events(event);
CREATE INDEX IF NOT EXISTS idx_audit_events_actor ON audit_events(actor)
    WHERE actor != '';
CREATE INDEX IF NOT EXISTS idx_audit_events_project ON audit_events(project_id)
    WHERE project_id != '';
CREATE INDEX IF NOT EXISTS idx_audit_events_outcome ON audit_events(outcome);
"""

CREATE_AUDIT_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS audit_schema_version (
    id INTEGER PRIMARY KEY DEFAULT 1,
    version INTEGER NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    CHECK (id = 1)
);
"""


def ensure_audit_schema(conn) -> None:
    """Create audit tables and indexes if they don't exist.

    Performs forward-only schema migration: checks current stored version,
    applies any pending migrations, and updates the version record.

    Raises RuntimeError if the stored version is *newer* than the code
    (indicates a rollback scenario that needs manual intervention).
    """
    cur = conn.cursor()
    try:
        # Bootstrap tables
        cur.execute(CREATE_AUDIT_EVENTS_TABLE)
        cur.execute(CREATE_AUDIT_INDEXES)
        cur.execute(CREATE_AUDIT_SCHEMA_VERSION_TABLE)
        conn.commit()

        # Check existing version
        cur.execute("SELECT version FROM audit_schema_version WHERE id = 1")
        row = cur.fetchone()
        stored_version = row[0] if row else 0

        if stored_version > AUDIT_STORE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Audit schema version {stored_version} is newer than code version "
                f"{AUDIT_STORE_SCHEMA_VERSION}. Cannot downgrade. "
                "Deploy the newer code version or restore from backup."
            )

        if stored_version < AUDIT_STORE_SCHEMA_VERSION:
            _apply_audit_migrations(
                conn, cur, stored_version, AUDIT_STORE_SCHEMA_VERSION
            )

        # Upsert version
        cur.execute(
            "INSERT INTO audit_schema_version (id, version) VALUES (1, %s) "
            "ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version, "
            "updated_at = NOW()",
            (AUDIT_STORE_SCHEMA_VERSION,),
        )
        conn.commit()
        logger.info(
            "Audit schema ensured (version %d, migrated from %d)",
            AUDIT_STORE_SCHEMA_VERSION,
            stored_version,
        )
    except RuntimeError:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


def _apply_audit_migrations(conn, cur, from_version: int, to_version: int) -> None:
    """Apply sequential migrations between versions.

    Each migration is a function keyed by target version number.
    Add new migrations here as the schema evolves.
    """
    migrations: dict[int, str] = {
        # Version 1: initial schema (tables already created above)
        # Future example:
        # 2: "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS severity VARCHAR(16) DEFAULT 'info'",
    }

    for version in range(from_version + 1, to_version + 1):
        sql = migrations.get(version)
        if sql:
            logger.info("Applying audit schema migration to version %d", version)
            cur.execute(sql)
            conn.commit()


class AuditStore:
    """Postgres-backed persistent audit event store.

    Thread-safe: each method acquires and releases its own cursor.
    """

    def __init__(self, conn) -> None:
        self.conn = conn

    def write(self, record: dict[str, Any]) -> None:
        """Persist a single audit event record."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "INSERT INTO audit_events "
                "(event_id, schema_version, ts, event, actor, project_id, "
                " request_id, outcome, scope, details) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (event_id) DO NOTHING",
                (
                    record.get("event_id", ""),
                    record.get("schema_version", "1"),
                    record.get("ts", time.time()),
                    record.get("event", ""),
                    record.get("actor", ""),
                    record.get("project_id", ""),
                    record.get("request_id", ""),
                    record.get("outcome", "success"),
                    record.get("scope", ""),
                    json.dumps(record.get("details", {}), default=str),
                ),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            logger.debug("Audit write failed", exc_info=True)
        finally:
            cur.close()

    def query(
        self,
        *,
        request_id: str = "",
        event: str = "",
        actor: str = "",
        project_id: str = "",
        since: float = 0.0,
        until: float = 0.0,
        outcome: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query persisted audit events with indexed filters.

        Parameters
        ----------
        since : float
            Unix timestamp lower bound (inclusive).
        until : float
            Unix timestamp upper bound (inclusive).

        Returns most-recent-first, limited to ``limit`` rows.
        """
        conditions: list[str] = []
        params: list[Any] = []

        if request_id:
            conditions.append("request_id = %s")
            params.append(request_id)
        if event:
            conditions.append("event LIKE %s")
            params.append(f"%{event}%")
        if actor:
            conditions.append("actor = %s")
            params.append(actor)
        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)
        if since > 0:
            conditions.append("ts >= %s")
            params.append(since)
        if until > 0:
            conditions.append("ts <= %s")
            params.append(until)
        if outcome:
            conditions.append("outcome = %s")
            params.append(outcome)

        where = " AND ".join(conditions) if conditions else "TRUE"
        params.append(min(limit, 500))

        sql = (
            f"SELECT event_id, schema_version, ts, event, actor, project_id, "
            f"request_id, outcome, scope, details "
            f"FROM audit_events WHERE {where} "
            f"ORDER BY ts DESC LIMIT %s"
        )

        cur = self.conn.cursor()
        try:
            cur.execute(sql, params)
            rows = cur.fetchall()
        except Exception:
            self.conn.rollback()
            logger.debug("Audit query failed", exc_info=True)
            return []
        finally:
            cur.close()

        return [
            {
                "event_id": r[0],
                "schema_version": r[1],
                "ts": r[2],
                "event": r[3],
                "actor": r[4],
                "project_id": r[5],
                "request_id": r[6],
                "outcome": r[7],
                "scope": r[8],
                "details": r[9] if isinstance(r[9], dict) else {},
            }
            for r in rows
        ]

    def count(self) -> int:
        """Return total number of persisted audit events."""
        cur = self.conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM audit_events")
            return cur.fetchone()[0]
        except Exception:
            self.conn.rollback()
            return 0
        finally:
            cur.close()

    def delete_before(self, cutoff_ts: float) -> int:
        """Delete audit events older than cutoff_ts. Returns count deleted."""
        cur = self.conn.cursor()
        try:
            cur.execute(
                "DELETE FROM audit_events WHERE ts < %s",
                (cutoff_ts,),
            )
            count = cur.rowcount
            self.conn.commit()
            return count
        except Exception:
            self.conn.rollback()
            logger.warning("Audit retention cleanup failed", exc_info=True)
            return 0
        finally:
            cur.close()
