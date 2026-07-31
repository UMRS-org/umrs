"""Postgres-backed scientific memory store (durable K1 implementation).

Implements the same interface as :class:`InMemoryScientificMemoryStore`
against the ``scientific_memory_entries`` table (migration 0059). The
in-memory store remains the reference implementation and test fake.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from . import (
    DuplicateMemoryEntryError,
    MemoryEntry,
    MemoryEntryStatus,
    MemoryEntryType,
)

logger = logging.getLogger(__name__)

_COLUMNS = (
    "entry_id, project_id, experiment_id, entry_type, source, content, "
    "content_uri, summary, metadata, tags, status, confidence, refs, "
    "created_at, updated_at, created_by"
)


def _row_to_entry(row: dict[str, Any]) -> MemoryEntry:
    data = dict(row)
    data["references"] = data.pop("refs", []) or []
    for key in ("metadata", "tags", "references"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    return MemoryEntry.from_dict(data)


def _filters(clauses: list[str], params: list[Any]) -> tuple[str, list[Any]]:
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


class PgScientificMemoryStore:
    """Durable scientific memory store backed by Postgres."""

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

    def add(self, entry: MemoryEntry) -> str:
        """Add a new entry; raises DuplicateMemoryEntryError on conflict."""
        try:
            with self._cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO scientific_memory_entries ({_COLUMNS})
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                            %s, %s, %s::jsonb, %s, %s, %s)
                    """,
                    (
                        entry.entry_id,
                        entry.project_id,
                        entry.experiment_id,
                        entry.entry_type.value,
                        entry.source,
                        entry.content,
                        entry.content_uri,
                        entry.summary,
                        json.dumps(entry.metadata),
                        json.dumps(entry.tags),
                        entry.status.value,
                        entry.confidence,
                        json.dumps(entry.references),
                        entry.created_at,
                        entry.updated_at,
                        entry.created_by,
                    ),
                )
        except Exception as exc:
            if _is_unique_violation(exc):
                raise DuplicateMemoryEntryError(
                    f"Entry {entry.entry_id} already exists"
                ) from exc
            raise
        return entry.entry_id

    def get_creation_ids(self) -> list[str]:
        """Entry IDs in insertion order (parity with reference store's
        ``_creation_order`` reconstructed deterministically from the
        (created_at, entry_id) unique pair)."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT entry_id FROM scientific_memory_entries "
                "ORDER BY created_at ASC, entry_id ASC"
            )
            return [str(row["entry_id"]) for row in cur.fetchall()]

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Get an entry by ID, or None."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM scientific_memory_entries WHERE entry_id = %s",
                (entry_id,),
            )
            row = cur.fetchone()
        return _row_to_entry(row) if row else None

    def list_entries(
        self,
        *,
        project_id: str | None = None,
        entry_type: MemoryEntryType | None = None,
        status: MemoryEntryStatus | None = None,
        experiment_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEntry]:
        """List entries with optional filters, newest first."""
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if entry_type:
            clauses.append("entry_type = %s")
            params.append(entry_type.value)
        if status:
            clauses.append("status = %s")
            params.append(status.value)
        if experiment_id:
            clauses.append("experiment_id = %s")
            params.append(experiment_id)
        where, params = _filters(clauses, params)
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM scientific_memory_entries {where} "
                "ORDER BY created_at DESC LIMIT %s",
                params,
            )
            return [_row_to_entry(row) for row in cur.fetchall()]

    def search(
        self,
        query: str,
        *,
        project_id: str | None = None,
        entry_type: MemoryEntryType | None = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Search content, summary, or tags (case-insensitive substring)."""
        clauses = [
            "(lower(content) LIKE %s OR lower(summary) LIKE %s "
            "OR EXISTS (SELECT 1 FROM jsonb_array_elements_text(tags) t "
            "WHERE lower(t) LIKE %s))"
        ]
        needle = f"%{query.lower()}%"
        params: list[Any] = [needle, needle, needle]
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if entry_type:
            clauses.append("entry_type = %s")
            params.append(entry_type.value)
        where, params = _filters(clauses, params)
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM scientific_memory_entries {where} "
                "ORDER BY created_at DESC LIMIT %s",
                params,
            )
            return [_row_to_entry(row) for row in cur.fetchall()]

    def update_status(self, entry_id: str, status: MemoryEntryStatus) -> bool:
        """Update entry status; True when the entry exists."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE scientific_memory_entries SET status = %s, "
                "updated_at = %s WHERE entry_id = %s",
                (status.value, now, entry_id),
            )
            return cur.rowcount > 0

    def count(
        self,
        *,
        project_id: str | None = None,
        entry_type: MemoryEntryType | None = None,
        status: MemoryEntryStatus | None = None,
    ) -> int:
        """Count entries with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if entry_type:
            clauses.append("entry_type = %s")
            params.append(entry_type.value)
        if status:
            clauses.append("status = %s")
            params.append(status.value)
        where, params = _filters(clauses, params)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT count(*) AS n FROM scientific_memory_entries {where}",
                params,
            )
            return int(cur.fetchone()["n"])

    def get_timeline(self, project_id: str) -> list[MemoryEntry]:
        """All entries for a project, oldest first."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM scientific_memory_entries "
                "WHERE project_id = %s ORDER BY created_at ASC",
                (project_id,),
            )
            return [_row_to_entry(row) for row in cur.fetchall()]


def _is_unique_violation(exc: Exception) -> bool:
    try:
        from psycopg2 import errors as pg_errors

        return isinstance(exc, pg_errors.UniqueViolation)
    except ImportError:
        return False
