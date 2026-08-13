"""Postgres-backed knowledge model store (durable K1 implementation).

Implements the same interface as :class:`InMemoryKnowledgeModelStore`
against the ``sat_knowledge_objects`` / ``sat_knowledge_object_versions`` /
``sat_knowledge_tombstones`` tables (migration 0060). The in-memory store
remains the reference implementation and test fake.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from . import (
    DuplicateKnowledgeObjectError,
    ImmutableObjectError,
    KnowledgeObject,
    KnowledgeObjectNotFoundError,
    KnowledgeObjectStatus,
    KnowledgeObjectType,
    ObjectVersion,
    Tombstone,
)

logger = logging.getLogger(__name__)

_OBJ_COLUMNS = (
    "object_id, project_id, object_type, content, metadata, version, "
    "parent_version_id, status, is_immutable, created_at, updated_at, created_by"
)
_VER_COLUMNS = (
    "version_id, object_id, version, content, metadata, created_at, created_by"
)
_TOMB_COLUMNS = "object_id, object_type, deleted_at, deleted_by, deletion_reason"


def _sanitize_pg_json(candidate: Any) -> Any:
    """Replace NUL bytes (accepted by json.dumps but rejected by postgres
    jsonb) with the unicode replacement char, recursively."""
    if isinstance(candidate, str):
        return candidate.replace("\x00", "\ufffd")
    if isinstance(candidate, list):
        return [_sanitize_pg_json(v) for v in candidate]
    if isinstance(candidate, dict):
        out: dict[Any, Any] = {}
        for k, v in candidate.items():
            key = k.replace("\x00", "\ufffd") if isinstance(k, str) else k
            out[key] = _sanitize_pg_json(v)
        return out
    return candidate


def _row_to_object(row: dict[str, Any]) -> KnowledgeObject:
    data = dict(row)
    for key in ("content", "metadata"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    return KnowledgeObject.from_dict(data)


def _row_to_version(row: dict[str, Any]) -> ObjectVersion:
    data = dict(row)
    for key in ("content", "metadata"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    return ObjectVersion(
        version_id=str(data["version_id"]),
        object_id=str(data["object_id"]),
        version=int(data["version"]),
        content=dict(data["content"] or {}),
        metadata=dict(data["metadata"] or {}),
        created_at=str(data.get("created_at") or ""),
        created_by=str(data.get("created_by") or ""),
    )


def _row_to_tombstone(row: dict[str, Any]) -> Tombstone:
    return Tombstone(
        object_id=str(row["object_id"]),
        object_type=KnowledgeObjectType(str(row["object_type"])),
        deleted_at=str(row.get("deleted_at") or ""),
        deleted_by=str(row.get("deleted_by") or ""),
        deletion_reason=str(row.get("deletion_reason") or ""),
    )


class PgKnowledgeModelStore:
    """Durable knowledge model store backed by Postgres."""

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

    # ── writes ───────────────────────────────────────────────────────────

    def create_object(
        self,
        *,
        project_id: str,
        object_type: KnowledgeObjectType,
        content: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        is_immutable: bool = False,
        created_by: str = "",
    ) -> str:
        """Create an object + initial version; raises Duplicate on conflict."""
        # NUL-bytes are legal in JSON strings but postgres jsonb rejects them
        # (`psycopg2.errors.UntranslatableCharacter`). Real payloads arrive via
        # extracted knowledge from raw user uploads (live crash 2026-08-13).
        obj = KnowledgeObject(
            project_id=project_id,
            object_type=object_type,
            content=_sanitize_pg_json(candidate=content or {}),
            metadata=_sanitize_pg_json(candidate=metadata or {}),
            is_immutable=is_immutable,
            created_by=created_by,
        )
        try:
            with self._cursor() as cur:
                cur.execute(
                    f"INSERT INTO sat_knowledge_objects ({_OBJ_COLUMNS}) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, "
                    "%s, %s, %s)",
                    (
                        obj.object_id,
                        obj.project_id,
                        obj.object_type.value,
                        json.dumps(obj.content),
                        json.dumps(obj.metadata),
                        obj.version,
                        obj.parent_version_id,
                        obj.status.value,
                        obj.is_immutable,
                        obj.created_at,
                        obj.updated_at,
                        obj.created_by,
                    ),
                )
                self._insert_version(
                    cur,
                    obj.object_id,
                    1,
                    obj.content,
                    obj.metadata,
                    created_at=obj.created_at,
                    created_by=created_by,
                )
        except Exception as exc:
            if _is_unique_violation(exc):
                raise DuplicateKnowledgeObjectError(
                    f"Object {obj.object_id} already exists"
                ) from exc
            raise
        return obj.object_id

    def get_object(self, object_id: str) -> KnowledgeObject | None:
        """Get an object by ID, or None."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_OBJ_COLUMNS} FROM sat_knowledge_objects "
                "WHERE object_id = %s",
                (object_id,),
            )
            row = cur.fetchone()
        return _row_to_object(row) if row else None

    def update_object(
        self,
        object_id: str,
        *,
        content: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update content/metadata, creating a new version; True when found.

        Raises ImmutableObjectError for immutable objects.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT content, metadata, version, is_immutable "
                "FROM sat_knowledge_objects WHERE object_id = %s FOR UPDATE",
                (object_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            if row["is_immutable"]:
                raise ImmutableObjectError(
                    f"Object {object_id} is immutable and cannot be updated"
                )
            new_content = _coalesce_json(content, row["content"])
            new_metadata = _coalesce_json(metadata, row["metadata"])
            new_version = int(row["version"]) + 1
            now = datetime.now(timezone.utc).isoformat()
            cur.execute(
                "UPDATE sat_knowledge_objects SET content = %s::jsonb, "
                "metadata = %s::jsonb, version = %s, updated_at = %s "
                "WHERE object_id = %s",
                (
                    json.dumps(_sanitize_pg_json(new_content)),
                    json.dumps(_sanitize_pg_json(new_metadata)),
                    new_version,
                    now,
                    object_id,
                ),
            )
            self._insert_version(
                cur,
                object_id,
                new_version,
                new_content,
                new_metadata,
                created_at=now,
                created_by="",
            )
            return True

    @staticmethod
    def _insert_version(
        cur,
        object_id: str,
        version: int,
        content: dict[str, Any],
        metadata: dict[str, Any],
        *,
        created_at: str,
        created_by: str,
    ) -> None:
        cur.execute(
            f"INSERT INTO sat_knowledge_object_versions ({_VER_COLUMNS}) "
            "VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)",
            (
                f"ver-{uuid4().hex[:12]}",
                object_id,
                version,
                json.dumps(_sanitize_pg_json(content)),
                json.dumps(_sanitize_pg_json(metadata)),
                created_at,
                created_by,
            ),
        )

    def delete_object(
        self,
        object_id: str,
        *,
        deleted_by: str = "",
        deletion_reason: str = "",
    ) -> bool:
        """Soft delete via tombstone; True when the object existed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "SELECT object_type FROM sat_knowledge_objects "
                "WHERE object_id = %s FOR UPDATE",
                (object_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            cur.execute(
                f"INSERT INTO sat_knowledge_tombstones ({_TOMB_COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (object_id) DO UPDATE SET "
                "deleted_at = excluded.deleted_at, "
                "deleted_by = excluded.deleted_by, "
                "deletion_reason = excluded.deletion_reason",
                (object_id, str(row["object_type"]), now, deleted_by, deletion_reason),
            )
            cur.execute(
                "UPDATE sat_knowledge_objects SET status = %s, updated_at = %s "
                "WHERE object_id = %s",
                (KnowledgeObjectStatus.DELETED.value, now, object_id),
            )
            return True

    # ── reads ────────────────────────────────────────────────────────────

    def list_objects(
        self,
        *,
        project_id: str | None = None,
        object_type: KnowledgeObjectType | None = None,
        status: KnowledgeObjectStatus | None = None,
        limit: int = 100,
    ) -> list[KnowledgeObject]:
        """List objects (ACTIVE by default), newest first."""
        effective_status = (
            status if status is not None else KnowledgeObjectStatus.ACTIVE
        )
        clauses = ["status = %s"]
        params: list[Any] = [effective_status.value]
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if object_type:
            clauses.append("object_type = %s")
            params.append(object_type.value)
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_OBJ_COLUMNS} FROM sat_knowledge_objects "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY created_at DESC LIMIT %s",
                params,
            )
            return [_row_to_object(row) for row in cur.fetchall()]

    def search_objects(
        self,
        query: str,
        *,
        project_id: str | None = None,
        object_type: KnowledgeObjectType | None = None,
        limit: int = 20,
    ) -> list[KnowledgeObject]:
        """Search content, metadata, or object_id (case-insensitive)."""
        clauses = [
            "status != %s",
            "(lower(content::text) LIKE %s "
            "OR lower(metadata::text) LIKE %s OR lower(object_id) LIKE %s)",
        ]
        needle = f"%{query.lower()}%"
        params: list[Any] = [
            KnowledgeObjectStatus.DELETED.value,
            needle,
            needle,
            needle,
        ]
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if object_type:
            clauses.append("object_type = %s")
            params.append(object_type.value)
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_OBJ_COLUMNS} FROM sat_knowledge_objects "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY created_at DESC LIMIT %s",
                params,
            )
            return [_row_to_object(row) for row in cur.fetchall()]

    def get_object_history(self, object_id: str) -> list[ObjectVersion]:
        """Version history, ascending; raises when the object is missing."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT 1 FROM sat_knowledge_objects WHERE object_id = %s",
                (object_id,),
            )
            if not cur.fetchone():
                raise KnowledgeObjectNotFoundError(f"Object {object_id} not found")
            cur.execute(
                f"SELECT {_VER_COLUMNS} FROM sat_knowledge_object_versions "
                "WHERE object_id = %s ORDER BY version ASC",
                (object_id,),
            )
            return [_row_to_version(row) for row in cur.fetchall()]

    def get_tombstone(self, object_id: str) -> Tombstone | None:
        """Tombstone for a deleted object, or None."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_TOMB_COLUMNS} FROM sat_knowledge_tombstones "
                "WHERE object_id = %s",
                (object_id,),
            )
            row = cur.fetchone()
        return _row_to_tombstone(row) if row else None


def _coalesce_json(new: dict[str, Any] | None, current: Any) -> dict[str, Any]:
    """New value when provided, else parse the stored JSONB value."""
    if new is not None:
        return new
    if isinstance(current, str):
        return json.loads(current)
    return dict(current or {})


def _is_unique_violation(exc: Exception) -> bool:
    try:
        from psycopg2 import errors as pg_errors

        return isinstance(exc, pg_errors.UniqueViolation)
    except ImportError:
        return False
