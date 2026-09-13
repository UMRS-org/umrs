"""Durable ontology registry store + pure payload normalization (ADR-049 Leg A).

Revocation-shaped retirement (``retired_at``), never deletes — same
lifecycle law as asset_grants. Uniqueness of the live (ontology_id,
version) coordinate is enforced by the 0091 partial index; this store
surfaces the conflict as a typed error instead of leaking the PG message.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from .ontology import Ontology, ontology_from_dict


class OntologyCoordinateConflict(ValueError):
    """A live record already owns this (ontology_id, version) coordinate."""


@dataclass(frozen=True)
class OntologyRecord:
    """One (possibly retired) registry row."""

    record_id: str
    ontology: Ontology
    owner_id: str
    project_id: Optional[str]
    status: str
    retired_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            **self.ontology.to_dict(),
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "status": self.status,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


def _row_to_record(row: Mapping[str, Any]) -> OntologyRecord:
    raw = {k: row[k] for k in (
        "record_id", "ontology_id", "version", "entity_types", "relation_types",
        "transitive_relations", "description", "owner_id", "project_id",
        "status", "retired_at", "created_at", "updated_at",
    )}
    # jsonb columns arrive as list (asyncpg codec) or string (psycopg2) — normalize.
    for key in ("entity_types", "relation_types", "transitive_relations"):
        value = raw[key]
        if isinstance(value, str):
            import json

            raw[key] = json.loads(value)
    ontology = ontology_from_dict(raw)
    return OntologyRecord(
        record_id=raw["record_id"],
        ontology=ontology,
        owner_id=raw["owner_id"],
        project_id=raw["project_id"],
        status=raw["status"],
        retired_at=raw["retired_at"],
        created_at=raw["created_at"],
        updated_at=raw["updated_at"],
    )


def _execute(conn: Any, sql: str, params: tuple) -> Any:
    """psycopg/psycopg2 pool-agnostic execute returning a cursor."""
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur


class OntologyRegistryStore:
    """Sync store over a psycopg-shaped connection/connaccessor.

    Lifecycle: register → (live) → retire (tombstone). No DELETE anywhere;
    a retired coordinate may be re-registered (the live-index is partial).
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # -- reads ---------------------------------------------------------------

    def live_coordinate(self, ontology_id: str, version: str) -> Optional[OntologyRecord]:
        cur = _execute(
            self._conn,
            """SELECT * FROM knowledge_ontologies
               WHERE ontology_id = %s AND version = %s AND retired_at IS NULL""",
            (ontology_id, version),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_record(_row_mapping(cur, row))

    def get_for_read(self, ontology_id: str, version: str) -> Optional[OntologyRecord]:
        """Alias for live_coordinate: exclusively live rows ever resolve."""
        return self.live_coordinate(ontology_id, version)

    def list(
        self,
        *,
        project_id: str | None = None,
        owner_id: str | None = None,
        include_retired: bool = False,
    ) -> list[OntologyRecord]:
        sql = "SELECT * FROM knowledge_ontologies WHERE TRUE"
        params: list[Any] = []
        if not include_retired:
            sql += " AND retired_at IS NULL"
        if project_id is not None:
            sql += " AND (project_id = %s OR project_id IS NULL)"
            params.append(project_id)
        if owner_id is not None:
            sql += " AND owner_id = %s"
            params.append(owner_id)
        sql += " ORDER BY ontology_id, version, created_at"
        cur = _execute(self._conn, sql, tuple(params))
        return [_row_to_record(_row_mapping(cur, row)) for row in cur.fetchall()]

    # -- writes --------------------------------------------------------------

    def register(
        self,
        ontology: Ontology,
        *,
        owner_id: str,
        project_id: str | None = None,
    ) -> OntologyRecord:
        """Insert a live record; 409-shaped conflict on live-coordinate reuse."""
        live = self.live_coordinate(ontology.ontology_id, ontology.version)
        if live is not None:
            raise OntologyCoordinateConflict(
                f"'{ontology.ontology_id}' v{ontology.version} already registered "
                f"live (record {live.record_id}) — retire it deliberately first"
            )
        record_id = f"onto:{ontology.ontology_id}:{ontology.version}:{secrets.token_hex(6)}"
        import json

        now = datetime.now(timezone.utc)
        _execute(
            self._conn,
            """INSERT INTO knowledge_ontologies
               (record_id, ontology_id, version, entity_types, relation_types,
                transitive_relations, description, owner_id, project_id,
                status, retired_at, created_at, updated_at)
               VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s,
                       'active', NULL, %s, %s)""",
            (
                record_id,
                ontology.ontology_id,
                ontology.version,
                json.dumps(sorted(ontology.entity_types)),
                json.dumps(sorted(ontology.relation_types)),
                json.dumps(sorted(ontology.transitive_relations)),
                ontology.description,
                owner_id,
                project_id,
                now,
                now,
            ),
        )
        self._conn.commit()
        return OntologyRecord(
            record_id=record_id,
            ontology=ontology,
            owner_id=owner_id,
            project_id=project_id,
            status="active",
            retired_at=None,
            created_at=now,
            updated_at=now,
        )

    def retire(self, ontology_id: str, version: str) -> OntologyRecord | None:
        """Tombstone the live record; None when nothing live is registered."""
        row = self.live_coordinate(ontology_id, version)
        if row is None:
            return None
        now = datetime.now(timezone.utc)
        _execute(
            self._conn,
            """UPDATE knowledge_ontologies
               SET status = 'retired', retired_at = %s, updated_at = %s
               WHERE record_id = %s""",
            (now, now, row.record_id),
        )
        self._conn.commit()
        return OntologyRecord(
            record_id=row.record_id,
            ontology=row.ontology,
            owner_id=row.owner_id,
            project_id=row.project_id,
            status="retired",
            retired_at=now,
            created_at=row.created_at,
            updated_at=now,
        )


def _row_mapping(cur: Any, row: Any) -> Mapping[str, Any]:
    """Map a cursor row to a name-keyed mapping (psycopg2 tuple or dict-row)."""
    if isinstance(row, Mapping):
        return row
    columns = [c[0] for c in cur.description]
    return dict(zip(columns, row))


__all__ = [
    "OntologyCoordinateConflict",
    "OntologyRecord",
    "OntologyRegistryStore",
]
