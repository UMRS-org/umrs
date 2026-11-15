"""Durable ontology registry store + pure payload normalization (ADR-049 Leg A).

Revocation-shaped retirement (``retired_at``), never deletes — same
lifecycle law as asset_grants. A live (ontology_id, version) coordinate is
unique per scope: each project, and the global scope, has its own namespace
(the 0139 partial index on ``(ontology_id, version, coalesce(project_id,
''))``, which replaced the cross-project 0091 index). This store surfaces
the conflict as a typed error instead of leaking the PG message.

Lookups read the caller's project first, then the global scope; another
project's record never resolves. Retiring a version blocks new bindings
only: a graph already bound to it keeps resolving the retired record
(:meth:`OntologyRegistryStore.resolve_coordinate`).

Migration 0139 (ADR-116 D-4) adds ``spec`` (the full ``Ontology.to_dict()``,
so ``subtype_of``, ``extraction_fields`` and the ADR-116 fields survive a
round trip), ``source_format``, ``source_text`` and ``source_sha256``. Rows
written before 0139 have an empty ``spec`` and are read from the 0091
columns.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from .ontology import Ontology, ontology_from_dict


class OntologyCoordinateConflict(ValueError):
    """A live record of the same scope already owns this (ontology_id,
    version) coordinate."""


@dataclass(frozen=True)
class OntologyRecord:
    """One (possibly retired) registry row."""

    record_id: str
    ontology: Ontology
    owner_id: str
    project_id: str | None
    status: str
    retired_at: datetime | None
    created_at: datetime
    updated_at: datetime
    source_format: str = ""
    source_sha256: str | None = None
    source_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            **self.ontology.to_dict(),
            "owner_id": self.owner_id,
            "project_id": self.project_id,
            "status": self.status,
            "retired_at": _iso(self.retired_at),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "source_format": self.source_format,
            "source_sha256": self.source_sha256,
        }

    def summary(self) -> dict[str, Any]:
        """The record shape of the ontology API (ADR-116 D-5)."""
        return {
            "record_id": self.record_id,
            "ontology_id": self.ontology.ontology_id,
            "version": self.ontology.version,
            "description": self.ontology.description,
            "project_id": self.project_id,
            "owner_id": self.owner_id,
            "status": self.status,
            "created_at": _iso(self.created_at),
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
        }


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


_TEXT_COLUMNS = ("ontology_id", "version", "description")
_JSONB_COLUMNS = ("entity_types", "relation_types", "transitive_relations")


def _json_value(value: Any) -> Any:
    """jsonb arrives decoded (psycopg2, asyncpg codec) or as text — normalize."""
    return json.loads(value) if isinstance(value, str) else value


def _ontology_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """The serialized ontology of a row: 0091 columns overlaid by the 0139 spec."""
    raw = {key: row[key] for key in _TEXT_COLUMNS}
    raw.update({key: _json_value(row[key]) for key in _JSONB_COLUMNS})
    spec = _json_value(row.get("spec"))
    if isinstance(spec, Mapping) and spec:
        raw.update(spec)
        # The coordinate columns are authoritative (the live index keys on them).
        raw["ontology_id"] = row["ontology_id"]
        raw["version"] = row["version"]
    return raw


def _row_to_record(row: Mapping[str, Any]) -> OntologyRecord:
    return OntologyRecord(
        record_id=row["record_id"],
        ontology=ontology_from_dict(_ontology_payload(row)),
        owner_id=row["owner_id"],
        project_id=row["project_id"],
        status=row["status"],
        retired_at=row["retired_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        source_format=str(row.get("source_format") or ""),
        source_sha256=row.get("source_sha256"),
        source_text=row.get("source_text"),
    )


def source_sha256(source_text: str | None) -> str | None:
    """sha256 hex of the registered source text (None without a source)."""
    if source_text is None:
        return None
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


def _version_key(version: str) -> tuple[tuple[int, Any], ...]:
    """Sort key ordering ``2`` before ``10`` and ``1.2`` before ``1.10``."""
    parts = re.split(r"[.+_-]", version)
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts)


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

    def _records_at(self, ontology_id: str, version: str) -> list[OntologyRecord]:
        """Every row (live or retired, any scope) at this coordinate."""
        cur = _execute(
            self._conn,
            """SELECT * FROM knowledge_ontologies
               WHERE ontology_id = %s AND version = %s""",
            (ontology_id, version),
        )
        return [_row_to_record(_row_mapping(cur, row)) for row in cur.fetchall()]

    def live_coordinate(
        self, ontology_id: str, version: str, project_id: str | None = None
    ) -> OntologyRecord | None:
        """The live record a caller in ``project_id`` reads: the project's
        own, else the global one (None = global only)."""
        records = self._records_at(ontology_id, version)
        return _pick(records, project_id, live_only=True)

    def resolve_coordinate(
        self, ontology_id: str, version: str, project_id: str | None = None
    ) -> OntologyRecord | None:
        """The record a graph of ``project_id`` pinned to this coordinate
        reads: the live one (as :meth:`live_coordinate`), else the newest
        retired one of the project, then of the global scope. Retiring
        blocks new bindings only; graphs bound before keep their ontology."""
        records = self._records_at(ontology_id, version)
        return _pick(records, project_id, live_only=False)

    def get_for_read(
        self, ontology_id: str, version: str, project_id: str | None = None
    ) -> OntologyRecord | None:
        """Alias for live_coordinate."""
        return self.live_coordinate(ontology_id, version, project_id)

    def latest_live_version(
        self, ontology_id: str, project_id: str | None = None
    ) -> str | None:
        """Highest live version of ``ontology_id`` that ``project_id`` reads
        (its own and global records; None = global only), or None.

        Versions compare numerically part by part (``2`` before ``10``).
        """
        cur = _execute(
            self._conn,
            "SELECT version, project_id, created_at FROM knowledge_ontologies "
            "WHERE ontology_id = %s AND retired_at IS NULL",
            (ontology_id,),
        )
        scope = project_id or None
        rows = [_row_mapping(cur, row) for row in cur.fetchall()]
        rows = [r for r in rows if r["project_id"] in (None, scope)]
        if not rows:
            return None
        best = max(
            rows, key=lambda r: (_version_key(str(r["version"])), str(r["created_at"]))
        )
        return str(best["version"])

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
        source_format: str = "",
        source_text: str | None = None,
    ) -> OntologyRecord:
        """Insert a live record; 409-shaped conflict when the same scope
        (``project_id``, None = global) already has this coordinate live.
        Another project's record is neither a conflict nor named.

        ``spec`` stores the full ``ontology.to_dict()``; the source text and
        its sha256 are kept for ``show`` and audit (ADR-116 D-4).
        """
        scope = project_id or None
        records = self._records_at(ontology.ontology_id, ontology.version)
        live = [r for r in records if r.retired_at is None and r.project_id == scope]
        if live:
            where = f"project {scope!r}" if scope else "the global scope"
            raise OntologyCoordinateConflict(
                f"'{ontology.ontology_id}' v{ontology.version} is already registered "
                f"live in {where} (record {live[0].record_id}): raise the version, "
                "or retire it deliberately first"
            )
        record_id = (
            f"onto:{ontology.ontology_id}:{ontology.version}:{secrets.token_hex(6)}"
        )
        now = datetime.now(timezone.utc)
        sha = source_sha256(source_text)
        _execute(
            self._conn,
            """INSERT INTO knowledge_ontologies
               (record_id, ontology_id, version, entity_types, relation_types,
                transitive_relations, description, owner_id, project_id,
                status, retired_at, created_at, updated_at,
                spec, source_format, source_text, source_sha256)
               VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s,
                       'active', NULL, %s, %s, %s::jsonb, %s, %s, %s)""",
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
                json.dumps(ontology.to_dict()),
                source_format,
                source_text,
                sha,
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
            source_format=source_format,
            source_sha256=sha,
            source_text=source_text,
        )

    def retire(
        self, ontology_id: str, version: str, project_id: str | None = None
    ) -> OntologyRecord | None:
        """Tombstone the live record ``project_id`` reads (as
        :meth:`live_coordinate`); None when nothing live is registered.
        New bindings refuse it; graphs already bound to it keep resolving
        it (:meth:`resolve_coordinate`)."""
        row = self.live_coordinate(ontology_id, version, project_id)
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
        return replace(row, status="retired", retired_at=now, updated_at=now)


def _pick(
    records: list[OntologyRecord], project_id: str | None, *, live_only: bool
) -> OntologyRecord | None:
    """The record ``project_id`` reads among one coordinate's rows: live
    before retired, the project's own before global, then the newest.
    Another project's record is never picked."""
    scope = project_id or None
    readable = [
        r
        for r in records
        if r.project_id in (None, scope) and not (live_only and r.retired_at)
    ]
    newest_first = sorted(
        readable, key=lambda r: r.retired_at or r.created_at, reverse=True
    )
    ranked = sorted(
        newest_first, key=lambda r: (r.retired_at is not None, r.project_id is None)
    )
    return ranked[0] if ranked else None


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
    "source_sha256",
]
