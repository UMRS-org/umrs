"""I3: Resumable upload-manifest store.

One ``IntakeUploadManifest`` row per chunked upload. Store factory
mirrors the connectors sync-state pattern: PG when resolvable,
in-memory fallback otherwise (``UMRS_INTAKE_UPLOAD_STORE_BACKEND`` =
``pg`` | ``memory`` | ``auto``, default auto).

Manifest lifecycle::

    uploading -> assembled   (successful complete)
             -> failed       (assembly error, error_note set)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

STATUS_UPLOADING = "uploading"
STATUS_ASSEMBLED = "assembled"
STATUS_FAILED = "failed"

_STORE_BACKEND_ENV = "UMRS_INTAKE_UPLOAD_STORE_BACKEND"


@dataclass
class IntakeUploadManifest:
    """One chunked-upload manifest."""

    manifest_id: str
    project_id: str
    target_key: str
    expected_parts: int
    owner: str = ""
    target_bucket: str = ""
    received_parts: list[int] = field(default_factory=list)
    parts_detail: dict[str, Any] = field(default_factory=dict)
    status: str = STATUS_UPLOADING
    final_etag: str | None = None
    size_bytes: int | None = None
    error_note: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    assembled_at: float | None = None

    def missing_parts(self) -> list[int]:
        have = set(self.received_parts)
        return [n for n in range(1, self.expected_parts + 1) if n not in have]

    def is_complete(self) -> bool:
        return not self.missing_parts()

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "project_id": self.project_id,
            "owner": self.owner,
            "target_key": self.target_key,
            "target_bucket": self.target_bucket,
            "expected_parts": self.expected_parts,
            "received_parts": list(self.received_parts),
            "missing_parts": self.missing_parts(),
            "parts_detail": dict(self.parts_detail),
            "status": self.status,
            "final_etag": self.final_etag,
            "size_bytes": self.size_bytes,
            "error_note": self.error_note,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "assembled_at": self.assembled_at,
        }


def new_manifest(
    *, project_id: str, target_key: str, expected_parts: int, owner: str
) -> IntakeUploadManifest:
    return IntakeUploadManifest(
        manifest_id=f"ul-{uuid.uuid4().hex[:12]}",
        project_id=project_id,
        owner=owner,
        target_key=target_key,
        expected_parts=expected_parts,
    )


# ── In-memory store ───────────────────────────────────────────────────────────


class InMemoryIntakeUploadStore:
    """Process-local store (tests + degraded operation)."""

    def __init__(self) -> None:
        self._manifests: dict[str, IntakeUploadManifest] = {}

    def create(self, manifest: IntakeUploadManifest) -> str:
        self._manifests[manifest.manifest_id] = manifest
        return manifest.manifest_id

    def get(self, manifest_id: str) -> IntakeUploadManifest | None:
        return self._manifests.get(manifest_id)

    def save(self, manifest: IntakeUploadManifest) -> bool:
        if manifest.manifest_id not in self._manifests:
            return False
        manifest.updated_at = time.time()
        self._manifests[manifest.manifest_id] = manifest
        return True

    def list(
        self, *, project_id: str | None = None, limit: int = 50
    ) -> list[IntakeUploadManifest]:
        manifests = list(self._manifests.values())
        if project_id:
            manifests = [m for m in manifests if m.project_id == project_id]
        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests[:limit]


# ── Postgres store ────────────────────────────────────────────────────────────


class PgIntakeUploadStore:
    """Postgres-backed store (``intake_upload_manifests`` table)."""

    def __init__(self, db_pool: Any) -> None:
        self._pool = db_pool

    def create(self, manifest: IntakeUploadManifest) -> str:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO intake_upload_manifests
                       (manifest_id, project_id, owner, target_key, target_bucket,
                        expected_parts, received_parts, parts_detail, status,
                        created_at, updated_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s,
                               to_timestamp(%s), to_timestamp(%s))""",
                    (
                        manifest.manifest_id,
                        manifest.project_id,
                        manifest.owner,
                        manifest.target_key,
                        manifest.target_bucket,
                        manifest.expected_parts,
                        json.dumps(manifest.received_parts),
                        json.dumps(manifest.parts_detail),
                        manifest.status,
                        manifest.created_at,
                        manifest.updated_at,
                    ),
                )
            conn.commit()
            return manifest.manifest_id
        finally:
            self._pool.putconn(conn)

    def get(self, manifest_id: str) -> IntakeUploadManifest | None:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT manifest_id, project_id, owner, target_key,
                              target_bucket, expected_parts, received_parts,
                              parts_detail, status, final_etag, size_bytes,
                              error_note, created_at, updated_at, assembled_at
                       FROM intake_upload_manifests WHERE manifest_id = %s""",
                    (manifest_id,),
                )
                row = cur.fetchone()
        finally:
            self._pool.putconn(conn)
        if row is None:
            return None
        return _manifest_from_row(row)

    def save(self, manifest: IntakeUploadManifest) -> bool:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE intake_upload_manifests
                       SET received_parts = %s::jsonb, parts_detail = %s::jsonb,
                           status = %s, final_etag = %s, size_bytes = %s,
                           error_note = %s, updated_at = to_timestamp(%s),
                           assembled_at = CASE WHEN %s IS NULL THEN NULL
                                               ELSE to_timestamp(%s) END
                       WHERE manifest_id = %s""",
                    (
                        json.dumps(manifest.received_parts),
                        json.dumps(manifest.parts_detail),
                        manifest.status,
                        manifest.final_etag,
                        manifest.size_bytes,
                        manifest.error_note,
                        manifest.updated_at,
                        manifest.assembled_at,
                        manifest.assembled_at,
                        manifest.manifest_id,
                    ),
                )
                updated = cur.rowcount == 1
            conn.commit()
            return updated
        finally:
            self._pool.putconn(conn)

    def list(
        self, *, project_id: str | None = None, limit: int = 50
    ) -> list[IntakeUploadManifest]:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                if project_id:
                    cur.execute(
                        """SELECT manifest_id, project_id, owner, target_key,
                                  target_bucket, expected_parts, received_parts,
                                  parts_detail, status, final_etag, size_bytes,
                                  error_note, created_at, updated_at, assembled_at
                           FROM intake_upload_manifests
                           WHERE project_id = %s
                           ORDER BY created_at DESC LIMIT %s""",
                        (project_id, limit),
                    )
                else:
                    cur.execute(
                        """SELECT manifest_id, project_id, owner, target_key,
                                  target_bucket, expected_parts, received_parts,
                                  parts_detail, status, final_etag, size_bytes,
                                  error_note, created_at, updated_at, assembled_at
                           FROM intake_upload_manifests
                           ORDER BY created_at DESC LIMIT %s""",
                        (limit,),
                    )
                rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)
        return [_manifest_from_row(row) for row in rows]


def _manifest_from_row(row: Any) -> IntakeUploadManifest:
    received = row[6] if isinstance(row[6], list) else json.loads(row[6] or "[]")
    detail = row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}")
    return IntakeUploadManifest(
        manifest_id=row[0],
        project_id=row[1],
        owner=row[2] or "",
        target_key=row[3],
        target_bucket=row[4] or "",
        expected_parts=row[5],
        received_parts=list(received),
        parts_detail=dict(detail),
        status=row[8],
        final_etag=row[9],
        size_bytes=row[10],
        error_note=row[11],
        created_at=_ts_to_epoch(row[12]),
        updated_at=_ts_to_epoch(row[13]),
        assembled_at=_ts_to_epoch(row[14]) if row[14] else None,
    )


def _ts_to_epoch(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(value.timestamp())


# ── Factory ───────────────────────────────────────────────────────────────────


def build_default_store(db_pool: Any = None) -> Any:
    """Resolve the active store per env configuration."""
    backend = os.getenv(_STORE_BACKEND_ENV, "auto").lower()
    if backend == "memory":
        return InMemoryIntakeUploadStore()

    from ..shared.pg import resolve_satellite_pool

    pool = resolve_satellite_pool(backend, db_pool, "Intake upload manifests")
    if pool is not None:
        logger.info("Intake upload manifests: using Postgres store")
        return PgIntakeUploadStore(pool)

    if backend == "pg":
        raise RuntimeError("UMRS_INTAKE_UPLOAD_STORE_BACKEND=pg but no pool resolvable")
    logger.warning("Intake upload manifests: Postgres unavailable, in-memory fallback")
    return InMemoryIntakeUploadStore()


__all__ = [
    "InMemoryIntakeUploadStore",
    "IntakeUploadManifest",
    "PgIntakeUploadStore",
    "build_default_store",
    "new_manifest",
]
