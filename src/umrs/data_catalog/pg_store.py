"""Postgres-backed catalog metadata store (source of truth)."""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .scoring import CatalogHit, allowed, score_meta, tokenize
from .schema import CatalogObjectMeta, CatalogState

logger = logging.getLogger(__name__)


def _dict_cursor(conn):
    from psycopg2.extras import RealDictCursor

    return conn.cursor(cursor_factory=RealDictCursor)


def derive_project_id(meta: CatalogObjectMeta, *, fallback: str | None = None) -> str:
    """Derive canonical project id from acl_scope or s3_key prefix."""
    scope = (meta.acl_scope or "").strip()
    if scope:
        return scope
    if "/" in meta.s3_key:
        return meta.s3_key.split("/", 1)[0]
    return fallback or ""


def _parse_updated_at(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value or "")


def _row_to_meta(row: dict[str, Any]) -> CatalogObjectMeta:
    tags = row.get("tags") or []
    filename_signals = row.get("filename_signals") or []
    if isinstance(tags, str):
        tags = json.loads(tags)
    if isinstance(filename_signals, str):
        filename_signals = json.loads(filename_signals)
    return CatalogObjectMeta(
        s3_key=str(row["s3_key"]),
        object_etag=str(row.get("object_etag") or ""),
        state=CatalogState(str(row["state"])),
        owner=str(row.get("owner") or ""),
        acl_scope=str(row.get("acl_scope") or ""),
        updated_at=_parse_updated_at(row.get("updated_at")),
        summary=str(row.get("summary") or ""),
        deterministic_summary=str(row.get("deterministic_summary") or ""),
        perception_summary=row.get("perception_summary"),
        perception_model=row.get("perception_model"),
        tags=list(tags),
        filename_signals=list(filename_signals),
        detected_modality=str(row.get("detected_modality") or "unknown"),
        extracted_text_ref=row.get("extracted_text_ref"),
        extracted_knowledge_ref=row.get("extracted_knowledge_ref"),
        error_note=row.get("error_note"),
    )


class CatalogMetaStore:
    """Durable catalog metadata store backed by ``catalog_objects``."""

    def __init__(self, db_pool: Any) -> None:
        self.db_pool = db_pool

    @contextmanager
    def _get_conn(self):
        conn = self.db_pool.getconn()
        try:
            yield conn
        finally:
            self.db_pool.putconn(conn)

    def upsert(self, meta: CatalogObjectMeta, project_id: str) -> None:
        """Insert or update a catalog metadata row."""
        canonical_project = (project_id or derive_project_id(meta)).strip()
        if not canonical_project:
            raise ValueError("project_id is required for catalog upsert")

        now = datetime.now(timezone.utc)
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO catalog_objects (
                    project_id, s3_key, object_etag, state, owner, acl_scope,
                    summary, deterministic_summary, perception_summary, perception_model,
                    tags, filename_signals, detected_modality,
                    extracted_text_ref, extracted_knowledge_ref, error_note,
                    updated_at, created_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s,
                    %s, %s, %s,
                    %s, %s
                )
                ON CONFLICT (project_id, s3_key) DO UPDATE SET
                    object_etag = EXCLUDED.object_etag,
                    state = EXCLUDED.state,
                    owner = EXCLUDED.owner,
                    acl_scope = EXCLUDED.acl_scope,
                    summary = EXCLUDED.summary,
                    deterministic_summary = EXCLUDED.deterministic_summary,
                    perception_summary = EXCLUDED.perception_summary,
                    perception_model = EXCLUDED.perception_model,
                    tags = EXCLUDED.tags,
                    filename_signals = EXCLUDED.filename_signals,
                    detected_modality = EXCLUDED.detected_modality,
                    extracted_text_ref = EXCLUDED.extracted_text_ref,
                    extracted_knowledge_ref = EXCLUDED.extracted_knowledge_ref,
                    error_note = EXCLUDED.error_note,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    canonical_project,
                    meta.s3_key,
                    meta.object_etag,
                    meta.state.value,
                    meta.owner,
                    meta.acl_scope,
                    meta.summary,
                    meta.deterministic_summary,
                    meta.perception_summary,
                    meta.perception_model,
                    json.dumps(meta.tags),
                    json.dumps(meta.filename_signals),
                    meta.detected_modality,
                    meta.extracted_text_ref,
                    meta.extracted_knowledge_ref,
                    meta.error_note,
                    now,
                    now,
                ),
            )
            conn.commit()

    def get(self, project_id: str, s3_key: str) -> CatalogObjectMeta | None:
        """Return metadata for one object after ACL check."""
        row = self._fetch_row(project_id, s3_key)
        if row is None:
            return None
        meta = _row_to_meta(row)
        if not allowed(meta, project_id):
            return None
        return meta

    def get_by_s3_key(self, s3_key: str) -> CatalogObjectMeta | None:
        """Return metadata by s3_key alone (internal / reconciliation use)."""
        with self._get_conn() as conn, _dict_cursor(conn) as cur:
            cur.execute(
                """
                SELECT * FROM catalog_objects
                WHERE s3_key = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (s3_key,),
            )
            row = cur.fetchone()
        return _row_to_meta(row) if row else None

    def _fetch_row(self, project_id: str, s3_key: str) -> dict[str, Any] | None:
        with self._get_conn() as conn, _dict_cursor(conn) as cur:
            cur.execute(
                """
                SELECT * FROM catalog_objects
                WHERE project_id = %s AND s3_key = %s
                """,
                (project_id, s3_key),
            )
            return cur.fetchone()

    def list_by_project(
        self,
        project_id: str,
        *,
        prefix: str | None = None,
        include_pending: bool = True,
        include_error: bool = False,
    ) -> list[CatalogObjectMeta]:
        """List catalog metadata visible to a project."""
        normalized_prefix = prefix or f"{project_id}/"
        if not normalized_prefix.endswith("/"):
            normalized_prefix = f"{normalized_prefix}/"

        clauses = [
            "(project_id = %s OR s3_key LIKE %s)",
        ]
        params: list[Any] = [project_id, f"{project_id}/%"]

        if prefix is not None:
            clauses.append("s3_key LIKE %s")
            params.append(f"{normalized_prefix}%")

        if not include_pending:
            clauses.append("state != 'pending'")
        if not include_error:
            clauses.append("state != 'error'")

        where_sql = " AND ".join(clauses)
        with self._get_conn() as conn, _dict_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT * FROM catalog_objects
                WHERE {where_sql}
                ORDER BY updated_at DESC
                """,
                tuple(params),
            )
            rows = cur.fetchall()

        metas: list[CatalogObjectMeta] = []
        for row in rows:
            meta = _row_to_meta(row)
            if not allowed(meta, project_id):
                continue
            if prefix is not None and not meta.s3_key.startswith(normalized_prefix):
                continue
            metas.append(meta)
        return metas

    def search(
        self,
        project_id: str,
        query: str,
        *,
        limit: int = 20,
    ) -> list[CatalogHit]:
        """Lexical search over catalog metadata with project ACL filtering."""
        tokens = tokenize(query)
        if not tokens:
            return []

        ts_query = " & ".join(_escape_ts_token(t) for t in tokens if t)
        params: list[Any] = [project_id, f"{project_id}/%"]
        fts_clause = ""
        if ts_query:
            fts_clause = "AND search_vector @@ to_tsquery('simple', %s)"
            params.append(ts_query)

        with self._get_conn() as conn, _dict_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT * FROM catalog_objects
                WHERE (project_id = %s OR s3_key LIKE %s)
                  AND state != 'error'
                  {fts_clause}
                ORDER BY updated_at DESC
                """,
                tuple(params),
            )
            rows = cur.fetchall()

        hits: list[CatalogHit] = []
        for row in rows:
            meta = _row_to_meta(row)
            if not allowed(meta, project_id):
                continue
            score = score_meta(meta, tokens)
            if score <= 0:
                continue
            hits.append(CatalogHit(meta=meta, score=score))

        if not hits and ts_query:
            return self._search_fallback(project_id, tokens, limit=limit)

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def _search_fallback(
        self,
        project_id: str,
        tokens: list[str],
        *,
        limit: int,
    ) -> list[CatalogHit]:
        """Score all project rows when FTS pre-filter returns no hits."""
        metas = self.list_by_project(project_id, include_pending=True, include_error=False)
        hits: list[CatalogHit] = []
        for meta in metas:
            score = score_meta(meta, tokens)
            if score <= 0:
                continue
            hits.append(CatalogHit(meta=meta, score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def delete(self, project_id: str, s3_key: str) -> bool:
        """Remove catalog metadata for a deleted object."""
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM catalog_objects
                WHERE project_id = %s AND s3_key = %s
                """,
                (project_id, s3_key),
            )
            deleted = cur.rowcount > 0
            conn.commit()
        return deleted

    def project_stats(self, project_id: str) -> dict[str, Any]:
        """Aggregate counts by state and modality (replaces S3 manifest)."""
        with self._get_conn() as conn, _dict_cursor(conn) as cur:
            cur.execute(
                """
                SELECT state, detected_modality, COUNT(*) AS count
                FROM catalog_objects
                WHERE project_id = %s OR s3_key LIKE %s
                GROUP BY state, detected_modality
                ORDER BY state, detected_modality
                """,
                (project_id, f"{project_id}/%"),
            )
            rows = cur.fetchall()

        by_state: dict[str, int] = {}
        by_modality: dict[str, int] = {}
        total = 0
        for row in rows:
            count = int(row["count"])
            total += count
            state = str(row["state"])
            modality = str(row["detected_modality"])
            by_state[state] = by_state.get(state, 0) + count
            by_modality[modality] = by_modality.get(modality, 0) + count

        return {
            "project_id": project_id,
            "object_count": total,
            "states": by_state,
            "modalities": by_modality,
        }

    def list_s3_keys_for_project(self, project_id: str) -> list[str]:
        """Return all cataloged s3_keys for reconciliation."""
        with self._get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT s3_key FROM catalog_objects
                WHERE project_id = %s OR s3_key LIKE %s
                """,
                (project_id, f"{project_id}/%"),
            )
            return [row[0] for row in cur.fetchall()]


def _escape_ts_token(token: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", token)
    return cleaned or token


@dataclass
class ReconcileResult:
    """Result of a catalog reconciliation sweep."""

    checked: int = 0
    deleted: int = 0
    missing_keys: list[str] | None = None


def reconcile_project(
    store: CatalogMetaStore,
    s3_client: Any,
    bucket: str,
    project_id: str,
    *,
    dry_run: bool = False,
) -> ReconcileResult:
    """Delete catalog rows whose S3 objects no longer exist."""
    keys = store.list_s3_keys_for_project(project_id)
    missing: list[str] = []
    deleted = 0

    for s3_key in keys:
        if s3_key.startswith("_data_catalog/"):
            continue
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
        except Exception:
            missing.append(s3_key)
            if not dry_run:
                if store.delete(project_id, s3_key):
                    deleted += 1

    return ReconcileResult(checked=len(keys), deleted=deleted, missing_keys=missing)
