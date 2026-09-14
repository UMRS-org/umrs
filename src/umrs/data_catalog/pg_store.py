"""Postgres-backed catalog metadata store (source of truth)."""

from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from umrs.knowledge.access_scope import VISIBILITY_SYSTEM, visibility_predicate

from .schema import CatalogObjectMeta, CatalogState
from .scoring import CatalogHit, allowed, score_meta, tokenize

logger = logging.getLogger(__name__)


def _dict_cursor(conn):
    from psycopg2.extras import RealDictCursor

    return conn.cursor(cursor_factory=RealDictCursor)


def _visibility_clause(include_ambient: bool) -> str:
    """ADR-050 tier gate for the catalog lane (R-02/0093).

    Ownership keeps its catalog shape (project row OR s3_key prefix);
    system rows widen; ambient rows leave unless the admin lane asks.
    """
    return visibility_predicate(
        owned_sql="(project_id = %s OR s3_key LIKE %s)",
        include_ambient=include_ambient,
    )


def _search_filters(
    project_id: str,
    *,
    modality: str | None,
    state: str | None,
    tag: str | None,
    prefix: str | None,
    include_ambient: bool,
) -> tuple[list[str], list[Any]]:
    """Assemble the WHERE clauses + params for a catalog search call."""
    clauses = [_visibility_clause(include_ambient), "state != 'error'"]
    params: list[Any] = [project_id, f"{_like_literal(project_id)}/%"]
    if modality:
        clauses.append("detected_modality = %s")
        params.append(modality)
    if state:
        clauses.append("state = %s")
        params.append(state)
    if tag:
        clauses.append("tags @> %s::jsonb")
        params.append(json.dumps([tag]))
    if prefix:
        normalized = prefix if prefix.endswith("/") else f"{prefix}/"
        clauses.append("s3_key LIKE %s")
        params.append(f"{_like_literal(normalized)}%")
    return clauses, params


def _like_literal(value: str) -> str:
    """Escape LIKE metacharacters in a literal used inside a LIKE pattern.

    Project ids and prefixes are literal text: an ``_`` in a project id must
    not act as a single-char wildcard (``my_proj/%`` used to also match
    ``myXproj/…`` rows at the SQL layer). The python-side ``allowed()``
    filter still re-checks visibility, but the over-match distorted the
    bounded candidate window (``_MAX_SEARCH_CANDIDATES``) and wasted rows.
    Postgres LIKE default escape character is backslash.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


# Upper bound on rows pulled from Postgres before Python-side scoring. Without
# this, a broad WHERE match (or the no-FTS-hit fallback) loads every row for
# a project into process memory before `limit` is applied in Python.
_MAX_SEARCH_CANDIDATES = 500


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


def _json_list(value: Any) -> list:
    """JSONB list column as a Python list (accepts a JSON string or None)."""
    if isinstance(value, str):
        value = json.loads(value)
    return list(value or [])


def _text(row: dict[str, Any], key: str, default: str = "") -> str:
    """Text column with NULL/empty collapsed to ``default``."""
    return str(row.get(key) or default)


def _row_to_meta(row: dict[str, Any]) -> CatalogObjectMeta:
    return CatalogObjectMeta(
        s3_key=str(row["s3_key"]),
        object_etag=_text(row, "object_etag"),
        state=CatalogState(str(row["state"])),
        owner=_text(row, "owner"),
        acl_scope=_text(row, "acl_scope"),
        updated_at=_parse_updated_at(row.get("updated_at")),
        summary=_text(row, "summary"),
        deterministic_summary=_text(row, "deterministic_summary"),
        perception_summary=row.get("perception_summary"),
        perception_model=row.get("perception_model"),
        tags=_json_list(row.get("tags")),
        filename_signals=_json_list(row.get("filename_signals")),
        detected_modality=_text(row, "detected_modality", "unknown"),
        extracted_text_ref=row.get("extracted_text_ref"),
        extracted_knowledge_ref=row.get("extracted_knowledge_ref"),
        error_note=row.get("error_note"),
    )


def _normalized_prefix(project_id: str, prefix: str | None) -> str:
    normalized = prefix or f"{project_id}/"
    return normalized if normalized.endswith("/") else f"{normalized}/"


def _list_where(
    project_id: str,
    *,
    prefix: str | None,
    normalized_prefix: str,
    include_pending: bool,
    include_error: bool,
) -> tuple[list[str], list[Any]]:
    """WHERE clauses + params for ``list_by_project`` (ownership + state filters)."""
    clauses = ["(project_id = %s OR s3_key LIKE %s)"]
    params: list[Any] = [project_id, f"{_like_literal(project_id)}/%"]
    if prefix is not None:
        clauses.append("s3_key LIKE %s")
        params.append(f"{_like_literal(normalized_prefix)}%")
    if not include_pending:
        clauses.append("state != 'pending'")
    if not include_error:
        clauses.append("state != 'error'")
    return clauses, params


def _fallback_match(
    meta: CatalogObjectMeta,
    *,
    modality: str | None,
    state: str | None,
    tag: str | None,
    prefix: str | None,
) -> bool:
    """Structured-filter re-check for the recency-window search fallback."""
    if modality and meta.detected_modality != modality:
        return False
    if state and meta.state.value != state:
        return False
    if tag and tag not in meta.tags:
        return False
    if prefix:
        normalized = prefix if prefix.endswith("/") else f"{prefix}/"
        return meta.s3_key.startswith(normalized)
    return True


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
        limit: int | None = None,
    ) -> list[CatalogObjectMeta]:
        """List catalog metadata visible to a project.

        ``limit`` is a SQL-level cap applied before Python filtering — pass it
        for recency-biased sampling (e.g. search fallback); leave it ``None``
        for callers that need the full authoritative artifact set.
        """
        normalized_prefix = _normalized_prefix(project_id, prefix)
        clauses, params = _list_where(
            project_id,
            prefix=prefix,
            normalized_prefix=normalized_prefix,
            include_pending=include_pending,
            include_error=include_error,
        )

        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT %s"
            params.append(limit)

        where_sql = " AND ".join(clauses)
        with self._get_conn() as conn, _dict_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT * FROM catalog_objects
                WHERE {where_sql}
                ORDER BY updated_at DESC
                {limit_sql}
                """,
                tuple(params),
            )
            rows = cur.fetchall()

        metas = [meta for meta in map(_row_to_meta, rows) if allowed(meta, project_id)]
        if prefix is None:
            return metas
        return [meta for meta in metas if meta.s3_key.startswith(normalized_prefix)]

    def search(
        self,
        project_id: str,
        query: str,
        *,
        limit: int = 20,
        modality: str | None = None,
        state: str | None = None,
        tag: str | None = None,
        prefix: str | None = None,
        include_ambient: bool = False,
    ) -> list[CatalogHit]:
        """Lexical search over catalog metadata with optional structured filters."""
        tokens = tokenize(query) if query.strip() else []

        clauses, params = _search_filters(
            project_id,
            modality=modality,
            state=state,
            tag=tag,
            prefix=prefix,
            include_ambient=include_ambient,
        )
        ts_query = " & ".join(_escape_ts_token(t) for t in tokens if t)
        fts_clause = ""
        if ts_query:
            fts_clause = "AND search_vector @@ to_tsquery('simple', %s)"
            params.append(ts_query)

        # Rank by FTS relevance (when a query is present) before truncating to
        # the candidate cap, so the bounded window keeps the best textual
        # matches rather than merely the most recently updated rows.
        order_sql = "updated_at DESC"
        order_params: list[Any] = []
        if ts_query:
            order_sql = "ts_rank_cd(search_vector, to_tsquery('simple', %s)) DESC, updated_at DESC"
            order_params.append(ts_query)

        where_sql = " AND ".join(clauses)
        exec_params = [*params, *order_params, _MAX_SEARCH_CANDIDATES]
        with self._get_conn() as conn, _dict_cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT * FROM catalog_objects
                WHERE {where_sql}
                  {fts_clause}
                ORDER BY {order_sql}
                LIMIT %s
                """,
                tuple(exec_params),
            )
            rows = cur.fetchall()

        hits: list[CatalogHit] = []
        for row in rows:
            meta = _row_to_meta(row)
            # System-tier rows are visible to every caller (ADR-050); the
            # project ACL re-check applies to the project tier only.
            if row.get("visibility") != VISIBILITY_SYSTEM and not allowed(
                meta, project_id
            ):
                continue
            if tokens:
                score = score_meta(meta, tokens, query=query)
                if score <= 0:
                    continue
            else:
                score = 1.0
            hits.append(CatalogHit(meta=meta, score=score))

        if not hits and ts_query:
            return self._search_fallback(
                project_id,
                tokens,
                limit=limit,
                query=query,
                modality=modality,
                state=state,
                tag=tag,
                prefix=prefix,
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def _search_fallback(
        self,
        project_id: str,
        tokens: list[str],
        *,
        limit: int,
        query: str = "",
        modality: str | None = None,
        state: str | None = None,
        tag: str | None = None,
        prefix: str | None = None,
    ) -> list[CatalogHit]:
        """Score a bounded, recency-ordered window when strict FTS AND-match misses.

        Strict AND semantics (all tokens present) miss legitimate multi-word
        queries where tokens land in different fields (e.g. one word in
        ``summary``, another only in ``tags``). This recency-biased fallback
        trades completeness for a bounded, predictable cost — it will not
        surface an old, exact match buried past ``_MAX_SEARCH_CANDIDATES``
        recently-updated rows.
        """
        metas = self.list_by_project(
            project_id,
            include_pending=True,
            include_error=False,
            limit=_MAX_SEARCH_CANDIDATES,
        )
        hits: list[CatalogHit] = []
        for meta in metas:
            if not _fallback_match(
                meta, modality=modality, state=state, tag=tag, prefix=prefix
            ):
                continue
            score = score_meta(meta, tokens, query=query)
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
                (project_id, f"{_like_literal(project_id)}/%"),
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
                (project_id, f"{_like_literal(project_id)}/%"),
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
