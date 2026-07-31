"""Postgres-backed evidence store (durable K1 implementation).

Implements the same interface as :class:`InMemoryEvidenceStore` against the
``evidence_items`` table (migration 0059). The in-memory store remains the
reference implementation and test fake.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from . import (
    DuplicateEvidenceError,
    Evidence,
    EvidenceSourceType,
    EvidenceStatus,
)

logger = logging.getLogger(__name__)

_COLUMNS = (
    "evidence_id, project_id, claim_id, source_type, source_id, content, "
    "content_uri, relevance_score, citation, status, metadata, "
    "created_at, updated_at, created_by"
)


def _row_to_evidence(row: dict[str, Any]) -> Evidence:
    data = dict(row)
    if isinstance(data.get("metadata"), str):
        data["metadata"] = json.loads(data["metadata"])
    return Evidence.from_dict(data)


def _scoped(
    project_id: str | None,
    source_type: EvidenceSourceType | None,
    status: EvidenceStatus | None,
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if project_id:
        clauses.append("project_id = %s")
        params.append(project_id)
    if source_type:
        clauses.append("source_type = %s")
        params.append(source_type.value)
    if status:
        clauses.append("status = %s")
        params.append(status.value)
    return clauses, params


class PgEvidenceStore:
    """Durable evidence store backed by Postgres."""

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

    def add_evidence(self, evidence: Evidence) -> str:
        """Add evidence; raises DuplicateEvidenceError on id conflict."""
        try:
            with self._cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO evidence_items ({_COLUMNS})
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb,
                            %s, %s, %s)
                    """,
                    (
                        evidence.evidence_id,
                        evidence.project_id,
                        evidence.claim_id,
                        evidence.source_type.value,
                        evidence.source_id,
                        evidence.content,
                        evidence.content_uri,
                        evidence.relevance_score,
                        evidence.citation,
                        evidence.status.value,
                        json.dumps(evidence.metadata),
                        evidence.created_at,
                        evidence.updated_at,
                        evidence.created_by,
                    ),
                )
        except Exception as exc:
            if _is_unique_violation(exc):
                raise DuplicateEvidenceError(
                    f"Evidence {evidence.evidence_id} already exists"
                ) from exc
            raise
        return evidence.evidence_id

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Get evidence by ID, or None."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM evidence_items WHERE evidence_id = %s",
                (evidence_id,),
            )
            row = cur.fetchone()
        return _row_to_evidence(row) if row else None

    def list_evidence(
        self,
        *,
        project_id: str | None = None,
        source_type: EvidenceSourceType | None = None,
        status: EvidenceStatus | None = None,
        claim_id: str | None = None,
        limit: int = 100,
    ) -> list[Evidence]:
        """List evidence with filters, most relevant first.

        Defaults to ACTIVE items only (parity with the in-memory store).
        """
        effective_status = status if status is not None else EvidenceStatus.ACTIVE
        clauses, params = _scoped(project_id, source_type, effective_status)
        if claim_id:
            clauses.append("claim_id = %s")
            params.append(claim_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM evidence_items {where} "
                "ORDER BY relevance_score DESC LIMIT %s",
                params,
            )
            return [_row_to_evidence(row) for row in cur.fetchall()]

    def search_evidence(
        self,
        query: str,
        *,
        project_id: str | None = None,
        limit: int = 20,
    ) -> list[Evidence]:
        """Search ACTIVE evidence by content or citation substring."""
        clauses = ["status = %s", "(lower(content) LIKE %s OR lower(citation) LIKE %s)"]
        needle = f"%{query.lower()}%"
        params: list[Any] = [EvidenceStatus.ACTIVE.value, needle, needle]
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM evidence_items "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY relevance_score DESC LIMIT %s",
                params,
            )
            return [_row_to_evidence(row) for row in cur.fetchall()]

    def get_evidence_for_claim(self, claim_id: str) -> list[Evidence]:
        """All ACTIVE evidence for a claim, most relevant first."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM evidence_items "
                "WHERE claim_id = %s AND status = %s "
                "ORDER BY relevance_score DESC",
                (claim_id, EvidenceStatus.ACTIVE.value),
            )
            return [_row_to_evidence(row) for row in cur.fetchall()]

    def update_status(self, evidence_id: str, status: EvidenceStatus) -> bool:
        """Update evidence status; True when the item exists."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "UPDATE evidence_items SET status = %s, updated_at = %s "
                "WHERE evidence_id = %s",
                (status.value, now, evidence_id),
            )
            return cur.rowcount > 0

    def count(
        self,
        *,
        project_id: str | None = None,
        source_type: EvidenceSourceType | None = None,
        status: EvidenceStatus | None = None,
    ) -> int:
        """Count evidence with optional filters."""
        clauses, params = _scoped(project_id, source_type, status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._cursor() as cur:
            cur.execute(f"SELECT count(*) AS n FROM evidence_items {where}", params)
            return int(cur.fetchone()["n"])


def _is_unique_violation(exc: Exception) -> bool:
    try:
        from psycopg2 import errors as pg_errors

        return isinstance(exc, pg_errors.UniqueViolation)
    except ImportError:
        return False
