"""
Egress Audit REST API (A-1204).

Persists egress proxy decisions emitted by the Gateway
(`gateway/gateway/routes/egress.py`) and exposes a query API for
`r4 agent audit` (A-1209).

Endpoints
---------
- POST /umrs/egress/audit         Record one egress decision.
- GET  /umrs/egress/audit         List/filter audit entries.

Storage
-------
If ``UMRS_DB_URL`` is set and ``psycopg2`` is importable, rows are
persisted to the ``egress_requests`` table created by migration
``0005_egress_audit.sql``. Otherwise the router falls back to an
in-memory ring buffer (development / unit-test mode).

Writes are best-effort — a database failure must not propagate to the
Gateway, which only treats the audit sink as advisory.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections import deque
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Deque, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/egress", tags=["egress-audit"])


# =============================================================================
# Models
# =============================================================================


class EgressAuditRecord(BaseModel):
    """One audited egress decision."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    project_id: str
    agent_session_id: Optional[str] = None
    tool_name: Optional[str] = None
    destination_url: str
    destination_domain: str
    method: str = "GET"
    payload_hash: str = ""
    payload_size_bytes: int = 0
    sensitivity_flags: Dict[str, Any] = Field(default_factory=dict)
    user_tags: List[str] = Field(default_factory=list)
    action: str = Field(..., pattern="^(allowed|denied)$")
    deny_reason: Optional[str] = None
    response_status: Optional[int] = None
    response_size_bytes: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EgressAuditCreateResponse(BaseModel):
    request_id: str
    persisted: bool


class EgressAuditListResponse(BaseModel):
    items: List[EgressAuditRecord]
    count: int


# =============================================================================
# Backend (DB optional, in-memory fallback)
# =============================================================================


_INMEMORY_LIMIT = int(os.getenv("UMRS_EGRESS_AUDIT_MEMORY_LIMIT", "10000"))


class _EgressAuditStore:
    """Thread-safe egress audit sink with optional Postgres backend."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._buffer: Deque[EgressAuditRecord] = deque(maxlen=_INMEMORY_LIMIT)
        self._db_url = os.getenv("UMRS_DB_URL", "").strip() or None
        self._psycopg2 = None
        if self._db_url:
            try:
                import psycopg2  # type: ignore
                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning(
                    "UMRS_DB_URL set but psycopg2 unavailable; "
                    "egress audit will use in-memory store only"
                )

    @property
    def db_enabled(self) -> bool:
        return self._psycopg2 is not None and self._db_url is not None

    def insert(self, record: EgressAuditRecord) -> bool:
        """Append to in-memory buffer and (best-effort) persist to DB."""
        with self._lock:
            self._buffer.append(record)
        if not self.db_enabled:
            return False
        try:
            assert self._psycopg2 is not None and self._db_url is not None
            with self._psycopg2.connect(self._db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO egress_requests (
                            request_id, user_id, project_id, agent_session_id,
                            tool_name, destination_url, destination_domain,
                            method, payload_hash, payload_size_bytes,
                            sensitivity_flags, user_tags, action, deny_reason,
                            response_status, response_size_bytes, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s::jsonb, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (request_id) DO NOTHING
                        """,
                        (
                            record.request_id,
                            record.user_id,
                            record.project_id,
                            record.agent_session_id,
                            record.tool_name,
                            record.destination_url,
                            record.destination_domain,
                            record.method,
                            record.payload_hash,
                            record.payload_size_bytes,
                            self._json_dumps(record.sensitivity_flags),
                            record.user_tags,
                            record.action,
                            record.deny_reason,
                            record.response_status,
                            record.response_size_bytes,
                            record.created_at,
                        ),
                    )
                conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001 - audit must never break callers
            logger.warning("Failed to persist egress audit row: %s", exc)
            return False

    def query(
        self,
        *,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[EgressAuditRecord]:
        if self.db_enabled:
            try:
                return self._query_db(
                    user_id=user_id,
                    project_id=project_id,
                    action=action,
                    since=since,
                    limit=limit,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("DB query failed, falling back to memory: %s", exc)
        return self._query_memory(
            user_id=user_id,
            project_id=project_id,
            action=action,
            since=since,
            limit=limit,
        )

    # -------- internal helpers --------

    def _query_memory(
        self,
        *,
        user_id: Optional[str],
        project_id: Optional[str],
        action: Optional[str],
        since: Optional[datetime],
        limit: int,
    ) -> List[EgressAuditRecord]:
        with self._lock:
            items = list(self._buffer)
        items.reverse()  # newest first
        out: List[EgressAuditRecord] = []
        for r in items:
            if user_id and r.user_id != user_id:
                continue
            if project_id and r.project_id != project_id:
                continue
            if action and r.action != action:
                continue
            if since and r.created_at < since:
                continue
            out.append(r)
            if len(out) >= limit:
                break
        return out

    def _query_db(
        self,
        *,
        user_id: Optional[str],
        project_id: Optional[str],
        action: Optional[str],
        since: Optional[datetime],
        limit: int,
    ) -> List[EgressAuditRecord]:
        assert self._psycopg2 is not None and self._db_url is not None
        clauses: List[str] = []
        params: List[Any] = []
        if user_id:
            clauses.append("user_id = %s")
            params.append(user_id)
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if action:
            clauses.append("action = %s")
            params.append(action)
        if since:
            clauses.append("created_at >= %s")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT request_id, user_id, project_id, agent_session_id, "
            "tool_name, destination_url, destination_domain, method, "
            "payload_hash, payload_size_bytes, sensitivity_flags, user_tags, "
            "action, deny_reason, response_status, response_size_bytes, "
            f"created_at FROM egress_requests {where} "
            "ORDER BY created_at DESC LIMIT %s"
        )
        params.append(limit)
        with self._psycopg2.connect(self._db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: tuple) -> EgressAuditRecord:
        return EgressAuditRecord(
            request_id=str(row[0]),
            user_id=row[1],
            project_id=row[2],
            agent_session_id=str(row[3]) if row[3] else None,
            tool_name=row[4],
            destination_url=row[5],
            destination_domain=row[6],
            method=row[7],
            payload_hash=row[8],
            payload_size_bytes=row[9] or 0,
            sensitivity_flags=row[10] or {},
            user_tags=list(row[11] or []),
            action=row[12],
            deny_reason=row[13],
            response_status=row[14],
            response_size_bytes=row[15],
            created_at=row[16],
        )

    @staticmethod
    def _json_dumps(value: Any) -> str:
        import json
        return json.dumps(value or {}, default=str)


_store: Optional[_EgressAuditStore] = None


def get_store() -> _EgressAuditStore:
    global _store
    if _store is None:
        _store = _EgressAuditStore()
    return _store


# =============================================================================
# Routes
# =============================================================================


@router.post(
    "/audit",
    response_model=EgressAuditCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_audit(record: EgressAuditRecord) -> EgressAuditCreateResponse:
    persisted = get_store().insert(record)
    return EgressAuditCreateResponse(
        request_id=record.request_id,
        persisted=persisted,
    )


@router.get("/audit", response_model=EgressAuditListResponse)
def list_audit(
    user_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None, pattern="^(allowed|denied)$"),
    since: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
) -> EgressAuditListResponse:
    items = get_store().query(
        user_id=user_id,
        project_id=project_id,
        action=action,
        since=since,
        limit=limit,
    )
    return EgressAuditListResponse(items=items, count=len(items))


__all__ = [
    "router",
    "EgressAuditRecord",
    "EgressAuditCreateResponse",
    "EgressAuditListResponse",
    "get_store",
]
