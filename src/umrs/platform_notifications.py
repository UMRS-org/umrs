"""Durable platform notifications (migration 0109; D-36 leg B).

The store behind the web dashboard's notifications surface and the
budget-stop emission seam: one row per notification, ``read_at`` marks
consumption (never a DELETE — the row is the audit of what surfaced).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class NotificationStore:
    """psycopg2-backed durable notification rows (0109)."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def emit(
        self,
        *,
        kind: str,
        title: str = "",
        detail: dict[str, Any] | None = None,
        user_id: str = "",
        project_id: str = "",
    ) -> dict[str, Any]:
        """Insert one notification row; returns the created row view."""
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO platform_notifications
                   (kind, title, detail, user_id, project_id)
               VALUES (%s, %s, %s, %s, %s)
               RETURNING id, kind, title, detail, user_id, project_id,
                         created_at, read_at""",
            (
                kind,
                title,
                json.dumps(detail or {}),
                user_id,
                project_id,
            ),
        )
        row = cur.fetchone()
        self._conn.commit()
        return self._row_dict(row)

    def list(
        self,
        *,
        user_id: str | None = None,
        project_id: str | None = None,
        unread_only: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Newest-first; optional owner/project/unread filters."""
        where = []
        params: list[Any] = []
        if user_id:
            where.append("user_id = %s")
            params.append(user_id)
        if project_id:
            where.append("project_id = %s")
            params.append(project_id)
        if unread_only:
            where.append("read_at IS NULL")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, kind, title, detail, user_id, project_id, created_at, read_at"
            f" FROM platform_notifications{clause}"
            " ORDER BY created_at DESC LIMIT %s",
            (*params, limit),
        )
        return [self._row_dict(r) for r in cur.fetchall()]

    def mark_read(self, notification_id: int) -> dict[str, Any] | None:
        """Stamp read_at (idempotent); None when the id is unknown."""
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE platform_notifications SET read_at = now()"
            " WHERE id = %s AND read_at IS NULL"
            " RETURNING id, kind, title, detail, user_id, project_id, created_at, read_at",
            (notification_id,),
        )
        row = cur.fetchone()
        self._conn.commit()
        return self._row_dict(row) if row else None

    @staticmethod
    def _row_dict(row: Any) -> dict[str, Any]:
        return {
            "id": row[0],
            "kind": row[1],
            "title": row[2],
            "detail": row[3]
            if isinstance(row[3], dict)
            else json.loads(row[3] or "{}"),
            "user_id": row[4],
            "project_id": row[5],
            "created_at": row[6].isoformat()
            if hasattr(row[6], "isoformat")
            else row[6],
            "read_at": row[7].isoformat() if hasattr(row[7], "isoformat") else row[7],
        }


# ── the emission seam (the foundation-layer budget stop → the store) ────────

_EMITTER: Any = None


def set_emitter(pool: Any) -> None:
    """Install (or reset) the emitter pool — the api layer wires it at boot."""
    global _EMITTER
    _EMITTER = pool


def notify(
    *,
    kind: str,
    title: str = "",
    detail: dict[str, Any] | None = None,
    user_id: str = "",
    project_id: str = "",
) -> bool:
    """Best-effort emission: a notification failure NEVER blocks its trigger.

    Returns True when a row landed; False when unwired (named in the log —
    never a silent no-op claim)."""
    if _EMITTER is None:
        logger.warning(
            "notification emitter unwired — kind=%s project=%s skipped",
            kind,
            project_id,
        )
        return False
    try:
        with _conn(_EMITTER) as conn:
            NotificationStore(conn).emit(
                kind=kind,
                title=title,
                detail=detail,
                user_id=user_id,
                project_id=project_id,
            )
        return True
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        logger.warning("notification emit failed (kind=%s): %s", kind, exc)
        return False


@contextmanager
def _conn(pool: Any) -> Iterator[Any]:
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)
