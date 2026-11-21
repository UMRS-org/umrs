"""Durable chat-turn approval intents (ADR-057 D-3).

Same inbox contract as the loop-scoped approval inbox (0081) but keyed by
(session_id, turn_id): a chat turn raises the intent, any client of the
session's owner answers the decision. Terminal transitions only.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

_COLUMNS = (
    "approval_id",
    "session_id",
    "turn_id",
    "user_id",
    "action",
    "status",
    "decision",
    "decided_by",
    "decided_at",
    "created_at",
)

_TABLE = "chat_turn_approvals"

_APPROVED = frozenset({"once", "always"})
_REJECTED = frozenset({"reject"})
VALID_DECISIONS = frozenset(_APPROVED | _REJECTED)


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _row_to_record(row: Any) -> dict[str, Any]:
    cols = dict(zip(list(_COLUMNS), row, strict=False))
    action = cols.get("action")
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except (TypeError, ValueError):
            action = {}
    cols["action"] = action or {}
    cols["decided_at"] = _iso(cols.get("decided_at"))
    cols["created_at"] = _iso(cols.get("created_at"))
    return cols


class ChatTurnApprovalsStore:
    """Postgres primary (pool-borrowed), sqlite fallback for tests/local."""

    def __init__(self, conn=None, *, path: str | None = None) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        if self._conn is None:
            path = path or os.getenv("R4_CHAT_APPROVALS_DB", ":memory:")
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.row_factory = None
            self._sqlite = True
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_turn_approvals (
                    approval_id TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    turn_id     TEXT NOT NULL DEFAULT '',
                    user_id     TEXT NOT NULL DEFAULT '',
                    action      TEXT NOT NULL DEFAULT '{}',
                    status      TEXT NOT NULL DEFAULT 'pending'
                                CHECK (status IN ('pending','approved','rejected')),
                    decision    TEXT NOT NULL DEFAULT '',
                    decided_by  TEXT NOT NULL DEFAULT '',
                    decided_at  TEXT,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_chat_turn_approvals_pending
                    ON chat_turn_approvals(status, user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_chat_turn_approvals_session
                    ON chat_turn_approvals(session_id, turn_id);
                """
            )
        else:
            self._sqlite = False

    def _q(self, sql: str) -> str:
        return sql if not self._sqlite else sql.replace("%s", "?")

    def _fetch_one(self, cur, approval_id: str) -> dict[str, Any] | None:
        cur.execute(
            self._q(
                f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} WHERE approval_id = %s"
            ),
            (approval_id,),
        )
        row = cur.fetchone()
        return _row_to_record(row) if row else None

    def request(
        self,
        session_id: str,
        turn_id: str,
        action: dict[str, Any],
        *,
        approval_id: str | None = None,
        user_id: str = "",
    ) -> dict[str, Any]:
        """Register an intent; idempotent when ``approval_id`` is given."""
        approval_id = approval_id or f"appr-{uuid4().hex[:12]}"
        with self._lock:
            cur = self._conn.cursor()
            verb = "INSERT OR IGNORE" if self._sqlite else "INSERT"
            suffix = "" if self._sqlite else "ON CONFLICT (approval_id) DO NOTHING"
            cur.execute(
                self._q(
                    f"{verb} INTO {_TABLE} (approval_id, session_id, turn_id, user_id, action) "
                    f"VALUES (%s, %s, %s, %s, %s) {suffix}"
                ),
                (
                    approval_id,
                    session_id,
                    turn_id or "",
                    user_id or "",
                    json.dumps(dict(action)),
                ),
            )
            self._conn.commit()
            record = self._fetch_one(cur, approval_id)
        assert record is not None
        return record

    def get(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._fetch_one(self._conn.cursor(), approval_id)

    def pending(
        self, *, session_id: str | None = None, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} WHERE status = 'pending'"
        params: list[Any] = []
        if session_id:
            sql += " AND session_id = %s"
            params.append(session_id)
        if user_id is not None:
            sql += " AND (user_id = %s OR user_id = '')"
            params.append(user_id)
        sql += " ORDER BY created_at, approval_id"
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(self._q(sql), tuple(params))
            return [_row_to_record(r) for r in cur.fetchall()]

    def pending_decided(
        self, *, session_id: str, status: str
    ) -> list[dict[str, Any]]:
        if status not in ("approved", "rejected"):
            raise ValueError("status must be approved|rejected")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                self._q(
                    f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE} "
                    "WHERE status = %s AND session_id = %s "
                    "ORDER BY created_at, approval_id"
                ),
                (status, session_id),
            )
            return [_row_to_record(r) for r in cur.fetchall()]

    def decide(self, approval_id: str, decision: str, decided_by: str) -> dict[str, Any]:
        """Flip a pending intent; terminal transitions only."""
        if decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")
        status = "approved" if decision in _APPROVED else "rejected"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                self._q(
                    f"UPDATE {_TABLE} SET status = %s, decision = %s, "
                    "decided_by = %s, decided_at = %s "
                    "WHERE approval_id = %s AND status = 'pending'"
                ),
                (status, decision, decided_by, now, approval_id),
            )
            changed = (cur.rowcount or 0) > 0
            self._conn.commit()
            record = self._fetch_one(cur, approval_id)
        if record is None:
            raise KeyError(f"approval not found: {approval_id}")
        if not changed:
            raise ValueError(f"approval {approval_id} already decided")
        return record

    def clear_all(self) -> None:
        with self._lock:
            self._conn.cursor().execute(f"DELETE FROM {_TABLE}")
            self._conn.commit()


__all__ = ["VALID_DECISIONS", "ChatTurnApprovalsStore"]
