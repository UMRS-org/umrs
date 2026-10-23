"""Durable ask_user questions of remote sessions (ADR-099 D-8, migration 0127).

A remote session's worker raises the question through its rail and polls it;
the session's owner answers it from any attached client. Terminal
transitions only: ``pending`` → ``answered`` | ``expired`` | ``cancelled``.

Same storage pattern as :class:`ChatTurnApprovalsStore`: a borrowed Postgres
connection in production, a sqlite fallback for tests and local use.
Answers are conversation content: admin views never read this table.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

_TABLE = "chat_turn_questions"
_COLUMNS = (
    "question_id",
    "session_id",
    "turn_id",
    "user_id",
    "question",
    "options",
    "status",
    "answer",
    "answered_by",
    "answered_at",
    "created_at",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM {_TABLE}"

STATUSES = ("pending", "answered", "expired", "cancelled")
MAX_QUESTION_CHARS = 4000
MAX_ANSWER_CHARS = 8000
MAX_OPTIONS = 24
MAX_OPTION_CHARS = 500

_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS chat_turn_questions (
    question_id TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    turn_id     TEXT NOT NULL DEFAULT '',
    user_id     TEXT NOT NULL,
    question    TEXT NOT NULL,
    options     TEXT NOT NULL DEFAULT '[]',
    status      TEXT NOT NULL DEFAULT 'pending'
                CHECK (status IN ('pending','answered','expired','cancelled')),
    answer      TEXT NOT NULL DEFAULT '',
    answered_by TEXT NOT NULL DEFAULT '',
    answered_at TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f+00:00','now'))
);
CREATE INDEX IF NOT EXISTS idx_chat_turn_questions_session
    ON chat_turn_questions(session_id, status, created_at);
"""


class QuestionClosed(ValueError):
    """An answer to a question that is no longer pending (HTTP 409)."""


def _iso(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _options(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _record(row: Any) -> dict[str, Any]:
    cols = dict(zip(_COLUMNS, row, strict=False))
    cols["options"] = _options(cols.get("options"))
    cols["answered_at"] = _iso(cols.get("answered_at"))
    cols["created_at"] = _iso(cols.get("created_at"))
    return cols


def clean_options(options: Any) -> list[str]:
    """Bounded, non-empty option strings (the overlay's choices)."""
    if not isinstance(options, (list, tuple)):
        return []
    out = [str(item)[:MAX_OPTION_CHARS] for item in options if str(item).strip()]
    return out[:MAX_OPTIONS]


class ChatTurnQuestionsStore:
    """Postgres primary (pool-borrowed connection), sqlite fallback."""

    def __init__(self, conn: Any = None, *, path: str | None = None) -> None:
        self._lock = threading.Lock()
        self._sqlite = conn is None
        if conn is None:
            path = path or os.getenv("R4_CHAT_QUESTIONS_DB", ":memory:")
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.executescript(_SQLITE_DDL)
        self._conn = conn

    def _q(self, sql: str) -> str:
        return sql.replace("%s", "?") if self._sqlite else sql

    def _fetch_one(self, cur: Any, question_id: str) -> dict[str, Any] | None:
        cur.execute(self._q(f"{_SELECT} WHERE question_id = %s"), (question_id,))
        row = cur.fetchone()
        return _record(row) if row else None

    def ask(
        self,
        *,
        session_id: str,
        turn_id: str,
        user_id: str,
        question: str,
        options: list[str],
        question_id: str | None = None,
    ) -> dict[str, Any]:
        """Register a pending question; idempotent when ``question_id`` is given."""
        question_id = question_id or f"q-{uuid4().hex[:16]}"
        verb = "INSERT OR IGNORE" if self._sqlite else "INSERT"
        suffix = "" if self._sqlite else "ON CONFLICT (question_id) DO NOTHING"
        params = (
            question_id,
            session_id,
            turn_id or "",
            user_id,
            question[:MAX_QUESTION_CHARS],
            json.dumps(clean_options(options)),
        )
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                self._q(
                    f"{verb} INTO {_TABLE} "
                    "(question_id, session_id, turn_id, user_id, question, options) "
                    f"VALUES (%s, %s, %s, %s, %s, %s) {suffix}"
                ),
                params,
            )
            self._conn.commit()
            record = self._fetch_one(cur, question_id)
        if record is None:
            raise KeyError(f"question not stored: {question_id}")
        return record

    def get(self, question_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._fetch_one(self._conn.cursor(), question_id)

    def pending(self, *, session_id: str) -> list[dict[str, Any]]:
        return self.by_status(session_id=session_id, status="pending")

    def by_status(
        self, *, session_id: str, status: str = "pending"
    ) -> list[dict[str, Any]]:
        """One session's questions, oldest first; ``status="all"`` for every state."""
        sql = f"{_SELECT} WHERE session_id = %s"
        params: list[Any] = [session_id]
        if status != "all":
            if status not in STATUSES:
                raise ValueError(f"status must be one of {[*STATUSES, 'all']}")
            sql += " AND status = %s"
            params.append(status)
        sql += " ORDER BY created_at, question_id"
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(self._q(sql), tuple(params))
            return [_record(row) for row in cur.fetchall()]

    def _close(
        self, question_id: str, status: str, answer: str, answered_by: str
    ) -> tuple[dict[str, Any] | None, bool]:
        now = datetime.now(timezone.utc).isoformat() if status == "answered" else None
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                self._q(
                    f"UPDATE {_TABLE} SET status = %s, answer = %s, answered_by = %s, "
                    "answered_at = %s WHERE question_id = %s AND status = 'pending'"
                ),
                (status, answer, answered_by, now, question_id),
            )
            changed = (cur.rowcount or 0) > 0
            self._conn.commit()
            return self._fetch_one(cur, question_id), changed

    def answer(self, question_id: str, answer: str, answered_by: str) -> dict[str, Any]:
        """Answer a pending question. KeyError when absent, QuestionClosed when
        it is no longer pending."""
        record, changed = self._close(
            question_id, "answered", str(answer)[:MAX_ANSWER_CHARS], answered_by
        )
        if record is None:
            raise KeyError(f"question not found: {question_id}")
        if not changed:
            raise QuestionClosed(f"question {question_id} is {record['status']}")
        return record

    def expire(self, question_id: str) -> dict[str, Any]:
        """Mark a pending question expired; a closed one is returned unchanged."""
        record, _changed = self._close(question_id, "expired", "", "")
        if record is None:
            raise KeyError(f"question not found: {question_id}")
        return record

    def table_present(self) -> bool:
        """False on a database without migration 0127 (the sqlite fallback
        creates its table itself)."""
        if self._sqlite:
            return True
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(f"SELECT to_regclass('{_TABLE}') IS NOT NULL")
            row = cur.fetchone()
        return bool(row and row[0])

    def cancel_pending(self, *, session_id: str, user_id: str) -> int:
        """Close one owner's pending questions of a session as ``cancelled``
        (a new worker, a stop or the lease end: nobody polls them any more).
        The number of questions closed."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                self._q(
                    f"UPDATE {_TABLE} SET status = 'cancelled' WHERE session_id = %s "
                    "AND user_id = %s AND status = 'pending'"
                ),
                (session_id, user_id),
            )
            closed = int(cur.rowcount or 0)
            self._conn.commit()
        return closed


def cancel_session_questions(
    pool: Any,
    *,
    session_id: str,
    user_id: str,
    factory: Callable[[Any], ChatTurnQuestionsStore] | None = None,
) -> int:
    """Best effort: cancel a session's pending questions on a pooled
    connection (ADR-099 D-8). A venue without migration 0127, or a database
    error, cancels nothing and never fails the caller's start or stop."""
    if pool is None:
        return 0
    conn = pool.getconn()
    try:
        store = factory(conn) if factory is not None else ChatTurnQuestionsStore(conn)
        if not store.table_present():
            return 0
        return store.cancel_pending(session_id=session_id, user_id=user_id)
    except Exception:  # cleanup never blocks a start or a stop
        logger.warning(
            "pending questions of %s not cancelled", session_id, exc_info=True
        )
        _rollback(conn)
        return 0
    finally:
        pool.putconn(conn)


def _rollback(conn: Any) -> None:
    rollback = getattr(conn, "rollback", None)
    if callable(rollback):
        try:
            rollback()
        except Exception:  # the pool discards a broken connection
            logger.debug("rollback failed", exc_info=True)


__all__ = [
    "MAX_ANSWER_CHARS",
    "MAX_QUESTION_CHARS",
    "STATUSES",
    "ChatTurnQuestionsStore",
    "QuestionClosed",
    "cancel_session_questions",
    "clean_options",
]
