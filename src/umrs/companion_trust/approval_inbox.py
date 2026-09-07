"""Durable approval inbox — psycopg production store, sqlite fallback.

ADR-036 amendment (a) (WP-05 round 1): pending tool-grain approval intents
raised by UMRS-hosted loop drivers, keyed by ``approval_id`` and carrying the
loop / job / attempt they belong to plus the loop owner's ``user_id``. Both
the intent and its terminal decision are rows (migration 0081), so a UMRS
restart neither loses the question nor the answer — the parked loop reads the
recorded decision and continues instead of re-asking.

Sync facade like :class:`~umrs.companion_trust.store.TrustAllowlistStore`:
HTTP handlers and the engine driver thread call sync methods; tests and
DB-less deployments run on the sqlite3 fallback. One lock serialises every
operation — the store is shared between the driver thread and the request
threadpool on purpose (single chokepoint, honest ``durability``).

Semantics:
- ``request`` is idempotent under ``approval_id`` (rehydration after restart
  returns the existing row, decided or not — decided rows are terminal)
- ``decide`` transitions ``pending → approved|rejected`` exactly once;
  unknown id → ``KeyError``; already decided → ``ValueError``
- ``pending(user_id=…)`` returns the caller's rows plus ownerless rows
  (loops started without a gateway identity); never another owner's rows
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .schema import CREATE_APPROVAL_INBOX_TABLE
from .store import _sqliteify

# Decision vocabulary (kept identical to the companion route contract).
APPROVED_DECISIONS = frozenset({"once", "always"})
REJECTED_DECISIONS = frozenset({"no"})
VALID_DECISIONS = APPROVED_DECISIONS | REJECTED_DECISIONS

_COLUMNS = (
    "approval_id",
    "loop_id",
    "iteration",
    "user_id",
    "job_id",
    "attempt_id",
    "action",
    "status",
    "decision",
    "decided_by",
    "decided_at",
    "created_at",
)
_SELECT = "SELECT " + ", ".join(_COLUMNS) + " FROM companion_approval_inbox"


def _iso(value: Any) -> str:
    if value is None or value == "":
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _action_dict(raw: Any) -> dict[str, Any]:
    """JSONB arrives parsed (psycopg) or as text (sqlite)."""
    if isinstance(raw, (str, bytes)):
        raw = json.loads(raw or "{}")
    return dict(raw or {})


def _row_to_record(row: Any) -> dict[str, Any]:
    data = dict(zip(_COLUMNS, list(row)))  # sqlite3.Row / psycopg tuples: positional
    text = {k: (data[k] or "") for k in ("user_id", "job_id", "attempt_id", "decision", "decided_by")}
    return {
        "id": data["approval_id"],
        "loop_id": data["loop_id"],
        "iteration": int(data["iteration"] or 0),
        "action": _action_dict(data["action"]),
        "status": data["status"],
        "decided_at": _iso(data["decided_at"]),
        "created_at": _iso(data["created_at"]),
        **text,
    }


class ApprovalInboxStore:
    """Sync store over a DB-API connection (psycopg or sqlite3).

    ``conn`` may be a plain connection or a pool with ``getconn``/``putconn``.
    Placeholders are written ``%s`` — sqlite needs ``?`` (translated in ``_q``).
    """

    def __init__(self, conn=None, *, path: str | None = None) -> None:
        self._lock = threading.RLock()
        self._pool = conn if hasattr(conn, "getconn") else None
        self._path = path
        if conn is None and path is not None:
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._sqlite = True
        elif conn is not None and self._pool is None:
            self._conn = conn
            self._sqlite = isinstance(conn, sqlite3.Connection)
        elif self._pool is not None:
            self._conn = None
            self._sqlite = False
        else:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._sqlite = True

        if self._sqlite and self._conn is not None:
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_sqliteify(CREATE_APPROVAL_INBOX_TABLE))

    @property
    def durability(self) -> str:
        """``postgres`` (durable), ``sqlite`` (file-durable), ``process``."""
        if not self._sqlite:
            return "postgres"
        return "sqlite" if self._path and self._path != ":memory:" else "process"

    # -- plumbing ------------------------------------------------------------ #

    def _borrow(self):
        return self._pool.getconn() if self._pool is not None else self._conn

    def _return(self, conn) -> None:
        if self._pool is not None:
            self._pool.putconn(conn)

    def _q(self, sql: str) -> str:
        if not self._sqlite:
            return sql
        return sql.replace("%s", "?").replace("now()", "datetime('now')")

    def _commit(self, conn) -> None:
        if self._sqlite or not getattr(conn, "autocommit", False):
            conn.commit()

    def _json_param(self, payload: dict[str, Any]) -> Any:
        if self._sqlite:
            return json.dumps(payload)
        from psycopg2.extras import Json

        return Json(payload)

    def _fetch_one(self, cur, approval_id: str) -> dict[str, Any] | None:
        cur.execute(self._q(_SELECT + " WHERE approval_id = %s"), (approval_id,))
        row = cur.fetchone()
        return _row_to_record(row) if row is not None else None

    # -- API ----------------------------------------------------------------- #

    def request(
        self,
        loop_id: str,
        iteration: int,
        action: dict[str, Any],
        *,
        approval_id: str | None = None,
        user_id: str = "",
        job_id: str = "",
        attempt_id: str = "",
    ) -> dict[str, Any]:
        """Register an approval intent; idempotent when ``approval_id`` is given."""
        approval_id = approval_id or f"appr-{uuid4().hex[:12]}"
        with self._lock:
            conn = self._borrow()
            try:
                cur = conn.cursor()
                cur.execute(
                    self._q(
                        "INSERT INTO companion_approval_inbox "
                        "(approval_id, loop_id, iteration, user_id, job_id, attempt_id, action) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (approval_id) DO NOTHING"
                    ),
                    (
                        approval_id,
                        loop_id,
                        int(iteration),
                        user_id or "",
                        job_id or "",
                        attempt_id or "",
                        self._json_param(dict(action)),
                    ),
                )
                self._commit(conn)
                record = self._fetch_one(cur, approval_id)
            finally:
                self._return(conn)
        assert record is not None  # the row was just inserted or already present
        return record

    def get(self, approval_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._borrow()
            try:
                return self._fetch_one(conn.cursor(), approval_id)
            finally:
                self._return(conn)

    def pending(
        self, *, loop_id: str | None = None, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Pending intents, optionally per loop and/or visible to one owner."""
        sql = _SELECT + " WHERE status = 'pending'"
        params: list[Any] = []
        if loop_id:
            sql += " AND loop_id = %s"
            params.append(loop_id)
        if user_id is not None:
            sql += " AND (user_id = %s OR user_id = '')"
            params.append(user_id)
        sql += " ORDER BY created_at, approval_id"
        with self._lock:
            conn = self._borrow()
            try:
                cur = conn.cursor()
                cur.execute(self._q(sql), tuple(params))
                return [_row_to_record(r) for r in cur.fetchall()]
            finally:
                self._return(conn)

    def decide(
        self, approval_id: str, decision: str, decided_by: str
    ) -> dict[str, Any]:
        """Flip a pending intent to approved/rejected. Terminal transitions only."""
        if decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")
        status = "approved" if decision in APPROVED_DECISIONS else "rejected"
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = self._borrow()
            try:
                cur = conn.cursor()
                cur.execute(
                    self._q(
                        "UPDATE companion_approval_inbox SET status = %s, decision = %s, "
                        "decided_by = %s, decided_at = %s "
                        "WHERE approval_id = %s AND status = 'pending'"
                    ),
                    (status, decision, decided_by, now, approval_id),
                )
                changed = (cur.rowcount or 0) > 0
                self._commit(conn)
                record = self._fetch_one(cur, approval_id)
            finally:
                self._return(conn)
        if record is None:
            raise KeyError(f"approval not found: {approval_id}")
        if not changed:
            raise ValueError(f"approval {approval_id} already decided")
        return record

    def clear_all(self) -> None:
        """Test utility: wipe every intent."""
        with self._lock:
            conn = self._borrow()
            try:
                conn.cursor().execute("DELETE FROM companion_approval_inbox")
                self._commit(conn)
            finally:
                self._return(conn)


__all__ = [
    "APPROVED_DECISIONS",
    "REJECTED_DECISIONS",
    "VALID_DECISIONS",
    "ApprovalInboxStore",
]
