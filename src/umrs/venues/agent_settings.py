"""The user's agent settings for a remote session (ADR-099 D-8, migration 0128).

A remote session behaves like the user's local session (M-6, 2026-10-23).
The owner's client computes the settings its local bridge applies and sends
them before every remote start and on every change; the worker reads them
through its rail and applies them from the next turn when ``rev`` grew.

One row per session, owned by ``user_id``: a write by another user never
replaces it, and the rail read is filtered by the lease owner. Unknown keys
(a newer client's) are kept; the worker ignores them.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from typing import Any

SETTINGS_SCHEMA = 1
PERMISSION_MODES = ("ask_always", "auto_safe", "trust", "deny")
INTERACTION_MODES = ("chat", "plan", "autopilot")
ASK_USER_MODES = ("ask", "auto")
MAX_SETTINGS_BYTES = 16 * 1024
MAX_APPROVED_TOOLS = 256
MAX_APPROVAL_TTL_S = 7 * 24 * 3600
_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


class SettingsInvalid(ValueError):
    """A settings document the store refuses (``field`` names the key)."""

    def __init__(self, field: str, message: str) -> None:
        super().__init__(f"{field}: {message}")
        self.field = field


def _choice(allowed: tuple[str, ...]) -> Callable[[str, Any], Any]:
    def check(field: str, value: Any) -> Any:
        if value not in allowed:
            raise SettingsInvalid(field, f"must be one of {list(allowed)}")
        return value

    return check


def _flag(field: str, value: Any) -> Any:
    if not isinstance(value, bool):
        raise SettingsInvalid(field, "must be true or false")
    return value


def _int_range(low: int, high: int) -> Callable[[str, Any], Any]:
    def check(field: str, value: Any) -> Any:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not low <= value <= high
        ):
            raise SettingsInvalid(field, f"must be an integer in [{low}, {high}]")
        return value

    return check


def _tools(field: str, value: Any) -> Any:
    if not isinstance(value, list) or len(value) > MAX_APPROVED_TOOLS:
        raise SettingsInvalid(
            field, f"must be a list of at most {MAX_APPROVED_TOOLS} tool names"
        )
    if not all(isinstance(name, str) and _TOOL_NAME.match(name) for name in value):
        raise SettingsInvalid(field, "tool names are [A-Za-z0-9_.:-], 1-128 characters")
    return sorted(set(value))


_CHECKS: dict[str, Callable[[str, Any], Any]] = {
    "schema": _int_range(1, 1000),
    "permission_mode": _choice(PERMISSION_MODES),
    "interaction_mode": _choice(INTERACTION_MODES),
    "write_requires_plan": _flag,
    "workflow_armed": _flag,
    "ask_user_mode": _choice(ASK_USER_MODES),
    "approval_ttl_s": _int_range(0, MAX_APPROVAL_TTL_S),
    "session_approved_tools": _tools,
}


def validate_agent_settings(doc: Any) -> dict[str, Any]:
    """The stored form of a settings document; SettingsInvalid when refused."""
    if not isinstance(doc, dict):
        raise SettingsInvalid("settings", "must be a JSON object")
    if len(json.dumps(doc, default=str)) > MAX_SETTINGS_BYTES:
        raise SettingsInvalid("settings", f"larger than {MAX_SETTINGS_BYTES} bytes")
    out = dict(doc)
    for key, check in _CHECKS.items():
        if key in out:
            out[key] = check(key, out[key])
    out.setdefault("schema", SETTINGS_SCHEMA)
    return out


class SettingsOwnerConflict(PermissionError):
    """The session's settings row belongs to another user."""


_PUT_SQL = """
INSERT INTO companion_venue_agent_settings (session_id, user_id, settings, rev, updated_at)
VALUES (%s, %s, %s::jsonb, 1, now())
ON CONFLICT (session_id) DO UPDATE
    SET settings = EXCLUDED.settings,
        rev = companion_venue_agent_settings.rev + 1,
        updated_at = now()
    WHERE companion_venue_agent_settings.user_id = EXCLUDED.user_id
RETURNING rev
"""


class VenueAgentSettingsStore:
    """In-memory (pool=None) or Postgres-backed settings rows."""

    def __init__(self, pool: Any = None) -> None:
        self._pool = pool
        self._rows: dict[str, tuple[str, int, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str, *, user_id: str = "") -> tuple[int, dict[str, Any]]:
        """``(rev, settings)``; ``(0, {})`` when none were sent (or another
        user's row, when ``user_id`` is given)."""
        if self._pool is None:
            owner, rev, settings = self._rows.get(session_id, ("", 0, {}))
            if user_id and owner != user_id:
                return 0, {}
            return rev, dict(settings)
        row = self._fetch(session_id, user_id)
        if row is None:
            return 0, {}
        settings = row[1] if isinstance(row[1], dict) else json.loads(row[1] or "{}")
        return int(row[0] or 0), dict(settings)

    def put(self, *, session_id: str, user_id: str, settings: dict[str, Any]) -> int:
        """Store ``settings`` and return the new ``rev`` (1 on the first write)."""
        if self._pool is None:
            return self._put_memory(session_id, user_id, settings)
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(_PUT_SQL, (session_id, user_id, json.dumps(settings)))
                row = cur.fetchone()
            commit = getattr(conn, "commit", None)
            if callable(commit):
                commit()
        finally:
            self._pool.putconn(conn)
        if row is None:
            raise SettingsOwnerConflict(
                f"settings of session {session_id} belong to another user"
            )
        return int(row[0])

    def _put_memory(
        self, session_id: str, user_id: str, settings: dict[str, Any]
    ) -> int:
        with self._lock:
            owner, rev, _old = self._rows.get(session_id, (user_id, 0, {}))
            if owner != user_id:
                raise SettingsOwnerConflict(
                    f"settings of session {session_id} belong to another user"
                )
            self._rows[session_id] = (user_id, rev + 1, dict(settings))
            return rev + 1

    def _fetch(self, session_id: str, user_id: str) -> Any:
        sql = "SELECT rev, settings FROM companion_venue_agent_settings WHERE session_id = %s"
        params: tuple[str, ...] = (session_id,)
        if user_id:
            sql += " AND user_id = %s"
            params = (session_id, user_id)
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()
        finally:
            self._pool.putconn(conn)


__all__ = [
    "ASK_USER_MODES",
    "INTERACTION_MODES",
    "PERMISSION_MODES",
    "SETTINGS_SCHEMA",
    "SettingsInvalid",
    "SettingsOwnerConflict",
    "VenueAgentSettingsStore",
    "validate_agent_settings",
]
