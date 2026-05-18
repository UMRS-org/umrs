"""Session-scoped tool output cache (MEM-A3).

Caches active tool outputs within a session/execution context with
TTL-based expiration.  Avoids re-executing identical tool calls within
the same workflow run.

The cache key is derived from (session_id, tool_name, inputs_hash).
Entries expire after a configurable TTL (default 300s).

This module lives in ``umrs.jobs`` because it reads/writes durable state
tied to jobs/executions and does not import from ``umrs.api`` or
``umrs.server``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 300  # 5 minutes


def _generate_entry_id() -> str:
    return f"toc-{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _compute_inputs_hash(tool_name: str, inputs: Any) -> str:
    """Deterministic hash of tool name + inputs for cache lookup."""
    raw = json.dumps(
        {"tool": tool_name, "inputs": inputs},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class CachedToolOutput:
    """A single cached tool output entry."""

    entry_id: str
    session_id: str
    tool_name: str
    inputs_hash: str
    inputs_summary: dict[str, Any] | None = None  # first 1KB for display
    output: dict[str, Any] | None = None  # inline output (<64KB)
    output_ref: str | None = None  # S3 ref for large outputs
    output_size_bytes: int = 0
    hit_count: int = 0
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    created_at: datetime = field(default_factory=_now)
    expires_at: datetime | None = None
    last_accessed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "session_id": self.session_id,
            "tool_name": self.tool_name,
            "inputs_hash": self.inputs_hash,
            "inputs_summary": self.inputs_summary,
            "output": self.output,
            "output_ref": self.output_ref,
            "output_size_bytes": self.output_size_bytes,
            "hit_count": self.hit_count,
            "ttl_seconds": self.ttl_seconds,
            "created_at": _iso(self.created_at),
            "expires_at": _iso(self.expires_at),
            "last_accessed_at": _iso(self.last_accessed_at),
        }

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return _now() >= self.expires_at


_MAX_INLINE_BYTES = 65536  # 64KB


class ToolOutputCache:
    """Session-scoped tool output cache backed by Postgres.

    Falls back to in-memory storage when ``db_url`` is not set (dev mode).
    """

    def __init__(self, db_url: str | None = None) -> None:
        self._db_url = db_url
        self._psycopg2 = None
        if db_url:
            try:
                import psycopg2  # type: ignore

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning(
                    "db_url set but psycopg2 unavailable; "
                    "tool output cache will use in-memory store"
                )
        # In-memory fallback: session_id -> {inputs_hash -> CachedToolOutput}
        self._cache: dict[str, dict[str, CachedToolOutput]] = {}

    @property
    def db_enabled(self) -> bool:
        return self._psycopg2 is not None and self._db_url is not None

    def _get_conn(self):
        return self._psycopg2.connect(self._db_url)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def put(
        self,
        session_id: str,
        tool_name: str,
        inputs: Any,
        output: Any,
        output_ref: str | None = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> CachedToolOutput:
        """Cache a tool output for a session.

        If an entry with the same (session_id, tool_name, inputs_hash)
        already exists, it is replaced.
        """
        inputs_hash = _compute_inputs_hash(tool_name, inputs)
        now = _now()
        expires_at = now + timedelta(seconds=ttl_seconds)

        # Build inputs summary (first 1KB for display)
        inputs_summary = None
        if inputs is not None:
            try:
                raw = json.dumps(inputs, sort_keys=True, default=str)
                if len(raw) > 1024:
                    inputs_summary = {"_truncated": True, "preview": raw[:1024]}
                else:
                    inputs_summary = inputs
            except (TypeError, ValueError):
                pass

        # Determine inline vs ref
        inline_output = None
        output_size = 0
        if output is not None:
            try:
                raw = json.dumps(output, sort_keys=True, default=str)
                output_size = len(raw.encode())
                if output_size < _MAX_INLINE_BYTES:
                    inline_output = output
            except (TypeError, ValueError):
                pass

        entry = CachedToolOutput(
            entry_id=_generate_entry_id(),
            session_id=session_id,
            tool_name=tool_name,
            inputs_hash=inputs_hash,
            inputs_summary=inputs_summary,
            output=inline_output,
            output_ref=output_ref,
            output_size_bytes=output_size,
            ttl_seconds=ttl_seconds,
            created_at=now,
            expires_at=expires_at,
        )

        if self.db_enabled:
            try:
                self._db_put(entry)
                return entry
            except Exception as exc:
                logger.warning("DB put tool output failed, using memory: %s", exc)

        session_cache = self._cache.setdefault(session_id, {})
        session_cache[inputs_hash] = entry
        return entry

    def get(
        self, session_id: str, tool_name: str, inputs: Any
    ) -> CachedToolOutput | None:
        """Look up a cached tool output. Returns None on miss or expiry."""
        inputs_hash = _compute_inputs_hash(tool_name, inputs)

        if self.db_enabled:
            try:
                entry = self._db_get(session_id, inputs_hash)
                if entry is not None and not entry.is_expired:
                    self._db_record_hit(entry.entry_id)
                    entry.hit_count += 1
                    entry.last_accessed_at = _now()
                    return entry
                return None
            except Exception as exc:
                logger.warning("DB get tool output failed: %s", exc)

        session_cache = self._cache.get(session_id, {})
        entry = session_cache.get(inputs_hash)
        if entry is None or entry.is_expired:
            return None
        entry.hit_count += 1
        entry.last_accessed_at = _now()
        return entry

    def list_session(self, session_id: str) -> list[CachedToolOutput]:
        """List all non-expired cached outputs for a session."""
        now = _now()

        if self.db_enabled:
            try:
                return self._db_list_session(session_id, now)
            except Exception as exc:
                logger.warning("DB list_session failed: %s", exc)

        session_cache = self._cache.get(session_id, {})
        return [e for e in session_cache.values() if not e.is_expired]

    def invalidate(self, session_id: str, tool_name: str | None = None) -> int:
        """Invalidate cached entries for a session.

        If tool_name is given, only that tool's entries are removed.
        Returns count of removed entries.
        """
        if self.db_enabled:
            try:
                return self._db_invalidate(session_id, tool_name)
            except Exception as exc:
                logger.warning("DB invalidate failed: %s", exc)

        session_cache = self._cache.get(session_id, {})
        if not session_cache:
            return 0

        if tool_name is None:
            count = len(session_cache)
            session_cache.clear()
            return count

        to_remove = [
            k for k, v in session_cache.items() if v.tool_name == tool_name
        ]
        for k in to_remove:
            del session_cache[k]
        return len(to_remove)

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now = _now()

        if self.db_enabled:
            try:
                return self._db_cleanup_expired(now)
            except Exception as exc:
                logger.warning("DB cleanup_expired failed: %s", exc)

        count = 0
        for session_id in list(self._cache):
            sc = self._cache[session_id]
            expired = [k for k, v in sc.items() if v.is_expired]
            for k in expired:
                del sc[k]
                count += 1
            if not sc:
                del self._cache[session_id]
        return count

    # ------------------------------------------------------------------
    # Postgres implementation
    # ------------------------------------------------------------------

    def _db_put(self, entry: CachedToolOutput) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tool_output_cache
                        (entry_id, session_id, tool_name, inputs_hash,
                         inputs_summary, output, output_ref,
                         output_size_bytes, ttl_seconds, created_at,
                         expires_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id, inputs_hash)
                    DO UPDATE SET
                        entry_id = EXCLUDED.entry_id,
                        tool_name = EXCLUDED.tool_name,
                        inputs_summary = EXCLUDED.inputs_summary,
                        output = EXCLUDED.output,
                        output_ref = EXCLUDED.output_ref,
                        output_size_bytes = EXCLUDED.output_size_bytes,
                        ttl_seconds = EXCLUDED.ttl_seconds,
                        created_at = EXCLUDED.created_at,
                        expires_at = EXCLUDED.expires_at,
                        hit_count = 0,
                        last_accessed_at = NULL
                    """,
                    (
                        entry.entry_id,
                        entry.session_id,
                        entry.tool_name,
                        entry.inputs_hash,
                        json.dumps(entry.inputs_summary) if entry.inputs_summary else None,
                        json.dumps(entry.output) if entry.output else None,
                        entry.output_ref,
                        entry.output_size_bytes,
                        entry.ttl_seconds,
                        entry.created_at,
                        entry.expires_at,
                    ),
                )
            conn.commit()

    def _db_get(self, session_id: str, inputs_hash: str) -> CachedToolOutput | None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT entry_id, session_id, tool_name, inputs_hash,
                           inputs_summary, output, output_ref,
                           output_size_bytes, hit_count, ttl_seconds,
                           created_at, expires_at, last_accessed_at
                    FROM tool_output_cache
                    WHERE session_id = %s AND inputs_hash = %s
                    """,
                    (session_id, inputs_hash),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_entry(row)

    def _db_record_hit(self, entry_id: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tool_output_cache
                    SET hit_count = hit_count + 1,
                        last_accessed_at = NOW()
                    WHERE entry_id = %s
                    """,
                    (entry_id,),
                )
            conn.commit()

    def _db_list_session(
        self, session_id: str, now: datetime
    ) -> list[CachedToolOutput]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT entry_id, session_id, tool_name, inputs_hash,
                           inputs_summary, output, output_ref,
                           output_size_bytes, hit_count, ttl_seconds,
                           created_at, expires_at, last_accessed_at
                    FROM tool_output_cache
                    WHERE session_id = %s
                      AND (expires_at IS NULL OR expires_at > %s)
                    ORDER BY created_at DESC
                    """,
                    (session_id, now),
                )
                return [self._row_to_entry(row) for row in cur.fetchall()]

    def _db_invalidate(self, session_id: str, tool_name: str | None) -> int:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                if tool_name is None:
                    cur.execute(
                        "DELETE FROM tool_output_cache WHERE session_id = %s",
                        (session_id,),
                    )
                else:
                    cur.execute(
                        "DELETE FROM tool_output_cache WHERE session_id = %s AND tool_name = %s",
                        (session_id, tool_name),
                    )
                count = cur.rowcount
            conn.commit()
            return count

    def _db_cleanup_expired(self, now: datetime) -> int:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM tool_output_cache WHERE expires_at IS NOT NULL AND expires_at <= %s",
                    (now,),
                )
                count = cur.rowcount
            conn.commit()
            return count

    @staticmethod
    def _row_to_entry(row) -> CachedToolOutput:
        return CachedToolOutput(
            entry_id=row[0],
            session_id=row[1],
            tool_name=row[2],
            inputs_hash=row[3],
            inputs_summary=json.loads(row[4]) if row[4] else None,
            output=json.loads(row[5]) if row[5] else None,
            output_ref=row[6],
            output_size_bytes=row[7],
            hit_count=row[8],
            ttl_seconds=row[9],
            created_at=row[10],
            expires_at=row[11],
            last_accessed_at=row[12],
        )
