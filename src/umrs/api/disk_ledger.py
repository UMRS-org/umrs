"""Per-user disk budget ledger (ADR-051 D-3).

Attribution ledger: rows `user_disk_usage` (per-user sum) + `user_disk_objects`
(hash-dedup per uploader — each unique content hash is charged against the
uploader ONCE; a re-upload of the same hash under the same user is free).

Detour accounting deliberately does NOT exist (ambient rows still count).
The module guards ONE write point per lane (multipart upload receipts and
s3-staged durable uploads downstream of it).
"""

from __future__ import annotations

import os
from typing import Any

from umrs.api.typed_errors import PlatformError


def disk_bytes_max() -> int:
    """Per-user disk budget in bytes (``0`` = uncapped)."""
    return max(0, int(os.environ.get("UMRS_USER_DISK_BYTES_MAX", str(100 * (1 << 30)))))


class DiskLedger:
    """psycopg2-shaped sync store over the 0096 ledger tables."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def usage(self, user_id: str) -> int:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT bytes_used FROM user_disk_usage WHERE user_id = %s", (user_id,)
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def _hash_charged(self, user_id: str, content_hash: str) -> bool:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT 1 FROM user_disk_objects WHERE user_id = %s AND content_hash = %s",
            (user_id, content_hash),
        )
        return cur.fetchone() is not None

    def charge(
        self, user_id: str, *, content_hash: str, byte_size: int
    ) -> dict[str, int]:
        """Charge ``byte_size`` of one unique object against one user, or
        raise typed quota.disk_bytes_exceeded (never a silent overshoot).

        Re-uploads of the same hash cost nothing (named dedup contract)."""
        if self._hash_charged(user_id, content_hash):
            usage = self.usage(user_id)
            return {"charged": 0, "usage": usage, "limit": disk_bytes_max()}
        cur = self._conn.cursor()
        usage = self.usage(user_id)
        limit = disk_bytes_max()
        if limit > 0 and usage + byte_size > limit:
            raise_typed_quota(
                limit=limit, usage=usage, byte_size=byte_size, user_id=user_id
            )
        byte_size = int(byte_size or 0)
        cur.execute(
            """INSERT INTO user_disk_usage (user_id, bytes_used, updated_at)
               VALUES (%s, %s, now())
               ON CONFLICT (user_id) DO UPDATE
                 SET bytes_used = user_disk_usage.bytes_used + %s,
                     updated_at = now()""",
            (user_id, byte_size, byte_size),
        )
        cur.execute(
            """INSERT INTO user_disk_objects (user_id, content_hash, bytes)
               VALUES (%s, %s, %s)
               ON CONFLICT (content_hash, user_id) DO NOTHING""",
            (user_id, content_hash, byte_size),
        )
        self._conn.commit()
        return {"charged": byte_size, "usage": usage + byte_size, "limit": limit}


def raise_typed_quota(*, limit: int, usage: int, byte_size: int, user_id: str) -> None:
    from fastapi import HTTPException

    raise HTTPException(
        **PlatformError.quota(
            "disk_bytes_exceeded",
            f"disk budget exceeded for user {user_id}: "
            f"current={usage} extra={byte_size} limit={limit}",
        ).to_http_parts()
    )


__all__ = ["DiskLedger", "disk_bytes_max"]
