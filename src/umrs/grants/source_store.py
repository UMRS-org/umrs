"""Source grant store (R3-1, ADR-046 §R3 + ADR-041 lineage).

Owner shares a sealed provider source with another user. Store semantics
mirror the asset-grant store (0083): tombstone-on-revoke (never DELETE),
live-grant uniqueness on (source_id, grantee), re-grant frees after revoke.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SourceGrant:
    grant_id: str
    source_id: str
    owner: str
    grantee: str
    allow_paths: tuple[str, ...]
    budget_usd: Any | None
    expires_at: Any | None
    tombstoned_at: Any | None
    created_by: str
    created_at: Any

    def to_dict(self) -> dict[str, Any]:
        def _iso(v: Any) -> Any:
            return v.isoformat() if hasattr(v, "isoformat") else v

        return {
            "grant_id": self.grant_id,
            "source_id": self.source_id,
            "owner": self.owner,
            "grantee": self.grantee,
            "allow_paths": list(self.allow_paths),
            "budget_usd": float(self.budget_usd) if self.budget_usd is not None else None,
            "expires_at": _iso(self.expires_at),
            "tombstoned_at": _iso(self.tombstoned_at),
            "created_by": self.created_by,
            "created_at": _iso(self.created_at),
        }


def _row_to_grant(row: Any) -> SourceGrant:
    return SourceGrant(
        grant_id=str(row["grant_id"]),
        source_id=str(row["source_id"]),
        owner=str(row["owner"]),
        grantee=str(row["grantee"]),
        allow_paths=tuple(str(p) for p in (row["allow_paths"] or [])),
        budget_usd=row["budget_usd"],
        expires_at=row["expires_at"],
        tombstoned_at=row["tombstoned_at"],
        created_by=str(row["created_by"]),
        created_at=row["created_at"],
    )


class SourceGrantStore:
    """Postgres store for source_grants (same pool discipline as the sealed store)."""

    def __init__(self, pool: Any) -> None:
        if pool is None:
            raise RuntimeError("source grant store requires the shared umrs_db pool")
        self._pool = pool

    def effective_granted_source_id(
        self, pool_row: Any, user_id: str, source_id: str
    ) -> bool:
        """True when user_id owns or holds a live grant on source_id."""
        # ownership check happens at the sealed-source layer (meta read);
        # here: live grant?
        if pool_row is not None:
            return True
        return self.has_live_grant(user_id=user_id, source_id=source_id)

    def has_live_grant(self, *, user_id: str, source_id: str) -> bool:
        conn = self._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT 1 FROM source_grants WHERE source_id = %s AND grantee = %s "
                    "AND tombstoned_at IS NULL AND (expires_at IS NULL OR expires_at > now())",
                    (source_id, user_id),
                )
                return cur.fetchone() is not None
        finally:
            self._pool.putconn(conn)

    def create_grant(
        self,
        *,
        source_id: str,
        owner: str,
        grantee: str,
        allow_paths: tuple[str, ...],
        budget_usd: Any | None,
        expires_at: Any | None,
        now: datetime,
    ) -> SourceGrant:
        del now  # API consistency with the asset-grant store; DB side stamps created_at
        grant_id = f"sg-{uuid.uuid4().hex[:12]}"
        conn = self._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "INSERT INTO source_grants (grant_id, source_id, owner, grantee, "
                    "allow_paths, budget_usd, expires_at, created_by) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "RETURNING grant_id, source_id, owner, grantee, allow_paths, "
                    "budget_usd, expires_at, tombstoned_at, created_by, created_at",
                    (
                        grant_id,
                        source_id,
                        owner,
                        grantee,
                        list(allow_paths),
                        budget_usd,
                        expires_at,
                        owner,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        finally:
            self._pool.putconn(conn)
        return _row_to_grant(row)

    def revoke_grant(self, grant_id: str, *, actor: str, now: datetime) -> bool:
        del actor  # stamped via the audit row the route writes, not the tombstone
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE source_grants SET tombstoned_at = %s WHERE grant_id = %s "
                    "AND tombstoned_at IS NULL",
                    (now, grant_id),
                )
                moved = cur.rowcount > 0
            conn.commit()
        finally:
            self._pool.putconn(conn)
        return moved

    def purge_tombstoned_older_than(self, days: int) -> int:
        """R3-2: hard-delete grants whose tombstone predates the retention window.

        Tombstones are an audit trail within TTL; beyond it the table is pruned
        (detail stays in user_source_audit until its own TTL fires).
        """
        if days <= 0:
            return 0
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM source_grants WHERE tombstoned_at IS NOT NULL "
                    "AND tombstoned_at < now() - make_interval(days => %s)",
                    (days,),
                )
                removed = cur.rowcount
            conn.commit()
        finally:
            self._pool.putconn(conn)
        return removed

    def get_grant(self, grant_id: str) -> SourceGrant | None:
        conn = self._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT grant_id, source_id, owner, grantee, allow_paths, "
                    "budget_usd, expires_at, tombstoned_at, created_by, created_at "
                    "FROM source_grants WHERE grant_id = %s",
                    (grant_id,),
                )
                row = cur.fetchone()
        finally:
            self._pool.putconn(conn)
        return _row_to_grant(row) if row else None

    def list_for_owner(self, owner: str) -> list[SourceGrant]:
        return self._list("owner = %s", (owner,))

    def list_for_grantee(self, grantee: str) -> list[SourceGrant]:
        return self._list(
            "grantee = %s AND tombstoned_at IS NULL AND (expires_at IS NULL OR expires_at > now())",
            (grantee,),
        )

    def list_for_source(self, source_id: str) -> list[SourceGrant]:
        return self._list("source_id = %s", (source_id,))

    def _list(self, where: str, params: tuple) -> list[SourceGrant]:
        conn = self._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT grant_id, source_id, owner, grantee, allow_paths, "
                    "budget_usd, expires_at, tombstoned_at, created_by, created_at "
                    f"FROM source_grants WHERE {where} ORDER BY created_at DESC",
                    params,
                )
                rows = cur.fetchall() or []
        finally:
            self._pool.putconn(conn)
        return [_row_to_grant(r) for r in rows]
