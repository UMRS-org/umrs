"""Durable asset-grant store + pure authorization predicates (ADR-041).

Revocation tombstones (``tombstoned_at``), never deletes. The SQL surface is
thin; every authorization decision is a pure function over fetched rows so
the lifecycle (create → live → revoked/expired/pin-mismatched) is fully
testable without a database.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# WP-06 round 2 leg-b: two more kinds; 'bundle' stays deferred (no durable
# bundle-owner rows exist today and fabricating one would lie).
# WP-06 r2 leg-c (ADR-050 §3 groundwork 2026-09-16): 'group' joins the
# ladder — the groups+membership authority (0097) exists to resolve it.
ASSET_KINDS = ("model", "knowledge_graph", "endpoint", "workflow")
GRANTEE_KINDS = ("user", "project", "group")


@dataclass(frozen=True)
class AssetGrant:
    """One (possibly tombstoned) grant row."""

    grant_id: str
    asset_kind: str
    asset_id: str
    grantee_kind: str
    grantee_id: str
    version_pin: str | None = None
    expires_at: datetime | None = None
    tombstoned_at: datetime | None = None
    created_by: str = ""
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        def _iso(value: datetime | None) -> str | None:
            return value.isoformat() if isinstance(value, datetime) else None

        return {
            "grant_id": self.grant_id,
            "asset_kind": self.asset_kind,
            "asset_id": self.asset_id,
            "grantee_kind": self.grantee_kind,
            "grantee_id": self.grantee_id,
            "version_pin": self.version_pin,
            "expires_at": _iso(self.expires_at),
            "tombstoned_at": _iso(self.tombstoned_at),
            "created_by": self.created_by,
            "created_at": _iso(self.created_at),
        }


def grant_is_live(grant: AssetGrant, *, now: datetime) -> bool:
    """A grant authorizes only while untombstoned and unexpired."""
    if grant.tombstoned_at is not None:
        return False
    return grant.expires_at is None or now < grant.expires_at


def grant_authorizes(
    grant: AssetGrant,
    *,
    asset_kind: str,
    asset_id: str,
    user_id: str,
    project_id: str,
    asset_version: str | None = None,
    now: datetime,
) -> bool:
    """Does this grant authorize the caller's read of the asset?

    - asset identity must match exactly
    - grantee matches on user id, or on the caller's verified project id
    - a version pin authorizes only that exact version; assets that do not
      expose a version (``asset_version=None``) can never satisfy a pin —
      fail-closed by design (ADR-041)
    """
    if grant.asset_kind != asset_kind or grant.asset_id != asset_id:
        return False
    if not grant_is_live(grant, now=now):
        return False
    grantee_match = (grant.grantee_kind == "user" and grant.grantee_id == user_id) or (
        grant.grantee_kind == "project"
        and bool(project_id)
        and grant.grantee_id == project_id
    )
    if not grantee_match:
        return False
    if grant.version_pin is not None:
        return asset_version is not None and grant.version_pin == asset_version
    return True


def _as_aware(value: datetime) -> datetime:
    """Normalize a possibly-naive datetime to UTC for comparisons."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class GrantStore:
    """psycopg2-backed grant persistence. Revoke tombstones; never DELETE."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def create_grant(
        self,
        *,
        asset_kind: str,
        asset_id: str,
        grantee_kind: str,
        grantee_id: str,
        created_by: str,
        version_pin: str | None = None,
        expires_at: datetime | None = None,
    ) -> AssetGrant:
        if asset_kind not in ASSET_KINDS:
            raise ValueError(f"unknown asset_kind: {asset_kind}")
        if grantee_kind not in GRANTEE_KINDS:
            raise ValueError(f"unknown grantee_kind: {grantee_kind}")
        grant = AssetGrant(
            grant_id=f"grt-{secrets.token_hex(8)}",
            asset_kind=asset_kind,
            asset_id=asset_id,
            grantee_kind=grantee_kind,
            grantee_id=grantee_id,
            version_pin=version_pin,
            expires_at=expires_at,
            created_by=created_by,
        )
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO asset_grants
                    (grant_id, asset_kind, asset_id, grantee_kind, grantee_id,
                     version_pin, expires_at, created_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    grant.grant_id,
                    grant.asset_kind,
                    grant.asset_id,
                    grant.grantee_kind,
                    grant.grantee_id,
                    grant.version_pin,
                    grant.expires_at,
                    grant.created_by,
                ),
            )
            self._conn.commit()
        return grant

    def revoke_grant(self, grant_id: str, *, now: datetime) -> bool:
        """Tombstone a live grant. Returns True when a row was tombstoned."""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE asset_grants
                SET tombstoned_at = %s
                WHERE grant_id = %s AND tombstoned_at IS NULL
                """,
                (_as_aware(now), grant_id),
            )
            revoked = cur.rowcount > 0
            self._conn.commit()
        return revoked

    def get_grant(self, grant_id: str) -> AssetGrant | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT grant_id, asset_kind, asset_id, grantee_kind, grantee_id,
                       version_pin, expires_at, tombstoned_at, created_by, created_at
                FROM asset_grants WHERE grant_id = %s
                """,
                (grant_id,),
            )
            row = cur.fetchone()
        return _row_to_grant(row) if row else None

    def list_for_asset(
        self, asset_kind: str, asset_id: str, *, include_tombstoned: bool = False
    ) -> list[AssetGrant]:
        clause = "" if include_tombstoned else " AND tombstoned_at IS NULL"
        return self._select(
            f"WHERE asset_kind = %s AND asset_id = %s{clause} ORDER BY created_at",
            (asset_kind, asset_id),
        )

    def list_for_grantee(self, grantee_kind: str, grantee_id: str) -> list[AssetGrant]:
        """Live grants received by one grantee."""
        return self._select(
            "WHERE grantee_kind = %s AND grantee_id = %s"
            " AND tombstoned_at IS NULL ORDER BY created_at",
            (grantee_kind, grantee_id),
        )

    def _select(self, where: str, params: tuple[Any, ...]) -> list[AssetGrant]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                SELECT grant_id, asset_kind, asset_id, grantee_kind, grantee_id,
                       version_pin, expires_at, tombstoned_at, created_by, created_at
                FROM asset_grants
                """
                + where,
                params,
            )
            rows = cur.fetchall()
        return [_row_to_grant(row) for row in rows]


def _row_to_grant(row: Any) -> AssetGrant:
    (
        grant_id,
        asset_kind,
        asset_id,
        grantee_kind,
        grantee_id,
        version_pin,
        expires_at,
        tombstoned_at,
        created_by,
        created_at,
    ) = row
    return AssetGrant(
        grant_id=grant_id,
        asset_kind=asset_kind,
        asset_id=asset_id,
        grantee_kind=grantee_kind,
        grantee_id=grantee_id,
        version_pin=version_pin,
        expires_at=expires_at,
        tombstoned_at=tombstoned_at,
        created_by=created_by,
        created_at=created_at,
    )
