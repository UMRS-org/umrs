"""Sealed user-credential store (R1-1, design §3.1).

Persists ``user_sources_sealed`` rows. Two hard rules:

1. Metadata reads NEVER select ``dek_wrapped``/``nonce``/``ciphertext``
   (column grants mirror the discipline; the code never depends on them
   being present, and tests assert the SELECT lists).
2. The unseal path is the inference proxy's — this module exposes
   ``fetch_for_unseal`` only for that caller class.

All SQL is parameterized; the validator in ``provider_url`` runs BEFORE a
write reaches this store.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

_METADATA_COLUMNS = (
    "id, owner, name, provider_kind, base_url, allow_host, allow_paths, "
    "key_fingerprint, seal_key_id, created_at, updated_at, tombstoned_at"
)

_INSERT_SQL = """
INSERT INTO user_sources_sealed (
    owner, name, provider_kind, base_url, allow_host, allow_paths,
    key_fingerprint, seal_key_id, dek_wrapped, nonce, ciphertext
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""

_LIST_SQL = f"""
SELECT {_METADATA_COLUMNS}
  FROM user_sources_sealed
 WHERE owner = %s AND tombstoned_at IS NULL
 ORDER BY created_at
"""

_GET_META_SQL = f"""
SELECT {_METADATA_COLUMNS}
  FROM user_sources_sealed
 WHERE owner = %s AND name = %s AND tombstoned_at IS NULL
"""

_GET_BY_ID_META_SQL = f"""
SELECT {_METADATA_COLUMNS} FROM user_sources_sealed WHERE id = %s
"""

_TOMBSTONE_SQL = """
UPDATE user_sources_sealed
   SET tombstoned_at = now(), updated_at = now()
 WHERE owner = %s AND name = %s AND tombstoned_at IS NULL
RETURNING id
"""

_UNSEAL_SQL = """
SELECT id, owner, name, provider_kind, base_url, allow_host, allow_paths,
       key_fingerprint, seal_key_id, dek_wrapped, nonce, ciphertext
  FROM user_sources_sealed
 WHERE id = %s AND tombstoned_at IS NULL
"""

_AUDIT_SQL = """
INSERT INTO user_source_audit (action, actor_type, owner, source_id, detail)
VALUES (%s, %s, %s, %s, %s::jsonb)
"""


class SealedStoreUnavailable(RuntimeError):
    """Raised when no Postgres pool is available for the sealed store."""


@dataclass(frozen=True)
class SealedSourceMeta:
    """Metadata row (never carries key material)."""

    id: str
    owner: str
    name: str
    provider_kind: str
    base_url: str
    allow_host: str
    allow_paths: tuple[str, ...]
    key_fingerprint: str
    seal_key_id: str
    created_at: Any
    updated_at: Any
    tombstoned_at: Any
    active_leases: int | None = None

    def to_dict(self) -> dict[str, Any]:
        def _iso(v: Any) -> Any:
            return v.isoformat() if hasattr(v, "isoformat") else v

        payload = {
            "id": self.id,
            "owner": self.owner,
            "name": self.name,
            "provider_kind": self.provider_kind,
            "base_url": self.base_url,
            "allow_host": self.allow_host,
            "allow_paths": list(self.allow_paths),
            "key_fingerprint": self.key_fingerprint,
            "seal_key_id": self.seal_key_id,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "tombstoned_at": _iso(self.tombstoned_at),
        }
        if self.active_leases is not None:
            payload["active_leases"] = self.active_leases
        return payload


@dataclass(frozen=True)
class UnsealFetch:
    """Full row for the inference proxy's unseal path only."""

    id: str
    owner: str
    name: str
    provider_kind: str
    allow_host: str
    allow_paths: tuple[str, ...]
    key_fingerprint: str
    dek_wrapped: bytes
    nonce: bytes
    ciphertext: bytes


class SealedSourceStore:
    """Postgres store for sealed user credentials (plain pool, RealDict rows)."""

    def __init__(self, pool: Any) -> None:
        if pool is None:
            raise SealedStoreUnavailable(
                "sealed source store requires the shared umrs_db pool"
            )
        self._pool = pool

    def _fetchone(self, sql: str, params: tuple) -> Any | None:
        conn = self._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
            conn.commit()
            return row
        finally:
            self._pool.putconn(conn)

    def _fetchall(self, sql: str, params: tuple) -> list:
        conn = self._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
                return cur.fetchall() or []
        finally:
            self._pool.putconn(conn)

    @staticmethod
    def _meta_from_row(row: Any) -> SealedSourceMeta:
        return SealedSourceMeta(
            id=str(row["id"]),
            owner=str(row["owner"]),
            name=str(row["name"]),
            provider_kind=str(row["provider_kind"]),
            base_url=str(row["base_url"]),
            allow_host=str(row["allow_host"]),
            allow_paths=tuple(str(p) for p in (row["allow_paths"] or [])),
            key_fingerprint=str(row["key_fingerprint"]),
            seal_key_id=str(row["seal_key_id"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            tombstoned_at=row["tombstoned_at"],
            active_leases=int(row["active_leases"]) if "active_leases" in row else None,
        )

    def insert(
        self,
        *,
        owner: str,
        name: str,
        provider_kind: str,
        base_url: str,
        allow_host: str,
        allow_paths: tuple[str, ...],
        key_fingerprint: str,
        seal_key_id: str,
        dek_wrapped: bytes,
        nonce: bytes,
        ciphertext: bytes,
    ) -> str:
        """Write one sealed row (re-seal updates after a prior tombstone)."""
        row = self._fetchone(
            _INSERT_SQL,
            (
                owner,
                name,
                provider_kind,
                base_url,
                allow_host,
                list(allow_paths),
                key_fingerprint,
                seal_key_id,
                dek_wrapped,
                nonce,
                ciphertext,
            ),
        )
        return str(row["id"])

    def list_mine(
        self, owner: str, *, with_lease_counts: bool = False
    ) -> list[SealedSourceMeta]:
        if with_lease_counts:
            # Per row count of live leases (N+1 avoided via lateral join).
            rows = []
            conn = self._pool.getconn()
            try:
                from psycopg2.extras import RealDictCursor

                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        f"SELECT {_METADATA_COLUMNS} FROM user_sources_sealed "
                        "WHERE owner = %s AND tombstoned_at IS NULL ORDER BY created_at",
                        (owner,),
                    )
                    metas = [self._meta_from_row(r) for r in cur.fetchall() or []]
                for meta in metas:
                    with conn.cursor(cursor_factory=RealDictCursor) as cur2:
                        cur2.execute(
                            "SELECT count(*) AS c FROM secret_leases l "
                            "WHERE %s::uuid = ANY (l.source_ids) "
                            "AND NOT l.is_revoked AND l.expires_at > now()",
                            (meta.id,),
                        )
                        count = int(cur2.fetchone()["c"])
                    rows.append(
                        SealedSourceMeta(**{**meta.__dict__, "active_leases": count})
                    )
                return rows
            finally:
                self._pool.putconn(conn)
        return [self._meta_from_row(r) for r in self._fetchall(_LIST_SQL, (owner,))]

    def get_meta(self, owner: str, name: str) -> SealedSourceMeta | None:
        row = self._fetchone(_GET_META_SQL, (owner, name))
        return self._meta_from_row(row) if row else None

    def get_meta_by_id(self, source_id: str) -> SealedSourceMeta | None:
        row = self._fetchone(_GET_BY_ID_META_SQL, (source_id,))
        return self._meta_from_row(row) if row else None

    def tombstone(self, owner: str, name: str) -> str | None:
        """Soft-delete; the revoke cascade runs at the route layer."""
        row = self._fetchone(_TOMBSTONE_SQL, (owner, name))
        return str(row["id"]) if row else None

    def fetch_for_unseal(self, source_id: str) -> UnsealFetch | None:
        """Proxy-only read incl. ciphertext columns."""
        row = self._fetchone(_UNSEAL_SQL, (source_id,))
        if row is None:
            return None
        return UnsealFetch(
            id=str(row["id"]),
            owner=str(row["owner"]),
            name=str(row["name"]),
            provider_kind=str(row["provider_kind"]),
            allow_host=str(row["allow_host"]),
            allow_paths=tuple(str(p) for p in (row["allow_paths"] or [])),
            key_fingerprint=str(row["key_fingerprint"]),
            dek_wrapped=bytes(row["dek_wrapped"]),
            nonce=bytes(row["nonce"]),
            ciphertext=bytes(row["ciphertext"]),
        )

    def _execute(self, sql: str, params: tuple) -> None:
        conn = self._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, params)
            conn.commit()
        finally:
            self._pool.putconn(conn)

    def audit(
        self,
        *,
        action: str,
        owner: str,
        source_id: str | None,
        detail: dict[str, Any],
        actor_type: str = "user",
    ) -> None:
        import json

        self._execute(
            _AUDIT_SQL,
            (action, actor_type, owner, source_id, json.dumps(detail)),
        )


def load_seal_pubkey_configured() -> str | None:
    """Configured seal pubkey from env, or None (push must be feature-off)."""
    return os.environ.get("UMRS_SEAL_PUBKEY_B64", "").strip() or None
