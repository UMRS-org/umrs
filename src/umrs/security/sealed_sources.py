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
    next_lease_expiry: Any = None  # latest expires_at across live leases (badge countdown)

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
        if self.next_lease_expiry is not None:
            payload["next_lease_expiry"] = _iso(self.next_lease_expiry)
        return payload


@dataclass(frozen=True)
class UnsealFetch:
    """Full row for the inference proxy's unseal path only."""

    id: str
    owner: str
    name: str
    provider_kind: str
    base_url: str
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
                            "SELECT count(*) AS c, max(l.expires_at) AS latest "
                            "FROM secret_leases l "
                            "WHERE %s::uuid = ANY (l.source_ids) "
                            "AND NOT l.is_revoked AND l.expires_at > now()",
                            (meta.id,),
                        )
                        row2 = cur2.fetchone()
                    rows.append(
                        SealedSourceMeta(
                            **{
                                **meta.__dict__,
                                "active_leases": int(row2["c"]),
                                "next_lease_expiry": row2["latest"],
                            }
                        )
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
            base_url=str(row["base_url"]),
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

    @staticmethod
    def _pubkey_from_privkey(privkey_b64: str) -> str:
        """X25519 public half derives from the private key bytes."""
        import base64 as _b64

        from .seal import _crypto

        c = _crypto()
        priv = c["X25519PrivateKey"].from_private_bytes(_b64.b64decode(privkey_b64))
        raw = priv.public_key().public_bytes(
            c["serialization"].Encoding.Raw,
            c["serialization"].PublicFormat.Raw,
        )
        return _b64.b64encode(raw).decode()

    def purge_audit_older_than(self, days: int) -> dict[str, int]:
        """R3-2 retention TTL. Deletes audit rows older than ``days`` days.

        Pair rule: user_source_audit + secret_audit are purged together so
        the two halves of one security event expire as one.
        Returns row counts per table.
        """
        if days <= 0:
            return {"user_source_audit": 0, "secret_audit": 0}
        counts = {"user_source_audit": 0, "secret_audit": 0}
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM user_source_audit WHERE created_at < now() - make_interval(days => %s)",
                    (days,),
                )
                counts["user_source_audit"] = cur.rowcount
                cur.execute(
                    "DELETE FROM secret_audit WHERE created_at < now() - make_interval(days => %s)",
                    (days,),
                )
                counts["secret_audit"] = cur.rowcount
            conn.commit()
        finally:
            self._pool.putconn(conn)
        return counts

    def export_audit(
        self, *, owner: str | None = None, since_days: int = 30, limit: int = 10_000
    ) -> list[dict[str, Any]]:
        """R3-2 export hook: flattened audit rows for accounting (owner filter optional)."""
        conn = self._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if owner:
                    cur.execute(
                        "SELECT created_at, action, actor_type, owner, source_id, detail "
                        "FROM user_source_audit WHERE owner = %s AND created_at > now() - make_interval(days => %s) "
                        "ORDER BY created_at LIMIT %s",
                        (owner, since_days, limit),
                    )
                else:
                    cur.execute(
                        "SELECT created_at, action, actor_type, owner, source_id, detail "
                        "FROM user_source_audit WHERE created_at > now() - make_interval(days => %s) "
                        "ORDER BY created_at LIMIT %s",
                        (since_days, limit),
                    )
                rows = cur.fetchall() or []
        finally:
            self._pool.putconn(conn)
        return [
            {
                "created_at": r["created_at"].isoformat() if hasattr(r["created_at"], "isoformat") else r["created_at"],
                "action": r["action"],
                "actor_type": r["actor_type"],
                "owner": r["owner"],
                "source_id": str(r["source_id"]) if r["source_id"] else None,
                "detail": r["detail"],
            }
            for r in rows
        ]

    def rewrap_all(self, *, old_privkey_b64: str, new_pubkey_b64: str) -> dict[str, int]:
        """Rotation job (R2-5): rewrite the OLD key-epoch rows' DEK wrapping in one txn.

        Scope: rows with ``seal_key_id == <old pub id>`` (foreign-epoch rows
        never move). Unwrap all first; any failure aborts before a single write.
        Returns {"rows_rewrapped": n, "seal_key_id": new_id}.
        """
        from .rotation import rewrap_sealed_rows
        from .seal import seal_key_id

        old_pub = self._pubkey_from_privkey(old_privkey_b64)
        old_id = seal_key_id(old_pub)

        conn = self._pool.getconn()
        try:
            from psycopg2.extras import RealDictCursor

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, dek_wrapped FROM user_sources_sealed "
                    "WHERE tombstoned_at IS NULL AND seal_key_id = %s",
                    (old_id,),
                )
                rows = [(str(r["id"]), bytes(r["dek_wrapped"])) for r in cur.fetchall() or []]

            # First: rewrap everything in memory (fail before any write).
            rewrapped = list(
                rewrap_sealed_rows(
                    rows,
                    old_privkey_b64=old_privkey_b64,
                    new_pubkey_b64=new_pubkey_b64,
                    row_attrs=lambda r: r,
                )
            )
            new_seal_id = seal_key_id(new_pubkey_b64)

            with conn.cursor() as cur:
                for row_id, new_wrapped, new_id in rewrapped:
                    cur.execute(
                        "UPDATE user_sources_sealed "
                        "SET dek_wrapped = %s, seal_key_id = %s, updated_at = now() "
                        "WHERE id = %s",
                        (new_wrapped, new_id, row_id),
                    )
            conn.commit()
            return {"rows_rewrapped": len(rewrapped), "seal_key_id": new_seal_id}
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
