"""Site accounts and project links (migrations 0134/0135; ADR-112 wave B1).

Every umrs-front shares ``umrs_db``. A store is bound to one venue and every
statement it runs names that venue (invariant I-11), so one front never reads
or changes another front's rows.

Rows come back as the wire contract's views (plan §4 B1.5a): the account view
and the link view. Neither can carry a private key: no column holds one.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

ACCOUNT_STATUSES = ("pending", "active", "suspended", "revoked")
GATE_PROBES = ("refused", "bypassed", "unknown")
LINK_STATUSES = ("active", "suspended")

# The account view of the wire contract, in its order.
ACCOUNT_VIEW = (
    "venue",
    "username",
    "site_login",
    "method",
    "status",
    "verified_login",
    "verified_assocs",
    "verified_at",
    "gate_version",
    "gate_sha256",
    "gate_probe",
    "transport_key_id",
    "transport_fingerprint",
    "signer_fingerprints",
    "uv_required",
)
ACCOUNT_COLUMNS = (*ACCOUNT_VIEW, "created_at", "updated_at")
# The link view of the wire contract (suspended_reason stays server-side).
LINK_COLUMNS = (
    "venue",
    "project_id",
    "username",
    "slurm_account",
    "status",
    "created_at",
)

# Fields ``record_verified`` writes; everything else is the store's own.
VERIFIED_FIELDS = (
    "site_login",
    "verified_login",
    "verified_assocs",
    "gate_version",
    "gate_sha256",
    "signer_fingerprints",
    "uv_required",
)
_JSON_FIELDS = frozenset({"verified_assocs", "signer_fingerprints"})


def _iso(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def account_view(row: dict[str, Any]) -> dict[str, Any]:
    """The wire contract's account view of a row (public information only)."""
    return {key: _iso(row.get(key)) for key in ACCOUNT_VIEW}


def link_view(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _iso(row.get(key)) for key in LINK_COLUMNS}


# ── Postgres ──────────────────────────────────────────────────────────────

_ACCOUNT_SELECT = f"SELECT {', '.join(ACCOUNT_COLUMNS)} FROM site_accounts"
_ACCOUNT_RETURNING = f" RETURNING {', '.join(ACCOUNT_COLUMNS)}"
_LINK_SELECT = f"SELECT {', '.join(LINK_COLUMNS)} FROM project_site_allocations"
_LINK_RETURNING = f" RETURNING {', '.join(LINK_COLUMNS)}"

_ISSUE_KEY_SQL = (
    "INSERT INTO site_accounts (venue, username, transport_key_id, transport_fingerprint)"
    " VALUES (%s, %s, %s, %s)"
    " ON CONFLICT (venue, username) DO UPDATE SET"
    " transport_key_id = EXCLUDED.transport_key_id,"
    " transport_fingerprint = EXCLUDED.transport_fingerprint,"
    " status = 'pending', gate_probe = 'unknown', updated_at = now()"
    + _ACCOUNT_RETURNING
)
_VERIFIED_SQL = (
    "UPDATE site_accounts SET site_login = %s, verified_login = %s,"
    " verified_assocs = %s::jsonb, gate_version = %s, gate_sha256 = %s,"
    " signer_fingerprints = %s::jsonb, uv_required = %s, verified_at = now(),"
    " gate_probe = 'refused', status = 'active', updated_at = now()"
    " WHERE venue = %s AND username = %s AND transport_key_id = %s" + _ACCOUNT_RETURNING
)
# D-7: an account is active only while the server reaches the gate. A probe
# that the gate did not refuse demotes an active account in the same write.
_PROBE_SQL = (
    "UPDATE site_accounts SET gate_probe = %s,"
    " status = CASE WHEN %s <> 'refused' AND status = 'active'"
    " THEN 'suspended' ELSE status END, updated_at = now()"
    " WHERE venue = %s AND username = %s"
)
_SUSPEND_ACCOUNT_SQL = (
    "UPDATE site_accounts SET status = 'suspended', updated_at = now()"
    " WHERE venue = %s AND username = %s AND status = 'active' RETURNING username"
)
_REVOKE_SQL = (
    "UPDATE site_accounts SET status = 'revoked', updated_at = now()"
    " WHERE venue = %s AND username = %s RETURNING username"
)
_UPSERT_LINK_SQL = (
    "INSERT INTO project_site_allocations (venue, project_id, username, slurm_account)"
    " VALUES (%s, %s, %s, %s)"
    " ON CONFLICT (venue, project_id, username) DO UPDATE SET"
    " slurm_account = EXCLUDED.slurm_account, status = 'active', suspended_reason = ''"
    + _LINK_RETURNING
)
_DELETE_LINK_SQL = (
    "DELETE FROM project_site_allocations"
    " WHERE venue = %s AND username = %s AND project_id = %s RETURNING project_id"
)
_SUSPEND_LINKS_SQL = (
    "UPDATE project_site_allocations SET status = 'suspended', suspended_reason = %s"
    " WHERE venue = %s AND username = %s AND status = 'active'"
    " AND (%s = '' OR project_id = %s) RETURNING project_id"
)


class SiteAccountStore:
    """``site_accounts`` / ``project_site_allocations`` of one venue over a
    psycopg2 pool (``getconn`` / ``putconn``)."""

    def __init__(self, pool: Any, venue: str) -> None:
        if not venue:
            raise ValueError("a site account store needs its venue (site name)")
        self._pool = pool
        self.venue = venue

    def _rows(self, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() if cur.description else []
            conn.commit()
            return list(rows)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _account(self, sql: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
        rows = self._rows(sql, params)
        return account_view(dict(zip(ACCOUNT_COLUMNS, rows[0]))) if rows else None

    def _links(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [
            link_view(dict(zip(LINK_COLUMNS, row))) for row in self._rows(sql, params)
        ]

    # accounts

    def get_account(self, username: str) -> dict[str, Any] | None:
        sql = _ACCOUNT_SELECT + " WHERE venue = %s AND username = %s"
        return self._account(sql, (self.venue, username))

    def list_accounts(self) -> list[dict[str, Any]]:
        rows = self._rows(
            _ACCOUNT_SELECT + " WHERE venue = %s ORDER BY username", (self.venue,)
        )
        return [account_view(dict(zip(ACCOUNT_COLUMNS, row))) for row in rows]

    def issue_key(self, username: str, key_id: str, fingerprint: str) -> dict[str, Any]:
        """Record a new transport key: the account (re)starts as pending."""
        row = self._account(_ISSUE_KEY_SQL, (self.venue, username, key_id, fingerprint))
        if row is None:
            raise RuntimeError(f"site account upsert lost ({self.venue}/{username})")
        return row

    def record_verified(
        self, username: str, key_id: str, verified: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Store what the gate reported and activate the account, only while
        ``key_id`` is still its transport key (None otherwise)."""
        values = tuple(
            json.dumps(verified[k]) if k in _JSON_FIELDS else verified[k]
            for k in VERIFIED_FIELDS
        )
        return self._account(_VERIFIED_SQL, (*values, self.venue, username, key_id))

    def record_probe(self, username: str, probe: str) -> None:
        """Store the probe answer; anything but ``refused`` suspends an
        active account (a failed re-verify never leaves it active)."""
        self._rows(_PROBE_SQL, (probe, probe, self.venue, username))

    def suspend_account(self, username: str) -> bool:
        """An active account the gate no longer vouches for: suspended."""
        return bool(self._rows(_SUSPEND_ACCOUNT_SQL, (self.venue, username)))

    def revoke_account(self, username: str) -> bool:
        return bool(self._rows(_REVOKE_SQL, (self.venue, username)))

    # links

    def list_links(self, username: str = "") -> list[dict[str, Any]]:
        sql = _LINK_SELECT + (
            " WHERE venue = %s AND (%s = '' OR username = %s) ORDER BY username, project_id"
        )
        return self._links(sql, (self.venue, username, username))

    def upsert_link(
        self, username: str, project_id: str, slurm_account: str
    ) -> dict[str, Any]:
        links = self._links(
            _UPSERT_LINK_SQL, (self.venue, project_id, username, slurm_account)
        )
        if not links:
            raise RuntimeError(
                f"link upsert lost ({self.venue}/{project_id}/{username})"
            )
        return links[0]

    def delete_link(self, username: str, project_id: str) -> bool:
        return bool(self._rows(_DELETE_LINK_SQL, (self.venue, username, project_id)))

    def suspend_links(self, username: str, reason: str, project_id: str = "") -> int:
        params = (reason, self.venue, username, project_id, project_id)
        return len(self._rows(_SUSPEND_LINKS_SQL, params))


# ── in-memory twin (tests, and a server without a database in tests) ─────


@dataclass
class InMemorySiteAccountStore:
    """The same API over dicts. Pass one ``tables`` to several venues' stores
    to model the shared database."""

    venue: str
    tables: dict[str, dict[tuple[str, ...], dict[str, Any]]] = field(
        default_factory=lambda: {"accounts": {}, "links": {}}
    )

    @property
    def _accounts(self) -> dict[tuple[str, ...], dict[str, Any]]:
        return self.tables["accounts"]

    @property
    def _link_rows(self) -> dict[tuple[str, ...], dict[str, Any]]:
        return self.tables["links"]

    def get_account(self, username: str) -> dict[str, Any] | None:
        row = self._accounts.get((self.venue, username))
        return account_view(copy.deepcopy(row)) if row else None

    def list_accounts(self) -> list[dict[str, Any]]:
        rows = [r for (v, _u), r in sorted(self._accounts.items()) if v == self.venue]
        return [account_view(copy.deepcopy(r)) for r in rows]

    def issue_key(self, username: str, key_id: str, fingerprint: str) -> dict[str, Any]:
        row = self._accounts.setdefault(
            (self.venue, username), _blank_account(self.venue, username)
        )
        row.update(
            transport_key_id=key_id,
            transport_fingerprint=fingerprint,
            status="pending",
            gate_probe="unknown",
            updated_at=_now(),
        )
        return account_view(copy.deepcopy(row))

    def record_verified(
        self, username: str, key_id: str, verified: dict[str, Any]
    ) -> dict[str, Any] | None:
        row = self._accounts.get((self.venue, username))
        if row is None or row["transport_key_id"] != key_id:
            return None
        row.update({k: copy.deepcopy(verified[k]) for k in VERIFIED_FIELDS})
        row.update(
            verified_at=_now(), gate_probe="refused", status="active", updated_at=_now()
        )
        return account_view(copy.deepcopy(row))

    def record_probe(self, username: str, probe: str) -> None:
        row = self._accounts.get((self.venue, username))
        if row is None:
            return
        row.update(gate_probe=probe, updated_at=_now())
        if probe != "refused" and row["status"] == "active":
            row["status"] = "suspended"

    def suspend_account(self, username: str) -> bool:
        row = self._accounts.get((self.venue, username))
        if row is None or row["status"] != "active":
            return False
        row.update(status="suspended", updated_at=_now())
        return True

    def revoke_account(self, username: str) -> bool:
        row = self._accounts.get((self.venue, username))
        if row is None:
            return False
        row.update(status="revoked", updated_at=_now())
        return True

    def list_links(self, username: str = "") -> list[dict[str, Any]]:
        rows = [
            r
            for (v, _p, u), r in sorted(
                self._link_rows.items(), key=lambda kv: (kv[0][2], kv[0][1])
            )
            if v == self.venue and (not username or u == username)
        ]
        return [link_view(copy.deepcopy(r)) for r in rows]

    def upsert_link(
        self, username: str, project_id: str, slurm_account: str
    ) -> dict[str, Any]:
        if (self.venue, username) not in self._accounts:
            raise ValueError("a link needs its site account (0135 foreign key)")
        key = (self.venue, project_id, username)
        row = self._link_rows.setdefault(
            key, _blank_link(self.venue, project_id, username)
        )
        row.update(slurm_account=slurm_account, status="active", suspended_reason="")
        return link_view(copy.deepcopy(row))

    def delete_link(self, username: str, project_id: str) -> bool:
        return self._link_rows.pop((self.venue, project_id, username), None) is not None

    def suspend_links(self, username: str, reason: str, project_id: str = "") -> int:
        hits = [
            row
            for (v, p, u), row in self._link_rows.items()
            if (v, u) == (self.venue, username)
            and row["status"] == "active"
            and project_id in ("", p)
        ]
        for row in hits:
            row.update(status="suspended", suspended_reason=reason)
        return len(hits)


def _blank_account(venue: str, username: str) -> dict[str, Any]:
    now = _now()
    return {
        "venue": venue,
        "username": username,
        "site_login": "",
        "method": "gated_key",
        "status": "pending",
        "verified_login": "",
        "verified_assocs": [],
        "verified_at": None,
        "gate_version": "",
        "gate_sha256": "",
        "gate_probe": "unknown",
        "transport_key_id": "",
        "transport_fingerprint": "",
        "signer_fingerprints": [],
        "uv_required": False,
        "created_at": now,
        "updated_at": now,
    }


def _blank_link(venue: str, project_id: str, username: str) -> dict[str, Any]:
    return {
        "venue": venue,
        "project_id": project_id,
        "username": username,
        "slurm_account": "",
        "status": "active",
        "created_at": _now(),
        "suspended_reason": "",
    }


__all__ = [
    "ACCOUNT_COLUMNS",
    "ACCOUNT_STATUSES",
    "ACCOUNT_VIEW",
    "GATE_PROBES",
    "LINK_COLUMNS",
    "LINK_STATUSES",
    "VERIFIED_FIELDS",
    "InMemorySiteAccountStore",
    "SiteAccountStore",
    "account_view",
    "link_view",
]
