"""Host identity without a platform CA (ADR-098 D-5, D-6; slice SH-H1b).

An agent host holds one ed25519 private key; the server holds only the
public key it enrolled (``umrs.hosts.enroll``). The host authenticates each
request with a **short-lived host token** it signs itself::

    umrs-host.v1.<b64url(payload json)>.<b64url(ed25519 signature)>

    payload = {"aud": "umrs-host", "venue": <site>, "host": <host id>,
               "iat": <unix s>, "exp": <unix s>, "nonce": <hex>}

The signature covers ``umrs-host.v1.<payload b64>``. A token lives at most
``MAX_TOKEN_SECONDS``; an expired one is refused even when its signature is
good (a replay), and so is one issued in the future beyond ``CLOCK_SKEW_S``.
The venue in the token must be this site's, so a token minted for one site
never passes at another that shares the database.

Leases are bound to a host (D-6): ``secret_leases.host_id``, '' meaning the
inline host. :func:`lease_host_refusal` is the one rule the worker rail and
the inference proxy both apply, after the host token and before trusting
the lease:

    lease host   presented host   verdict
    inline       none             ok (inline sandbox; the IP check applies)
    inline       H                rail.host_mismatch (403)
    agent A      none             host_token_required (401)
    agent A      A                ok (no IP check: pasta defeats it)
    agent A      B                rail.host_mismatch (403)

Refusals are named; no token, key or signature is ever logged.

Dual backend like ``HostRegistry``: in memory without a pool (tests, dev),
Postgres (migration ``<n>_host_identity.sql``) with one. Every query is
scoped to this store's venue.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from umrs.hosts.runtime import INLINE_HOST_ID

#: Header an agent host (or its sidecar) presents the host token in.
HOST_TOKEN_HEADER = "x-umrs-host-token"
TOKEN_PREFIX = "umrs-host.v1."
TOKEN_AUDIENCE = "umrs-host"
#: Host tokens are short-lived: an agent signs a fresh one every few minutes.
MAX_TOKEN_SECONDS = 300
CLOCK_SKEW_S = 30
#: Typed code of a lease presented from another host (ADR-098 D-6).
HOST_MISMATCH = "rail.host_mismatch"
#: Enrollment refusal: only a ``--rekey`` token may replace an enrolled key.
ALREADY_ENROLLED = "enroll.already_enrolled"
#: 401 reasons; :data:`HOST_MISMATCH` is the only 403.
HOST_AUTH_REASONS = (
    "host_token_invalid",
    "host_token_expired",
    "host_unenrolled",
    "host_token_required",
)


class HostAuthError(Exception):
    """A refused host token; ``reason`` is one of :data:`HOST_AUTH_REASONS`."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    @property
    def status(self) -> int:
        return 403 if self.reason == HOST_MISMATCH else 401


class EnrollmentError(ValueError):
    """A refused enrollment; ``code`` is the typed reason (``enroll.*``)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _now() -> datetime:
    return datetime.now(timezone.utc)


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def unb64url(text: str) -> bytes:
    """Strict base64url (no padding needed); ValueError on anything else."""
    pad = "=" * (-len(text) % 4)
    try:
        return base64.urlsafe_b64decode((text + pad).encode("ascii"))
    except (binascii.Error, UnicodeEncodeError) as exc:
        raise ValueError("not base64url") from exc


def key_id(public_key: bytes) -> str:
    """Short public fingerprint shown to admins (``sha256:`` + 16 hex)."""
    return "sha256:" + hashlib.sha256(public_key).hexdigest()[:16]


# ── key store ─────────────────────────────────────────────────────────────


@dataclass
class HostKey:
    host_id: str
    venue: str
    public_key: bytes
    key_id: str
    enrolled_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        enrolled = self.enrolled_at.isoformat() if self.enrolled_at else None
        return {
            "venue": self.venue,
            "host_id": self.host_id,
            "key_id": self.key_id,
            "enrolled_at": enrolled,
        }


@dataclass
class _Enrollment:
    host_id: str
    expires_at: datetime
    used_at: datetime | None = None
    #: Issued with ``--rekey``: may replace an enrolled key (else refused).
    rekey: bool = False


@dataclass
class _MemoryIdentity:
    enrollments: dict[tuple[str, str], _Enrollment] = field(default_factory=dict)
    keys: dict[tuple[str, str], HostKey] = field(default_factory=dict)

    def add_enrollment(
        self,
        venue: str,
        token_hash: str,
        host_id: str,
        ttl_s: int,
        _by: str,
        rekey: bool = False,
    ) -> datetime:
        expires = _now() + timedelta(seconds=ttl_s)
        row = _Enrollment(host_id, expires, rekey=rekey)
        self.enrollments[(venue, token_hash)] = row
        return expires

    def consume(self, venue: str, token_hash: str, key: HostKey) -> str:
        row = self.enrollments.get((venue, token_hash))
        reason = _enrollment_refusal(row, _now())
        if reason:
            return reason
        assert row is not None
        if not row.rekey and (venue, row.host_id) in self.keys:
            return ALREADY_ENROLLED
        row.used_at = _now()
        key.host_id, key.enrolled_at = row.host_id, _now()
        self.keys[(venue, row.host_id)] = key
        return ""

    def key(self, venue: str, host_id: str) -> HostKey | None:
        return self.keys.get((venue, host_id))

    def keys_of(self, venue: str) -> list[HostKey]:
        return [k for (v, _), k in sorted(self.keys.items()) if v == venue]

    def delete(self, venue: str, host_id: str) -> bool:
        pending = [
            k
            for k, e in self.enrollments.items()
            if k[0] == venue and e.host_id == host_id and e.used_at is None
        ]
        for item in pending:
            del self.enrollments[item]
        return self.keys.pop((venue, host_id), None) is not None or bool(pending)


def _enrollment_refusal(row: Any, now: datetime) -> str:
    """Named reason an enrollment row may not be redeemed ("" = ok)."""
    if row is None:
        return "enroll.token_invalid"
    if row.used_at is not None:
        return "enroll.token_used"
    if row.expires_at <= now:
        return "enroll.token_expired"
    return ""


_CONSUME_SQL = (
    "UPDATE host_enrollments SET used_at = now() WHERE venue = %s"
    " AND token_hash = %s AND used_at IS NULL AND expires_at > now()"
    " RETURNING host_id, rekey"
)
_PROBE_SQL = (
    "SELECT host_id, expires_at, used_at FROM host_enrollments"
    " WHERE venue = %s AND token_hash = %s"
)
#: A plain (not ``--rekey``) token only adds a key: an enrolled host is kept.
_KEY_INSERT_SQL = (
    "INSERT INTO host_keys (venue, host_id, public_key, key_id)"
    " VALUES (%s, %s, %s, %s) ON CONFLICT (venue, host_id) DO NOTHING"
    " RETURNING host_id"
)
_KEY_UPSERT_SQL = (
    "INSERT INTO host_keys (venue, host_id, public_key, key_id)"
    " VALUES (%s, %s, %s, %s) ON CONFLICT (venue, host_id) DO UPDATE SET"
    " public_key = EXCLUDED.public_key, key_id = EXCLUDED.key_id,"
    " enrolled_at = now()"
)
_KEY_COLUMNS = "host_id, venue, public_key, key_id, enrolled_at"
_DELETE_PENDING_SQL = (
    "DELETE FROM host_enrollments WHERE venue = %s AND host_id = %s"
    " AND used_at IS NULL RETURNING host_id"
)
_DELETE_KEY_SQL = (
    "DELETE FROM host_keys WHERE venue = %s AND host_id = %s RETURNING host_id"
)


def _key_row(row: Any) -> HostKey:
    host_id, venue, public_key, kid, enrolled_at = row
    return HostKey(
        str(host_id), str(venue), unb64url(str(public_key)), str(kid), enrolled_at
    )


class _PgIdentity:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def add_enrollment(
        self,
        venue: str,
        token_hash: str,
        host_id: str,
        ttl_s: int,
        by: str,
        rekey: bool = False,
    ) -> datetime:
        sql = (
            "INSERT INTO host_enrollments (token_hash, venue, host_id, created_by,"
            " expires_at, rekey) VALUES (%s, %s, %s, %s,"
            " now() + make_interval(secs => %s), %s) RETURNING expires_at"
        )
        params = (token_hash, venue, host_id, by, float(ttl_s), bool(rekey))
        rows = self._run([(sql, params)])
        return rows[0][0]

    def consume(self, venue: str, token_hash: str, key: HostKey) -> str:
        conn = self._pool.getconn()
        try:
            reason = self._consume_in(conn, venue, token_hash, key)
            conn.commit() if not reason else conn.rollback()
            return reason
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    @staticmethod
    def _consume_in(conn: Any, venue: str, token_hash: str, key: HostKey) -> str:
        """One transaction: the token is used and the key stored, or neither."""
        with conn.cursor() as cur:
            cur.execute(_CONSUME_SQL, (venue, token_hash))
            row = cur.fetchone()
            if row is None:
                cur.execute(_PROBE_SQL, (venue, token_hash))
                probe = cur.fetchone()
                found = _Enrollment(*probe) if probe else None
                return _enrollment_refusal(found, _now()) or "enroll.token_invalid"
            key.host_id = str(row[0])
            params = (venue, key.host_id, b64url(key.public_key), key.key_id)
            cur.execute(_KEY_UPSERT_SQL if row[1] else _KEY_INSERT_SQL, params)
            if not row[1] and cur.fetchone() is None:
                return ALREADY_ENROLLED  # rolled back: the token stays unused
        return ""

    def key(self, venue: str, host_id: str) -> HostKey | None:
        sql = f"SELECT {_KEY_COLUMNS} FROM host_keys WHERE venue = %s AND host_id = %s"
        rows = self._run([(sql, (venue, host_id))])
        return _key_row(rows[0]) if rows else None

    def keys_of(self, venue: str) -> list[HostKey]:
        sql = f"SELECT {_KEY_COLUMNS} FROM host_keys WHERE venue = %s ORDER BY host_id"
        return [_key_row(r) for r in self._run([(sql, (venue,))])]

    def delete(self, venue: str, host_id: str) -> bool:
        params = (venue, host_id)
        statements = [(_DELETE_PENDING_SQL, params), (_DELETE_KEY_SQL, params)]
        return bool(self._run(statements))

    def _run(self, statements: list[tuple[str, tuple]]) -> list[Any]:
        """Statements in one transaction; the rows of all of them."""
        conn = self._pool.getconn()
        try:
            rows: list[Any] = []
            with conn.cursor() as cur:
                for sql, params in statements:
                    cur.execute(sql, params)
                    rows.extend(cur.fetchall() if cur.description else [])
            conn.commit()
            return rows
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)


class HostKeyStore:
    """This site's enrollment tokens and enrolled host public keys."""

    def __init__(self, pool: Any | None = None, *, venue: str | None = None) -> None:
        self._backend: Any = (
            _PgIdentity(pool) if pool is not None else _MemoryIdentity()
        )
        self._venue = venue

    @property
    def venue(self) -> str:
        if self._venue is None:
            from umrs.jobs.store import configured_site_name

            self._venue = configured_site_name()
        return self._venue

    def add_enrollment(
        self,
        token_hash: str,
        host_id: str,
        ttl_s: int,
        created_by: str = "",
        *,
        rekey: bool = False,
    ) -> datetime:
        return self._backend.add_enrollment(
            self.venue, token_hash, host_id, ttl_s, created_by, rekey
        )

    def consume_enrollment(self, token_hash: str, public_key: bytes) -> HostKey:
        """Use the token once and store ``public_key`` for its host, atomically.

        Raises :class:`EnrollmentError` (``enroll.token_invalid``,
        ``enroll.token_used``, ``enroll.token_expired``, or
        ``enroll.already_enrolled`` when a token issued without ``rekey``
        meets a host that enrolled meanwhile; that token stays unused)."""
        key = HostKey("", self.venue, public_key, key_id(public_key))
        reason = self._backend.consume(self.venue, token_hash, key)
        if reason:
            raise EnrollmentError(reason, "enrollment token refused")
        return key

    def public_key(self, host_id: str) -> HostKey | None:
        return self._backend.key(self.venue, host_id)

    def keys(self) -> list[HostKey]:
        return self._backend.keys_of(self.venue)

    def unenroll(self, host_id: str) -> bool:
        """Forget the host's key and its unused enrollment tokens."""
        return self._backend.delete(self.venue, host_id)


# ── host tokens ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HostClaims:
    host_id: str
    venue: str
    issued_at: int
    expires_at: int


def sign_host_token(
    private_key: Any,
    *,
    venue: str,
    host_id: str,
    ttl_s: int = MAX_TOKEN_SECONDS,
    now: float | None = None,
) -> str:
    """Host side: a fresh token signed with the host's ed25519 private key."""
    issued = int(time.time() if now is None else now)
    payload = {
        "aud": TOKEN_AUDIENCE,
        "venue": venue,
        "host": host_id,
        "iat": issued,
        "exp": issued + min(int(ttl_s), MAX_TOKEN_SECONDS),
        "nonce": secrets.token_hex(8),
    }
    body = b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signature = private_key.sign((TOKEN_PREFIX + body).encode("ascii"))
    return f"{TOKEN_PREFIX}{body}.{b64url(signature)}"


def _split(token: str) -> tuple[str, bytes, bytes]:
    """(signed text, payload bytes, signature) or ValueError."""
    if not token.startswith(TOKEN_PREFIX):
        raise ValueError("prefix")
    body, _, sig = token[len(TOKEN_PREFIX) :].partition(".")
    if not body or not sig or "." in sig:
        raise ValueError("shape")
    return TOKEN_PREFIX + body, unb64url(body), unb64url(sig)


def _claims(payload: bytes, venue: str) -> HostClaims:
    data = json.loads(payload)
    if not isinstance(data, dict) or data.get("aud") != TOKEN_AUDIENCE:
        raise ValueError("audience")
    claims = HostClaims(
        str(data["host"]), str(data["venue"]), int(data["iat"]), int(data["exp"])
    )
    if not claims.host_id or claims.venue != venue:
        raise ValueError("venue")
    if claims.expires_at - claims.issued_at > MAX_TOKEN_SECONDS:
        raise ValueError("lifetime")
    return claims


def _parse(token: str, venue: str) -> tuple[str, bytes, HostClaims]:
    try:
        signed, payload, signature = _split(token)
        return signed, signature, _claims(payload, venue)
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError) as exc:
        # Claims are read before the signature, so this is unauthenticated
        # input: inf (OverflowError) and deep nesting (RecursionError) too.
        raise HostAuthError("host_token_invalid") from exc


def _check_signature(key: HostKey, signed: str, signature: bytes) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(key.public_key).verify(
            signature, signed.encode("ascii")
        )
    except (InvalidSignature, ValueError) as exc:
        raise HostAuthError("host_token_invalid") from exc


def _check_window(claims: HostClaims, now: float) -> None:
    if claims.issued_at > now + CLOCK_SKEW_S:
        raise HostAuthError("host_token_invalid")
    if claims.expires_at <= now:
        raise HostAuthError("host_token_expired")


def verify_host_token(
    token: str, store: HostKeyStore, *, now: float | None = None
) -> HostClaims:
    """Server side: the claims of a valid token, else :class:`HostAuthError`.

    Order: shape and venue → enrolled key → signature → time window, so an
    expired token is reported expired only when it was genuinely signed."""
    signed, signature, claims = _parse(token, store.venue)
    key = store.public_key(claims.host_id)
    if key is None:
        raise HostAuthError("host_unenrolled")
    _check_signature(key, signed, signature)
    _check_window(claims, time.time() if now is None else now)
    return claims


def presented_host(
    headers: Any, store: HostKeyStore | None, *, now: float | None = None
) -> str:
    """The verified host of a request ('' when no host token was sent).

    A token sent to a server without a key store is refused, never ignored."""
    token = headers.get(HOST_TOKEN_HEADER) if headers is not None else None
    if not token:
        return ""
    if store is None:
        raise HostAuthError("host_unenrolled")
    return verify_host_token(str(token), store, now=now).host_id


def host_keys_from_state(state: Any) -> HostKeyStore | None:
    """The site's enrolled host keys, cached on an app state (tests inject
    ``host_key_store``); None without a database, where any presented host
    token is refused ``host_unenrolled``."""
    cached = getattr(state, "host_key_store", None)
    if cached is not None:
        return cached
    pool = getattr(state, "umrs_db_pool", None)
    if pool is None:
        return None
    store = HostKeyStore(pool)
    state.host_key_store = store
    return store


def lease_host(lease: Any) -> str:
    """The host a lease is bound to; '' and ``local`` are the inline host."""
    bound = str(getattr(lease, "host_id", "") or "")
    return bound or INLINE_HOST_ID


def lease_host_refusal(lease: Any, presented: str) -> str:
    """'' when the lease may be used from ``presented`` (see module table)."""
    bound = lease_host(lease)
    if not presented:
        return "" if bound == INLINE_HOST_ID else "host_token_required"
    return "" if presented == bound else HOST_MISMATCH


__all__ = [
    "ALREADY_ENROLLED",
    "CLOCK_SKEW_S",
    "HOST_AUTH_REASONS",
    "HOST_MISMATCH",
    "HOST_TOKEN_HEADER",
    "MAX_TOKEN_SECONDS",
    "EnrollmentError",
    "HostAuthError",
    "HostClaims",
    "HostKey",
    "HostKeyStore",
    "host_keys_from_state",
    "key_id",
    "lease_host",
    "lease_host_refusal",
    "presented_host",
    "sign_host_token",
    "verify_host_token",
]
