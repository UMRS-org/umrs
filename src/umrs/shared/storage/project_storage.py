"""Per-project external storage binding (ADR-077) — foundation layer.

A project may point its object store (an S3 URL) and its Postgres (a libpq
URI) at stores outside the UMRS front-end — typically next to an HPC site's
large filesystem. Unbound projects keep the server's own stores. This module
owns:

* parsing and validation of the two locators — the credentials ride the
  URL's userinfo once, from the client to UMRS, and are never returned;
* the target-host policy: never loopback / link-local / unspecified /
  multicast / reserved; ``UMRS_PROJECT_STORAGE_ALLOWED`` (comma-separated
  CIDRs or host names) narrows targets further and is what lets a project
  owner (not only an admin) bind a store;
* sealing of both locators at rest (AES-256-GCM, AAD = project + field; the
  key is ``UMRS_PROJECT_STORAGE_KEY`` or is derived from ``JWT_SECRET``);
* the ``project_storage_bindings`` store (migration 0114);
* the resolver the data paths call: answers are cached for a short TTL, a
  stale answer survives a failed refresh, and a project that was never
  resolved fails closed — a project's data is never written to the wrong
  store because the registry was briefly unreadable;
* probes that prove a binding works from this server.

Foundation module: never imports from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import base64
import hashlib
import ipaddress
import logging
import os
import re
import secrets
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

from umrs.shared.storage.s3_config import S3Config

logger = logging.getLogger(__name__)

KEY_ENV = "UMRS_PROJECT_STORAGE_KEY"
ALLOWED_ENV = "UMRS_PROJECT_STORAGE_ALLOWED"
TTL_ENV = "UMRS_PROJECT_STORAGE_TTL_SEC"

_HKDF_SALT = b"umrs-project-storage"
_HKDF_INFO = b"umrs-project-storage-v1"
_SEAL_VERSION = "v1"
_NONCE_LEN = 12

_S3_QUERY_KEYS = frozenset({"region", "verify", "backend"})
_S3_BACKENDS = frozenset({"seaweedfs", "minio"})
_PG_SCHEMES = frozenset({"postgresql", "postgres"})
# libpq keywords a client may carry; anything that names a file on this
# server (sslrootcert, sslkey, passfile, service, ...) or re-targets the
# connection (host, hostaddr) is refused.
_PG_QUERY_KEYS = frozenset({"sslmode", "connect_timeout", "target_session_attrs"})
_PG_SSLMODES = frozenset(
    {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
)
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")
_PROBE_PREFIX = ".umrs-probe/"


class StorageBindingError(ValueError):
    """Validation or sealing failure with a stable reason code.

    The message never carries credential material.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class ProjectStorageUnavailable(RuntimeError):
    """The binding registry could not be read for a never-resolved project."""


# -- locators -----------------------------------------------------------------


@dataclass(frozen=True)
class S3Locator:
    """An S3 endpoint with its credentials (from ``http(s)://AK:SK@host:port``)."""

    endpoint_url: str  # scheme://host[:port], no credentials, no path
    host: str
    port: int
    access_key: str
    secret_key: str
    region: str = "us-east-1"
    bucket: str = ""  # optional default bucket (the URL path)
    verify: bool = True  # TLS verification for https endpoints
    backend: str = "seaweedfs"

    @property
    def use_ssl(self) -> bool:
        """True for https endpoints."""
        return self.endpoint_url.startswith("https://")

    def redacted(self) -> dict[str, Any]:
        """Display form: no secret key, access-key id shortened."""
        return {
            "endpoint_url": self.endpoint_url,
            "bucket": self.bucket,
            "region": self.region,
            "backend": self.backend,
            "verify": self.verify,
            "access_key_hint": _hint(self.access_key),
        }


@dataclass(frozen=True)
class PostgresLocator:
    """A libpq connection URI (``postgresql://user:pw@host:port/db``)."""

    uri: str  # full URI including the password
    host: str
    port: int
    dbname: str
    user: str
    sslmode: str = ""

    def redacted(self) -> dict[str, Any]:
        """Display form: everything but the password."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "sslmode": self.sslmode,
        }


@dataclass(frozen=True)
class StorageBinding:
    """A project's bound stores; ``None`` means the server default."""

    project_id: str
    s3: S3Locator | None = None
    postgres: PostgresLocator | None = None
    updated_by: str = ""
    updated_at: str | None = None

    @property
    def bound(self) -> bool:
        """True when at least one store is bound."""
        return self.s3 is not None or self.postgres is not None

    def redacted(self) -> dict[str, Any]:
        """API response shape — never carries credentials."""
        return {
            "project_id": self.project_id,
            "bound": self.bound,
            "s3": self.s3.redacted() if self.s3 else None,
            "postgres": self.postgres.redacted() if self.postgres else None,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at,
        }


def _hint(value: str) -> str:
    return f"{value[:4]}…" if len(value) > 4 else "…"


def _port_of(parts: Any, default: int) -> int:
    try:
        return parts.port or default
    except ValueError:
        raise StorageBindingError("port_invalid") from None


def _bounded(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        raise StorageBindingError("locator_empty")
    if len(raw) > 2048:
        raise StorageBindingError("locator_too_long")
    return raw


def _check_parts(parts: Any, schemes: frozenset[str]) -> None:
    if parts.scheme.lower() not in schemes:
        raise StorageBindingError("scheme_unsupported", parts.scheme or "(none)")
    if "," in parts.netloc or not parts.hostname:
        raise StorageBindingError("host_invalid")
    if parts.fragment:
        raise StorageBindingError("fragment_forbidden")
    if parts.username is None or parts.password is None:
        raise StorageBindingError("credentials_required")


def _split(text: str, *, schemes: frozenset[str]) -> Any:
    parts = urlsplit(_bounded(text))
    _check_parts(parts, schemes)
    return parts


def _s3_query(parts: Any) -> dict[str, str]:
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    unknown = sorted(set(query) - _S3_QUERY_KEYS)
    if unknown:
        raise StorageBindingError("query_key_unsupported", ",".join(unknown))
    backend = (query.get("backend") or "seaweedfs").strip().lower()
    if backend not in _S3_BACKENDS:
        raise StorageBindingError("backend_unsupported", backend)
    query["backend"] = backend
    return query


def _s3_bucket(path: str) -> str:
    bucket = path.strip("/")
    if bucket and ("/" in bucket or not _BUCKET_RE.match(bucket)):
        raise StorageBindingError("bucket_invalid", bucket)
    return bucket


def parse_s3_url(text: str) -> S3Locator:
    """Parse ``http(s)://ACCESS_KEY:SECRET_KEY@host[:port][/bucket][?...]``.

    Query keys: ``region`` (default us-east-1), ``verify`` (``false``
    disables TLS verification for a self-signed https endpoint) and
    ``backend`` (``seaweedfs`` | ``minio``). Credentials are URL-decoded.
    """
    parts = _split(text, schemes=frozenset({"http", "https"}))
    query = _s3_query(parts)
    scheme = parts.scheme.lower()
    host = parts.hostname.lower()
    port = _port_of(parts, 443 if scheme == "https" else 80)
    shown_host = f"[{host}]" if ":" in host else host
    return S3Locator(
        endpoint_url=f"{scheme}://{shown_host}:{port}",
        host=host,
        port=port,
        access_key=unquote(parts.username),
        secret_key=unquote(parts.password),
        region=(query.get("region") or "us-east-1").strip(),
        bucket=_s3_bucket(parts.path),
        verify=(query.get("verify") or "true").strip().lower() != "false",
        backend=query["backend"],
    )


def _pg_query(parts: Any) -> dict[str, str]:
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    unknown = sorted(set(query) - _PG_QUERY_KEYS)
    if unknown:
        raise StorageBindingError("query_key_unsupported", ",".join(unknown))
    sslmode = query.get("sslmode", "")
    if sslmode and sslmode not in _PG_SSLMODES:
        raise StorageBindingError("sslmode_invalid", sslmode)
    return query


def parse_postgres_uri(text: str) -> PostgresLocator:
    """Parse ``postgresql://user:password@host[:port]/dbname[?sslmode=...]``."""
    parts = _split(text, schemes=_PG_SCHEMES)
    query = _pg_query(parts)
    dbname = unquote(parts.path.lstrip("/"))
    if not dbname or "/" in dbname:
        raise StorageBindingError("dbname_required")
    if not parts.username:
        raise StorageBindingError("credentials_required")
    return PostgresLocator(
        uri=text.strip(),
        host=parts.hostname.lower(),
        port=_port_of(parts, 5432),
        dbname=dbname,
        user=unquote(parts.username),
        sslmode=query.get("sslmode", ""),
    )


# -- target-host policy ---------------------------------------------------------


def _forbidden_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(
        ip.is_loopback
        or ip.is_link_local
        or ip.is_unspecified
        or ip.is_multicast
        or ip.is_reserved
    )


def _resolve(host: str) -> set[str]:
    try:
        infos = socket.getaddrinfo(host, None, 0, socket.SOCK_STREAM)
    except socket.gaierror:
        raise StorageBindingError("host_unresolvable", host) from None
    return {info[4][0] for info in infos}


def allowed_targets() -> tuple[str, ...]:
    """Operator allowlist from ``UMRS_PROJECT_STORAGE_ALLOWED`` (may be empty)."""
    raw = os.environ.get(ALLOWED_ENV, "")
    return tuple(item.strip().lower() for item in raw.split(",") if item.strip())


def _matches_allowlist(
    host: str, addresses: set[str], allowed: tuple[str, ...]
) -> bool:
    if host in allowed:
        return True
    networks = []
    for item in allowed:
        try:
            networks.append(ipaddress.ip_network(item, strict=False))
        except ValueError:
            continue
    return bool(addresses) and all(
        any(ipaddress.ip_address(addr) in net for net in networks) for addr in addresses
    )


def check_target_host(host: str, *, allowed: tuple[str, ...] | None = None) -> None:
    """Refuse infrastructure addresses; enforce the allowlist when one is set.

    Every resolved address must pass (a DNS name mixing a private and a
    loopback answer is refused).
    """
    addresses = _resolve(host)
    if not addresses:
        raise StorageBindingError("host_unresolvable", host)
    for addr in addresses:
        if _forbidden_address(ipaddress.ip_address(addr)):
            raise StorageBindingError("address_forbidden", host)
    allowed = allowed_targets() if allowed is None else allowed
    if allowed and not _matches_allowlist(host, addresses, allowed):
        raise StorageBindingError("host_not_allowlisted", host)


def binding_refusal(*, is_admin: bool, is_owner: bool, allowed: tuple[str, ...]) -> str:
    """Who may set a binding: ``""`` when allowed, else the refusal reason.

    Owners may bind only when the operator declared an allowlist (it bounds
    where a casual project can send traffic from this server); admins may
    bind any target that passes :func:`check_target_host`.
    """
    if not (is_admin or is_owner):
        return "forbidden"
    if not is_admin and not allowed:
        return "admin_required"
    return ""


# -- sealing at rest --------------------------------------------------------------


def _key_material() -> bytes:
    explicit = os.environ.get(KEY_ENV, "").strip()
    if explicit:
        return explicit.encode("utf-8")
    jwt_secret = os.environ.get("JWT_SECRET", "").strip()
    if jwt_secret:
        return jwt_secret.encode("utf-8")
    raise StorageBindingError("sealing_key_unavailable", f"set {KEY_ENV} or JWT_SECRET")


def _sealing_key() -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(
        algorithm=hashes.SHA256(), length=32, salt=_HKDF_SALT, info=_HKDF_INFO
    ).derive(_key_material())


def _key_id(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:8]


def _aad(project_id: str, field: str) -> bytes:
    return f"{project_id}\x1f{field}".encode()


def seal(plaintext: str, *, project_id: str, field: str) -> str:
    """Seal *plaintext* bound to (project, field): ``v1:<key id>:<b64>``."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = _sealing_key()
    nonce = secrets.token_bytes(_NONCE_LEN)
    blob = AESGCM(key).encrypt(
        nonce, plaintext.encode("utf-8"), _aad(project_id, field)
    )
    payload = base64.b64encode(nonce + blob).decode("ascii")
    return f"{_SEAL_VERSION}:{_key_id(key)}:{payload}"


def unseal(token: str, *, project_id: str, field: str) -> str:
    """Inverse of :func:`seal`; a changed key or a moved ciphertext fails."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    version, _, rest = (token or "").partition(":")
    key_id, _, payload = rest.partition(":")
    if version != _SEAL_VERSION or not payload:
        raise StorageBindingError("sealed_value_invalid")
    key = _sealing_key()
    if key_id != _key_id(key):
        raise StorageBindingError("sealing_key_changed", "re-set the binding")
    raw = base64.b64decode(payload)
    try:
        plain = AESGCM(key).decrypt(
            raw[:_NONCE_LEN], raw[_NONCE_LEN:], _aad(project_id, field)
        )
    except InvalidTag:
        raise StorageBindingError("sealed_value_invalid") from None
    return plain.decode("utf-8")


# -- store (migration 0114) --------------------------------------------------------

_SELECT = (
    "SELECT project_id, s3_url_sealed, pg_uri_sealed, updated_by, updated_at"
    " FROM project_storage_bindings WHERE project_id = %s"
)


class ProjectStorageStore:
    """psycopg2-shaped store over ``project_storage_bindings``."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def get(self, project_id: str) -> StorageBinding | None:
        """The unsealed binding, or ``None`` when the project has none."""
        cur = self._conn.cursor()
        cur.execute(_SELECT, (project_id,))
        row = cur.fetchone()
        if not row:
            return None
        s3 = (
            parse_s3_url(unseal(row[1], project_id=row[0], field="s3"))
            if row[1]
            else None
        )
        pg = (
            parse_postgres_uri(unseal(row[2], project_id=row[0], field="postgres"))
            if row[2]
            else None
        )
        updated = row[4].isoformat() if hasattr(row[4], "isoformat") else row[4]
        return StorageBinding(
            row[0], s3=s3, postgres=pg, updated_by=row[3], updated_at=updated
        )

    def _sealed_row(self, project_id: str) -> tuple[str, str]:
        """The stored sealed values ``(s3, postgres)``; empty when unbound."""
        cur = self._conn.cursor()
        cur.execute(_SELECT, (project_id,))
        row = cur.fetchone()
        return (row[1] or "", row[2] or "") if row else ("", "")

    def put(
        self,
        project_id: str,
        *,
        s3_url: str | None,
        postgres_uri: str | None,
        updated_by: str,
    ) -> StorageBinding | None:
        """Upsert. ``None`` keeps a store's current value, ``""`` unbinds it.

        A binding that ends with neither store is deleted. Kept values are
        carried over sealed, as stored.
        """
        kept_s3, kept_pg = self._sealed_row(project_id)
        s3_sealed = _next_sealed(project_id, "s3", s3_url, kept_s3)
        pg_sealed = _next_sealed(project_id, "postgres", postgres_uri, kept_pg)
        if not s3_sealed and not pg_sealed:
            self.clear(project_id)
            return None
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO project_storage_bindings
                   (project_id, s3_url_sealed, pg_uri_sealed, updated_by, updated_at)
               VALUES (%s, %s, %s, %s, NOW())
               ON CONFLICT (project_id) DO UPDATE SET
                   s3_url_sealed = EXCLUDED.s3_url_sealed,
                   pg_uri_sealed = EXCLUDED.pg_uri_sealed,
                   updated_by = EXCLUDED.updated_by,
                   updated_at = NOW()""",
            (project_id, s3_sealed, pg_sealed, updated_by),
        )
        self._conn.commit()
        return self.get(project_id)

    def clear(self, project_id: str) -> bool:
        """Delete the binding; True when a row existed."""
        cur = self._conn.cursor()
        cur.execute(
            "DELETE FROM project_storage_bindings WHERE project_id = %s", (project_id,)
        )
        existed = bool(getattr(cur, "rowcount", 0))
        self._conn.commit()
        return existed


def _next_sealed(project_id: str, field: str, value: str | None, kept: str) -> str:
    """``None`` keeps *kept*; blank unbinds; anything else is sealed anew."""
    if value is None:
        return kept
    if not value.strip():
        return ""
    return seal(value.strip(), project_id=project_id, field=field)


# -- resolver --------------------------------------------------------------------


def _is_undefined_table(exc: BaseException) -> bool:
    return getattr(exc, "pgcode", None) == "42P01"


def _default_pool() -> Any:
    from umrs.shared.pg import get_default_sync_pool, pg_configured_from_env

    return get_default_sync_pool() if pg_configured_from_env() else None


def _end_read(conn: Any) -> None:
    try:
        conn.rollback()  # never leave a borrowed connection idle in transaction
    except Exception:  # the pool discards broken connections
        logger.debug("rollback after binding read failed", exc_info=True)


def _borrowed(pool: Any, read: Callable[[Any], Any], *, absent: Any) -> Any:
    """Run *read* on a borrowed conn; *absent* when there is no registry.

    No pool (no Postgres configured) and a missing table (migration 0114 not
    applied yet) both mean "no bindings exist" — never an error.
    """
    if pool is None:
        return absent
    conn = pool.getconn()
    try:
        return read(conn)
    except Exception as exc:
        if _is_undefined_table(exc):
            return absent
        raise
    finally:
        _end_read(conn)
        pool.putconn(conn)


def _read_bound_ids(conn: Any) -> frozenset[str]:
    cur = conn.cursor()
    cur.execute("SELECT project_id FROM project_storage_bindings")
    return frozenset(str(row[0]) for row in cur.fetchall())


class ProjectStorageResolver:
    """TTL-cached project → binding lookups for the data paths.

    One cheap query per TTL loads the SET of bound project ids, so an
    unbound project (the common case) costs no per-project read and is
    unaffected by a single bound row that fails to unseal. A failed refresh
    serves the last good answer; only a process that never read the
    registry fails closed (:class:`ProjectStorageUnavailable`).
    """

    def __init__(
        self,
        *,
        pool_getter: Callable[[], Any] = _default_pool,
        ttl_sec: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pool_getter = pool_getter
        self._ttl = float(os.environ.get(TTL_ENV, "30")) if ttl_sec is None else ttl_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._bound_ids: tuple[float, frozenset[str]] | None = None
        self._cache: dict[str, tuple[float, StorageBinding | None]] = {}

    def _fresh(self, stamp: float) -> bool:
        return self._clock() - stamp < self._ttl

    def bound_ids(self, *, pool: Any = None) -> frozenset[str]:
        """Ids of projects that carry a binding (TTL-cached)."""
        with self._lock:
            cached = self._bound_ids
        if cached is not None and self._fresh(cached[0]):
            return cached[1]
        try:
            ids = _borrowed(
                pool if pool is not None else self._pool_getter(),
                _read_bound_ids,
                absent=frozenset(),
            )
        except Exception as exc:  # noqa: BLE001 — classified below
            return self._stale_or_raise("(all projects)", cached, exc)
        with self._lock:
            self._bound_ids = (self._clock(), ids)
        return ids

    def binding(self, project_id: str, *, pool: Any = None) -> StorageBinding | None:
        """The project's binding (``None`` = server default stores)."""
        pid = (project_id or "").strip()
        if not pid or pid not in self.bound_ids(pool=pool):
            return None
        with self._lock:
            cached = self._cache.get(pid)
        if cached is not None and self._fresh(cached[0]):
            return cached[1]
        try:
            fresh = _borrowed(
                pool if pool is not None else self._pool_getter(),
                lambda conn: ProjectStorageStore(conn).get(pid),
                absent=None,
            )
        except Exception as exc:  # noqa: BLE001 — classified below
            return self._stale_or_raise(pid, cached, exc)
        with self._lock:
            self._cache[pid] = (self._clock(), fresh)
        return fresh

    def _stale_or_raise(self, what: str, cached: Any, exc: Exception) -> Any:
        if cached is not None:
            logger.warning(
                "binding refresh failed for %s; serving the cached answer", what
            )
            return cached[1]
        raise ProjectStorageUnavailable(
            f"storage binding registry unreadable for {what} ({type(exc).__name__})"
        ) from exc

    def s3_locator(self, project_id: str, *, pool: Any = None) -> S3Locator | None:
        """The project's bound object store, or ``None``."""
        binding = self.binding(project_id, pool=pool)
        return binding.s3 if binding else None

    def invalidate(self, project_id: str | None = None) -> None:
        """Drop cached answers after a set/clear (the id set always goes)."""
        with self._lock:
            self._bound_ids = None
            if project_id is None:
                self._cache.clear()
            else:
                self._cache.pop(project_id, None)


_resolver: ProjectStorageResolver | None = None
_resolver_lock = threading.Lock()


def get_project_storage_resolver() -> ProjectStorageResolver:
    """Process-wide resolver (lazily built)."""
    global _resolver
    with _resolver_lock:
        if _resolver is None:
            _resolver = ProjectStorageResolver()
        return _resolver


def set_project_storage_resolver(resolver: ProjectStorageResolver | None) -> None:
    """Install a resolver (tests) or reset to lazy construction (None)."""
    global _resolver
    with _resolver_lock:
        _resolver = resolver


# -- clients -------------------------------------------------------------------------


def s3_client_for(locator: S3Locator, *, probe: bool = False) -> Any:
    """A path-style, SigV4 boto3 client for a bound store.

    ``probe`` clients fail fast (3 s connect, one attempt): a probe runs
    inside a request the gateway times out after 30 s, and an unreachable
    store must come back as a report, not as a gateway timeout.
    """
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=locator.endpoint_url,
        aws_access_key_id=locator.access_key,
        aws_secret_access_key=locator.secret_key,
        region_name=locator.region,
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=3 if probe else 10,
            read_timeout=10 if probe else 120,
            retries={"max_attempts": 1 if probe else 3, "mode": "standard"},
        ),
        use_ssl=locator.use_ssl,
        verify=locator.verify if locator.use_ssl else False,
    )


class ProjectS3Config(S3Config):
    """``S3Config`` for a bound project: same TTL policy, the bound store.

    ``presign_endpoint`` pins presigned URLs to the bound endpoint — the
    server-wide ``S3_PRESIGN_ENDPOINT`` names the default store only.
    """

    def __init__(
        self, locator: S3Locator, *, default_bucket: str | None = None
    ) -> None:
        super().__init__()
        self.locator = locator
        self.endpoint_url = locator.endpoint_url
        self.access_key = locator.access_key
        self.secret_key = locator.secret_key
        self.region = locator.region
        self.bucket = locator.bucket or default_bucket or self.bucket
        self.use_ssl = locator.use_ssl
        self.backend = locator.backend
        self.iam_endpoint_url = None
        self.presign_endpoint = locator.endpoint_url

    def get_client(self) -> Any:
        """boto3 client for the bound store."""
        return s3_client_for(self.locator)


def project_s3_config(
    project_id: str, *, pool: Any = None, resolver: ProjectStorageResolver | None = None
) -> ProjectS3Config | None:
    """The bound ``S3Config`` for *project_id*, or ``None`` (server default).

    Raises:
        ProjectStorageUnavailable: the registry is unreadable and the
            project was never resolved by this process.
    """
    locator = (resolver or get_project_storage_resolver()).s3_locator(
        project_id, pool=pool
    )
    if locator is None:
        return None
    return ProjectS3Config(locator)


# -- probes ----------------------------------------------------------------------------


def _probe_buckets(client: Any, bucket: str) -> list[str]:
    from botocore.exceptions import ClientError

    created = []
    for name in (bucket, f"{bucket}-output"):
        try:
            client.head_bucket(Bucket=name)
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            client.create_bucket(Bucket=name)
            created.append(name)
    return created


def probe_s3(
    locator: S3Locator, *, bucket: str, ensure_buckets: bool = True
) -> dict[str, Any]:
    """Prove the store answers from here: buckets, then a put/get/delete.

    With *ensure_buckets* the project bucket and its ``-output`` twin are
    created when absent (the workflow output convention).
    """
    started = time.monotonic()
    fast = s3_client_for(locator, probe=True)
    if ensure_buckets:
        created = _probe_buckets(fast, bucket)
    else:
        fast.head_bucket(Bucket=bucket)
        created = []
    # The store answered: the round-trip uses the retrying client — SeaweedFS
    # answers the first write into a fresh bucket with a transient 500 while
    # it grows the collection's volumes (2026-10-01).
    client = s3_client_for(locator)
    key = f"{_PROBE_PREFIX}{secrets.token_hex(8)}"
    body = secrets.token_bytes(64)
    client.put_object(Bucket=bucket, Key=key, Body=body)
    echoed = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    client.delete_object(Bucket=bucket, Key=key)
    if echoed != body:
        raise StorageBindingError("s3_roundtrip_mismatch")
    return {
        "ok": True,
        "endpoint_url": locator.endpoint_url,
        "bucket": bucket,
        "buckets_created": created,
        "roundtrip_ms": round((time.monotonic() - started) * 1000, 1),
    }


def _pg_connect(locator: PostgresLocator) -> Any:
    import psycopg2

    return psycopg2.connect(
        locator.uri, connect_timeout=5, application_name="umrs-binding-probe"
    )


def _pg_facts(cur: Any) -> dict[str, Any]:
    cur.execute(
        "SELECT current_user, current_database(), current_setting('server_version')"
    )
    user, dbname, version = cur.fetchone()
    cur.execute("SELECT ssl FROM pg_stat_ssl WHERE pid = pg_backend_pid()")
    ssl_row = cur.fetchone()
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    vector_row = cur.fetchone()
    return {
        "user": user,
        "dbname": dbname,
        "server_version": version,
        "tls": bool(ssl_row and ssl_row[0]),
        "pgvector": vector_row[0] if vector_row else None,
    }


def _pg_vector_roundtrip(cur: Any) -> bool:
    cur.execute("CREATE TEMP TABLE umrs_probe (id int, v vector(3))")
    cur.execute("INSERT INTO umrs_probe VALUES (1, '[1,2,3]')")
    cur.execute("SELECT v <-> '[1,2,4]' FROM umrs_probe WHERE id = 1")
    return cur.fetchone()[0] == 1.0


def probe_postgres(locator: PostgresLocator) -> dict[str, Any]:
    """Connect, report server facts, and round-trip a vector when pgvector is there.

    Only a session-scoped TEMP table is written; the transaction is rolled back.
    """
    started = time.monotonic()
    conn = _pg_connect(locator)
    try:
        cur = conn.cursor()
        facts = _pg_facts(cur)
        if facts["pgvector"]:
            facts["vector_roundtrip"] = _pg_vector_roundtrip(cur)
        conn.rollback()
    finally:
        conn.close()
    facts.update(
        ok=True,
        host=locator.host,
        port=locator.port,
        latency_ms=round((time.monotonic() - started) * 1000, 1),
    )
    return facts


def probe_binding(binding: StorageBinding, *, default_bucket: str) -> dict[str, Any]:
    """Probe every bound store; failures are reported per store, never raised."""
    report: dict[str, Any] = {
        "project_id": binding.project_id,
        "s3": None,
        "postgres": None,
    }
    if binding.s3 is not None:
        bucket = binding.s3.bucket or default_bucket
        report["s3"] = _guarded(lambda: probe_s3(binding.s3, bucket=bucket))
    if binding.postgres is not None:
        report["postgres"] = _guarded(lambda: probe_postgres(binding.postgres))
    stores = [r for r in (report["s3"], report["postgres"]) if r is not None]
    report["ok"] = bool(stores) and all(r.get("ok") for r in stores)
    return report


def _guarded(probe: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return probe()
    except Exception as exc:  # noqa: BLE001 — reported per store
        return {"ok": False, "error": type(exc).__name__, "detail": _safe_detail(exc)}


def _safe_detail(exc: Exception) -> str:
    """First line of the error, with anything URI-shaped masked."""
    text = str(exc).splitlines()[0] if str(exc) else ""
    return re.sub(r"[a-z][a-z0-9+.-]*://[^\s]+", "<uri>", text)[:300]


__all__ = [
    "ALLOWED_ENV",
    "KEY_ENV",
    "PostgresLocator",
    "ProjectS3Config",
    "ProjectStorageResolver",
    "ProjectStorageStore",
    "ProjectStorageUnavailable",
    "S3Locator",
    "StorageBinding",
    "StorageBindingError",
    "allowed_targets",
    "binding_refusal",
    "check_target_host",
    "get_project_storage_resolver",
    "parse_postgres_uri",
    "parse_s3_url",
    "probe_binding",
    "probe_postgres",
    "probe_s3",
    "project_s3_config",
    "s3_client_for",
    "seal",
    "set_project_storage_resolver",
    "unseal",
]
