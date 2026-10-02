"""Self-service personal buckets (ADR-079).

A verified user creates ``<owner>.<project>.<alias>`` (and its ``-output``
pair) on the server's own object store through UMRS; the ``user_buckets``
row (migration 0115) records the owner. Two consumers ask :func:`owner_of`:

* the HPC lane's staging owner check (``umrs.server.step_staging``), next to
  the provisioned buckets of the user configs (``BucketsCfg``);
* the files API, which lets a caller upload / list / download / delete only
  in buckets they own.

No credential is stored or returned: the server performs every S3 operation
(constitution §2 — storage is written only through UMRS).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: One DNS-label-ish segment of a bucket name (S3 rules: lowercase letters,
#: digits, hyphens; no leading/trailing hyphen).
_SEGMENT_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,40}[a-z0-9])?")
_OUTPUT_SUFFIX = "-output"
#: S3 bucket names are 3..63 characters; ``-output`` must still fit.
_MAX_BUCKET_LEN = 63 - len(_OUTPUT_SUFFIX)
_TTL_SEC = 30.0


class UserBucketError(ValueError):
    """A bucket request that can never succeed as given (name, owner)."""


@dataclass(frozen=True)
class UserBucket:
    bucket: str
    owner_id: str
    project: str
    alias: str

    @property
    def output_bucket(self) -> str:
        return self.bucket + _OUTPUT_SUFFIX

    def to_dict(self) -> dict[str, str]:
        return {
            "bucket": self.bucket,
            "output_bucket": self.output_bucket,
            "owner": self.owner_id,
            "project": self.project,
            "alias": self.alias,
        }


def _segment(value: str) -> str:
    """``make_path_to_bucket_string`` rules: lowercase, spaces/_ → '-'."""
    return str(value or "").strip().lower().replace(" ", "-").replace("_", "-")


def bucket_name(owner: str, project: str, alias: str) -> str:
    """``<owner>.<project>.<alias>``, refused unless every part is a valid
    bucket segment — the same shape the staging owner check expects."""
    parts = [_segment(owner), _segment(project), _segment(alias)]
    for label, part in zip(("owner", "project", "alias"), parts, strict=True):
        if not _SEGMENT_RE.fullmatch(part):
            raise UserBucketError(
                f"{label} {part!r} cannot be part of a bucket name "
                "(lowercase letters, digits and inner hyphens only)"
            )
    name = ".".join(parts)
    if len(name) > _MAX_BUCKET_LEN:
        raise UserBucketError(
            f"bucket name {name!r} is longer than {_MAX_BUCKET_LEN} characters"
        )
    return name


def input_bucket_of(bucket: str) -> str:
    """The input bucket a name refers to (an ``-output`` pair → its input)."""
    return bucket.removesuffix(_OUTPUT_SUFFIX)


# -- store -------------------------------------------------------------------


def _default_pool() -> Any:
    from umrs.shared.pg import get_default_sync_pool, pg_configured_from_env

    return get_default_sync_pool() if pg_configured_from_env() else None


def _borrowed(pool: Any, work: Callable[[Any], Any], *, absent: Any) -> Any:
    """Run *work* on a borrowed connection; *absent* without a registry (no
    Postgres, or migration 0115 not applied yet). The connection is always
    returned, never idle in transaction."""
    if pool is None:
        return absent
    conn = pool.getconn()
    try:
        return work(conn)
    except Exception as exc:
        if getattr(exc, "pgcode", None) == "42P01":  # undefined_table
            return absent
        raise
    finally:
        try:
            conn.rollback()
        except Exception:
            logger.debug("rollback after user-bucket access failed", exc_info=True)
        pool.putconn(conn)


class UserBucketStore:
    """``user_buckets`` rows plus a short owner cache for the hot paths."""

    def __init__(self, pool_factory: Callable[[], Any] = _default_pool) -> None:
        self._pool_factory = pool_factory
        self._owners: dict[str, tuple[float, str | None]] = {}
        self._lock = threading.Lock()

    def _pool(self) -> Any:
        return self._pool_factory()

    def add(self, entry: UserBucket) -> bool:
        """Record *entry*; False when the bucket already has a row."""

        def work(conn: Any) -> bool:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO user_buckets (bucket, owner_id, project, alias) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT (bucket) DO NOTHING",
                (entry.bucket, entry.owner_id, entry.project, entry.alias),
            )
            conn.commit()
            return cur.rowcount == 1

        pool = self._pool()
        if pool is None:
            raise UserBucketError("personal buckets need the UMRS Postgres store")
        added = bool(_borrowed(pool, work, absent=False))
        with self._lock:
            self._owners.pop(entry.bucket, None)
        return added

    def get(self, bucket: str) -> UserBucket | None:
        def work(conn: Any) -> UserBucket | None:
            cur = conn.cursor()
            cur.execute(
                "SELECT bucket, owner_id, project, alias FROM user_buckets WHERE bucket = %s",
                (bucket,),
            )
            row = cur.fetchone()
            return UserBucket(*row) if row else None

        return _borrowed(self._pool(), work, absent=None)

    def list_for(self, owner: str) -> list[UserBucket]:
        def work(conn: Any) -> list[UserBucket]:
            cur = conn.cursor()
            cur.execute(
                "SELECT bucket, owner_id, project, alias FROM user_buckets "
                "WHERE owner_id = %s ORDER BY bucket",
                (owner,),
            )
            return [UserBucket(*row) for row in cur.fetchall()]

        return _borrowed(self._pool(), work, absent=[])

    def owner_of(self, bucket: str) -> str | None:
        """Owner of *bucket* (or of the input bucket of an ``-output`` name);
        None when no personal-bucket row exists. Cached for a few seconds."""
        name = input_bucket_of(bucket)
        now = time.monotonic()
        with self._lock:
            hit = self._owners.get(name)
            if hit is not None and now - hit[0] < _TTL_SEC:
                return hit[1]
        entry = self.get(name)
        owner = entry.owner_id if entry is not None else None
        with self._lock:
            self._owners[name] = (now, owner)
        return owner


_store: UserBucketStore | None = None
_store_lock = threading.Lock()


def get_user_bucket_store() -> UserBucketStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = UserBucketStore()
        return _store


def set_user_bucket_store(store: UserBucketStore | None) -> None:
    """Tests / boot code replace the process store."""
    global _store
    with _store_lock:
        _store = store


def owner_of(bucket: str) -> str | None:
    """Process-wide :meth:`UserBucketStore.owner_of`; None on any failure
    (a lookup error never grants ownership)."""
    try:
        return get_user_bucket_store().owner_of(bucket)
    except Exception:
        logger.warning("personal-bucket owner lookup failed for %s", bucket, exc_info=True)
        return None
