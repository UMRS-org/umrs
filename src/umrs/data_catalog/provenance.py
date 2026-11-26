"""Object provenance: server-computed SHA-256 and origin per object version
(ADR-115 D-2, slice MB-S2a).

One Postgres row per ``(bucket, key, etag)`` in ``object_provenance``
(migration ``0142_object_provenance.sql``). The writers are UMRS code
paths only, and they hash the bytes as they pass through the server:

* the files upload route (``umrs.api.files``): ``source_kind=local_upload``;
* step-lane output collection (``umrs.server.step_staging.collect_out``):
  ``source_kind=step_output`` with the execution id and step.

A row is written ``verified`` only when the server itself hashed every byte
it stored and the stored version's ETag is the MD5 of those bytes. An
object without a row is **legacy**: :func:`lookup` returns a
``source_kind="legacy"`` record with no hash, which ``OBJECT_VERIFY``
(slice MB-S2c) hashes on demand.

Recording is best-effort by design: a provenance write never fails the
upload or the step that produced the object (the object is then simply
legacy and verifiable later). The same holds with no database at all, so a
server without Postgres keeps working.

The ETag is the store's, read back with one ``head_object`` after the
upload. It must belong to the bytes the server hashed, or a write racing
on the same key (a second upload, an ADR-080 grantee writing to the
owner's bucket) would get this server's hash recorded as ``verified``:

* a single-part ETag must equal the MD5 the server computed beside the
  SHA-256: then the row is ``verified``;
* a multipart ETag (``<md5>-<parts>``) is not a body MD5, so it cannot be
  tied to the bytes; with a matching ``ContentLength`` the row is written
  ``pending`` (``OBJECT_VERIFY`` re-hashes it), otherwise no row;
* any other ETag, or none: no row, the object stays legacy.

(2026-11-24, review of MB-S2a: the first cut marked any ETag the HEAD
returned ``verified``.)
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_KINDS = frozenset(
    {"local_upload", "nextcloud", "step_output", "review_record", "legacy"}
)
SHA256_STATES = frozenset({"pending", "verified", "mismatch"})
DEFAULT_CONTENT_TYPE = "application/octet-stream"

_COLUMNS = (
    "bucket",
    "key",
    "etag",
    "size",
    "sha256",
    "sha256_state",
    "content_type",
    "source_kind",
    "source_ref",
    "uploader",
    "project_id",
    "execution_id",
    "step",
)


@dataclass(frozen=True)
class ObjectProvenance:
    """One object version's provenance (the columns this slice writes)."""

    bucket: str
    key: str
    etag: str
    size: int
    sha256: str | None
    sha256_state: str
    content_type: str | None
    source_kind: str
    source_ref: dict[str, Any] = field(default_factory=dict)
    uploader: str | None = None
    project_id: str | None = None
    execution_id: str | None = None
    step: str | None = None

    @property
    def legacy(self) -> bool:
        return self.source_kind == "legacy"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def legacy_record(bucket: str, key: str, etag: str = "") -> ObjectProvenance:
    """An object UMRS holds no provenance for: origin unknown, never hashed
    by the server, ``pending`` until ``OBJECT_VERIFY`` hashes it."""
    return ObjectProvenance(
        bucket=bucket,
        key=key,
        etag=etag,
        size=0,
        sha256=None,
        sha256_state="pending",
        content_type=None,
        source_kind="legacy",
    )


class Sha256Reader:
    """File-like view that SHA-256-hashes (and MD5-hashes, for the quota
    ledger's dedup key) and counts every byte the store's uploader reads.

    Non-seekable on purpose: boto3 then streams it and never re-reads a
    part the hash has already counted."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5(usedforsecurity=False)
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._raw.read(size)
        self.sha256.update(chunk)
        self.md5.update(chunk)
        self.size += len(chunk)
        return chunk

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False


def content_type_for(key: str) -> str:
    """ContentType from the key's extension (``video/mp4``, ``application/json``)."""
    guessed, _encoding = mimetypes.guess_type(key, strict=False)
    return guessed or DEFAULT_CONTENT_TYPE


def _head(s3_client: Any, bucket: str, key: str) -> dict[str, Any]:
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
    except Exception:  # no ETag means no row, never a failure
        logger.debug("provenance: head %s/%s failed", bucket, key, exc_info=True)
        return {}
    return head if isinstance(head, dict) else {}


def _etag_of(head: dict[str, Any]) -> str | None:
    etag = head.get("ETag")
    if not isinstance(etag, str):
        return None
    return etag.strip('"') or None


def live_etag(s3_client: Any, bucket: str, key: str) -> str | None:
    """The stored object's ETag without quotes; None when unreadable."""
    return _etag_of(_head(s3_client, bucket, key))


# ── the store ───────────────────────────────────────────────────────────────

_pool_lock = threading.Lock()
_pool: Any = None


def set_provenance_pool(pool: Any) -> None:
    """Install (server boot) or forget (tests, ``None``) the process pool the
    step lane records through. The files route passes its app pool itself."""
    global _pool
    with _pool_lock:
        _pool = pool


def provenance_pool() -> Any:
    with _pool_lock:
        return _pool


_INSERT = (
    f"INSERT INTO object_provenance ({', '.join(_COLUMNS)}, verified_at) "
    f"VALUES ({', '.join(['%s'] * len(_COLUMNS))}, "
    "CASE WHEN %s = 'verified' THEN now() END) "
    "ON CONFLICT (bucket, key, etag) DO NOTHING"
)


def _row_values(row: ObjectProvenance) -> tuple[Any, ...]:
    values = [getattr(row, column) for column in _COLUMNS]
    values[_COLUMNS.index("source_ref")] = json.dumps(row.source_ref, default=str)
    return (*values, row.sha256_state)


def _check(row: ObjectProvenance) -> None:
    if row.source_kind not in SOURCE_KINDS:
        raise ValueError(f"unknown provenance source_kind {row.source_kind!r}")
    if row.sha256_state not in SHA256_STATES:
        raise ValueError(f"unknown sha256_state {row.sha256_state!r}")


def record(conn: Any, row: ObjectProvenance) -> None:
    """Insert *row* on *conn* (the caller commits). The first provenance of
    an object version wins: a re-write of the same ``(bucket, key, etag)``
    keeps the row that is already there."""
    _check(row)
    with conn.cursor() as cur:
        cur.execute(_INSERT, _row_values(row))


def record_best_effort(pool: Any, row: ObjectProvenance) -> bool:
    """:func:`record` on a pooled connection, committed; False (logged) on
    any failure or when *pool* is None. Never raises."""
    if pool is None:
        return False
    try:
        conn = pool.getconn()
    except Exception:  # provenance never fails the write it describes
        logger.warning(
            "provenance: no connection for %s/%s", row.bucket, row.key, exc_info=True
        )
        return False
    try:
        record(conn, row)
        conn.commit()
        return True
    except Exception:  # same: the object stays legacy, verifiable later
        _rollback(conn)
        logger.warning("provenance: %s/%s not recorded", row.bucket, row.key, exc_info=True)
        return False
    finally:
        pool.putconn(conn)


def _rollback(conn: Any) -> None:
    try:
        conn.rollback()
    except Exception:  # the warning above already names the row
        logger.debug("provenance: rollback failed", exc_info=True)


_SELECT = (
    f"SELECT {', '.join(_COLUMNS)} FROM object_provenance "
    "WHERE bucket = %s AND key = %s AND (%s = '' OR etag = %s) "
    "ORDER BY created_at DESC LIMIT 1"
)


def lookup(conn: Any, bucket: str, key: str, etag: str = "") -> ObjectProvenance:
    """The provenance of ``bucket/key`` (at *etag*, else its newest row);
    :func:`legacy_record` when UMRS holds none."""
    with conn.cursor() as cur:
        cur.execute(_SELECT, (bucket, key, etag, etag))
        found = cur.fetchone()
    if found is None:
        return legacy_record(bucket, key, etag)
    return _row_from(found)


_SELECT_MANY = (
    f"SELECT {', '.join(_COLUMNS)} FROM object_provenance "
    "WHERE bucket = ANY(%s) AND key = ANY(%s) ORDER BY created_at DESC"
)

#: One object a page or a run names: ``(bucket, key, etag)``; ``etag`` is
#: the store's current one ("" when the caller never read it).
ObjectRef = tuple[str, str, str]


def _row_from(found: tuple[Any, ...]) -> ObjectProvenance:
    values = dict(zip(_COLUMNS, found, strict=True))
    ref = values["source_ref"]
    values["source_ref"] = json.loads(ref) if isinstance(ref, str) else dict(ref or {})
    return ObjectProvenance(**values)


def _pick(rows: list[ObjectProvenance], ref: ObjectRef) -> ObjectProvenance:
    """The newest row of *ref*'s object at its etag (any etag when the caller
    has none); a row of another version never describes this one."""
    bucket, key, etag = ref
    for row in rows:
        if row.bucket == bucket and row.key == key and etag in ("", row.etag):
            return row
    return legacy_record(bucket, key, etag)


def lookup_many(conn: Any, refs: list[ObjectRef]) -> dict[tuple[str, str], ObjectProvenance]:
    """:func:`lookup` for a whole page in **one** SELECT (ADR-115 D-4: never
    a query or a ``head_object`` per key), keyed by ``(bucket, key)``."""
    if not refs:
        return {}
    buckets = sorted({bucket for bucket, _key, _etag in refs})
    keys = sorted({key for _bucket, key, _etag in refs})
    with conn.cursor() as cur:
        cur.execute(_SELECT_MANY, (buckets, keys))
        rows = [_row_from(found) for found in cur.fetchall()]
    return {(ref[0], ref[1]): _pick(rows, ref) for ref in refs}


def _all_legacy(refs: list[ObjectRef]) -> dict[tuple[str, str], ObjectProvenance]:
    return {(bucket, key): legacy_record(bucket, key, etag) for bucket, key, etag in refs}


def lookup_many_best_effort(
    pool: Any, refs: list[ObjectRef]
) -> dict[tuple[str, str], ObjectProvenance]:
    """:func:`lookup_many` on a pooled connection; every object reads as
    legacy (unverified, the safe answer) with no pool or on any failure, so
    a listing never fails because the provenance table is unreachable."""
    if pool is None or not refs:
        return _all_legacy(refs)
    try:
        conn = pool.getconn()
    except Exception:  # a listing never fails on provenance
        logger.warning(
            "provenance: no connection for a %d-object lookup", len(refs), exc_info=True
        )
        return _all_legacy(refs)
    try:
        return lookup_many(conn, refs)
    except Exception:  # same: unverified is the safe reading
        _rollback(conn)
        logger.warning("provenance: lookup of %d objects failed", len(refs), exc_info=True)
        return _all_legacy(refs)
    finally:
        pool.putconn(conn)


# ── the writers' one call each ──────────────────────────────────────────────


@dataclass(frozen=True)
class HashedObject:
    """What a writer knows after streaming one object through a reader."""

    bucket: str
    key: str
    size: int
    sha256: str
    content_type: str
    #: MD5 of the same bytes: ties a single-part ETag to what was hashed.
    md5: str = ""


def hashed(bucket: str, key: str, reader: Sha256Reader, content_type: str) -> HashedObject:
    return HashedObject(
        bucket,
        key,
        reader.size,
        reader.sha256.hexdigest(),
        content_type,
        reader.md5.hexdigest(),
    )


def _state_for(obj: HashedObject, etag: str, length: Any) -> str | None:
    """``verified`` when the stored version is provably the hashed bytes,
    ``pending`` when it may be (multipart, same size), else None."""
    if "-" in etag:
        return "pending" if length == obj.size else None
    if obj.md5 and etag.lower() == obj.md5:
        return "verified"
    return None


def provenance_row(
    s3_client: Any, obj: HashedObject, source_kind: str, **origin: Any
) -> ObjectProvenance | None:
    """The row for an object the server just hashed and stored (``origin``:
    uploader, project_id, execution_id, step, source_ref). None when the
    store returns no ETag, or one that is not this server's bytes (a write
    raced on the key): the object then stays legacy."""
    head = _head(s3_client, obj.bucket, obj.key)
    etag = _etag_of(head)
    state = None if etag is None else _state_for(obj, etag, head.get("ContentLength"))
    if etag is None or state is None:
        return None
    return ObjectProvenance(
        bucket=obj.bucket,
        key=obj.key,
        etag=etag,
        size=obj.size,
        sha256=obj.sha256,
        sha256_state=state,
        content_type=obj.content_type,
        source_kind=source_kind,
        **origin,
    )


def record_step_outputs(
    s3_client: Any, outputs: list[HashedObject], *, execution_id: str, step: str
) -> int:
    """Provenance for a step's collected outputs through the process pool;
    the number of rows written (0 with no pool installed)."""
    pool = provenance_pool()
    if pool is None or not outputs:
        return 0
    written = 0
    for obj in outputs:
        row = provenance_row(
            s3_client, obj, "step_output", execution_id=execution_id, step=step
        )
        written += bool(row is not None and record_best_effort(pool, row))
    return written


__all__ = [
    "DEFAULT_CONTENT_TYPE",
    "SHA256_STATES",
    "SOURCE_KINDS",
    "HashedObject",
    "ObjectProvenance",
    "ObjectRef",
    "Sha256Reader",
    "content_type_for",
    "hashed",
    "legacy_record",
    "live_etag",
    "lookup",
    "lookup_many",
    "lookup_many_best_effort",
    "provenance_pool",
    "provenance_row",
    "record",
    "record_best_effort",
    "record_step_outputs",
    "set_provenance_pool",
]
