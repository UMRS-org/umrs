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

Slice MB-S2c (2026-11-27) adds:

* :func:`verify_object` (the ``OBJECT_VERIFY`` job, ``umrs.jobs.object_verify``,
  and in-process callers): re-read the object inside UMRS and record
  ``verified`` or ``mismatch`` (named reason) on its row; a legacy object
  gets a row on demand. It raises instead of guessing when there is no
  database or the object changed while it was read;
* :func:`output_prefix_rule`: ``runs/`` in an ``-output`` bucket is
  server-only, ``reviews/`` create-only (ADR-115 D-11);
* :func:`reserve_create_only`: the reservation row (``etag`` =
  :data:`RESERVED_ETAG`) a create-only save inserts before its PUT, so two
  concurrent saves to one key cannot both win (D-10). Reads skip it.
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

#: The ``etag`` of a create-only reservation row (ADR-115 D-10/D-11, slice
#: MB-S2c): never a store ETag (those are hex, with an optional ``-<parts>``),
#: so it never describes an object version and every read skips it.
RESERVED_ETAG = "create-only:reserved"

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
    f"AND etag <> '{RESERVED_ETAG}' "
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


def _pick(
    rows: dict[tuple[str, str], list[ObjectProvenance]], ref: ObjectRef
) -> ObjectProvenance:
    """The newest row of *ref*'s object at its etag (any etag when the caller
    has none); a row of another version never describes this one."""
    bucket, key, etag = ref
    for row in rows.get((bucket, key), ()):
        if row.etag != RESERVED_ETAG and etag in ("", row.etag):
            return row
    return legacy_record(bucket, key, etag)


def _by_object(found: list[tuple[Any, ...]]) -> dict[tuple[str, str], list[ObjectProvenance]]:
    """Rows (newest first) grouped by object, so a 1000-key page is matched
    in one pass rather than keys x rows (review MB-S2b, 2026-11-27)."""
    grouped: dict[tuple[str, str], list[ObjectProvenance]] = {}
    for values in found:
        row = _row_from(values)
        grouped.setdefault((row.bucket, row.key), []).append(row)
    return grouped


def lookup_many(conn: Any, refs: list[ObjectRef]) -> dict[tuple[str, str], ObjectProvenance]:
    """:func:`lookup` for a whole page in **one** SELECT (ADR-115 D-4: never
    a query or a ``head_object`` per key), keyed by ``(bucket, key)``."""
    if not refs:
        return {}
    buckets = sorted({bucket for bucket, _key, _etag in refs})
    keys = sorted({key for _bucket, key, _etag in refs})
    with conn.cursor() as cur:
        cur.execute(_SELECT_MANY, (buckets, keys))
        rows = _by_object(cur.fetchall())
    return {(ref[0], ref[1]): _pick(rows, ref) for ref in refs}


def _all_legacy(refs: list[ObjectRef]) -> dict[tuple[str, str], ObjectProvenance]:
    return {(bucket, key): legacy_record(bucket, key, etag) for bucket, key, etag in refs}


def _on_pooled(pool: Any, read: Any, fallback: Any, what: str) -> Any:
    """``read(conn)`` on a pooled connection; *fallback* with no pool or on
    any failure (a read path never fails because provenance is
    unreachable: unverified is the safe reading)."""
    if pool is None:
        return fallback
    try:
        conn = pool.getconn()
    except Exception:  # never fail the caller on provenance
        logger.warning("provenance: no connection for %s", what, exc_info=True)
        return fallback
    try:
        return read(conn)
    except Exception:  # same: unverified is the safe reading
        _rollback(conn)
        logger.warning("provenance: %s failed", what, exc_info=True)
        return fallback
    finally:
        pool.putconn(conn)


def lookup_many_best_effort(
    pool: Any, refs: list[ObjectRef]
) -> dict[tuple[str, str], ObjectProvenance]:
    """:func:`lookup_many` on a pooled connection; every object reads as
    legacy (unverified, the safe answer) with no pool or on any failure, so
    a listing never fails because the provenance table is unreachable."""
    if not refs:
        return {}
    return _on_pooled(
        pool,
        lambda conn: lookup_many(conn, refs),
        _all_legacy(refs),
        f"a lookup of {len(refs)} objects",
    )


_SELECT_EXECUTION = (
    f"SELECT {', '.join(_COLUMNS)} FROM ("
    f"SELECT DISTINCT ON (bucket, key) {', '.join(_COLUMNS)}, created_at "
    "FROM object_provenance "
    "WHERE execution_id = %s AND bucket = ANY(%s) AND source_kind = 'step_output' "
    "ORDER BY bucket, key, created_at DESC"
    ") newest ORDER BY created_at, key"
)


def rows_of_execution(
    conn: Any, execution_id: str, buckets: list[str]
) -> list[ObjectProvenance]:
    """The ``step_output`` rows one run wrote into *buckets* (its own bucket
    pair), the newest version of each object, oldest object first. The
    outputs route reads these when the run's in-memory record is gone (a
    server restart): the rows still say what each step stored (review
    MB-S2b, 2026-11-27)."""
    with conn.cursor() as cur:
        cur.execute(_SELECT_EXECUTION, (execution_id, sorted(set(buckets))))
        return [_row_from(found) for found in cur.fetchall()]


def rows_of_execution_best_effort(
    pool: Any, execution_id: str, buckets: list[str]
) -> list[ObjectProvenance]:
    """:func:`rows_of_execution` on a pooled connection; none on failure."""
    return list(
        _on_pooled(
            pool,
            lambda conn: rows_of_execution(conn, execution_id, buckets),
            [],
            f"the outputs of {execution_id}",
        )
    )


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



# ── server-only and create-only prefixes (ADR-115 D-11, slice MB-S2c) ─────────

#: Run output buckets are ``<input bucket>-output`` (``workflow_runner``).
OUTPUT_BUCKET_SUFFIX = "-output"
#: Written by step collection only (``runs/<run_id>/<step>/``): a client
#: write through the files routes is refused.
SERVER_ONLY_PREFIXES = ("runs/",)
#: Write-once records (review saves): a client write must create the key.
CREATE_ONLY_PREFIXES = ("reviews/",)

SERVER_ONLY = "server_only"
CREATE_ONLY = "create_only"


def output_prefix_rule(bucket: str, key: str) -> str | None:
    """:data:`SERVER_ONLY`, :data:`CREATE_ONLY` or None for a client write of
    *key* into *bucket*. Only ``-output`` buckets carry the rule; the key is
    read as the store will name it (a leading ``/`` is dropped, as the files
    route drops it)."""
    if not str(bucket or "").endswith(OUTPUT_BUCKET_SUFFIX):
        return None
    path = str(key or "").lstrip("/")
    if path.startswith(SERVER_ONLY_PREFIXES):
        return SERVER_ONLY
    if path.startswith(CREATE_ONLY_PREFIXES):
        return CREATE_ONLY
    return None


_RESERVE = (
    "INSERT INTO object_provenance "
    "(bucket, key, etag, size, sha256_state, source_kind, uploader, project_id) "
    "VALUES (%s, %s, %s, 0, 'pending', %s, %s, %s) "
    "ON CONFLICT (bucket, key, etag) DO NOTHING"
)
_RELEASE = "DELETE FROM object_provenance WHERE bucket = %s AND key = %s AND etag = %s"


def reserve_create_only(
    conn: Any,
    bucket: str,
    key: str,
    *,
    source_kind: str = "review_record",
    uploader: str | None = None,
    project_id: str | None = None,
) -> bool:
    """Claim *key* before its PUT (the caller commits): True for the one
    caller that inserted the reservation row, False when a save to that key
    already claimed it. The primary key ``(bucket, key, etag)`` with the
    fixed :data:`RESERVED_ETAG` makes the insert the arbiter, so two
    concurrent saves cannot both win. The row stays after the PUT: the key
    is written once, ever. A crash between the claim and the PUT leaves the
    key claimed (a later save is 409; the reviewer saves under a new
    timestamped key)."""
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"unknown provenance source_kind {source_kind!r}")
    with conn.cursor() as cur:
        cur.execute(
            _RESERVE, (bucket, key, RESERVED_ETAG, source_kind, uploader, project_id)
        )
        return int(getattr(cur, "rowcount", 0) or 0) == 1


def release_create_only(conn: Any, bucket: str, key: str) -> None:
    """Drop the reservation of a save whose PUT failed (the caller commits),
    so the reviewer can retry the same key."""
    with conn.cursor() as cur:
        cur.execute(_RELEASE, (bucket, key, RESERVED_ETAG))


# ── OBJECT_VERIFY: re-read an object inside UMRS (ADR-115 D-5, MB-S2c) ────────

_CHUNK = 1024 * 1024
#: Concurrent re-reads per process: a verify re-reads up to 8 GiB, and a
#: bounded pool keeps the store and the API workers free (plan risk row).
_VERIFY_SLOTS = threading.BoundedSemaphore(2)


class ObjectUnreadable(RuntimeError):
    """The object is gone, has no ETag, or changed while it was re-read: no
    verdict (the job fails and the scheduler may retry it)."""


@dataclass(frozen=True)
class VerifyOutcome:
    """One ``OBJECT_VERIFY`` verdict on the version the store holds now."""

    bucket: str
    key: str
    etag: str
    size: int
    sha256: str
    sha256_state: str
    #: Why the state is ``mismatch`` (None when ``verified``).
    reason: str | None = None
    #: The row's origin (``legacy`` when UMRS held none before this verify).
    source_kind: str = "legacy"
    #: The hash of the bytes this verify read (differs from ``sha256``, the
    #: recorded one, on a ``sha256_differs`` mismatch).
    observed_sha256: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Reread:
    size: int
    sha256: str
    md5: str


def _reread(s3_client: Any, bucket: str, key: str) -> _Reread:
    """Hash every byte of the object as the store serves it now."""
    try:
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"]
    except Exception as exc:
        raise ObjectUnreadable(f"{bucket}/{key} could not be read: {exc}") from exc
    reader = Sha256Reader(body)
    try:
        while reader.read(_CHUNK):
            pass
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    return _Reread(reader.size, reader.sha256.hexdigest(), reader.md5.hexdigest())


def _mismatch_reason(
    got: _Reread, head: dict[str, Any], etag: str, expected: tuple[str | None, str]
) -> str | None:
    """The first way the bytes differ from what they should be, else None.

    *expected* is ``(recorded sha256, declared sha256)``: the hash UMRS
    recorded when it stored this version, and one the caller declared (an
    intake item's). A single-part ETag is the MD5 of the version's bytes, the
    same rule the writers use (``_state_for``), so bytes changed under an
    unchanged ETag are caught even for a legacy object."""
    recorded, declared = expected
    length = head.get("ContentLength")
    checks = (
        (isinstance(length, int) and length != got.size, "size_differs"),
        ("-" not in etag and etag.lower() != got.md5, "etag_not_body_md5"),
        (bool(recorded) and recorded != got.sha256, "sha256_differs"),
        (bool(declared) and declared.lower() != got.sha256, "declared_sha256_differs"),
    )
    return next((reason for failed, reason in checks if failed), None)


_UPSERT_VERIFIED = (
    "INSERT INTO object_provenance "
    "(bucket, key, etag, size, sha256, sha256_state, content_type, source_kind, "
    "source_ref, verified_at) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, "
    "CASE WHEN %s = 'verified' THEN now() END) "
    "ON CONFLICT (bucket, key, etag) DO UPDATE SET "
    "sha256_state = EXCLUDED.sha256_state, "
    "sha256 = COALESCE(object_provenance.sha256, EXCLUDED.sha256), "
    "verified_at = EXCLUDED.verified_at"
)


def _write_verdict(conn: Any, outcome: VerifyOutcome, content_type: Any) -> None:
    """Upsert the verdict on the version's row. A recorded hash is kept (it
    is what the bytes should be); a legacy object gets a row carrying the
    hash UMRS just computed."""
    ref = json.dumps({"verified_by": "object_verify", "reason": outcome.reason})
    with conn.cursor() as cur:
        cur.execute(
            _UPSERT_VERIFIED,
            (
                outcome.bucket,
                outcome.key,
                outcome.etag,
                outcome.size,
                outcome.sha256,
                outcome.sha256_state,
                content_type if isinstance(content_type, str) else None,
                outcome.source_kind,
                ref,
                outcome.sha256_state,
            ),
        )


def _live_version(s3_client: Any, bucket: str, key: str) -> tuple[dict[str, Any], str]:
    head = _head(s3_client, bucket, key)
    etag = _etag_of(head)
    if etag is None:
        raise ObjectUnreadable(f"{bucket}/{key} has no readable version")
    return head, etag


def verify_object(
    s3_client: Any,
    conn: Any,
    bucket: str,
    key: str,
    *,
    declared_sha256: str = "",
) -> VerifyOutcome:
    """Re-read ``bucket/key`` inside UMRS, compare, and record the verdict on
    *conn* (the caller commits).

    ``verified`` when the bytes match everything known about the version
    (its ETag, the hash UMRS recorded, a declared hash); ``mismatch``
    otherwise, with the reason. A legacy object (no row) is attested on
    demand: it gets a row with the computed hash. Raises
    :class:`ObjectUnreadable` when the object is gone or changed while it
    was read (the ETag before and after differ): no verdict then."""
    head, etag = _live_version(s3_client, bucket, key)
    with _VERIFY_SLOTS:
        got = _reread(s3_client, bucket, key)
    if _live_version(s3_client, bucket, key)[1] != etag:
        raise ObjectUnreadable(f"{bucket}/{key} changed while it was verified")
    row = lookup(conn, bucket, key, etag)
    reason = _mismatch_reason(got, head, etag, (row.sha256, declared_sha256))
    outcome = VerifyOutcome(
        bucket=bucket,
        key=key,
        etag=etag,
        size=got.size,
        sha256=row.sha256 or got.sha256,
        sha256_state="mismatch" if reason else "verified",
        reason=reason,
        source_kind=row.source_kind,
        observed_sha256=got.sha256,
    )
    _write_verdict(conn, outcome, head.get("ContentType"))
    return outcome


def verify_object_pooled(
    pool: Any, s3_client: Any, bucket: str, key: str, *, declared_sha256: str = ""
) -> VerifyOutcome:
    """:func:`verify_object` on a pooled connection, committed. Unlike the
    writers' best-effort calls this one raises: a verify with nowhere to
    record its verdict has not happened."""
    if pool is None:
        raise ObjectUnreadable("no provenance database: OBJECT_VERIFY needs one")
    conn = pool.getconn()
    try:
        outcome = verify_object(
            s3_client, conn, bucket, key, declared_sha256=declared_sha256
        )
        conn.commit()
        return outcome
    except Exception:
        _rollback(conn)
        raise
    finally:
        pool.putconn(conn)


__all__ = [
    "CREATE_ONLY",
    "CREATE_ONLY_PREFIXES",
    "DEFAULT_CONTENT_TYPE",
    "OUTPUT_BUCKET_SUFFIX",
    "RESERVED_ETAG",
    "SERVER_ONLY",
    "SERVER_ONLY_PREFIXES",
    "SHA256_STATES",
    "SOURCE_KINDS",
    "HashedObject",
    "ObjectProvenance",
    "ObjectRef",
    "ObjectUnreadable",
    "Sha256Reader",
    "VerifyOutcome",
    "content_type_for",
    "hashed",
    "legacy_record",
    "live_etag",
    "lookup",
    "lookup_many",
    "lookup_many_best_effort",
    "output_prefix_rule",
    "provenance_pool",
    "provenance_row",
    "record",
    "record_best_effort",
    "record_step_outputs",
    "release_create_only",
    "reserve_create_only",
    "rows_of_execution",
    "rows_of_execution_best_effort",
    "set_provenance_pool",
    "verify_object",
    "verify_object_pooled",
]
