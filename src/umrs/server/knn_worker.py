"""KNN: cosine nearest neighbours of one object in its bucket's embedding table.

Port of upstream 1798ac05 / e04b3d4e / b7d5821a / bd0e6bc1 / 8912c150 /
ef8b7037 / a30136cb (plan U1 A6). The query object is read from S3; its
embedding is reused from the table when any row of the same bucket (and
model) carries the object's ETag, else computed with the bucket's model.
Rows with that ETag (the query itself, byte-identical copies) are never
returned as neighbours. Scope rules:

* the caller must own the bucket (``bucket_owned_by``, the rule
  PROCESS_OBJECTS, EXPORT_EMBEDDINGS and WAIT_OUTPUT share); another user's
  bucket reads like a missing one (checked by ``KnnHandler`` before launch);
* every query only sees rows of that bucket (``s3_object`` starts with
  ``<bucket>/``, or ``<bucket>/<subdir>/`` with a sub-directory) and, where
  the table records it, the bucket's current model: a table several buckets
  share never answers with another bucket's objects;
* ``subdir`` is a normalised relative path: ``..``, absolute paths and
  backslashes are refused;
* ``k`` must be >= 1 and is capped at ``UMRS_KNN_MAX_K`` (default 100).

All SQL is parameterised; the table name goes through ``sql.Identifier``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass

from psycopg2 import sql

from umrs.core.TaskCmdStatus import TaskStatus

logger = logging.getLogger(__name__)

DEFAULT_MAX_K = 100


class KnnError(Exception):
    """A KNN request that cannot run; the message goes back to the caller."""


class KnnRequestError(KnnError, ValueError):
    """A malformed KNN request (refused before any thread is launched)."""


@dataclass(frozen=True)
class KnnRequest:
    """One validated KNN request."""

    n: int
    s3_object: str
    bucket: str
    subdir: str | None = None

    def scope_prefix(self) -> str:
        """The ``s3_object`` prefix every result must carry. The DB stores
        ``<bucket>/<key>`` with no ``s3://`` scheme (b7d5821a)."""
        if self.subdir:
            return f"{self.bucket}/{self.subdir}/"
        return f"{self.bucket}/"


@dataclass
class KnnOutcome:
    """What one search found."""

    results: list
    cached: bool
    n_indexed: int


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def max_k() -> int:
    """``UMRS_KNN_MAX_K`` (default 100); an invalid value keeps the default."""
    raw = os.getenv("UMRS_KNN_MAX_K", "").strip()
    if not raw:
        return DEFAULT_MAX_K
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning("UMRS_KNN_MAX_K=%r is not a positive integer; using %d",
                       raw, DEFAULT_MAX_K)
        return DEFAULT_MAX_K
    return value


def _parse_k(value) -> int:
    """``k`` >= 1, capped at :func:`max_k`."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise KnnRequestError("n must be a positive integer")
    try:
        k = int(value)
    except ValueError:
        raise KnnRequestError("n must be a positive integer") from None
    if k <= 0:
        raise KnnRequestError(f"n must be >= 1 (got {k})")
    limit = max_k()
    if k > limit:
        logger.info("KNN: n=%d capped at UMRS_KNN_MAX_K=%d", k, limit)
    return min(k, limit)


def _path_parts(raw, what: str) -> list[str]:
    """Segments of a relative S3 path; refuses absolute paths and ``..``."""
    if not isinstance(raw, str) or not raw.strip():
        raise KnnRequestError(f"{what} must be a non-empty string")
    if raw.startswith("/") or "\\" in raw or "\x00" in raw:
        raise KnnRequestError(f"{what} must be a relative path: {raw!r}")
    parts = raw.split("/")
    if ".." in parts:
        raise KnnRequestError(f"{what} must not contain '..': {raw!r}")
    return parts


def normalise_subdir(raw) -> str | None:
    """``subdir`` as a clean ``a/b`` prefix (no empty or ``.`` segments)."""
    if raw is None or raw == "":
        return None
    parts = [p for p in _path_parts(raw, "subdir") if p not in ("", ".")]
    return "/".join(parts) or None


def _object_key(raw) -> str:
    """The query object's key, unchanged once it is a safe relative key."""
    _path_parts(raw, "object")
    return raw


def _bucket_name(raw) -> str:
    if not isinstance(raw, str) or not raw or "/" in raw:
        raise KnnRequestError("bucket must be a bucket name")
    return raw


def parse_knn_request(args_str) -> KnnRequest:
    """``{"n", "object", "bucket"[, "subdir"]}`` -> :class:`KnnRequest`."""
    try:
        args = json.loads(args_str)
    except (TypeError, ValueError) as exc:
        raise KnnRequestError(f"bad params: {exc}") from None
    if not isinstance(args, dict):
        raise KnnRequestError("bad params: expected a JSON object")
    return KnnRequest(
        n=_parse_k(args.get("n")),
        s3_object=_object_key(args.get("object")),
        bucket=_bucket_name(args.get("bucket")),
        subdir=normalise_subdir(args.get("subdir")),
    )


def searchable_entry(buckets_cfg, bucket: str, user: str):
    """The bucket's server entry when *user* may search it, else :class:`KnnError`."""
    from umrs.server.indexed_prefilter import plain_embedding_bucket
    from umrs.server.step_staging import bucket_owned_by

    entry = buckets_cfg.get(bucket)
    if entry is None or not bucket_owned_by(bucket, user):
        raise KnnError(f"No bucket entry found for path: {bucket}")
    if not plain_embedding_bucket(entry):
        raise KnnError(f"Bucket {bucket} has no embedding table to search")
    return entry


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class _Scope:
    """Table, row prefix and model filter one request may read."""

    table: str
    prefix: str
    bucket_prefix: str
    model: str | None
    has_model: bool

    @classmethod
    def of(cls, entry, request: KnnRequest) -> _Scope:
        return cls(
            table=entry.table_name,
            prefix=request.scope_prefix(),
            bucket_prefix=f"{request.bucket}/",
            model=getattr(entry.model, "model_name", None),
            has_model=bool(entry.data_base.has_column(entry.table_name, "model")),
        )

    def model_filter(self) -> tuple[sql.Composable, list]:
        if not self.has_model:
            return sql.SQL(""), []
        return sql.SQL(" AND model IS NOT DISTINCT FROM %s"), [self.model]


def _vector_literal(embedding) -> str:
    """pgvector text form of a stored (``str``) or computed embedding."""
    if isinstance(embedding, str):
        return embedding
    values = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def _cached_embedding(cur, scope: _Scope, etag: str):
    """A stored vector with the query's ETag anywhere in the caller's bucket
    (a30136cb: the copy may live at another key; never another bucket)."""
    model_sql, model_params = scope.model_filter()
    query = sql.SQL(
        "SELECT embedding::text FROM {} WHERE checksum = %s"
        " AND starts_with(s3_object, %s) AND embedding IS NOT NULL{} LIMIT 1"
    ).format(sql.Identifier(scope.table), model_sql)
    cur.execute(query, [etag, scope.bucket_prefix, *model_params])
    row = cur.fetchone()
    return row[0] if row else None


def _query_vector(cur, scope, etag, obj, model) -> tuple[str, bool]:
    """``(vector literal, cached)`` for the query object."""
    cached = _cached_embedding(cur, scope, etag)
    if cached is not None:
        return cached, True
    embeddings = model.compute_embeddings([obj])
    return _vector_literal(embeddings[0]), False


def _nearest(cur, scope: _Scope, vector: str, etag: str, k: int) -> list[dict]:
    """The *k* closest rows in scope, minus exact copies of the query
    (same ETag, a30136cb)."""
    model_sql, model_params = scope.model_filter()
    query = sql.SQL(
        "SELECT s3_object, 1.0 - (embedding <=> %s::vector) AS score FROM {}"
        " WHERE starts_with(s3_object, %s) AND checksum IS DISTINCT FROM %s"
        " AND embedding IS NOT NULL{}"
        " ORDER BY embedding <=> %s::vector LIMIT %s"
    ).format(sql.Identifier(scope.table), model_sql)
    cur.execute(query, [vector, scope.prefix, etag, *model_params, vector, k])
    return [{"object": row[0], "score": float(row[1])} for row in cur.fetchall()]


def _count_indexed(cur, scope: _Scope) -> int:
    """Distinct indexed objects in the searched scope (ef8b7037)."""
    model_sql, model_params = scope.model_filter()
    query = sql.SQL(
        "SELECT COUNT(DISTINCT s3_object) FROM {}"
        " WHERE starts_with(s3_object, %s) AND embedding IS NOT NULL{}"
    ).format(sql.Identifier(scope.table), model_sql)
    cur.execute(query, [scope.prefix, *model_params])
    return int(cur.fetchone()[0])


def _load_query(entry, request: KnnRequest):
    """``(etag, object)`` of the query object in the caller's bucket."""
    from umrs.shared.storage.s3 import load_data_from_s3

    try:
        objects, _meta, checksums, _sizes, _zero = load_data_from_s3(
            entry.s3_connection, request.bucket, [request.s3_object],
            entry.modality_name,
        )
    except Exception as exc:
        raise KnnError(f"S3 load failed for {request.s3_object}: {exc}") from exc
    return checksums[0], objects[0]


def search(entry, request: KnnRequest) -> KnnOutcome:
    """Run one KNN search against *entry* (ownership already checked)."""
    etag, obj = _load_query(entry, request)
    scope = _Scope.of(entry, request)
    data_base = entry.data_base
    conn = data_base.get_conn()
    try:
        with conn.cursor() as cur:
            vector, cached = _query_vector(cur, scope, etag, obj, entry.model)
            results = _nearest(cur, scope, vector, etag, request.n)
            n_indexed = _count_indexed(cur, scope)
    finally:
        with contextlib.suppress(Exception):
            conn.rollback()  # a read must not leave the conn idle in transaction
        data_base.put_conn(conn)
    return KnnOutcome(results=results, cached=cached, n_indexed=n_indexed)


def _ok_reply(request: KnnRequest, outcome: KnnOutcome, duration_s: float) -> dict:
    from umrs.shared.pretty_print import pretty_print_knn

    return {
        "status": "ok",
        "n": request.n,
        "query": request.s3_object,
        "bucket": request.bucket,
        "subdir": request.subdir,
        "results": outcome.results,
        "duration_s": duration_s,
        "cached": outcome.cached,
        "n_indexed": outcome.n_indexed,
        "summary": pretty_print_knn(
            request.s3_object, request.bucket, request.n, outcome.results,
            duration_s, outcome.cached, outcome.n_indexed,
        ),
    }


def run_knn(entry, request: KnnRequest) -> tuple[dict, str]:
    """``(reply, final task status)``; never raises."""
    t0 = time.perf_counter()
    try:
        outcome = search(entry, request)
    except KnnError as exc:
        logger.warning("KNN refused for %s/%s: %s", request.bucket, request.s3_object, exc)
        return {"status": "error", "message": str(exc)}, TaskStatus.ERROR
    except Exception as exc:
        logger.exception("KNN search failed for %s/%s", request.bucket, request.s3_object)
        return {"status": "error", "message": f"KNN search failed: {exc}"}, TaskStatus.ERROR
    duration_s = round(time.perf_counter() - t0, 3)
    return _ok_reply(request, outcome, duration_s), TaskStatus.COMPLETED


def knn_search_worker(
    active_threads,
    user,
    request: KnnRequest,
    entry,
    control_mq,
    zmq_sender_id,
    key,
    update_status_callback=None,
):
    """Pool worker for the KNN command: search, reply, bookkeeping."""
    if update_status_callback:
        update_status_callback(TaskStatus.RUNNING)
    reply, status = run_knn(entry, request)
    with contextlib.suppress(RuntimeError):
        control_mq.send_string(json.dumps(reply, ensure_ascii=False), zmq_sender_id)
    active_threads.update_status_to_done()
    active_threads.record_status_for_users_by_key([user], key)
    active_threads.remove_from_running_list()
    if update_status_callback:
        update_status_callback(status)
