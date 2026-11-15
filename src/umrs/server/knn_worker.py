"""KNN: cosine nearest neighbours of one object in its bucket's embedding table.

Port of upstream 1798ac05 / e04b3d4e / b7d5821a / bd0e6bc1 / 8912c150 /
ef8b7037 / a30136cb (plan U1 A6). The query object is read from S3; its
embedding is reused from the table when any row of the same bucket (and
model) carries the object's ETag, else computed with the bucket's model.
Rows with that ETag (the query itself, byte-identical copies) are never
returned as neighbours. Scope rules:

* the caller must own the bucket or be admitted by its ADR-080 access list
  (``step_staging.accessible_bucket_entry``, the rule EXPORT_EMBEDDINGS
  shares); another user's bucket reads like a missing one (checked by
  ``KnnHandler`` before launch);
* every query only sees rows of that bucket (``s3_object`` starts with
  ``<bucket>/``, or ``<bucket>/<subdir>/`` with a sub-directory) and, where
  the table records it, the bucket's current model: a table several buckets
  share never answers with another bucket's objects;
* ``subdir`` is a normalised relative path: ``..``, absolute paths and
  backslashes are refused;
* ``k`` must be >= 1 and is capped at ``UMRS_KNN_MAX_K`` (default 100);
* ``query_name`` (optional) is only what the reply shows as the query (the
  CLI uploads under a temporary key); it is validated like ``object``.

All SQL is parameterised; the table name goes through ``sql.Identifier``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from dataclasses import dataclass

from psycopg2 import errors as pg_errors
from psycopg2 import sql

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.shared.env_knobs import env_int

logger = logging.getLogger(__name__)

DEFAULT_MAX_K = 100
DEFAULT_EXACT_TIMEOUT_MS = 10_000
HNSW_DEFAULT_EF_SEARCH = 40  # pgvector's default hnsw.ef_search
HNSW_MAX_EF_SEARCH = 1000  # pgvector's upper bound for hnsw.ef_search

# ``umrs knn`` uploads its query image under ``[<subdir>/]<QUERY_UPLOAD_DIR>/``
# with a unique name and deletes it after the reply; such keys are never
# searched or counted, should one ever be indexed.
QUERY_UPLOAD_DIR = ".umrs-knn"
_QUERY_UPLOAD_SEGMENT = f"/{QUERY_UPLOAD_DIR}/"


class KnnError(Exception):
    """A KNN request that cannot run; the message goes back to the caller,
    so it never carries backend exception text."""

    code = "KNN_REFUSED"


class KnnRequestError(KnnError, ValueError):
    """A malformed KNN request (refused before any thread is launched)."""

    code = "KNN_BAD_REQUEST"


class KnnStorageError(KnnError):
    """The query object could not be read from the bucket."""

    code = "KNN_S3_UNAVAILABLE"


class KnnTimeoutError(KnnError):
    """The search hit UMRS_KNN_EXACT_TIMEOUT_MS before any answer."""

    code = "KNN_TIMEOUT"


@dataclass(frozen=True)
class KnnRequest:
    """One validated KNN request."""

    n: int
    s3_object: str
    bucket: str
    subdir: str | None = None
    query_name: str | None = None

    def display_query(self) -> str:
        """The query as the caller named it (else its S3 key)."""
        return self.query_name or self.s3_object

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
    n_indexed: int | None  # None when the bounded count timed out


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------


def max_k() -> int:
    """``UMRS_KNN_MAX_K`` (default 100, >= 1); an invalid value keeps the
    default with a warning (``env_knobs`` rule)."""
    return env_int("UMRS_KNN_MAX_K", DEFAULT_MAX_K, lo=1)


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


def _object_key(raw, what: str = "object") -> str:
    """The query object's key, unchanged once it is a safe relative key."""
    _path_parts(raw, what)
    return raw


def _query_name(raw) -> str | None:
    """Optional display name of the query, checked like ``object``."""
    if raw is None:
        return None
    return _object_key(raw, "query_name")


def _bucket_name(raw) -> str:
    if not isinstance(raw, str) or not raw or "/" in raw:
        raise KnnRequestError("bucket must be a bucket name")
    return raw


def parse_knn_request(args_str) -> KnnRequest:
    """``{"n", "object", "bucket"[, "subdir", "query_name"]}`` -> :class:`KnnRequest`."""
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
        query_name=_query_name(args.get("query_name")),
    )


def searchable_entry(buckets_cfg, bucket: str, user: str):
    """The bucket's server entry when *user* may search it, else :class:`KnnError`.

    Access is the export rule (own bucket or ADR-080 access list), shared
    through ``step_staging.accessible_bucket_entry``; KNN adds only the
    plain-embedding-table check.
    """
    from umrs.server import step_staging
    from umrs.server.indexed_prefilter import plain_embedding_bucket

    entry = step_staging.accessible_bucket_entry(buckets_cfg, bucket, user)
    if entry is None:
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


def _nearest(cur, scope: _Scope, vector: str, etag: str, k: int) -> list[dict]:
    """The *k* closest rows in scope, minus exact copies of the query
    (same ETag, a30136cb) and uploaded query objects."""
    model_sql, model_params = scope.model_filter()
    query = sql.SQL(
        "SELECT s3_object, 1.0 - (embedding <=> %s::vector) AS score FROM {}"
        " WHERE starts_with(s3_object, %s) AND strpos(s3_object, %s) = 0"
        " AND checksum IS DISTINCT FROM %s AND embedding IS NOT NULL{}"
        " ORDER BY embedding <=> %s::vector LIMIT %s"
    ).format(sql.Identifier(scope.table), model_sql)
    cur.execute(query, [vector, scope.prefix, _QUERY_UPLOAD_SEGMENT, etag, *model_params,
                        vector, k])
    return [{"object": row[0], "score": _finite_score(row[1])} for row in cur.fetchall()]


def _finite_score(value) -> float:
    """pgvector's ``<=>`` is NaN when a vector has zero norm: score it 0.0
    so the reply stays valid JSON and the summary renders."""
    score = float(value)
    return score if math.isfinite(score) else 0.0


def _count_indexed(cur, scope: _Scope) -> int:
    """Distinct indexed objects in the searched scope (ef8b7037)."""
    model_sql, model_params = scope.model_filter()
    query = sql.SQL(
        "SELECT COUNT(DISTINCT s3_object) FROM {}"
        " WHERE starts_with(s3_object, %s) AND strpos(s3_object, %s) = 0"
        " AND embedding IS NOT NULL{}"
    ).format(sql.Identifier(scope.table), model_sql)
    cur.execute(query, [scope.prefix, _QUERY_UPLOAD_SEGMENT, *model_params])
    return int(cur.fetchone()[0])


def _has_more_candidates(cur, scope: _Scope, etag: str, found: int) -> bool:
    """Whether the scope holds more returnable rows than the *found* ones.

    Stops after ``found + 1`` rows instead of counting the whole scope (the
    query's own ETag rows are never returned, so they do not count).
    """
    model_sql, model_params = scope.model_filter()
    query = sql.SQL(
        "SELECT COUNT(*) FROM (SELECT 1 FROM {}"
        " WHERE starts_with(s3_object, %s) AND strpos(s3_object, %s) = 0"
        " AND checksum IS DISTINCT FROM %s AND embedding IS NOT NULL{} LIMIT %s) AS c"
    ).format(sql.Identifier(scope.table), model_sql)
    cur.execute(query, [scope.prefix, _QUERY_UPLOAD_SEGMENT, etag, *model_params, found + 1])
    return int(cur.fetchone()[0]) > found


def _bounded(cur, name: str, run, on_timeout):
    """``run()`` in a savepoint; when it hits the statement timeout the
    transaction stays usable and *on_timeout* is the answer."""
    savepoint = sql.Identifier(name)
    cur.execute(sql.SQL("SAVEPOINT {}").format(savepoint))
    try:
        value = run()
    except pg_errors.QueryCanceled:
        cur.execute(sql.SQL("ROLLBACK TO SAVEPOINT {}").format(savepoint))
        logger.warning("KNN: %s timed out (UMRS_KNN_EXACT_TIMEOUT_MS)", name)
        return on_timeout
    cur.execute(sql.SQL("RELEASE SAVEPOINT {}").format(savepoint))
    return value


def _ef_search(k: int) -> int:
    """hnsw.ef_search for *k*: at least pgvector's default, at most its bound."""
    return min(HNSW_MAX_EF_SEARCH, max(HNSW_DEFAULT_EF_SEARCH, k))


def exact_timeout_ms() -> int:
    """``UMRS_KNN_EXACT_TIMEOUT_MS``: statement_timeout of the exact pass."""
    return env_int("UMRS_KNN_EXACT_TIMEOUT_MS", DEFAULT_EXACT_TIMEOUT_MS, lo=100, hi=600_000)


def _index_pass(cur, scope: _Scope, vector: str, etag: str, k: int) -> list[dict]:
    """The first pass, under :func:`exact_timeout_ms` like every later one.

    It is an index pass only when the table has its HNSW index; creating it
    is best effort (pgvector indexes at most 2000 dimensions), and without it
    this is a full scan and sort of the shared table (review U1). Cut off,
    it answers :class:`KnnTimeoutError`.
    """
    cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", [str(_ef_search(k))])
    cur.execute("SELECT set_config('statement_timeout', %s, true)", [str(exact_timeout_ms())])
    results = _bounded(cur, "knn_index", lambda: _nearest(cur, scope, vector, etag, k), None)
    if results is None:
        raise KnnTimeoutError("KNN search timed out")
    return results


def _scoped_nearest(cur, scope: _Scope, vector: str, etag: str, k: int):
    """``(results, n_indexed)``, exact within the caller's scope.

    A table several buckets share carries an HNSW index, which yields at most
    ``hnsw.ef_search`` candidates from the whole table before the scope filter.
    ef_search is raised to k for this transaction (within pgvector's 1..1000),
    so a k above pgvector's default (40) no longer falls short on its own
    (review U1). Every scan (the first pass, the scope count, the candidate
    check, the exact pass) runs under :func:`exact_timeout_ms`: these read
    the shared table, other tenants' rows included, and hold a pooled
    connection. When the first pass answers fewer than k rows and the
    scope has more (another tenant's crowd of near-duplicates took the
    candidates), the query runs again as an exact scan. A timed-out first
    pass fails the search (:class:`KnnTimeoutError`); a timed-out count
    reports ``n_indexed`` as None; a timed-out check or exact pass keeps the
    first pass's answer.
    """
    results = _index_pass(cur, scope, vector, etag, k)
    n_indexed = _bounded(cur, "knn_count", lambda: _count_indexed(cur, scope), None)
    if len(results) < k and _bounded(
        cur, "knn_candidates", lambda: _has_more_candidates(cur, scope, etag, len(results)),
        False,
    ):
        results = _exact_nearest(cur, scope, vector, etag, k, results)
    return results, n_indexed


def _exact_nearest(cur, scope: _Scope, vector, etag, k, approximate) -> list[dict]:
    """The query as an exact scan, local to this transaction; when it hits
    the statement timeout, the index's *approximate* answer stands."""
    cur.execute("SAVEPOINT knn_exact")
    cur.execute("SELECT set_config('statement_timeout', %s, true)", [str(exact_timeout_ms())])
    cur.execute("SET LOCAL enable_indexscan = off")
    cur.execute("SET LOCAL enable_seqscan = on")
    try:
        results = _nearest(cur, scope, vector, etag, k)
    except pg_errors.QueryCanceled:
        cur.execute("ROLLBACK TO SAVEPOINT knn_exact")
        logger.warning("KNN: exact pass over %s timed out; answering with the index's %d row(s)",
                       scope.prefix, len(approximate))
        return approximate
    cur.execute("RELEASE SAVEPOINT knn_exact")
    return results


def _with_cursor(data_base, work):
    """Run ``work(cursor)`` on a pooled connection, then roll back and return
    it: a read must not leave the conn idle in transaction."""
    conn = data_base.get_conn()
    try:
        with conn.cursor() as cur:
            return work(cur)
    finally:
        with contextlib.suppress(Exception):
            conn.rollback()
        data_base.put_conn(conn)


def _load_query(entry, request: KnnRequest):
    """``(etag, object)`` of the query object in the caller's bucket."""
    from umrs.shared.storage.s3 import load_data_from_s3

    try:
        objects, _meta, checksums, _sizes, _zero = load_data_from_s3(
            entry.s3_connection, request.bucket, [request.s3_object],
            entry.modality_name,
        )
    except Exception as exc:
        # The S3 error names the server-side endpoint: log it, never reply it.
        logger.warning("KNN: S3 load of %s/%s failed: %s",
                       request.bucket, request.s3_object, exc)
        raise KnnStorageError(
            f"Could not read the query object {request.s3_object} from the bucket"
        ) from exc
    return checksums[0], objects[0]


def search(entry, request: KnnRequest) -> KnnOutcome:
    """Run one KNN search against *entry* (ownership already checked).

    No connection is held while the model computes the query embedding: the
    per-experience pool is small and raises when exhausted, so concurrent
    searches would starve ingestion into the same database (review U1).
    """
    etag, obj = _load_query(entry, request)
    scope = _Scope.of(entry, request)
    data_base = entry.data_base
    vector = _with_cursor(data_base, lambda cur: _cached_embedding(cur, scope, etag))
    cached = vector is not None
    if not cached:
        vector = _vector_literal(entry.model.compute_embeddings([obj])[0])
    results, n_indexed = _with_cursor(
        data_base, lambda cur: _scoped_nearest(cur, scope, vector, etag, request.n)
    )
    return KnnOutcome(results=results, cached=cached, n_indexed=n_indexed)


def _ok_reply(request: KnnRequest, outcome: KnnOutcome, duration_s: float) -> dict:
    from umrs.shared.pretty_print import pretty_print_knn

    return {
        "status": "ok",
        "n": request.n,
        "query": request.display_query(),
        "bucket": request.bucket,
        "subdir": request.subdir,
        "results": outcome.results,
        "duration_s": duration_s,
        "cached": outcome.cached,
        "n_indexed": outcome.n_indexed,
        "summary": pretty_print_knn(
            request.display_query(), request.bucket, request.n, outcome.results,
            duration_s, outcome.cached, outcome.n_indexed,
        ),
    }


def error_reply(code: str, message: str) -> dict:
    """The KNN error shape, one for the handler's refusals and the worker's."""
    return {"status": "error", "code": code, "message": message}


_FAILED = ("KNN_FAILED", "KNN search failed")


def run_knn(entry, request: KnnRequest) -> tuple[dict, str]:
    """``(reply, final task status)``; never raises.

    Building the reply is guarded too (a summary that cannot render must not
    leave the caller waiting). Only :class:`KnnError` messages, which never
    carry backend text, reach the caller; anything else is logged here and
    answered with a fixed message (review U1).
    """
    t0 = time.perf_counter()
    try:
        outcome = search(entry, request)
        duration_s = round(time.perf_counter() - t0, 3)
        return _ok_reply(request, outcome, duration_s), TaskStatus.COMPLETED
    except KnnError as exc:
        logger.warning("KNN refused for %s/%s: %s", request.bucket, request.s3_object, exc)
        return error_reply(exc.code, str(exc)), TaskStatus.ERROR
    except Exception:
        logger.exception("KNN search failed for %s/%s", request.bucket, request.s3_object)
        return error_reply(*_FAILED), TaskStatus.ERROR


def _close_knn_work(active_threads, user, key, update_status_callback, status) -> None:
    """Close the STATUS entry and the task, whatever happened before."""
    try:
        active_threads.update_status_to_done()
        active_threads.record_status_for_users_by_key([user], key)
        active_threads.remove_from_running_list()
    finally:
        if update_status_callback:
            update_status_callback(status)


def _report_running(update_status_callback) -> None:
    """RUNNING for the task; a task-store failure here only logs, so the
    caller still gets its reply and the task still closes (review U1)."""
    if not update_status_callback:
        return
    try:
        update_status_callback(TaskStatus.RUNNING)
    except Exception:
        logger.warning("KNN: could not mark the task RUNNING", exc_info=True)


def knn_search_worker(
    active_threads,
    user,
    request: KnnRequest,
    entry,
    control_mq,
    zmq_sender_id,
    key,
    update_status_callback=None,
    buckets_cfg=None,
):
    """Pool worker for the KNN command: register, search, reply, bookkeeping.

    The STATUS entry is registered here on the pool thread (as
    ``process_model_group`` does), so a search that ends in milliseconds
    cannot finish before its entry exists. The reply and the bookkeeping run
    in ``finally``: the caller always gets an answer and the entry closes.
    """
    reply, status = error_reply(*_FAILED), TaskStatus.ERROR
    try:
        _report_running(update_status_callback)
        if buckets_cfg is not None:
            active_threads.update_thread_dict(buckets_cfg, {request.bucket: None}, key, user=user)
        reply, status = run_knn(entry, request)
    finally:
        with contextlib.suppress(RuntimeError):
            control_mq.send_string(json.dumps(reply, ensure_ascii=False), zmq_sender_id)
        _close_knn_work(active_threads, user, key, update_status_callback, status)
