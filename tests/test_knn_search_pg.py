"""KNN search against a real pgvector table (plan U1 A6).

Port of upstream 1798ac05 / e04b3d4e / b7d5821a / ef8b7037 / a30136cb. The
table is shared by two tenants' buckets on purpose: every query must stay
inside the caller's bucket (and model). Skips when ``POSTGRES_HOST`` is unset.
"""

from __future__ import annotations

import io
import os
import uuid
from types import SimpleNamespace

import pytest
from umrs.core.TaskCmdStatus import TaskStatus
from umrs.server import knn_worker
from umrs.server.knn_worker import KnnRequest

ALICE = "alice.xp.images"
BOB = "bob.xp.images"
_PG = os.environ.get("POSTGRES_HOST")


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class _S3:
    """get_object for the query object: fixed ETag, a tiny PNG body."""

    def __init__(self, etags: dict[tuple[str, str], str]):
        self.etags = etags

    def get_object(self, Bucket, Key):
        body = _png()
        return {"ETag": f'"{self.etags[(Bucket, Key)]}"', "Body": io.BytesIO(body),
                "ContentLength": len(body), "Metadata": {}}


class _Model:
    model_name = "clip"
    type = "local"
    return_type = "embedding"

    def __init__(self, vector=(1.0, 0.0, 0.0)):
        self.vector = list(vector)
        self.calls = 0

    def compute_embeddings(self, objects):
        self.calls += 1
        return [list(self.vector) for _ in objects]


@pytest.fixture
def pg_table():
    if not _PG:
        pytest.skip("POSTGRES_HOST unset")
    from psycopg2 import pool as pg_pool
    from umrs.server.sql import DataBase

    db_pool = pg_pool.SimpleConnectionPool(
        1, 3, host=_PG, port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        dbname=os.environ.get("POSTGRES_DB", "postgres"),
    )
    table = f"knn_{uuid.uuid4().hex[:10]}"
    data_base = DataBase(db_pool, os.environ.get("POSTGRES_DB", "postgres"), "test")
    _exec(data_base, "CREATE EXTENSION IF NOT EXISTS vector")
    _exec(data_base, f"CREATE TABLE {table} (id SERIAL PRIMARY KEY, s3_object TEXT, "
                     "checksum TEXT, embedding VECTOR(3), model TEXT, UNIQUE(s3_object))")
    data_base.add_table(table, {"id": "int", "s3_object": "text", "checksum": "text",
                                "embedding": "vector", "model": "text"})
    yield data_base, table
    _exec(data_base, f"DROP TABLE IF EXISTS {table}")
    db_pool.closeall()


def _exec(data_base, statement, params=None, many=False):
    conn = data_base.get_conn()
    try:
        with conn.cursor() as cur:
            (cur.executemany if many else cur.execute)(statement, params)
        conn.commit()
    finally:
        data_base.put_conn(conn)


def _insert(data_base, table, rows):
    """rows: (s3_object, checksum, [x, y, z], model)."""
    _exec(data_base,
          f"INSERT INTO {table} (s3_object, checksum, embedding, model) "
          "VALUES (%s, %s, %s::vector, %s)",
          [(obj, etag, str(vec), model) for obj, etag, vec, model in rows], many=True)


def _entry(data_base, table, model=None, etags=None):
    return SimpleNamespace(
        data_base=data_base, table_name=table, model=model or _Model(),
        modality_name="image", s3_connection=_S3(etags or {(ALICE, "q.png"): "eq"}),
    )


def _objects(outcome):
    return [hit["object"] for hit in outcome.results]


def _search(entry, n=10, s3_object="q.png", bucket=ALICE, subdir=None):
    return knn_worker.search(entry, KnnRequest(n=n, s3_object=s3_object, bucket=bucket,
                                               subdir=subdir))


# --- ordering ------------------------------------------------------------------


def test_neighbours_are_ordered_by_cosine_similarity(pg_table):
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{ALICE}/far.png", "e1", [-1, 0, 0], "clip"),
        (f"{ALICE}/near.png", "e2", [1, 0.1, 0], "clip"),
        (f"{ALICE}/mid.png", "e3", [1, 1, 0], "clip"),
        (f"{ALICE}/side.png", "e4", [0, 1, 0], "clip"),
    ])
    outcome = _search(_entry(data_base, table), n=3)

    assert _objects(outcome) == [f"{ALICE}/near.png", f"{ALICE}/mid.png", f"{ALICE}/side.png"]
    scores = [hit["score"] for hit in outcome.results]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1 / (1.01 ** 0.5), abs=1e-5)
    assert scores[1] == pytest.approx(2 ** -0.5, abs=1e-5)
    assert scores[2] == pytest.approx(0.0, abs=1e-6)
    assert outcome.cached is False
    assert len(data_base.db_pool._used) == 0


# --- scope -----------------------------------------------------------------------


def test_each_tenant_sees_only_its_own_bucket(pg_table):
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{BOB}/secret.png", "b1", [1, 0, 0], "clip"),   # the best match, bob's
        (f"{ALICE}/a.png", "a1", [1, 1, 0], "clip"),
        (f"{ALICE}x/look-alike.png", "a2", [1, 0, 0], "clip"),  # bucket-name prefix trap
    ])
    etags = {(ALICE, "q.png"): "eq", (BOB, "q.png"): "eq"}

    alice = _search(_entry(data_base, table, etags=etags))
    bob = _search(_entry(data_base, table, etags=etags), bucket=BOB)

    assert _objects(alice) == [f"{ALICE}/a.png"]
    assert alice.n_indexed == 1
    assert _objects(bob) == [f"{BOB}/secret.png"]
    assert bob.n_indexed == 1


def test_rows_of_another_model_are_not_searched(pg_table):
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{ALICE}/old.png", "o1", [1, 0, 0], "resnet"),
        (f"{ALICE}/new.png", "n1", [1, 1, 0], "clip"),
    ])
    outcome = _search(_entry(data_base, table))
    assert _objects(outcome) == [f"{ALICE}/new.png"]
    assert outcome.n_indexed == 1


def test_subdir_limits_results_to_its_prefix(pg_table):
    """b7d5821a: the DB stores ``<bucket>/<key>`` with no ``s3://`` prefix,
    so a prefix built as ``s3://<bucket>/<subdir>/`` matched nothing."""
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{ALICE}/pets/cat.png", "c1", [1, 0.2, 0], "clip"),
        (f"{ALICE}/pets/dog.png", "c2", [1, 0.5, 0], "clip"),
        (f"{ALICE}/pets2/rat.png", "c3", [1, 0, 0], "clip"),
        (f"{ALICE}/cars/car.png", "c4", [1, 0, 0], "clip"),
    ])
    outcome = _search(_entry(data_base, table), subdir="pets")

    assert _objects(outcome) == [f"{ALICE}/pets/cat.png", f"{ALICE}/pets/dog.png"]


def test_n_indexed_counts_the_searched_subdir_only(pg_table):
    """ef8b7037: n_indexed is the scoped count, not the whole bucket."""
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{ALICE}/pets/cat.png", "c1", [1, 0, 0], "clip"),
        (f"{ALICE}/pets/dog.png", "c2", [0, 1, 0], "clip"),
        (f"{ALICE}/cars/car.png", "c3", [0, 0, 1], "clip"),
        (f"{BOB}/pets/x.png", "c4", [0, 0, 1], "clip"),
    ])
    entry = _entry(data_base, table)

    assert _search(entry, n=1, subdir="pets").n_indexed == 2
    assert _search(entry, n=1).n_indexed == 3


def test_a_quoted_subdir_is_data_not_sql(pg_table):
    data_base, table = pg_table
    _insert(data_base, table, [(f"{ALICE}/a.png", "a1", [1, 0, 0], "clip")])
    subdir = knn_worker.normalise_subdir(f"x'; DROP TABLE {table}; --")

    outcome = _search(_entry(data_base, table), subdir=subdir)

    assert outcome.results == []
    assert _search(_entry(data_base, table)).n_indexed == 1  # table still there


# --- query embedding cache ---------------------------------------------------------


def test_the_stored_embedding_of_the_query_object_is_reused(pg_table):
    """b7d5821a: the lookup used the raw key, never ``<bucket>/<key>``."""
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{ALICE}/q.png", "eq", [0, 1, 0], "clip"),
        (f"{ALICE}/up.png", "e1", [0, 1, 0.1], "clip"),
        (f"{ALICE}/right.png", "e2", [1, 0, 0], "clip"),
    ])
    model = _Model(vector=(1, 0, 0))
    outcome = _search(_entry(data_base, table, model=model), n=1)

    assert outcome.cached is True
    assert model.calls == 0


def test_a_cached_vector_is_never_taken_from_another_tenant(pg_table):
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{BOB}/q.png", "eq", [0, 1, 0], "clip"),
        (f"{ALICE}/a.png", "a1", [1, 0, 0], "clip"),
    ])
    model = _Model()
    outcome = _search(_entry(data_base, table, model=model))

    assert outcome.cached is False
    assert model.calls == 1


def test_an_identical_copy_at_another_key_answers_the_cache(pg_table):
    """a30136cb: the cache matches the ETag anywhere in the bucket, so a
    query uploaded at ``q.png`` reuses the vector of ``pets/cat.png``."""
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{ALICE}/pets/cat.png", "eq", [0, 1, 0], "clip"),
        (f"{ALICE}/pets/dog.png", "e1", [0, 1, 0.1], "clip"),
    ])
    model = _Model(vector=(1, 0, 0))
    outcome = _search(_entry(data_base, table, model=model), n=1, subdir="pets")

    assert outcome.cached is True
    assert model.calls == 0


def test_exact_duplicates_of_the_query_are_not_neighbours(pg_table):
    """a30136cb: the query object (and byte-identical copies, same ETag) is
    not its own nearest neighbour."""
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{ALICE}/q.png", "eq", [1, 0, 0], "clip"),
        (f"{ALICE}/copies/q-copy.png", "eq", [1, 0, 0], "clip"),
        (f"{ALICE}/near.png", "e1", [1, 0.1, 0], "clip"),
        (f"{ALICE}/far.png", "e2", [0, 1, 0], "clip"),
    ])
    outcome = _search(_entry(data_base, table), n=2)

    assert _objects(outcome) == [f"{ALICE}/near.png", f"{ALICE}/far.png"]
    assert outcome.cached is True


# --- connection hygiene ---------------------------------------------------------------


class _PoolWatchingModel(_Model):
    """Records how many pool connections are checked out during inference."""

    def __init__(self, data_base):
        super().__init__()
        self.data_base = data_base
        self.used_during_compute = []

    def compute_embeddings(self, objects):
        self.used_during_compute.append(len(self.data_base.db_pool._used))
        return super().compute_embeddings(objects)


def test_no_connection_is_held_through_the_forward_pass(pg_table):
    """Review U1: the per-experience pool (maxconn 5, raises when full) lent a
    connection to every KNN for the whole model forward pass."""
    data_base, table = pg_table
    _insert(data_base, table, [(f"{ALICE}/a.png", "a1", [1, 0, 0], "clip")])
    model = _PoolWatchingModel(data_base)

    outcome = _search(_entry(data_base, table, model=model))

    assert outcome.cached is False
    assert model.used_during_compute == [0]
    assert len(data_base.db_pool._used) == 0


@pytest.fixture
def pg_hnsw_table(pg_table):
    """``pg_table`` plus the HNSW index every embedding table gets, read
    through connections whose planner always takes it (as on a large table)."""
    from psycopg2 import pool as pg_pool
    from umrs.server.sql import DataBase

    data_base, table = pg_table
    _exec(data_base, f"CREATE INDEX ON {table} USING hnsw (embedding vector_cosine_ops)")
    index_pool = pg_pool.SimpleConnectionPool(
        1, 3, host=_PG, port=os.environ.get("POSTGRES_PORT", "5432"),
        user=os.environ.get("POSTGRES_USER", "postgres"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        dbname=os.environ.get("POSTGRES_DB", "postgres"),
        options="-c enable_seqscan=off",
    )
    index_db = DataBase(index_pool, os.environ.get("POSTGRES_DB", "postgres"), "test")
    index_db.add_table(table, {"id": "int", "s3_object": "text", "checksum": "text",
                               "embedding": "vector", "model": "text"})
    yield data_base, index_db, table
    index_pool.closeall()


def test_another_tenants_crowd_cannot_starve_the_search(pg_hnsw_table):
    """Review U1: HNSW filters after its scan (ef_search = 40 candidates from
    the whole table). Bob's 60 near-copies of the query used to fill every
    candidate, so Alice got no result although her bucket holds 5 rows."""
    data_base, index_db, table = pg_hnsw_table
    crowd = [(f"{BOB}/copy{i}.png", f"b{i}", [1, 0.001 * i, 0], "clip") for i in range(60)]
    mine = [(f"{ALICE}/a{i}.png", f"a{i}", [1, 1 + i, 0], "clip") for i in range(5)]
    _insert(data_base, table, crowd + mine)

    outcome = _search(_entry(index_db, table), n=5)

    assert outcome.n_indexed == 5
    assert _objects(outcome) == [f"{ALICE}/a{i}.png" for i in range(5)]
    assert len(index_db.db_pool._used) == 0


class _Recorder:
    """Cursor proxy: records every statement the search runs."""

    def __init__(self, cur):
        self._cur = cur
        self.statements: list[str] = []

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else query.as_string(self._cur)
        self.statements.append(text)
        return self._cur.execute(query, params)

    def __getattr__(self, name):
        return getattr(self._cur, name)


def _scoped(data_base, table, n, vector="[1,0,0]", etag="eq"):
    """``(results, n_indexed, statements)`` of one scoped search."""
    request = KnnRequest(n=n, s3_object="q.png", bucket=ALICE)
    scope = knn_worker._Scope.of(_entry(data_base, table), request)
    recorders: list[_Recorder] = []

    def _work(cur):
        recorders.append(_Recorder(cur))
        return knn_worker._scoped_nearest(recorders[0], scope, vector, etag, n)

    results, n_indexed = knn_worker._with_cursor(data_base, _work)
    return results, n_indexed, recorders[0].statements


def _ran_exact_pass(statements):
    return any("enable_indexscan" in s for s in statements)


def test_k_above_the_default_ef_search_stays_on_the_index(pg_hnsw_table):
    """Review U1: HNSW yields at most hnsw.ef_search (40) rows, so every k
    above 40 re-ran the query as a scan of the whole shared table."""
    data_base, index_db, table = pg_hnsw_table
    _insert(data_base, table,
            [(f"{ALICE}/a{i}.png", f"a{i}", [1, 0.01 * i, 0], "clip") for i in range(80)])

    results, n_indexed, statements = _scoped(index_db, table, n=60)

    assert n_indexed == 80
    assert len(results) == 60
    assert not _ran_exact_pass(statements)


def test_an_indexed_query_object_does_not_force_the_exact_pass(pg_table):
    """Review U1: n_indexed counts the query's own row, which the search
    excludes, so k >= n_indexed always looked short by one."""
    data_base, table = pg_table
    _insert(data_base, table, [(f"{ALICE}/q.png", "eq", [1, 0, 0], "clip")] + [
        (f"{ALICE}/a{i}.png", f"a{i}", [1, 0.1 * i, 0], "clip") for i in range(1, 10)
    ])

    results, n_indexed, statements = _scoped(data_base, table, n=20)

    assert n_indexed == 10
    assert len(results) == 9
    assert not _ran_exact_pass(statements)


def test_uploaded_query_objects_are_never_neighbours(pg_table):
    """`umrs knn` uploads its query under <subdir>/.umrs-knn/; should one of
    those ever be indexed it is neither returned nor counted."""
    data_base, table = pg_table
    _insert(data_base, table, [
        (f"{ALICE}/pets/.umrs-knn/0a1b-q.png", "x1", [1, 0, 0], "clip"),
        (f"{ALICE}/.umrs-knn/0a1c-q.png", "x2", [1, 0, 0], "clip"),
        (f"{ALICE}/pets/cat.png", "c1", [1, 1, 0], "clip"),
    ])
    outcome = _search(_entry(data_base, table))

    assert _objects(outcome) == [f"{ALICE}/pets/cat.png"]
    assert outcome.n_indexed == 1


def test_a_failed_query_returns_the_connection(pg_table):
    data_base, _table = pg_table
    entry = _entry(data_base, f"missing_{uuid.uuid4().hex[:8]}")
    baseline = len(data_base.db_pool._used)

    reply, status = knn_worker.run_knn(entry, KnnRequest(n=2, s3_object="q.png", bucket=ALICE))

    assert status == TaskStatus.ERROR
    assert reply["status"] == "error"
    assert len(data_base.db_pool._used) == baseline


class _Slow(_Recorder):
    """The statement holding *marker* takes 2 s (a shared table of millions
    of rows) and then answers *stand_in*'s valid rows."""

    def __init__(self, cur, marker, stand_in):
        super().__init__(cur)
        self.marker, self.stand_in = marker, stand_in

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else query.as_string(self._cur)
        if self.marker in text:
            self.statements.append(text)
            return self._cur.execute(self.stand_in)
        return super().execute(query, params)


def _timed_scoped(data_base, table, slow):
    """``(outcome or the exception, seconds)`` of a k=3 search through *slow*."""
    import time

    request = KnnRequest(n=3, s3_object="q.png", bucket=ALICE)
    scope = knn_worker._Scope.of(_entry(data_base, table), request)
    t0 = time.monotonic()
    try:
        outcome = knn_worker._with_cursor(
            data_base, lambda cur: knn_worker._scoped_nearest(slow(cur), scope, "[1,0,0]",
                                                              "eq", 3))
    except knn_worker.KnnError as exc:
        outcome = exc
    return outcome, time.monotonic() - t0


def test_the_scope_count_is_bounded_by_the_statement_timeout(pg_table, monkeypatch):
    """Review U1: the COUNT(DISTINCT) over the shared table ran with no
    statement_timeout, holding one of the 5 pooled connections for as long
    as the scan took; now it is cut off and n_indexed is unknown. The stand-in
    answers a valid count (7), so an unbounded run cannot pass as a cut-off."""
    data_base, table = pg_table
    _insert(data_base, table,
            [(f"{ALICE}/a{i}.png", f"a{i}", [1, 0.1 * i, 0], "clip") for i in range(3)])
    monkeypatch.setenv("UMRS_KNN_EXACT_TIMEOUT_MS", "200")

    (results, n_indexed), seconds = _timed_scoped(
        data_base, table, lambda cur: _Slow(cur, "COUNT(DISTINCT", "SELECT 7 FROM pg_sleep(2)"))

    assert n_indexed is None
    assert len(results) == 3
    assert seconds < 1.5


def test_the_first_pass_is_bounded_without_an_hnsw_index(pg_table, monkeypatch):
    """Review U1: the first pass ran before any statement_timeout. pg_table
    has no HNSW index (creating one is best effort; pgvector refuses it above
    2000 dimensions), so that pass is a full scan and sort of the shared
    table: cut off, the search answers a timeout and frees its connection."""
    data_base, table = pg_table
    _insert(data_base, table, [(f"{ALICE}/a0.png", "a0", [1, 0, 0], "clip")])
    monkeypatch.setenv("UMRS_KNN_EXACT_TIMEOUT_MS", "200")
    baseline = len(data_base.db_pool._used)

    outcome, seconds = _timed_scoped(
        data_base, table,
        lambda cur: _Slow(cur, "ORDER BY embedding", "SELECT 'x', 1.0 FROM pg_sleep(2)"))

    assert isinstance(outcome, knn_worker.KnnTimeoutError)
    assert seconds < 1.5
    assert len(data_base.db_pool._used) == baseline


def test_k_above_pgvectors_ef_search_bound_still_answers(pg_hnsw_table):
    """Review U1: hnsw.ef_search accepts 1..1000; set to k=2000 (an admin's
    UMRS_KNN_MAX_K=5000) Postgres refused it and the KNN failed."""
    data_base, index_db, table = pg_hnsw_table
    _insert(data_base, table,
            [(f"{ALICE}/a{i}.png", f"a{i}", [1, 0.01 * i, 0], "clip") for i in range(5)])
    request = KnnRequest(n=2000, s3_object="q.png", bucket=ALICE)
    scope = knn_worker._Scope.of(_entry(index_db, table), request)

    def _work(cur):
        cur.execute("SELECT '[1,0,0]'::vector")  # pgvector loaded on this backend
        return knn_worker._scoped_nearest(cur, scope, "[1,0,0]", "eq", 2000)

    results, n_indexed = knn_worker._with_cursor(index_db, _work)

    assert n_indexed == 5
    assert len(results) == 5
