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


def test_a_failed_query_returns_the_connection(pg_table):
    data_base, _table = pg_table
    entry = _entry(data_base, f"missing_{uuid.uuid4().hex[:8]}")
    baseline = len(data_base.db_pool._used)

    reply, status = knn_worker.run_knn(entry, KnnRequest(n=2, s3_object="q.png", bucket=ALICE))

    assert status == TaskStatus.ERROR
    assert reply["status"] == "error"
    assert len(data_base.db_pool._used) == baseline
