"""PROCESS_OBJECTS_BATCH pre-filter of already-indexed objects (plan U1 A5).

Port of upstream 85a1cf06 (skip objects whose ETag is already stored),
b0c2d345 (the DB connection goes back to the pool on every path) and
8f9fe001 (STATUS shows what was submitted, not what survived the filter).
Hermetic: S3 is a fake; the DB is a fake pool, plus one test against the
lane Postgres that skips when ``POSTGRES_HOST`` is unset.
"""

from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest
from umrs.server import indexed_prefilter, processing


class _Cursor:
    def __init__(self, rows, error):
        self._rows, self._error = rows, error
        self.queries: list = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, _query, params):
        self.queries.append(params)
        if self._error is not None:
            raise self._error

    def fetchall(self):
        return list(self._rows)


class _FakeDataBase:
    """``get_conn``/``put_conn`` with a checkout counter, like the pool."""

    def __init__(self, rows=(), error=None, columns=("s3_object", "checksum", "model")):
        self.rows, self.error, self.columns = rows, error, set(columns)
        self.checked_out = 0
        self.cursors: list[_Cursor] = []

    def has_column(self, _table, column):
        return column in self.columns

    def get_conn(self):
        self.checked_out += 1
        return SimpleNamespace(cursor=self._cursor)

    def _cursor(self):
        cursor = _Cursor(self.rows, self.error)
        self.cursors.append(cursor)
        return cursor

    def put_conn(self, _conn, close=False):  # noqa: ARG002 — pool signature
        self.checked_out -= 1


class _FakeS3:
    def __init__(self, etags):
        self.etags = etags
        self.heads: list[str] = []

    def head_object(self, Bucket, Key):  # noqa: N803 — boto3 signature
        self.heads.append(f"{Bucket}/{Key}")
        if Key not in self.etags:
            raise KeyError(Key)
        return {"ETag": f'"{self.etags[Key]}"'}


class _Model:
    model_name = "clip"
    type = None
    return_type = "embedding"


def _entry(data_base, s3, model=None, store=True, modality="image", task="embedding"):
    return SimpleNamespace(
        data_base=data_base, table_name="tbl", s3_connection=s3,
        model=model or _Model(), store_embeddings_flag=store,
        modality_name=modality, task=task, s3_username="alice",
    )


class _Cfg:
    def __init__(self, entries):
        self.entries = entries

    def get(self, bucket, fields=None, **_kw):
        entry = self.entries.get(bucket)
        if entry is None or fields is None:
            return entry
        return tuple(getattr(entry, f) for f in fields)


@pytest.fixture(autouse=True)
def _owner_rule(monkeypatch):
    """alice owns bucket ``a`` only; bob owns ``b`` only."""
    owners = {"a": "alice", "b": "bob"}
    monkeypatch.setattr(processing, "_client_may_use", lambda b, c: owners.get(b) == c)


def test_connection_returned_when_query_fails():
    """Regression for b0c2d345: the pool checkout count goes back to baseline."""
    db = _FakeDataBase(error=RuntimeError("relation does not exist"))
    cfg = _Cfg({"a": _entry(db, _FakeS3({"k1": "e1"}))})
    entries = [("alice", ["s3://a/k1"])]

    result = indexed_prefilter.filter_already_indexed(cfg, entries)

    assert db.checked_out == 0
    assert result.entries == entries  # fail open: nothing skipped
    assert result.skipped == {}


def test_connection_returned_on_success():
    db = _FakeDataBase(rows=[("a/k1", "e1", "clip")])
    cfg = _Cfg({"a": _entry(db, _FakeS3({"k1": "e1"}))})
    indexed_prefilter.filter_already_indexed(cfg, [("alice", ["s3://a/k1"])])
    assert db.checked_out == 0


def test_skips_only_objects_whose_etag_and_model_match():
    db = _FakeDataBase(rows=[
        ("a/same", "e1", "clip"),
        ("a/changed", "old", "clip"),
        ("a/other-model", "e3", "resnet"),
        ("a/chunked", "e5", "clip"), ("a/chunked", "e5", "clip"),
        ("a/half", "e6", "clip"), ("a/half", "stale", "clip"),
    ])
    s3 = _FakeS3({"same": "e1", "changed": "e2", "other-model": "e3", "new": "e4",
                  "chunked": "e5", "half": "e6"})
    cfg = _Cfg({"a": _entry(db, s3)})
    objects = [f"s3://a/{k}" for k in ("same", "changed", "other-model", "new", "chunked", "half")]

    result = indexed_prefilter.filter_already_indexed(cfg, [("alice", objects)])

    assert result.entries == [("alice", ["s3://a/changed", "s3://a/other-model",
                                         "s3://a/new", "s3://a/half"])]
    assert result.skipped == {"a": 2}
    assert result.submitted == {"a": 6}
    assert result.done_clients == []
    # One query for the bucket, scoped to "<bucket>/<key>".
    assert len(db.cursors) == 1
    assert sorted(db.cursors[0].queries[0][0]) == sorted(o[len("s3://"):] for o in objects)


def test_table_without_model_column_compares_checksum_only():
    db = _FakeDataBase(rows=[("a/k1", "e1")], columns=("s3_object", "checksum"))
    cfg = _Cfg({"a": _entry(db, _FakeS3({"k1": "e1"}))})
    result = indexed_prefilter.filter_already_indexed(cfg, [("alice", ["s3://a/k1"])])
    assert result.entries == []
    assert result.done_clients == ["alice"]
    assert result.fully_skipped() == {"a": 1}


def test_refused_client_is_never_filtered_by_another_tenants_index():
    """bob names alice's bucket: he passes through untouched (and is refused
    later by prepare_model_groups); alice's index is never consulted for him."""
    db = _FakeDataBase(rows=[("a/k1", "e1", "clip")])
    s3 = _FakeS3({"k1": "e1"})
    cfg = _Cfg({"a": _entry(db, s3)})

    result = indexed_prefilter.filter_already_indexed(cfg, [("bob", ["s3://a/k1"])])

    assert result.entries == [("bob", ["s3://a/k1"])]
    assert result.done_clients == []
    assert db.cursors == [] and s3.heads == []


def test_buckets_are_scoped_separately():
    """Same key in two tenants' buckets: each bucket's own table decides."""
    db_a = _FakeDataBase(rows=[("a/k", "e", "clip")])
    db_b = _FakeDataBase(rows=[])
    cfg = _Cfg({"a": _entry(db_a, _FakeS3({"k": "e"})), "b": _entry(db_b, _FakeS3({"k": "e"}))})

    result = indexed_prefilter.filter_already_indexed(
        cfg, [("alice", ["s3://a/k"]), ("bob", ["s3://b/k"])]
    )

    assert result.entries == [("bob", ["s3://b/k"])]
    assert result.done_clients == ["alice"]
    assert db_b.cursors[0].queries[0][0] == ["b/k"]


@pytest.mark.parametrize("entry_kw", [
    {"store": False},
    {"model": SimpleNamespace(model_name="m", type="user_script", return_type="embedding")},
    {"model": SimpleNamespace(model_name="m", type="user_workflow", return_type="embedding")},
    {"model": SimpleNamespace(model_name="m", type=None, return_type="file")},
    {"task": "segmentation"},
])
def test_non_embedding_buckets_always_run(entry_kw):
    db = _FakeDataBase(rows=[("a/k1", "e1", "m")])
    s3 = _FakeS3({"k1": "e1"})
    cfg = _Cfg({"a": _entry(db, s3, **entry_kw)})

    result = indexed_prefilter.filter_already_indexed(cfg, [("alice", ["s3://a/k1"])])

    assert result.entries == [("alice", ["s3://a/k1"])]
    assert db.cursors == [] and s3.heads == []


def test_unreadable_object_is_not_skipped():
    db = _FakeDataBase(rows=[("a/gone", "e1", "clip")])
    cfg = _Cfg({"a": _entry(db, _FakeS3({}))})
    result = indexed_prefilter.filter_already_indexed(cfg, [("alice", ["s3://a/gone"])])
    assert result.entries == [("alice", ["s3://a/gone"])]


@pytest.mark.parametrize(("raw", "enabled"), [
    (None, True), ("1", True), ("0", False), ("off", False), ("False", False),
])
def test_prefilter_knob(monkeypatch, raw, enabled):
    if raw is None:
        monkeypatch.delenv("UMRS_PREFILTER_INDEXED", raising=False)
    else:
        monkeypatch.setenv("UMRS_PREFILTER_INDEXED", raw)
    assert indexed_prefilter.prefilter_enabled() is enabled


# --- Postgres: the real query shape and the real pool ------------------------

_PG = os.environ.get("POSTGRES_HOST")


@pytest.fixture
def pg_database():
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
    table = f"prefilter_{uuid.uuid4().hex[:10]}"
    data_base = DataBase(db_pool, os.environ.get("POSTGRES_DB", "postgres"), "test")
    conn = data_base.get_conn()
    with conn.cursor() as cur:
        cur.execute(f"CREATE TABLE {table} (id SERIAL PRIMARY KEY, s3_object TEXT, "
                    "checksum TEXT, model TEXT)")
    conn.commit()
    data_base.put_conn(conn)
    data_base.add_table(table, {"id": "int", "s3_object": "text", "checksum": "text",
                                "model": "text"})
    yield data_base, table
    conn = data_base.get_conn()
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    conn.commit()
    data_base.put_conn(conn)
    db_pool.closeall()


def _insert(data_base, table, rows):
    conn = data_base.get_conn()
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {table} (s3_object, checksum, model) VALUES (%s, %s, %s)", rows
        )
    conn.commit()
    data_base.put_conn(conn)


def test_postgres_prefilter_skips_indexed_objects(pg_database):
    data_base, table = pg_database
    _insert(data_base, table, [("a/k1", "e1", "clip"), ("a/k2", "old", "clip"),
                               ("b/k1", "e1", "clip")])
    entry = _entry(data_base, _FakeS3({"k1": "e1", "k2": "e2", "k3": "e3"}))
    entry.table_name = table
    cfg = _Cfg({"a": entry})

    result = indexed_prefilter.filter_already_indexed(
        cfg, [("alice", ["s3://a/k1", "s3://a/k2", "s3://a/k3"])]
    )

    assert result.entries == [("alice", ["s3://a/k2", "s3://a/k3"])]
    assert result.skipped == {"a": 1}
    assert len(data_base.db_pool._used) == 0


def test_postgres_connection_returned_when_table_is_missing(pg_database):
    data_base, _table = pg_database
    entry = _entry(data_base, _FakeS3({"k1": "e1"}))
    entry.table_name = f"missing_{uuid.uuid4().hex[:8]}"
    baseline = len(data_base.db_pool._used)

    result = indexed_prefilter.filter_already_indexed(
        _Cfg({"a": entry}), [("alice", ["s3://a/k1"])]
    )

    assert result.entries == [("alice", ["s3://a/k1"])]
    assert len(data_base.db_pool._used) == baseline
