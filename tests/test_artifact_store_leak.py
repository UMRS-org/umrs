"""Regression: artifact-store connections must return to the pool on every path.

Root-cause defect: the API route's finally block called
``db_pool.putconn(conn_factory())`` — acquiring a SECOND connection in order
to return one, leaking exactly one connection per HTTP request until the
psycopg2 pool exhausted and the artifacts route returned 500
(`psycopg2.pool.PoolError: connection pool exhausted`, live 2026-08-12).
"""

from __future__ import annotations

import uuid

from umrs.artifact_registry.store import ArtifactStore


class _FakeCur:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, *_sql, **_params):
        return None

    def fetchall(self):
        return []

    def fetchone(self):
        return None


class _FakeConn:
    def __init__(self):
        self.closed_count = 0

    def cursor(self):
        return _FakeCur(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed_count += 1


class _FakePool:
    def __init__(self, maxconn: int = 2):
        self.maxconn = maxconn
        self._all: list[_FakeConn] = []
        self._free: list[_FakeConn] = []

    def getconn(self):
        if self._free:
            return self._free.pop()
        if len(self._all) >= self.maxconn:
            raise RuntimeError("connection pool exhausted")
        conn = _FakeConn()
        self._all.append(conn)
        return conn

    def putconn(self, conn):
        self._free.append(conn)


def _artifact():
    from umrs.artifact_registry.models import Artifact

    return Artifact(
        id=uuid.uuid4(),
        artifact_type="file",
        name="a",
        owner_id="o",
        project_id="p",
        uri="s3://b/k",
        bucket="b",
        s3_key="k",
        size_bytes=1,
        checksum=None,
        metadata={},
        execution_id=None,
        workflow_id=None,
        job_id=None,
        version=1,
    )


def test_leased_connections_return_to_pool_on_close():
    pool = _FakePool(maxconn=2)
    store = ArtifactStore(pool.getconn, conn_release=pool.putconn)
    store.list(
        type(
            "Q",
            (),
            {
                "artifact_type": None,
                "owner_id": None,
                "project_id": None,
                "execution_id": None,
                "name_prefix": None,
                "limit": 10,
                "offset": 0,
            },
        )()
    )
    store.get(uuid.uuid4())
    store.close()
    assert len(pool._free) == 2, f"every leased conn returned: {pool._free}"

    # Pool is reusable afterwards — no exhaustion is possible.
    for _ in range(3):
        s2 = ArtifactStore(pool.getconn, conn_release=pool.putconn)
        s2.get(uuid.uuid4())
        s2.close()


def test_close_is_idempotent_and_safe_without_release():
    calls = {"made": 0}

    def factory():
        calls["made"] += 1
        return _FakeConn()

    store = ArtifactStore(factory)  # no conn_release → falls back to conn.close
    c1 = store._get_conn()
    store.close()
    store.close()  # idempotent
    assert c1.closed_count == 1


def test_create_then_close_releases_conn():
    pool = _FakePool(maxconn=1)
    store = ArtifactStore(pool.getconn, conn_release=pool.putconn)
    store._get_conn()  # simulate any method
    store.close()
    # Second lease works — proves the one-slot pool is cycled correctly.
    store2 = ArtifactStore(pool.getconn, conn_release=pool.putconn)
    store2._get_conn()
    store2.close()
