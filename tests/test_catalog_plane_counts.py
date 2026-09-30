"""Per-plane row counts on the catalog stats route (ADR-031 §Consequences 6).

Pins: the projection counts canonical documents/chunks/vector rows joined to
the project; pool absence or query failure omits the fields (honest absence),
never a fabricated zero.
"""

from __future__ import annotations

from catalog_store_fake import FakeCatalogMetaStore
from umrs.api.catalog import router as catalog_router
from umrs.api.catalog_store import set_catalog_meta_store
from umrs.data_catalog.plane_counts import compute_plane_counts


class _Cur:
    def __init__(self, script):
        self._script = script
        self._n = 0

    def execute(self, sql, params=None):
        for probe, count in self._script.items():
            if probe in sql:
                self._n = count if params and params[0] == "proj-1" else 0
                return
        raise AssertionError(f"unexpected SQL: {sql}")

    def fetchone(self):
        return (self._n,)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Conn:
    def __init__(self, script):
        self._script = script

    def cursor(self):
        return _Cur(self._script)


class _Pool:
    def __init__(self, script):
        self._script = script

    def getconn(self):
        return _Conn(self._script)

    def putconn(self, conn):
        return None


_SCRIPT = {
    "'document'": 3,
    "'chunk'": 11,
    "knowledge_node_embeddings": 7,
}


def test_compute_plane_counts_reads_three_planes():
    counts = compute_plane_counts(_Pool(_SCRIPT), "proj-1")
    assert counts.canonical_documents == 3
    assert counts.chunk_nodes == 11
    assert counts.vector_rows == 7
    payload = counts.to_dict()
    assert set(payload) == {"canonical_documents", "chunk_nodes", "vector_rows"}


def test_compute_plane_counts_requires_pool():
    import pytest

    with pytest.raises(RuntimeError, match="shared umrs_db pool"):
        compute_plane_counts(None, "proj-1")


def _stats_app(pool=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(catalog_router)
    set_catalog_meta_store(FakeCatalogMetaStore())
    if pool is not None:
        app.state.umrs_db_pool = pool
    return TestClient(app)


def test_stats_route_includes_plane_counts_when_pool_present():
    with _stats_app(pool=_Pool(_SCRIPT)) as client:
        response = client.get("/umrs/catalog/stats", headers={"X-Project-ID": "proj-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["canonical_documents"] == 3
    assert body["chunk_nodes"] == 11
    assert body["vector_rows"] == 7


def test_stats_route_omits_plane_counts_when_pool_absent():
    with _stats_app(pool=None) as client:
        response = client.get("/umrs/catalog/stats", headers={"X-Project-ID": "proj-1"})
    assert response.status_code == 200
    body = response.json()
    # Honest absence: fields present but null, never a fabricated zero.
    assert body["canonical_documents"] is None
    assert body["chunk_nodes"] is None
    assert body["vector_rows"] is None


class _FailingPool:
    def getconn(self):
        raise RuntimeError("pool dead")

    def putconn(self, conn):
        return None


def test_stats_route_omits_plane_counts_when_query_fails():
    with _stats_app(pool=_FailingPool()) as client:
        response = client.get("/umrs/catalog/stats", headers={"X-Project-ID": "proj-1"})
    assert response.status_code == 200
    body = response.json()
    assert body["canonical_documents"] is None
    assert body["vector_rows"] is None
