"""P2 wave 2 (ADR-027): K3 facade embeddings backend over migration 0075.

Table-reader lane: honest hits from knowledge_node_embeddings joined to
nodes, recency-ranked, with model/dim provenance in metadata. Wiring is
gated on the table existing (regclass probe) — otherwise the facade keeps
its honesty reason.
"""

from __future__ import annotations

from datetime import datetime, timezone

from umrs.api.knowledge_query import (
    _build_unified_search,
    _embeddings_backend_for,
    _EmbeddingsSearchBackend,
)


class _Cur:
    def __init__(self, behavior) -> None:
        self._behavior = behavior
        self.queries: list[tuple[str, tuple]] = []

    def execute(self, sql, params=None):
        self.queries.append((sql, params))
        self._rows = self._behavior(sql, params)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Conn:
    def __init__(self, behavior) -> None:
        self._cur = _Cur(behavior)

    def cursor(self):
        return self._cur

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _Pool:
    def __init__(self, behavior) -> None:
        self._conn = _Conn(behavior)
        self.returned = 0

    def getconn(self):
        return self._conn

    def putconn(self, _conn):
        self.returned += 1


_ROWS = [
    (
        "ragdoc-doc-1-chunk0",
        "intfloat/e5-base-v2",
        768,
        datetime(2026, 8, 23, tzinfo=timezone.utc),
        "chunk 0 of scan.h5",
        "spring8",
    ),
    (
        "ragdoc-doc-2-chunk3",
        "intfloat/e5-base-v2",
        768,
        datetime(2026, 8, 22, tzinfo=timezone.utc),
        "chunk 3 of run.md",
        "spring8",
    ),
]


def _emb_hits(limit=5):
    pool = _Pool(lambda _sql, _params: _ROWS)
    backend = _EmbeddingsSearchBackend(pool)
    return pool, backend.search_embeddings("e5", project_id="spring8", limit=limit)


def test_backend_sql_is_project_scoped_join():
    pool, hits = _emb_hits()
    assert len(hits) == 2
    sql, params = pool._conn._cur.queries[0]
    assert "knowledge_node_embeddings e" in sql
    assert "JOIN knowledge_nodes" in sql
    assert params[0] == "spring8" and params[-1] == 5


def test_backend_hit_carries_vector_provenance():
    _, hits = _emb_hits()
    first = hits[0]
    assert first["source_id"] == "ragdoc-doc-1-chunk0"
    assert first["source_type"] == "knowledge_node_embedding"
    meta = first["metadata"]
    assert meta["model_name"] == "intfloat/e5-base-v2"
    assert meta["dimension"] == 768
    assert meta["lane"] == "table-reader"
    assert first["score"] == 0.5


def test_wiring_gated_on_table_presence():
    pool = _Pool(lambda _sql, _params: [("knowledge_node_embeddings",)])
    backend, reason = _embeddings_backend_for(pool)
    assert backend is not None and reason is None
    assert pool.returned == 1


def test_wiring_reports_reason_when_table_missing():
    pool = _Pool(lambda _sql, _params: [None])
    backend, reason = _embeddings_backend_for(pool)
    assert backend is None
    assert "0075 not applied" in reason


def test_wiring_reports_reason_without_pool():
    backend, reason = _embeddings_backend_for(None)
    assert backend is None
    assert "requires the database" in reason


def test_build_unified_search_marks_embeddings_wired():
    class _StubStore:
        def search(self, *_a, **_k):
            return []

    backend = _EmbeddingsSearchBackend(_Pool(lambda _sql, _params: _ROWS))
    _, wired, unwired = _build_unified_search(_StubStore(), None, "no pool", backend, None)
    assert "embeddings" in wired
    assert "embeddings" not in unwired
    assert unwired["knowledge_graph"] == "no pool"

    _, wired2, unwired2 = _build_unified_search(_StubStore(), None, "no pool", None, "probe failed")
    assert "embeddings" not in wired2
    assert unwired2["embeddings"] == "probe failed"
