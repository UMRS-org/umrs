"""G5 (ADR-027 P1b): node-keyed pgvector rows via migration 0075.

Verifies the store API emits the intended upsert/nearest SQL and that the
vector literal is a valid pgvector text form. No live database — SQL
capture via the shared fake-pool pattern.
"""

from __future__ import annotations

import asyncio

from umrs.knowledge.store import KnowledgeStore


class _Conn:
    def __init__(self) -> None:
        self.executed: list[tuple] = []
        self.fetched: list[tuple] = []

    async def execute(self, sql: str, *params: object) -> str:
        self.executed.append((sql, params))
        return "INSERT 0 1"

    async def fetch(self, sql: str, *params: object) -> list[dict]:
        self.fetched.append((sql, params))
        return [
            {"node_id": "doc-1-chunk0", "distance": 0.01},
            {"node_id": "doc-1-chunk1", "distance": 0.20},
        ]


class _AcquireCtx:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *_exc: object) -> None:
        pass


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self._conn)


def _run(coro):
    """Fresh loop per call — safe under pytest-asyncio's auto mode."""
    return asyncio.new_event_loop().run_until_complete(coro)


def test_upsert_node_embedding_emits_cast_and_conflict():
    conn = _Conn()
    store = KnowledgeStore(_Pool(conn))
    _run(
        store.upsert_node_embedding(
            "doc-1-chunk0", "e5-large", [0.5, 0.25, -1.0], content_hash="abc123"
        )
    )
    assert len(conn.executed) == 1
    sql, params = conn.executed[0]
    assert "knowledge_node_embeddings" in sql
    assert "$4::vector" in sql
    assert "ON CONFLICT (node_id, model_name)" in sql
    node_id, model, dim, vec, h = params
    assert (node_id, model, dim, h) == ("doc-1-chunk0", "e5-large", 3, "abc123")
    assert vec == "[0.5,0.25,-1.0]"


def test_nearest_node_ids_scopes_by_project_and_model():
    conn = _Conn()
    store = KnowledgeStore(_Pool(conn))
    rows = _run(
        store.nearest_node_ids([0.1, 0.2], project_id="spring8", model_name="e5-large", limit=7)
    )
    sql, params = conn.fetched[0]
    assert "JOIN knowledge_nodes" in sql
    assert "e.model_name = $1" in sql and "n.project_id = $2" in sql
    assert params[0] == "e5-large" and params[1] == "spring8" and params[2] == 7
    assert rows == [("doc-1-chunk0", 0.01), ("doc-1-chunk1", 0.2)]


def test_vector_literal_is_parsable_pgvector_form():
    literal = KnowledgeStore._vector_literal([1, 0.125, -3])
    assert literal == "[1.0,0.125,-3.0]"
    assert literal.startswith("[") and literal.endswith("]")
    floats = [float(x) for x in literal[1:-1].split(",")]
    assert floats == [1.0, 0.125, -3.0]
