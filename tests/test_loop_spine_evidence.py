"""The research loop finds its evidence in the knowledge spine.

Plan §3c.2 [REMOTE TOO], reproduced on the stand-alone stack 2026-10-09: the
server-side loop engine was built with no knowledge store, so in real mode
every iteration retrieved nothing and stopped at the evidence gate ("refusing
to publish an unevidenced finding") — although the project held a note that
answered the goal. The engine now has a spine store, bound to each loop's
project.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from umrs.api import knowledge_query, rag_spine
from umrs.knowledge.loop import AutonomousResearchLoop


def test_the_app_engine_has_a_knowledge_store(monkeypatch):
    from umrs.api import loop as loop_api

    monkeypatch.setattr(loop_api, "_loop_engine", None)
    monkeypatch.setattr(loop_api, "_default_workflow_port", lambda: None)
    engine = loop_api._get_engine()
    assert isinstance(engine._knowledge_store, rag_spine.SpineKnowledgeStore)


def test_the_store_is_bound_to_the_loops_project():
    engine = AutonomousResearchLoop(knowledge_store=rag_spine.SpineKnowledgeStore())
    store = engine._store_for(SimpleNamespace(project_id="p-zeolite"))
    assert store._project_id == "p-zeolite"


def test_an_unscoped_store_is_used_as_is():
    plain = SimpleNamespace(search_artifacts=lambda _q: [])
    engine = AutonomousResearchLoop(knowledge_store=plain)
    assert engine._store_for(SimpleNamespace(project_id="p")) is plain


def test_search_returns_passages_as_loop_artifacts(monkeypatch):
    passages = [{"source_id": "node-chunk-art-1-0", "title": "chunk", "score": 0.8, "metadata": {}}]
    seen = {}

    def retrieve(pool, query, project_id, *, top_k, threshold):
        seen.update(query=query, project=project_id, threshold=threshold)
        return passages, ""

    def attach(pool, ps, fetch):
        return [
            {
                **ps[0],
                "text": "Ion exchange with sodium raises its CO2 uptake.",
                "source_name": "zeolite-note.md",
                "artifact_id": "art-1",
                "source_ref": "projects/p/knowledge/artifacts/abc",
            }
        ]

    monkeypatch.setattr(rag_spine, "retrieve_passages", retrieve)
    monkeypatch.setattr(rag_spine, "attach_texts", attach)
    store = rag_spine.SpineKnowledgeStore(pool=lambda: None).for_project("p")
    [artifact] = store.search_artifacts("sodium exchange ZSM-5")
    assert seen == {"query": "sodium exchange ZSM-5", "project": "p", "threshold": 0.7}
    assert artifact["artifact_id"] == "art-1" and artifact["filename"] == "zeolite-note.md"
    assert artifact["content"].startswith("Ion exchange") and artifact["s3_key"].startswith("projects/")


def test_list_and_get_are_scoped_to_the_project():
    executed = []

    class _Cur:
        def execute(self, sql, params):
            executed.append((sql, params))

        def fetchall(self):
            return [("art-1", "document", "projects/p/k/abc", "note.md", "Zirconia note")]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    conn = SimpleNamespace(cursor=lambda: _Cur(), rollback=lambda: None)
    pool = SimpleNamespace(getconn=lambda: conn, putconn=lambda _c: None)
    store = rag_spine.SpineKnowledgeStore(pool=lambda: pool).for_project("p")
    [row] = store.list_artifacts()
    assert row == {
        "artifact_id": "art-1",
        "kind": "document",
        "content": "Zirconia note",
        "filename": "note.md",
        "summary": "Zirconia note",
        "s3_key": "projects/p/k/abc",
    }
    assert store.get_artifact("art-1")["artifact_id"] == "art-1"
    assert executed[0][1] == ("p",) and executed[1][1] == ("p", "art-1")
    assert "k.project_id = %s" in executed[0][0]


# ── the semantic lane, through the real retrieval code ──────────────────────


class _PgConn:
    """psycopg2's shape: ``with conn`` ends the transaction on exit, and a
    closed connection raises."""

    def __init__(self, answers):
        self.answers = answers
        self.closed = False
        self._rows = []

    def cursor(self):
        return self

    def execute(self, sql, _params=None):
        self._rows = next(rows for marker, rows in self.answers if marker in sql)

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        if self.closed:
            raise RuntimeError("connection already closed")
        return False


class _ClosingPool:
    """psycopg2's ThreadedConnectionPool past ``minconn``: putconn CLOSES."""

    def __init__(self, answers):
        self.answers = answers

    def getconn(self):
        return _PgConn(self.answers)

    def putconn(self, conn):
        conn.closed = True


def test_search_takes_the_semantic_lane(monkeypatch):
    """Live 2026-10-09: every probe failed with "connection already closed", and
    the loop's evidence came from the list-every-artifact gap fill instead. The
    retrieval code wrote ``with pool.getconn() as conn: try … finally:
    putconn(conn)`` — the transaction ended after the connection was returned,
    and psycopg2's pool closes a returned connection whenever it already holds
    ``minconn`` idle ones (always, at minconn=0; under any concurrency on the
    shared minconn=1 pool)."""
    note = {"artifact_id": "art-1", "text": "Ion exchange with sodium raises its CO2 uptake."}
    pool = _ClosingPool(
        [
            ("to_regclass", [("knowledge_node_embeddings",)]),
            ("<=>", [("node-chunk-art-1-0", "e5-base-v2", 768, None, "chunk", "p", 0.1, "{}")]),
            ("ANY(%s)", [("node-chunk-art-1-0", json.dumps(note), "s3/ref", "zeolite-note.md", "")]),
        ]
    )
    monkeypatch.setattr(knowledge_query, "_query_encoder_url", lambda: "http://rag")
    monkeypatch.setattr(knowledge_query, "_encode_query", lambda _q: ("e5-base-v2", [0.1], ""))
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: lambda _ref: None)
    store = rag_spine.SpineKnowledgeStore(pool=lambda: pool).for_project("p")
    [artifact] = store.search_artifacts("sodium exchange ZSM-5")
    assert (artifact["artifact_id"], artifact["filename"]) == ("art-1", "zeolite-note.md")


def test_the_default_pool_keeps_connections_open_across_putconn(monkeypatch):
    from psycopg2.pool import ThreadedConnectionPool
    from umrs.jobs import knowledge_adapters

    monkeypatch.setattr(rag_spine, "_KNOWLEDGE_POOL", None)
    monkeypatch.setattr(knowledge_adapters, "_resolve_knowledge_dsn", lambda: "dbname=x")
    pool = rag_spine._knowledge_pool()
    assert isinstance(pool, ThreadedConnectionPool) and rag_spine._knowledge_pool() is pool
