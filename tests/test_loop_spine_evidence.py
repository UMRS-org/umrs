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

from umrs.api import rag_retrieve, rag_spine
from umrs.knowledge.loop import AutonomousResearchLoop
from umrs.knowledge.retrieval import hydrate as hydrate_mod

from r4_contracts.retrieval import RetrievalSufficiency, RetrievedPassage, RetrieveResult


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


def _result(status, passages=()):
    return RetrieveResult(
        query="q",
        passages=tuple(passages),
        sufficiency=RetrievalSufficiency(status=status, basis="z_null", value=1.0, threshold=2.0),
        searched=40,
    )


def _passage(rank, artifact_id="art-1"):
    return RetrievedPassage(
        rank=rank,
        source_id=f"node-chunk-{artifact_id}-{rank}",
        artifact_id=artifact_id,
        source_name="zeolite-note.md",
        text="Ion exchange with sodium raises its CO2 uptake.",
    )


class _RefPool:
    """Answers the store's one source_ref lookup."""

    def __init__(self):
        self.sql = []

    def getconn(self):
        pool = self

        class _Cur:
            def execute(self, sql, params=None):
                pool.sql.append((sql, params))

            def fetchall(self):
                return [("art-1", "projects/p/knowledge/artifacts/abc")]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        return _Conn(_Cur)

    def putconn(self, _conn):
        pass


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_search_returns_core_passages_as_loop_artifacts(monkeypatch):
    """ADR-093 D-2: the store reads the retrieval core — k_max 4, adaptive."""
    from umrs.knowledge.retrieval import pipeline

    seen = {}

    def retrieve(pool, spec, **kwargs):
        seen.update(spec=spec)
        return _result("sufficient", [_passage(1)])

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: lambda _ref: None)
    pool = _RefPool()
    store = rag_spine.SpineKnowledgeStore(pool=lambda: pool, encode=lambda _t: ("m", None, "x"))
    [artifact] = store.for_project("p").search_artifacts("sodium exchange ZSM-5")
    spec = seen["spec"]
    assert (spec.queries, spec.project_id, spec.k_max, spec.threshold) == (
        ("sodium exchange ZSM-5",),
        "p",
        4,
        None,
    )
    assert artifact["artifact_id"] == "art-1" and artifact["filename"] == "zeolite-note.md"
    assert artifact["content"].startswith("Ion exchange") and artifact["s3_key"].startswith(
        "projects/"
    )


def test_an_insufficient_verdict_is_no_evidence(monkeypatch):
    from umrs.knowledge.retrieval import pipeline

    monkeypatch.setattr(
        pipeline, "retrieve", lambda *_a, **_k: _result("insufficient", [_passage(1)])
    )
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: lambda _ref: None)
    store = rag_spine.SpineKnowledgeStore(
        pool=lambda: _RefPool(), encode=lambda _t: ("m", None, "x")
    )
    assert store.for_project("p").search_artifacts("best pizza in Naples") == []


def test_a_partial_verdict_is_at_most_two_passages(monkeypatch):
    from umrs.knowledge.retrieval import pipeline

    passages = [_passage(i) for i in (1, 2, 3)]
    monkeypatch.setattr(pipeline, "retrieve", lambda *_a, **_k: _result("partial", passages))
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: lambda _ref: None)
    store = rag_spine.SpineKnowledgeStore(
        pool=lambda: _RefPool(), encode=lambda _t: ("m", None, "x")
    )
    assert len(store.for_project("p").search_artifacts("zeolite")) == 2


def test_an_unrelated_goal_yields_no_evidence_not_the_newest_artifacts():
    """MD-6 (ADR-093 D-2): the gap fill returned the five most recent
    artifacts when nothing matched, and the loop cited recency as evidence."""
    newest = [
        {"artifact_id": f"a{i}", "filename": "zeolite-note.md", "summary": "zeolite uptake"}
        for i in range(6)
    ]
    store = SimpleNamespace(list_artifacts=lambda: newest)
    engine = AutonomousResearchLoop(knowledge_store=store)
    assert engine._gap_fill_from_store("perovskite encapsulation", store) == []
    assert engine._gap_fill_from_store("", store) == []
    assert len(engine._gap_fill_from_store("zeolite uptake", store)) == 6
    assert len(engine._gap_fill_from_store("zeolite", store)) == 6  # one word, one hit


def _cohort_artifacts(n=200):
    summary = "The study measured outcomes in the cohort."
    return [{"artifact_id": f"a{i}", "summary": summary, "content": summary} for i in range(n)]


def test_a_stop_word_never_makes_an_artifact_evidence():
    """Review of ADR-093 (2026-10-10): 'the' matched as a substring of every
    excerpt, so a plain-sentence goal took the newest 200 artifacts."""
    store = SimpleNamespace(list_artifacts=_cohort_artifacts)
    engine = AutonomousResearchLoop(knowledge_store=store)
    assert engine._gap_fill_from_store("What is the best pizza in Naples?", store) == []


def test_the_gap_fill_matches_word_starts_and_wants_two_of_them():
    training = [{"artifact_id": "a1", "summary": "Training runs of zeolites under heat."}]
    store = SimpleNamespace(list_artifacts=lambda: training)
    engine = AutonomousResearchLoop(knowledge_store=store)
    assert engine._gap_fill_from_store("rain heat", store) == []  # 'rain' inside 'training'
    assert engine._gap_fill_from_store("pizza heat", store) == []  # one of two words
    assert engine._gap_fill_from_store("zeolite heat", store) == training  # 'zeolites'


def test_the_spine_stores_insufficient_verdict_is_never_overridden(monkeypatch):
    """The spine store returns [] when the scorer says nothing answers the
    goal; the loop's token gap fill must not turn that into evidence."""
    pizza = [{"artifact_id": "a1", "summary": "best pizza in Naples", "content": "pizza Naples"}]
    monkeypatch.setattr(
        rag_spine.SpineKnowledgeStore, "search_artifacts", lambda _self, _q, _t=None: []
    )
    monkeypatch.setattr(
        rag_spine.SpineKnowledgeStore, "list_artifacts", lambda _self, _t=None: pizza
    )
    engine = AutonomousResearchLoop(knowledge_store=rag_spine.SpineKnowledgeStore())
    loop = SimpleNamespace(
        project_id="p", research_goal="best pizza in Naples", plan=None, current_iteration=1
    )
    refs, excerpts = engine._search_knowledge(loop, SimpleNamespace(retrieval_record_ids=[]))
    assert (refs, excerpts) == ([], [])


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
    shared minconn=1 pool). Since ADR-093 the store reads the retrieval core,
    whose every statement ends its transaction before ``putconn``."""
    note = {
        "kind": "chunk",
        "artifact_id": "art-1",
        "chunk_index": 0,
        "text": "Ion exchange with sodium raises its CO2 uptake.",
    }
    pool = _ClosingPool(
        [
            ("information_schema", []),
            ("<=>", [("node-chunk-art-1-0", json.dumps(note), 0.1, 1)]),
            (
                "LEFT JOIN",
                [
                    (
                        "node-chunk-art-1-0",
                        "g",
                        json.dumps(note),
                        "s3/ref",
                        "zeolite-note.md",
                        "",
                        None,
                    )
                ],
            ),
            ("GROUP BY", []),
            ("FROM knowledge_artifacts WHERE", [("art-1", "s3/ref")]),
        ]
    )
    monkeypatch.setattr(hydrate_mod, "_SEARCH_TEXT", None)
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: lambda _ref: None)
    encode = lambda texts: ("e5-base-v2", [[0.1] for _ in texts], "")  # noqa: E731
    store = rag_spine.SpineKnowledgeStore(pool=lambda: pool, encode=encode).for_project("p")
    [artifact] = store.search_artifacts("sodium exchange ZSM-5")
    assert (artifact["artifact_id"], artifact["filename"]) == ("art-1", "zeolite-note.md")


def test_the_default_pool_keeps_connections_open_across_putconn(monkeypatch):
    from psycopg2.pool import ThreadedConnectionPool
    from umrs.jobs import knowledge_adapters

    monkeypatch.setattr(rag_spine, "_KNOWLEDGE_POOL", None)
    monkeypatch.setattr(knowledge_adapters, "_resolve_knowledge_dsn", lambda: "dbname=x")
    pool = rag_spine._knowledge_pool()
    assert isinstance(pool, ThreadedConnectionPool) and rag_spine._knowledge_pool() is pool
