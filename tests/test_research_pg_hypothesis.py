"""Backend conformance tests for hypothesis lifecycle + closed loop (K1 tail).

Every behavioral test runs against the in-memory reference and the Postgres
implementation (migration 0061). PG tests skip when no database is reachable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.hypothesis_lifecycle import (
    HypothesisLifecycle,
    HypothesisStatus,
)

_MIGRATION = (
    Path(__file__).resolve().parents[1] / "src/umrs/migrations/0061_hypothesis_lifecycle.sql"
)


def _pg_pool_or_skip():
    pytest.importorskip("psycopg2")
    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(_MIGRATION.read_text())
            cur.execute("TRUNCATE hypothesis_history, hypotheses CASCADE")
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture(params=["memory", "pg"])
def lifecycle(request):
    if request.param == "memory":
        return HypothesisLifecycle()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    from umrs.hypothesis_lifecycle_pg import PgHypothesisLifecycle

    return PgHypothesisLifecycle(pool)


def test_propose_get_round_trip(lifecycle):
    hid = lifecycle.propose("Transformers generalize", planner_version="v1")
    hyp = lifecycle.get(hid)
    assert hyp is not None
    assert hyp.statement == "Transformers generalize"
    assert hyp.status == HypothesisStatus.PROPOSED
    assert hyp.originating_planner_version == "v1"
    assert lifecycle.get("missing") is None


def test_transition_validates_and_logs_history(lifecycle):
    hid = lifecycle.propose("H1")
    lifecycle.transition(hid, HypothesisStatus.TESTING)
    lifecycle.transition(hid, HypothesisStatus.SUPPORTED)
    assert lifecycle.get(hid).status == HypothesisStatus.SUPPORTED
    history = lifecycle.get_history(hid)
    assert [s for s, _ in history] == [
        HypothesisStatus.PROPOSED,
        HypothesisStatus.TESTING,
        HypothesisStatus.SUPPORTED,
    ]


def test_transition_invalid_raises(lifecycle):
    hid = lifecycle.propose("H1")
    with pytest.raises(ValueError, match="Invalid transition"):
        lifecycle.transition(hid, HypothesisStatus.ACCEPTED)


def test_transition_missing_raises_key_error(lifecycle):
    with pytest.raises(KeyError):
        lifecycle.transition("missing", HypothesisStatus.TESTING)


def test_add_evidence_updates_confidence(lifecycle):
    hid = lifecycle.propose("H1")
    lifecycle.add_evidence(hid, "evd-1", supports=True)
    lifecycle.add_evidence(hid, "evd-2", supports=True)
    lifecycle.add_evidence(hid, "evd-3", supports=False)
    hyp = lifecycle.get(hid)
    assert hyp.supporting_evidence == ["evd-1", "evd-2"]
    assert hyp.contradicting_evidence == ["evd-3"]
    assert hyp.confidence == pytest.approx(2 / 3)


def test_add_evidence_missing_raises(lifecycle):
    with pytest.raises(KeyError):
        lifecycle.add_evidence("missing", "evd", True)


def test_list_hypotheses_filters_status(lifecycle):
    lifecycle.propose("H1")
    h2 = lifecycle.propose("H2")
    lifecycle.transition(h2, HypothesisStatus.TESTING)
    assert len(lifecycle.list_hypotheses()) == 2
    only_testing = lifecycle.list_hypotheses(status=HypothesisStatus.TESTING)
    assert [h.hypothesis_id for h in only_testing] == [h2]


def test_refine_creates_linked_child(lifecycle):
    parent = lifecycle.propose("vague claim")
    child = lifecycle.refine(parent, "precise claim")
    assert lifecycle.get(parent).status == HypothesisStatus.REFINED
    child_hyp = lifecycle.get(child)
    assert child_hyp.parent_hypothesis_id == parent
    assert child_hyp.statement == "precise claim"
    parent_history = [s for s, _ in lifecycle.get_history(parent)]
    assert parent_history == [HypothesisStatus.PROPOSED, HypothesisStatus.REFINED]


def test_refine_missing_parent_raises(lifecycle):
    with pytest.raises((KeyError, Exception)):
        lifecycle.refine("missing", "child")


# ── Closed-loop durability wiring ────────────────────────────────────────


def test_closed_loop_defaults_in_memory_without_db(monkeypatch):
    from umrs.closed_loop_learning import ClosedLoopLearning
    from umrs.hypothesis_lifecycle import HypothesisLifecycle
    from umrs.knowledge_graph_api import InMemoryKnowledgeGraph
    from umrs.scientific_memory import InMemoryScientificMemoryStore

    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("UMRS_SCIENTIFIC_MEMORY_STORE_BACKEND", raising=False)
    monkeypatch.delenv("UMRS_KNOWLEDGE_GRAPH_STORE_BACKEND", raising=False)
    monkeypatch.delenv("UMRS_HYPOTHESIS_LIFECYCLE_STORE_BACKEND", raising=False)
    loop = ClosedLoopLearning()
    assert isinstance(loop._memory, InMemoryScientificMemoryStore)
    assert isinstance(loop._graph, InMemoryKnowledgeGraph)
    assert isinstance(loop._hypothesis_lifecycle, HypothesisLifecycle)


def test_closed_loop_honors_explicit_memory_backend(monkeypatch):
    from umrs.closed_loop_learning import ClosedLoopLearning
    from umrs.hypothesis_lifecycle import HypothesisLifecycle
    from umrs.knowledge_graph_api import InMemoryKnowledgeGraph
    from umrs.scientific_memory import InMemoryScientificMemoryStore

    monkeypatch.setenv("UMRS_SCIENTIFIC_MEMORY_STORE_BACKEND", "memory")
    monkeypatch.setenv("UMRS_KNOWLEDGE_GRAPH_STORE_BACKEND", "memory")
    monkeypatch.setenv("UMRS_HYPOTHESIS_LIFECYCLE_STORE_BACKEND", "memory")
    loop = ClosedLoopLearning()
    assert isinstance(loop._memory, InMemoryScientificMemoryStore)
    assert isinstance(loop._graph, InMemoryKnowledgeGraph)
    assert isinstance(loop._hypothesis_lifecycle, HypothesisLifecycle)
