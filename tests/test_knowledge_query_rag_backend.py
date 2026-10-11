"""_RAGSearchBackend pins (WP-15, ADR-027 P1b facade leg; ADR-093 C7).

The facade's ``rag`` scope answers from the node-keyed central corpus
(``knowledge_nodes``, migration 0055) — never the endpoint-coupled
mini-RiVault corpus. Since ADR-093 C7 the lane is the retrieval core's
lexical chunk lane (IDF-weighted OR full-text search, C6) instead of a
whole-phrase ILIKE ordered by recency with a fixed 0.5 score. These pins lock:
the lane call and its spec (project, graph selection), the hit shape (venue +
lane in metadata, the lane's own score, never fabricated content), honest
unavailability when the core or its column is missing, wiring gates
(regclass probe), and the unified composition reporting the backend as wired.
"""

from __future__ import annotations

import sys
import types

import pytest
from umrs.api.knowledge_query import (
    _build_unified_search,
    _GraphScope,
    _rag_backend_for,
    _RAGSearchBackend,
)
from umrs.knowledge.retrieval.types import Candidate, LaneResult

_LEXICAL = "umrs.knowledge.retrieval.lexical"


class _Cur:
    def __init__(self, behavior) -> None:
        self.queries = []
        self._behavior = behavior

    def execute(self, sql, params=None):
        self.queries.append((sql, params))

    def fetchall(self):
        return self._behavior(None, None)

    def fetchone(self):
        return self._behavior(None, None)[0] if self._behavior(None, None) else (None,)

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


class _UntouchedPool:
    """The backend hands the pool to the lane; it runs no SQL of its own."""

    def getconn(self):
        raise AssertionError("the rag backend must not query the pool itself")


def _candidates() -> list[Candidate]:
    return [
        Candidate(
            node_id="node-chunk-art-1-0",
            artifact_id="art-1",
            chunk_index=0,
            content={"kind": "chunk", "excerpt": "detector calibration constants for run 12"},
            lane_scores={"lex": 7.5},
        ),
        Candidate(
            node_id="node-chunk-art-2-3",
            artifact_id="art-2",
            chunk_index=3,
            content={"kind": "chunk", "artifact_id": "art-2"},
            lane_scores={"lex": 3.25},
            text="calibration of the beamline monochromator",
        ),
    ]


def _install_lane(monkeypatch, result=None, *, raises=None) -> list[tuple]:
    calls: list[tuple] = []
    module = types.ModuleType(_LEXICAL)

    def lexical_lane(pool, query, spec, *, limit=50):
        calls.append((pool, query, spec, limit))
        if raises is not None:
            raise raises
        return result

    module.lexical_lane = lexical_lane
    monkeypatch.setitem(sys.modules, _LEXICAL, module)
    return calls


def _ok_lane() -> LaneResult:
    return LaneResult(name="lex", candidates=_candidates())


def test_the_rag_scope_runs_the_lexical_chunk_lane(monkeypatch):
    calls = _install_lane(monkeypatch, _ok_lane())
    pool = _UntouchedPool()
    hits = _RAGSearchBackend(pool).search_rag("calibration", project_id="spring8", limit=5)
    assert [h["source_id"] for h in hits] == ["node-chunk-art-1-0", "node-chunk-art-2-3"]
    lane_pool, query, _spec, limit = calls[0]
    assert (lane_pool, query, limit) == (pool, "calibration", 5)


def test_the_lane_spec_is_the_project_under_the_legacy_rule(monkeypatch):
    calls = _install_lane(monkeypatch, _ok_lane())
    _RAGSearchBackend(_UntouchedPool()).search_rag("calibration", project_id="spring8")
    spec = calls[0][2]
    assert (spec.queries, spec.project_id) == (("calibration",), "spring8")
    assert (spec.graph_ids, spec.include_ambient) == (None, False)


def test_hits_carry_the_lane_score_venue_and_chunk(monkeypatch):
    _install_lane(monkeypatch, _ok_lane())
    first, second = _RAGSearchBackend(_UntouchedPool()).search_rag("calibration", project_id="p")
    assert first["score"] == 7.5 and second["score"] == 3.25  # IDF sums, not a fixed 0.5
    assert first["source_type"] == "knowledge_corpus_node"
    assert first["title"] == "chunk 0 of art-1"
    assert first["metadata"] == {
        "lane": "lexical",
        "corpus_venue": "central-pg",
        "artifact_id": "art-1",
        "chunk_index": 0,
        "clipped": False,  # wave F: the full chunk text up to the server cap
    }


def test_hit_text_prefers_the_hydrated_chunk_then_the_excerpt(monkeypatch):
    _install_lane(monkeypatch, _ok_lane())
    first, second = _RAGSearchBackend(_UntouchedPool()).search_rag("calibration", project_id="p")
    assert first["content"] == "detector calibration constants for run 12"
    assert second["content"] == "calibration of the beamline monochromator"


def test_the_lane_report_and_candidates_are_kept_for_the_response(monkeypatch):
    _install_lane(monkeypatch, _ok_lane())
    backend = _RAGSearchBackend(_UntouchedPool())
    backend.search_rag("calibration", project_id="p")
    assert backend.lane_report == {"lane": "lexical", "status": "ok", "n": 2}
    assert [c.node_id for c in backend.lexical_candidates] == [
        "node-chunk-art-1-0",
        "node-chunk-art-2-3",
    ]


def test_a_graph_selection_reaches_the_lane(monkeypatch):
    calls = _install_lane(monkeypatch, _ok_lane())
    scope = _GraphScope(graph_ids=("g-exp", "g-def"), default_projects=("p",))
    backend = _RAGSearchBackend(_UntouchedPool(), include_ambient=True, graph_scope=scope)
    backend.search_rag("q", project_id="p")
    spec = calls[0][2]
    assert spec.graph_ids == ("g-exp", "g-def") and spec.include_ambient is True


def test_a_missing_lexical_module_is_reported_never_answered(monkeypatch):
    monkeypatch.setitem(sys.modules, _LEXICAL, None)
    backend = _RAGSearchBackend(_UntouchedPool())
    assert backend.search_rag("q", project_id="p") == []
    assert backend.lane_report["status"] == "unavailable"
    assert "lexical lane not installed" in backend.lane_report["reason"]
    assert backend.lexical_candidates is None


@pytest.mark.parametrize(
    ("status", "reason"),
    [("unavailable", "knowledge_nodes.search_text missing (migration 0120)"), ("off", "")],
)
def test_a_lane_that_did_not_run_returns_nothing_and_says_why(monkeypatch, status, reason):
    _install_lane(monkeypatch, LaneResult(name="lex", status=status, reason=reason))
    backend = _RAGSearchBackend(_UntouchedPool())
    assert backend.search_rag("q", project_id="p") == []
    assert backend.lane_report["status"] == status
    assert backend.lane_report.get("reason", "") == reason
    assert backend.lexical_candidates is None


def test_a_failing_lane_is_reported_not_raised(monkeypatch):
    _install_lane(monkeypatch, raises=RuntimeError("relation does not exist"))
    backend = _RAGSearchBackend(_UntouchedPool())
    assert backend.search_rag("q", project_id="p") == []
    assert backend.lane_report["reason"] == "lexical lane failed (RuntimeError)"


def test_the_page_is_capped_at_the_limit(monkeypatch):
    many = [Candidate(node_id=f"n{i}", lane_scores={"lex": 10.0 - i}) for i in range(9)]
    _install_lane(monkeypatch, LaneResult(name="lex", candidates=many))
    hits = _RAGSearchBackend(_UntouchedPool()).search_rag("q", project_id="p", limit=4)
    assert [h["source_id"] for h in hits] == ["n0", "n1", "n2", "n3"]
    assert hits[0]["title"] == "n0"  # no artifact: the node id, never a made-up name


def test_wiring_gated_on_table_presence():
    pool = _Pool(lambda _sql, _params: [("knowledge_nodes",)])
    backend, reason = _rag_backend_for(pool)
    assert backend is not None and reason is None
    assert pool.returned == 1


def test_wiring_reports_reason_when_table_missing():
    pool = _Pool(lambda _sql, _params: [None])
    backend, reason = _rag_backend_for(pool)
    assert backend is None
    assert "corpus unavailable" in (reason or "")


def test_wiring_reports_reason_without_pool():
    backend, reason = _rag_backend_for(None)
    assert backend is None
    assert "requires the database" in (reason or "")


class _StubStore:
    def search_catalog(self, *_a, **_k):
        return []


def test_unified_search_marks_rag_wired():
    backend = _RAGSearchBackend(_UntouchedPool())
    _, wired, unwired = _build_unified_search(
        _StubStore(), None, "no pool", None, None, backend, None
    )
    assert "rag" in wired
    assert "rag" not in unwired


def test_unified_search_keeps_honesty_reason_when_unwired():
    _, wired2, unwired2 = _build_unified_search(
        _StubStore(), None, "no pool", None, None, None, "corpus probe failed"
    )
    assert "rag" not in wired2
    assert unwired2["rag"] == "corpus probe failed"


def test_unwired_backends_default_truth_central_corpus():
    from umrs.api.knowledge_query import _UNWIRED_BACKENDS

    # The default honesty reason now names the central corpus — not the
    # retired "requires a deployed mini-rivault instance".
    assert "mini-rivault" not in _UNWIRED_BACKENDS["rag"]
    assert "central knowledge corpus" in _UNWIRED_BACKENDS["rag"]
