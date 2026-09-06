"""_RAGSearchBackend pins (WP-15, ADR-027 P1b facade leg).

The facade's ``rag`` scope answers from the node-keyed central corpus
(``knowledge_nodes``, migration 0055) — never the endpoint-coupled
mini-RiVault corpus. These pins lock: project scoping + ordering SQL,
honest hit shape (venue + lane in metadata, fixed score, never
fabricated content), wiring gates (regclass probe), and the unified
composition reporting the backend as wired.
"""

from __future__ import annotations

from datetime import datetime, timezone

from umrs.api.knowledge_query import (
    _build_unified_search,
    _node_snippet,
    _rag_backend_for,
    _RAGSearchBackend,
)


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


_ROWS = [
    (
        "ragdoc-doc-1-chunk0",
        "chunk 0 of scan.h5",
        "entity",
        '{"text": "detector calibration constants for run 12"}',
        "ab" * 32,
        datetime(2026, 9, 7, tzinfo=timezone.utc),
    ),
    (
        "ragdoc-doc-2",
        "run.md",
        "file",
        '{"filename": "run.md"}',
        "cd" * 32,
        datetime(2026, 9, 6, tzinfo=timezone.utc),
    ),
]


def _rag_hits(limit=5):
    pool = _Pool(lambda _sql, _params: _ROWS)
    backend = _RAGSearchBackend(pool)
    return pool, backend.search_rag("calibration", project_id="spring8", limit=limit)


def test_backend_sql_is_project_scoped_corpus_read():
    pool, hits = _rag_hits()
    assert len(hits) == 2
    sql, params = pool._conn._cur.queries[0]
    assert "knowledge_nodes n" in sql
    assert params[0] == "spring8" and params[-1] == 5


def test_hit_carries_central_venue_and_lane():
    _, hits = _rag_hits()
    first = hits[0]
    assert first["source_type"] == "knowledge_corpus_node"
    meta = first["metadata"]
    assert meta["lane"] == "central-corpus"
    assert meta["corpus_venue"] == "central-pg"
    assert meta["node_type"] == "entity"
    assert first["score"] == 0.5  # honest fixed score — no relevance model


def test_hit_snippet_prefers_text_field():
    _, hits = _rag_hits()
    assert hits[0]["content"] == "detector calibration constants for run 12"
    # second node has no text-ish key -> honest content-ref pointer
    assert hits[1]["content"].startswith("(content ref:")


def test_node_snippet_never_fabricates():
    assert _node_snippet("", "") == "(no snippet)"
    assert _node_snippet('{"x": 1}', "") == "(no snippet)"
    assert _node_snippet("not json raw text", "").startswith("not json")
    assert _node_snippet('{"preview": "p"}', "") == "p"


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
    pool = _Pool(lambda _sql, _params: _ROWS)
    backend = _RAGSearchBackend(pool)
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
