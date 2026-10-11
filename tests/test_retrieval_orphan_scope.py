"""K3 and the retrieval lanes never read an orphan chunk; K3 hits carry their
full text up to a server cap (ADR-093 D-11, wave F, 2026-10-11).

- K3's graph lane ran its LIKE over every node, chunks included: two orphan
  chunks of the stand-alone stack (node-chunk-art-d7a806cc4b75-0,
  node-chunk-art-f262129bf8a4-0) were in its top 20 for "zeolite CO2 uptake".
  Chunk nodes are the corpus lanes' now.
- The corpus lanes read a chunk the embed job wrote only while its artifact
  row exists. Chunks of other writers (rag.upload minting, the legacy
  canonizer) carry ``document_id``, not ``artifact_id``, and pass unchanged.
  Dense legacy mode (acceptance receipt 4) keeps its old statement.
- K3's lexical and graph lanes clipped a hit to 240/200 characters with no
  flag, so the caller's per-hit budget could not reach its text.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest
from umrs.api import knowledge_query as kq
from umrs.knowledge.access_scope import artifact_alive_predicate
from umrs.knowledge.retrieval import calibrate, dense, lexical
from umrs.knowledge.retrieval.types import Candidate, RetrieveSpec

ALIVE = artifact_alive_predicate("n")


def _spec(**kw) -> RetrieveSpec:
    return RetrieveSpec(queries=("zeolite",), project_id="p", **kw)


# ── the predicate, evaluated ────────────────────────────────────────────────


def _visible(rows: list[tuple[str, str, dict]], artifacts: list[tuple[str, str]]) -> list[str]:
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE knowledge_nodes (node_id TEXT, project_id TEXT, content TEXT)")
    db.execute("CREATE TABLE knowledge_artifacts (artifact_id TEXT, project_id TEXT)")
    db.executemany(
        "INSERT INTO knowledge_nodes VALUES (?, ?, ?)",
        [(nid, project, json.dumps(content)) for nid, project, content in rows],
    )
    db.executemany("INSERT INTO knowledge_artifacts VALUES (?, ?)", artifacts)
    found = db.execute(f"SELECT node_id FROM knowledge_nodes n WHERE {ALIVE} ORDER BY 1")
    return [row[0] for row in found]


def test_an_embed_chunk_is_read_only_while_its_artifact_row_exists():
    embed = {"kind": "chunk", "origin": "knowledge_embed"}
    rows = [
        ("live", "p", {**embed, "artifact_id": "art-1"}),
        ("orphan", "p", {**embed, "artifact_id": "art-gone"}),
        ("elsewhere", "p", {**embed, "artifact_id": "art-q"}),  # row in another project
    ]
    assert _visible(rows, [("art-1", "p"), ("art-q", "q")]) == ["live"]


def test_chunks_of_other_writers_pass_unchanged():
    """rag.upload minting and the legacy canonizer key chunks by document_id."""
    rows = [
        ("rag", "p", {"kind": "chunk", "origin": "rag.upload", "document_id": "d-1"}),
        ("legacy", "p", {"kind": "chunk", "document_id": "d-2"}),
        ("file", "p", {"artifact_id": "art-gone"}),
    ]
    assert _visible(rows, []) == ["file", "legacy", "rag"]


# ── where it applies ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec",
    [_spec(), _spec(graph_ids=("g-1",)), _spec(exclude_graph_ids=("g-2",))],
)
def test_the_dense_and_lexical_scopes_read_only_live_chunks(spec):
    dense_where, _ = dense.scope_clause(spec, "/m/e5-base-v2")
    lexical_where, _ = lexical.scope_predicate(spec)
    assert ALIVE in dense_where and ALIVE in lexical_where


def test_the_calibrated_null_and_the_graph_hop_share_the_dense_scope():
    """calibrate and graph_lane build their WHERE from dense.scope_clause."""
    import inspect

    from umrs.knowledge.retrieval import graph_lane

    assert "scope_clause(spec, model)" in inspect.getsource(calibrate.scope_signature)
    assert "scope_clause(spec, model)" in inspect.getsource(calibrate.probe_cosines)
    assert "scope_clause(spec, model)" in inspect.getsource(graph_lane)


def test_dense_legacy_mode_keeps_its_old_statement():
    """Acceptance receipt 4: legacy mode returns exactly the pre-ADR-093 ids."""
    sql = dense._legacy_sql(_spec(threshold=0.5))
    assert "knowledge_artifacts" not in sql and "origin" not in sql


# ── K3 ───────────────────────────────────────────────────────────────────────


def test_k3s_graph_lane_reads_no_chunk_node():
    sql = kq._graph_lane_sql("(n.project_id = %s)", 2)
    haystack = sql.split("), m AS", 1)[0]
    assert "(content->>'kind') IS DISTINCT FROM 'chunk'" in haystack


def test_k3s_embeddings_lane_reads_only_live_chunks():
    backend = kq._EmbeddingsSearchBackend(pool=None)
    gate, params = backend._gate("p")
    assert ALIVE in gate and gate.count("%s") == len(params)
    backend._graph_scope = SimpleNamespace(owned_sql=lambda _a: "TRUE", params=lambda: ())
    gate, _ = backend._gate("p")
    assert ALIVE in gate


# ── K3 hit text: full, bounded, flagged ─────────────────────────────────────


def _lexical(text: str) -> dict:
    candidate = Candidate(node_id="node-chunk-art-1-0", artifact_id="art-1", content={})
    candidate.text = text
    candidate.lane_scores["lex"] = 1.0
    return kq._lexical_hit(candidate, "lex")


def _graph(content: str) -> dict:
    backend = kq._GraphSearchBackend.__new__(kq._GraphSearchBackend)
    backend._by_id, backend._default_by_project = {}, {}
    return backend._hit(1, ("node-sem-1", "p", "zeolite", "concept", content, "g", 1))


def test_a_lexical_hit_carries_its_full_chunk_text():
    body = "zeolite " * 100  # 800 characters: clipped to 240 before
    hit = _lexical(body)
    assert hit["content"] == body and hit["metadata"]["clipped"] is False


def test_a_graph_hit_carries_its_full_text():
    content = "x" * 700  # clipped to 200 before
    hit = _graph(content)
    assert hit["content"] == content and hit["metadata"]["clipped"] is False


def test_the_server_cap_bounds_a_hit_and_says_so(monkeypatch):
    monkeypatch.setenv(kq.HIT_TEXT_MAX_ENV, "50")
    lexical_hit, graph_hit = _lexical("y" * 80), _graph("z" * 80)
    assert lexical_hit["content"] == "y" * 50 and lexical_hit["metadata"]["clipped"] is True
    assert graph_hit["content"] == "z" * 50 and graph_hit["metadata"]["clipped"] is True


def test_the_cap_survives_the_search_api_conversion():
    from umrs.search_api import SearchResultType, UnifiedSearch

    (result,) = UnifiedSearch._convert_results(
        None, [_lexical("w" * 300)], SearchResultType.RAG_RESULT
    )
    assert len(result.content) == 300 and result.metadata["clipped"] is False
