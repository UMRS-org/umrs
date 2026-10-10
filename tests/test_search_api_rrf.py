"""ADR-093 C7: K3 ``scope=all`` fused by weighted RRF, and the catalogue floor.

Before C7, ``UnifiedSearch.search`` sorted raw scores across lanes that do
not share a scale (a catalogue token count of 1.25-4.0, a cosine of 0.7-1.0,
a fixed 0.5 or 0.6/0.9), so catalogue rows and title matches outranked
passages (KGC-03); ``total_results`` was the length of the cut page; and every
READY catalogue row scored +0.25 without matching anything, so a nonsense
query returned rows and ``knowledge_insufficient`` never fired (LIVE-06).
"""

from __future__ import annotations

import pytest
from fake_pg_pool import FakePool
from umrs.data_catalog.pg_store import CatalogMetaStore
from umrs.data_catalog.schema import CatalogObjectMeta, CatalogState
from umrs.data_catalog.scoring import score_meta, token_matches
from umrs.knowledge.adapters import UMRSSearchAdapter
from umrs.search_api import (
    LANE_WEIGHTS,
    RRF_K,
    SearchResultType,
    SearchScope,
    UnifiedSearch,
    catalog_quota,
)

# ── fakes ────────────────────────────────────────────────────────────────


def _hit(source_id: str, score: float, **extra) -> dict:
    return {
        "title": source_id,
        "content": f"text of {source_id}",
        "score": score,
        "source_id": source_id,
        "source_type": extra.pop("source_type", "node"),
        "metadata": extra,
    }


class _Lane:
    """A lane returning fixed hits under any of the four port method names."""

    def __init__(self, hits: list[dict]) -> None:
        self.hits = hits
        self.limits: list[int] = []

    def _search(self, _query: str, *, project_id=None, limit=20) -> list[dict]:
        del project_id
        self.limits.append(limit)
        return [dict(h, metadata=dict(h["metadata"])) for h in self.hits]

    search_catalog = search_knowledge_graph = search_embeddings = search_rag = _search


def _search(**lanes: list[dict]) -> UnifiedSearch:
    return UnifiedSearch(**{name: _Lane(hits) for name, hits in lanes.items()})


def _ids(results) -> list[str]:
    return [r.source_id for r in results.results]


# ── weighted RRF ─────────────────────────────────────────────────────────


def test_raw_scores_no_longer_decide_the_order_across_lanes():
    """A catalogue token count of 4.0 used to beat a passage at cosine 0.8."""
    search = _search(
        catalog=[_hit("s3://cat", 4.0)],
        knowledge_graph=[_hit("kg-node", 0.9)],
        embeddings=[_hit("dense-node", 0.8)],
        rag=[_hit("lex-node", 3.2)],
    )
    results = search.search("q", scope=SearchScope.ALL)
    # Every lane put its hit first, so the lane weights decide; equal weights
    # (embeddings, rag) keep the heavier-first fusion order.
    assert _ids(results) == ["dense-node", "lex-node", "kg-node", "s3://cat"]


def test_the_weights_and_k_are_the_adr_values():
    assert RRF_K == 60
    assert LANE_WEIGHTS == {
        SearchScope.EMBEDDINGS: 1.0,
        SearchScope.RAG: 1.0,
        SearchScope.KNOWLEDGE_GRAPH: 0.7,
        SearchScope.CATALOG: 0.5,
    }


def test_agreement_between_lanes_beats_one_lane_rank_one():
    search = _search(
        embeddings=[_hit("a", 0.90), _hit("b", 0.89), _hit("c", 0.88)],
        rag=[_hit("c", 7.0), _hit("x", 6.0)],
    )
    results = search.search("q", scope=SearchScope.ALL)
    # c: 1/63 + 1/61 > a: 1/61
    assert _ids(results)[0] == "c"


def test_a_source_two_lanes_found_is_one_result_naming_both_lanes():
    search = _search(
        embeddings=[_hit("n1", 0.91), _hit("n2", 0.85)],
        rag=[_hit("n2", 5.5), _hit("n3", 2.0)],
    )
    results = search.search("q", scope=SearchScope.ALL)
    assert sorted(_ids(results)) == ["n1", "n2", "n3"]
    merged = next(r for r in results.results if r.source_id == "n2")
    assert merged.metadata["lanes"] == ["embeddings", "rag"]
    assert merged.metadata["lane_score"] == {"embeddings": 0.85, "rag": 5.5}
    assert merged.metadata["lane_rank"] == {"embeddings": 2, "rag": 1}
    assert merged.metadata["rrf"] == pytest.approx(1 / 62 + 1 / 61)


def test_the_strongest_lane_speaks_for_a_merged_hit():
    search = _search(
        knowledge_graph=[_hit("n1", 1.0, source_type="knowledge_graph_node")],
        embeddings=[_hit("n0", 0.9), _hit("n1", 0.8, source_type="embedding")],
    )
    merged = next(
        r for r in search.search("q", scope=SearchScope.ALL).results if r.source_id == "n1"
    )
    # embeddings rank 2 (1/62) outweighs graph rank 1 (0.7/61)
    assert merged.result_type == SearchResultType.EMBEDDING_MATCH
    assert merged.source_type == "embedding"


def test_score_is_the_fused_value_over_the_most_attainable():
    search = _search(
        catalog=[_hit("s3://c", 2.0)],
        knowledge_graph=[_hit("n1", 1.0)],
        embeddings=[_hit("n1", 0.9)],
        rag=[_hit("n1", 3.0)],
    )
    results = search.search("q", scope=SearchScope.ALL)
    top = results.results[0]
    assert top.source_id == "n1"
    # rank 1 in three lanes of a four-lane search: 2.7 of 3.2 attainable
    assert top.score == pytest.approx(2.7 / 3.2, abs=1e-6)


def test_rank_one_in_every_lane_that_ran_scores_one():
    search = _search(embeddings=[_hit("n1", 0.7)], rag=[_hit("n1", 0.1)])
    assert search.search("q", scope=SearchScope.ALL).results[0].score == 1.0


def test_a_lane_counts_each_source_once_at_its_best_rank():
    search = _search(embeddings=[_hit("n1", 0.9), _hit("n1", 0.5), _hit("n2", 0.8)])
    results = search.search("q", scope=SearchScope.ALL)
    assert _ids(results) == ["n1", "n2"]
    assert results.results[0].metadata["lane_rank"] == {"embeddings": 1}


def test_hits_without_a_source_id_are_never_merged():
    search = _search(embeddings=[_hit("", 0.9), _hit("", 0.8)], rag=[_hit("", 1.0)])
    results = search.search("q", scope=SearchScope.ALL)
    assert results.total_results == 3


def test_fused_results_do_not_alias_the_lane_metadata():
    lane_meta = {"graph_id": "g-1"}
    search = _search(knowledge_graph=[_hit("n1", 1.0, **lane_meta)])
    result = search.search("q", scope=SearchScope.ALL).results[0]
    assert result.metadata["graph_id"] == "g-1"
    assert result.metadata["lanes"] == ["knowledge_graph"]


# ── catalogue quota and the page ─────────────────────────────────────────


@pytest.mark.parametrize(("limit", "quota"), [(1, 2), (4, 2), (10, 2), (20, 5), (100, 25)])
def test_catalog_quota_is_a_quarter_of_the_page_at_least_two(limit, quota):
    assert catalog_quota(limit) == quota


def _kinds(results) -> list[SearchResultType]:
    return [r.result_type for r in results.results]


def test_catalog_rows_cannot_crowd_graph_nodes_out_of_the_page():
    """Deep in a page, catalogue rank 1 (0.5/61) outranks graph rank 26+ (0.7/86):
    without the quota about 31 of 100 slots would be catalogue rows."""
    catalog = [_hit(f"s3://c{i}", 10.0 - i / 100) for i in range(100)]
    nodes = [_hit(f"n{i}", 0.9 - i / 1000) for i in range(100)]
    search = _search(catalog=catalog, knowledge_graph=nodes)
    results = search.search("q", scope=SearchScope.ALL, limit=100)
    assert len(results.results) == 100
    assert _kinds(results).count(SearchResultType.CATALOG_HIT) == catalog_quota(100) == 25
    assert results.total_results == 200


def test_raw_catalog_scores_cannot_lift_rows_above_passages():
    catalog = [_hit(f"s3://c{i}", 10.0 - i) for i in range(8)]
    passages = [_hit(f"n{i}", 0.9 - i / 100) for i in range(8)]
    results = _search(catalog=catalog, embeddings=passages).search(
        "q", scope=SearchScope.ALL, limit=8
    )
    assert _kinds(results) == [SearchResultType.EMBEDDING_MATCH] * 8


def test_catalog_rows_past_the_quota_backfill_a_page_nothing_else_fills():
    nodes = [_hit("n1", 0.9), _hit("n2", 0.8)]
    catalog = [_hit(f"s3://c{i}", 3.0 - i / 10) for i in range(6)]
    results = _search(catalog=catalog, knowledge_graph=nodes).search(
        "q", scope=SearchScope.ALL, limit=10
    )
    assert _ids(results) == ["n1", "n2"] + [f"s3://c{i}" for i in range(6)]
    assert results.total_results == 8


def test_total_results_counts_every_distinct_hit_before_the_cut():
    search = _search(
        embeddings=[_hit(f"n{i}", 0.9) for i in range(7)],
        rag=[_hit(f"n{i}", 1.0) for i in range(3, 10)],
    )
    results = search.search("q", scope=SearchScope.ALL, limit=5)
    assert len(results.results) == 5
    assert results.total_results == 10  # n0..n9, merged, before the page


def test_nothing_found_is_zero_results():
    search = _search(catalog=[], knowledge_graph=[], embeddings=[], rag=[])
    results = search.search("nonsense", scope=SearchScope.ALL)
    assert results.total_results == 0 and results.results == []


def test_every_lane_is_asked_for_the_page_size():
    lanes = {name: _Lane([]) for name in ("catalog", "knowledge_graph", "embeddings", "rag")}
    UnifiedSearch(**lanes).search("q", scope=SearchScope.ALL, limit=7)
    assert all(lane.limits == [7] for lane in lanes.values())


# ── one scope: the lane as it is ─────────────────────────────────────────


def test_a_single_scope_keeps_its_raw_scores_and_order():
    """Callers that threshold a lane's own score (cosine >= 0.7) keep working."""
    search = _search(
        embeddings=[_hit("n2", 0.71), _hit("n1", 0.93), _hit("n3", 0.65)],
        catalog=[_hit("s3://c", 9.0)],
    )
    results = search.search("q", scope=SearchScope.EMBEDDINGS, limit=2)
    assert [(r.source_id, r.score) for r in results.results] == [("n1", 0.93), ("n2", 0.71)]
    assert results.total_results == 3  # counted before the cut


# ── the other UnifiedSearch user ─────────────────────────────────────────


def test_the_knowledge_search_adapter_returns_the_fused_order():
    """knowledge/adapters.py delegates ``search`` unchanged: it sees RRF order."""
    adapter = UMRSSearchAdapter(
        _search(
            catalog=[_hit("s3://cat", 4.0)],
            knowledge_graph=[_hit("kg-node", 0.9)],
            embeddings=[_hit("dense-node", 0.8)],
            rag=[_hit("lex-node", 3.2)],
        )
    )
    results = adapter.search("q", scope=SearchScope.ALL, limit=3)
    assert _ids(results) == ["dense-node", "lex-node", "kg-node"]
    assert results.total_results == 4


# ── the catalogue floor (LIVE-06) ────────────────────────────────────────


def _meta(state: CatalogState = CatalogState.READY, **fields) -> CatalogObjectMeta:
    base = {
        "s3_key": "proj-1/uploads/batch-3-yields.csv",
        "object_etag": "e",
        "state": state,
        "owner": "u",
        "acl_scope": "proj-1",
        "updated_at": "2026-10-01T00:00:00",
        "summary": "reaction yields per batch",
        "detected_modality": "structured",
        "tags": ["yield_pct"],
        "filename_signals": ["batch-3-yields.csv"],
    }
    base.update(fields)
    return CatalogObjectMeta(**base)


def test_a_ready_row_no_token_matches_scores_zero():
    assert score_meta(_meta(), ["zebra", "quantum"], query="zebra quantum") == 0.0


def test_a_ready_row_with_a_token_match_keeps_its_bonus():
    ready = score_meta(_meta(), ["yields"], query="yields")
    errored = score_meta(_meta(state=CatalogState.ERROR), ["yields"], query="yields")
    assert ready == pytest.approx(errored + 0.25)


def test_a_pending_row_is_still_halved():
    ready = score_meta(_meta(), ["yields"], query="yields")
    pending = score_meta(_meta(state=CatalogState.PENDING), ["yields"], query="yields")
    assert pending == pytest.approx((ready - 0.25) / 2)


def test_token_matches_counts_tokens_found_anywhere_in_the_row():
    meta = _meta()
    assert token_matches(meta, ["yield", "batch", "zebra"]) == 2
    assert token_matches(meta, []) == 0


def _row(meta: CatalogObjectMeta) -> dict:
    return {
        "s3_key": meta.s3_key,
        "object_etag": meta.object_etag,
        "state": meta.state.value,
        "owner": meta.owner,
        "acl_scope": meta.acl_scope,
        "updated_at": meta.updated_at,
        "summary": meta.summary,
        "deterministic_summary": "",
        "tags": list(meta.tags),
        "filename_signals": list(meta.filename_signals),
        "detected_modality": meta.detected_modality,
        "visibility": "project",
    }


def _fallback_search(query: str, metas: list[CatalogObjectMeta]):
    # 1st execute: the strict FTS AND-match misses; 2nd: the recency window.
    pool = FakePool([[], [_row(m) for m in metas]])
    return CatalogMetaStore(pool).search("proj-1", query, limit=20)


def test_the_fallback_returns_nothing_for_a_query_nothing_matches():
    """LIVE-06: nonsense used to return every READY row (and every structured
    row when the query said "data"), so total_results was never zero."""
    rows = [_meta(), _meta(s3_key="proj-1/uploads/notes.md", detected_modality="document")]
    assert _fallback_search("zebra data quantum", rows) == []


def test_the_fallback_keeps_rows_whose_tokens_land_in_different_fields():
    rows = [_meta(), _meta(s3_key="proj-1/uploads/other.md", summary="unrelated", tags=[])]
    hits = _fallback_search("reaction yield_pct", rows)
    assert [h.meta.s3_key for h in hits] == ["proj-1/uploads/batch-3-yields.csv"]
