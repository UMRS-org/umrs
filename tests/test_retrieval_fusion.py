"""ADR-093 C6: weighted reciprocal-rank fusion of the retrieval lanes."""

from __future__ import annotations

import pytest
from umrs.knowledge.retrieval import fusion
from umrs.knowledge.retrieval.types import Candidate


@pytest.fixture(autouse=True)
def _site_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (*fusion.WEIGHT_ENV.values(), fusion.FUSION_ENV):
        monkeypatch.delenv(name, raising=False)


def _lane(lane: str, *ids: str, scores: list[float] | None = None) -> list[Candidate]:
    out = []
    for rank, node_id in enumerate(ids, 1):
        score = scores[rank - 1] if scores else 1.0 / rank
        cosine = score if lane.startswith("dense") else None
        out.append(
            Candidate(
                node_id,
                artifact_id=f"art-{node_id}",
                lane_scores={fusion.base_lane(lane): score},
                lane_ranks={fusion.base_lane(lane): rank},
                cosine=cosine,
            )
        )
    return out


def _by_id(fused: list[Candidate]) -> dict[str, Candidate]:
    return {c.node_id: c for c in fused}


def test_rrf_sums_weight_over_k_plus_rank() -> None:
    fused = fusion.fuse({"dense": _lane("dense", "a", "b", "c"), "lex": _lane("lex", "c", "d")})
    got = _by_id(fused)
    assert got["a"].fused == pytest.approx(1.0 / 61)
    assert got["c"].fused == pytest.approx(1.0 / 63 + 1.0 / 61)
    assert got["d"].fused == pytest.approx(1.0 / 62)
    # b and d tie at 1/62; b carries a cosine, so it goes first.
    assert [c.node_id for c in fused] == ["c", "a", "b", "d"]


def test_default_weights_give_ident_half_weight() -> None:
    assert fusion.default_weights() == {"dense": 1.0, "lex": 1.0, "ident": 0.5}
    fused = _by_id(fusion.fuse({"ident": _lane("ident", "x")}))
    assert fused["x"].fused == pytest.approx(0.5 / 61)


def test_site_weights_come_from_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UMRS_RAG_W_LEX", "1.5")
    monkeypatch.setenv("UMRS_RAG_W_IDENT", "not-a-number")
    assert fusion.default_weights() == {"dense": 1.0, "lex": 1.5, "ident": 0.5}
    fused = _by_id(fusion.fuse({"lex": _lane("lex", "x")}))
    assert fused["x"].fused == pytest.approx(1.5 / 61)


def test_call_weights_override_the_site_and_k_is_a_parameter() -> None:
    fused = _by_id(fusion.fuse({"lex": _lane("lex", "x")}, weights={"lex": 0.75}, k=10))
    assert fused["x"].fused == pytest.approx(0.75 / 11)


def test_variant_lanes_take_their_base_lanes_weight() -> None:
    lanes = {"dense#0": _lane("dense#0", "a", "b"), "dense#1": _lane("dense#1", "b", "a")}
    fused = _by_id(fusion.fuse(lanes, weights={"dense": 2.0}))
    assert fused["a"].fused == pytest.approx(2.0 / 61 + 2.0 / 62)
    assert fused["a"].fused == pytest.approx(fused["b"].fused)
    assert fusion.lane_weight("dense#3", {"dense": 2.0, "dense#3": 0.5}) == 0.5
    assert fusion.lane_weight("unknown", {}) == 1.0


def test_merge_keeps_every_lanes_scores_ranks_and_the_cosine() -> None:
    dense = _lane("dense", "a", "b", scores=[0.91, 0.85])
    lex = _lane("lex", "b", scores=[7.5])
    lex[0].lane_scores["lex:idf_sum"] = 7.4
    fused = _by_id(fusion.fuse({"dense": dense, "lex": lex}))
    b = fused["b"]
    assert b.lane_scores == {"dense": 0.85, "lex": 7.5, "lex:idf_sum": 7.4}
    assert b.lane_ranks == {"dense": 2, "lex": 1}
    assert b.cosine == 0.85 and b.artifact_id == "art-b"


def test_lexical_only_candidates_keep_no_cosine() -> None:
    fused = _by_id(fusion.fuse({"dense": _lane("dense", "a"), "lex": _lane("lex", "z")}))
    assert fused["z"].cosine is None


def test_fusion_leaves_its_inputs_untouched() -> None:
    dense, lex = _lane("dense", "a", "b"), _lane("lex", "b")
    before = [(c.lane_scores.copy(), c.lane_ranks.copy(), c.fused) for c in dense + lex]
    fused = fusion.fuse({"dense": dense, "lex": lex})
    after = [(c.lane_scores, c.lane_ranks, c.fused) for c in dense + lex]
    assert before == after
    assert all(c not in dense + lex for c in fused if c.node_id == "b")


def test_a_zero_weight_lane_is_left_out() -> None:
    fused = fusion.fuse(
        {"dense": _lane("dense", "a"), "lex": _lane("lex", "z")}, weights={"lex": 0}
    )
    assert [c.node_id for c in fused] == ["a"]


def test_a_chunk_counts_once_per_lane_at_its_best_rank() -> None:
    lane = _lane("lex", "a", "b") + _lane("lex", "a")
    fused = _by_id(fusion.fuse({"lex": lane}))
    assert fused["a"].fused == pytest.approx(1.0 / 61)
    assert fused["b"].fused == pytest.approx(1.0 / 62)


def test_ties_break_on_best_rank_then_cosine_then_id() -> None:
    dense = _lane("dense", "d1", scores=[0.9])
    lex = _lane("lex", "l1", scores=[5.0])
    fused = fusion.fuse({"lex": lex, "dense": dense})
    assert fused[0].fused == pytest.approx(fused[1].fused)
    assert [c.node_id for c in fused] == ["d1", "l1"]  # the cosine-bearing chunk first


def test_convex_combination_normalises_raw_lane_scores() -> None:
    dense = _lane("dense", "a", "b", "c", scores=[0.9, 0.8, 0.7])
    lex = _lane("lex", "c", "a", scores=[10.0, 2.0])
    fused = _by_id(fusion.fuse({"dense": dense, "lex": lex}, method="convex"))
    assert fused["a"].fused == pytest.approx(1.0 + 0.0)
    assert fused["b"].fused == pytest.approx(0.5)
    assert fused["c"].fused == pytest.approx(0.0 + 1.0)


def test_fusion_method_follows_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert fusion.fusion_method() == "rrf"
    monkeypatch.setenv(fusion.FUSION_ENV, "convex")
    assert fusion.fusion_method() == "convex"
    assert fusion.fusion_method("rrf") == "rrf"
    assert fusion.fusion_method("bogus") == "rrf"
    lanes = {"lex": _lane("lex", "a", "b", scores=[3.0, 3.0])}
    assert [c.fused for c in fusion.fuse(lanes)] == [1.0, 1.0]  # equal raw scores: both 1


def test_empty_lanes_fuse_to_nothing() -> None:
    assert fusion.fuse({}) == []
    assert fusion.fuse({"dense": [], "lex": []}, method="convex") == []
