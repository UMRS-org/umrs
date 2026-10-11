"""Deterministic relevance cut and calibrated sufficiency (ADR-093 D-3, C2).

Under e5-base-v2 every cosine sits in 0.7-1.0: the fixed 0.7 neither shrank
the context nor caught nonsense (0.699 for "best pizza in Naples", 0.875 for
the relevant note, validation §11 of 2026-10-09). The cut is relative, the
verdict is calibrated, and neither asks the 3B.
"""

from __future__ import annotations

import pytest
from umrs.knowledge.retrieval import cutoff
from umrs.knowledge.retrieval.calibrate import NullStats
from umrs.knowledge.retrieval.types import Candidate, RetrieveSpec

NULL = NullStats(mu=0.70, sigma=0.02, n=30, model="e5-base-v2")


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    for name in cutoff._DEFAULTS:
        monkeypatch.delenv(name, raising=False)


def _spec(**overrides):
    return RetrieveSpec(**{"queries": ("q",), "project_id": "p", **overrides})


def _c(cosine, node=None, *, lex=None, logit=None, text=""):
    candidate = Candidate(
        node_id=node or f"n{cosine}", cosine=cosine, rerank_logit=logit, text=text
    )
    if lex is not None:
        candidate.lane_scores["lex"] = lex
    return candidate


def _ids(kept):
    return [c.node_id for c in kept]


# ── the dense cut ───────────────────────────────────────────────────────────


def test_dense_keeps_rank_one_and_the_ones_within_delta():
    ranked = [_c(0.90, "a"), _c(0.89, "b"), _c(0.88, "c"), _c(0.86, "d")]
    assert _ids(cutoff.cut(ranked, _spec(), mode="dense")) == ["a", "b", "c"]


def test_dense_stops_at_the_largest_gap_of_at_least_gamma():
    # gaps 0.005, 0.025, 0.002 against gamma 0.015 (ADR-093 Phase 9)
    ranked = [_c(0.900, "a"), _c(0.895, "b"), _c(0.870, "c"), _c(0.868, "d")]
    assert _ids(cutoff.cut(ranked, _spec(), mode="dense")) == ["a", "b"]


def test_dense_never_keeps_more_than_k_max_nor_fewer_than_one():
    ranked = [_c(0.90 - i * 0.001, f"n{i}") for i in range(8)]
    assert len(cutoff.cut(ranked, _spec(k_max=4), mode="dense")) == 4
    assert _ids(cutoff.cut([_c(0.70, "only")], _spec(), mode="dense")) == ["only"]
    assert cutoff.cut([], _spec(), mode="dense") == []


def test_delta_and_gamma_read_the_environment(monkeypatch):
    ranked = [_c(0.90, "a"), _c(0.84, "b")]
    assert _ids(cutoff.cut(ranked, _spec(), mode="dense")) == ["a"]
    monkeypatch.setenv("UMRS_RAG_DELTA", "0.1")
    monkeypatch.setenv("UMRS_RAG_GAMMA", "1.0")
    assert _ids(cutoff.cut(ranked, _spec(), mode="dense")) == ["a", "b"]


def test_fixed_cutoff_keeps_the_top_k_max():
    ranked = [_c(0.90, "a"), _c(0.70, "b"), _c(0.60, "c")]
    assert _ids(cutoff.cut(ranked, _spec(cutoff="fixed", k_max=2), mode="dense")) == ["a", "b"]


# ── the fused cut ───────────────────────────────────────────────────────────


def test_fused_keeps_by_the_dense_rule_against_the_best_cosine_of_the_top_ten():
    ranked = [_c(0.86, "lexfirst", lex=9.0), _c(0.90, "dense1"), _c(0.80, "weak")]
    assert _ids(cutoff.cut(ranked, _spec(), mode="fused")) == ["lexfirst", "dense1"]


def test_fused_keeps_a_strong_lexical_match_the_dense_rule_drops():
    ranked = [_c(0.90, "a", lex=8.0), _c(0.70, "b", lex=4.5), _c(0.70, "c", lex=3.0)]
    assert _ids(cutoff.cut(ranked, _spec(), mode="fused")) == ["a", "b"]


def test_fused_respects_k_max():
    ranked = [_c(0.90, f"n{i}") for i in range(6)]
    assert len(cutoff.cut(ranked, _spec(k_max=3), mode="fused")) == 3


# ── the reranker cut ────────────────────────────────────────────────────────


def test_rerank_keeps_while_probable_and_within_the_logit_window():
    ranked = [
        _c(0.8, "a", logit=4.0),
        _c(0.8, "b", logit=3.5),
        _c(0.8, "c", logit=-0.5),
        _c(0.8, "d", logit=-3.0),
    ]
    assert _ids(cutoff.cut(ranked, _spec(), mode="rerank")) == ["a", "b"]


def test_rerank_stops_at_the_largest_gap_of_two_logits():
    ranked = [_c(0.8, "a", logit=5.0), _c(0.8, "b", logit=4.8), _c(0.8, "c", logit=2.5)]
    assert _ids(cutoff.cut(ranked, _spec(), mode="rerank")) == ["a", "b"]


def test_rerank_ignores_candidates_outside_the_scored_pool():
    ranked = [_c(0.8, "a", logit=2.0), _c(0.9, "unscored")]
    assert _ids(cutoff.cut(ranked, _spec(), mode="rerank")) == ["a"]


# ── the legacy filter ───────────────────────────────────────────────────────


def test_legacy_is_the_absolute_filter_on_the_top_k():
    ranked = [_c(0.80, "a"), _c(0.69, "b"), _c(0.75, "c")]
    assert _ids(cutoff.cut(ranked, _spec(threshold=0.7, k_max=5), mode="legacy")) == ["a", "c"]
    assert _ids(cutoff.cut(ranked, _spec(threshold=0.7, k_max=1), mode="legacy")) == ["a"]


def test_legacy_clips_negative_cosines_at_zero_like_the_k3_score():
    assert _ids(cutoff.cut([_c(-0.2, "neg")], _spec(threshold=0.0), mode="legacy")) == ["neg"]


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        cutoff.cut([_c(0.9)], _spec(), mode="graph")


# ── sufficiency ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("cosine", "status"),
    [(0.875, "sufficient"), (0.76, "sufficient"), (0.745, "partial"), (0.699, "insufficient")],
)
def test_dense_sufficiency_reads_z_null(cosine, status):
    verdict = cutoff.sufficiency(
        [_c(cosine)], mode="dense", null=NULL, lexical_hit=False, searched=100
    )
    assert verdict.status == status
    assert verdict.basis == "z_null" and verdict.value == pytest.approx(
        (cosine - 0.70) / 0.02, abs=1e-3
    )


def test_a_lexical_hit_lifts_a_partial_z_null_to_sufficient():
    verdict = cutoff.sufficiency(
        [_c(0.745)], mode="dense", null=NULL, lexical_hit=True, searched=100
    )
    assert (verdict.status, verdict.basis) == ("sufficient", "lexical")
    assert verdict.value == pytest.approx(2.25) and "lifted" in verdict.detail


def test_no_vectors_in_scope_is_insufficient():
    verdict = cutoff.sufficiency([], mode="dense", null=None, lexical_hit=False, searched=0)
    assert (verdict.status, verdict.basis) == ("insufficient", "no_vectors")


def test_without_a_null_only_the_lexical_rule_decides():
    hit = cutoff.sufficiency([_c(0.9)], mode="dense", null=None, lexical_hit=True, searched=12)
    miss = cutoff.sufficiency([_c(0.9)], mode="dense", null=None, lexical_hit=False, searched=12)
    assert (hit.status, hit.basis) == ("sufficient", "lexical")
    assert (miss.status, miss.basis) == ("insufficient", "lexical")


@pytest.mark.parametrize(
    ("logit", "status"), [(-3.5, "insufficient"), (-1.0, "partial"), (0.5, "sufficient")]
)
def test_rerank_sufficiency_reads_p_one(logit, status):
    verdict = cutoff.sufficiency(
        [_c(0.8, logit=logit)], mode="rerank", null=NULL, lexical_hit=True, searched=100
    )
    assert (verdict.status, verdict.basis) == (status, "rerank_p")


def test_legacy_sufficiency_reports_the_threshold():
    kept = cutoff.sufficiency(
        [_c(0.8)], mode="legacy", null=None, lexical_hit=False, searched=5, threshold=0.7
    )
    none = cutoff.sufficiency(
        [], mode="legacy", null=None, lexical_hit=False, searched=5, threshold=0.7
    )
    assert (kept.status, kept.basis, kept.threshold) == ("sufficient", "threshold", 0.7)
    assert none.status == "insufficient" and none.detail == "no passage scored at or above 0.7"


def test_the_z_thresholds_read_the_environment(monkeypatch):
    monkeypatch.setenv("UMRS_RAG_Z_SUFFICIENT", "20")
    verdict = cutoff.sufficiency(
        [_c(0.875)], mode="dense", null=NULL, lexical_hit=False, searched=100
    )
    assert verdict.status == "partial"


# ── lexical evidence ────────────────────────────────────────────────────────


def test_content_terms_drop_stop_words_and_short_tokens():
    assert cutoff.content_terms("What is the best pizza restaurant in Naples?") == [
        "pizza",
        "restaurant",
        "naples",
    ]
    assert cutoff.content_terms("IL-6 and TiO2 in ZSM-5") == ["il-6", "tio2", "zsm-5"]


def test_coverage_counts_terms_that_start_a_word():
    text = "Encapsulation with alumina slows moisture ingress in perovskites."
    assert cutoff.term_coverage("Does alumina encapsulation slow perovskite moisture?", text) == 1.0
    assert cutoff.term_coverage("zeolite sorption", text) == 0.0


def test_the_s1_lexical_hit_is_half_the_terms_in_the_top_passage():
    top = _c(0.8, text="Encapsulation with alumina slows moisture ingress.")
    assert cutoff.lexical_hit("alumina encapsulation zeolite", [top])
    assert not cutoff.lexical_hit("zeolite sorption capacity", [top])
    assert not cutoff.lexical_hit("anything", [])


def test_the_s3_lexical_hit_is_a_rare_matched_lexeme():
    top = _c(0.8, text="unrelated")
    assert cutoff.lexical_hit("zeolite sorption", [top], strength=3.2)
    assert not cutoff.lexical_hit("zeolite sorption", [top], strength=2.9)
    assert cutoff.LEXICAL_HIT_IDF == 3.0


def test_sigmoid_is_stable_at_the_extremes():
    assert cutoff.sigmoid(0.0) == 0.5
    assert cutoff.sigmoid(-800.0) == pytest.approx(0.0) and cutoff.sigmoid(800.0) == pytest.approx(
        1.0
    )


# ── ADR-093 review fixes (2026-10-10): what lexical evidence may decide ─────


def _lex(node, cosine, idf_max, *, text=""):
    """A lex-lane candidate whose best matched lexeme has IDF ``idf_max``."""
    candidate = _c(cosine, node, lex=idf_max, text=text)
    candidate.lane_scores.update({"lex:idf_sum": idf_max, "lex:idf_max": idf_max})
    return candidate


_NAPLES = "What is the best pizza restaurant in Naples?"


def test_a_single_rare_lexeme_never_overrides_an_insufficient_z():
    """core-1: 'restaurant' occurs in one abstract of 9.4k chunks (idf ~8)."""
    kept = [
        _c(0.70, "d0", text="Statin therapy and cardiovascular outcomes."),
        _lex("L1", 0.70, 8.1, text="Sodium content of restaurant meals."),
    ]
    hit = cutoff.lexical_hit(_NAPLES, kept, strength=8.1)
    verdict = cutoff.sufficiency(kept, mode="fused", null=NULL, lexical_hit=hit, searched=9377)
    assert (verdict.status, verdict.basis) == ("insufficient", "z_null")


def test_a_lexical_hit_never_overrides_an_insufficient_z():
    verdict = cutoff.sufficiency(
        [_c(0.71)], mode="dense", null=NULL, lexical_hit=True, searched=100
    )
    assert (verdict.status, verdict.basis) == ("insufficient", "z_null")


def test_a_kept_rare_lexeme_lifts_a_partial_z_to_sufficient():
    kept = [_lex("L1", 0.75, 8.1, text="Trypanosomiasis is transmitted by tsetse flies.")]
    lifted = cutoff.sufficiency(
        kept,
        mode="fused",
        null=NULL,
        lexical_hit=cutoff.lexical_evidence("trypanosomiasis outbreaks in cattle", kept),
        searched=100,
    )
    assert (lifted.status, lifted.basis) == ("sufficient", "lexical")
    plain = [_c(0.75, "d", text="Unrelated.")]
    verdict = cutoff.sufficiency(
        plain,
        mode="dense",
        null=NULL,
        lexical_hit=cutoff.lexical_evidence("trypanosomiasis outbreaks", plain),
        searched=100,
    )
    assert verdict.status == "partial"


def test_strength_is_read_from_the_kept_passages_only():
    kept = [_c(0.75, "d", text="Unrelated."), _lex("L", 0.70, 6.0)]
    assert cutoff.lexical_evidence("zeolite", kept).strength == 6.0
    assert cutoff.lexical_evidence("zeolite", kept[:1]).strength == 0.0


def test_coverage_needs_two_matched_terms_unless_the_query_has_one():
    top = _c(0.8, text="Sodium content of restaurant meals.")
    two = cutoff.lexical_evidence("pizza restaurant", [top])
    assert (two.coverage, two.matched, two.terms, two.covers) == (0.5, 1, 2, False)
    assert cutoff.lexical_evidence("restaurant", [top]).covers
    assert cutoff.lexical_evidence("restaurant meals sodium pizza", [top]).covers


def test_coverage_takes_the_best_covered_query_variant():
    top = _c(0.8, text="Zeolite cracking of heavy oil fractions.")
    evidence = cutoff.lexical_evidence(("it", "zeolite cracking it"), [top])
    assert evidence.covers and evidence.coverage == 1.0


def test_without_a_null_only_coverage_establishes_sufficiency():
    """core-4: no z means no 'partial'; a nonsense question is insufficient."""
    top = _c(0.9, text="Encapsulation with alumina slows moisture ingress.")
    covered = cutoff.sufficiency(
        [top],
        mode="dense",
        null=None,
        lexical_hit=cutoff.lexical_evidence("alumina encapsulation", [top]),
        searched=12,
    )
    assert (covered.status, covered.basis) == ("sufficient", "lexical")
    miss = cutoff.sufficiency([_c(0.70)], mode="dense", null=None, lexical_hit=False, searched=12)
    assert (miss.status, miss.basis) == ("insufficient", "lexical")


def test_without_a_null_a_rare_lexeme_alone_is_not_sufficient():
    kept = [_lex("L1", 0.70, 8.7, text="Sodium content of restaurant meals.")]
    verdict = cutoff.sufficiency(
        kept,
        mode="fused",
        null=None,
        lexical_hit=cutoff.lexical_hit(_NAPLES, kept, strength=8.7),
        searched=12,
    )
    assert verdict.status == "insufficient"


def test_no_vectors_in_this_space_is_insufficient_even_with_lexical_rows():
    kept = [_lex("L1", None, 8.7, text="alumina encapsulation")]
    verdict = cutoff.sufficiency(kept, mode="fused", null=None, lexical_hit=True, searched=0)
    assert (verdict.status, verdict.basis) == ("insufficient", "no_vectors")


def test_the_fused_cut_without_any_cosine_applies_the_lexical_rule_alone():
    """core-5: a missing cosine is no pass of the dense rule (encoder outage)."""
    ranked = [
        Candidate(node_id=f"L{i}", lane_scores={"lex": s})
        for i, s in enumerate((9.0, 1.0, 0.9, 0.8))
    ]
    assert _ids(cutoff.cut(ranked, _spec(k_max=4), mode="fused")) == ["L0"]


def test_the_fused_anchor_is_the_best_known_cosine():
    ranked = [
        Candidate(node_id="L0", lane_scores={"lex": 9.0}),
        _c(0.90, "d1"),
        _c(0.89, "d2"),
        _c(0.70, "d3", lex=1.0),
    ]
    assert _ids(cutoff.cut(ranked, _spec(k_max=4), mode="fused")) == ["L0", "d1", "d2"]
