"""Fusion of lane rankings (ADR-093 D-6, component C6; ADR-085 D-1's fusion part).

Weighted reciprocal-rank fusion: a candidate at 1-based rank ``r`` in lane
``l`` earns ``w_l / (k + r)``, summed over the lanes that found it (k = 60).
The same function fuses query variants: lanes named ``dense#0``, ``dense#1``
each take the weight of their base lane (``dense``).

Weights default to dense 1.0, lex 1.0, ident 0.5, overridable per site by
``UMRS_RAG_W_DENSE`` / ``UMRS_RAG_W_LEX`` / ``UMRS_RAG_W_IDENT`` and per call
by ``weights``. A lane weighted 0 is left out. The alternative the tune split
measures against RRF, a min-max convex combination of the raw lane scores,
is ``method="convex"`` (or ``UMRS_RAG_FUSION=convex``).

Candidates are merged by node id: the result holds one copy per chunk with
every lane's score and rank; the input candidates are not modified.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping, Sequence

from .types import Candidate

RRF_K = 60
DEFAULT_WEIGHTS: Mapping[str, float] = {"dense": 1.0, "lex": 1.0, "ident": 0.5}
WEIGHT_ENV: Mapping[str, str] = {
    "dense": "UMRS_RAG_W_DENSE",
    "lex": "UMRS_RAG_W_LEX",
    "ident": "UMRS_RAG_W_IDENT",
}
FUSION_ENV = "UMRS_RAG_FUSION"
FUSION_METHODS: tuple[str, ...] = ("rrf", "convex")

Lanes = Mapping[str, Sequence[Candidate]]


def base_lane(name: str) -> str:
    """``dense#1`` -> ``dense``: a query variant's lane."""
    return name.split("#", 1)[0]


def _env_weight(lane: str, default: float) -> float:
    raw = os.environ.get(WEIGHT_ENV[lane], "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def default_weights() -> dict[str, float]:
    """The site's lane weights: the defaults, overridden by the env knobs."""
    return {lane: _env_weight(lane, value) for lane, value in DEFAULT_WEIGHTS.items()}


def lane_weight(name: str, weights: Mapping[str, float]) -> float:
    """A lane's weight: its own entry, else its base lane's, else 1.0."""
    if name in weights:
        return float(weights[name])
    return float(weights.get(base_lane(name), 1.0))


def fusion_method(method: str | None = None) -> str:
    """``rrf`` unless ``method`` (or ``UMRS_RAG_FUSION``) names ``convex``."""
    choice = (method or os.environ.get(FUSION_ENV, "") or "rrf").strip().lower()
    return choice if choice in FUSION_METHODS else "rrf"


def _copy(candidate: Candidate) -> Candidate:
    return dataclasses.replace(
        candidate,
        lane_scores=dict(candidate.lane_scores),
        lane_ranks=dict(candidate.lane_ranks),
        fused=0.0,
    )


def _absorb(into: Candidate, other: Candidate) -> None:
    """Fold another lane's copy of the same chunk into ``into``."""
    for key, score in other.lane_scores.items():
        into.lane_scores[key] = max(score, into.lane_scores.get(key, score))
    for key, rank in other.lane_ranks.items():
        into.lane_ranks[key] = min(rank, into.lane_ranks.get(key, rank))
    if other.cosine is not None and (into.cosine is None or other.cosine > into.cosine):
        into.cosine = other.cosine
    if into.rerank_logit is None:
        into.rerank_logit = other.rerank_logit
    if not into.content:
        into.content = other.content
    into.artifact_id = into.artifact_id or other.artifact_id


def _merged(
    merged: dict[str, Candidate], candidate: Candidate, lane: str, rank: int
) -> Candidate:
    """The merged copy of ``candidate``, created on first sight."""
    target = merged.get(candidate.node_id)
    if target is None:
        target = merged[candidate.node_id] = _copy(candidate)
    else:
        _absorb(target, candidate)
    target.lane_ranks.setdefault(lane, rank)
    return target


def _order_key(candidate: Candidate) -> tuple[float, int, float, str]:
    best_rank = min(candidate.lane_ranks.values(), default=1 << 30)
    cosine = candidate.cosine if candidate.cosine is not None else -2.0
    return (-candidate.fused, best_rank, -cosine, candidate.node_id)


def _first_occurrences(ranked: Sequence[Candidate]) -> list[Candidate]:
    """A lane counts each chunk once, at its best rank."""
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in ranked:
        if candidate.node_id not in seen:
            seen.add(candidate.node_id)
            unique.append(candidate)
    return unique


def _rrf(lanes: Lanes, weights: Mapping[str, float], k: int) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for lane, ranked in lanes.items():
        weight = lane_weight(lane, weights)
        if weight <= 0:
            continue
        for rank, candidate in enumerate(_first_occurrences(ranked), 1):
            target = _merged(merged, candidate, lane, rank)
            target.fused += weight / (k + rank)
    return sorted(merged.values(), key=_order_key)


def _raw_score(candidate: Candidate, lane: str) -> float:
    for key in (lane, base_lane(lane)):
        if key in candidate.lane_scores:
            return float(candidate.lane_scores[key])
    if base_lane(lane) == "dense" and candidate.cosine is not None:
        return float(candidate.cosine)
    return 0.0


def _min_max(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    if high <= low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def _convex(lanes: Lanes, weights: Mapping[str, float]) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for lane, ranked in lanes.items():
        weight = lane_weight(lane, weights)
        if weight <= 0 or not ranked:
            continue
        unique = _first_occurrences(ranked)
        normalised = _min_max([_raw_score(c, lane) for c in unique])
        for rank, (candidate, value) in enumerate(zip(unique, normalised), 1):
            target = _merged(merged, candidate, lane, rank)
            target.fused += weight * value
    return sorted(merged.values(), key=_order_key)


def fuse(
    lanes: Lanes,
    weights: Mapping[str, float] | None = None,
    k: int = RRF_K,
    *,
    method: str | None = None,
) -> list[Candidate]:
    """Merge lane rankings into one, best first; ``Candidate.fused`` holds the score.

    ``lanes`` maps a lane name to its candidates in rank order (the list order
    is the rank). ``weights`` override the site weights per lane; a variant
    lane (``dense#0``) without its own entry takes its base lane's weight.
    """
    resolved = {**default_weights(), **(weights or {})}
    if fusion_method(method) == "convex":
        return _convex(lanes, resolved)
    return _rrf(lanes, resolved, k)


__all__ = [
    "DEFAULT_WEIGHTS",
    "RRF_K",
    "base_lane",
    "default_weights",
    "fuse",
    "fusion_method",
    "lane_weight",
]
