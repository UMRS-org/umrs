"""Deterministic relevance cut and calibrated sufficiency (ADR-093 D-3, C2).

How many candidates reach the model, and whether any should, is decided
here by scores — never by the generator (a 3B judges relevance and
sufficiency poorly, CRAG). Every adaptive mode keeps at least one and at
most ``k_max`` candidates:

- ``dense`` (S1): keep rank 1, then rank i while ``cos_i >= cos_1 - delta``;
  stop at the largest consecutive gap ``>= gamma`` inside the top ``k_max``.
- ``fused`` (S3): in fused order while a candidate passes the dense rule
  against the best cosine of the top 10, or the lexical rule
  (``lex >= 0.5 x best lex``).
- ``rerank`` (S4): ``p = sigmoid(logit)``; keep while ``p >= 0.10`` and
  ``logit >= logit_1 - 4.0``; stop at the largest logit gap ``>= 2.0``.
- ``legacy``: ``threshold`` set — the absolute filter, unchanged.

``cutoff=fixed`` keeps the top ``k_max`` in any adaptive mode.

Sufficiency (``sufficient | partial | insufficient``, with basis, value and
threshold): dense/fused by ``z_null = (cos_1 - mu) / sigma`` against the
calibrated null (ADR-083 D-5, ``calibrate.py``); rerank by ``p_1``; legacy
by the threshold. Lexical evidence (:class:`LexicalEvidence`, over the KEPT
passages) only ever lifts a partial ``z_null`` (``2 <= z < 3``) to
sufficient, and decides alone only when there is no null (fewer than 30
vectors in scope, or the probes could not be encoded) — and then only by the
coverage rule. It never overrides an insufficient ``z_null``: off-domain
words are the RAREST in a corpus, so one matched rare lexeme is no evidence
(review core-1, 2026-10-10: 29 of the 30 NONSENSE_EVAL_V1 probes share such
a word with SciFact). A scope without vectors in the query's space is
insufficient whatever the lexical lanes found. Every constant reads the
environment at call time and is tuned on the SciFact tune split only.

Composition-layer code: nothing here imports ``umrs.api``.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from r4_contracts.retrieval import RetrievalSufficiency

from .types import Candidate, RetrieveSpec

CUT_MODES: tuple[str, ...] = ("dense", "fused", "rerank", "legacy")

_DEFAULTS: dict[str, float] = {
    "UMRS_RAG_DELTA": 0.03,
    # Tuned on SciFact train (2026-10-10, benchmarks/retrieval/tune_cut.py):
    # 0.015 kept ctx_recall 0.893 against a 0.903 floor; 0.02 meets it at
    # ctx_precision 0.601 (top-5: 0.243).
    "UMRS_RAG_GAMMA": 0.02,
    "UMRS_RAG_Z_SUFFICIENT": 3.0,
    "UMRS_RAG_Z_PARTIAL": 2.0,
    "UMRS_RAG_NULL_P": 0.05,
    "UMRS_RAG_RERANK_P_MIN": 0.10,
    "UMRS_RAG_RERANK_WINDOW": 4.0,
    "UMRS_RAG_RERANK_GAP": 2.0,
    "UMRS_RAG_RERANK_P_SUFFICIENT": 0.5,
    "UMRS_RAG_LEX_RATIO": 0.5,
    "UMRS_RAG_LEX_COVERAGE": 0.5,
    "UMRS_RAG_LEX_MIN_TERMS": 2.0,
    "UMRS_RAG_LEX_HIT_IDF": 3.0,
}
_ANCHOR_TOP = 10
#: The S3 lexical floor: the IDF of one matched query lexeme in a kept
#: passage (``lexical.lexical_hit_strength(..., per_term=True)``). It may lift
#: a partial z_null to sufficient, never more.
LEXICAL_HIT_IDF = _DEFAULTS["UMRS_RAG_LEX_HIT_IDF"]
#: ``lexical.LEX_IDF_MAX``, read here so the rule needs no lane module.
_LEX_IDF_MAX = "lex:idf_max"

_TERM_RE = re.compile(r"[^\W_][\w\-]*", re.UNICODE)
_STOPWORD_TEXT = """a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers how i if in into is it its
    itself just me more most my no nor not now of off on once only or other our ours out
    over own same she should so some such than that the their theirs them then there these
    they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours whether tell show find give explain
    describe best way much many"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def constant(name: str) -> float:
    """A tuning constant: the environment's value, else the ADR default."""
    try:
        return float(os.environ.get(name, _DEFAULTS[name]))
    except ValueError:
        return _DEFAULTS[name]


def sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    expo = math.exp(logit)
    return expo / (1.0 + expo)


# ── the cut ─────────────────────────────────────────────────────────────────


def _largest_gap_cut(
    kept: list[Candidate], values: list[float], min_gap: float
) -> list[Candidate]:
    """Cut after the largest consecutive drop, when it is at least ``min_gap``."""
    if len(kept) < 2:
        return kept
    gaps = [values[i] - values[i + 1] for i in range(len(kept) - 1)]
    widest = max(range(len(gaps)), key=lambda i: (gaps[i], -i))
    return kept[: widest + 1] if gaps[widest] >= min_gap else kept


def _cos(candidate: Candidate) -> float:
    return candidate.cosine if candidate.cosine is not None else -1.0


def _cut_dense(candidates: list[Candidate], k_max: int) -> list[Candidate]:
    floor = _cos(candidates[0]) - constant("UMRS_RAG_DELTA")
    kept = [candidates[0]]
    for candidate in candidates[1:]:
        if _cos(candidate) < floor:
            break
        kept.append(candidate)
    kept = kept[:k_max]
    return _largest_gap_cut(kept, [_cos(c) for c in kept], constant("UMRS_RAG_GAMMA"))


def _best_lexical(candidates: Sequence[Candidate]) -> float:
    return max((c.lane_scores.get("lex", 0.0) for c in candidates), default=0.0)


def _passes_fused(
    candidate: Candidate, floor: float | None, lex_floor: float | None
) -> bool:
    """The dense rule (a candidate without a cosine never passes it) or the
    lexical rule."""
    cosine = candidate.cosine
    if floor is not None and cosine is not None and cosine >= floor:
        return True
    lex = candidate.lane_scores.get("lex")
    return lex_floor is not None and lex is not None and lex >= lex_floor


def _dense_floor(candidates: Sequence[Candidate]) -> float | None:
    """``best known cosine of the top 10 - delta``; ``None`` when none of them
    has a cosine (an encoder outage, or no vector in the query's space): the
    lexical rule then decides alone (review core-5)."""
    known = [c.cosine for c in candidates[:_ANCHOR_TOP] if c.cosine is not None]
    return max(known) - constant("UMRS_RAG_DELTA") if known else None


def _cut_fused(candidates: list[Candidate], k_max: int) -> list[Candidate]:
    floor = _dense_floor(candidates)
    best_lex = _best_lexical(candidates)
    lex_floor = constant("UMRS_RAG_LEX_RATIO") * best_lex if best_lex > 0 else None
    kept = [candidates[0]]
    for candidate in candidates[1:]:
        if len(kept) >= k_max or not _passes_fused(candidate, floor, lex_floor):
            break
        kept.append(candidate)
    return kept


def _logit(candidate: Candidate) -> float:
    return float(candidate.rerank_logit or 0.0)


def _cut_rerank(candidates: list[Candidate], k_max: int) -> list[Candidate]:
    scored = [c for c in candidates if c.rerank_logit is not None]
    if not scored:
        return candidates[:1]
    top = _logit(scored[0])
    p_min, window = (
        constant("UMRS_RAG_RERANK_P_MIN"),
        constant("UMRS_RAG_RERANK_WINDOW"),
    )
    kept = [scored[0]]
    for candidate in scored[1:]:
        logit = _logit(candidate)
        if sigmoid(logit) < p_min or logit < top - window:
            break
        kept.append(candidate)
    kept = kept[:k_max]
    return _largest_gap_cut(
        kept, [_logit(c) for c in kept], constant("UMRS_RAG_RERANK_GAP")
    )


def _cut_legacy(candidates: list[Candidate], spec: RetrieveSpec) -> list[Candidate]:
    """The pre-ADR-093 filter: top ``k`` at or above the absolute threshold
    (the K3 score, a cosine clipped at 0)."""
    threshold = float(spec.threshold or 0.0)
    return [c for c in candidates[: spec.k_max] if max(0.0, _cos(c)) >= threshold]


_CUTS = {"dense": _cut_dense, "fused": _cut_fused, "rerank": _cut_rerank}


def cut(
    candidates: list[Candidate], spec: RetrieveSpec, *, mode: str
) -> list[Candidate]:
    """The candidates the model may see, best first."""
    if mode not in CUT_MODES:
        raise ValueError(f"unknown cut mode {mode!r}; one of {CUT_MODES}")
    if mode == "legacy":
        return _cut_legacy(candidates, spec)
    k_max = max(1, spec.k_max)
    if not candidates:
        return []
    if spec.cutoff == "fixed":
        return list(candidates[:k_max])
    return _CUTS[mode](list(candidates), k_max)


# ── lexical evidence ────────────────────────────────────────────────────────


def content_terms(query: str) -> list[str]:
    """The query's content words: lower-case, three characters or more, no
    stop words, first occurrence order."""
    seen: dict[str, None] = {}
    for term in _TERM_RE.findall(query.lower()):
        if len(term) >= 3 and term not in _STOPWORDS:
            seen.setdefault(term, None)
    return list(seen)


def _matched_terms(query: str, text: str) -> tuple[int, int]:
    """``(content terms that start a word in text, content terms)``."""
    terms = content_terms(query)
    lowered = text.lower()
    found = sum(1 for t in terms if re.search(r"(?<!\w)" + re.escape(t), lowered))
    return found, len(terms)


def term_coverage(query: str, text: str) -> float:
    """Share of the query's content terms that start a word in ``text``."""
    found, total = _matched_terms(query, text)
    return found / total if total else 0.0


@dataclass(frozen=True)
class LexicalEvidence:
    """What the lexical rule sees in the KEPT passages (ADR-083 D-5, C2).

    ``coverage``/``matched``/``terms``: the best-covered query variant's
    content terms that start a word in the TOP kept passage. ``strength``:
    the largest IDF of one matched query lexeme in any kept passage (the lex
    lane's ``lex:idf_max``; 0 without the lane). Truthy when either signal
    holds; :func:`sufficiency` reads which one.
    """

    coverage: float = 0.0
    matched: int = 0
    terms: int = 0
    strength: float = 0.0

    @property
    def covers(self) -> bool:
        """At least ``UMRS_RAG_LEX_COVERAGE`` of the terms, and at least
        ``UMRS_RAG_LEX_MIN_TERMS`` of them (every term of a shorter query):
        one shared word never covers a longer question."""
        need = max(1, min(int(constant("UMRS_RAG_LEX_MIN_TERMS")), self.terms))
        enough = self.coverage >= constant("UMRS_RAG_LEX_COVERAGE")
        return self.terms > 0 and self.matched >= need and enough

    @property
    def strong(self) -> bool:
        """A matched lexeme at least ``UMRS_RAG_LEX_HIT_IDF`` rare."""
        return self.strength >= constant("UMRS_RAG_LEX_HIT_IDF")

    def __bool__(self) -> bool:
        return self.covers or self.strong

    def describe(self) -> str:
        if self.covers:
            return f"{self.matched} of {self.terms} query terms in the top passage"
        return f"a matched query lexeme of idf {self.strength:.2f} in a kept passage"


def _best_coverage(queries: Sequence[str], text: str) -> tuple[float, int, int]:
    best = (0.0, 0, 0)
    for query in queries:
        found, total = _matched_terms(query, text)
        best = max(best, (found / total if total else 0.0, found, total))
    return best


def lexical_evidence(
    query: str | Sequence[str],
    kept: Sequence[Candidate],
    *,
    strength: float | None = None,
) -> LexicalEvidence:
    """The lexical rule's inputs over the passages the reader is shown.

    ``kept`` must be the KEPT passages, best first — never the whole lexical
    candidate list (the pipeline and K3 both call this). ``query`` is the
    question or its variants; the best-covered one counts. ``strength``
    overrides the largest ``lex:idf_max`` among ``kept``.
    """
    if not kept:
        return LexicalEvidence()
    queries = (query,) if isinstance(query, str) else tuple(query)
    share, found, total = _best_coverage(queries, kept[0].text)
    if strength is None:
        strength = max(float(c.lane_scores.get(_LEX_IDF_MAX, 0.0)) for c in kept)
    return LexicalEvidence(share, found, total, float(strength))


def lexical_hit(
    query: str | Sequence[str],
    kept: Sequence[Candidate],
    *,
    strength: float | None = None,
) -> LexicalEvidence:
    """The name K3 and older callers use for :func:`lexical_evidence`; the
    result is truthy on either signal, and :func:`sufficiency` decides what
    each may do (a rare lexeme only lifts a partial ``z_null``)."""
    return lexical_evidence(query, kept, strength=strength)


def _as_evidence(hit: bool | LexicalEvidence) -> LexicalEvidence:
    """A bare ``True`` is coverage-grade evidence (the S1 rule's answer)."""
    if isinstance(hit, LexicalEvidence):
        return hit
    return LexicalEvidence(1.0, 1, 1, 0.0) if hit else LexicalEvidence()


# ── sufficiency ─────────────────────────────────────────────────────────────


def _verdict(
    status: str,
    basis: str,
    value: float | None,
    threshold: float | None,
    detail: str = "",
) -> RetrievalSufficiency:
    rounded = None if value is None else round(float(value), 4)
    return RetrievalSufficiency(
        status=status, basis=basis, value=rounded, threshold=threshold, detail=detail
    )


def _rerank_sufficiency(kept: Sequence[Candidate]) -> RetrievalSufficiency:
    p_1 = sigmoid(_logit(kept[0]))
    floor = constant("UMRS_RAG_NULL_P")
    if p_1 < floor:
        return _verdict("insufficient", "rerank_p", p_1, floor)
    enough = constant("UMRS_RAG_RERANK_P_SUFFICIENT")
    return _verdict(
        "sufficient" if p_1 >= enough else "partial", "rerank_p", p_1, enough
    )


def _z_sufficiency(
    kept: Sequence[Candidate], null: Any, evidence: LexicalEvidence
) -> RetrievalSufficiency:
    """``z_null`` decides; lexical evidence only lifts the partial band."""
    best = max(_cos(c) for c in kept)
    z = (best - float(null.mu)) / max(float(null.sigma), 1e-6)
    high, low = constant("UMRS_RAG_Z_SUFFICIENT"), constant("UMRS_RAG_Z_PARTIAL")
    if z >= high:
        return _verdict("sufficient", "z_null", z, high)
    if z < low:
        detail = (
            "a lexical hit never overrides an insufficient z_null" if evidence else ""
        )
        return _verdict("insufficient", "z_null", z, low, detail)
    if evidence:
        lift = f"partial z_null lifted by {evidence.describe()}"
        return _verdict("sufficient", "lexical", z, high, lift)
    return _verdict("partial", "z_null", z, high)


def _legacy_sufficiency(
    kept: Sequence[Candidate], threshold: float | None
) -> RetrievalSufficiency:
    if not kept:
        return _verdict(
            "insufficient",
            "threshold",
            None,
            threshold,
            f"no passage scored at or above {threshold}",
        )
    return _verdict("sufficient", "threshold", max(0.0, _cos(kept[0])), threshold)


def _no_null(evidence: LexicalEvidence) -> RetrievalSufficiency:
    """No calibrated null (fewer than 30 vectors in scope, or the probes could
    not be encoded): there is no z, so no ``partial``; only the coverage rule
    can establish sufficiency (C2), never a rare lexeme alone."""
    floor = constant("UMRS_RAG_LEX_COVERAGE")
    if evidence.covers:
        detail = f"no calibrated null; {evidence.describe()}"
        return _verdict("sufficient", "lexical", evidence.coverage, floor, detail)
    detail = "no calibrated null, and the query's terms do not cover the top passage"
    return _verdict("insufficient", "lexical", evidence.coverage, floor, detail)


def sufficiency(
    kept: Sequence[Candidate],
    *,
    mode: str,
    null: Any | None,
    lexical_hit: bool | LexicalEvidence,
    searched: int,
    threshold: float | None = None,
) -> RetrievalSufficiency:
    """Whether ``kept`` answers anything, and on what basis.

    ``lexical_hit`` is :func:`lexical_evidence` over ``kept`` (a bare bool is
    read as coverage-grade evidence). ``searched`` is the number of vectors
    in scope in the query's space: none means insufficient in the dense and
    fused modes, whatever the lexical lanes found (C2).
    """
    if mode == "legacy":
        return _legacy_sufficiency(kept, threshold)
    if not kept:
        basis = "no_vectors" if searched == 0 else "empty"
        return _verdict("insufficient", basis, None, None, "nothing in scope matched")
    if mode == "rerank":
        return _rerank_sufficiency(kept)
    if searched <= 0:
        detail = "no vector in scope in the query's space"
        return _verdict("insufficient", "no_vectors", None, None, detail)
    evidence = _as_evidence(lexical_hit)
    if null is None:
        return _no_null(evidence)
    return _z_sufficiency(kept, null, evidence)
