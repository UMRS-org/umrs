"""Catalog search scoring helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import CatalogObjectMeta, CatalogState

_STRUCTURED_QUERY_HINTS = frozenset(
    {"csv", "tsv", "json", "column", "columns", "table", "batch", "yield", "data"}
)
# Words that decide nothing: as substrings "in", "a" and "the" occur in almost
# every key or summary, so a nonsense query matched every READY row (LIVE-06
# survived its first fix; review of ADR-093, 2026-10-10).
_MIN_TERM_CHARS = 3
_STOP_WORD_TEXT = """a an as at be by do he i if in is it me my no of on or so to up we
    about above after again against all also and any are because been before
    being below between both but can could did does doing down during each few for
    from further had has have having her here hers him his how into its itself
    just more most not now off once only other our ours out over own same she
    should some such than that the their theirs them then there these they this
    those through too under until very was were what when where which while who
    whom why will with would you your yours"""
_STOP_WORDS = frozenset(_STOP_WORD_TEXT.split())
_WORD_SPLIT = re.compile(r"[\W_]+")
# Scripts written without spaces (Han, kana, Hangul): a Japanese summary is
# one "word", so a term matches anywhere inside it, never only at a word
# start (review of ADR-093, 2026-10-10: word starts zeroed every Japanese
# catalogue query). Two such characters already make a word (収率, 測定).
_CJK = re.compile(
    "[\u3040-\u30ff\u31f0-\u31ff\u3400-\u4dbf\u4e00-\u9fff"
    "\uf900-\ufaff\uac00-\ud7af\uff66-\uff9f]"
)
# One kana alone is a particle (の, は, を), which decides nothing.
_LONE_KANA = re.compile("[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]")
_MIN_CJK_TERM_CHARS = 2


@dataclass
class CatalogHit:
    """One catalog search result."""

    meta: CatalogObjectMeta
    score: float


def allowed(meta: CatalogObjectMeta, project_id: str) -> bool:
    if meta.acl_scope == project_id:
        return True
    return meta.s3_key.startswith(f"{project_id}/")


def tokenize(query: str) -> list[str]:
    return [t for t in re.split(r"\W+", query.lower()) if t]


def _haystack(meta: CatalogObjectMeta) -> str:
    """Every searchable text field of a row, lowercased."""
    return " ".join(
        [
            meta.summary,
            meta.deterministic_summary,
            meta.perception_summary or "",
            meta.detected_modality,
            meta.s3_key,
            " ".join(meta.tags),
            " ".join(meta.filename_signals),
        ]
    ).lower()


def is_cjk(term: str) -> bool:
    """The term holds a character of a script written without spaces."""
    return bool(_CJK.search(term))


def _decides(term: str) -> bool:
    return term not in _STOP_WORDS and not _LONE_KANA.fullmatch(term)


def _long(term: str) -> bool:
    if is_cjk(term):
        return len(term) >= _MIN_CJK_TERM_CHARS
    return len(term) >= _MIN_TERM_CHARS


def content_terms(tokens: list[str]) -> list[str]:
    """The query tokens that can decide a match: three characters or more
    (two in Han, kana or Hangul) and no stop word or lone particle; a query
    made only of short tokens keeps its non-stop ones."""
    kept = [t for t in tokens if _decides(t)]
    long_terms = [t for t in kept if _long(t)]
    return long_terms or kept


def _words(haystack: str) -> set[str]:
    """Whole words of the row, also split at ``_`` and ``-`` (``yield_pct``
    holds ``yield_pct``, ``yield`` and ``pct``)."""
    return set(re.split(r"\W+", haystack)) | set(_WORD_SPLIT.split(haystack))


def _term_hits(term: str, words: set[str], haystack: str) -> bool:
    """A term matches a word it begins (``yield`` matches ``yields``); a
    Han/kana/Hangul term matches anywhere in the row's text, which has no
    word boundaries to start at (``収率`` in ``バッチ3の収率の測定結果``)."""
    if is_cjk(term):
        return term in haystack
    return any(word.startswith(term) for word in words if word)


def token_matches(meta: CatalogObjectMeta, tokens: list[str]) -> int:
    """How many of the query's content terms begin a word of the row (or,
    in a script written without spaces, occur in it)."""
    haystack = _haystack(meta)
    words = _words(haystack)
    return sum(1 for term in content_terms(tokens) if _term_hits(term, words, haystack))


def _signal_score(token: str, signals: list[str]) -> float:
    """Filename-signal credit: substring 0.5, whole stem word another 0.5."""
    score = 0.0
    for signal in signals:
        signal_lower = signal.lower()
        if token in signal_lower:
            score += 0.5
        stem = signal_lower.rsplit(".", 1)[0]
        if token in stem.replace("-", " ").replace("_", " ").split():
            score += 0.5
    return score


def _token_score(meta: CatalogObjectMeta, tokens: list[str], haystack: str) -> float:
    tag_blob = " ".join(meta.tags).lower()
    score = 0.0
    for token in tokens:
        score += 1.0 if token in haystack else 0.0
        score += 0.75 if token in tag_blob else 0.0
        score += _signal_score(token, meta.filename_signals)
    return score


def _structured_hint(meta: CatalogObjectMeta, query: str) -> float:
    if meta.detected_modality != "structured":
        return 0.0
    query_lower = query.lower()
    return 0.5 if any(hint in query_lower for hint in _STRUCTURED_QUERY_HINTS) else 0.0


def score_meta(
    meta: CatalogObjectMeta,
    tokens: list[str],
    *,
    query: str = "",
) -> float:
    """Lexical relevance of one catalogue row to the query tokens.

    A row no content term matches (as the start of one of its words) scores
    zero, bonuses included (ADR-093 C7, LIVE-06): a floor on every ready row,
    then substring matches of "in" or "the", made a nonsense query return the
    catalogue, so ``knowledge_insufficient`` (zero results) could never fire.
    """
    haystack = _haystack(meta)
    if not haystack or not token_matches(meta, tokens):
        return 0.0
    terms = content_terms(tokens)
    score = _token_score(meta, terms, haystack) + _structured_hint(meta, query)
    if meta.state == CatalogState.READY:
        score += 0.25
    elif meta.state == CatalogState.PENDING:
        score *= 0.5
    return score
