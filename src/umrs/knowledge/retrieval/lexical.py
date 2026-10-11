"""Lexical recall lanes of the retrieval core (ADR-093 D-6, component C6).

``lex``: OR-mode full-text search over ``knowledge_nodes.search_text`` (the
migration 0120 column), weighted by corpus IDF. Postgres' built-in ranking
knows no corpus statistics, so the lane counts them itself: the document
frequency of each query lexeme and the number of indexed chunks in the
caller's scope, both through the expression index, cached for ten minutes.

``ident``: identifier tokens (IL-6, TiO2, ZSM-5, BRCA1), which English
stemming splits or mangles, matched by escaped ILIKE on the trigram index.

Composition-layer code: nothing here imports ``umrs.api``. User text reaches
SQL only as a bound parameter. The OR tsquery is assembled inside SQL from the
lexemes Postgres extracted (``tsvector_to_array`` + ``quote_literal``), never
by concatenating query text. Both lanes read exactly the dense lane's scope
(``dense.gate_clause`` and its exclusions): chunk rows, the ADR-050
visibility predicate, and the graph predicate with the legacy
``graph_id = ''`` rule when graphs are selected.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from umrs.knowledge.access_scope import artifact_alive_predicate

from .dense import _exclude_clause, gate_clause
from .types import Candidate, LaneResult, RetrieveSpec

logger = logging.getLogger(__name__)

LEX_LANE = "lex"
IDENT_LANE = "ident"
#: Occurrences of the identifier tokens in a chunk, beside ``lane_scores["ident"]``.
IDENT_HITS = "ident:hits"
#: Sub-scores the lex lane adds beside ``lane_scores["lex"]``: the summed and
#: the largest IDF of the query lexemes a chunk contains (sufficiency rules).
LEX_IDF_SUM = "lex:idf_sum"
LEX_IDF_MAX = "lex:idf_max"

LEXICAL_ENV = "UMRS_RAG_LEXICAL"
STATS_TTL_S = 600.0
MAX_DF_RATIO = 0.3
KEEP_RAREST = 3
MAX_LEXEMES = 12
RANK_CD_WEIGHT = 0.1
RANK_CD_NORMALIZATION = 32  # rank / (rank + 1)
LEX_LIMIT = 50
IDENT_LIMIT = 20
IDENT_MAX_TOKENS = 4
IDENT_MIN_CHARS = 3
#: Without the trigram index the ident lane is a sequential scan: bounded.
IDENT_SCAN_MAX_ROWS = 50_000
FTS_INDEX = "idx_knowledge_nodes_search_text_fts"
TRGM_INDEX = "idx_knowledge_nodes_search_text_trgm"
BACKFILL_HINT = "python -m umrs.knowledge.retrieval.backfill --project <id>"

_SCHEMA_SQL = (
    "/* lex:schema */ SELECT EXISTS (SELECT 1 FROM pg_attribute "
    "WHERE attrelid = to_regclass('knowledge_nodes') AND attname = 'search_text' "
    "AND NOT attisdropped), to_regclass(%s) IS NOT NULL, to_regclass(%s) IS NOT NULL"
)

# Lexemes of the query, as Postgres' english configuration derives them. A
# lexeme holding a backslash would make quote_literal emit E'' syntax that
# the tsquery parser rejects; such tokens are dropped.
_LEXEMES_SQL = (
    "/* lex:lexemes */ SELECT DISTINCT lex "
    "FROM unnest(tsvector_to_array(to_tsvector('english', %s))) AS lex "
    "WHERE strpos(lex, chr(92)) = 0 ORDER BY lex"
)

_MATCH_ONE = "to_tsvector('english', n.search_text) @@ quote_literal(t.lex)::tsquery"


@dataclass(frozen=True)
class _Schema:
    column: bool
    fts_index: bool
    trgm_index: bool


class _TTLCache:
    """A small thread-safe LRU whose entries expire after ``ttl`` seconds."""

    def __init__(self, ttl: float, max_entries: int = 8192) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self.clock: Callable[[], float] = time.monotonic
        self._data: OrderedDict[Hashable, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            if self.clock() - item[1] > self.ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return item[0]

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._data[key] = (value, self.clock())
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_STATS = _TTLCache(STATS_TTL_S)


def clear_caches() -> None:
    """Forget the cached corpus statistics and schema probe (tests, re-ingest)."""
    _STATS.clear()


# ── scope ────────────────────────────────────────────────────────────────────


def scope_predicate(spec: RetrieveSpec) -> tuple[str, list[Any]]:
    """``(sql, params)`` over alias ``n``: the chunk rows the dense lane may
    rank for ``spec`` (its ownership/visibility gate, graph selection with
    the legacy ``graph_id = ''`` rule, and exclusions), without the space —
    and, like it, only chunks whose artifact is alive (wave F)."""
    gate, params = gate_clause(spec)
    exclude, exclude_params = _exclude_clause(spec)
    alive = artifact_alive_predicate("n")
    return f"{gate} AND n.content->>'kind' = 'chunk' AND {alive}{exclude}", [
        *params,
        *exclude_params,
    ]


def _scope_key(spec: RetrieveSpec) -> tuple[Any, ...]:
    graphs = None if spec.graph_ids is None else tuple(sorted(spec.graph_ids))
    return (
        spec.project_id,
        graphs,
        tuple(sorted(spec.exclude_graph_ids)),
        spec.include_ambient,
    )


# ── switches and pure helpers ────────────────────────────────────────────────


def lexical_enabled(mode: str) -> bool:
    """Resolve a request's ``lexical`` (auto|on|off) against ``UMRS_RAG_LEXICAL``.

    ``auto`` follows the site; a site ``auto`` runs the lanes wherever the
    column and its index exist (they report ``unavailable`` otherwise).
    """
    choice = (mode or "auto").strip().lower()
    if choice == "auto":
        choice = os.environ.get(LEXICAL_ENV, "auto").strip().lower()
    return choice != "off"


def idf(df: int, total: int) -> float:
    """BM25 IDF, ``ln(1 + (N - df + 0.5) / (df + 0.5))``; never negative."""
    return math.log1p((max(total - df, 0) + 0.5) / (df + 0.5))


def select_terms(
    frequencies: Mapping[str, int], total: int
) -> tuple[list[tuple[str, int, float]], list[str]]:
    """``(kept, dropped)``: kept ``(lexeme, df, idf)`` rarest first.

    Lexemes in no indexed chunk cannot match and are skipped. Lexemes in more
    than ``MAX_DF_RATIO`` of the scope are dropped, unless none survives, then
    the ``KEEP_RAREST`` rarest stay. At most ``MAX_LEXEMES`` are kept.
    """
    present = _rarest_first(frequencies)
    total = max(total, 1)
    rare = [(lex, df) for lex, df in present if df / total <= MAX_DF_RATIO]
    kept = (rare or present[:KEEP_RAREST])[:MAX_LEXEMES]
    names = {lex for lex, _ in kept}
    dropped = [lex for lex, _ in present if lex not in names]
    return [(lex, df, idf(df, total)) for lex, df in kept], dropped


def _rarest_first(frequencies: Mapping[str, int]) -> list[tuple[str, int]]:
    """Lexemes that occur in scope, rarest first (ties by lexeme)."""
    present = [(lex, df) for lex, df in frequencies.items() if df > 0]
    return sorted(present, key=lambda item: (item[1], item[0]))


_TOKEN_RE = re.compile(r"\w[\w\-+.]*")


def _is_identifier(token: str) -> bool:
    if len(token) < IDENT_MIN_CHARS or not any(ch.isalpha() for ch in token):
        return False
    return any(ch.isdigit() for ch in token) or sum(ch.isupper() for ch in token) >= 2


def identifier_tokens(query: str) -> list[str]:
    """Up to ``IDENT_MAX_TOKENS`` identifier-like tokens, in query order.

    A token of 3 or more characters with a digit or 2 or more capitals (IL-6,
    TiO2, ZSM-5, BRCA1). Purely numeric tokens (years, doses) are not
    identifiers. Case-insensitive duplicates count once (ILIKE).
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in _TOKEN_RE.findall(query or ""):
        token = raw.rstrip(".-")
        if not _is_identifier(token) or token.lower() in seen:
            continue
        seen.add(token.lower())
        tokens.append(token)
    return tokens[:IDENT_MAX_TOKENS]


def like_literal(value: str) -> str:
    """Escape LIKE metacharacters (backslash is Postgres' default escape)."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def lexical_hit_strength(
    candidates: Iterable[Candidate], *, per_term: bool = False
) -> float:
    """The best lexical evidence among ``candidates``: their largest IDF sum
    over matched query lexemes, or with ``per_term`` the largest single
    matched IDF; ``0.0`` when no candidate came through the lex lane."""
    key = LEX_IDF_MAX if per_term else LEX_IDF_SUM
    return max((float(c.lane_scores.get(key, 0.0)) for c in candidates), default=0.0)


def _as_dict(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    try:
        value = json.loads(content or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _candidate(
    node_id: str, content: Any, lane: str, rank: int, scores: dict[str, float]
) -> Candidate:
    body = _as_dict(content)
    return Candidate(
        node_id=str(node_id),
        artifact_id=str(body.get("artifact_id") or ""),
        chunk_index=_as_int(body.get("chunk_index")),
        content=body,
        lane_scores=scores,
        lane_ranks={lane: rank},
    )


# ── database plumbing ────────────────────────────────────────────────────────


@contextmanager
def _cursor(pool: Any) -> Iterator[Any]:
    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                yield cur
    finally:
        pool.putconn(conn)


def _schema(cur: Any) -> _Schema:
    cached = _STATS.get(("schema",))
    if cached is not None:
        return cached
    cur.execute(_SCHEMA_SQL, (FTS_INDEX, TRGM_INDEX))
    row = cur.fetchone() or (False, False, False)
    schema = _Schema(bool(row[0]), bool(row[1]), bool(row[2]))
    _STATS.put(("schema",), schema)
    return schema


def _scope_rows(cur: Any, spec: RetrieveSpec) -> int:
    """Chunk rows with ``search_text`` in the spec's scope (cached)."""
    key = ("rows", _scope_key(spec))
    cached = _STATS.get(key)
    if cached is not None:
        return int(cached)
    scope, params = scope_predicate(spec)
    cur.execute(
        "/* lex:rows */ SELECT count(*) FROM knowledge_nodes n "
        f"WHERE {scope} AND n.search_text IS NOT NULL",
        params,
    )
    row = cur.fetchone()
    total = int(row[0]) if row else 0
    _STATS.put(key, total)
    return total


def _query_lexemes(cur: Any, query: str) -> list[str]:
    cur.execute(_LEXEMES_SQL, (query,))
    return [str(row[0]) for row in cur.fetchall()]


def _frequencies(cur: Any, spec: RetrieveSpec, lexemes: list[str]) -> dict[str, int]:
    """Document frequency per lexeme in scope; one statement for the uncached."""
    scope_key = _scope_key(spec)
    out: dict[str, int] = {}
    missing: list[str] = []
    for lex in lexemes:
        cached = _STATS.get(("df", scope_key, lex))
        if cached is None:
            missing.append(lex)
        else:
            out[lex] = int(cached)
    if missing:
        scope, params = scope_predicate(spec)
        cur.execute(
            "/* lex:df */ SELECT t.lex, (SELECT count(*) FROM knowledge_nodes n "
            f"WHERE {scope} AND n.search_text IS NOT NULL AND {_MATCH_ONE}) "
            "FROM unnest(%s::text[]) AS t(lex)",
            [*params, missing],
        )
        for lex, df in cur.fetchall():
            out[str(lex)] = int(df)
            _STATS.put(("df", scope_key, str(lex)), int(df))
    return out


def _search_sql(scope: str) -> str:
    """The lex lane statement.

    The IDF sum comes from the index alone: one parameterized GIN probe per
    kept lexeme, whose heap visits only check the scope (a plain-lexeme GIN
    match needs no recheck), so no chunk is re-tokenized to score it. Only
    the chunks the cover-density term can still reorder into the top
    ``limit`` (IDF sum within ``RANK_CD_WEIGHT`` of the limit-th best; at most
    ``2 x limit``) pay for ``to_tsvector`` (about 0.1 ms per 500 characters,
    measured on the stand-alone Postgres 15, 2026-10-10). Ties on the IDF sum
    (every match of a one-lexeme query) go to the SHORTER chunk, BM25's
    length normalisation for equal matches, never to node-id order (review
    core-12); ``octet_length`` reads the TOAST header, not the text.

    Parameter order: lexemes, idfs, scope params, limit, rank weight,
    rank pool, rank weight, rank normalization, limit.
    """
    return (
        "/* lex:search */ WITH terms AS MATERIALIZED ("
        " SELECT t.lex, t.idf, quote_literal(t.lex)::tsquery AS tsq"
        " FROM unnest(%s::text[], %s::float8[]) AS t(lex, idf)"
        "), matched AS MATERIALIZED ("
        " SELECT m.node_id, min(m.len) AS len,"
        " sum(t.idf) AS idf_sum, max(t.idf) AS idf_max"
        " FROM terms t CROSS JOIN LATERAL ("
        " SELECT n.node_id, octet_length(n.search_text) AS len FROM knowledge_nodes n"
        f" WHERE {scope} AND n.search_text IS NOT NULL"
        " AND to_tsvector('english', n.search_text) @@ t.tsq) m"
        " GROUP BY m.node_id"
        "), shortlist AS MATERIALIZED ("
        " SELECT node_id, len, idf_sum, idf_max FROM matched"
        " WHERE idf_sum >= (SELECT coalesce(min(b.idf_sum), 0) FROM ("
        " SELECT idf_sum FROM matched ORDER BY idf_sum DESC LIMIT %s) b) - %s"
        " ORDER BY idf_sum DESC, len, node_id LIMIT %s"
        ") SELECT n.node_id, n.content,"
        " p.idf_sum + %s * ts_rank_cd(to_tsvector('english', n.search_text),"
        " (SELECT string_agg(quote_literal(lex), ' | ')::tsquery FROM terms), %s) AS score,"
        " p.idf_sum, p.idf_max"
        " FROM shortlist p JOIN knowledge_nodes n ON n.node_id = p.node_id"
        " ORDER BY score DESC, p.len, n.node_id LIMIT %s"
    )


def _search(
    cur: Any, spec: RetrieveSpec, terms: list[tuple[str, int, float]], limit: int
) -> list[Candidate]:
    scope, params = scope_predicate(spec)
    cur.execute(
        _search_sql(scope),
        [
            [lex for lex, _, _ in terms],
            [weight for _, _, weight in terms],
            *params,
            limit,
            RANK_CD_WEIGHT,
            2 * limit,
            RANK_CD_WEIGHT,
            RANK_CD_NORMALIZATION,
            limit,
        ],
    )
    return [
        _candidate(
            node_id,
            content,
            LEX_LANE,
            rank,
            {
                LEX_LANE: float(score),
                LEX_IDF_SUM: float(idf_sum),
                LEX_IDF_MAX: float(idf_max),
            },
        )
        for rank, (node_id, content, score, idf_sum, idf_max) in enumerate(
            cur.fetchall(), 1
        )
    ]


# ── lanes ────────────────────────────────────────────────────────────────────


def _unavailable(lane: str, reason: str, **info: Any) -> LaneResult:
    return LaneResult(name=lane, status="unavailable", reason=reason, info=dict(info))


def _lex_preconditions(cur: Any, spec: RetrieveSpec) -> tuple[int, LaneResult | None]:
    """``(indexed rows in scope, None)``, or a result saying why the lane cannot run."""
    schema = _schema(cur)
    if not schema.column:
        return 0, _unavailable(
            LEX_LANE, "knowledge_nodes.search_text is missing (migration 0120)"
        )
    if not schema.fts_index:
        return 0, _unavailable(
            LEX_LANE, f"index {FTS_INDEX} is missing (migration 0120)"
        )
    total = _scope_rows(cur, spec)
    if total == 0:
        return 0, _unavailable(
            LEX_LANE, f"no chunk in scope has search_text yet: run {BACKFILL_HINT}"
        )
    return total, None


def _no_terms_reason(lexemes: list[str]) -> str:
    if not lexemes:
        return "the query has no indexable terms"
    return "no query term occurs in the indexed chunks"


def _run_lex(pool: Any, query: str, spec: RetrieveSpec, limit: int) -> LaneResult:
    with _cursor(pool) as cur:
        total, blocked = _lex_preconditions(cur, spec)
        if blocked is not None:
            return blocked
        lexemes = _query_lexemes(cur, query)
        terms, dropped = select_terms(_frequencies(cur, spec, lexemes), total)
        candidates = _search(cur, spec, terms, limit) if terms else []
    info: dict[str, Any] = {
        "searched": total,
        "terms": [
            {"lexeme": lex, "df": df, "idf": round(w, 4)} for lex, df, w in terms
        ],
    }
    if dropped:
        info["dropped"] = dropped
    reason = "" if terms else _no_terms_reason(lexemes)
    return LaneResult(name=LEX_LANE, candidates=candidates, reason=reason, info=info)


def _timed(lane: str, run: Callable[[], LaneResult]) -> LaneResult:
    """Run a lane; a failure is reported by its exception TYPE only — the
    message (hosts, users, SQL) goes to the server log (review security-5)."""
    started = time.perf_counter()
    try:
        result = run()
    except Exception as exc:  # noqa: BLE001 — a lane failure is reported, never raised (ADR-083 D-4)
        logger.warning("%s lane failed: %s", lane, exc)
        result = _unavailable(lane, f"{lane} lane failed ({type(exc).__name__})")
    result.ms = round((time.perf_counter() - started) * 1000.0, 2)
    return result


def lexical_lane(
    pool: Any, query: str, spec: RetrieveSpec, *, limit: int = LEX_LIMIT
) -> LaneResult:
    """Lane ``lex``: IDF-weighted OR full-text search over chunk ``search_text``.

    score = sum of IDF over the kept query lexemes a chunk contains, plus
    ``0.1 x ts_rank_cd(., 32)``. ``off`` when ``spec.lexical`` resolves off;
    ``unavailable`` (with the reason) when migration 0120 is missing, nothing
    in scope is indexed yet, or the statement fails.
    """
    if not lexical_enabled(spec.lexical):
        return LaneResult(name=LEX_LANE, status="off", reason="lexical lanes are off")
    return _timed(LEX_LANE, lambda: _run_lex(pool, query, spec, limit))


#: Case-insensitive occurrences of one token (bound twice) in a chunk.
_IDENT_COUNT = (
    "((length(lower(n.search_text)) - length(replace(lower(n.search_text), "
    "lower(%s), ''))) / length(lower(%s)))"
)


def _ident_sql(scope: str, n_tokens: int) -> str:
    """Ranked by tokens matched, then by their occurrences, then shorter chunk
    first (BM25's tie rule), never by node id alone (review core-12).

    Parameter order: patterns (matched count), each token twice (occurrences),
    scope params, patterns, limit.
    """
    hit = "(n.search_text ILIKE %s)"
    matched = " + ".join([f"{hit}::int"] * n_tokens)
    hits = " + ".join([_IDENT_COUNT] * n_tokens)
    any_hit = " OR ".join([hit] * n_tokens)
    return (
        f"/* ident:search */ SELECT n.node_id, n.content, ({matched}) AS matched, "
        f"({hits}) AS hits "
        f"FROM knowledge_nodes n WHERE {scope} AND n.search_text IS NOT NULL "
        f"AND ({any_hit}) "
        "ORDER BY matched DESC, hits DESC, octet_length(n.search_text), n.node_id "
        "LIMIT %s"
    )


def _ident_scores(row: tuple) -> dict[str, float]:
    scores = {IDENT_LANE: float(row[2])}
    if len(row) > 3 and row[3] is not None:
        scores[IDENT_HITS] = float(row[3])
    return scores


def _ident_blocked(cur: Any, spec: RetrieveSpec) -> tuple[str, LaneResult | None]:
    """``(access path, None)``, or a result saying why the lane cannot run."""
    schema = _schema(cur)
    if not schema.column:
        return "", _unavailable(
            IDENT_LANE, "knowledge_nodes.search_text is missing (migration 0120)"
        )
    if schema.trgm_index:
        return "trigram", None
    total = _scope_rows(cur, spec)
    if total > IDENT_SCAN_MAX_ROWS:
        return "", _unavailable(
            IDENT_LANE,
            f"no trigram index (pg_trgm) and {total} chunk rows in scope "
            f"(a scan runs only up to {IDENT_SCAN_MAX_ROWS})",
        )
    return "scan", None


def _run_ident(
    pool: Any, tokens: list[str], spec: RetrieveSpec, limit: int
) -> LaneResult:
    patterns = [f"%{like_literal(token)}%" for token in tokens]
    counted = [value for token in tokens for value in (token, token)]
    with _cursor(pool) as cur:
        access, blocked = _ident_blocked(cur, spec)
        if blocked is not None:
            return blocked
        scope, params = scope_predicate(spec)
        cur.execute(
            _ident_sql(scope, len(patterns)),
            [*patterns, *counted, *params, *patterns, limit],
        )
        rows = cur.fetchall()
    candidates = [
        _candidate(row[0], row[1], IDENT_LANE, rank, _ident_scores(row))
        for rank, row in enumerate(rows, 1)
    ]
    return LaneResult(
        name=IDENT_LANE,
        candidates=candidates,
        info={"tokens": tokens, "access": access},
    )


def ident_lane(
    pool: Any, query: str, spec: RetrieveSpec, *, limit: int = IDENT_LIMIT
) -> LaneResult:
    """Lane ``ident``: chunks containing the query's identifier tokens, ranked
    by how many of them they contain (escaped ILIKE, trigram index)."""
    if not lexical_enabled(spec.lexical):
        return LaneResult(name=IDENT_LANE, status="off", reason="lexical lanes are off")
    tokens = identifier_tokens(query)
    if not tokens:
        return LaneResult(
            name=IDENT_LANE,
            reason="no identifier tokens in the query",
            info={"tokens": []},
        )
    return _timed(IDENT_LANE, lambda: _run_ident(pool, tokens, spec, limit))


__all__ = [
    "IDENT_HITS",
    "IDENT_LANE",
    "LEX_IDF_MAX",
    "LEX_IDF_SUM",
    "LEX_LANE",
    "clear_caches",
    "ident_lane",
    "identifier_tokens",
    "idf",
    "lexical_enabled",
    "lexical_hit_strength",
    "lexical_lane",
    "like_literal",
    "scope_predicate",
    "select_terms",
]
