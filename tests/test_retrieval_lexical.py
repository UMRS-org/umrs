"""ADR-093 C6: the lexical lanes (``lex`` and ``ident``) of the retrieval core.

The pool is a routing fake: each statement carries a ``/* lane:step */`` tag,
and the fake answers by tag, so the tests pin behaviour, not statement order.
The SQL shapes themselves were run read-only against the stand-alone
Postgres 15 (2026-10-10); the EXPLAIN check is the integration test in
``test_migration_0120.py``.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest
from umrs.knowledge.retrieval import dense, lexical
from umrs.knowledge.retrieval.types import Candidate, RetrieveSpec

_TAG = re.compile(r"/\*\s*([\w:]+)\s*\*/")


class _Cursor:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool
        self._rows: list[tuple] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        match = _TAG.search(sql)
        tag = match.group(1) if match else "?"
        self._pool.calls.append((tag, sql, list(params or [])))
        answer = self._pool.answers.get(tag, [])
        if isinstance(answer, Exception):
            raise answer
        self._rows = list(answer(sql, list(params or [])) if callable(answer) else answer)

    def fetchone(self) -> Any:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple]:
        return list(self._rows)


class _Conn:
    def __init__(self, pool: _Pool) -> None:
        self._pool = pool

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def cursor(self) -> _Cursor:
        return _Cursor(self._pool)


class _Pool:
    def __init__(self, answers: dict[str, Any]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, str, list[Any]]] = []

    def getconn(self) -> _Conn:
        return _Conn(self)

    def putconn(self, _conn: _Conn) -> None:
        return None

    def tags(self) -> list[str]:
        return [tag for tag, _, _ in self.calls]

    def call(self, tag: str) -> tuple[str, list[Any]]:
        for got, sql, params in self.calls:
            if got == tag:
                return sql, params
        raise AssertionError(f"no {tag} statement in {self.tags()}")


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(lexical.LEXICAL_ENV, raising=False)
    lexical.clear_caches()
    yield
    lexical.clear_caches()


def _spec(**kw: Any) -> RetrieveSpec:
    return RetrieveSpec(queries=(kw.pop("query", "q"),), project_id="proj", **kw)


def _content(artifact: str, index: int) -> dict[str, Any]:
    return {"kind": "chunk", "artifact_id": artifact, "chunk_index": index, "chunk_md5": "m"}


def _lex_answers(**over: Any) -> dict[str, Any]:
    answers: dict[str, Any] = {
        "lex:schema": [(True, True, True)],
        "lex:rows": [(100,)],
        "lex:lexemes": [("brca1",), ("cell",), ("mutat",)],
        "lex:df": lambda _sql, params: [
            (lex, {"brca1": 2, "cell": 60, "mutat": 10}[lex]) for lex in params[-1]
        ],
        "lex:search": [
            ("n-1", _content("art-a", 0), 7.25, 7.2, 4.0),
            ("n-2", _content("art-b", 3), 4.01, 4.0, 4.0),
        ],
    }
    answers.update(over)
    return answers


# ── pure helpers ─────────────────────────────────────────────────────────────


def test_idf_is_the_bm25_form() -> None:
    for df, total in ((1, 100), (30, 100), (100, 100), (0, 10)):
        expected = math.log(1 + (total - df + 0.5) / (df + 0.5))
        assert lexical.idf(df, total) == pytest.approx(expected)
    assert lexical.idf(120, 100) > 0  # stale df above N never goes negative


def test_select_terms_drops_frequent_lexemes_and_orders_rarest_first() -> None:
    kept, dropped = lexical.select_terms({"cell": 60, "brca1": 2, "mutat": 10, "absent": 0}, 100)
    assert [lex for lex, _, _ in kept] == ["brca1", "mutat"]
    assert dropped == ["cell"]  # df/N 0.6 > 0.3; a df of 0 cannot match and is skipped
    assert kept[0][2] == pytest.approx(lexical.idf(2, 100))


def test_select_terms_keeps_three_rarest_when_none_survives() -> None:
    kept, dropped = lexical.select_terms({"a": 90, "b": 40, "c": 50, "d": 70}, 100)
    assert [lex for lex, _, _ in kept] == ["b", "c", "d"]
    assert dropped == ["a"]


def test_select_terms_caps_at_twelve() -> None:
    frequencies = {f"t{i:02d}": i + 1 for i in range(20)}
    kept, dropped = lexical.select_terms(frequencies, 1000)
    assert len(kept) == lexical.MAX_LEXEMES == 12
    assert [lex for lex, _, _ in kept] == [f"t{i:02d}" for i in range(12)]
    assert len(dropped) == 8


def test_identifier_tokens() -> None:
    query = "Does IL-6 or BRCA1 bind TiO2 and ZSM-5, unlike DNA in 1998 at 2.5 mM? il-6 again."
    assert lexical.identifier_tokens(query) == ["IL-6", "BRCA1", "TiO2", "ZSM-5"]
    assert lexical.identifier_tokens("DNA repair in 1998, dose 2.5") == ["DNA"]
    assert lexical.identifier_tokens("plain words only, The End") == []


def test_like_literal_escapes_metacharacters() -> None:
    assert lexical.like_literal(r"a_b%c\d") == r"a\_b\%c\\d"


def test_lexical_switch_follows_request_then_site(monkeypatch: pytest.MonkeyPatch) -> None:
    assert lexical.lexical_enabled("auto") is True
    monkeypatch.setenv(lexical.LEXICAL_ENV, "off")
    assert lexical.lexical_enabled("auto") is False
    assert lexical.lexical_enabled("on") is True
    monkeypatch.setenv(lexical.LEXICAL_ENV, "on")
    assert lexical.lexical_enabled("off") is False


def test_lexical_hit_strength() -> None:
    a = Candidate("a", lane_scores={"lex": 7.3, lexical.LEX_IDF_SUM: 7.2, lexical.LEX_IDF_MAX: 4.0})
    b = Candidate("b", lane_scores={"lex": 3.1, lexical.LEX_IDF_SUM: 3.0, lexical.LEX_IDF_MAX: 3.0})
    dense_only = Candidate("c", lane_scores={"dense": 0.9})
    assert lexical.lexical_hit_strength([b, a, dense_only]) == 7.2
    assert lexical.lexical_hit_strength([b, dense_only], per_term=True) == 3.0
    assert lexical.lexical_hit_strength([dense_only]) == 0.0
    assert lexical.lexical_hit_strength([]) == 0.0


# ── scope ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spec",
    [
        _spec(),
        _spec(graph_ids=("g-1", "g-2")),
        _spec(graph_ids=(), exclude_graph_ids=("g-3",)),
        _spec(include_ambient=True, exclude_graph_ids=("g-3",)),
    ],
)
def test_scope_is_the_dense_lanes(spec: RetrieveSpec) -> None:
    from umrs.knowledge.access_scope import artifact_alive_predicate

    sql, params = lexical.scope_predicate(spec)
    gate, gate_params = dense.gate_clause(spec)
    exclude, exclude_params = dense._exclude_clause(spec)
    alive = artifact_alive_predicate("n")
    assert sql == f"{gate} AND n.content->>'kind' = 'chunk' AND {alive}{exclude}"
    assert params == [*gate_params, *exclude_params]
    assert sql.count("%s") == len(params)


def test_scope_keeps_the_legacy_default_graph_rule() -> None:
    from umrs.knowledge.graph_registry import default_graph_id

    sql, params = lexical.scope_predicate(_spec(graph_ids=(default_graph_id("proj"),)))
    assert f"n.graph_id = '' AND {dense.DEFAULT_GRAPH_SQL} = ANY(%s)" in sql
    # legacy '' rows of the project whose default graph is selected stay visible
    assert params[:2] == [[default_graph_id("proj")], [default_graph_id("proj")]]


# ── lex lane ─────────────────────────────────────────────────────────────────


def test_lex_lane_ranks_by_idf_sum() -> None:
    pool = _Pool(_lex_answers())
    result = lexical.lexical_lane(pool, "BRCA1 mutations in cells", _spec())
    assert (result.status, result.name) == ("ok", "lex")
    assert [c.node_id for c in result.candidates] == ["n-1", "n-2"]
    first = result.candidates[0]
    assert first.lane_scores == {"lex": 7.25, "lex:idf_sum": 7.2, "lex:idf_max": 4.0}
    assert [c.lane_ranks for c in result.candidates] == [{"lex": 1}, {"lex": 2}]
    assert (first.artifact_id, first.chunk_index) == ("art-a", 0)


def test_lex_lane_reports_its_terms_and_binds_their_idf() -> None:
    pool = _Pool(_lex_answers())
    result = lexical.lexical_lane(pool, "BRCA1 mutations in cells", _spec())
    assert [t["lexeme"] for t in result.info["terms"]] == ["brca1", "mutat"]
    assert (result.info["dropped"], result.info["searched"]) == (["cell"], 100)
    _, params = pool.call("lex:search")
    assert params[0] == ["brca1", "mutat"]
    assert params[1] == pytest.approx([lexical.idf(2, 100), lexical.idf(10, 100)])
    assert params[-6:] == [
        50,
        lexical.RANK_CD_WEIGHT,
        100,
        lexical.RANK_CD_WEIGHT,
        lexical.RANK_CD_NORMALIZATION,
        50,
    ]


def test_lex_scores_from_the_index_and_tokenizes_only_the_shortlist() -> None:
    # Re-tokenizing every OR match costs ~0.1 ms per 500 characters: the IDF
    # sum comes from per-lexeme index probes; only the shortlist (IDF sum
    # within the rank weight of the limit-th best, at most 2 x limit) is
    # tokenized for ts_rank_cd.
    pool = _Pool(_lex_answers())
    lexical.lexical_lane(pool, "BRCA1 mutations", _spec(), limit=7)
    sql, params = pool.call("lex:search")
    assert sql.count("to_tsvector(") == 2
    assert "to_tsvector('english', n.search_text) @@ t.tsq" in sql
    assert "FROM shortlist p JOIN knowledge_nodes n" in sql
    assert params[-6:-3] == [7, lexical.RANK_CD_WEIGHT, 14]


def test_lex_query_text_is_only_ever_a_bound_parameter() -> None:
    hostile = "x') OR 1=1; DROP TABLE knowledge_nodes; --"
    pool = _Pool(_lex_answers())
    lexical.lexical_lane(pool, hostile, _spec())
    for _tag, sql, _params in pool.calls:
        assert hostile not in sql and "DROP TABLE" not in sql
    _, params = pool.call("lex:lexemes")
    assert params == [hostile]


def test_lex_or_query_is_assembled_inside_sql() -> None:
    pool = _Pool(_lex_answers())
    lexical.lexical_lane(pool, "BRCA1 mutations", _spec())
    lexemes_sql, _ = pool.call("lex:lexemes")
    assert "tsvector_to_array(to_tsvector('english', %s))" in lexemes_sql
    search_sql, _ = pool.call("lex:search")
    assert "string_agg(quote_literal(lex), ' | ')::tsquery" in search_sql
    # The indexed expression of migration 0120, so the GIN index applies.
    assert "to_tsvector('english', n.search_text) @@" in search_sql
    assert "ts_rank_cd(to_tsvector('english', n.search_text)," in search_sql
    assert search_sql.count("%s") == len(pool.call("lex:search")[1])


def test_lex_statistics_are_cached_for_ten_minutes() -> None:
    pool = _Pool(_lex_answers())
    now = [1000.0]
    lexical._STATS.clock = lambda: now[0]
    try:
        lexical.lexical_lane(pool, "BRCA1 mutations", _spec())
        first = pool.tags()
        assert {"lex:schema", "lex:rows", "lex:df"} <= set(first)
        pool.calls.clear()
        now[0] += 599.0
        lexical.lexical_lane(pool, "BRCA1 mutations", _spec())
        assert pool.tags() == ["lex:lexemes", "lex:search"]
        pool.calls.clear()
        now[0] += 2.0
        lexical.lexical_lane(pool, "BRCA1 mutations", _spec())
        assert {"lex:schema", "lex:rows", "lex:df"} <= set(pool.tags())
    finally:
        lexical._STATS.clock = __import__("time").monotonic


def test_lex_df_is_counted_per_scope_and_only_for_uncached_lexemes() -> None:
    pool = _Pool(_lex_answers())
    lexical.lexical_lane(pool, "BRCA1 mutations", _spec())
    pool.answers["lex:lexemes"] = [("brca1",), ("tio2",)]
    pool.answers["lex:df"] = lambda _sql, params: [(lex, 1) for lex in params[-1]]
    pool.calls.clear()
    lexical.lexical_lane(pool, "BRCA1 TiO2", _spec())
    _, params = pool.call("lex:df")
    assert params[-1] == ["tio2"]
    pool.calls.clear()
    lexical.lexical_lane(pool, "BRCA1 TiO2", _spec(graph_ids=("g-1",)))
    _, params = pool.call("lex:df")
    assert params[-1] == ["brca1", "tio2"]  # another scope, its own statistics


def test_lex_lane_off_touches_no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _Pool(_lex_answers())
    assert lexical.lexical_lane(pool, "q", _spec(lexical="off")).status == "off"
    monkeypatch.setenv(lexical.LEXICAL_ENV, "off")
    assert lexical.lexical_lane(pool, "q", _spec()).status == "off"
    assert pool.calls == []


@pytest.mark.parametrize(
    ("schema", "rows", "needle"),
    [
        ((False, False, False), 0, "search_text is missing"),
        ((True, False, False), 0, lexical.FTS_INDEX),
        ((True, True, False), 0, "backfill"),
    ],
)
def test_lex_lane_unavailable_reasons(schema: tuple, rows: int, needle: str) -> None:
    pool = _Pool(_lex_answers(**{"lex:schema": [schema], "lex:rows": [(rows,)]}))
    result = lexical.lexical_lane(pool, "BRCA1", _spec())
    assert result.status == "unavailable" and needle in result.reason
    assert "lex:search" not in pool.tags()


def test_lex_lane_reports_a_failing_statement_instead_of_raising() -> None:
    pool = _Pool(_lex_answers(**{"lex:search": RuntimeError("boom")}))
    result = lexical.lexical_lane(pool, "BRCA1", _spec())
    assert result.status == "unavailable" and "RuntimeError" in result.reason
    assert "boom" not in result.reason  # the message stays in the server log
    assert result.ms >= 0.0


def test_lex_lane_without_matching_terms_skips_the_search() -> None:
    pool = _Pool(_lex_answers(**{"lex:lexemes": []}))
    result = lexical.lexical_lane(pool, "the of and", _spec())
    assert result.status == "ok" and result.candidates == []
    assert "no indexable terms" in result.reason
    assert "lex:search" not in pool.tags()


# ── ident lane ───────────────────────────────────────────────────────────────


def _ident_answers(schema: tuple = (True, True, True), rows: int = 10) -> dict[str, Any]:
    return {
        "lex:schema": [schema],
        "lex:rows": [(rows,)],
        "ident:search": [("n-9", _content("art-z", 1), 2), ("n-3", _content("art-c", 0), 1)],
    }


def test_ident_lane_ranks_by_tokens_matched() -> None:
    pool = _Pool(_ident_answers())
    result = lexical.ident_lane(pool, "ZSM_5 and IL-6 in 50% of cases", _spec())
    assert result.status == "ok"
    assert result.info == {"tokens": ["ZSM_5", "IL-6"], "access": "trigram"}
    assert [c.node_id for c in result.candidates] == ["n-9", "n-3"]
    assert result.candidates[0].lane_scores == {"ident": 2.0}
    assert result.candidates[0].lane_ranks == {"ident": 1}


def test_ident_lane_binds_escaped_patterns() -> None:
    pool = _Pool(_ident_answers())
    lexical.ident_lane(pool, "ZSM_5 and IL-6 in 50% of cases", _spec())
    sql, params = pool.call("ident:search")
    patterns = [r"%ZSM\_5%", "%IL-6%"]  # "_" would be a wildcard unescaped
    assert params[:2] == patterns
    assert params[-3:] == [*patterns, 20]
    assert sql.count("n.search_text ILIKE %s") == 4
    assert "ZSM" not in sql and "ORDER BY matched DESC" in sql


def test_ident_lane_scans_only_small_scopes_without_trigrams() -> None:
    small = _Pool(_ident_answers(schema=(True, True, False), rows=lexical.IDENT_SCAN_MAX_ROWS))
    result = lexical.ident_lane(small, "BRCA1", _spec())
    assert result.status == "ok" and result.info["access"] == "scan"
    lexical.clear_caches()
    large = _Pool(_ident_answers(schema=(True, True, False), rows=lexical.IDENT_SCAN_MAX_ROWS + 1))
    result = lexical.ident_lane(large, "BRCA1", _spec())
    assert result.status == "unavailable" and "pg_trgm" in result.reason
    assert "ident:search" not in large.tags()


def test_ident_lane_without_tokens_or_switched_off_touches_no_database() -> None:
    pool = _Pool(_ident_answers())
    result = lexical.ident_lane(pool, "no identifiers here", _spec())
    assert result.status == "ok" and result.candidates == [] and result.info == {"tokens": []}
    assert lexical.ident_lane(pool, "BRCA1", _spec(lexical="off")).status == "off"
    assert pool.calls == []


def test_ident_lane_unavailable_without_the_column() -> None:
    pool = _Pool(_ident_answers(schema=(False, False, False)))
    result = lexical.ident_lane(pool, "BRCA1", _spec())
    assert result.status == "unavailable" and "migration 0120" in result.reason


# ── review fixes (2026-10-10) ────────────────────────────────────────────────


def test_ident_ties_break_by_occurrences_then_length_never_by_id_first() -> None:
    """core-12: with one identifier every match ties on ``matched``; id order
    handed an arbitrary 20 of them an RRF boost."""
    pool = _Pool(_ident_answers())
    lexical.ident_lane(pool, "IL-6 in sepsis", _spec())
    sql, params = pool.call("ident:search")
    assert "ORDER BY matched DESC, hits DESC, octet_length(n.search_text), n.node_id" in sql
    assert params[:3] == ["%IL-6%", "IL-6", "IL-6"]  # the raw token counts occurrences
    assert "IL-6" not in sql and sql.count("%s") == len(params)


def test_ident_hits_are_kept_beside_the_matched_count() -> None:
    answers = _ident_answers()
    answers["ident:search"] = [
        ("n-9", _content("art-z", 1), 1, 3),
        ("n-3", _content("art-c", 0), 1, 1),
    ]
    result = lexical.ident_lane(_Pool(answers), "IL-6", _spec())
    assert result.candidates[0].lane_scores == {"ident": 1.0, "ident:hits": 3.0}


def test_lex_shortlist_breaks_idf_ties_by_length_not_by_id() -> None:
    """core-12: a one-lexeme query ties every match on its IDF sum."""
    pool = _Pool(_lex_answers())
    lexical.lexical_lane(pool, "BRCA1 mutations", _spec())
    sql, _ = pool.call("lex:search")
    assert "ORDER BY idf_sum DESC, len, node_id LIMIT %s" in sql
    assert "octet_length(n.search_text) AS len" in sql
    assert "ORDER BY score DESC, p.len, n.node_id LIMIT %s" in sql


_LEAK = 'connection to server at "postgres" (10.89.1.4), port 5432 failed: user "umrs"'


@pytest.mark.parametrize("lane", ["lex", "ident"])
def test_a_failing_lane_reports_its_type_never_its_message(lane: str) -> None:
    """security-5: hosts, users and SQL stay in the server log."""

    class _Broken(_Pool):
        def getconn(self) -> _Conn:
            raise RuntimeError(_LEAK)

    run = lexical.lexical_lane if lane == "lex" else lexical.ident_lane
    result = run(_Broken({}), "BRCA1 mutations", _spec())
    assert result.status == "unavailable" and "RuntimeError" in result.reason
    assert "10.89.1.4" not in result.reason and "umrs" not in result.reason
