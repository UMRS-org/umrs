"""The graph bridge lane for large-extraction sites (ADR-093 D-14, C20).

The rows below are the shapes the live stand-alone graph returned on
2026-10-10 (project ``default``; the T2 notes were ingested with extraction
on, ``scifact-eval`` was not): semantic nodes
``node-sem-{scope}{slug}-{artifact}`` with ``content.semantic_type`` and
``content.artifact_id``, ``mentions`` edges from the file node, and the
query lexemes Postgres' ``english`` configuration derives (``nife-ldh`` keeps
its hyphenated lexeme, ``university`` stems to ``univers``).

Pinned: no fire on a one-document query; a fire on a two-entity,
two-document query; kept passages never displaced; the KG-05 scope on every
statement; ``auto`` off for a 3B extraction model and on for kimi-k3 — once
the lane is measured (review of ADR-093, 2026-10-10: ``auto`` stays off until
the multi-hop gate passes, and a value that is not off|auto|on is off).
"""

from __future__ import annotations

import json
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import rag as rag_api
from umrs.api import rag_retrieve, rag_spine
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge.retrieval import calibrate, dense, graph_lane, pipeline
from umrs.knowledge.retrieval import hydrate as hydrate_mod
from umrs.knowledge.retrieval.calibrate import NullStats
from umrs.knowledge.retrieval.types import Candidate, LaneResult, RetrieveSpec

from r4_contracts.retrieval import RetrievalSufficiency, RetrievedPassage, RetrieveResult

NOTES = "g-16565937d32d441b"  # the T2 notes' named graph
LDH = "art-7e49373d907a"  # NiFe-LDH / OER note (Tohoku University)
ZEOLITES = ("art-5db849714bf7", "art-6c3e72c70be4", "art-94c84357e74b")  # CO2 uptake notes
QUERY = (
    "How does NiFe-LDH from Tohoku University change the oxygen evolution reaction and CO2 uptake?"
)
QUERY_LEX = [
    "chang",
    "co2",
    "evolut",
    "ldh",
    "nife",
    "nife-ldh",
    "oxygen",
    "reaction",
    "tohoku",
    "univers",
    "uptak",
]


def _sem_id(slug: str, artifact: str) -> str:
    scope = f"{NOTES}-" if artifact == LDH else ""
    return f"node-sem-{scope}{slug}-{artifact}"


def _row(slug, label, artifact, semantic_type, label_lex, aliases=(), query_lex=QUERY_LEX):
    content = {"source": label, "artifact_id": artifact, "semantic_type": semantic_type}
    return (
        _sem_id(slug, artifact),
        label,
        json.dumps(content),
        list(label_lex),
        list(aliases),
        [artifact],
        list(query_lex),
    )


# The live link rows for QUERY (2026-10-10), strongest label match first.
LIVE_ROWS = [
    _row("nife-ldh", "NiFe-LDH", LDH, "material", ["ldh", "nife", "nife-ldh"]),
    _row(
        "oxygen-evolution-reaction",
        "oxygen evolution reaction",
        LDH,
        "process",
        ["evolut", "oxygen", "reaction"],
    ),
    *(_row("co2-uptake", "CO2 uptake", a, "property", ["co2", "uptak"]) for a in ZEOLITES),
    _row("tohoku-university", "Tohoku University", LDH, "organization", ["tohoku", "univers"]),
    _row(
        "nagoya-university",
        "Nagoya University",
        "art-4b6acb04bb8a",
        "application",
        ["nagoya", "univers"],
    ),
    *(
        _row(
            "hokkaido-university", "Hokkaido University", a, "organization", ["hokkaido", "univers"]
        )
        for a in ZEOLITES
    ),
]


def _hop_row(artifact: str, distance: float, index: int = 0) -> tuple:
    content = {"kind": "chunk", "artifact_id": artifact, "chunk_index": index}
    return (f"node-chunk-{artifact}-{index}", json.dumps(content), distance)


class FakePool:
    """psycopg2's pool shape; answers each statement by its marker and
    records every call (sql, params)."""

    def __init__(self, *, exists=True, link=LIVE_ROWS, hop=(), fail=None, scope=None):
        self.answers = {
            "graph:scope": [scope or (False, 59 if exists else 0)],
            "graph:link": list(link),
            "graph:hop": list(hop),
        }
        self.fail = fail
        self.calls: list[tuple[str, list]] = []

    def getconn(self):
        return _Conn(self)

    def putconn(self, _conn):
        pass

    def answer(self, sql, params):
        self.calls.append((sql, list(params or ())))
        if self.fail and self.fail in sql:
            raise RuntimeError("statement timeout on host db.internal for user umrs")
        return next((rows for marker, rows in self.answers.items() if marker in sql), [])

    def statements(self, marker):
        return [(sql, params) for sql, params in self.calls if marker in sql]


class _Conn:
    def __init__(self, pool):
        self.pool, self.rows = pool, []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        self.rows = self.pool.answer(sql, params)

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    for name in ("UMRS_RAG_GRAPH_LANE", "UMRS_RAG_PROFILE", "UMRS_RAG_MERGE", "UMRS_AGENT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("UMRS_RAG_GRAPH_LANE_MEASURED", raising=False)
    monkeypatch.setattr(hydrate_mod, "_SEARCH_TEXT", None)
    calibrate.clear_cache()


def _spec(**overrides) -> RetrieveSpec:
    base = {"queries": (QUERY,), "project_id": "default", "lexical": "off", "rerank": "off"}
    return RetrieveSpec(**{**base, **overrides})


def _kept(*artifacts: str) -> list[Candidate]:
    return [Candidate(node_id=f"node-chunk-{a}-0", artifact_id=a) for a in artifacts]


# ── the gate ────────────────────────────────────────────────────────────────


@pytest.fixture
def measured(monkeypatch):
    """The lane passed its multi-hop gate (the maintainer's switch)."""
    monkeypatch.setenv("UMRS_RAG_GRAPH_LANE_MEASURED", "1")


@pytest.mark.parametrize(
    ("model", "lane"),
    [("llama3.2:3b", "off"), ("kimi-k3", "on"), ("glm-5.3", "on"), (None, "off")]
    # Lead decision 2026-10-10: one size rule, the EFFECTIVE size: a 30B-A3B
    # extracts like a ~9.5B model (medium), so its graph earns no lane.
    + [("qwen3-30b-a3b", "off"), ("gpt-oss-20b", "off"), ("gpt-oss-120b", "on")]
    # Review of ADR-093 (profiles-graph-2): ollama's tags for the same MoE
    # models carry no -a<n>b tag; they were classed large and turned it on.
    + [("qwen3:30b", "off"), ("qwen3-coder:30b", "off"), ("deepseek-v2:16b", "off")]
    + [("qwen3:32b", "on")],
)
def test_auto_is_on_only_for_a_large_extraction_model(measured, model, lane):
    setting, how = graph_lane.resolve_gate(lambda: model)
    assert setting == lane and how.startswith("auto: extraction model")


@pytest.mark.parametrize("model", ["kimi-k3", "glm-5.3", "gpt-oss-120b"])
def test_auto_stays_off_until_the_lane_passes_its_multi_hop_gate(monkeypatch, model):
    """Review of ADR-093 (profiles-graph-5): D-14 says no default flips
    without a multi-hop probe, and every other ADR-093 stage is off until its
    gate passes; unset, the lane was on at every large-extraction site."""
    setting, how = graph_lane.resolve_gate(lambda: model)
    assert setting == "off" and "UMRS_RAG_GRAPH_LANE_MEASURED unset" in how
    for value in ("0", "no", "maybe"):
        monkeypatch.setenv("UMRS_RAG_GRAPH_LANE_MEASURED", value)
        assert graph_lane.resolve_gate(lambda m=model: m)[0] == "off"
    monkeypatch.setenv("UMRS_RAG_GRAPH_LANE_MEASURED", " TRUE ")
    assert graph_lane.resolve_gate(lambda m=model: m)[0] == "on"


@pytest.mark.parametrize("mode", ["off", "on"])
def test_an_explicit_setting_never_reads_the_extraction_model(monkeypatch, mode):
    monkeypatch.setenv("UMRS_RAG_GRAPH_LANE", mode.upper())

    def never():
        raise AssertionError("the registry was read")

    assert graph_lane.resolve_gate(never) == (mode, f"UMRS_RAG_GRAPH_LANE={mode}")


@pytest.mark.parametrize("value", ["false", "0", "no", "disabled", "sometimes", "of"])
def test_a_value_that_is_not_off_auto_or_on_is_off(monkeypatch, measured, value):
    """Review of ADR-093 (profiles-graph-5): an operator's 'false' resolved to
    auto, which is on at a kimi-k3 site. A bad value now fails closed."""
    monkeypatch.setenv("UMRS_RAG_GRAPH_LANE", value)

    def never():
        raise AssertionError("the registry was read")

    assert graph_lane.graph_lane_mode() == "off"
    assert graph_lane.resolve_gate(never) == (
        "off",
        f"UMRS_RAG_GRAPH_LANE={value!r} is not off|auto|on: off",
    )


def test_a_bad_value_is_logged_once(monkeypatch, caplog):
    monkeypatch.setattr(graph_lane, "_warned_modes", set())
    monkeypatch.setenv("UMRS_RAG_GRAPH_LANE", "disable-it-please")
    for _ in range(3):
        graph_lane.graph_lane_mode()
    assert sum("disable-it-please" in r.getMessage() for r in caplog.records) == 1


@pytest.mark.parametrize("value", [None, "", "  ", "AUTO"])
def test_unset_or_empty_is_auto(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv("UMRS_RAG_GRAPH_LANE", value)
    assert graph_lane.graph_lane_mode() == "auto"


# ── link and route ──────────────────────────────────────────────────────────


def test_link_groups_a_label_over_documents_and_drops_half_matches():
    entities = {e.label: e for e in graph_lane.link(LIVE_ROWS)}
    assert set(entities) == {
        "NiFe-LDH",
        "oxygen evolution reaction",
        "CO2 uptake",
        "Tohoku University",
    }
    assert entities["CO2 uptake"].artifacts == set(ZEOLITES)
    assert all(e.score == 1.0 for e in entities.values())  # "Nagoya University" is 0.5


def test_link_ranks_by_score_then_matched_terms_and_keeps_eight():
    half = [_row(f"h{i}", f"H {i}", f"art-h{i}", "concept", [f"h{i}", "x"]) for i in range(3)]
    assert graph_lane.link(half) == []  # one of two terms is below the 0.6 floor
    query = ["alpha", "beta", "gamma", *(f"e{i}" for i in range(12))]
    rows = [
        _row(f"e{i}", f"E{i}", f"art-{i}", "concept", [f"e{i}"], query_lex=query) for i in range(12)
    ]
    rows.append(
        _row(
            "abg",
            "alpha beta gamma",
            "art-abg",
            "concept",
            ["alpha", "beta", "gamma"],
            query_lex=query,
        )
    )
    entities = graph_lane.link(rows)
    assert len(entities) == graph_lane.MAX_ENTITIES
    assert (entities[0].label, entities[0].matched) == ("alpha beta gamma", 3)


def test_an_alias_links_an_entity_its_label_does_not():
    row = _row("oer", "OER", LDH, "process", ["oer"], aliases=["oxygen evolut reaction"])
    [entity] = graph_lane.link([row])
    assert (entity.label, entity.score) == ("OER", 1.0)


def test_a_row_naming_no_document_is_never_linked():
    row = (_sem_id("x", LDH), "x", json.dumps({"semantic_type": "concept"}), ["x"], [], [], ["x"])
    assert graph_lane.link([row]) == []


def test_no_fire_on_a_single_document_query():
    rows = LIVE_ROWS[:2]  # NiFe-LDH and the oxygen evolution reaction: one note
    documents, why = graph_lane.route(graph_lane.link(rows), set())
    assert documents == [] and why == "the linked entities sit in one document"


def test_no_fire_on_one_entity_however_many_documents_mention_it():
    rows = LIVE_ROWS[2:5]  # CO2 uptake, three notes
    documents, why = graph_lane.route(graph_lane.link(rows), set())
    assert documents == [] and why == "fewer than two linked entities"


def test_fires_on_two_entities_in_two_documents_toward_the_unkept_ones():
    documents, why = graph_lane.route(graph_lane.link(LIVE_ROWS), {LDH})
    assert why == "" and documents == sorted(ZEOLITES)


def test_no_fire_when_every_linked_document_is_already_kept():
    documents, why = graph_lane.route(graph_lane.link(LIVE_ROWS), {LDH, *ZEOLITES})
    assert documents == [] and why == "every linked document is already kept"


# Review of ADR-093 (profiles-graph-1): one named entity under two labels,
# or two entities that always sit in the same documents, is no bridge. Live
# 2026-10-10: "Tell me about ZSM-5 zeolite" linked "ZSM-5 zeolite" (5 notes)
# and "ZSM-5 zeolite adsorption" (3 of 4 terms, one of those notes).
ZSM_LEX = ["-5", "tell", "zeolit", "zsm"]
Z0, Z1, Z2 = ZEOLITES


def _zsm_rows() -> list[tuple]:
    zsm = ["-5", "zeolit", "zsm"]
    rows = [
        _row("zsm-5-zeolite", "ZSM-5 zeolite", a, "material", zsm, query_lex=ZSM_LEX)
        for a in ZEOLITES
    ]
    variant = ["-5", "adsorpt", "zeolit", "zsm"]
    rows.append(
        _row(
            "zsm-5-zeolite-adsorption",
            "ZSM-5 zeolite adsorption",
            Z0,
            "process",
            variant,
            query_lex=ZSM_LEX,
        )
    )
    return rows


def test_one_named_entity_with_a_longer_label_variant_does_not_fire():
    entities = graph_lane.link(_zsm_rows())
    assert [e.label for e in entities] == ["ZSM-5 zeolite", "ZSM-5 zeolite adsorption"]
    documents, why = graph_lane.route(entities, {Z0})
    assert documents == [] and why == "fewer than two linked entities"


def test_label_variants_join_one_concept_with_every_document():
    [concept] = graph_lane.concepts(graph_lane.link(_zsm_rows()))
    assert (concept.label, concept.score, concept.artifacts) == (
        "ZSM-5 zeolite",
        1.0,
        set(ZEOLITES),
    )


def test_co_located_entities_do_not_fire():
    """CO2 uptake and sodium sit in the same three notes: no document holds
    one without the other, so there is nothing to bridge to."""
    query = ["co2", "sodium", "uptak"]
    rows = [
        _row("co2-uptake", "CO2 uptake", a, "property", ["co2", "uptak"], query_lex=query)
        for a in ZEOLITES
    ]
    rows += [_row("sodium", "sodium", a, "material", ["sodium"], query_lex=query) for a in ZEOLITES]
    documents, why = graph_lane.route(graph_lane.link(rows), {Z0})
    assert documents == [] and why == "no two linked entities sit in different documents"


def test_a_nested_pair_is_no_bridge_but_an_apart_pair_bridges_its_difference():
    query = ["alpha", "beta", "gamma"]
    rows = [_row("alpha", "alpha", "art-a1", "concept", ["alpha"], query_lex=query)]
    rows += [
        _row("beta", "beta", a, "concept", ["beta"], query_lex=query) for a in ("art-a1", "art-b2")
    ]
    nested = graph_lane.link(rows)
    assert graph_lane.route(nested, set())[0] == []  # alpha's notes all hold beta
    rows += [
        _row("gamma", "gamma", a, "concept", ["gamma"], query_lex=query)
        for a in ("art-b2", "art-g3")
    ]
    documents, why = graph_lane.route(graph_lane.link(rows), {"art-b2"})
    # beta {a1, b2} and gamma {b2, g3} are apart; alpha {a1} and gamma too.
    assert why == "" and documents == ["art-a1", "art-g3"]


# ── the lane over its statements ────────────────────────────────────────────


def _lane(pool, spec=None, kept=(LDH,), vector=(1.0, 0.0)):
    return graph_lane.bridge_lane(
        pool,
        spec or _spec(),
        vector=list(vector) if vector else None,
        model="/models/e5-base-v2",
        kept=_kept(*kept),
    )


def _fired():
    hop = [_hop_row(ZEOLITES[0], 0.20), _hop_row(ZEOLITES[1], 0.10), _hop_row(ZEOLITES[2], 0.30)]
    pool = FakePool(hop=hop)
    return pool, _lane(pool)


def test_the_lane_adds_at_most_two_bridges_closest_first():
    pool, lane = _fired()
    assert lane.status == "ok" and lane.info["fired"] is True
    assert [c.artifact_id for c in lane.candidates] == [ZEOLITES[1], ZEOLITES[0]]
    assert [c.lane_ranks for c in lane.candidates] == [{"graph": 1}, {"graph": 2}]
    [(_sql, params)] = pool.statements("graph:hop")
    assert params[-1] == sorted(ZEOLITES)  # never a kept passage's document


def test_the_lane_reports_each_bridges_entity_and_its_counts():
    _pool, lane = _fired()
    assert lane.info["bridges"][0] == {
        "source_id": f"node-chunk-{ZEOLITES[1]}-0",
        "bridge_entity": "CO2 uptake",
    }
    assert (lane.info["linked"], lane.info["documents"]) == (4, 4)


def test_no_hop_statement_when_the_route_does_not_fire():
    pool = FakePool(link=LIVE_ROWS[:2])
    lane = _lane(pool)
    assert lane.status == "ok" and lane.candidates == [] and lane.info["fired"] is False
    assert lane.reason == "the linked entities sit in one document"
    assert pool.statements("graph:hop") == []


def test_no_semantic_graph_in_scope_is_unavailable():
    pool = FakePool(exists=False)
    lane = _lane(pool)
    assert (lane.status, lane.reason) == ("unavailable", "no semantic graph in scope")
    assert pool.statements("graph:link") == []


def test_a_scope_too_large_to_scan_without_the_label_index_is_refused_at_once():
    """Review of ADR-093 (profiles-graph-4): the link statement computed every
    label's lexemes, so past ~70k semantic nodes every retrieve waited the
    whole statement budget for a timeout. Without the label index the lane
    now refuses a scope over LABEL_SCAN_MAX_ROWS before the link runs."""
    over = graph_lane.LABEL_SCAN_MAX_ROWS + 1
    pool = FakePool(scope=(False, over))
    lane = _lane(pool)
    assert lane.status == "unavailable" and lane.reason.startswith(
        f"no label index ({graph_lane.LABEL_INDEX}) and more than"
    )
    assert pool.statements("graph:link") == [] and pool.statements("graph:hop") == []


def test_with_the_label_index_a_large_scope_still_links():
    pool = FakePool(
        scope=(True, graph_lane.LABEL_SCAN_MAX_ROWS + 1), hop=[_hop_row(ZEOLITES[0], 0.2)]
    )
    assert _lane(pool).status == "ok" and len(pool.statements("graph:link")) == 1


def test_the_label_match_is_written_for_the_label_index():
    """The index's own expression, in the node scan's WHERE (an index scan
    once the index exists), not a materialized pass over every label."""
    label_match = "to_tsvector('english', coalesce(n.label, '')) @@ (SELECT q.tsq FROM q)"
    nodes = graph_lane._LINK_SQL.split("), d AS")[0]
    assert label_match in nodes and "s AS MATERIALIZED" not in nodes


def test_the_link_statement_groups_labels_before_its_limit():
    """Review of ADR-093 (profiles-graph-3): LIMIT 400 counted semantic
    nodes, one per document, so 'cells' in 400 notes pushed TP53 out before
    grouping. The statement now groups by label first (read-only check on the
    live Postgres with 401 'cells' nodes: TP53 and apoptosis kept)."""
    sql = graph_lane._LINK_SQL
    assert sql.index("GROUP BY d.label_lex") < sql.rindex("LIMIT %s")
    assert "array_agg(DISTINCT d.artifact)" in sql and "LIMIT %s" not in sql.split("GROUP BY")[0]


def test_a_grouped_row_carries_every_document_of_its_label():
    query = ["tp53", "regul", "apoptosi", "cell"]
    many = [f"art-{i:04d}" for i in range(401)]
    grouped = [
        ("node-sem-cells-art-0000", "cells", "{}", ["cell"], [], many, query),
        ("node-sem-tp53-art-x", "TP53", "{}", ["tp53"], [], ["art-x"], query),
        ("node-sem-apoptosis-art-y", "apoptosis", "{}", ["apoptosi"], [], ["art-y"], query),
    ]
    entities = {e.label: e for e in graph_lane.link(grouped)}
    assert (
        set(entities) == {"cells", "TP53", "apoptosis"} and len(entities["cells"].artifacts) == 401
    )


# The link statement on a real Postgres (the DB-backed CI job; locally with
# UMRS_TEST_DB_URL). It reads no table: the test's rows stand in for
# knowledge_nodes/knowledge_edges as CTEs, so the statement stays read-only.
_CELLS_NODES = (
    "SELECT 'node-sem-cells-art-' || lpad(g::text, 4, '0') AS node_id, 'cells'::text AS label,"
    " jsonb_build_object('artifact_id', 'art-' || lpad(g::text, 4, '0'),"
    " 'semantic_type', 'concept') AS content, 'default'::text AS project_id,"
    " 'project'::text AS visibility, 'entity'::text AS node_type, ''::text AS graph_id"
    " FROM generate_series(0, 400) AS g"
    " UNION ALL SELECT 'node-sem-tp53-art-x', 'TP53',"
    " jsonb_build_object('artifact_id', 'art-x', 'semantic_type', 'gene'),"
    " 'default', 'project', 'entity', ''"
    " UNION ALL SELECT 'node-sem-apoptosis-art-y', 'apoptosis',"
    " jsonb_build_object('artifact_id', 'art-y', 'semantic_type', 'process'),"
    " 'default', 'project', 'entity', ''"
)
_NO_EDGES = (
    "SELECT ''::text AS target_id, ''::text AS relation_type, '{}'::jsonb AS content WHERE false"
)


def _pg_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    host = os.environ.get("POSTGRES_HOST")
    if explicit or not host:
        return explicit or None
    user = os.environ.get("POSTGRES_USER", "test")
    secret = os.environ.get("POSTGRES_PASSWORD", "test")
    port, name = os.environ.get("POSTGRES_PORT", "5432"), os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{secret}@{host}:{port}/{name}"


class _Shadowed:
    """A cursor that runs each statement over the test's rows (CTEs named
    like the tables shadow them) instead of the tables."""

    HEAD = "WITH q AS MATERIALIZED ("

    def __init__(self, cur):
        self.cur = cur

    def execute(self, sql, params):
        shadow = (
            f"WITH knowledge_nodes AS ({_CELLS_NODES}), knowledge_edges AS ({_NO_EDGES}), "
            f"q AS MATERIALIZED ("
        )
        assert sql.count(self.HEAD) == 1
        self.cur.execute(sql.replace(self.HEAD, shadow), params)

    def fetchall(self):
        return self.cur.fetchall()


@pytest.mark.skipif(not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*")
def test_on_postgres_a_label_four_hundred_documents_share_leaves_the_rest_linked():
    """profiles-graph-3 on Postgres: 401 'cells' nodes and one TP53 node; the
    old statement's LIMIT 400 kept apoptosis and 399 'cells' rows only."""
    psycopg2 = pytest.importorskip("psycopg2")
    conn = psycopg2.connect(_pg_url())
    try:
        conn.set_session(readonly=True)
        with conn, conn.cursor() as cur:
            spec = _spec(queries=("Does TP53 regulate apoptosis in cells?",))
            entities = graph_lane._linked(_Shadowed(cur), spec)
    finally:
        conn.close()
    found = {e.label: e for e in entities}
    assert {"TP53", "apoptosis", "cells"} <= set(found)
    assert len(found["cells"].artifacts) == 401 and found["TP53"].artifacts == {"art-x"}


def test_no_query_vector_is_unavailable_and_reads_nothing():
    pool = FakePool()
    assert _lane(pool, vector=None).status == "unavailable" and pool.calls == []


def test_a_failed_statement_reports_its_type_only():
    lane = _lane(FakePool(fail="graph:link"))
    assert (lane.status, lane.reason) == ("unavailable", "graph lane failed (RuntimeError)")


def test_the_query_reaches_sql_only_as_a_parameter_and_each_statement_is_bounded():
    pool = FakePool(hop=[_hop_row(ZEOLITES[0], 0.2)])
    _lane(pool)
    assert all(QUERY not in sql for sql, _ in pool.calls)
    [(_sql, params)] = pool.statements("graph:link")
    assert params[0] == QUERY and params[-1] == graph_lane.LINK_GROUP_LIMIT
    assert pool.calls[0] == ("SELECT set_config('statement_timeout', %s, true)", ["1500"])


def test_every_statement_reads_the_dense_lanes_kg05_scope():
    """Selected graphs, exclusions and the ADR-050 gate: the predicate the
    dense lane ranks under, on the semantic nodes and on the hop."""
    spec = _spec(graph_ids=(NOTES,), exclude_graph_ids=("g-u2-personal",))
    pool = FakePool(hop=[_hop_row(ZEOLITES[0], 0.2)])
    _lane(pool, spec=spec)
    gate, gate_params = dense.gate_clause(spec)
    semantic = [*gate_params, "node-sem-%", ["g-u2-personal"]]
    expected = {
        "graph:scope": [graph_lane.LABEL_INDEX, *semantic, graph_lane.LABEL_SCAN_MAX_ROWS + 1],
        "graph:link": [QUERY, *semantic, *semantic, graph_lane.LINK_GROUP_LIMIT],
    }
    for marker, params_expected in expected.items():
        [(sql, params)] = pool.statements(marker)
        assert gate in sql and "n.graph_id <> ALL(%s)" in sql and "n.visibility != 'ambient'" in sql
        assert params == params_expected and gate_params[0] == [NOTES]
    [(sql, params)] = pool.statements("graph:hop")
    scope, scope_params = dense.scope_clause(spec, "/models/e5-base-v2")
    assert scope in sql and params[1:-1] == scope_params


# ── in the pipeline: appended, never displacing ─────────────────────────────


def _cand(node_id: str, cosine: float, artifact: str) -> Candidate:
    return Candidate(
        node_id=node_id,
        artifact_id=artifact,
        content={"kind": "chunk", "artifact_id": artifact},
        cosine=cosine,
        lane_scores={"dense": cosine},
        lane_ranks={"dense": 1},
    )


def _hydrate(_pool, candidates, _fetch, *, documents=None):
    for c in candidates:
        if not c.node_id.endswith("-notext"):
            c.text, c.text_source, c.source_name = (
                f"text of {c.node_id}",
                "artifact",
                f"{c.artifact_id}.md",
            )


def _stages(found, graph=None, mu=0.70):
    def lane(_pool, _vectors, _model, _spec, *, limit):
        for rank, c in enumerate(found, 1):
            c.lane_ranks["dense"] = rank
        return LaneResult(name="dense", candidates=list(found)[:limit], info={"in_scope": 100})

    null = NullStats(mu=mu, sigma=0.02, n=30, model="e5-base-v2")
    stages = {"dense": lane, "hydrate": _hydrate, "null": lambda *_a, **_k: null}
    if graph is not None:
        stages["graph"] = graph
    return stages


def _encode(texts):
    return "e5-base-v2", [[1.0, 0.0] for _ in texts], ""


def _found():
    return [_cand("node-chunk-ldh-0", 0.90, LDH), _cand("node-chunk-ldh-1", 0.89, LDH)]


def _bridge_lane(calls, *ids):
    def lane(pool, spec, *, vector, model, kept):
        calls.append({"kept": [c.node_id for c in kept], "vector": vector})
        found = [
            Candidate(
                node_id=i,
                artifact_id=i.split("|")[0],
                cosine=0.8,
                lane_scores={"graph": 1.0},
                lane_ranks={"graph": n},
            )
            for n, i in enumerate(ids, 1)
        ]
        info = {
            "fired": True,
            "bridges": [{"source_id": i, "bridge_entity": "CO2 uptake"} for i in ids],
        }
        return LaneResult(name="graph", candidates=found, info=info)

    return lane


def _plain_and_bridged():
    plain = pipeline.retrieve(object(), _spec(), encode=_encode, lanes=_stages(_found()))
    calls: list = []
    graph = _bridge_lane(calls, f"{ZEOLITES[0]}|b1", f"{ZEOLITES[1]}|b2")
    bridged = pipeline.retrieve(
        object(),
        _spec(graph_lane="on", graph_gate="test"),
        encode=_encode,
        lanes=_stages(_found(), graph),
    )
    return plain, bridged, calls


def test_bridges_are_appended_after_the_kept_passages_never_displacing_one():
    plain, bridged, calls = _plain_and_bridged()
    kept = [p.source_id for p in plain.passages]
    assert [p.source_id for p in bridged.passages][: len(kept)] == kept
    assert [p.lanes for p in bridged.passages[len(kept) :]] == [("graph",), ("graph",)]
    assert [p.rank for p in bridged.passages] == list(range(1, len(kept) + 3))
    assert calls == [{"kept": kept, "vector": [1.0, 0.0]}]


def test_the_bridged_result_reports_the_lane_and_its_score():
    plain, bridged, _calls = _plain_and_bridged()
    assert bridged.sufficiency == plain.sufficiency  # bridges never change the verdict
    report = bridged.lanes["graph"]
    assert (report["status"], report["n"], report["bridges"][0]["bridge_entity"]) == (
        "ok",
        2,
        "CO2 uptake",
    )
    assert bridged.passages[-1].scores["graph"] == 1.0


def test_a_bridge_without_text_is_dropped_from_passages_and_report():
    graph = _bridge_lane([], f"{ZEOLITES[0]}|b1-notext", f"{ZEOLITES[1]}|b2")
    result = pipeline.retrieve(
        object(), _spec(graph_lane="on"), encode=_encode, lanes=_stages(_found(), graph)
    )
    assert [p.source_id for p in result.passages if p.lanes == ("graph",)] == [f"{ZEOLITES[1]}|b2"]
    assert [b["source_id"] for b in result.lanes["graph"]["bridges"]] == [f"{ZEOLITES[1]}|b2"]


def test_no_bridge_from_an_insufficient_verdict():
    calls: list = []
    result = pipeline.retrieve(
        object(),
        _spec(graph_lane="on"),
        encode=_encode,
        lanes=_stages([_cand("n1", 0.705, LDH)], _bridge_lane(calls, "b|1")),
    )
    assert result.sufficiency.status == "insufficient" and calls == []
    assert result.lanes["graph"]["fired"] is False


def test_off_is_reported_with_how_the_api_resolved_it_and_never_runs():
    calls: list = []
    result = pipeline.retrieve(
        object(),
        _spec(graph_gate="auto: extraction model llama3.2:3b is small, not large"),
        encode=_encode,
        lanes=_stages(_found(), _bridge_lane(calls, "b|1")),
    )
    assert calls == [] and result.lanes["graph"] == {
        "status": "off",
        "n": 0,
        "reason": "auto: extraction model llama3.2:3b is small, not large",
    }
    unresolved = pipeline.retrieve(object(), _spec(), encode=_encode, lanes=_stages(_found()))
    assert "graph" not in unresolved.lanes


def test_legacy_threshold_mode_never_bridges():
    calls: list = []
    result = pipeline.retrieve(
        object(),
        _spec(graph_lane="on", threshold=0.7),
        encode=_encode,
        lanes=_stages(_found(), _bridge_lane(calls, "b|1")),
    )
    assert calls == [] and result.lanes["graph"]["reason"] == "legacy threshold mode"


def test_the_real_lane_runs_in_the_pipeline_over_the_live_shapes():
    pool = FakePool(hop=[_hop_row(ZEOLITES[0], 0.25)])
    result = pipeline.retrieve(
        pool, _spec(graph_lane="on"), encode=_encode, lanes=_stages(_found())
    )
    assert [p.source_id for p in result.passages][-1] == f"node-chunk-{ZEOLITES[0]}-0"
    assert result.passages[-1].lanes == ("graph",)
    assert result.lanes["graph"]["bridges"] == [
        {"source_id": f"node-chunk-{ZEOLITES[0]}-0", "bridge_entity": "CO2 uptake"}
    ]


# ── the api resolves auto from the extraction model ─────────────────────────


@pytest.fixture
def route(monkeypatch):
    specs: list[RetrieveSpec] = []

    def retrieve(_pool, spec, **_kwargs):
        specs.append(spec)
        return RetrieveResult(query="q")

    monkeypatch.setattr(rag_api, "_require_rag_scope", lambda *_a, **_k: ("u1", "default"))
    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(rag_retrieve, "hidden_graphs", lambda *_a: ())
    monkeypatch.setattr(rag_retrieve, "batch_encoder", lambda: "ENCODER")
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: "FETCH")
    app = FastAPI()
    app.include_router(rag_api.router)
    app.state.umrs_db_pool = object()
    client = TestClient(app)

    def get(extraction_model):
        monkeypatch.setattr(
            ka,
            "_extract_chat_endpoint",
            lambda _p: ("http://llm", extraction_model) if extraction_model else None,
        )
        assert client.get("/umrs/rag/retrieve", params={"q": QUERY}).status_code == 200
        return specs[-1]

    return get


@pytest.mark.parametrize(
    ("model", "lane"), [("llama3.2:3b", "off"), ("kimi-k3", "on"), (None, "off")]
)
def test_the_route_resolves_auto_from_the_sites_extraction_model(route, measured, model, lane):
    spec = route(model)
    assert spec.graph_lane == lane and spec.graph_gate.startswith("auto: extraction model")


def test_the_route_honours_an_explicit_site_setting(route, monkeypatch):
    monkeypatch.setenv("UMRS_RAG_GRAPH_LANE", "off")
    assert (route("kimi-k3").graph_lane, route("kimi-k3").graph_gate) == (
        "off",
        "UMRS_RAG_GRAPH_LANE=off",
    )


def test_the_compose_path_gates_by_its_answer_model(monkeypatch, measured):
    seen = {}

    def retrieve(_pool, spec, **_kwargs):
        seen["spec"] = spec
        return RetrieveResult(query="q", sufficiency=RetrievalSufficiency(status="insufficient"))

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: None)
    monkeypatch.setattr(ka, "_extract_chat_endpoint", lambda _p: ("http://rikyu", "glm-5.3"))
    rag_spine.answer_from_spine(
        object(), "q", "default", encode=lambda _t: ("m", None, "x"), fetch_bytes=lambda _r: None
    )
    assert seen["spec"].graph_lane == "on"


# ── a partial verdict keeps its bridges ─────────────────────────────────────


def _passage(rank: int, lanes=("dense",)) -> RetrievedPassage:
    return RetrievedPassage(
        rank=rank,
        source_id=f"n{rank}",
        artifact_id=f"a{rank}",
        source_name="n.md",
        text=f"t{rank}",
        lanes=lanes,
    )


def test_a_partial_view_caps_the_cut_and_keeps_the_bridges():
    passages = [_passage(1), _passage(2), _passage(3), _passage(4, ("graph",))]
    assert [p.source_id for p in rag_spine.partial_view(passages, "partial", 2)] == [
        "n1",
        "n2",
        "n4",
    ]
    assert len(rag_spine.partial_view(passages, "sufficient", 2)) == 4
