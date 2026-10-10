"""The retrieval core: one scored retrieve path (ADR-093 D-2, C1).

Pins the stage order and its reports, the dense lane's scope, hydration,
the batch encoder, and two acceptance receipts:

- receipt 3 — the retrieve path never calls a chat model;
- receipt 4 — legacy mode (threshold 0.7, stages off, k 5) returns exactly
  the ids, in exactly the order, that ``rag_spine.retrieve_passages``
  returned before ADR-093 (its statement is frozen below).
"""

from __future__ import annotations

import ast
import json
import math
import random
import sys
from pathlib import Path

import httpx
import pytest
from umrs.knowledge.retrieval import calibrate, dense, pipeline
from umrs.knowledge.retrieval import hydrate as hydrate_mod
from umrs.knowledge.retrieval.calibrate import NullStats
from umrs.knowledge.retrieval.types import Candidate, LaneResult, RetrieveSpec

_RETRIEVAL_DIR = Path(pipeline.__file__).parent


# ── fakes ───────────────────────────────────────────────────────────────────


class FakePool:
    """psycopg2's pool shape; answers each statement by the first marker it
    contains (rows, or a callable of (sql, params)). Records every call."""

    def __init__(self, answers=()):
        self.answers = list(answers)
        self.calls: list[tuple[str, tuple]] = []

    def getconn(self):
        return _Conn(self)

    def putconn(self, _conn):
        pass

    def answer(self, sql, params):
        self.calls.append((sql, tuple(params or ())))
        for marker, rows in self.answers:
            if marker in sql:
                return list(rows(sql, params) if callable(rows) else rows)
        return []


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
    monkeypatch.setattr(hydrate_mod, "_SEARCH_TEXT", None)
    for name in ("UMRS_RAG_MERGE", "UMRS_RAG_RERANK_DEFAULT", "UMRS_RAG_LEXICAL", "UMRS_RAG_DELTA"):
        monkeypatch.delenv(name, raising=False)
    calibrate.clear_cache()


def _content(artifact, index, start=0, end=100, **extra):
    return {
        "kind": "chunk",
        "artifact_id": artifact,
        "chunk_index": index,
        "chunk_start": start,
        "chunk_end": end,
        **extra,
    }


def _cand(node_id, cosine, artifact="a", index=0, start=0, end=100, rank=1):
    return Candidate(
        node_id=node_id,
        artifact_id=artifact,
        chunk_index=index,
        content=_content(artifact, index, start, end),
        cosine=cosine,
        lane_scores={"dense": cosine},
        lane_ranks={"dense": rank},
    )


def _dense_lane(candidates, in_scope=100):
    def lane(pool, vectors, model, spec, *, limit):
        ranked = sorted(candidates, key=lambda c: -(c.cosine or 0))[:limit]
        for rank, c in enumerate(ranked, 1):
            c.lane_ranks["dense"] = rank
        return LaneResult(
            name="dense", candidates=ranked, info={"in_scope": in_scope, "space": "e5-base-v2"}
        )

    return lane


def _fake_hydrate(pool, candidates, fetch_bytes, *, documents=None):
    for c in candidates:
        if c.text_source:
            continue
        span = (c.content.get("chunk_start", 0), c.content.get("chunk_end", 0))
        c.text = c.content.get("_text", "x" * (span[1] - span[0]))
        c.text_source = "artifact"
        c.span = span
        c.source_name = f"{c.artifact_id}.md"


def _encode(texts):
    return "e5-base-v2", [[1.0, 0.0] for _ in texts], ""


def _spec(**overrides):
    base = {
        "queries": ("perovskite encapsulation",),
        "project_id": "p1",
        "lexical": "off",
        "rerank": "off",
    }
    return RetrieveSpec(**{**base, **overrides})


def _lanes(candidates, null=NullStats(mu=0.70, sigma=0.02, n=30, model="e5-base-v2"), **more):
    return {
        "dense": _dense_lane(candidates),
        "hydrate": _fake_hydrate,
        "null": lambda *_a, **_k: null,
        **more,
    }


# ── stage order, cut and reports ────────────────────────────────────────────


def test_dense_mode_keeps_the_close_ones_and_judges_sufficiency():
    found = [_cand("n1", 0.90, "a"), _cand("n2", 0.89, "b"), _cand("n3", 0.80, "c")]
    result = pipeline.retrieve(object(), _spec(), encode=_encode, lanes=_lanes(found))
    assert [p.source_id for p in result.passages] == ["n1", "n2"]
    assert (result.sufficiency.status, result.sufficiency.basis) == ("sufficient", "z_null")
    assert (result.searched, result.space) == (100, "e5-base-v2")
    first = result.passages[0]
    assert (first.scores["dense"], first.lanes) == (0.9, ("dense",))


def test_every_stage_reports_its_status_and_timing():
    result = pipeline.retrieve(object(), _spec(), encode=_encode, lanes=_lanes([_cand("n1", 0.9)]))
    statuses = {name: report["status"] for name, report in result.lanes.items()}
    assert statuses == {
        "dense": "ok",
        "lex": "off",
        "ident": "off",
        "fusion": "off",
        "rerank": "off",
        "null": "ok",
        "merge": "off",
    }
    assert {"encode", "dense", "lexical", "fuse", "cut", "hydrate", "total"} <= set(
        result.timing_ms
    )


def test_a_nonsense_like_top_score_is_insufficient_but_still_reported():
    found = [_cand("n1", 0.705, "a"), _cand("n2", 0.70, "b")]
    result = pipeline.retrieve(
        object(), _spec(queries=("violet harbor ledger",)), encode=_encode, lanes=_lanes(found)
    )
    assert result.sufficiency.status == "insufficient" and result.sufficiency.basis == "z_null"
    assert len(result.passages) >= 1  # the consumer reads the verdict, never re-cuts


def test_no_vectors_in_scope_is_insufficient():
    result = pipeline.retrieve(
        object(),
        _spec(),
        encode=_encode,
        lanes={**_lanes([]), "dense": _dense_lane([], in_scope=0)},
    )
    assert result.passages == () and result.sufficiency.basis == "no_vectors"
    assert result.sufficiency.status == "insufficient"


def test_an_encoder_outage_is_reported_never_answered():
    down = lambda _texts: ("", None, "query encoder unavailable (ConnectError)")  # noqa: E731
    result = pipeline.retrieve(object(), _spec(), encode=down, lanes=_lanes([_cand("n1", 0.9)]))
    assert result.passages == ()
    assert result.lanes["dense"] == {
        "status": "unavailable",
        "n": 0,
        "reason": "query encoder unavailable (ConnectError)",
    }
    assert (result.sufficiency.status, result.sufficiency.basis) == ("insufficient", "unavailable")


def test_a_missing_lane_module_is_unavailable_with_the_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "umrs.knowledge.retrieval.lexical", None)
    lanes = pipeline.default_lanes()
    assert (
        isinstance(lanes["lex"], pipeline.Unavailable)
        and "ModuleNotFoundError" in lanes["lex"].reason
    )
    found = [_cand("n1", 0.9)]
    stages = {
        **_lanes(found),
        "lex": lanes["lex"],
        "ident": lanes["ident"],
        "lex_strength": lanes["lex_strength"],
    }
    result = pipeline.retrieve(object(), _spec(lexical="on"), encode=_encode, lanes=stages)
    assert (
        result.lanes["lex"]["status"] == "unavailable"
        and "ModuleNotFoundError" in result.lanes["lex"]["reason"]
    )
    assert [p.source_id for p in result.passages] == ["n1"]


def test_lexical_only_finds_get_their_cosines_in_one_statement():
    found = [_cand("n1", 0.90, "a"), _cand("n2", 0.86, "b")]
    lex_only = Candidate(
        node_id="n9",
        artifact_id="z",
        content=_content("z", 0),
        lane_scores={"lex": 9.0},
        lane_ranks={"lex": 1},
    )
    twin = Candidate(node_id="n2", lane_scores={"lex": 4.0}, lane_ranks={"lex": 2})
    pool = FakePool([("LEAST(", [("n9", 0.12)])])
    lex = lambda *_a, **_k: LaneResult(name="lex", candidates=[lex_only, twin])  # noqa: E731
    stages = {
        **_lanes(found),
        "lex": lex,
        "ident": lambda *_a, **_k: LaneResult(name="ident"),
        "lex_strength": lambda *_a, **_k: 0.0,
    }
    result = pipeline.retrieve(pool, _spec(lexical="on"), encode=_encode, lanes=stages)
    statements = [params for sql, params in pool.calls if "LEAST(" in sql]
    assert [params[1] for params in statements] == [["n9"]]  # ONE statement, lexical-only ids
    assert (twin.cosine, twin.artifact_id) == (0.86, "b")  # adopted from the dense lane
    found_n9 = next(p for p in result.passages if p.source_id == "n9")
    fused = result.lanes["fusion"]["status"]
    assert (round(found_n9.scores["dense"], 6), found_n9.lanes, fused) == (0.88, ("lex",), "ok")


def test_at_most_per_artifact_chunks_of_one_artifact_reach_the_cut():
    found = [
        _cand(f"node-chunk-a-{i}", 0.90 - i * 0.001, "a", i, i * 300, i * 300 + 290)
        for i in range(4)
    ]
    found.append(_cand("node-chunk-b-0", 0.89, "b"))
    result = pipeline.retrieve(
        object(), _spec(per_artifact=2, k_max=4), encode=_encode, lanes=_lanes(found)
    )
    assert [p.source_id for p in result.passages] == [
        "node-chunk-a-0",
        "node-chunk-a-1",
        "node-chunk-b-0",
    ]
    assert pipeline.cap_per_artifact(found, 1, top=2) == [found[0], found[4]]


def test_a_short_tail_folds_into_its_predecessor_and_takes_one_slot():
    """Fixed windows leave 15-140 character tails that e5 ranks high alone;
    with ``merge`` on (D-5, the S2 knob) the tail folds into its host."""
    tail = _cand("node-chunk-g-a-1", 0.91, "a", 1, 1000, 1060)
    head = _cand("node-chunk-g-a-0", 0.89, "a", 0, 0, 1000)
    other = _cand("node-chunk-g-b-0", 0.90, "b", 0, 0, 1000)
    tail.content["_text"], head.content["_text"] = "t" * 60, "h" * 1000
    stages = {**_lanes([tail, head, other]), "merge": lambda _pool, kept, **_k: kept}
    result = pipeline.retrieve(object(), _spec(k_max=4, merge="on"), encode=_encode, lanes=stages)
    ids = [p.source_id for p in result.passages]
    assert ids == ["node-chunk-g-a-0", "node-chunk-g-b-0"]
    first = result.passages[0]
    assert first.text == "h" * 1000 + "t" * 60 and first.span == (0, 1060)
    assert first.scores["dense"] == 0.91 and first.section == "part 1-2"


def test_rerank_asked_but_not_staged_is_unavailable_and_keeps_the_order():
    found = [_cand("n1", 0.90, "a"), _cand("n2", 0.89, "b")]
    result = pipeline.retrieve(
        object(), _spec(rerank="on"), encode=_encode, rerank=None, lanes=_lanes(found)
    )
    assert result.lanes["rerank"]["status"] == "unavailable"
    assert [p.source_id for p in result.passages] == ["n1", "n2"]
    assert result.sufficiency.basis == "z_null"  # not judged as reranked


def test_a_staged_reranker_orders_and_cuts_by_probability():
    found = [_cand("n1", 0.90, "a"), _cand("n2", 0.89, "b"), _cand("n3", 0.88, "c")]
    reranker = lambda _q, passages: ([-6.0, 3.0, 2.5][: len(passages)], {"model": "m"})  # noqa: E731
    result = pipeline.retrieve(
        object(), _spec(rerank="on"), encode=_encode, rerank=reranker, lanes=_lanes(found)
    )
    assert [p.source_id for p in result.passages] == ["n2", "n3"]
    assert result.sufficiency.basis == "rerank_p" and result.sufficiency.status == "sufficient"
    assert result.passages[0].scores["rerank_p"] == pytest.approx(0.9526, abs=1e-3)


def test_merge_on_runs_the_merge_stage_with_the_shared_artifact_cache():
    seen = {}

    def merge(pool, kept, **kwargs):
        seen.update(kwargs)
        return kept

    found = [_cand("n1", 0.9)]
    result = pipeline.retrieve(
        object(), _spec(merge="on"), encode=_encode, lanes={**_lanes(found), "merge": merge}
    )
    assert result.lanes["merge"]["status"] == "ok"
    assert seen["max_passage_tokens"] == 400 and isinstance(seen["documents"], dict)


def test_trace_lists_the_candidates_and_the_mode():
    found = [_cand("n1", 0.9), _cand("n2", 0.5, "b")]
    result = pipeline.retrieve(object(), _spec(trace=True), encode=_encode, lanes=_lanes(found))
    assert result.trace["mode"] == "dense"
    assert [row["node_id"] for row in result.trace["candidates"]] == ["n1", "n2"]
    plain = pipeline.retrieve(object(), _spec(), encode=_encode, lanes=_lanes(found))
    assert plain.trace == {}


def test_too_few_vectors_for_a_null_leaves_only_the_lexical_rule():
    found = [_cand("n1", 0.9)]
    found[0].content["_text"] = "Perovskite encapsulation with alumina."
    lanes = {**_lanes(found), "dense": _dense_lane(found, in_scope=5)}
    hit = pipeline.retrieve(object(), _spec(), encode=_encode, lanes=lanes)
    assert (hit.sufficiency.status, hit.sufficiency.basis) == ("sufficient", "lexical")
    miss = pipeline.retrieve(
        object(), _spec(queries=("zeolite sorption",)), encode=_encode, lanes=lanes
    )
    # No null means no z, so no "partial" (C2; review core-4).
    assert (miss.sufficiency.status, miss.sufficiency.basis) == ("insufficient", "lexical")


def test_a_partial_verdict_shows_at_most_two_passages():
    """Weak evidence comes with few distractors, for every reader alike."""
    found = [_cand(f"n{i}", 0.745 - i * 0.001, f"a{i}") for i in range(4)]
    result = pipeline.retrieve(object(), _spec(), encode=_encode, lanes=_lanes(found))
    assert result.sufficiency.status == "partial"
    assert [p.source_id for p in result.passages] == ["n0", "n1"]


def test_the_null_report_says_why_it_was_not_consulted():
    lanes = {**_lanes([_cand("n1", 0.9)]), "dense": _dense_lane([_cand("n1", 0.9)], in_scope=3)}
    result = pipeline.retrieve(object(), _spec(), encode=_encode, lanes=lanes)
    assert result.lanes["null"] == {
        "status": "off",
        "n": 0,
        "reason": "fewer than 30 vectors in scope",
    }


# ── receipt 3: no chat call on the retrieve path ────────────────────────────


class _Response:
    def __init__(self, body, status=200):
        self._body, self.status_code = body, status

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "x", request=httpx.Request("POST", "http://x"), response=None
            )


def _chat_trap(monkeypatch):
    """Every outbound call is recorded; only /rag/embed is answered."""
    calls: list[str] = []

    def post(url, *args, **kwargs):
        calls.append(str(url))
        if str(url).endswith("/rag/embed"):
            texts = kwargs["json"]["texts"]
            return _Response(
                {"model": "/models/e5-base-v2", "vectors": [[1.0, 0.0] for _ in texts], "dim": 2}
            )
        return _Response({}, status=599)

    def client_post(self, url, *args, **kwargs):
        calls.append(str(url))
        return _Response({}, status=599)

    def chat(*_a, **_k):
        calls.append("chat:_default_litellm_chat")
        raise AssertionError("the retrieve path called a chat model")

    from umrs.knowledge import agent_executor

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx.Client, "post", client_post)
    monkeypatch.setattr(agent_executor, "_default_litellm_chat", chat)
    return calls


def _corpus_pool(rows):
    """Dense rows (node_id, content, distance, in_scope) and hydrate rows."""
    hydrate_rows = [(n, "g", c, None, f"{n}.md", None, None) for n, c, _d, _s in rows]
    probes = [(i, 0.3) for i in range(len(calibrate.NULL_PROBES_V1))]
    return FakePool([("unnest(", probes), ("<=>", rows), ("LEFT JOIN", hydrate_rows)])


def test_the_retrieve_path_never_calls_a_chat_model(monkeypatch):
    calls = _chat_trap(monkeypatch)
    text = "Alumina encapsulation slows perovskite moisture ingress."
    content = json.dumps({**_content("a", 0, 0, len(text)), "text": text})
    pool = _corpus_pool([("node-chunk-a-0", content, 0.1, 40)])
    spec = _spec(lexical="on", rerank="on", merge="on", trace=True)
    result = pipeline.retrieve(
        pool,
        spec,
        encode=dense.http_batch_encoder("http://rag:8003"),
        rerank=None,
        fetch_bytes=lambda _r: None,
    )
    assert [p.source_id for p in result.passages] == ["node-chunk-a-0"]
    assert calls and all(url.endswith("/rag/embed") for url in calls), calls


def _imports(source: str) -> set[str]:
    """Every imported module and name of a module's source."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            found.add(node.module or "")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.update(alias.name for alias in node.names)
    return found


def _chat_reach(path: Path) -> list[str]:
    source = path.read_text()
    imports = _imports(source)
    hits = [
        name for name in imports if name.endswith("agent_executor") or name.startswith("umrs.api")
    ]
    hits += ["_default_litellm_chat"] if "_default_litellm_chat" in imports else []
    return hits + (["chat/completions"] if "chat/completions" in source else [])


def test_no_retrieval_module_imports_a_chat_client():
    """Static half of receipt 3: the core never imports the chat helper."""
    reach = {path.name: _chat_reach(path) for path in sorted(_RETRIEVAL_DIR.glob("*.py"))}
    assert {name: hits for name, hits in reach.items() if hits} == {}


# ── receipt 4: legacy mode is B0 ────────────────────────────────────────────

# Frozen from api/knowledge_query.py at parent 2dba21bb / umrs 809387a1 (the
# statement rag_spine.retrieve_passages ran through _similarity_hits, B0).
_B0_SQL = (
    "SELECT e.node_id, e.model_name, e.dimension, e.created_at, "
    "n.label, n.project_id, (e.embedding <=> %s::vector) AS distance, "
    "n.content::text "
    "FROM knowledge_node_embeddings e "
    "JOIN knowledge_nodes n ON n.node_id = e.node_id "
    "WHERE ((n.project_id = %s OR n.visibility = 'system') AND n.visibility != 'ambient') "
    "AND (e.model_name = %s OR e.model_name LIKE %s) "
    "ORDER BY distance ASC LIMIT %s"
)


def _cos_distance(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return 1.0 - dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


class _VectorDB:
    """Postgres' exact scan in miniature: ``<=>``, a stable sort, LIMIT."""

    def __init__(self, seed=20261010, n=60):
        rng = random.Random(seed)
        self.rows = []
        for i in range(n):
            vec = [rng.uniform(-1, 1) for _ in range(6)]
            text = "" if i % 7 == 3 else f"passage {i}"
            content = _content(f"art-{i % 20}", i // 20, 0, 50, **({"text": text} if text else {}))
            self.rows.append((f"node-{i}", vec, json.dumps(content)))
        self.rows.insert(10, ("node-tie", list(self.rows[9][1]), self.rows[9][2]))  # an exact tie

    def ranked(self, sql, params):
        query = json.loads(params[0])
        scored = sorted(
            ((_cos_distance(query, vec), nid, content) for nid, vec, content in self.rows),
            key=lambda r: r[0],
        )
        top = scored[: params[-1]]
        if sql.startswith("SELECT e.node_id, e.model_name"):
            return [(nid, "/models/e5-base-v2", 6, None, "chunk", "p1", d, c) for d, nid, c in top]
        return [(nid, c, d) for d, nid, c in top]

    def pool(self):
        texts = {nid: content for nid, _v, content in self.rows}
        hydrate = lambda _sql, params: [(n, "g", texts[n], None, n, None, None) for n in params[0]]  # noqa: E731
        return FakePool([("<=>", self.ranked), ("LEFT JOIN", hydrate)])


def _b0_retrieve(pool, vector, *, top_k=5, threshold=0.7):
    """Pre-ADR-093 retrieve_passages + attach_texts, by their semantics."""
    params = (
        dense.vector_literal(vector),
        "p1",
        "/models/e5-base-v2",
        dense.same_space_pattern("/models/e5-base-v2"),
        top_k,
    )
    rows = pool.answer(_B0_SQL, params)
    kept = [r for r in rows if max(0.0, 1.0 - float(r[6])) >= threshold]
    return [r[0] for r in kept if json.loads(r[7]).get("text")]


def test_legacy_mode_issues_the_b0_statement():
    db = _VectorDB()
    pool = db.pool()
    vector = [0.3, -0.2, 0.5, 0.1, 0.0, 0.4]
    encode = lambda texts: ("/models/e5-base-v2", [vector for _ in texts], "")  # noqa: E731
    pipeline.retrieve(pool, _spec(threshold=0.7, k_max=5), encode=encode, lanes={"null": None})
    [(sql, params)] = [(s, p) for s, p in pool.calls if "<=>" in s]
    assert sql[sql.index("FROM ") :] == _B0_SQL[_B0_SQL.index("FROM ") :]
    assert params == (
        dense.vector_literal(vector),
        "p1",
        "/models/e5-base-v2",
        dense.same_space_pattern("/models/e5-base-v2"),
        5,
    )


def test_legacy_mode_returns_b0_ids_in_b0_order():
    db = _VectorDB()
    rng = random.Random(7)
    checked = 0
    for _ in range(40):
        vector = [rng.uniform(-1, 1) for _ in range(6)]
        encode = lambda texts, v=vector: ("/models/e5-base-v2", [v for _ in texts], "")  # noqa: E731
        threshold = rng.choice([0.0, 0.1, 0.3, 0.5])
        expected = _b0_retrieve(db.pool(), vector, threshold=threshold)
        result = pipeline.retrieve(db.pool(), _spec(threshold=threshold, k_max=5), encode=encode)
        assert [p.source_id for p in result.passages] == expected
        assert result.sufficiency.basis == "threshold"
        assert result.lanes["lex"]["status"] == "off" and result.lanes["rerank"]["status"] == "off"
        checked += bool(expected)
    assert checked >= 20  # most probes keep something


# ── the dense lane ──────────────────────────────────────────────────────────


def test_the_dense_lane_scans_one_space_chunks_only_with_the_graph_rule():
    pool = FakePool([("<=>", [("node-chunk-a-0", json.dumps(_content("a", 0)), 0.2, 12)])])
    from umrs.knowledge.graph_registry import default_graph_id

    spec = _spec(graph_ids=("g-x", default_graph_id("p1")), exclude_graph_ids=("g-y",))
    lane = dense.dense_lane(pool, [[1.0, 0.0]], "/models/e5-base-v2", spec, limit=50)
    [(sql, params)] = pool.calls
    assert "n.content->>'kind' = 'chunk'" in sql and "count(*) OVER ()" in sql
    assert "(n.graph_id = ANY(%s) OR (n.graph_id = '' AND ('g-' || left(" in sql
    assert "n.graph_id <> ALL(%s)" in sql and "ORDER BY distance ASC, e.node_id LIMIT %s" in sql
    selected = ["g-x", default_graph_id("p1")]
    assert params[1:3] == (selected, selected)
    assert params[-3:] == ("%/e5-base-v2", ["g-y"], 50)
    [c] = lane.candidates
    assert (c.cosine, c.artifact_id, lane.info["in_scope"]) == (pytest.approx(0.8), "a", 12)


def test_a_shared_graph_never_reads_legacy_rows_of_another_project():
    pool = FakePool()
    dense.dense_lane(pool, [[1.0]], "m", _spec(graph_ids=("g-shared",)), limit=5)
    sql, params = pool.calls[0]
    # '' rows only of a project whose DEFAULT graph id is selected: g-shared is
    # no project's default graph, so none.
    assert f"(n.graph_id = '' AND {dense.DEFAULT_GRAPH_SQL} = ANY(%s))" in sql
    assert params[1:3] == (["g-shared"], ["g-shared"])


def test_query_variants_rank_separately_for_fusion():
    rows = {0: [("x", "{}", 0.1, 3), ("y", "{}", 0.2, 3)], 1: [("y", "{}", 0.05, 3)]}
    state = {"i": 0}

    def answer(_sql, _params):
        out = rows[state["i"]]
        state["i"] += 1
        return out

    lane = dense.dense_lane(
        FakePool([("<=>", answer)]), [[1.0], [0.5]], "m", _spec(queries=("a", "b")), limit=5
    )
    by_id = {c.node_id: c for c in lane.candidates}
    assert by_id["y"].lane_ranks == {"dense": 1, "dense#0": 2, "dense#1": 1}
    assert by_id["y"].cosine == pytest.approx(0.95)


def test_a_failing_scan_is_an_unavailable_lane():
    class Broken(FakePool):
        def answer(self, _sql, _params):
            raise RuntimeError("relation does not exist")

    lane = dense.dense_lane(Broken(), [[1.0]], "m", _spec(), limit=5)
    assert lane.status == "unavailable" and "RuntimeError" in lane.reason


# ── the batch encoder ───────────────────────────────────────────────────────


def test_the_batch_encoder_sends_every_variant_in_one_request(monkeypatch):
    sent = []

    def post(url, json=None, timeout=None):
        sent.append(json)
        return _Response({"model": "e5", "vectors": [[0.1], [0.2]], "dim": 1})

    monkeypatch.setattr(httpx, "post", post)
    model, vectors, reason = dense.http_batch_encoder("http://rag/")(["a", "b"])
    assert (model, vectors, reason) == ("e5", [[0.1], [0.2]], "") and sent == [
        {"texts": ["a", "b"]}
    ]


def test_an_older_rag_service_is_asked_one_text_at_a_time(monkeypatch):
    sent = []

    def post(url, json=None, timeout=None):
        sent.append(json)
        if "texts" in json:
            return _Response({"detail": "text required"}, status=422)
        return _Response({"model": "e5", "vector": [float(len(json["text"]))]})

    monkeypatch.setattr(httpx, "post", post)
    model, vectors, _ = dense.http_batch_encoder("http://rag")(["a", "bb"])
    assert (model, vectors) == ("e5", [[1.0], [2.0]])
    assert sent == [{"texts": ["a", "bb"]}, {"text": "a"}, {"text": "bb"}]


def test_no_encoder_url_is_a_reason():
    assert dense.http_batch_encoder("")(["q"]) == ("", None, "no query encoder configured")


# ── review fixes (2026-10-10): nonsense that shares rare words with a corpus ─

# Biomedical abstract sentences written for this test (no SciFact text). Off-
# domain words are the RAREST in such a corpus: each appears in one chunk of a
# 9,377-chunk scope (the live scifact-eval size), so its IDF is about 8.4.
_VOCABULARY = {
    "node-chunk-g-r-0": "Sodium content of restaurant meals exceeded dietary guidelines.",
    "node-chunk-g-v-0": "Focal dystonia in professional violin players responded to toxin.",
    "node-chunk-g-s-0": "Sleeping sickness is transmitted by tsetse flies in rural Uganda.",
    "node-chunk-g-u-0": "An umbrella review of vitamin D supplementation and fracture risk.",
    "node-chunk-g-b-0": "Helmet use among bicycle commuters reduced head injury admissions.",
    "node-chunk-g-e-0": "Egg allergy resolved by age four in most infants of a birth cohort.",
    "node-chunk-g-a-0": "A cohort from Australia linked ultraviolet exposure to melanoma.",
    "node-chunk-g-w-0": "Moderate red wine intake raised plasma resveratrol levels.",
    "node-chunk-g-l-0": "Lighthouse keepers showed circadian disruption on night shifts.",
    "node-chunk-g-k-0": "Kettle descaling agents caused accidental poisoning in children.",
}
_SCOPE_CHUNKS = 9377
#: NONSENSE_EVAL_V1-style probes (word salads, off-domain questions), written
#: here; each shares at least one rare word with the vocabulary above.
_RARE_WORD_NONSENSE = (
    "What is the best pizza restaurant in Naples?",
    "Which restaurant in Lisbon serves the best grilled sardines?",
    "the violin of tuesday borrows a lantern from the quiet harbor",
    "a sleeping teapot audits the umbrella of a nervous canal",
    "How do I patch a bicycle inner tube before a long ride?",
    "How long should I boil an egg for a picnic salad?",
    "What is the tallest mountain in Australia?",
    "red wine marmalade negotiates with the cobalt staircase",
    "the anxious lighthouse rents its kettle to seven polite moths",
)


def _vocabulary_hits(query):
    from umrs.knowledge.retrieval import cutoff

    terms = cutoff.content_terms(query)
    return {
        node: [t for t in terms if cutoff.term_coverage(t, text) == 1.0]
        for node, text in _VOCABULARY.items()
    }


def _vocabulary_lex(pool, query, spec, *, limit):
    """The lex lane over the vocabulary: IDF-weighted OR match, like C6."""
    from umrs.knowledge.retrieval import lexical

    hits = {node: terms for node, terms in _vocabulary_hits(query).items() if terms}
    df = {t: sum(t in terms for terms in hits.values()) for t in set().union(*hits.values())}
    scored = sorted(
        (-sum(lexical.idf(df[t], _SCOPE_CHUNKS) for t in terms), node, terms)
        for node, terms in hits.items()
    )
    return LaneResult(name="lex", candidates=[_vocabulary_lex_cand(r, df) for r in scored][:limit])


def _vocabulary_lex_cand(row, df):
    from umrs.knowledge.retrieval import lexical

    neg_sum, node, terms = row
    best = max(lexical.idf(df[t], _SCOPE_CHUNKS) for t in terms)
    scores = {"lex": -neg_sum, "lex:idf_sum": -neg_sum, "lex:idf_max": best}
    return Candidate(
        node_id=node, artifact_id=node, lane_scores=scores, content=_vocabulary_content(node)
    )


def _vocabulary_content(node):
    text = _VOCABULARY[node]
    return _content(node, 0, 0, len(text), _text=text)


def _vocabulary_run(query, cosines=None):
    """Dense cosines flat around the null mean (z within +-1) unless given."""
    found = []
    for i, node in enumerate(_VOCABULARY):
        cosine = (cosines or {}).get(node, 0.685 + 0.003 * i)
        found.append(
            Candidate(
                node_id=node, artifact_id=node, content=_vocabulary_content(node), cosine=cosine
            )
        )
    stages = {
        **_lanes(found),
        "dense": _dense_lane(found, in_scope=_SCOPE_CHUNKS),
        "lex": _vocabulary_lex,
        "ident": lambda *_a, **_k: LaneResult(name="ident"),
    }
    spec = _spec(queries=(query,), lexical="on")
    return pipeline.retrieve(object(), spec, encode=_encode, lanes=stages)


@pytest.mark.parametrize("query", _RARE_WORD_NONSENSE)
def test_nonsense_sharing_a_rare_corpus_word_is_insufficient(query):
    """core-1: one matched rare lexeme never makes nonsense sufficient."""
    result = _vocabulary_run(query)
    assert result.lanes["lex"]["n"] >= 1 and result.lanes["fusion"]["status"] == "ok"
    assert any("lex" in p.lanes for p in result.passages)  # the rare match was kept
    assert result.sufficiency.status == "insufficient", result.sufficiency


def test_evidence_in_the_same_vocabulary_stays_sufficient():
    clear = _vocabulary_run(
        "botulinum toxin for focal dystonia in violin players",
        cosines={"node-chunk-g-v-0": 0.90},
    )
    assert (clear.sufficiency.status, clear.sufficiency.basis) == ("sufficient", "z_null")
    lifted = _vocabulary_run("sodium in restaurant meals", cosines={"node-chunk-g-r-0": 0.75})
    assert (lifted.sufficiency.status, lifted.sufficiency.basis) == ("sufficient", "lexical")
    assert lifted.passages[0].source_id == "node-chunk-g-r-0"


def test_a_partial_z_without_lexical_evidence_stays_partial():
    result = _vocabulary_run("glacier hydrology", cosines={"node-chunk-g-k-0": 0.75})
    assert result.sufficiency.status == "partial"


# ── review fixes: small-to-big stays behind its knob (core-7) ───────────────


def test_with_merge_off_a_tail_is_not_folded_into_its_host():
    """D-5: read-time small-to-big is off until S2 — tail folding included."""
    host = _cand("node-chunk-g-a-0", 0.89, "a", 0, 0, 1000)
    tail = _cand("node-chunk-g-a-1", 0.90, "a", 1, 1000, 1100)
    host.content["_text"], tail.content["_text"] = "h" * 1000, "t" * 100
    result = pipeline.retrieve(
        object(), _spec(k_max=4, merge="off"), encode=_encode, lanes=_lanes([tail, host])
    )
    by_id = {p.source_id: p for p in result.passages}
    assert set(by_id) == {"node-chunk-g-a-0", "node-chunk-g-a-1"}
    assert by_id["node-chunk-g-a-0"].span == (0, 1000)
    assert by_id["node-chunk-g-a-0"].scores["dense"] == 0.89
    assert by_id["node-chunk-g-a-0"].text == "h" * 1000


# ── review fixes: every query variant reaches the lexical lanes (core-13) ───


def _variant_dense(candidates, variants=2, in_scope=100):
    def lane(pool, vectors, model, spec, *, limit):
        for rank, c in enumerate(candidates, 1):
            c.lane_ranks.update({"dense": rank, **{f"dense#{i}": rank for i in range(variants)}})
        return LaneResult(name="dense", candidates=list(candidates), info={"in_scope": in_scope})

    return lane


def test_every_query_variant_reaches_the_lexical_lanes():
    seen = []

    def lane(name):
        def run(_pool, query, _spec, *, limit):
            seen.append((name, query))
            found = [
                Candidate(
                    node_id="z1",
                    artifact_id="z",
                    content=_content("z", 0),
                    lane_scores={"lex": 7.0},
                )
            ]
            return LaneResult(name=name, candidates=found if "zeolite" in query else [])

        return run

    stages = {
        **_lanes([]),
        "dense": _variant_dense([_cand("n1", 0.9)]),
        "cosines": lambda *_a, **_k: {"z1": 0.7},
        "lex": lane("lex"),
        "ident": lane("ident"),
    }
    spec = _spec(queries=("it", "zeolite cracking it"), lexical="on")
    result = pipeline.retrieve(object(), spec, encode=_encode, lanes=stages)
    assert seen == [
        ("lex", "it"),
        ("lex", "zeolite cracking it"),
        ("ident", "it"),
        ("ident", "zeolite cracking it"),
    ]
    assert "lex#1" in result.lanes["fusion"]["rankings"]
    assert [v["n"] for v in result.lanes["lex"]["variants"]] == [0, 1]


def test_coverage_reads_the_best_covered_variant():
    found = [_cand("n1", 0.9)]
    found[0].content["_text"] = "Perovskite encapsulation with alumina."
    lanes = {**_lanes(found), "dense": _variant_dense(found, in_scope=5)}
    spec = _spec(queries=("it", "perovskite encapsulation it"))
    result = pipeline.retrieve(object(), spec, encode=_encode, lanes=lanes)
    assert (result.sufficiency.status, result.sufficiency.basis) == ("sufficient", "lexical")


# ── review fixes: no exception text reaches the caller (security-5) ─────────

_LEAK = (
    'connection to server at "postgres" (10.89.1.4), port 5432 failed: '
    'FATAL: password authentication failed for user "umrs"'
)


def _raise(*_a, **_k):
    raise RuntimeError(_LEAK)


def test_a_failing_lexical_lane_reports_its_type_never_its_message():
    stages = {**_lanes([_cand("n1", 0.9)]), "lex": _raise, "ident": _raise}
    result = pipeline.retrieve(object(), _spec(lexical="on"), encode=_encode, lanes=stages)
    lanes = result.to_dict()["lanes"]
    for name in ("lex", "ident"):
        assert lanes[name]["status"] == "unavailable" and "RuntimeError" in lanes[name]["reason"]
        assert "10.89.1.4" not in lanes[name]["reason"] and "umrs" not in lanes[name]["reason"]


def test_a_failing_null_reports_its_type_never_its_message():
    result = pipeline.retrieve(
        object(), _spec(), encode=_encode, lanes={**_lanes([_cand("n1", 0.9)]), "null": _raise}
    )
    reason = result.to_dict()["lanes"]["null"]["reason"]
    assert "RuntimeError" in reason and "10.89.1.4" not in reason and "postgres" not in reason


# ── review fixes: an encoder outage is never judged by lexical rows (core-4) ─


def test_an_encoder_outage_with_lexical_finds_is_still_unscored():
    down = lambda _texts: ("", None, "query encoder unavailable (ConnectError)")  # noqa: E731
    lex_only = Candidate(
        node_id="L0",
        artifact_id="z",
        content=_content("z", 0, _text="perovskite encapsulation"),
        lane_scores={"lex": 9.0, "lex:idf_max": 8.0},
    )
    stages = {
        **_lanes([]),
        "lex": lambda *_a, **_k: LaneResult(name="lex", candidates=[lex_only]),
        "ident": lambda *_a, **_k: LaneResult(name="ident"),
    }
    result = pipeline.retrieve(object(), _spec(lexical="on"), encode=down, lanes=stages)
    assert (result.sufficiency.status, result.sufficiency.basis) == ("insufficient", "unavailable")


# ── review fixes: the legacy '' rule is K3's (core-10, data-deploy-9) ───────

#: What dense.DEFAULT_GRAPH_SQL computes in Postgres, in Python.
_SQL_DEFAULT_GRAPH = (
    "('g-' || left(encode(sha256(convert_to('project:' || n.project_id, 'UTF8')), 'hex'), 16))"
)


def _sql_default_graph(project):
    import hashlib

    return "g-" + hashlib.sha256(("project:" + project).encode("utf-8")).hexdigest()[:16]


@pytest.mark.parametrize("project", ["default", "pB", "proj with spaces", "プロジェクト"])
def test_the_sql_default_graph_formula_is_the_registry_rule(project):
    from umrs.knowledge.graph_registry import default_graph_id

    assert _sql_default_graph(project) == default_graph_id(project)


def test_a_selected_default_graph_of_another_project_reads_its_legacy_rows():
    """K3's _GraphScope.of: the '' rows of EVERY project whose default graph
    is selected (the api layer passes only readable graphs), not only the
    caller's; every core lane renders this one clause."""
    from umrs.knowledge.graph_registry import default_graph_id

    selected = [default_graph_id("pB"), "g-x"]
    gate, params = dense.gate_clause(_spec(project_id="pA", graph_ids=tuple(selected)))
    assert f"(n.graph_id = '' AND {_SQL_DEFAULT_GRAPH} = ANY(%s))" in gate
    assert params == [selected, selected] and gate.count("%s") == 2


# ── review fix: the scope's last re-embed keys the null (core-11) ───────────


def test_the_dense_lane_reads_the_scopes_last_re_embed():
    row = ("node-chunk-a-0", json.dumps(_content("a", 0)), 0.2, 12, "2026-10-10 01:00:00+00")
    pool = FakePool([("<=>", [row])])
    lane = dense.dense_lane(pool, [[1.0, 0.0]], "/models/e5-base-v2", _spec(), limit=50)
    [(sql, _params)] = pool.calls
    assert "count(*) OVER () AS in_scope, max(e.updated_at) OVER () AS stamp" in sql
    assert (lane.info["in_scope"], lane.info["stamp"]) == (12, "2026-10-10 01:00:00+00")


def test_the_scope_stamp_keys_the_null_and_stays_off_the_report():
    seen = {}

    def null(*_a, **kwargs):
        seen.update(kwargs)
        return NullStats(mu=0.70, sigma=0.02, n=30, model="e5-base-v2")

    def stamped(pool, vectors, model, spec, *, limit):
        info = {"in_scope": 100, "space": "e5-base-v2", "stamp": "t1"}
        return LaneResult(name="dense", candidates=[_cand("n1", 0.9)], info=info)

    stages = {**_lanes([]), "dense": stamped, "null": null}
    result = pipeline.retrieve(object(), _spec(), encode=_encode, lanes=stages)
    assert seen == {"vectors": 100, "stamp": "t1"}
    assert "stamp" not in result.lanes["dense"]
