"""ADR-093 wave A end to end: the retrieval core with its REAL lanes.

``pipeline.retrieve`` runs every stage — encode, dense, lexical and ident,
the one cosine statement for lexical-only finds, RRF fusion (variants and
lanes), tail folding, the per-artifact cap, hydration, the cross-encoder
lane, the cut, the calibrated null and sufficiency, the neighbour merge —
over a corpus shaped like the stand-alone SciFact graph: fixed-window chunks
with md5s and spans, short tail chunks that rank high, a non-chunk node with
a vector, a second embedding space, and e5-like cosines that all sit in
0.7-1.0. Only the database, the query encoder and the reranker are fakes;
the fake database evaluates the core's own SQL over that corpus.

Checked here: legacy parity with the pre-ADR-093 K3 path, the cut sizes of
every mode, an insufficient verdict on nonsense, and no chat (or any HTTP)
call on the retrieve path; then the GET route's JSON through
``RetrieveResult.from_dict``, the fetch handle, and K3's lexical lane and
sufficiency agreeing with the core (integrator, 2026-10-10).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import knowledge_query as kq
from umrs.api import rag as rag_api
from umrs.api import rag_retrieve, rag_spine
from umrs.knowledge import agent_executor
from umrs.knowledge.retrieval import calibrate, lexical, pipeline
from umrs.knowledge.retrieval import hydrate as hydrate_mod
from umrs.knowledge.retrieval.types import RetrieveSpec

from r4_contracts.retrieval import RetrieveResult

DIM = 24
MODEL = "/models/e5-base-v2"
OTHER_MODEL = "/models/bge-small-en-v1.5"
PROJECT = "p"
COMMON = 1.6  # the shared direction that keeps every e5 cosine in 0.7-1.0
FILLER = (
    "The study measured results across samples and reported values under controlled conditions. "
)
TOPICS: tuple[tuple[str, ...], ...] = (
    ("hydrogel", "swelling", "moisture", "sorption"),
    ("tokamak", "plasma", "confinement", "divertor"),
    ("zeolite", "catalyst", "methanol", "cracking"),
    ("perovskite", "photovoltaic", "bandgap", "passivation"),
    ("microbiome", "colitis", "butyrate", "intestinal"),
    ("seismic", "inversion", "waveform", "tomography"),
    ("graphene", "phonon", "conductivity", "lattice"),
    ("interleukin", "sepsis", "cytokine", "macrophage"),
)
IDENTIFIER = {7: "IL-6"}
CROSS = (1, 1)  # this artifact's first chunk also mentions "hydrogel"
DECISIVE = "decisive"  # the reranker's favourite (third artifact of a topic)
STRENGTH = (1.0, 0.3, 1.2)  # chunk 0, chunk 1, the short tail
NONSENSE = (
    "best pizza in Naples tonight",
    "velvet giraffes audition for the municipal echo",
)
_STOP = frozenset({"how", "and", "the", "does", "what", "for", "are", "with", "under"})


def question(topic: int) -> str:
    words = TOPICS[topic]
    ident = f" {IDENTIFIER[topic]}" if topic in IDENTIFIER else ""
    return f"How do {words[0]} and {words[1]} change {words[2]}{ident}?"


def chunk_id(topic: int, j: int, index: int) -> str:
    return f"node-chunk-art-{topic}-{j}-{index}"


# ── the corpus ──────────────────────────────────────────────────────────────


def _rng(seed: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.md5(seed.encode()).hexdigest()[:8], 16))


def _vector(topic: int | None, strength: float, seed: str, noise: float) -> np.ndarray:
    vector = np.zeros(DIM)
    vector[0] = COMMON
    if topic is not None:
        vector[1 + topic] = strength
    vector[9:] += noise * _rng(seed).standard_normal(DIM - 9)
    return vector / np.linalg.norm(vector)


def _unrelated(text: str) -> np.ndarray:
    """A question about nothing in the corpus (null probes, nonsense)."""
    rng = _rng(text)
    vector = np.zeros(DIM)
    vector[0] = COMMON
    tail = rng.standard_normal(DIM - 9)
    vector[9:] = tail / np.linalg.norm(tail)
    vector[1:9] = 0.35 * rng.standard_normal(8)
    return vector / np.linalg.norm(vector)


def query_vector(text: str) -> np.ndarray:
    lowered = text.lower()
    for topic, words in enumerate(TOPICS):
        if words[0] in lowered:
            return _vector(topic, 1.0, text, 0.02)
    return _unrelated(text)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def lexemes(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in _STOP}


@dataclass
class Node:
    node_id: str
    kind: str
    artifact_id: str
    index: int
    text: str
    span: tuple[int, int] | None
    vectors: dict[str, np.ndarray] = field(default_factory=dict)

    def content(self) -> dict[str, Any]:
        if self.kind != "chunk":
            return {"kind": self.kind, "artifact_id": self.artifact_id, "excerpt": self.text}
        return {
            "kind": "chunk",
            "artifact_id": self.artifact_id,
            "chunk_index": self.index,
            "chunk_start": self.span[0],
            "chunk_end": self.span[1],
            "chunk_md5": _md5(self.text),
            "excerpt": self.text[:500],
            "origin": "knowledge_embed",
        }


def _chunk_texts(topic: int, j: int) -> list[str]:
    words = TOPICS[topic]
    tone = DECISIVE if j == 2 else "clear"
    first = f"{words[0].capitalize()} {words[1]} and {words[2]} {words[3]} findings were {tone}."
    if topic in IDENTIFIER:
        first += f" {IDENTIFIER[topic]} rose."
    if (topic, j) == CROSS:
        first += f" It cites {TOPICS[0][0]} work."
    texts = [first + " " + (FILLER * 3).strip(), f"A secondary note on {words[0]}. " + FILLER * 6]
    texts = [t.strip() for t in texts]
    if j == 0:
        texts.append(f"{words[0]} {words[1]} tail.")
    return texts


class Corpus:
    """Twenty-four SciFact-like artifacts in eight topics, 56 chunk vectors."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.documents: dict[str, str] = {}
        for topic in range(len(TOPICS)):
            for j in range(3):
                self._artifact(topic, j)
        file_node = Node("node-file-art-0-0", "file", "art-0-0", 0, "Hydrogel report.", None)
        file_node.vectors[MODEL] = _vector(0, 1.1, file_node.node_id, 0.01)
        self.nodes[file_node.node_id] = file_node
        # The same chunk in another space: never ranked beside e5 vectors.
        self.nodes[chunk_id(0, 0, 0)].vectors[OTHER_MODEL] = query_vector(question(0))

    def _artifact(self, topic: int, j: int) -> None:
        artifact = f"art-{topic}-{j}"
        texts = _chunk_texts(topic, j)
        self.documents[artifact] = " ".join(texts)
        start = 0
        for index, text in enumerate(texts):
            node = Node(chunk_id(topic, j, index), "chunk", artifact, index, text, None)
            node.span = (start, start + len(text))
            node.vectors[MODEL] = _vector(topic, STRENGTH[index], node.node_id, 0.01)
            self.nodes[node.node_id] = node
            start += len(text) + 1

    def chunks(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.kind == "chunk"]

    def embedded(self, space: str, *, chunks_only: bool) -> list[tuple[Node, np.ndarray]]:
        rows = []
        for node in self.nodes.values():
            for model, vector in node.vectors.items():
                if PurePosixPath(model).name == space and (node.kind == "chunk" or not chunks_only):
                    rows.append((node, vector))
        return rows

    def ranked(self, query: np.ndarray, space: str, *, chunks_only: bool) -> list:
        scored = [
            (1.0 - float(vector @ query), node)
            for node, vector in self.embedded(space, chunks_only=chunks_only)
        ]
        return sorted(scored, key=lambda pair: (pair[0], pair[1].node_id))

    def fetch(self, source_ref: str) -> bytes | None:
        document = self.documents.get(PurePosixPath(source_ref).stem)
        return document.encode("utf-8") if document is not None else None


CORPUS = Corpus()


# ── a database that evaluates the core's SQL over the corpus ─────────────────


def _literal(value: str) -> np.ndarray:
    vector = np.asarray(json.loads(value), dtype=float)
    return vector / np.linalg.norm(vector)


def _space(params: list[Any]) -> str:
    pattern = next(p for p in params if isinstance(p, str) and p.startswith("%/"))
    return pattern[2:].replace("\\_", "_").replace("\\%", "%")


def _needle(pattern: str) -> str:
    return pattern.strip("%").replace("\\_", "_").replace("\\%", "%").lower()


class _Cursor:
    def __init__(self, db: FakeDB) -> None:
        self._db, self._rows = db, []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = ()) -> None:
        self._rows = self._db.run(sql, list(params or ()))

    def fetchall(self) -> list[tuple]:
        return list(self._rows)

    def fetchone(self) -> tuple | None:
        return self._rows[0] if self._rows else None


class _Conn:
    def __init__(self, db: FakeDB) -> None:
        self._db = db

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def cursor(self) -> _Cursor:
        return _Cursor(self._db)

    def rollback(self) -> None:
        return None


class FakeDB:
    """A psycopg2-shaped pool whose cursor answers the retrieval core's SQL."""

    def __init__(self, corpus: Corpus, *, search_text: bool = True) -> None:
        self.corpus, self.search_text = corpus, search_text
        self.statements: list[str] = []
        self._handlers = (
            ("/* lex:schema */", self._lex_schema),
            ("/* lex:rows */", self._lex_rows),
            ("/* lex:lexemes */", self._lex_lexemes),
            ("/* lex:df */", self._lex_df),
            ("/* lex:search */", self._lex_search),
            ("/* ident:search */", self._ident_search),
            ("information_schema.columns", self._column_probe),
            ("count(*) OVER () AS in_scope", self._dense),
            ("LEAST(", self._cosines),
            ("WITH scope AS MATERIALIZED", self._probes),
            (
                "SELECT count(*), max(e.updated_at) FROM knowledge_node_embeddings",
                self._vector_count,
            ),
            ("n.label, n.project_id, (e.embedding", self._k3_similarity),
            ("SELECT e.node_id, n.content::text, (e.embedding", self._legacy),
            ("LEFT JOIN knowledge_artifacts a", self._hydrate_rows),
            ("GROUP BY 1, 2", self._chunk_counts),
            ("n.visibility, n.content->>'chunk_index'", self._node_row),
            ("SELECT DISTINCT n.graph_id", self._project_graphs),
            ("n.visibility, n.content->>'artifact_id' FROM", self._merge_scopes),
            ("IS NOT DISTINCT FROM", self._scoped_neighbours),
        )

    def getconn(self) -> _Conn:
        return _Conn(self)

    def putconn(self, _conn: Any) -> None:
        return None

    def run(self, sql: str, params: list[Any]) -> list[tuple]:
        self.statements.append(sql)
        for marker, handler in self._handlers:
            if marker in sql:
                return handler(sql, params)
        raise AssertionError(f"unexpected SQL: {sql[:160]}")

    def count(self, marker: str) -> int:
        return sum(1 for sql in self.statements if marker in sql)

    # dense, legacy, K3 similarity, cosines, null probes

    def _dense(self, _sql: str, params: list[Any]) -> list[tuple]:
        ranked = self.corpus.ranked(_literal(params[0]), _space(params), chunks_only=True)
        return [
            (n.node_id, json.dumps(n.content()), d, len(ranked))
            for d, n in ranked[: int(params[-1])]
        ]

    def _legacy(self, _sql: str, params: list[Any]) -> list[tuple]:
        ranked = self.corpus.ranked(_literal(params[0]), _space(params), chunks_only=False)
        return [(n.node_id, json.dumps(n.content()), d) for d, n in ranked[: int(params[-1])]]

    def _k3_similarity(self, _sql: str, params: list[Any]) -> list[tuple]:
        ranked = self.corpus.ranked(_literal(params[0]), _space(params), chunks_only=False)
        return [
            (n.node_id, MODEL, DIM, None, n.node_id, PROJECT, d, json.dumps(n.content()))
            for d, n in ranked[: int(params[-1])]
        ]

    def _cosines(self, sql: str, params: list[Any]) -> list[tuple]:
        count = sql.count("e.embedding <=> %s::vector")
        queries, wanted = [_literal(p) for p in params[:count]], set(params[count])
        rows = []
        for node, vector in self.corpus.embedded(_space(params), chunks_only=False):
            if node.node_id in wanted:
                rows.append((node.node_id, min(1.0 - float(vector @ q) for q in queries)))
        return rows

    def _probes(self, _sql: str, params: list[Any]) -> list[tuple]:
        scope = self.corpus.embedded(_space(params), chunks_only=True)
        rows = []
        for index, literal in zip(params[-2], params[-1], strict=True):
            probe = _literal(literal)
            rows.append((index, min(1.0 - float(v @ probe) for _n, v in scope)))
        return rows

    def _vector_count(self, _sql: str, params: list[Any]) -> list[tuple]:
        return [(len(self.corpus.embedded(_space(params), chunks_only=True)),)]

    # hydration and the fetch handle

    def _column_probe(self, _sql: str, _params: list[Any]) -> list[tuple]:
        return [(1,)] if self.search_text else []

    def _hydrate_rows(self, sql: str, params: list[Any]) -> list[tuple]:
        with_text = "n.search_text FROM" in sql
        rows = []
        for node_id in params[0]:
            node = self.corpus.nodes.get(node_id)
            if node is None:
                continue
            ref = f"s3://bucket/{node.artifact_id}.txt"
            row = (node_id, "", json.dumps(node.content()), ref, f"{node.artifact_id}.md", None, "")
            rows.append((*row, node.text) if with_text else row)
        return rows

    def _chunk_counts(self, _sql: str, params: list[Any]) -> list[tuple]:
        counts: dict[str, int] = {}
        for node in self.corpus.chunks():
            if node.artifact_id in params[1]:
                counts[node.artifact_id] = counts.get(node.artifact_id, 0) + 1
        return [("", artifact, n) for artifact, n in counts.items()]

    def _node_row(self, _sql: str, params: list[Any]) -> list[tuple]:
        node = self.corpus.nodes.get(params[0])
        if node is None or node.kind != "chunk":
            return []
        return [(PROJECT, "", "project", str(node.index), node.artifact_id)]

    def _project_graphs(self, _sql: str, _params: list[Any]) -> list[tuple]:
        return []  # every corpus row is a legacy '' row: no graph to hide

    def _merge_scopes(self, _sql: str, params: list[Any]) -> list[tuple]:
        found = (self.corpus.nodes.get(n) for n in params[0])
        return [
            (n.node_id, PROJECT, "", "project", n.artifact_id)
            for n in found
            if n is not None and n.kind == "chunk"
        ]

    def _scoped_neighbours(self, _sql: str, params: list[Any]) -> list[tuple]:
        ids, _owner, _graph, _visibility, artifact = params
        found = (self.corpus.nodes.get(n) for n in ids)
        return [(n.node_id,) for n in found if n is not None and n.artifact_id == artifact]

    # the lexical lanes (migration 0120)

    def _indexed(self) -> list[Node]:
        return self.corpus.chunks() if self.search_text else []

    def _lex_schema(self, _sql: str, _params: list[Any]) -> list[tuple]:
        return [(self.search_text, self.search_text, False)]

    def _lex_rows(self, _sql: str, _params: list[Any]) -> list[tuple]:
        return [(len(self._indexed()),)]

    def _lex_lexemes(self, _sql: str, params: list[Any]) -> list[tuple]:
        return [(lex,) for lex in sorted(lexemes(params[0]))]

    def _lex_df(self, _sql: str, params: list[Any]) -> list[tuple]:
        chunks = [lexemes(n.text) for n in self._indexed()]
        return [(lex, sum(1 for words in chunks if lex in words)) for lex in params[-1]]

    def _lex_search(self, _sql: str, params: list[Any]) -> list[tuple]:
        weights = dict(zip(params[0], params[1], strict=True))
        rows = []
        for node in self._indexed():
            matched = [weights[w] for w in lexemes(node.text) if w in weights]
            if matched:
                rows.append(
                    (node.node_id, node.content(), sum(matched), sum(matched), max(matched))
                )
        rows.sort(key=lambda row: (-row[2], row[0]))
        return rows[: int(params[-1])]

    def _ident_search(self, sql: str, params: list[Any]) -> list[tuple]:
        needles = [_needle(p) for p in params[: sql.count("ILIKE %s)::int")]]
        rows = []
        for node in self._indexed():
            matched = sum(1 for needle in needles if needle in node.text.lower())
            if matched:
                rows.append((node.node_id, node.content(), matched))
        rows.sort(key=lambda row: (-row[2], row[0]))
        return rows[: int(params[-1])]


# ── fakes for the encoder and the cross-encoder ─────────────────────────────


class Encoder:
    """The rag service's batch ``/rag/embed``: one call per batch, recorded."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: Any) -> tuple[str, list[list[float]], str]:
        batch = list(texts)
        self.calls.append(batch)
        return MODEL, [query_vector(t).tolist() for t in batch], ""

    def one(self, text: str) -> tuple[str, list[float], str]:
        """K3's single-query encoder over the same model."""
        return MODEL, query_vector(text).tolist(), ""


def reranker(query: str, passages: Any) -> tuple[list[float], dict[str, Any]]:
    """Raw logits: topic words matched, plus a liking for the decisive study."""
    words = next((w for w in TOPICS if w[0] in query.lower()), ())
    logits = []
    for passage in passages:
        text = passage.lower()
        hits = sum(1 for word in words if word in text)
        logits.append(2.0 * hits - 4.0 + (1.5 if hits and DECISIVE in text else 0.0))
    return logits, {"model": "fake-cross-encoder", "revision": "r1", "ms": 1.0}


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    for name in [n for n in os.environ if n.startswith("UMRS_RAG_")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hydrate_mod, "_SEARCH_TEXT", None)
    calibrate.clear_cache()
    lexical.clear_caches()
    yield
    calibrate.clear_cache()
    lexical.clear_caches()


@pytest.fixture
def forbidden(monkeypatch):
    """Every model or HTTP call the retrieve path could make, refused."""
    calls: list[Any] = []

    def refuse(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        raise AssertionError("a model or HTTP call on the retrieve path")

    monkeypatch.setattr(agent_executor, "_default_litellm_chat", refuse)
    monkeypatch.setattr(httpx, "post", refuse)
    monkeypatch.setattr(httpx.Client, "post", refuse)
    return calls


@pytest.fixture
def db() -> FakeDB:
    return FakeDB(CORPUS)


@pytest.fixture
def encoder() -> Encoder:
    return Encoder()


def retrieve(db: FakeDB, encoder: Encoder, *queries: str, **knobs: Any) -> RetrieveResult:
    spec = RetrieveSpec(queries=tuple(queries), project_id=PROJECT, **knobs)
    return pipeline.retrieve(db, spec, encode=encoder, rerank=reranker, fetch_bytes=CORPUS.fetch)


def ids(result: RetrieveResult) -> list[str]:
    return [p.source_id for p in result.passages]


ADAPTIVE = {"lexical": "off", "rerank": "off", "merge": "off"}


# ── the stages are the real modules ─────────────────────────────────────────


def test_every_lazily_imported_stage_resolves():
    lanes = pipeline.default_lanes()
    missing = {k: v.reason for k, v in lanes.items() if isinstance(v, pipeline.Unavailable)}
    assert missing == {}
    assert lanes["lex"] is lexical.lexical_lane and lanes["ident"] is lexical.ident_lane


# ── legacy parity (acceptance receipt 4) ────────────────────────────────────


def _old_retrieve_passages(db: FakeDB, encoder: Encoder, query: str) -> list[str]:
    """rag_spine.retrieve_passages before ADR-093: K3 similarity, limit 5,
    kept at score >= 0.7 (every corpus node has text, so none is dropped)."""
    backend = kq._EmbeddingsSearchBackend(db, encoder=encoder.one)
    hits = backend.search_embeddings(query, project_id=PROJECT, limit=5)
    return [hit["source_id"] for hit in hits if hit["score"] >= 0.7]


@pytest.mark.parametrize("query", [*(question(t) for t in range(len(TOPICS))), *NONSENSE])
def test_legacy_mode_returns_the_pre_adr093_ids_in_order(db, encoder, query):
    result = retrieve(db, encoder, query, k_max=5, threshold=0.7)
    assert ids(result) == _old_retrieve_passages(db, encoder, query)
    assert db.count("count(*) OVER ()") == 0  # the frozen statement, not the adaptive one
    for stage in ("lex", "ident", "fusion", "rerank", "merge"):
        assert result.lanes[stage]["status"] == "off", stage
    assert result.sufficiency.basis == "threshold"


def test_legacy_mode_keeps_the_non_chunk_node_the_old_statement_ranked(db, encoder):
    """No kind predicate in legacy mode: the file node stays where K3 put it;
    every adaptive mode ranks chunks only."""
    legacy = retrieve(db, encoder, question(0), k_max=5, threshold=0.7)
    adaptive = retrieve(db, encoder, question(0), **ADAPTIVE)
    assert "node-file-art-0-0" in ids(legacy)
    assert "node-file-art-0-0" not in ids(adaptive)
    assert len(set(ids(adaptive))) == len(ids(adaptive))  # one space, no duplicate


# ── cut sizes ───────────────────────────────────────────────────────────────


def test_the_dense_cut_keeps_the_close_cluster(db, encoder):
    result = retrieve(db, encoder, question(3), k_max=4, **ADAPTIVE)
    assert set(ids(result)) == {
        chunk_id(3, 0, 0),
        chunk_id(3, 1, 0),
        chunk_id(3, 2, 0),
        # the short tail chunk 2 ranks above its host; merge off folds nothing
        # (D-5: read-time small-to-big stays behind the S2 knob)
        chunk_id(3, 0, 2),
    }
    assert (result.sufficiency.status, result.sufficiency.basis) == ("sufficient", "z_null")
    assert (result.searched, result.space) == (56, "e5-base-v2")


def test_a_short_tail_is_folded_into_its_predecessor(db, encoder):
    """With ``merge`` on (the neighbour merge itself stubbed out here)."""
    spec = RetrieveSpec(
        queries=(question(3),), project_id=PROJECT, k_max=4, lexical="off", rerank="off", merge="on"
    )
    result = pipeline.retrieve(
        db,
        spec,
        encode=encoder,
        fetch_bytes=CORPUS.fetch,
        lanes={"merge": lambda _pool, kept, **_kw: kept},
    )
    assert chunk_id(3, 0, 2) not in ids(result)  # one k slot for the pair
    host = next(p for p in result.passages if p.source_id == chunk_id(3, 0, 1))
    assert host.text.endswith(CORPUS.nodes[chunk_id(3, 0, 2)].text)
    assert host.span == (CORPUS.nodes[chunk_id(3, 0, 1)].span[0], len(CORPUS.documents["art-3-0"]))
    assert (host.section, host.text_source) == ("part 2-3/3", "search_text")


@pytest.mark.parametrize(
    ("knobs", "size"),
    [
        ({"k_max": 4, "per_artifact": 1}, 3),  # the cap drops the tail's host
        ({"k_max": 4, "per_artifact": 1, "cutoff": "fixed"}, 4),
        ({"k_max": 2}, 2),
        ({"k_max": 1}, 1),
    ],
)
def test_cut_sizes_stay_between_one_and_k_max(db, encoder, knobs, size):
    result = retrieve(db, encoder, question(5), **ADAPTIVE, **knobs)
    assert len(result.passages) == size
    artifacts = [p.artifact_id for p in result.passages]
    assert max(artifacts.count(a) for a in artifacts) <= knobs.get("per_artifact", 2)


def _fused(db: FakeDB, encoder: Encoder) -> RetrieveResult:
    return retrieve(
        db, encoder, question(0), pool=10, lexical="on", rerank="off", merge="off", trace=True
    )


def test_the_fused_mode_gives_lexical_only_finds_their_cosines_at_once(db, encoder):
    result = _fused(db, encoder)
    assert result.lanes["lex"]["status"] == "ok" and result.lanes["fusion"]["status"] == "ok"
    assert db.count("LEAST(") == 1  # one statement for every lexical-only find
    cross = next(r for r in result.trace["candidates"] if r["node_id"] == chunk_id(*CROSS, 0))
    assert cross["lanes"] == ["lex"] and 0.6 < cross["scores"]["dense"] < 0.8


def test_the_fused_cut_keeps_passages_both_lanes_found(db, encoder):
    result = _fused(db, encoder)
    assert result.trace["mode"] == "fused" and 1 <= len(result.passages) <= 4
    assert all("lex" in p.lanes and "lexical" in p.scores for p in result.passages)
    assert result.sufficiency.status == "sufficient"


def test_the_identifier_lane_fuses_beside_dense_and_lex(db, encoder):
    result = retrieve(db, encoder, question(7), lexical="on", rerank="off", merge="off")
    assert result.lanes["ident"]["status"] == "ok" and result.lanes["ident"]["n"] == 3
    assert result.lanes["ident"]["tokens"] == ["IL-6"]
    assert {"dense", "lex", "ident"} <= set(result.passages[0].lanes)


def test_query_variants_are_encoded_once_and_fused_per_variant(db, encoder):
    result = retrieve(db, encoder, question(2), "zeolite catalyst for methanol", **ADAPTIVE)
    assert encoder.calls[0] == [question(2), "zeolite catalyst for methanol"]
    assert result.lanes["fusion"]["rankings"] == ["dense#0", "dense#1"]
    assert {chunk_id(2, j, 0) for j in range(3)} <= set(ids(result))


def test_the_rerank_cut_reorders_by_the_cross_encoder(db, encoder):
    result = retrieve(db, encoder, question(4), lexical="on", rerank="on", merge="off")
    assert result.lanes["rerank"]["status"] == "ok"
    assert result.lanes["rerank"]["model"] == "fake-cross-encoder"
    assert ids(result)[0] == chunk_id(4, 2, 0)  # the decisive study, by logit
    assert set(ids(result)) == {chunk_id(4, j, 0) for j in range(3)}
    assert result.passages[0].scores["rerank_p"] > 0.99
    assert (result.sufficiency.status, result.sufficiency.basis) == ("sufficient", "rerank_p")


def test_the_merge_grows_kept_passages_and_joins_one_artifacts(db, encoder):
    result = retrieve(db, encoder, question(6), lexical="off", rerank="off", merge="on")
    assert result.lanes["merge"]["status"] == "ok"
    by_artifact = {p.artifact_id: p for p in result.passages}
    assert sorted(by_artifact) == ["art-6-0", "art-6-1", "art-6-2"]
    assert by_artifact["art-6-0"].text == CORPUS.documents["art-6-0"]  # 0 + 1 + tail
    assert by_artifact["art-6-0"].section == "part 1-3/3"
    assert by_artifact["art-6-1"].text == CORPUS.documents["art-6-1"]  # grew by chunk 1


def test_every_stage_reports_and_is_timed(db, encoder):
    result = retrieve(db, encoder, question(1), lexical="on", rerank="on", merge="on")
    reported = {k: v["status"] for k, v in result.lanes.items()}
    assert reported == {
        "dense": "ok",
        "lex": "ok",
        "ident": "ok",  # no identifier in this query: ran, found nothing
        "fusion": "ok",
        "rerank": "ok",
        "null": "off",  # rerank mode judges by p_1
        "merge": "ok",
    }
    for stage in ("encode", "dense", "lexical", "fuse", "hydrate", "rerank", "cut", "merge"):
        assert stage in result.timing_ms, stage


# ── sufficiency on nonsense ─────────────────────────────────────────────────


@pytest.mark.parametrize("query", NONSENSE)
@pytest.mark.parametrize("lexical_mode", ["off", "on"])
def test_nonsense_is_insufficient_against_the_calibrated_null(db, encoder, query, lexical_mode):
    result = retrieve(db, encoder, query, lexical=lexical_mode, rerank="off", merge="off")
    verdict = result.sufficiency
    assert (verdict.status, verdict.basis) == ("insufficient", "z_null")
    assert verdict.value < 2.0 and verdict.threshold == 2.0
    null = result.lanes["null"]
    assert null["status"] == "ok" and null["n"] == 30 and 0.7 < null["mu"] < 0.95


@pytest.mark.parametrize("query", NONSENSE)
def test_nonsense_is_insufficient_by_the_cross_encoder_too(db, encoder, query):
    verdict = retrieve(db, encoder, query, rerank="on", merge="off").sufficiency
    assert (verdict.status, verdict.basis) == ("insufficient", "rerank_p")


def test_legacy_mode_still_passes_nonsense_through_the_absolute_cut(db, encoder):
    """The defect ADR-093 D-3 retires: e5 cosines all clear 0.7."""
    result = retrieve(db, encoder, NONSENSE[0], k_max=5, threshold=0.7)
    assert result.sufficiency.status == "sufficient" and len(result.passages) == 5


def test_relevant_questions_are_sufficient_by_a_wide_z_margin(db, encoder):
    for topic in range(len(TOPICS)):
        verdict = retrieve(db, encoder, question(topic), **ADAPTIVE).sufficiency
        assert (verdict.status, verdict.basis) == ("sufficient", "z_null"), topic
        assert verdict.value > 5.0


# ── no chat call (acceptance receipt 3) ─────────────────────────────────────


def test_no_model_or_http_call_anywhere_on_the_retrieve_path(db, encoder, forbidden):
    for knobs in (
        {"threshold": 0.7, "k_max": 5},
        ADAPTIVE,
        {"lexical": "on", "rerank": "off", "merge": "off"},
        {"lexical": "on", "rerank": "on", "merge": "on"},
    ):
        for query in (question(0), NONSENSE[0]):
            retrieve(db, encoder, query, **knobs)
    refused = rag_spine.answer_from_spine(
        db, NONSENSE[1], PROJECT, encode=encoder, rerank=reranker, fetch_bytes=CORPUS.fetch
    )
    assert refused.answer == "" and refused.abstained
    assert refused.reason.startswith("z_null")
    assert forbidden == []


def test_a_sufficient_answer_costs_exactly_one_chat_call(db, encoder, monkeypatch):
    chats: list[dict[str, Any]] = []

    def chat(messages: Any, **kwargs: Any) -> str:
        chats.append(kwargs)
        return "Swelling slows moisture sorption [1]."

    monkeypatch.setattr(agent_executor, "_default_litellm_chat", chat)
    monkeypatch.setattr(
        "umrs.jobs.knowledge_adapters._extract_chat_endpoint", lambda _p: ("http://llm", "m")
    )
    answer = rag_spine.answer_from_spine(
        db, question(0), PROJECT, encode=encoder, rerank=reranker, fetch_bytes=CORPUS.fetch
    )
    assert len(chats) == 1 and chats[0]["temperature"] == 0.0
    assert answer.citations and not answer.abstained


# ── the GET route, the wire contract and the fetch handle ───────────────────


@pytest.fixture
def client(db, encoder, monkeypatch) -> TestClient:
    monkeypatch.setattr(
        rag_api, "_require_rag_scope", lambda _request, _target=None, **_kw: ("u", PROJECT)
    )
    monkeypatch.setattr(rag_retrieve, "batch_encoder", lambda: encoder)
    monkeypatch.setattr(rag_retrieve, "site_reranker", lambda: reranker)
    monkeypatch.setattr(rag_spine, "_artifact_fetcher", lambda: CORPUS.fetch)
    app = FastAPI()
    app.include_router(rag_api.router)
    app.state.umrs_db_pool = db
    return TestClient(app)


def test_the_route_serves_the_real_core_and_its_json_round_trips(client, forbidden):
    params = [("q", question(0)), ("lexical", "on"), ("rerank", "on"), ("trace", "1")]
    response = client.get("/umrs/rag/retrieve", params=params)
    assert response.status_code == 200, response.text
    body = response.json()
    assert RetrieveResult.from_dict(body).to_dict() == body  # the wire contract holds
    parsed = RetrieveResult.from_dict(body)
    assert parsed.sufficiency.status == "sufficient" and 1 <= len(parsed.passages) <= 4
    assert parsed.passages[0].source_name.startswith("art-0-")
    assert parsed.trace["candidates"] and parsed.lanes["rerank"]["status"] == "ok"
    assert forbidden == []


def test_the_route_judges_nonsense_insufficient(client):
    body = client.get("/umrs/rag/retrieve", params={"q": NONSENSE[0]}).json()
    assert body["sufficiency"]["status"] == "insufficient"
    assert body["lanes"]["lex"]["status"] in ("ok", "off")


def test_the_fetch_handle_stitches_a_verified_window(client):
    node = chunk_id(0, 0, 1)
    body = client.get(f"/umrs/rag/passages/{node}", params={"window": 1}).json()
    assert body["text"] == CORPUS.documents["art-0-0"]  # chunks 0, 1 and the tail
    assert body["chunks"] == [0, 2] and body["text_source"] == "search_text"
    assert client.get("/umrs/rag/passages/node-chunk-nope-0").status_code == 404


# ── before migration 0120: the lanes say why, hydration reads the artifact ──


def test_without_migration_0120_the_lexical_lanes_say_why(encoder):
    db = FakeDB(CORPUS, search_text=False)
    result = retrieve(db, encoder, question(7), lexical="on", rerank="off", merge="off")
    for lane in ("lex", "ident"):  # question 7 names IL-6, so both probe
        assert result.lanes[lane]["status"] == "unavailable"
        assert "migration 0120" in result.lanes[lane]["reason"]
    assert {p.text_source for p in result.passages} == {"artifact"}  # md5-verified slices
    assert result.sufficiency.status == "sufficient"


# ── K3: the lexical lane and sufficiency agree with the core ────────────────


def _k3_verdict(db: FakeDB, encoder: Encoder, query: str) -> tuple[dict[str, Any], list]:
    emb = kq._EmbeddingsSearchBackend(db, encoder=encoder.one)
    emb.search_embeddings(query, project_id=PROJECT, limit=10)
    rag = kq._RAGSearchBackend(db)
    hits = rag.search_rag(query, project_id=PROJECT, limit=10)
    spec = kq._retrieve_spec(query, PROJECT, graph_scope=None, include_ambient=False)
    verdict = kq._sufficiency_block(kq._SufficiencyInputs(db, spec, emb, rag))
    return verdict, hits


@pytest.mark.parametrize("query", [question(0), question(7), *NONSENSE])
def test_k3_lexical_lane_and_sufficiency_agree_with_the_core(db, encoder, monkeypatch, query):
    monkeypatch.setattr(rag_retrieve, "batch_encoder", lambda: encoder)
    verdict, hits = _k3_verdict(db, encoder, query)
    core = retrieve(db, encoder, query, lexical="on", rerank="off", merge="off")
    assert verdict["status"] == core.sufficiency.status
    if core.sufficiency.status == "sufficient":
        assert hits and hits[0]["metadata"]["lane"] == "lexical"
        assert hits[0]["source_id"] in {c.node_id for c in CORPUS.chunks()}
    else:
        assert hits == []  # nonsense shares no indexed term with the corpus
