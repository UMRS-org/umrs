"""ADR-116 D-7: entity-resolution proposals against a fake store (no DB).

The real-Postgres half (store SQL, establish, the graph lane's
``aliases_reviewed``) is ``test_entity_resolution_pg.py``.
"""

from __future__ import annotations

import math

import pytest
from umrs.knowledge import entity_resolution as er
from umrs.knowledge import review_queue
from umrs.knowledge.graph_registry import default_graph_id
from umrs.knowledge.retrieval import dense, graph_lane
from umrs.knowledge.store import SemanticNodeRow

PROJECT = "proj-er"
GRAPH = "g-named0000000001"


def _node(slug, label, artifact, semantic_type="material", aliases=()):
    return SemanticNodeRow(
        node_id=f"node-sem-{GRAPH}-{slug}-{artifact}",
        label=label,
        artifact_id=artifact,
        semantic_type=semantic_type,
        aliases=tuple(aliases),
    )


class FakeStore:
    """The store surface :func:`er.propose_same_as` uses, in memory."""

    def __init__(self, nodes, pairs=()):
        self.nodes = {n.node_id: n for n in nodes}
        self.pairs = set(pairs)
        self.edges = []
        self.vectors = {}
        self.nearest_calls = []
        self.fail_edges = set()

    async def list_semantic_nodes(self, project_id, *, graph_id, node_ids=None, limit=50_000):
        assert (project_id, graph_id) == (PROJECT, GRAPH)
        rows = sorted(self.nodes.values(), key=lambda n: n.node_id)
        if node_ids is not None:
            rows = [n for n in rows if n.node_id in node_ids]
        return rows[:limit]

    async def same_as_pairs(self, node_ids):
        return {p for p in self.pairs if set(p) & set(node_ids)}

    async def upsert_edge(self, edge):
        if edge.source_id in self.fail_edges:
            raise RuntimeError("fk")
        self.edges.append(edge)

    async def upsert_node_embedding(self, node_id, model_name, embedding, **_kw):
        self.vectors[(node_id, model_name)] = list(embedding)

    async def nearest_label_neighbours(
        self, queries, *, project_id, model_name, limit, graph_id=None
    ):
        """The SQL's rules: the node itself and its own artifact are left
        out before the limit."""
        self.nearest_calls.append((project_id, model_name, graph_id))
        return {
            node_id: self._nearest(node_id, artifact_id, vector, model_name, limit)
            for node_id, artifact_id, vector in queries
        }

    def _nearest(self, node_id, artifact_id, embedding, model_name, limit):
        scored = [
            (other, 1.0 - _cos(embedding, vec))
            for (other, model), vec in self.vectors.items()
            if model == model_name
            and other != node_id
            and self.nodes[other].artifact_id != artifact_id
        ]
        return sorted(scored, key=lambda r: r[1])[:limit]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


async def _propose(store, node_ids, **kwargs):
    return await er.propose_same_as(
        store, project_id=PROJECT, graph_id=GRAPH, node_ids=node_ids, job_id="job-1", **kwargs
    )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (er.ENABLE_ENV, er.THRESHOLD_ENV, er.MAX_PER_JOB_ENV):
        monkeypatch.delenv(name, raising=False)


# ── normalization and types ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "key"),
    [
        ("ZSM-5 Zeolites", "zsm 5 zeolite"),
        ("  metal_organic   frameworks ", "metal organic framework"),
        ("Properties", "property"),
        ("boxes", "box"),
        ("gas", "gas"),
        ("analysis", "analysis"),
        ("class", "class"),
        ("ＮｉＦｅ-LDH", "nife ldh"),
    ],
)
def test_normalize_label(raw, key):
    assert er.normalize_label(raw) == key


@pytest.mark.parametrize(
    ("a", "b", "ok"),
    [
        ("material", "material", True),
        ("material", "unclassified", True),
        ("concept", "organization", True),
        ("material", "organization", False),
    ],
)
def test_types_compatible(a, b, ok):
    assert er.types_compatible(a, b) is ok


# ── the label method ─────────────────────────────────────────────────────────


async def test_a_label_match_across_two_artifacts_writes_one_proposed_edge():
    old = _node("nife-ldh", "NiFe-LDH", "art-b")
    new = _node("nife-ldh", "NiFe LDH", "art-a")
    store = FakeStore([old, new])
    result = await _propose(store, [new.node_id])
    assert (result.proposed, result.candidates, result.notes) == (1, 1, [])
    [edge] = store.edges
    a, b = sorted([old.node_id, new.node_id])
    assert edge.edge_id == f"edge-sameas-{GRAPH}-{a}-{b}"
    assert (edge.source_id, edge.target_id) == (a, b)
    assert (edge.relation_type, edge.status, edge.confidence) == ("same_as", "proposed", 1.0)
    assert edge.content == {
        "score": 1.0,
        "method": "label",
        "job_id": "job-1",
        "origin_actor": "extracted",
    }
    assert edge.graph_id == GRAPH and edge.project_id == PROJECT


async def test_an_alias_matches_a_label():
    old = _node("mof", "MOF", "art-b", aliases=["metal-organic frameworks"])
    new = _node("metal-organic-framework", "metal organic framework", "art-a")
    store = FakeStore([old, new])
    assert (await _propose(store, [new.node_id])).proposed == 1


async def test_a_pair_from_one_artifact_is_skipped():
    a = _node("ldh", "LDH", "art-a")
    b = _node("ldhs", "LDHs", "art-a")
    store = FakeStore([a, b])
    result = await _propose(store, [a.node_id, b.node_id])
    assert (result.proposed, store.edges) == (0, [])


async def test_incompatible_types_are_skipped_but_concept_is_compatible():
    old = _node("mercury", "Mercury", "art-b", semantic_type="material")
    planet = _node("mercury", "Mercury", "art-a", semantic_type="organization")
    vague = _node("mercury", "Mercury", "art-c", semantic_type="concept")
    store = FakeStore([old, planet, vague])
    await _propose(store, [planet.node_id])
    assert [(e.source_id, e.target_id) for e in store.edges] == [
        tuple(sorted([planet.node_id, vague.node_id]))
    ]


async def test_an_existing_same_as_pair_is_not_proposed_again():
    old = _node("tp53", "TP53", "art-b")
    new = _node("tp53", "tp53", "art-a")
    store = FakeStore([old, new], pairs={tuple(sorted([old.node_id, new.node_id]))})
    result = await _propose(store, [new.node_id])
    assert (result.proposed, store.edges) == (0, [])


async def test_the_cap_writes_the_first_pairs_and_notes_the_rest(monkeypatch):
    monkeypatch.setenv(er.MAX_PER_JOB_ENV, "2")
    nodes = [_node("cells", "cells", f"art-{i}") for i in range(4)]
    store = FakeStore(nodes)
    result = await _propose(store, [nodes[0].node_id])
    assert (result.proposed, result.candidates) == (2, 3)
    assert result.notes == ["entity_resolution_capped:3>2"]


@pytest.mark.parametrize("value", ["0", "false", "FALSE"])
async def test_the_flag_turns_it_off(monkeypatch, value):
    monkeypatch.setenv(er.ENABLE_ENV, value)
    old, new = _node("x", "X ray", "art-b"), _node("x", "x-ray", "art-a")
    store = FakeStore([old, new])
    result = await _propose(store, [new.node_id])
    assert (result.proposed, result.notes, store.edges) == (0, ["entity_resolution:disabled"], [])


async def test_a_failing_store_is_a_note_never_an_exception():
    class Broken(FakeStore):
        async def list_semantic_nodes(self, *_a, **_k):
            raise ConnectionError("down")

    result = await _propose(Broken([]), ["node-sem-x-art-a"])
    assert result.notes == ["entity_resolution_failed:ConnectionError"]


async def test_one_failed_edge_write_is_counted_and_the_rest_written():
    nodes = [_node("cells", "cells", f"art-{i}") for i in range(3)]
    store = FakeStore(nodes)
    store.fail_edges = {nodes[0].node_id}
    result = await _propose(store, [nodes[2].node_id])
    assert result.proposed == 1 and result.notes == ["entity_resolution_write_failed:1"]


def test_the_default_graph_has_no_edge_id_scope():
    assert er.node_id_scope(PROJECT, default_graph_id(PROJECT)) == ""
    assert er.node_id_scope(PROJECT, "") == ""
    assert er.node_id_scope(PROJECT, GRAPH) == f"{GRAPH}-"


# ── the embedding method ─────────────────────────────────────────────────────

# Deterministic label vectors: two names of one alloy close, a third far.
_VECTORS = {
    "material: stainless steel 316L": [1.0, 0.02, 0.0],
    "material: SS316L alloy": [0.99, 0.05, 0.0],
    "material: copper": [0.0, 1.0, 0.0],
    "organization: SS316L alloy": [0.99, 0.05, 0.0],
}


def _embed(texts):
    return [_VECTORS[t] for t in texts]


async def test_the_embedding_method_pairs_close_labels_under_the_label_model():
    first = _node("ss", "stainless steel 316L", "art-b")
    store = FakeStore([first])
    await _propose(store, [first.node_id], embed_labels=_embed, embedder_name="/m/e5-base-v2")
    new = _node("ss316l", "SS316L alloy", "art-a")
    far = _node("cu", "copper", "art-a")
    store.nodes.update({new.node_id: new, far.node_id: far})
    result = await _propose(
        store, [new.node_id, far.node_id], embed_labels=_embed, embedder_name="/m/e5-base-v2"
    )
    assert result.proposed == 1
    [edge] = store.edges
    assert edge.content["method"] == "embedding" and edge.confidence >= 0.92
    assert edge.content["score"] == edge.confidence
    assert {m for _n, m in store.vectors} == {"/m/e5-base-v2#label"}
    assert all(call == (PROJECT, "/m/e5-base-v2#label", GRAPH) for call in store.nearest_calls)


async def test_the_embedding_search_is_capped_and_batched(monkeypatch):
    """One exact scan per new node grew with new nodes x graph size, and
    UMRS_KNOWLEDGE_ER_MAX_PER_JOB capped written edges, not searches."""
    monkeypatch.setenv(er.EMBED_MAX_ENV, "3")
    monkeypatch.setattr(er, "SEARCH_BATCH", 2)
    nodes = [_node(f"n{i}", f"name {i}", f"art-{i}") for i in range(5)]
    store = FakeStore(nodes)
    result = await _propose(
        store,
        [n.node_id for n in nodes],
        embed_labels=lambda texts: [[1.0, float(i)] for i, _t in enumerate(texts)],
        embedder_name="e5",
    )
    assert "entity_resolution_embedding_capped:3/5" in result.notes
    assert len(store.vectors) == 3
    assert len(store.nearest_calls) == 2  # 3 searches in batches of 2


async def test_the_embedding_method_keeps_the_type_rule():
    first = _node("ss", "stainless steel 316L", "art-b")
    store = FakeStore([first])
    await _propose(store, [first.node_id], embed_labels=_embed, embedder_name="e5")
    new = _node("ss316l", "SS316L alloy", "art-a", semantic_type="organization")
    store.nodes[new.node_id] = new
    assert (
        await _propose(store, [new.node_id], embed_labels=_embed, embedder_name="e5")
    ).proposed == 0


async def test_a_threshold_above_the_similarity_proposes_nothing(monkeypatch):
    monkeypatch.setenv(er.THRESHOLD_ENV, "0.99999")
    first = _node("ss", "stainless steel 316L", "art-b")
    new = _node("ss316l", "SS316L alloy", "art-a")
    store = FakeStore([first, new])
    result = await _propose(
        store, [first.node_id, new.node_id], embed_labels=_embed, embedder_name="e5"
    )
    assert result.proposed == 0


async def test_a_failing_embedder_is_a_note_and_label_matches_still_land():
    def broken(_texts):
        raise RuntimeError("no model")

    old, new = _node("tp53", "TP53", "art-b"), _node("tp53", "TP53", "art-a")
    store = FakeStore([old, new])
    result = await _propose(store, [new.node_id], embed_labels=broken, embedder_name="e5")
    assert result.proposed == 1
    assert result.notes == ["entity_resolution_embedding_failed:RuntimeError"]


async def test_an_embedder_without_a_name_is_skipped_by_name():
    new = _node("tp53", "TP53", "art-a")
    store = FakeStore([new])
    result = await _propose(store, [new.node_id], embed_labels=_embed)
    assert result.notes == ["entity_resolution_embedding_skipped:no_embedder_name"]
    assert store.vectors == {}


@pytest.mark.parametrize(
    "model", ["/opt/umrs/models/e5-base-v2", "e5-base-v2", "intfloat/e5-base-v2"]
)
def test_the_dense_lane_never_reads_label_vectors(model):
    """The dense lane reads ``model_name = %s OR model_name LIKE %s``; a
    ``#label`` name matches neither (LIKE matches the whole string)."""
    import re

    label_model = er.label_model_name(model)
    pattern = dense.same_space_pattern(model)
    regex = "^" + re.escape(pattern).replace("%", ".*").replace(r"\\_", "_") + "$"
    assert label_model != model and re.match(regex, label_model) is None
    assert re.match(regex, "/models/e5-base-v2") is not None  # the space itself still matches


# ── the review queue and the graph lane ──────────────────────────────────────


def test_the_review_queue_lists_a_proposed_same_as_edge_as_extracted_by_its_job():
    a, b = _node("x", "X", "art-a"), _node("x", "X", "art-b")
    edge = er.same_as_edge(
        er._Match(a.node_id, b.node_id, 1.0, "label"), er._Scope(PROJECT, GRAPH, "job-9")
    )
    row = {
        "edge_id": edge.edge_id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relation_type": edge.relation_type,
        "confidence": edge.confidence,
        "content": edge.content,
        "graph_id": edge.graph_id,
    }
    listed = review_queue._edge_dict(row)
    assert (listed["origin"], listed["run_id"], listed["relation_type"]) == (
        "extracted",
        "job-9",
        "same_as",
    )


def test_the_graph_lane_matches_reviewed_aliases():
    assert "content->'aliases_reviewed'" in graph_lane._ALIAS_VALUES
    assert "content->'aliases'" in graph_lane._ALIAS_VALUES
    assert "aliases_reviewed" in graph_lane._ALIAS_HIT
    assert graph_lane._LINK_SQL.count("aliases_reviewed") >= 3


# ── the store seams, without a database ──────────────────────────────────────


class _EdgeConn:
    def __init__(self, row):
        self.row, self.executed, self.in_transaction = row, [], False

    async def fetchrow(self, _sql, *_params):
        return self.row

    async def execute(self, sql, *params):
        assert self.in_transaction
        self.executed.append((sql, params))

    def transaction(self):
        return _Tx(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _Tx:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        self.conn.in_transaction = True

    async def __aexit__(self, *_exc):
        self.conn.in_transaction = False
        return False


class _EdgePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return self.conn


def _edge_row(relation):
    return {
        "status": "proposed",
        "content": {"job_id": "j"},
        "relation_type": relation,
        "source_id": "node-sem-a",
        "target_id": "node-sem-b",
        "project_id": "p-1",
    }


async def test_establishing_an_edge_writes_no_updated_at():
    from umrs.knowledge.store import KnowledgeStore

    conn = _EdgeConn(_edge_row("related_to"))
    assert await KnowledgeStore(_EdgePool(conn)).establish_edge("e", established_by="a") is True
    [(sql, params)] = conn.executed
    assert sql.startswith("UPDATE knowledge_edges") and "updated_at" not in sql
    assert params[1]["review"] == {"established_by": "a"} and params[1]["job_id"] == "j"


async def test_establishing_same_as_copies_aliases_in_the_same_transaction():
    from umrs.knowledge import store as store_mod

    conn = _EdgeConn(_edge_row("same_as"))
    await store_mod.KnowledgeStore(_EdgePool(conn)).establish_edge("e", established_by="a")
    assert [sql for sql, _p in conn.executed][1] == store_mod._COPY_REVIEWED_ALIASES_SQL
    # both nodes must sit in the edge's project (review fix: cross-project write)
    assert conn.executed[1][1] == ("node-sem-a", "node-sem-b", "p-1")
    assert "p.project_id = $3" in conn.executed[1][0] and "n.project_id = $3" in conn.executed[1][0]


def test_the_upsert_keeps_review_keys_unless_the_new_content_sets_them():
    from umrs.knowledge import store as store_mod

    for sql in (store_mod._UPSERT_NODE_SQL, store_mod._UPSERT_NODE_TEXT_SQL):
        update = sql.split("ON CONFLICT")[1]
        assert "SET content = $5," not in update
        assert "'review', 'aliases_reviewed'" in update
        assert "NOT EXCLUDED.content ? k.key" in update
