"""ADR-116 D-7/D-8 on a real Postgres (pgvector): the store SQL entity
resolution needs.

Runs when ``UMRS_TEST_DB_URL`` (or the CI job's ``POSTGRES_*``) names a
migrated database, and skips cleanly otherwise. Every row a test writes
belongs to a project id unique to that test and is deleted afterwards.

Pinned against the old code: a re-upsert dropped ``content.review`` and
``content.aliases_reviewed``; establishing an edge wrote ``updated_at``,
which ``knowledge_edges`` does not have (UndefinedColumn).
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager

import pytest
from umrs.knowledge import entity_resolution as er
from umrs.knowledge.retrieval import graph_lane
from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode, KnowledgeStore


def _pg_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    host = os.environ.get("POSTGRES_HOST")
    if explicit or not host:
        return explicit or None
    user = os.environ.get("POSTGRES_USER", "test")
    secret = os.environ.get("POSTGRES_PASSWORD", "test")
    port, name = os.environ.get("POSTGRES_PORT", "5432"), os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{secret}@{host}:{port}/{name}"


pytestmark = pytest.mark.skipif(
    not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*"
)

GRAPH = "g-er0000000000001"


async def _init_codecs(conn) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )


@pytest.fixture
async def db():
    asyncpg = pytest.importorskip("asyncpg")
    pool = await asyncpg.create_pool(dsn=_pg_url(), min_size=1, max_size=2, init=_init_codecs)
    project = f"er-it-{uuid.uuid4().hex[:10]}"
    try:
        yield KnowledgeStore(pool), pool, project
    finally:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM knowledge_nodes WHERE project_id = $1", project
            )  # edges and vectors cascade
        await pool.close()


def _sem(project, slug, artifact, label, semantic_type="material", **content):
    node_id = f"node-sem-{GRAPH}-{slug}-{artifact}-{project}"
    return KnowledgeNode(
        node_id=node_id,
        project_id=project,
        label=label,
        node_type="entity",
        content={"artifact_id": artifact, "semantic_type": semantic_type, **content},
        content_hash=node_id,
        graph_id=GRAPH,
        status="proposed",
    )


async def _content(pool, node_id):
    async with pool.acquire() as conn:
        return await conn.fetchval("SELECT content FROM knowledge_nodes WHERE node_id=$1", node_id)


# ── D-8: upsert keeps the review ─────────────────────────────────────────────


async def test_a_reupsert_keeps_review_and_reviewed_aliases(db):
    store, pool, project = db
    node = _sem(project, "tp53", "art-a", "TP53")
    await store.upsert_node(node)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE knowledge_nodes SET content = content || $2 WHERE node_id = $1",
            node.node_id,
            {"review": {"established_by": "alice"}, "aliases_reviewed": ["p53"]},
        )
    await store.upsert_node(_sem(project, "tp53", "art-a", "TP53", source="re-ingest"))
    content = await _content(pool, node.node_id)
    assert content["review"] == {"established_by": "alice"}
    assert content["aliases_reviewed"] == ["p53"]
    assert content["source"] == "re-ingest"


async def test_new_content_that_sets_the_keys_wins(db):
    store, pool, project = db
    node = _sem(project, "tp53", "art-a", "TP53", review={"established_by": "alice"})
    await store.upsert_node(node)
    await store.upsert_node(
        _sem(
            project, "tp53", "art-a", "TP53", review={"established_by": "bob"}, aliases_reviewed=[]
        )
    )
    content = await _content(pool, node.node_id)
    assert content["review"] == {"established_by": "bob"} and content["aliases_reviewed"] == []


async def test_a_chunk_upsert_keeps_its_search_text_rule(db):
    """The search_text statement carries the same content merge."""
    store, pool, project = db
    chunk = KnowledgeNode(
        node_id=f"node-chunk-{project}-0",
        project_id=project,
        label="c",
        node_type="entity",
        content={"kind": "chunk", "chunk_md5": "m", "review": {"established_by": "a"}},
        graph_id=GRAPH,
        search_text="body",
    )
    await store.upsert_node(chunk)
    again = KnowledgeNode(
        **{**chunk.__dict__, "content": {"kind": "chunk", "chunk_md5": "m"}, "search_text": None}
    )
    await store.upsert_node(again)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT content, search_text FROM knowledge_nodes WHERE node_id=$1", chunk.node_id
        )
    assert row["content"]["review"] == {"established_by": "a"} and row["search_text"] == "body"


# ── D-8: establishing an edge ────────────────────────────────────────────────


async def _edge(store, project, a, b, relation="related_to"):
    edge = KnowledgeEdge(
        edge_id=f"edge-{relation}-{a.node_id}-{b.node_id}",
        project_id=project,
        source_id=a.node_id,
        target_id=b.node_id,
        relation_type=relation,
        content={"job_id": "job-1"},
        status="proposed",
        graph_id=GRAPH,
    )
    await store.upsert_edge(edge)
    return edge


async def test_establishing_a_proposed_edge_works_on_postgres(db):
    store, pool, project = db
    a, b = _sem(project, "a", "art-a", "A"), _sem(project, "b", "art-b", "B")
    await store.upsert_node(a)
    await store.upsert_node(b)
    edge = await _edge(store, project, a, b)
    ok = await store.establish_edge(
        edge.edge_id, established_by="alice", evidence_refs=["doc-1"], confidence=0.8
    )
    assert ok is True
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT status, confidence, content FROM knowledge_edges WHERE edge_id=$1",
            edge.edge_id,
        )
    assert (row["status"], round(row["confidence"], 3)) == ("established", 0.8)
    assert row["content"]["review"] == {"established_by": "alice", "evidence_refs": ["doc-1"]}
    assert row["content"]["job_id"] == "job-1"
    assert await _content(pool, a.node_id) == a.content  # not same_as: nodes untouched


async def test_establishing_same_as_copies_each_label_into_the_other(db):
    store, pool, project = db
    a = _sem(project, "nife", "art-a", "NiFe-LDH", aliases_reviewed=["NiFe LDH"])
    b = _sem(project, "nickel-iron", "art-b", "nickel-iron layered double hydroxide")
    await store.upsert_node(a)
    await store.upsert_node(b)
    edge = await _edge(store, project, a, b, relation="same_as")
    assert await store.establish_edge(edge.edge_id, established_by="alice") is True
    assert (await _content(pool, a.node_id))["aliases_reviewed"] == [
        "NiFe LDH",
        "nickel-iron layered double hydroxide",
    ]
    assert (await _content(pool, b.node_id))["aliases_reviewed"] == ["NiFe-LDH"]
    # a second promotion (new evidence) adds no duplicate
    await store.establish_edge(edge.edge_id, established_by="bob", evidence_refs=["r"])
    assert (await _content(pool, b.node_id))["aliases_reviewed"] == ["NiFe-LDH"]


async def test_reviewed_aliases_stop_at_the_cap(db):
    store, pool, project = db
    full = [f"alias {i}" for i in range(50)]
    a = _sem(project, "a", "art-a", "A", aliases_reviewed=full)
    b = _sem(project, "b", "art-b", "B")
    await store.upsert_node(a)
    await store.upsert_node(b)
    edge = await _edge(store, project, a, b, relation="same_as")
    await store.establish_edge(edge.edge_id, established_by="alice")
    assert (await _content(pool, a.node_id))["aliases_reviewed"] == full
    assert (await _content(pool, b.node_id))["aliases_reviewed"] == ["A"]


# ── D-7: the reads and propose_same_as on Postgres ───────────────────────────


async def test_list_semantic_nodes_reads_extracted_nodes_only(db):
    store, _pool, project = db
    sem = _sem(project, "tp53", "art-a", "TP53", aliases=["p53"], aliases_reviewed=["TP-53"])
    chunk = KnowledgeNode(
        node_id=f"node-sem-{GRAPH}-chunk-{project}",
        project_id=project,
        label="chunk",
        node_type="entity",
        content={"kind": "chunk", "semantic_type": "material", "artifact_id": "art-a"},
        graph_id=GRAPH,
    )
    other_graph = KnowledgeNode(
        **{**sem.__dict__, "node_id": sem.node_id + "-x", "graph_id": "g-other"}
    )
    for node in (sem, chunk, other_graph):
        await store.upsert_node(node)
    rows = await store.list_semantic_nodes(project, graph_id=GRAPH)
    assert [(r.node_id, r.aliases) for r in rows] == [(sem.node_id, ("p53", "TP-53"))]
    assert await store.list_semantic_nodes(project, graph_id=GRAPH, node_ids=["nope"]) == []


async def test_nearest_node_ids_keeps_one_graph(db):
    store, _pool, project = db
    a = _sem(project, "a", "art-a", "A")
    b = KnowledgeNode(**{**_sem(project, "b", "art-b", "B").__dict__, "graph_id": "g-other"})
    for node in (a, b):
        await store.upsert_node(node)
        await store.upsert_node_embedding(node.node_id, "m#label", [1.0, 0.0])
    both = await store.nearest_node_ids([1.0, 0.0], project_id=project, model_name="m#label")
    one = await store.nearest_node_ids(
        [1.0, 0.0], project_id=project, model_name="m#label", graph_id=GRAPH
    )
    assert {n for n, _d in both} == {a.node_id, b.node_id}
    assert [n for n, _d in one] == [a.node_id]


async def test_propose_same_as_writes_one_proposed_edge_on_postgres(db):
    store, pool, project = db
    old = _sem(project, "nife-ldh", "art-b", "NiFe-LDH")
    new = _sem(project, "nife-ldh", "art-a", "NiFe LDHs")
    same_doc = _sem(project, "nife", "art-a", "NiFe LDH", semantic_type="concept")
    for node in (old, new, same_doc):
        await store.upsert_node(node)
    result = await er.propose_same_as(
        store, project_id=project, graph_id=GRAPH, node_ids=[new.node_id], job_id="job-7"
    )
    # new~old only: new~same_doc share an artifact, and same_doc is not new
    assert (result.proposed, result.notes) == (1, [])
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT edge_id, source_id, target_id, status, confidence, content, graph_id "
            "FROM knowledge_edges WHERE project_id=$1 AND relation_type='same_as' ORDER BY 1",
            project,
        )
    [row] = rows
    a, b = sorted([old.node_id, new.node_id])
    assert (row["source_id"], row["target_id"], row["status"]) == (a, b, "proposed")
    assert (row["edge_id"], row["graph_id"]) == (f"edge-sameas-{GRAPH}-{a}-{b}", GRAPH)
    assert row["content"] == {
        "score": 1.0,
        "method": "label",
        "job_id": "job-7",
        "origin_actor": "extracted",
    }
    again = await er.propose_same_as(
        store, project_id=project, graph_id=GRAPH, node_ids=[new.node_id], job_id="job-8"
    )
    assert again.proposed == 0  # existing same_as pairs are not proposed twice


async def test_propose_same_as_by_embedding_on_postgres(db):
    store, _pool, project = db
    vectors = {
        "material: SS316L": [1.0, 0.0, 0.01],
        "material: stainless steel 316L": [1.0, 0.0, 0.0],
    }
    a = _sem(project, "ss316l", "art-a", "SS316L")
    b = _sem(project, "ss", "art-b", "stainless steel 316L")
    for node in (a, b):
        await store.upsert_node(node)
    result = await er.propose_same_as(
        store,
        project_id=project,
        graph_id=GRAPH,
        node_ids=[a.node_id, b.node_id],
        job_id="job-1",
        embed_labels=lambda texts: [vectors[t] for t in texts],
        embedder_name="/m/e5-base-v2",
    )
    assert (result.proposed, result.notes) == (1, [])


# ── the graph lane over aliases_reviewed ─────────────────────────────────────

_SHADOW_NODES = (
    "SELECT 'node-sem-nife-art-a'::text AS node_id, 'NiFe-LDH'::text AS label,"
    " jsonb_build_object('artifact_id', 'art-a', 'semantic_type', 'material',"
    " 'aliases_reviewed', jsonb_build_array('nickel iron hydroxide')) AS content,"
    " 'default'::text AS project_id, 'project'::text AS visibility,"
    " 'entity'::text AS node_type, ''::text AS graph_id"
)
_NO_EDGES = (
    "SELECT ''::text AS target_id, ''::text AS relation_type, '{}'::jsonb AS content WHERE false"
)


class _Shadowed:
    HEAD = "WITH q AS MATERIALIZED ("

    def __init__(self, cur):
        self.cur = cur

    def execute(self, sql, params):
        shadow = (
            f"WITH knowledge_nodes AS ({_SHADOW_NODES}), knowledge_edges AS ({_NO_EDGES}), "
            "q AS MATERIALIZED ("
        )
        self.cur.execute(sql.replace(self.HEAD, shadow), params)

    def fetchall(self):
        return self.cur.fetchall()


def test_the_graph_lane_links_an_entity_by_a_reviewed_alias_on_postgres():
    from umrs.knowledge.retrieval.types import RetrieveSpec

    psycopg2 = pytest.importorskip("psycopg2")
    spec = RetrieveSpec(project_id="default", queries=("nickel iron hydroxide",))
    conn = psycopg2.connect(_pg_url())
    try:
        conn.set_session(readonly=True)
        with conn, conn.cursor() as cur:
            entities = graph_lane._linked(_Shadowed(cur), spec)
    finally:
        conn.close()
    assert [(e.label, e.artifacts) for e in entities] == [("NiFe-LDH", {"art-a"})]


# ── review fixes ─────────────────────────────────────────────────────────────


async def test_close_nodes_of_the_same_document_do_not_hide_a_cross_document_match(db):
    """The node itself and 11 near nodes of its own artifact filled LIMIT 10;
    the other document's node (cosine 0.949) was never seen."""
    store, _pool, project = db
    query = _sem(project, "q", "art-a", "Q")
    twins = [_sem(project, f"t{i}", "art-a", f"T{i}") for i in range(11)]
    other = _sem(project, "o", "art-b", "O")
    vectors = {"material: Q": [1.0, 0.0]}
    vectors.update({f"material: T{i}": [1.0, 0.001 * (i + 1)] for i in range(11)})
    vectors["material: O"] = [1.0, 0.333]  # cosine ~0.949, above 0.92
    for node in (query, *twins, other):
        await store.upsert_node(node)
    result = await er.propose_same_as(
        store,
        project_id=project,
        graph_id=GRAPH,
        node_ids=[n.node_id for n in (query, *twins, other)],
        job_id="job-er",
        embed_labels=lambda texts: [vectors[t] for t in texts],
        embedder_name="m",
    )
    # every art-a node pairs with the art-b node; no same-document pair
    assert (result.proposed, result.notes) == (12, [])


async def test_establishing_same_as_never_writes_into_another_project(db):
    store, pool, project = db
    foreign = f"{project}-x"
    mine = _sem(project, "a", "art-a", "Mine")
    theirs = _sem(foreign, "b", "art-b", "Secret label")
    try:
        await store.upsert_node(mine)
        await store.upsert_node(theirs)
        edge = await _edge(store, project, mine, theirs, relation="same_as")
        await store.establish_edge(edge.edge_id, established_by="mallory", evidence_refs=["x"])
        assert await _content(pool, theirs.node_id) == theirs.content
        assert "aliases_reviewed" not in await _content(pool, mine.node_id)
    finally:
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM knowledge_nodes WHERE project_id = $1", foreign)


class _Interleaving:
    """Runs ``during`` while establish_node sits between its content read and
    its write (on another connection), to show the row lock."""

    def __init__(self, pool, during) -> None:
        self._pool, self._during, self.task = pool, during, None

    @asynccontextmanager
    async def acquire(self):
        async with self._pool.acquire() as conn:
            yield _ConnProxy(conn, self)


class _ConnProxy:
    def __init__(self, conn, owner) -> None:
        self._conn, self._owner = conn, owner

    def __getattr__(self, name):
        return getattr(self._conn, name)

    async def fetchval(self, sql, *args):
        value = await self._conn.fetchval(sql, *args)
        if "SELECT content" in sql:
            self._owner.task = asyncio.create_task(self._owner._during())
            await asyncio.sleep(0.3)
        return value


async def test_a_node_promotion_does_not_overwrite_a_concurrent_reingest(db):
    store, pool, project = db
    node = _sem(project, "tp53", "art-a", "TP53")
    await store.upsert_node(node)
    reingest = _sem(project, "tp53", "art-a", "TP53", evidence=[{"quote": "new"}])
    racing = _Interleaving(pool, lambda: store.upsert_node(reingest))
    await KnowledgeStore(racing).establish_node(node.node_id, established_by="alice")
    await racing.task
    content = await _content(pool, node.node_id)
    assert content["review"] == {"established_by": "alice"}
    assert content["evidence"] == [{"quote": "new"}]  # the re-ingest is kept


async def test_label_vectors_never_become_the_bundle_embedding_pin(db):
    from pathlib import Path

    from umrs.knowledge.bundle_export import KnowledgeExportEngine

    store, _pool, project = db
    nodes = [_sem(project, f"n{i}", "art-a", f"N{i}") for i in range(4)]
    for i, node in enumerate(nodes):
        await store.upsert_node(node)
        await store.upsert_node_embedding(node.node_id, "/m/e5#label", [1.0, float(i)])
    await store.upsert_node_embedding(nodes[0].node_id, "/m/e5", [1.0, 0.0, 0.0])
    engine = KnowledgeExportEngine(store, Path("/nonexistent"))
    pin = await engine._pin_from_node_embeddings(project)
    assert pin is not None and pin["model_path"] == "/m/e5"
