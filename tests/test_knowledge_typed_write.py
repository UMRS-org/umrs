"""ADR-081 W2a typed writes through ``KnowledgeStore`` (KN-S7 acceptance).

Hermetic half (runs now): batch refusals, untyped writers keep the exact
pre-W2a statements, and the fallbacks when the database has no typed
columns or no seeded registry.

Composed half (``@pending`` until kn-s7-01…05 land; the Postgres tests also
need a migrated database): undeclared attr -> ``content.undeclared`` + gap;
unknown relation -> ``r4:relatedTo``; enforce refuses the whole batch and
keeps ``graph_version``; legacy rows stay valid; the raw-SQL mint agrees with
``prepare_row``; the bundled seed applies offline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from typing import Any

import pytest
from _kn_s7_pending import pending
from umrs.knowledge import rag_upload_mint as mint
from umrs.knowledge import store as store_mod
from umrs.knowledge import term_registry as reg
from umrs.knowledge import typed_write as tw
from umrs.knowledge.ontologies import bundled_by_ref, load_bundled
from umrs.knowledge.ontologies.seed_sql import render_seed_sql
from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode, KnowledgeStore
from umrs.knowledge.term_closure import admitted_set, build_closure
from umrs.knowledge.term_types import (
    EdgeDraft,
    NodeDraft,
    OntologyViolationError,
    TypingContext,
)
from umrs.knowledge.node_typing import prepare_row
from umrs.knowledge.edge_typing import prepare_edge

COMPOSED = (load_bundled, bundled_by_ref, build_closure, admitted_set, prepare_row, prepare_edge)


@pytest.fixture(autouse=True)
def _fresh_probe_memos() -> Any:
    store_mod._typed_probes.clear()
    store_mod._search_text_probes.clear()
    yield
    store_mod._typed_probes.clear()
    store_mod._search_text_probes.clear()


class _Conn:
    """Records writes; answers the typed-column probe and the registry reads."""

    def __init__(self, typed: bool, seeded: bool) -> None:
        self.typed, self.seeded = typed, seeded
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def fetchval(self, sql: str, *_a: Any) -> Any:
        if sql == tw.TYPED_COLUMNS_PROBE_SQL:
            return self.typed
        if sql == reg._SEEDED_SQL:
            return self.seeded
        raise AssertionError(f"unexpected read: {sql[:80]}")

    async def fetchrow(self, sql: str, *_a: Any) -> Any:
        assert sql == reg._GRAPH_SQL, sql[:80]
        return None

    async def execute(self, sql: str, *params: Any) -> str:
        self.executed.append((sql, params))
        return "INSERT 0 1"

    @contextlib.asynccontextmanager
    async def transaction(self) -> Any:
        yield self


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self.conn = conn

    @contextlib.asynccontextmanager
    async def acquire(self) -> Any:
        yield self.conn


def _node(**kw: Any) -> KnowledgeNode:
    base: dict[str, Any] = {"node_id": "n-1", "project_id": "p", "label": "img", "node_type": "file"}
    return KnowledgeNode(**{**base, **kw})


def _edge(**kw: Any) -> KnowledgeEdge:
    base: dict[str, Any] = {
        "edge_id": "e-1", "project_id": "p", "source_id": "a", "target_id": "b",
        "relation_type": "frobs",
    }
    return KnowledgeEdge(**{**base, **kw})


def _batch(**kw: Any) -> tw.WriteBatch:
    base: dict[str, Any] = {"graph_id": "g", "project_id": "p", "writer": "w", "origin_actor": "agent"}
    return tw.WriteBatch(**{**base, **kw})


# ── hermetic ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("batch", "needle"),
    [
        (_batch(origin_actor="loop"), "loop"),
        (_batch(graph_id=""), "graph_id"),
        (_batch(nodes=(_node(project_id="q"),)), "n-1"),
        (_batch(nodes=(_node(graph_id="other"),)), "n-1"),
        (_batch(edges=(_edge(project_id="q"),)), "e-1"),
    ],
)
def test_malformed_batch_is_refused_before_any_read(batch, needle):
    conn = _Conn(typed=True, seeded=True)
    with pytest.raises(ValueError, match=needle):
        asyncio.run(KnowledgeStore(_Pool(conn)).upsert_typed(batch))
    assert conn.executed == []


def test_untyped_node_takes_the_legacy_statement_without_a_probe():
    conn = _Conn(typed=True, seeded=True)
    asyncio.run(KnowledgeStore(_Pool(conn)).upsert_node(_node()))
    assert [sql for sql, _ in conn.executed] == [store_mod._UPSERT_NODE_SQL]
    assert store_mod._typed_probes == {}


def test_untyped_edge_takes_the_legacy_statement():
    conn = _Conn(typed=True, seeded=True)
    asyncio.run(KnowledgeStore(_Pool(conn)).upsert_edge(_edge()))
    sql, params = conn.executed[0]
    assert "relation_class" not in sql and params[4] == "frobs"


def test_typed_node_without_0145_is_written_untyped_with_attrs_kept():
    conn = _Conn(typed=False, seeded=True)
    node = _node(node_class="image", attrs={"pixel_size": 1}, content={"kind": "image"})
    asyncio.run(KnowledgeStore(_Pool(conn)).upsert_node(node))
    sql, params = conn.executed[-1]
    assert sql == store_mod._UPSERT_NODE_SQL
    assert params[4] == {"kind": "image", "undeclared": {"pixel_size": 1}}


def test_typed_node_with_an_unseeded_registry_is_written_untyped():
    conn = _Conn(typed=True, seeded=False)
    asyncio.run(KnowledgeStore(_Pool(conn)).upsert_node(_node(node_class="image")))
    assert [sql for sql, _ in conn.executed] == [store_mod._UPSERT_NODE_SQL]


def test_typed_edge_with_an_unseeded_registry_is_written_untyped():
    conn = _Conn(typed=True, seeded=False)
    asyncio.run(KnowledgeStore(_Pool(conn)).upsert_edge(_edge(relation_class="r4:cites")))
    sql, _ = conn.executed[-1]
    assert sql.lstrip().startswith("INSERT INTO knowledge_edges") and "relation_class" not in sql


def test_batch_with_an_unseeded_registry_writes_untyped_rows_and_no_audit():
    conn = _Conn(typed=True, seeded=False)
    batch = _batch(nodes=(_node(node_class="image", attrs={"x": 1}),), edges=(_edge(),))
    result = asyncio.run(KnowledgeStore(_Pool(conn)).upsert_typed(batch))
    assert result is None
    sqls = [sql for sql, _ in conn.executed]
    assert sqls[0] == store_mod._UPSERT_NODE_SQL
    assert conn.executed[0][1][9] == "g"  # '' graph -> the batch graph
    assert not any("knowledge_write_batches" in s for s in sqls)


# ── composed (pending the worker queue) ──────────────────────────────────────


def _core_ctx() -> TypingContext:
    return TypingContext.from_snapshot(reg.bundled_snapshot())


@pending(*COMPOSED)
def test_bundled_core_context_resolves_legacy_slugs():
    ctx = _core_ctx()
    assert ctx.resolve("part_of", "relation").term_id == "r4:partOf"  # type: ignore[union-attr]
    assert ctx.resolve("document", "class").term_id == "r4:Document"  # type: ignore[union-attr]
    assert ctx.is_a("r4:Paper", "prov:Entity")


@pending(*COMPOSED)
def test_raw_mint_classes_agree_with_prepare_row():
    ctx = _core_ctx()
    doc = prepare_row(ctx, NodeDraft("d", "file", mint.MINT_DOC_CLASS))
    chunk = prepare_row(ctx, NodeDraft("c", "entity", mint.MINT_CHUNK_CLASS))
    part = prepare_edge(ctx, EdgeDraft("e", "part_of"), source_class=chunk.node_class,
                        target_class=doc.node_class)
    assert (doc.node_class, doc.node_type, doc.violations) == (mint.MINT_DOC_CLASS, "file", ())
    assert (chunk.node_class, chunk.node_type, chunk.violations) == (
        mint.MINT_CHUNK_CLASS, "entity", (),
    )
    assert (part.relation_class, part.violations) == (mint.MINT_PART_OF_RELATION, ())


def _pg_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    host = os.environ.get("POSTGRES_HOST")
    if explicit or not host:
        return explicit or None
    user = os.environ.get("POSTGRES_USER", "test")
    secret = os.environ.get("POSTGRES_PASSWORD", "test")
    port = os.environ.get("POSTGRES_PORT", "5432")
    name = os.environ.get("POSTGRES_DB", "umrs_test")
    return f"postgresql://{user}:{secret}@{host}:{port}/{name}"


needs_pg = pytest.mark.skipif(not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*")


def _seed_bundled(url: str) -> None:
    """Apply the bundled seed (idempotent), as the seed migration does."""
    import psycopg2

    docs = bundled_by_ref()
    terms = [t for d in docs.values() for t in d.terms]
    sql = render_seed_sql(
        list(docs.values()), build_closure(terms), {ref: admitted_set(ref, docs) for ref in docs}
    )
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    finally:
        conn.close()


async def _codecs(conn: Any) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )


async def _pool(url: str) -> Any:
    import asyncpg

    return await asyncpg.create_pool(url, min_size=1, max_size=2, init=_codecs)


async def _cleanup(pool: Any, project: str) -> None:
    for table in ("knowledge_edges", "knowledge_nodes"):
        await pool.execute(f"DELETE FROM {table} WHERE project_id = $1", project)
    await pool.execute("DELETE FROM knowledge_write_batches WHERE project_id = $1", project)
    await pool.execute("DELETE FROM knowledge_graphs WHERE project_id = $1", project)


async def _warn_batch(url: str, project: str) -> dict[str, Any]:
    pool = await _pool(url)
    graph = f"g-{project}"
    try:
        result = await KnowledgeStore(pool).upsert_typed(
            tw.WriteBatch(
                graph_id=graph, project_id=project, writer="test.kn_s7", origin_actor="agent",
                nodes=(
                    KnowledgeNode(f"img-{project}", project, "frame", "entity",
                                  content={"kind": "image"}, node_class="image",
                                  attrs={"pixel_size": 0.3, "colour": "red"}),
                    KnowledgeNode(f"doc-{project}", project, "paper", "file",
                                  node_class="r4:Document"),
                ),
                edges=(
                    KnowledgeEdge(f"e1-{project}", project, f"img-{project}", f"doc-{project}",
                                  "frobs"),
                    KnowledgeEdge(f"e2-{project}", project, f"img-{project}", f"doc-{project}",
                                  "part_of"),
                ),
            )
        )
        nodes = {r["node_id"]: dict(r) for r in await pool.fetch(
            "SELECT node_id, node_class, node_type, attrs, content FROM knowledge_nodes "
            "WHERE project_id = $1", project)}
        edges = {r["edge_id"]: dict(r) for r in await pool.fetch(
            "SELECT edge_id, relation_type, relation_class, content FROM knowledge_edges "
            "WHERE project_id = $1", project)}
        gaps = {(r["kind"], r["value"]) for r in await pool.fetch(
            "SELECT kind, value FROM knowledge_vocab_gaps WHERE graph_id = $1", graph)}
        batch = await pool.fetchrow(
            "SELECT origin_actor, counts FROM knowledge_write_batches WHERE batch_id = $1",
            result.batch_id if result else "")
        return {"result": result, "nodes": nodes, "edges": edges, "gaps": gaps, "batch": batch}
    finally:
        await pool.execute("DELETE FROM knowledge_vocab_gaps WHERE graph_id = $1", graph)
        await _cleanup(pool, project)
        await pool.close()


@pytest.fixture(scope="module")
def warn_batch() -> dict[str, Any]:
    if not _pg_url():
        pytest.skip("needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*")
    _seed_bundled(_pg_url() or "")
    project = f"p-kns7-{uuid.uuid4().hex[:8]}"
    return {**asyncio.run(_warn_batch(_pg_url() or "", project)), "project": project}


@needs_pg
@pending(*COMPOSED, render_seed_sql)
def test_warn_batch_types_nodes_and_moves_undeclared_attrs(warn_batch):
    img = warn_batch["nodes"][f"img-{warn_batch['project']}"]
    assert (img["node_class"], img["node_type"]) == ("r4:Image", "file")
    assert img["attrs"] == {"r4:pixelSize": 0.3}
    assert img["content"] == {"kind": "image", "undeclared": {"colour": "red"}}


@needs_pg
@pending(*COMPOSED, render_seed_sql)
def test_warn_batch_unknown_relation_falls_back_to_related_to(warn_batch):
    project = warn_batch["project"]
    unknown, part = warn_batch["edges"][f"e1-{project}"], warn_batch["edges"][f"e2-{project}"]
    assert (unknown["relation_class"], unknown["relation_type"]) == ("r4:relatedTo", "frobs")
    assert unknown["content"]["rel_raw"] == "frobs"
    assert part["relation_class"] == "r4:partOf"


@needs_pg
@pending(*COMPOSED, render_seed_sql)
def test_warn_batch_counts_gaps_and_audits_the_actor(warn_batch):
    assert {("attribute", "colour"), ("relation", "frobs")} <= warn_batch["gaps"]
    assert warn_batch["batch"]["origin_actor"] == "agent"
    assert warn_batch["batch"]["counts"] == {"edges": 2, "nodes": 2}  # the pool decodes jsonb
    assert (warn_batch["result"].mode, warn_batch["result"].ontology_ref) == ("warn", "r4-core@2")


async def _enforce(url: str, project: str) -> dict[str, Any]:
    pool = await _pool(url)
    graph = f"g-{project}"
    store = KnowledgeStore(pool)
    try:
        await pool.execute(
            "INSERT INTO knowledge_graphs (graph_id, name, project_id, ontology_id, "
            "ontology_version, ontology_mode) VALUES ($1, $1, $2, 'r4-core', '2', 'enforce')",
            graph, project)
        bad = tw.WriteBatch(graph, project, "test.kn_s7", "user", nodes=(
            KnowledgeNode(f"ok-{project}", project, "ok", "file", node_class="r4:Document"),
            KnowledgeNode(f"bad-{project}", project, "bad", "entity", node_class="ex:Nope"),
        ))
        refused = None
        try:
            await store.upsert_typed(bad)
        except OntologyViolationError as exc:
            refused = exc
        written = await pool.fetchval(
            "SELECT count(*) FROM knowledge_nodes WHERE project_id = $1", project)
        version_after_refusal = await pool.fetchval(
            "SELECT graph_version FROM knowledge_graphs WHERE graph_id = $1", graph)
        await store.upsert_typed(tw.WriteBatch(graph, project, "test.kn_s7", "user", nodes=(
            KnowledgeNode(f"ok-{project}", project, "ok", "file", node_class="r4:Document"),)))
        version_after_write = await pool.fetchval(
            "SELECT graph_version FROM knowledge_graphs WHERE graph_id = $1", graph)
        return {"refused": refused, "written": written, "v0": version_after_refusal,
                "v1": version_after_write}
    finally:
        await _cleanup(pool, project)
        await pool.close()


@needs_pg
@pending(*COMPOSED, render_seed_sql)
def test_enforce_graph_refuses_the_whole_batch_and_keeps_its_version():
    _seed_bundled(_pg_url() or "")
    project = f"p-kns7-{uuid.uuid4().hex[:8]}"
    seen = asyncio.run(_enforce(_pg_url() or "", project))
    assert isinstance(seen["refused"], OntologyViolationError)
    assert [v.code for v in seen["refused"].violations] == ["class_unknown"]
    assert seen["written"] == 0 and seen["v0"] == 0
    assert seen["v1"] == 1


async def _legacy_and_typed(url: str, project: str) -> dict[str, Any]:
    pool = await _pool(url)
    store = KnowledgeStore(pool)
    try:
        await store.upsert_node(KnowledgeNode(f"old-{project}", project, "old", "concept"))
        await store.upsert_edge(KnowledgeEdge(f"e-{project}", project, f"old-{project}",
                                              f"old-{project}", "frobs", relation_class="frobs"))
        rows = await pool.fetch(
            "SELECT node_class, attrs FROM knowledge_nodes WHERE project_id = $1", project)
        edge = await pool.fetchrow(
            "SELECT relation_class, content FROM knowledge_edges WHERE project_id = $1", project)
        return {"rows": [dict(r) for r in rows], "edge": dict(edge)}
    finally:
        await _cleanup(pool, project)
        await pool.close()


@needs_pg
@pending(*COMPOSED, render_seed_sql)
def test_legacy_rows_stay_valid_and_a_single_typed_edge_falls_back():
    _seed_bundled(_pg_url() or "")
    project = f"p-kns7-{uuid.uuid4().hex[:8]}"
    seen = asyncio.run(_legacy_and_typed(_pg_url() or "", project))
    assert seen["rows"] == [{"node_class": None, "attrs": {}}]
    assert seen["edge"]["relation_class"] == "r4:relatedTo"
    assert seen["edge"]["content"] == {"rel_raw": "frobs"}
