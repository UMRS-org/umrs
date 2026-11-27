"""The scope overlay's write side (ADR-108 D-4, slice S-W3a).

A write under a work-level scope token lands proposed and tagged with the
level's path in the ``scope_path`` column (migration 0150); a root write is
unchanged. Acceptance: scoped ingest stores the column, root ingest is
unchanged, the migration is idempotent.

Sections: the pure rules (``scope_overlay``), the ingest/understand wiring
(``knowledge_adapters``, ``client_provenance``, ``store``), the client-spec
refusal, the worker tasks' modules (``scope_lineage``/``parent_scope``,
``scope_tagging``, ``scope_subtree``), and Postgres checks (env-gated:
``UMRS_TEST_DB_URL`` or the CI job's ``POSTGRES_*``) in a scratch schema
dropped afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.knowledge import scope_overlay as so
from umrs.knowledge import scope_subtree as sub
from umrs.knowledge import scope_tagging as tag
from umrs.knowledge import store as store_mod
from umrs.knowledge.client_provenance import attach_scope_path
from umrs.knowledge.session_turns import client_spec_refusal
from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode

_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "umrs"
    / "migrations"
    / "0150_knowledge_scope_path.sql"
)


# ── pure rules (scope_overlay, implemented with the seam) ──────────────────


@pytest.mark.parametrize("raw", [None, "", "   ", "\t\n"])
def test_an_empty_path_is_the_root_scope(raw: Any) -> None:
    assert so.clean_scope_path(raw) is None


def test_a_valid_path_is_kept_and_stripped() -> None:
    assert so.clean_scope_path(" s-ab12/loop-7/iter-3 ") == "s-ab12/loop-7/iter-3"


@pytest.mark.parametrize(
    "raw",
    ["a//b", "/a", "a/", "a b", "a/../b", "-a", "a/" * 8 + "a", "x" * 65, 7, ["a"]],
)
def test_a_malformed_path_raises_never_reads_as_root(raw: Any) -> None:
    with pytest.raises(so.ScopeOverlayError):
        so.clean_scope_path(raw)


def test_the_spec_path_is_read_and_checked() -> None:
    assert so.scoped_spec_path({"scope_path": "s-1/loop-2"}) == "s-1/loop-2"
    assert so.scoped_spec_path({}) is None
    assert so.scoped_spec_path(None) is None
    with pytest.raises(so.ScopeOverlayError):
        so.scoped_spec_path({"scope_path": "a//b"})


def test_a_scoped_row_is_always_proposed() -> None:
    assert so.scoped_tier("established", "s-1") == "proposed"
    assert so.scoped_tier("established", None) == "established"
    assert so.scoped_tier("proposed", None) == "proposed"


def test_the_tier_stamp_rule() -> None:
    vouched = {"origin_actor": "vouched", "review": {"established_by": "u"}}
    assert so.scoped_tier_stamp(False, vouched, None, "j") == (False, vouched)
    assert so.scoped_tier_stamp(False, vouched, "s-1", "j") == (
        True,
        {"origin_actor": "agent", "job_id": "j"},
    )
    agent = {"origin_actor": "agent", "job_id": "j0"}
    assert so.scoped_tier_stamp(True, agent, "s-1", "j") == (True, agent)


# ── attach_scope_path: the only place a descriptor gets its scope ──────────


def test_a_scoped_ingest_tags_each_descriptor_and_its_metadata() -> None:
    descs: list[dict[str, Any]] = [{"metadata": {"source_name": "a"}}, {}]
    attach_scope_path(descs, "s-1/loop-2")
    assert [d["scope_path"] for d in descs] == ["s-1/loop-2", "s-1/loop-2"]
    assert descs[0]["metadata"] == {"source_name": "a", "scope_path": "s-1/loop-2"}
    assert descs[1]["metadata"] == {"scope_path": "s-1/loop-2"}


def test_a_root_ingest_leaves_no_scope_on_the_descriptor() -> None:
    descs: list[dict[str, Any]] = [{"metadata": {"source_name": "a"}, "scope_path": "forged"}]
    attach_scope_path(descs, None)
    assert descs == [{"metadata": {"source_name": "a"}}]


def test_a_malformed_job_path_fails_the_ingest() -> None:
    with pytest.raises(so.ScopeOverlayError):
        attach_scope_path([{}], "a//b")


# ── the artifact insert ─────────────────────────────────────────────────────


def test_the_root_insert_statement_is_the_pre_0150_one() -> None:
    sql, extra = store_mod._artifact_insert(None)
    assert (sql, extra) == (store_mod._INSERT_ARTIFACT_SQL, ())
    assert "scope_path" not in sql and "tier" not in sql


def test_the_scoped_insert_lands_proposed_with_the_tag() -> None:
    sql, extra = store_mod._artifact_insert("s-1")
    assert extra == ("s-1",)
    assert "'proposed', $11" in sql and "scope_path" in sql


class _ArtifactStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def active_artifact_for(self, *_a: Any) -> None:
        return None

    async def insert_artifact(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return str(kwargs["artifact_id"])


def _store_desc(desc: dict[str, Any]) -> dict[str, Any]:
    fake = _ArtifactStore()
    asyncio.run(ka._store_one_artifact(fake, desc, "p", {}))
    return fake.calls[0]


def test_a_scoped_descriptor_is_inserted_with_its_scope() -> None:
    desc = {"artifact_id": "a1", "content_hash": "h", "scope_path": "s-1/loop-2"}
    assert _store_desc(desc)["scope_path"] == "s-1/loop-2"


def test_a_root_descriptor_is_inserted_exactly_as_before() -> None:
    call = _store_desc({"artifact_id": "a1", "content_hash": "h"})
    assert "scope_path" not in call


# ── understand: chain, tier, tagged write ───────────────────────────────────


def _node(node_id: str = "n-1") -> KnowledgeNode:
    return KnowledgeNode(node_id=node_id, project_id="p", label="x", node_type="concept")


def _edge(edge_id: str = "e-1") -> KnowledgeEdge:
    return KnowledgeEdge(
        edge_id=edge_id, project_id="p", source_id="n-1", target_id="n-2", relation_type="r"
    )


def _job(spec: dict[str, Any]) -> Any:
    return SimpleNamespace(job_id="j", spec=spec, user_id="u", priority=0)


def test_a_scoped_workflow_output_lands_proposed() -> None:
    """A scoped run's output ingest has no agent origin; still proposed."""
    nodes, edges = ka._tiered_rows([_node()], [_edge()], _job({"scope_path": "s-1"}), "")
    assert (nodes[0].status, edges[0].status) == ("proposed", "proposed")
    assert nodes[0].content["origin_actor"] == "agent"


def test_a_root_workflow_output_is_unchanged() -> None:
    nodes, edges = ka._tiered_rows([_node()], [_edge()], _job({}), "")
    assert (nodes[0].status, edges[0].status) == ("established", "established")


def test_the_job_scope_is_read_from_the_spec() -> None:
    assert ka._job_scope(_job({"scope_path": "s-1"}), "p") == ("p", "s-1")
    assert ka._job_scope(_job({}), "p") is None


def test_the_understand_chain_carries_the_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    submitted: list[Any] = []
    service = SimpleNamespace(submit_job=lambda s: submitted.append(s) or "us-1")
    monkeypatch.setattr("umrs.jobs.service.get_job_service", lambda: service)
    srcs = [{"artifact_id": "a"}]
    ka._enqueue_understand_job("p", "u", _job({"scope_path": "s-1/loop-2"}), srcs)
    ka._enqueue_understand_job("p", "u", _job({}), srcs)
    assert submitted[0].spec["scope_path"] == "s-1/loop-2"
    assert "scope_path" not in submitted[1].spec


class _Conn:
    def __init__(self) -> None:
        self.in_tx = False
        self.events: list[str] = []

    @contextlib.asynccontextmanager
    async def transaction(self) -> Any:
        self.in_tx = True
        yield self
        self.events.append("commit")
        self.in_tx = False


class _GraphStore:
    def __init__(self) -> None:
        self.conn = _Conn()
        self.writes: list[tuple[str, Any]] = []
        self._pool = self

    @contextlib.asynccontextmanager
    async def acquire(self) -> Any:
        yield self.conn

    async def upsert_node(self, node: Any, conn: Any = None) -> None:
        self.writes.append((node.node_id, conn))

    async def upsert_edge(self, edge: Any, conn: Any = None) -> None:
        self.writes.append((edge.edge_id, conn))


def _write(store: _GraphStore, scope: tuple[str, str] | None) -> bool:
    result = ka.KnowledgeUnderstandResult()
    return asyncio.run(ka._write_rows(store, [_node()], [_edge()], "a", result, scope))


def test_a_scoped_pass_writes_and_tags_in_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _GraphStore()
    tagged: list[tuple[Any, ...]] = []

    async def _tag(db: Any, *args: Any) -> tuple[int, int]:
        assert db is store.conn and db.in_tx, "tags must join the write's transaction"
        tagged.append(args)
        return 1, 1

    monkeypatch.setattr(tag, "tag_graph_rows", _tag)
    assert _write(store, ("p", "s-1"))
    assert store.writes == [("n-1", store.conn), ("e-1", store.conn)]
    assert tagged == [("p", ["n-1"], ["e-1"], "s-1")]
    assert store.conn.events == ["commit"]


def test_a_root_pass_writes_as_before(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _GraphStore()

    async def _never(*_a: Any) -> tuple[int, int]:
        raise AssertionError("a root pass tags nothing")

    monkeypatch.setattr(tag, "tag_graph_rows", _never)
    assert _write(store, None)
    assert store.writes == [("n-1", None), ("e-1", None)]


def test_a_failed_tag_fails_the_artifact_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom(*_a: Any) -> tuple[int, int]:
        raise RuntimeError("tag down")

    monkeypatch.setattr(tag, "tag_graph_rows", _boom)
    result = ka.KnowledgeUnderstandResult()
    ok = asyncio.run(ka._write_rows(_GraphStore(), [_node()], [], "a", result, ("p", "s")))
    assert not ok and "tag down" in result.failed_items[0]["error"]


# ── a client may not name the tag ───────────────────────────────────────────


def test_a_client_built_spec_naming_scope_path_is_refused() -> None:
    refusal = client_spec_refusal({"sources": [], "scope_path": "s-1"})
    assert refusal is not None and "scope_path" in refusal
    assert client_spec_refusal({"sources": []}) is None


# ── task s-w3a-01: scope_lineage, parent_scope ──────────────────────────────


@pytest.mark.parametrize(
    ("path", "want"),
    [
        ("a", ("a",)),
        ("a/b/c", ("a", "a/b", "a/b/c")),
        (" s-1/loop-2 ", ("s-1", "s-1/loop-2")),
        ("", ()),
        ("   ", ()),
    ],
)
def test_the_lineage_is_root_most_first(path: str, want: tuple) -> None:
    assert so.scope_lineage(path) == want


def test_a_lineage_of_a_malformed_path_raises() -> None:
    with pytest.raises(so.ScopeOverlayError):
        so.scope_lineage("a//b")


def test_the_deepest_lineage_has_eight_entries() -> None:
    path = "/".join(f"l{i}" for i in range(8))
    lineage = so.scope_lineage(path)
    assert len(lineage) == 8 and lineage[-1] == path and lineage[0] == "l0"


@pytest.mark.parametrize(
    ("path", "want"), [("a/b/c", "a/b"), ("a/b", "a"), ("a", None), (" a/b ", "a")]
)
def test_the_parent_scope(path: str, want: str | None) -> None:
    assert so.parent_scope(path) == want


@pytest.mark.parametrize("path", ["", "  ", "a//b"])
def test_the_root_and_a_malformed_path_have_no_parent(path: str) -> None:
    with pytest.raises(so.ScopeOverlayError):
        so.parent_scope(path)


# ── task s-w3a-02: scope_tagging ────────────────────────────────────────────


class _Recorder:
    def __init__(self, *statuses: str) -> None:
        self.statuses = list(statuses)
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.executed.append((sql, args))
        return self.statuses.pop(0)


@pytest.mark.parametrize(
    ("status", "want"),
    [
        ("UPDATE 3", 3),
        ("UPDATE 0", 0),
        (" UPDATE 12 ", 12),
        ("UPDATE", 0),
        ("UPDATE x", 0),
        ("UPDATE -1", 0),
        ("UPDATE \u00b2", 0),  # isdigit() but not int(): no crash
        ("UPDATE \u0663", 0),  # a non-ASCII decimal digit
        ("", 0),
        (None, 0),
    ],
)
def test_the_command_status_count(status: Any, want: int) -> None:
    assert tag.updated_count(status) == want


def test_the_tag_statements_are_the_seams() -> None:
    """Only untagged proposed rows of the caller's project; fixed text."""
    assert tag.TAG_NODES_SQL == (
        "UPDATE knowledge_nodes SET scope_path = $3"
        " WHERE project_id = $1 AND node_id = ANY($2::text[])"
        " AND scope_path IS NULL AND status = 'proposed'"
    )
    assert tag.TAG_EDGES_SQL == (
        "UPDATE knowledge_edges SET scope_path = $3"
        " WHERE project_id = $1 AND edge_id = ANY($2::text[])"
        " AND scope_path IS NULL AND status = 'proposed'"
    )


def test_nodes_and_edges_are_tagged_in_order() -> None:
    db = _Recorder("UPDATE 2", "UPDATE 1")
    got = asyncio.run(tag.tag_graph_rows(db, "p", ("n1", "n2"), ["e1"], "s-1"))
    assert got == (2, 1)
    assert db.executed == [
        (tag.TAG_NODES_SQL, ("p", ["n1", "n2"], "s-1")),
        (tag.TAG_EDGES_SQL, ("p", ["e1"], "s-1")),
    ]


def test_an_empty_list_sends_no_statement() -> None:
    db = _Recorder("UPDATE 1")
    assert asyncio.run(tag.tag_graph_rows(db, "p", [], ["e1"], "s-1")) == (0, 1)
    assert [sql for sql, _ in db.executed] == [tag.TAG_EDGES_SQL]
    none = _Recorder()
    assert asyncio.run(tag.tag_graph_rows(none, "p", [], [], "s-1")) == (0, 0)
    assert none.executed == []


def test_the_root_scope_is_never_written_as_a_tag() -> None:
    db = _Recorder("UPDATE 1")
    with pytest.raises(ValueError):
        asyncio.run(tag.tag_graph_rows(db, "p", ["n1"], [], ""))
    assert db.executed == []


def test_a_database_error_propagates() -> None:
    class _Down:
        async def execute(self, *_a: Any) -> str:
            raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="down"):
        asyncio.run(tag.tag_graph_rows(_Down(), "p", ["n1"], [], "s-1"))


# ── task s-w3a-03: scope_subtree ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "want"),
    [
        ("a/b", "a/b/%"),
        ("a_b", "a\\_b/%"),
        ("50%", "50\\%/%"),
        ("x\\y", "x\\\\y/%"),
        ("a_%\\", "a\\_\\%\\\\/%"),
        (" a ", " a /%"),
    ],
)
def test_the_like_pattern_escapes(path: str, want: str) -> None:
    assert sub.subtree_like_pattern(path) == want


@pytest.mark.parametrize("path", ["", "   "])
def test_the_root_has_no_subtree_pattern(path: str) -> None:
    with pytest.raises(ValueError):
        sub.subtree_like_pattern(path)


def test_the_subtree_condition() -> None:
    assert sub.subtree_condition("scope_path", 3) == (
        "(scope_path = $3 OR scope_path LIKE $4 ESCAPE '\\')"
    )
    assert sub.subtree_condition("n.scope_path", 1) == (
        "(n.scope_path = $1 OR n.scope_path LIKE $2 ESCAPE '\\')"
    )


@pytest.mark.parametrize(
    ("column", "first"),
    [("status", 1), ("scope_path; DROP TABLE x", 1), ("scope_path", 0), ("scope_path", True)],
)
def test_the_subtree_condition_refuses(column: str, first: Any) -> None:
    with pytest.raises(ValueError):
        sub.subtree_condition(column, first)


# ── Postgres (env-gated, scratch schema) ────────────────────────────────────


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


needs_pg = pytest.mark.skipif(
    not _pg_url(), reason="needs UMRS_TEST_DB_URL or the CI job's POSTGRES_*"
)

# The pre-0150 shape of the three tables, as far as 0150 and the store's
# artifact insert read it.
_BASE_TABLES = """
CREATE TABLE knowledge_artifacts (
    artifact_id TEXT PRIMARY KEY, content_hash TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'document', project_id TEXT NOT NULL,
    source_ref TEXT NOT NULL DEFAULT '', size_bytes BIGINT DEFAULT 0,
    mime_type TEXT DEFAULT '', status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB DEFAULT '{}'::jsonb, created_at TEXT, updated_at TEXT);
CREATE TABLE knowledge_nodes (
    node_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'established');
CREATE TABLE knowledge_edges (
    edge_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'established');
"""


@pytest.fixture()
def scratch() -> Any:
    """A scratch schema with the base tables and 0150 applied twice."""
    import psycopg2

    schema = f"sw3a_{uuid.uuid4().hex[:10]}"
    conn = psycopg2.connect(_pg_url())
    try:
        with conn.cursor() as cur:
            cur.execute(f"CREATE SCHEMA {schema}; SET search_path TO {schema}")
            cur.execute(_BASE_TABLES)
            cur.execute(_MIGRATION.read_text())
            cur.execute(_MIGRATION.read_text())  # idempotent
        conn.commit()
        yield conn, schema
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        conn.commit()
        conn.close()


def _columns(conn: Any, schema: str, table: str) -> dict[str, tuple[str, str | None]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, is_nullable, column_default FROM information_schema.columns"
            " WHERE table_schema = %s AND table_name = %s",
            (schema, table),
        )
        return {name: (nullable, default) for name, nullable, default in cur.fetchall()}


@needs_pg
def test_0150_adds_the_columns_idempotently(scratch: Any) -> None:
    conn, schema = scratch
    arts = _columns(conn, schema, "knowledge_artifacts")
    assert arts["scope_path"] == ("YES", None)
    assert arts["tier"][0] == "NO" and "established" in str(arts["tier"][1])
    for table in ("knowledge_nodes", "knowledge_edges"):
        assert _columns(conn, schema, table)["scope_path"] == ("YES", None)
    with conn.cursor() as cur:
        cur.execute("SELECT indexname FROM pg_indexes WHERE schemaname = %s", (schema,))
        names = {row[0] for row in cur.fetchall()}
    assert {"idx_artifact_scope_path", "idx_node_scope_path", "idx_edge_scope_path"} <= names


@needs_pg
def test_0150_refuses_an_unknown_tier_and_an_empty_tag(scratch: Any) -> None:
    import psycopg2

    conn, _schema = scratch
    for values in ("'a1', 'p', 'maybe', NULL", "'a2', 'p', 'proposed', ''"):
        with pytest.raises(psycopg2.errors.CheckViolation), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO knowledge_artifacts (artifact_id, project_id, tier, scope_path)"
                f" VALUES ({values})"
            )
        conn.rollback()


async def _codecs(conn: Any) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename, encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )


async def _pool(schema: str) -> Any:
    import asyncpg

    return await asyncpg.create_pool(
        _pg_url(),
        min_size=1,
        max_size=2,
        init=_codecs,
        server_settings={"search_path": schema},
    )


async def _insert_both(schema: str) -> list[Any]:
    pool = await _pool(schema)
    try:
        store = store_mod.KnowledgeStore(pool)
        await store.insert_artifact("root-1", "h1", "document", "p", metadata={})
        await store.insert_artifact(
            "scoped-1", "h2", "document", "p", metadata={}, scope_path="s-1/loop-2"
        )
        return list(
            await pool.fetch(
                "SELECT artifact_id, tier, scope_path FROM knowledge_artifacts ORDER BY artifact_id"
            )
        )
    finally:
        await pool.close()


@needs_pg
def test_a_scoped_artifact_row_lands_proposed_and_tagged(scratch: Any) -> None:
    _conn, schema = scratch
    rows = asyncio.run(_insert_both(schema))
    assert [tuple(r) for r in rows] == [
        ("root-1", "established", None),
        ("scoped-1", "proposed", "s-1/loop-2"),
    ]


async def _tag_rows(schema: str) -> tuple[tuple[int, int], list[Any]]:
    pool = await _pool(schema)
    try:
        await pool.execute(
            "INSERT INTO knowledge_nodes (node_id, project_id, status) VALUES"
            " ('n-new', 'p', 'proposed'), ('n-est', 'p', 'established'),"
            " ('n-other', 'q', 'proposed')"
        )
        await pool.execute(
            "INSERT INTO knowledge_nodes (node_id, project_id, status, scope_path)"
            " VALUES ('n-sib', 'p', 'proposed', 's-2')"
        )
        await pool.execute(
            "INSERT INTO knowledge_edges (edge_id, project_id, status) VALUES"
            " ('e-new', 'p', 'proposed')"
        )
        async with pool.acquire() as conn, conn.transaction():
            counts = await tag.tag_graph_rows(
                conn, "p", ["n-new", "n-est", "n-other", "n-sib"], ["e-new"], "s-1"
            )
        rows = await pool.fetch(
            "SELECT node_id AS id, scope_path FROM knowledge_nodes"
            " UNION ALL SELECT edge_id, scope_path FROM knowledge_edges ORDER BY 1"
        )
        return counts, [tuple(r) for r in rows]
    finally:
        await pool.close()


@needs_pg
def test_only_untagged_proposed_rows_of_the_project_are_tagged(scratch: Any) -> None:
    _conn, schema = scratch
    counts, rows = asyncio.run(_tag_rows(schema))
    assert counts == (1, 1)
    assert rows == [
        ("e-new", "s-1"),
        ("n-est", None),
        ("n-new", "s-1"),
        ("n-other", None),
        ("n-sib", "s-2"),
    ]


@needs_pg
def test_the_subtree_condition_selects_the_scope_and_beneath(scratch: Any) -> None:
    conn, _schema = scratch
    paths = ["a_b", "a_b/c", "a_b/c/d", "axb", "axb/c", "a_bc", "a"]
    where = sub.subtree_condition("scope_path", 1)
    with conn.cursor() as cur:
        for i, path in enumerate(paths):
            cur.execute(
                "INSERT INTO knowledge_nodes (node_id, project_id, status, scope_path)"
                " VALUES (%s, 'p', 'proposed', %s)",
                (f"n{i}", path),
            )
        sql = where.replace("$1", "%(path)s").replace("$2", "%(pattern)s")
        cur.execute(
            f"SELECT scope_path FROM knowledge_nodes WHERE {sql} ORDER BY 1",
            {"path": "a_b", "pattern": sub.subtree_like_pattern("a_b")},
        )
        got = [row[0] for row in cur.fetchall()]
    assert got == ["a_b", "a_b/c", "a_b/c/d"]
