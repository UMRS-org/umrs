"""The embed writer writes one artifact's chunks in ONE transaction (ADR-093
D-11, wave F, 2026-10-11).

S7's first live run found 3 chunk nodes with no artifact row, no part_of
edge and no vector. Each write had autocommitted on its own pool connection:
the chunk node committed, its edge then failed the 0055 FK (no file node),
and the vector was never written. These tests drive the real
``KnowledgeStore`` over a fake asyncpg connection that keeps a transaction's
writes pending until it commits, and discards them on an exception — so a
write made outside the transaction shows up as a committed stray row.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from umrs.jobs import knowledge_adapters as ka
from umrs.jobs.knowledge_adapters import KnowledgeEmbedResult
from umrs.knowledge.store import KnowledgeStore

PROJECT = "p"
ART = "art-1"
FILE_NODE = f"node-file-{ART}"


class _FkViolation(Exception):
    sqlstate = "23503"


class _Db:
    """Committed rows, plus the switches the tests flip."""

    def __init__(self, *, artifact_row: bool = True, file_node: bool = True) -> None:
        self.nodes: dict[str, str] = {FILE_NODE: "file"} if file_node else {}
        self.edges: list[str] = []
        self.vectors: list[str] = []
        self.artifacts = {ART} if artifact_row else set()
        self.fail_edges = False
        self.unreadable_files = False
        self.transactions = 0


class _Conn:
    def __init__(self, db: _Db) -> None:
        self.db = db
        self.pending: dict[str, list] | None = None

    @contextlib.asynccontextmanager
    async def transaction(self):
        self.db.transactions += 1
        self.pending = {"nodes": [], "edges": [], "vectors": []}
        try:
            yield self
        except BaseException:
            self.pending = None
            raise
        staged, self.pending = self.pending, None
        self.db.nodes.update((nid, "chunk") for nid in staged["nodes"])
        self.db.edges += staged["edges"]
        self.db.vectors += staged["vectors"]

    def _write(self, table: str, row: str) -> None:
        if self.pending is not None:
            self.pending[table].append(row)
        elif table == "nodes":
            self.db.nodes[row] = "chunk"  # autocommit
        else:
            getattr(self.db, table).append(row)

    def _node_exists(self, node_id: str) -> bool:
        staged = self.pending["nodes"] if self.pending is not None else []
        return node_id in self.db.nodes or node_id in staged

    async def execute(self, sql: str, *args: Any) -> str:
        text = sql.lstrip()
        if text.startswith("INSERT INTO knowledge_nodes"):
            self._write("nodes", args[0])
        elif text.startswith("INSERT INTO knowledge_edges"):
            if self.db.fail_edges or not self._node_exists(args[3]):
                raise _FkViolation("knowledge_edges_target_id_fkey")
            self._write("edges", args[0])
        elif "INSERT INTO knowledge_node_embeddings" in text:
            self._write("vectors", args[0])
        return "DELETE 0" if text.startswith("DELETE") else "UPDATE 1"

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if "FROM knowledge_artifacts" in sql:
            return 1 if args[0] in self.db.artifacts else None
        if "FROM knowledge_nodes" in sql:
            if self.db.unreadable_files:
                raise ConnectionError("connection reset")
            return 1 if args[0] in self.db.nodes else None
        return True  # the search_text column probe

    async def fetch(self, sql: str, *_args: Any) -> list:
        if self.db.unreadable_files and "FROM knowledge_nodes" in sql:
            raise ConnectionError("connection reset")
        return []


class _Pool:
    def __init__(self, db: _Db) -> None:
        self.db = db

    @contextlib.asynccontextmanager
    async def acquire(self):
        yield _Conn(self.db)


def _write(db: _Db) -> KnowledgeEmbedResult:
    chunks = [{"text": t, "start": 0, "end": len(t)} for t in ("first", "second")]
    embedded = {"chunks": chunks, "vectors": [[1.0, 0.0]] * 2}
    result = KnowledgeEmbedResult()
    store = KnowledgeStore(_Pool(db))
    asyncio.run(ka._write_isolated(store, {"artifact_id": ART}, PROJECT, "", embedded, result))
    return result


def _chunks(db: _Db) -> list[str]:
    return [nid for nid, kind in db.nodes.items() if kind == "chunk"]


@pytest.fixture(autouse=True)
def _site(monkeypatch):
    monkeypatch.setenv("UMRS_KNOWLEDGE_EMBED_MODEL", "/srv/models/e5")
    monkeypatch.delenv("UMRS_KNOWLEDGE_CHUNK_SCHEME", raising=False)


def test_a_good_write_commits_nodes_edges_and_vectors_in_one_transaction():
    db = _Db()
    result = _write(db)
    assert result.failed_items == [] and result.nodes_embedded == 2
    assert _chunks(db) == [f"node-chunk-{ART}-0", f"node-chunk-{ART}-1"]
    assert len(db.edges) == 2 and len(db.vectors) == 2
    assert db.transactions == 1


def test_an_edge_failure_leaves_no_chunk_node():
    db = _Db()
    db.fail_edges = True
    result = _write(db)
    (failed,) = result.failed_items
    assert failed["artifact_id"] == ART and "knowledge_edges_target_id_fkey" in failed["error"]
    assert _chunks(db) == [] and db.edges == [] and db.vectors == []
    assert result.nodes_embedded == 0


def test_an_unreadable_file_node_lookup_writes_nothing():
    """The lookup failed and the writer fell back to its own scope: the chunk
    node committed, then its edge failed the FK (the probe of 2026-10-11)."""
    db = _Db(file_node=False)
    db.unreadable_files = True
    result = _write(db)
    (failed,) = result.failed_items
    assert "unreadable" in failed["error"]
    assert _chunks(db) == [] and db.vectors == []


def test_an_artifact_without_its_row_writes_nothing_and_is_named():
    """art-3de698e03d87, art-d7a806cc4b75, art-f262129bf8a4: rowless ids sent
    down the chain. The writer refuses them by name, before any row."""
    db = _Db(artifact_row=False)
    result = _write(db)
    (failed,) = result.failed_items
    assert "artifact row absent" in failed["error"] and ART in failed["error"]
    assert _chunks(db) == [] and db.edges == [] and db.vectors == []


def test_the_writes_join_the_callers_connection():
    """``conn=`` puts a store write on the caller's transaction."""
    db = _Db()
    conn = _Conn(db)

    async def _in_rolled_back_transaction() -> None:
        from umrs.knowledge.store import KnowledgeNode

        store = KnowledgeStore(_Pool(db))
        with contextlib.suppress(RuntimeError):
            async with conn.transaction():
                await store.upsert_node(
                    KnowledgeNode(node_id="n-1", project_id=PROJECT, label="x", node_type="entity"),
                    conn=conn,
                )
                raise RuntimeError("roll back")

    asyncio.run(_in_rolled_back_transaction())
    assert "n-1" not in db.nodes
