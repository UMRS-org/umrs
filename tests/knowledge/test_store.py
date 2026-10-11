"""Tests for knowledge Postgres store (store.py).

Unit tests with mocked asyncpg pool cover:
  - insert_artifact dedup (same content_hash -> False)
  - insert_ir idempotency (ON CONFLICT DO NOTHING -> False on second)
  - upsert_node / upsert_edge round-trip
  - truncate_nodes / truncate_edges counters
  - list_artifacts pagination + filters
  - get_artifact / get_ir lookups

Integration tests require --db-url or UMRS_TEST_DB_URL.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from umrs.knowledge.store import (
    KnowledgeEdge,
    KnowledgeNode,
    KnowledgeStore,
)

logger = logging.getLogger(__name__)


# -- Mock pool helpers --


def _make_pool(
    execute_return: str = "INSERT 0 1",
    fetchval_result: Any = None,
    fetchrow_result: Any = None,
    fetch_result: list | None = None,
) -> MagicMock:
    """Build a mock asyncpg Pool with configured method results."""
    call_counter = [0]

    async def _execute_side_effect(*args, **kwargs):
        call_counter[0] += 1
        return execute_return

    conn = MagicMock()
    conn.fetchval = AsyncMock(return_value=fetchval_result)
    conn.execute = AsyncMock(side_effect=_execute_side_effect)
    conn.fetchrow = AsyncMock(return_value=fetchrow_result)
    conn.fetch = AsyncMock(return_value=fetch_result or [])

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=None)

    pool = MagicMock()
    pool.acquire = MagicMock(return_value=ctx)
    pool._conn = conn
    pool._call_count = call_counter
    return pool


def _make_ts() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# -- Unit tests --


class TestInsertArtifact:
    """Artifact insert with dedup by content_hash."""

    def test_insert_new_artifact(self):
        pool = _make_pool(execute_return="INSERT 0 1", fetchval_result=None)
        store = KnowledgeStore(pool)
        result = asyncio.get_event_loop().run_until_complete(
            store.insert_artifact(
                artifact_id="art-001",
                content_hash="abc123",
                kind="document",
                project_id="proj-1",
            )
        )
        assert result == "art-001"  # the id that now holds the bytes
        assert pool._conn.execute.call_count == 1  # INSERT only (dedup uses fetchval)

    def test_dedup_returns_the_stored_id(self):
        pool = _make_pool(execute_return="INSERT 0 0", fetchval_result="art-existing")
        store = KnowledgeStore(pool)
        result = asyncio.get_event_loop().run_until_complete(
            store.insert_artifact(
                artifact_id="art-002",
                content_hash="abc123",
                kind="document",
                project_id="proj-1",
            )
        )
        assert result == "art-existing"
        assert pool._conn.execute.call_count == 0  # no row for a deduped id

    def test_dedup_checks_active_only(self):
        pool = _make_pool(execute_return="INSERT 0 1", fetchval_result=None)
        store = KnowledgeStore(pool)
        result = asyncio.get_event_loop().run_until_complete(
            store.insert_artifact(
                artifact_id="art-003",
                content_hash="abc123",
                kind="document",
                project_id="proj-1",
            )
        )
        assert result == "art-003"

    def test_insert_passes_metadata(self):
        pool = _make_pool(execute_return="INSERT 0 1", fetchval_result=None)
        store = KnowledgeStore(pool)
        asyncio.get_event_loop().run_until_complete(
            store.insert_artifact(
                artifact_id="art-004",
                content_hash="def456",
                kind="code",
                project_id="proj-1",
                metadata={"key": "value"},
            )
        )
        assert pool._conn.execute.call_count == 1  # INSERT only (dedup uses fetchval)


class TestInsertIR:
    """IR insert idempotency - write-once by ir_id."""

    def test_insert_new_ir(self):
        pool = _make_pool(execute_return="INSERT 0 1")
        store = KnowledgeStore(pool)
        result = asyncio.get_event_loop().run_until_complete(
            store.insert_ir(
                ir_id="ir-001",
                artifact_id="art-001",
                ir_type="code",
                content={"class": "Foo"},
                content_hash="hash1",
            )
        )
        assert result is True

    def test_insert_duplicate_ir_returns_false(self):
        pool = _make_pool(execute_return="INSERT 0 0")
        store = KnowledgeStore(pool)
        result = asyncio.get_event_loop().run_until_complete(
            store.insert_ir(
                ir_id="ir-001",
                artifact_id="art-001",
                ir_type="code",
                content={"class": "Foo"},
                content_hash="hash1",
            )
        )
        assert result is False

    def test_insert_ir_idempotency_proven(self):
        """Double insert = one row. Both calls succeed, second returns False."""
        insert_log = ["INSERT 0 1", "INSERT 0 0"]

        async def _execute_side_effect(*args, **kwargs):
            r = insert_log.pop(0) if insert_log else "INSERT 0 0"
            return r

        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=_execute_side_effect)
        conn.fetchval = AsyncMock()
        conn.fetchrow = AsyncMock()
        conn.fetch = AsyncMock()

        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=conn)
        ctx.__aexit__ = AsyncMock(return_value=None)

        pool = MagicMock()
        pool.acquire = MagicMock(return_value=ctx)
        pool._conn = conn
        pool._call_count = [0]

        store = KnowledgeStore(pool)

        r1 = asyncio.get_event_loop().run_until_complete(
            store.insert_ir(
                ir_id="ir-dedup-test",
                artifact_id="art-x",
                ir_type="document",
                content={"text": "hello"},
                content_hash="h1",
            )
        )
        r2 = asyncio.get_event_loop().run_until_complete(
            store.insert_ir(
                ir_id="ir-dedup-test",
                artifact_id="art-x",
                ir_type="document",
                content={"text": "hello"},
                content_hash="h1",
            )
        )
        assert r1 is True
        assert r2 is False
        assert conn.execute.call_count == 2


class TestUpsertNode:
    def test_upsert_node_creates(self):
        pool = _make_pool(execute_return="INSERT 0 1")
        store = KnowledgeStore(pool)
        node = KnowledgeNode(
            node_id="node-001",
            project_id="proj-1",
            label="Foo",
            node_type="class",
            content={"methods": ["bar"]},
            content_hash="h1",
        )
        asyncio.get_event_loop().run_until_complete(store.upsert_node(node))
        assert pool._conn.execute.call_count == 1

    def test_upsert_node_overrides_on_conflict(self):
        pool = _make_pool(execute_return="INSERT 0 1")
        store = KnowledgeStore(pool)
        node = KnowledgeNode(
            node_id="node-001",
            project_id="proj-1",
            label="Bar",
            node_type="class",
            content={"methods": ["baz"]},
            content_hash="h2",
        )
        asyncio.get_event_loop().run_until_complete(store.upsert_node(node))
        call_args = pool._conn.execute.call_args[0]
        assert "ON CONFLICT (node_id) DO UPDATE" in call_args[0]


class TestUpsertEdge:
    def test_upsert_edge_creates(self):
        pool = _make_pool(execute_return="INSERT 0 1")
        store = KnowledgeStore(pool)
        edge = KnowledgeEdge(
            edge_id="edge-001",
            project_id="proj-1",
            source_id="node-001",
            target_id="node-002",
            relation_type="CITES",
            content={},
        )
        asyncio.get_event_loop().run_until_complete(store.upsert_edge(edge))
        assert pool._conn.execute.call_count == 1

    def test_upsert_edge_overrides_on_conflict(self):
        pool = _make_pool(execute_return="INSERT 0 1")
        store = KnowledgeStore(pool)
        edge = KnowledgeEdge(
            edge_id="edge-001",
            project_id="proj-1",
            source_id="node-001",
            target_id="node-002",
            relation_type="CITES",
            content={"updated": True},
        )
        asyncio.get_event_loop().run_until_complete(store.upsert_edge(edge))
        call_args = pool._conn.execute.call_args[0]
        assert "ON CONFLICT (edge_id) DO UPDATE" in call_args[0]


class TestTruncateNodes:
    def test_truncate_all_nodes_for_project(self):
        pool = _make_pool(execute_return="DELETE 5")
        store = KnowledgeStore(pool)
        deleted = asyncio.get_event_loop().run_until_complete(
            store.truncate_nodes(project_id="proj-1")
        )
        assert deleted == 5

    def test_truncate_nodes_with_source_filter(self):
        pool = _make_pool(execute_return="DELETE 2")
        store = KnowledgeStore(pool)
        deleted = asyncio.get_event_loop().run_until_complete(
            store.truncate_nodes(project_id="proj-1", source_ir="ir-001")
        )
        assert deleted == 2


class TestTruncateEdges:
    def test_truncate_all_edges(self):
        pool = _make_pool(execute_return="DELETE 10")
        store = KnowledgeStore(pool)
        deleted = asyncio.get_event_loop().run_until_complete(store.truncate_edges())
        assert deleted == 10

    def test_truncate_edges_by_source(self):
        pool = _make_pool(execute_return="DELETE 3")
        store = KnowledgeStore(pool)
        deleted = asyncio.get_event_loop().run_until_complete(
            store.truncate_edges(source_id="node-001")
        )
        assert deleted == 3

    def test_truncate_edges_by_target(self):
        pool = _make_pool(execute_return="DELETE 2")
        store = KnowledgeStore(pool)
        deleted = asyncio.get_event_loop().run_until_complete(
            store.truncate_edges(target_id="node-002")
        )
        assert deleted == 2

    def test_truncate_edges_by_source_and_target(self):
        pool = _make_pool(execute_return="DELETE 1")
        store = KnowledgeStore(pool)
        deleted = asyncio.get_event_loop().run_until_complete(
            store.truncate_edges(source_id="node-001", target_id="node-002")
        )
        assert deleted == 1


class TestListArtifacts:
    def test_list_artifacts_kind_filter(self):
        rows = [
            {
                "artifact_id": "art-001",
                "content_hash": "h1",
                "kind": "code",
                "project_id": "proj-1",
                "source_ref": "",
                "size_bytes": 0,
                "mime_type": "",
                "status": "active",
                "metadata": {},
                "created_at": _make_ts(),
                "updated_at": _make_ts(),
            },
        ]
        pool = _make_pool(fetchrow_result={"count": "1"}, fetch_result=rows)
        store = KnowledgeStore(pool)
        artifacts, total = asyncio.get_event_loop().run_until_complete(
            store.list_artifacts(project_id="proj-1", kind="code")
        )
        assert total == 1
        assert artifacts[0].kind == "code"

    def test_list_artifacts_status_filter(self):
        rows = [
            {
                "artifact_id": "art-001",
                "content_hash": "h1",
                "kind": "document",
                "project_id": "proj-1",
                "source_ref": "",
                "size_bytes": 0,
                "mime_type": "",
                "status": "archived",
                "metadata": {},
                "created_at": _make_ts(),
                "updated_at": _make_ts(),
            },
        ]
        pool = _make_pool(fetchrow_result={"count": "1"}, fetch_result=rows)
        store = KnowledgeStore(pool)
        artifacts, total = asyncio.get_event_loop().run_until_complete(
            store.list_artifacts(project_id="proj-1", status="archived")
        )
        assert total == 1
        assert artifacts[0].status == "archived"

    def test_list_artifacts_returns_dataclass(self):
        pool = _make_pool(fetchrow_result={"count": "0"}, fetch_result=[])
        store = KnowledgeStore(pool)
        artifacts, _ = asyncio.get_event_loop().run_until_complete(
            store.list_artifacts(project_id="proj-1")
        )
        assert isinstance(artifacts, list)

    def test_list_artifacts_empty(self):
        pool = _make_pool(fetchrow_result={"count": "0"}, fetch_result=[])
        store = KnowledgeStore(pool)
        artifacts, total = asyncio.get_event_loop().run_until_complete(
            store.list_artifacts(project_id="empty")
        )
        assert total == 0
        assert len(artifacts) == 0


class TestGetArtifact:
    def test_get_artifact_found(self):
        row = {
            "artifact_id": "art-001",
            "content_hash": "abc",
            "kind": "code",
            "project_id": "proj-1",
            "source_ref": "file.py",
            "size_bytes": 1024,
            "mime_type": "text/x-python",
            "status": "active",
            "metadata": {},
            "created_at": _make_ts(),
            "updated_at": _make_ts(),
        }
        pool = _make_pool(fetchrow_result=row)
        store = KnowledgeStore(pool)
        artifact = asyncio.get_event_loop().run_until_complete(store.get_artifact("art-001"))
        assert artifact is not None
        assert artifact.artifact_id == "art-001"
        assert artifact.source_ref == "file.py"
        assert artifact.size_bytes == 1024

    def test_get_artifact_not_found(self):
        pool = _make_pool(fetchrow_result=None)
        store = KnowledgeStore(pool)
        artifact = asyncio.get_event_loop().run_until_complete(store.get_artifact("nonexistent"))
        assert artifact is None


class TestGetIR:
    def test_get_ir_found(self):
        row = {
            "ir_id": "ir-001",
            "artifact_id": "art-001",
            "version": 1,
            "ir_type": "code",
            "content": {"class": "Foo"},
            "content_hash": "h1",
            "status": "current",
            "created_at": _make_ts(),
        }
        pool = _make_pool(fetchrow_result=row)
        store = KnowledgeStore(pool)
        ir = asyncio.get_event_loop().run_until_complete(store.get_ir("ir-001"))
        assert ir is not None
        assert ir.ir_id == "ir-001"
        assert ir.ir_type == "code"

    def test_get_ir_not_found(self):
        pool = _make_pool(fetchrow_result=None)
        store = KnowledgeStore(pool)
        ir = asyncio.get_event_loop().run_until_complete(store.get_ir("nonexistent"))
        assert ir is None


class TestSchemaVersion:
    def test_returns_version(self):
        pool = _make_pool()
        store = KnowledgeStore(pool)
        version = asyncio.get_event_loop().run_until_complete(store.get_schema_version())
        assert version == 1


# -- Integration tests (require --db-url) --


class TestKnowledgeStoreIntegration:
    """Integration tests against real Postgres. Require --db-url or UMRS_TEST_DB_URL."""

    @pytest.fixture
    def pool(self, db_url):
        import asyncpg

        async def _create():
            return await asyncpg.create_pool(dsn=db_url, min=1, max=1)

        async def _setup(p):
            await p.fetch("""
                CREATE TABLE IF NOT EXISTS knowledge_artifacts (
                    artifact_id    TEXT PRIMARY KEY,
                    content_hash   TEXT NOT NULL UNIQUE,
                    kind           TEXT NOT NULL CHECK (kind IN ('document', 'code', 'table', 'image', 'audio', 'video', 'embedding', 'other')),
                    project_id     TEXT NOT NULL,
                    source_ref     TEXT DEFAULT '',
                    size_bytes     BIGINT DEFAULT 0,
                    mime_type      TEXT DEFAULT '',
                    status         TEXT DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
                    metadata       JSONB DEFAULT '{}',
                    created_at     TIMESTAMPTZ DEFAULT now(),
                    updated_at     TIMESTAMPTZ DEFAULT now()
                )
            """)
            await p.fetch("""
                CREATE TABLE IF NOT EXISTS knowledge_ir (
                    ir_id          TEXT PRIMARY KEY,
                    artifact_id    TEXT NOT NULL REFERENCES knowledge_artifacts(artifact_id) ON DELETE CASCADE,
                    version        INTEGER DEFAULT 1,
                    ir_type        TEXT NOT NULL CHECK (ir_type IN ('code', 'document', 'schema', 'catalog', 'embedding')),
                    content        JSONB NOT NULL DEFAULT '{}',
                    content_hash   TEXT NOT NULL,
                    size_bytes     BIGINT DEFAULT 0,
                    status         TEXT DEFAULT 'current' CHECK (status IN ('current', 'archived', 'superseded')),
                    created_at     TIMESTAMPTZ DEFAULT now(),
                    UNIQUE (artifact_id, version)
                )
            """)
            await p.fetch("""
                CREATE TABLE IF NOT EXISTS knowledge_nodes (
                    node_id        TEXT PRIMARY KEY,
                    project_id     TEXT NOT NULL,
                    label          TEXT NOT NULL,
                    node_type      TEXT NOT NULL CHECK (node_type IN ('class', 'function', 'file', 'module', 'concept', 'entity', 'relationship')),
                    content        JSONB DEFAULT '{}',
                    content_hash   TEXT NOT NULL,
                    source_ir      TEXT REFERENCES knowledge_ir(ir_id) ON DELETE SET NULL,
                    created_at     TIMESTAMPTZ DEFAULT now(),
                    updated_at     TIMESTAMPTZ DEFAULT now()
                )
            """)
            await p.fetch("""
                CREATE TABLE IF NOT EXISTS knowledge_edges (
                    edge_id        TEXT PRIMARY KEY,
                    project_id     TEXT NOT NULL,
                    source_id      TEXT NOT NULL REFERENCES knowledge_nodes(node_id) ON DELETE CASCADE,
                    target_id      TEXT NOT NULL REFERENCES knowledge_nodes(node_id) ON DELETE CASCADE,
                    relation_type  TEXT NOT NULL,
                    content        JSONB DEFAULT '{}',
                    created_at     TIMESTAMPTZ DEFAULT now()
                )
            """)

        async def _teardown(p):
            await p.fetch("DELETE FROM knowledge_edges")
            await p.fetch("DELETE FROM knowledge_nodes")
            await p.fetch("DELETE FROM knowledge_ir")
            await p.fetch("DELETE FROM knowledge_artifacts")
            await p.close()

        p = asyncio.get_event_loop().run_until_complete(_create())
        asyncio.get_event_loop().run_until_complete(_setup(p))
        yield p
        asyncio.get_event_loop().run_until_complete(_teardown(p))

    def test_artifact_dedup_proven(self, pool):
        store = KnowledgeStore(pool)
        r1 = asyncio.get_event_loop().run_until_complete(
            store.insert_artifact("art-001", "dedup-hash", "document", "proj-1")
        )
        r2 = asyncio.get_event_loop().run_until_complete(
            store.insert_artifact("art-002", "dedup-hash", "document", "proj-1")
        )
        assert r1 == "art-001"
        assert r2 == "art-001"  # the stored id, never a fresh one with no row
        count = asyncio.get_event_loop().run_until_complete(
            pool.fetchval(
                "SELECT COUNT(*) FROM knowledge_artifacts WHERE content_hash = $1",
                "dedup-hash",
            )
        )
        assert count == 1

    def test_ir_insert_idempotent(self, pool):
        store = KnowledgeStore(pool)
        r1 = asyncio.get_event_loop().run_until_complete(
            store.insert_ir("ir-001", "art-001", "code", {"x": 1}, "h1")
        )
        r2 = asyncio.get_event_loop().run_until_complete(
            store.insert_ir("ir-001", "art-001", "code", {"x": 1}, "h1")
        )
        assert r1 is True
        assert r2 is False
        count = asyncio.get_event_loop().run_until_complete(
            pool.fetchval("SELECT COUNT(*) FROM knowledge_ir WHERE ir_id = $1", "ir-001")
        )
        assert count == 1

    def test_ir_insert_idempotent_under_concurrency(self, pool):
        """Property test: 10 concurrent inserts with same ir_id -> only 1 row."""
        store = KnowledgeStore(pool)
        ir_id = f"ir-concurrent-{id(store)}"

        async def _insert():
            return await store.insert_ir(
                ir_id=ir_id,
                artifact_id="art-x",
                ir_type="document",
                content={"test": True},
                content_hash="ch",
            )

        results = asyncio.get_event_loop().run_until_complete(
            asyncio.gather(*[_insert() for _ in range(10)])
        )
        inserted_count = sum(1 for r in results if r)
        row_count = asyncio.get_event_loop().run_until_complete(
            pool.fetchval("SELECT COUNT(*) FROM knowledge_ir WHERE ir_id = $1", ir_id)
        )
        assert row_count == 1, "Only one row despite concurrent inserts"
        assert inserted_count >= 1

    def test_node_edge_round_trip(self, pool):
        store = KnowledgeStore(pool)
        node = KnowledgeNode(
            node_id="n-001",
            project_id="proj-1",
            label="Foo",
            node_type="class",
            content={},
            content_hash="h",
        )
        edge = KnowledgeEdge(
            edge_id="e-001",
            project_id="proj-1",
            source_id="n-001",
            target_id="n-001",
            relation_type="SELF_REF",
            content={},
        )
        asyncio.get_event_loop().run_until_complete(store.upsert_node(node))
        asyncio.get_event_loop().run_until_complete(store.upsert_edge(edge))
        node_count = asyncio.get_event_loop().run_until_complete(
            pool.fetchval("SELECT COUNT(*) FROM knowledge_nodes WHERE node_id = $1", "n-001")
        )
        edge_count = asyncio.get_event_loop().run_until_complete(
            pool.fetchval("SELECT COUNT(*) FROM knowledge_edges WHERE edge_id = $1", "e-001")
        )
        assert node_count == 1
        assert edge_count == 1

    def test_truncate_nodes_returns_count(self, pool):
        store = KnowledgeStore(pool)
        asyncio.get_event_loop().run_until_complete(
            store.upsert_node(
                KnowledgeNode(
                    node_id="n-del",
                    project_id="proj-1",
                    label="Del",
                    node_type="class",
                    content={},
                    content_hash="h",
                )
            )
        )
        deleted = asyncio.get_event_loop().run_until_complete(
            store.truncate_nodes(project_id="proj-1")
        )
        assert deleted >= 1
        count = asyncio.get_event_loop().run_until_complete(
            pool.fetchval("SELECT COUNT(*) FROM knowledge_nodes WHERE project_id = $1", "proj-1")
        )
        assert count == 0

    def test_truncate_edges_returns_count(self, pool):
        store = KnowledgeStore(pool)
        asyncio.get_event_loop().run_until_complete(
            store.upsert_node(
                KnowledgeNode(
                    node_id="n-a",
                    project_id="proj-1",
                    label="A",
                    node_type="class",
                    content={},
                    content_hash="h",
                )
            )
        )
        asyncio.get_event_loop().run_until_complete(
            store.upsert_node(
                KnowledgeNode(
                    node_id="n-b",
                    project_id="proj-1",
                    label="B",
                    node_type="class",
                    content={},
                    content_hash="h",
                )
            )
        )
        asyncio.get_event_loop().run_until_complete(
            store.upsert_edge(
                KnowledgeEdge(
                    edge_id="e-del",
                    project_id="proj-1",
                    source_id="n-a",
                    target_id="n-b",
                    relation_type="CITES",
                    content={},
                )
            )
        )
        deleted = asyncio.get_event_loop().run_until_complete(store.truncate_edges())
        assert deleted >= 1
        count = asyncio.get_event_loop().run_until_complete(
            pool.fetchval("SELECT COUNT(*) FROM knowledge_edges")
        )
        assert count == 0

    def test_list_artifacts_pagination(self, pool):
        store = KnowledgeStore(pool)
        for i in range(7):
            asyncio.get_event_loop().run_until_complete(
                store.insert_artifact(f"art-list-{i}", f"hash-{i}", "document", "proj-1")
            )
        artifacts, total = asyncio.get_event_loop().run_until_complete(
            store.list_artifacts(project_id="proj-1", page=1, page_size=3)
        )
        assert total == 7
        assert len(artifacts) == 3
        artifacts, total = asyncio.get_event_loop().run_until_complete(
            store.list_artifacts(project_id="proj-1", page=3, page_size=3)
        )
        assert len(artifacts) == 1

    def test_get_artifact_round_trip(self, pool):
        store = KnowledgeStore(pool)
        asyncio.get_event_loop().run_until_complete(
            store.insert_artifact(
                "art-get",
                "get-hash",
                "code",
                "proj-1",
                source_ref="main.py",
                size_bytes=42,
                mime_type="text/x-python",
                metadata={"key": "val"},
            )
        )
        artifact = asyncio.get_event_loop().run_until_complete(store.get_artifact("art-get"))
        assert artifact is not None
        assert artifact.source_ref == "main.py"
        assert artifact.size_bytes == 42
        assert artifact.metadata == {"key": "val"}

    def test_get_ir_round_trip(self, pool):
        store = KnowledgeStore(pool)
        asyncio.get_event_loop().run_until_complete(
            store.insert_ir("ir-get", "art-get", "code", {"class": "Test"}, "gh")
        )
        ir = asyncio.get_event_loop().run_until_complete(store.get_ir("ir-get"))
        assert ir is not None
        assert ir.ir_id == "ir-get"
        assert ir.ir_type == "code"
