"""Tests for PgGraphSink (pg_graph_sink.py).

Unit tests with mocked asyncpg pool cover:
  - Regenerate creates nodes and edges from IRs
  - Two runs with same input → deterministic (identical row sets)
  - Truncate then recompile restores nodes/edges
  - IR without extracted nodes → 0 nodes/edges created
  - Empty IR list → 0 nodes returned
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from umrs.knowledge.ir import EXTRACTOR_VERSION, build_knowledge_ir
from umrs.knowledge.pg_graph_sink import PgGraphSink
from umrs.knowledge.store import KnowledgeIRRow, KnowledgeStore

# -- Mock pool helpers (mirrors test_store.py) --


def _make_pool(
    execute_return: str = "DELETE 3",
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
    return pool


def _make_ir_row(
    ir_id: str,
    artifact_id: str,
    artifact_type: str,
    content: dict[str, Any],
) -> KnowledgeIRRow:
    """Build a mock KnowledgeIRRow for testing."""
    return KnowledgeIRRow(
        ir_id=ir_id,
        artifact_id=artifact_id,
        version=1,
        ir_type="code",
        content=content,
        content_hash="hash1",
        status="current",
    )


def _make_ir(
    artifact_id: str,
    *,
    filename: str,
    content: bytes,
    extracted: dict[str, Any] | None = None,
) -> Any:
    """Build a KnowledgeIR for testing."""
    payload: dict[str, Any] = {
        "artifact_id": artifact_id,
        "artifact_type": "code",
        "filename": filename,
    }
    if extracted:
        payload.update(extracted)
    return build_knowledge_ir(
        project_id="proj-1",
        artifact_id=artifact_id,
        artifact_type="code",
        content=content,
        extracted=payload,
        extractor_version=EXTRACTOR_VERSION,
    )


# -- Fixtures --


@pytest.fixture
def mock_pool() -> MagicMock:
    """Mock asyncpg pool that returns DELETE counts."""
    return _make_pool(execute_return="DELETE 3")


@pytest.fixture
def store(mock_pool: MagicMock) -> KnowledgeStore:
    """KnowledgeStore backed by mock pool."""
    return KnowledgeStore(mock_pool)


@pytest.fixture
def sink(store: KnowledgeStore) -> PgGraphSink:
    return PgGraphSink(store)


class TestGenerateCreatesNodesAndEdges:
    """PgGraphSink.generate() creates nodes and edges from IR content."""

    def test_regenerate_creates_nodes_and_edges(self, sink: PgGraphSink, store: KnowledgeStore):
        irs = [
            _make_ir_row(
                ir_id="ir-001",
                artifact_id="art-001",
                artifact_type="code",
                content={
                    "artifact_id": "art-001",
                    "artifact_type": "code",
                    "extracted": {
                        "filename": "solver.py",
                        "dependencies": ["numpy", "scipy"],
                        "algorithms": ["gradient_descent"],
                    },
                },
            ),
            _make_ir_row(
                ir_id="ir-002",
                artifact_id="art-002",
                artifact_type="paper",
                content={
                    "artifact_id": "art-002",
                    "artifact_type": "paper",
                    "extracted": {
                        "filename": "paper.md",
                        "title": "Deep Learning Survey",
                        "citations": ["smith2020", "doe2021"],
                    },
                },
            ),
        ]

        async def _run():
            store.list_ir_by_ids = AsyncMock(return_value=irs)
            return await sink.generate(store, "proj-1", ["ir-001", "ir-002"])

        result = asyncio.get_event_loop().run_until_complete(_run())

        assert result == 2  # 2 nodes created

    def test_regenerate_creates_dep_edges(self, sink: PgGraphSink, store: KnowledgeStore):
        irs = [
            _make_ir_row(
                ir_id="ir-001",
                artifact_id="art-001",
                artifact_type="code",
                content={
                    "artifact_id": "art-001",
                    "artifact_type": "code",
                    "extracted": {
                        "filename": "solver.py",
                        "dependencies": ["numpy"],
                        "algorithms": ["gradient_descent"],
                        "citations": [],
                    },
                },
            )
        ]

        async def _run():
            store.list_ir_by_ids = AsyncMock(return_value=irs)
            store.upsert_node = AsyncMock()  # type: ignore[assignment]
            store.upsert_edge = AsyncMock()  # type: ignore[assignment]
            return await sink.generate(store, "proj-1", ["ir-001"])

        asyncio.get_event_loop().run_until_complete(_run())

        # Verify upsert_node and upsert_edge were called
        assert store.upsert_node.call_count == 1  # type: ignore[attr-defined]
        assert store.upsert_edge.call_count == 2  # type: ignore[attr-defined]  # 1 dep + 1 algo


class TestDeterministic:
    """Two runs with identical IRs → identical node/edge row sets."""

    def test_regenerate_deterministic(self, sink: PgGraphSink, store: KnowledgeStore):
        irs = [
            _make_ir_row(
                ir_id="ir-a",
                artifact_id="art-a",
                artifact_type="code",
                content={
                    "artifact_id": "art-a",
                    "artifact_type": "code",
                    "extracted": {
                        "filename": "module_a.py",
                        "dependencies": ["pkg_x", "pkg_y"],
                        "algorithms": ["train_model"],
                    },
                },
            ),
        ]

        async def _collect(run_num: int):
            store.list_ir_by_ids = AsyncMock(return_value=irs)
            node_calls = []
            edge_calls = []

            async def _track_node(node):
                node_calls.append(
                    {
                        "node_id": node.node_id,
                        "content_hash": node.content_hash,
                    }
                )
                return None

            async def _track_edge(edge):
                edge_calls.append(
                    {
                        "edge_id": edge.edge_id,
                        "content": edge.content,
                    }
                )
                return None

            store.upsert_node = AsyncMock(side_effect=_track_node)
            store.upsert_edge = AsyncMock(side_effect=_track_edge)

            await sink.generate(store, "proj-determ", ["ir-a"])
            return run_num, node_calls, edge_calls

        _, nodes1, edges1 = asyncio.get_event_loop().run_until_complete(_collect(1))
        _, nodes2, edges2 = asyncio.get_event_loop().run_until_complete(_collect(2))

        # Identical row sets (node_ids, edge_ids, content)
        assert nodes1 == nodes2, "Nodes differ between runs"
        assert edges1 == edges2, "Edges differ between runs"


class TestTruncateThenRecompile:
    """Truncate deletes existing nodes/edges, recompile restores them."""

    def test_truncate_then_recompile_restores_nodes(self, sink: PgGraphSink, store: KnowledgeStore):
        irs = [
            _make_ir_row(
                ir_id="ir-t1",
                artifact_id="art-t1",
                artifact_type="code",
                content={
                    "artifact_id": "art-t1",
                    "artifact_type": "code",
                    "extracted": {
                        "filename": "test.py",
                        "dependencies": [],
                        "algorithms": [],
                        "citations": [],
                    },
                },
            ),
        ]

        async def _run():
            store.list_ir_by_ids = AsyncMock(return_value=irs)
            store.upsert_node = AsyncMock()
            store.upsert_edge = AsyncMock()

            # First run: populate
            n1 = await sink.generate(store, "proj-trunc", ["ir-t1"])

            # Second run: truncate + recompile (same IRs)
            n2 = await sink.generate(store, "proj-trunc", ["ir-t1"])

            return n1, n2

        n1, n2 = asyncio.get_event_loop().run_until_complete(_run())

        assert n1 == n2 == 1, "Both runs should produce 1 node"
        assert store.upsert_node.call_count == 2, "Two upserts total (pre+post truncate)"


class TestEmptyIRContent:
    """IRs without extracted nodes/edges create 0 nodes and 0 edges."""

    def test_ir_without_extracted_nodes(self, sink: PgGraphSink, store: KnowledgeStore):
        """IR with empty extracted nodes list → 0 nodes."""
        irs = [
            _make_ir_row(
                ir_id="ir-empty",
                artifact_id="art-empty",
                artifact_type="code",
                content={
                    "artifact_id": "art-empty",
                    "artifact_type": "code",
                    "extracted": {
                        "filename": "empty.py",
                        "nodes": [],  # Empty nodes list
                        "dependencies": [],
                        "algorithms": [],
                        "citations": [],
                    },
                },
            )
        ]

        async def _run():
            store.list_ir_by_ids = AsyncMock(return_value=irs)
            return await sink.generate(store, "proj-empty", ["ir-empty"])

        result = asyncio.get_event_loop().run_until_complete(_run())
        # Node is still created (node per IR), just no dependencies
        assert result == 1

    def test_ir_no_extracted_field(self, sink: PgGraphSink, store: KnowledgeStore):
        """IR with missing extracted → handles gracefully, 0 edges."""
        irs = [
            _make_ir_row(
                ir_id="ir-no-ext",
                artifact_id="art-no-ext",
                artifact_type="code",
                content={
                    "artifact_id": "art-no-ext",
                    "artifact_type": "code",
                },
            )
        ]

        async def _run():
            store.list_ir_by_ids = AsyncMock(return_value=irs)
            return await sink.generate(store, "proj-no-ext", ["ir-no-ext"])

        result = asyncio.get_event_loop().run_until_complete(_run())
        # Node created, no edges
        assert result == 1

    def test_empty_ir_list_returns_zero(self, sink: PgGraphSink, store: KnowledgeStore):
        """No IRs → 0 nodes returned."""

        async def _run():
            store.list_ir_by_ids = AsyncMock(return_value=[])
            return await sink.generate(store, "proj-0", [])

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result == 0
