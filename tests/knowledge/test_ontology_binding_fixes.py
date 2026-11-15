"""ADR-116 fixes on the ontology read side.

* KG-02: the canonical view projected nodes by the 7-bucket ``node_type``,
  so a graph bound to a science ontology projected to zero nodes; it now
  also matches ``content.semantic_type``.
* PgGraphSink handed the psycopg2 pool itself to OntologyRegistryStore,
  whose ``cursor()`` call then raised AttributeError; it now takes a
  connection with getconn and returns it with putconn.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from umrs.knowledge.canonical_graph_view import in_memory_view
from umrs.knowledge.ontology import OntologyViolationError, builtin_ontology
from umrs.knowledge.pg_graph_sink import PgGraphSink
from umrs.knowledge.store import KnowledgeIRRow, KnowledgeStore
from umrs.shared import pg


class TestProjectionMatchesSemanticType:
    def _view(self):
        view = in_memory_view(project_id="p", graph_id="g", ontology=builtin_ontology("r4-science"))
        view.add_node(
            node_id="n-method",
            node_type="concept",
            title="DFT",
            metadata={"semantic_type": "method"},
        )
        view.add_node(
            node_id="n-dataset",
            node_type="entity",
            title="QM9",
            metadata={"semantic_type": "dataset"},
        )
        # node_type "entity" is a bucket, not an r4-science class, so only
        # the semantic type could keep this node, and it is not a class.
        view.add_node(
            node_id="n-other",
            node_type="entity",
            title="vibes",
            metadata={"semantic_type": "wormhole"},
        )
        return view

    def test_science_ontology_keeps_nodes_by_semantic_type(self) -> None:
        ids = {n.node_id for n in self._view().query()}
        assert ids == {"n-method", "n-dataset"}

    def test_type_query_matches_semantic_type(self) -> None:
        ids = {n.node_id for n in self._view().query(node_type="dataset")}
        assert ids == {"n-dataset"}


# ── PgGraphSink registry connection ──────────────────────────────────────

_REGISTERED_ROW = {
    "record_id": "onto:lab:1:abc",
    "ontology_id": "lab",
    "version": "1",
    "entity_types": [
        "class", "function", "file", "module", "concept", "entity", "relationship",
    ],
    "relation_types": ["depends_on"],
    "transitive_relations": ["depends_on"],
    "description": "",
    "owner_id": "alice",
    "project_id": "proj-a",
    "status": "active",
    "retired_at": None,
    "created_at": None,
    "updated_at": None,
}  # fmt: skip


class _Cursor:
    def __init__(self) -> None:
        self.params: tuple = ()

    def execute(self, _sql: str, params: tuple) -> None:
        self.params = params

    def fetchone(self) -> dict[str, Any] | None:
        return _REGISTERED_ROW if self.params == ("lab", "1") else None

    def fetchall(self) -> list[dict[str, Any]]:
        row = self.fetchone()
        return [row] if row is not None else []


class _Conn:
    def cursor(self) -> _Cursor:
        return _Cursor()


class _Pool:
    """psycopg2-pool shaped: no cursor(), only getconn/putconn."""

    def __init__(self) -> None:
        self.conn = _Conn()
        self.returned: list[Any] = []

    def getconn(self) -> _Conn:
        return self.conn

    def putconn(self, conn: Any) -> None:
        self.returned.append(conn)


def _ir_row() -> KnowledgeIRRow:
    return KnowledgeIRRow(
        ir_id="ir-1",
        artifact_id="art-ir-1",
        version=1,
        ir_type="code",
        content={
            "artifact_id": "art-ir-1",
            "artifact_type": "code",
            "extracted": {"filename": "ir-1.py", "citations": ["doi:x"]},
        },
        content_hash="hash-ir-1",
        status="current",
    )


def _store() -> MagicMock:
    store = MagicMock(spec=KnowledgeStore)
    store.list_ir_by_ids = AsyncMock(return_value=[_ir_row()])
    store.truncate_nodes = AsyncMock()
    store.truncate_edges = AsyncMock()
    store.upsert_node = AsyncMock()
    store.upsert_edge = AsyncMock()
    return store


def test_sink_resolves_the_registry_on_a_pooled_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = _Pool()
    monkeypatch.setattr(pg, "pg_configured_from_env", lambda: True)
    monkeypatch.setattr(pg, "get_default_sync_pool", lambda: pool)
    store = _store()
    graph = {"graph_id": "g-x", "ontology_id": "lab", "ontology_version": "1"}
    # The registered 'lab' ontology has no 'cites' relation, so the payload's
    # citation edge is refused: proof the registry row was read. The old
    # code raised AttributeError (the pool has no cursor()).
    with pytest.raises(OntologyViolationError, match="'lab'"):
        asyncio.run(
            PgGraphSink(store).generate(store, "proj-a", ["ir-1"], graph_id="g-x", graph=graph)
        )
    assert pool.returned == [pool.conn]
    store.truncate_nodes.assert_not_awaited()
