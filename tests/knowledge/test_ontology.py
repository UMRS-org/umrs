"""First-class ontology: module unit tests + PgGraphSink ontology gate.

Coverage:
  - DEFAULT_ONTOLOGY is built from the DB-enforced vocabulary (no drift)
  - ontology_for_graph resolution (default / pinned / inline / fail-loud)
  - validate_graph_payload violation reporting
  - PgGraphSink: violating payloads raise BEFORE truncation; valid payloads
    and legacy defaults behave exactly as before
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from umrs.knowledge.ontology import (
    CANONICAL_RELATION_TYPES,
    DEFAULT_ONTOLOGY,
    OntologyViolationError,
    ontology_for_graph,
    ontology_from_dict,
    validate_graph_payload,
)
from umrs.knowledge.pg_graph_sink import PgGraphSink
from umrs.knowledge.store import KnowledgeIRRow, KnowledgeStore


def _ir_row(ir_id: str, extracted: dict[str, Any]) -> KnowledgeIRRow:
    return KnowledgeIRRow(
        ir_id=ir_id,
        artifact_id=f"art-{ir_id}",
        version=1,
        ir_type="code",
        content={
            "artifact_id": f"art-{ir_id}",
            "artifact_type": "code",
            "extracted": {"filename": f"{ir_id}.py", **extracted},
        },
        content_hash=f"hash-{ir_id}",
        status="current",
    )


def _mock_store(irs: list[KnowledgeIRRow]) -> MagicMock:
    store = MagicMock(spec=KnowledgeStore)
    store.list_ir_by_ids = AsyncMock(return_value=irs)
    store.truncate_nodes = AsyncMock()
    store.truncate_edges = AsyncMock()
    store.upsert_node = AsyncMock()
    store.upsert_edge = AsyncMock()
    return store


_STRICT_ONTOLOGY_DOC = {
    "ontology_id": "proj-strict",
    "version": "1",
    "entity_types": [
        "class",
        "function",
        "file",
        "module",
        "concept",
        "entity",
        "relationship",
    ],
    "relation_types": ["depends_on", "implements"],
}


class TestDefaultsAndResolution:
    def test_default_ontology_derives_from_db_vocabulary(self):
        assert DEFAULT_ONTOLOGY.entity_types == KnowledgeStore.ALLOWED_NODE_TYPES
        assert DEFAULT_ONTOLOGY.relation_types == CANONICAL_RELATION_TYPES
        assert DEFAULT_ONTOLOGY.ontology_id == "r4-default"

    def test_absent_graph_resolves_default(self):
        assert ontology_for_graph(None) is DEFAULT_ONTOLOGY
        assert ontology_for_graph({}) is DEFAULT_ONTOLOGY
        assert ontology_for_graph({"graph_id": "g-x"}) is DEFAULT_ONTOLOGY

    def test_explicit_default_with_matching_version(self):
        graph = {"ontology_id": "r4-default", "ontology_version": "1"}
        assert ontology_for_graph(graph) is DEFAULT_ONTOLOGY

    def test_default_version_mismatch_fails_loud(self):
        graph = {"ontology_id": "r4-default", "ontology_version": "99"}
        with pytest.raises(ValueError, match="no version '99'"):
            ontology_for_graph(graph)

    def test_unknown_ontology_reference_fails_loud(self):
        graph = {"graph_id": "g-x", "ontology_id": "does-not-exist"}
        with pytest.raises(ValueError, match="unknown ontology 'does-not-exist'"):
            ontology_for_graph(graph)

    def test_inline_custom_ontology_resolves(self):
        graph = {
            "graph_id": "g-x",
            "ontology_id": "proj-strict",
            "ontology_version": "1",
            "ontology": _STRICT_ONTOLOGY_DOC,
        }
        ontology = ontology_for_graph(graph)
        assert ontology.ontology_id == "proj-strict"
        assert ontology.relation_types == frozenset({"depends_on", "implements"})

    def test_inline_via_provenance(self):
        graph = {
            "graph_id": "g-x",
            "ontology_id": "proj-strict",
            "provenance": {"ontology": _STRICT_ONTOLOGY_DOC},
        }
        assert ontology_for_graph(graph).ontology_id == "proj-strict"

    def test_inline_id_mismatch_fails_loud(self):
        graph = {
            "graph_id": "g-x",
            "ontology_id": "other",
            "ontology": _STRICT_ONTOLOGY_DOC,
        }
        with pytest.raises(ValueError, match="references ontology 'other'"):
            ontology_for_graph(graph)

    def test_serialized_roundtrip_and_strict_shape(self):
        ontology = ontology_from_dict(_STRICT_ONTOLOGY_DOC)
        assert ontology_from_dict(ontology.to_dict()) == ontology
        with pytest.raises(ValueError, match="missing"):
            ontology_from_dict({"ontology_id": "x"})


class TestPayloadValidation:
    def test_clean_payload_passes(self):
        validate_graph_payload(
            DEFAULT_ONTOLOGY,
            node_types=["file", "concept"],
            relation_types=["depends_on"],
        )

    def test_unknown_types_raise_with_details(self):
        with pytest.raises(OntologyViolationError) as excinfo:
            validate_graph_payload(
                DEFAULT_ONTOLOGY,
                node_types=["file", "wormhole"],
                relation_types=["depends_on", "enchants"],
            )
        err = excinfo.value
        assert err.ontology is DEFAULT_ONTOLOGY
        assert err.unknown_entity_types == ["wormhole"]
        assert err.unknown_relation_types == ["enchants"]
        assert "r4-default" in str(err)


class TestSinkOntologyGate:
    def _run(self, sink, store, graph=None):
        return asyncio.run(sink.generate(store, "proj-a", ["ir-1"], graph_id="g-x", graph=graph))

    def test_default_ontology_legacy_path_unchanged(self):
        irs = [_ir_row("ir-1", {"dependencies": ["numpy"], "citations": ["doi:x"]})]
        store = _mock_store(irs)
        written = self._run(PgGraphSink(store), store)
        assert written == 1
        store.truncate_nodes.assert_awaited_once()
        assert store.upsert_edge.await_count == 2  # depends_on + cites

    def test_violating_payload_blocks_before_truncate(self):
        irs = [_ir_row("ir-1", {"citations": ["doi:x"]})]
        store = _mock_store(irs)
        graph = {
            "graph_id": "g-x",
            "ontology_id": "proj-strict",
            "ontology": _STRICT_ONTOLOGY_DOC,
        }
        with pytest.raises(OntologyViolationError, match="'proj-strict'"):
            self._run(PgGraphSink(store), store, graph=graph)
        exc_store = store
        exc_store.truncate_nodes.assert_not_awaited()
        exc_store.truncate_edges.assert_not_awaited()
        exc_store.upsert_node.assert_not_awaited()

    def test_conforming_custom_ontology_writes(self):
        irs = [_ir_row("ir-1", {"dependencies": ["numpy"]})]
        store = _mock_store(irs)
        graph = {
            "graph_id": "g-x",
            "ontology_id": "proj-strict",
            "ontology": _STRICT_ONTOLOGY_DOC,
        }
        written = self._run(PgGraphSink(store), store, graph=graph)
        assert written == 1
        store.truncate_nodes.assert_awaited_once()

    def test_bad_graph_reference_fails_before_truncate(self):
        irs = [_ir_row("ir-1", {"dependencies": ["numpy"]})]
        store = _mock_store(irs)
        graph = {"graph_id": "g-x", "ontology_id": "r4-default", "ontology_version": "7"}
        with pytest.raises(ValueError, match="no version '7'"):
            self._run(PgGraphSink(store), store, graph=graph)
        store.truncate_nodes.assert_not_awaited()

    def test_empty_ir_list_preserves_legacy_truncate(self):
        store = _mock_store([])
        written = self._run(PgGraphSink(store), store)
        assert written == 0
        store.truncate_nodes.assert_awaited_once()
