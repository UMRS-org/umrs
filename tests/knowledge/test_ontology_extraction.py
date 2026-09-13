"""ADR-049 Leg D pins: ontology-driven extraction widening at the sink.

Contracts locked:

- A graph bound to an ontology declaring ``extraction_fields`` emits
  edges ADDITIVELY for those (content-key, relation) pairs — the base
  loops (dependencies/citations/algorithms) ride UNCHANGED.
- The consistency gate fails FAST (raise before truncation) when a
  declared extraction relation is outside the ontology's own
  relation_types — a mis-declared dictionary can never ride past
  an empty/no-op compile.
- Serialized round-trip dominates: to_dict without extraction_fields
  omits the key entirely (default paths byte-stable), from_dict
  tolerates absence, map-shaped payloads normalize ordered tuples.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from umrs.knowledge.ontology import Ontology, ontology_from_dict
from umrs.knowledge.pg_graph_sink import (
    PgGraphSink,
    _validate_against_ontology,
)


class _FakeIR:
    def __init__(self, ir_id: str, content: dict):
        self.ir_id = ir_id
        self.content = content


def _sink() -> PgGraphSink:
    return PgGraphSink(store=MagicMock())


def _onto(**extensions: str) -> Ontology:
    relations = {"cites", "supports", "depends_on", "implements"} | set(extensions.values())
    return Ontology(
        ontology_id="lab-extraction",
        version="1",
        entity_types=frozenset({"file", "function", "concept", "class", "module",
                                "entity", "relationship"}),
        relation_types=frozenset(relations),
        extraction_fields=tuple(sorted(extensions.items())) if extensions else (),
    )


class _CaptureStore:
    def __init__(self):
        self.nodes: list = []
        self.edges: list = []

    async def upsert_node(self, node):
        self.nodes.append(node)

    async def upsert_edge(self, edge):
        self.edges.append(edge)


_IR = _FakeIR(
    "ir-1",
    {
        "artifact_type": "code",
        "extracted": {
            "dependencies": ["libc"],
            "citations": ["paper-x"],
            "algorithms": ["fft-fast"],
            "reviewed_by": ["reviewer-9"],
            "reviewed_by_2_notes": [],
        },
    },
)


class TestLegDEmission:
    @pytest.mark.asyncio
    async def test_base_loops_unchanged_without_ontology(self) -> None:
        store = _CaptureStore()
        sink = PgGraphSink(store=store)
        await sink._upsert_edges(store, _IR, "proj", "g")  # ontology=None
        kinds = sorted(e.relation_type for e in store.edges)
        assert kinds == ["cites", "depends_on", "implements"]

    @pytest.mark.asyncio
    async def test_extension_fields_emit_additively(self) -> None:
        store = _CaptureStore()
        sink = PgGraphSink(store=store)
        onto = _onto(reviewed_by="supports")
        await sink._upsert_edges(store, _IR, "proj", "g", onto)
        rels = sorted(e.relation_type for e in store.edges)
        assert rels == ["cites", "depends_on", "implements", "supports"]
        ext = [e for e in store.edges if e.relation_type == "supports"][0]
        assert ext.target_id
        assert ext.content["extraction_field"] == "reviewed_by"
        assert ext.edge_id.endswith(":supports:reviewer-9")

    @pytest.mark.asyncio
    async def test_base_field_shadow_never_double_emits(self) -> None:
        store = _CaptureStore()
        sink = PgGraphSink(store=store)
        onto = _onto(dependencies="depends_on")  # declares what base loops own
        await sink._upsert_edges(store, _IR, "proj", "g", onto)
        deps = [e for e in store.edges if e.relation_type == "depends_on"]
        assert len(deps) == 1  # base loop owns it; extension loop skipped


class TestLegDConsistencyGate:
    def test_misdeclared_extraction_relation_fails_fast(self) -> None:
        onto = Ontology(
            ontology_id="bad-extraction",
            version="1",
            entity_types=frozenset({"concept"}),
            relation_types=frozenset({"cites"}),
            extraction_fields=(("evidence", "supports"),),
        )
        with pytest.raises(ValueError, match="mis-declares extraction"):
            _validate_against_ontology(onto, [_IR])
        # ...while a self-consistent dictionary validates clean
        _validate_against_ontology(_onto(reviewed_by="supports"), [_IR])

    def test_empty_ir_list_never_validates(self) -> None:
        bad = Ontology(
            ontology_id="b1", version="1",
            entity_types=frozenset({"concept"}), relation_types=frozenset({"cites"}),
            extraction_fields=(("evidence", "supports"),),
        )
        with pytest.raises(ValueError):
            _validate_against_ontology(bad, [_IR])


class TestLegDSerialization:
    def test_roundtrip_normalizes_pairs(self) -> None:
        onto = _onto(reviewed_by="supports", derived_from="depends_on")
        back = ontology_from_dict(onto.to_dict())
        assert back.extraction_fields == tuple(
            sorted(
                (("derived_from", "depends_on"), ("reviewed_by", "supports"))
            )
        )

    def test_payload_without_fields_maps_to_empty(self) -> None:
        back = ontology_from_dict(
            {
                "ontology_id": "x",
                "version": "1",
                "entity_types": ["concept"],
                "relation_types": ["cites"],
            }
        )
        assert back.extraction_fields == ()

    def test_to_dict_omits_field_when_empty(self) -> None:
        payload = _onto().to_dict()
        assert "extraction_fields" not in payload
        assert "transitive_relations" in payload
