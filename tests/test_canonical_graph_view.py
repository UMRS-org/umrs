"""CanonicalGraphView — protocol semantics + canonical write-through (R5).

Mirror parity with InMemoryKnowledgeGraph (dedupe, direction-agnostic
reads, BFS) and durable routing into the canonical KnowledgeStore.
"""

from __future__ import annotations

from umrs.knowledge.canonical_graph_view import CanonicalGraphView
from umrs.knowledge.ontology import canonical_relation
from umrs.knowledge_graph_api import RelationType


class FakeStore:
    """Minimal KnowledgeStore stand-in capturing writes."""

    def __init__(self):
        self.nodes: dict = {}
        self.edges: dict = {}

    async def upsert_node(self, node):
        self.nodes[node.node_id] = node

    async def upsert_edge(self, edge):
        self.edges[edge.edge_id] = edge

    async def list_nodes(self, project_id, graph_id=None):  # noqa: ARG002
        return list(self.nodes.values())

    async def list_edges(self, project_id, graph_id=None):  # noqa: ARG002
        return list(self.edges.values())


def _new_view(store=None, project="proj-a", graph="g-x"):
    return CanonicalGraphView(store=store, project_id=project, graph_id=graph)


def test_add_node_and_update_semantics():
    v = _new_view()
    v.add_node(node_id="n1", node_type="file", title="T1")
    v.add_node(node_id="n1", node_type="file", title="T2", metadata={"a": 1})
    node = v.get_node("n1")
    assert node.title == "T2"
    assert node.metadata == {"a": 1}


def test_add_edge_dedep_refreshes_weight_confidence():
    v = _new_view()
    e1 = v.add_edge(
        source_id="a",
        target_id="b",
        relation_type=RelationType.SUPPORTS,
        weight=0.4,
        confidence=0.5,
    )
    e2 = v.add_edge(
        source_id="a",
        target_id="b",
        relation_type=RelationType.SUPPORTS,
        weight=0.9,
        confidence=0.8,
    )
    assert e1 == e2
    assert len(v._edges) == 1
    edge = next(iter(v._edges.values()))
    assert (edge.weight, edge.confidence) == (0.9, 0.8)


def test_edge_autocreates_unknown_nodes():
    v = _new_view()
    v.add_edge(source_id="x", target_id="y", relation_type=RelationType.CITES)
    assert v.get_node("x") is not None and v.get_node("y") is not None
    assert v.get_node("x").node_type == "unknown"


def test_neighbors_and_related_are_direction_agnostic():
    v = _new_view()
    v.add_edge(source_id="a", target_id="b", relation_type=RelationType.SUPPORTS)
    v.add_edge(source_id="c", target_id="a", relation_type=RelationType.SUPPORTS)
    assert set(v.get_neighbors("a")) == {"b", "c"}
    assert set(v.get_related("a", RelationType.SUPPORTS)) == {"b", "c"}


def test_find_path_undirected_bfs():
    v = _new_view()
    v.add_edge(source_id="a", target_id="b", relation_type=RelationType.USES)
    v.add_edge(source_id="b", target_id="c", relation_type=RelationType.USES)
    assert v.find_path("a", "c") == ["a", "b", "c"]
    assert v.find_path("c", "a") == ["c", "b", "a"]
    assert v.find_path("a", "z") is None
    assert v.find_path("a", "a") == ["a"]


def test_query_type_and_relation_filters():
    v = _new_view()
    v.add_node(node_id="n1", node_type="file")
    v.add_node(node_id="n2", node_type="concept")
    v.add_edge(source_id="n1", target_id="n2", relation_type=RelationType.USES)
    assert {n.node_id for n in v.query(node_type="file")} == {"n1"}
    assert {n.node_id for n in v.query(relation_type=RelationType.USES)} == {"n1", "n2"}
    assert {n.node_id for n in v.query()} == {"n1", "n2"}


def test_write_through_to_canonical_store_mapping():
    """SAT relation projects onto canonical vocabulary with provenance."""
    store = FakeStore()
    v = _new_view(store=store)
    v.add_node(node_id="find-1", node_type="concept", title="F1")
    v.add_edge(
        source_id="find-1",
        target_id="hyp-1",
        relation_type=RelationType.SUPPORTS,
        confidence=0.7,
        weight=0.9,
    )
    row = store.edges[next(iter(store.edges))]
    _assert_row_scope(row)
    _assert_row_provenance(row)


def _assert_row_scope(row) -> None:
    assert row.relation_type == "supports"
    assert row.project_id == "proj-a"
    assert row.graph_id == "g-x"
    assert (row.source_id, row.target_id) == ("find-1", "hyp-1")


def _assert_row_provenance(row) -> None:
    assert row.content["sat_relation"] == "SUPPORTS"
    assert row.content["sat_inverse"] is False
    assert row.content["confidence"] == 0.7
    assert row.content["weight"] == 0.9


def test_inverse_relation_flips_storage_direction_only():
    store = FakeStore()
    v = _new_view(store=store)
    v.add_edge(source_id="a", target_id="b", relation_type=RelationType.ANSWERED_BY)
    row = next(iter(store.edges.values()))
    # canonical stores inverse direction; provenance keeps the original act
    assert (row.source_id, row.target_id) == ("b", "a")
    assert row.content["sat_relation"] == "ANSWERED_BY"
    assert row.content["sat_inverse"] is True


def test_node_type_falls_back_to_allowed_bucket():
    store = FakeStore()
    v = _new_view(store=store)
    v.add_node(node_id="n1", node_type="scientific_object")
    row = store.nodes["n1"]
    assert row.node_type == "entity"  # DB CHECK-bucket fallback


def test_refresh_rebuilds_mirror_with_provenance_edges():
    store = FakeStore()
    v = _new_view(store=store)
    v.add_edge(source_id="a", target_id="b", relation_type=RelationType.ANSWERED_BY)
    v2 = _new_view(store=store)

    import asyncio

    asyncio.run(v2.refresh())
    edge = next(iter(v2._edges.values()))
    # Provenance markers restore original orientation on reload
    assert (edge.source_id, edge.target_id) == ("a", "b")
    assert edge.relation_type == RelationType.ANSWERED_BY


def test_canonical_relation_fallback_for_unknown_name():
    rel, inverse = canonical_relation("NO_SUCH_RELATION_XYZ")
    assert (rel, inverse) == ("related_to", False)
