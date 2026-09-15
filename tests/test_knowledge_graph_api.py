"""Tests for the knowledge_graph_api subsystem (Phase 1, Step 2.3)."""

from __future__ import annotations

import pytest

from umrs.knowledge_graph_api import (
    InMemoryKnowledgeGraph,
    KnowledgeEdge,
    KnowledgeGraphAPI,
    KnowledgeNode,
    RelationType,
)


# ── Enum Tests ────────────────────────────────────────────────────────────


def test_relation_type_enum():
    """Test RelationType enum values."""
    assert RelationType.CITES == "cites"
    assert RelationType.REFERENCES == "references"
    assert RelationType.REFERENCED_BY == "referenced_by"
    assert RelationType.DERIVES_FROM == "derives_from"
    assert RelationType.PRODUCES == "produces"
    assert RelationType.PRODUCED_BY == "produced_by"
    assert RelationType.SUPPORTS == "supports"
    assert RelationType.CONTRADICTS == "contradicts"
    assert RelationType.EVIDENCE_FOR == "evidence_for"
    assert RelationType.EVIDENCE_AGAINST == "evidence_against"
    assert RelationType.IMPLEMENTS == "implements"
    assert RelationType.IMPLEMENTED_BY == "implemented_by"
    assert RelationType.DESCRIBES == "describes"
    assert RelationType.DESCRIBED_BY == "described_by"
    assert RelationType.PART_OF == "part_of"
    assert RelationType.CONTAINS == "contains"
    assert RelationType.DEPENDS_ON == "depends_on"
    assert RelationType.USES == "uses"
    assert RelationType.ANSWERS == "answers"
    assert RelationType.ANSWERED_BY == "answered_by"
    assert RelationType.ASSUMES == "assumes"
    assert RelationType.VERIFIES == "verifies"
    assert RelationType.REFUTES == "refutes"
    assert RelationType.RELATED_TO == "related_to"


# ── KnowledgeNode Tests ───────────────────────────────────────────────────


def test_knowledge_node_defaults():
    """Test KnowledgeNode defaults."""
    node = KnowledgeNode()

    assert node.node_id == ""
    assert node.node_type == ""
    assert node.title == ""
    assert node.metadata == {}


def test_knowledge_node_to_dict():
    """Test KnowledgeNode to_dict."""
    node = KnowledgeNode(
        node_id="node-001",
        node_type="paper",
        title="My Paper",
        metadata={"author": "Test"},
    )

    data = node.to_dict()

    assert data["node_id"] == "node-001"
    assert data["node_type"] == "paper"
    assert data["title"] == "My Paper"
    assert data["metadata"] == {"author": "Test"}


def test_knowledge_node_from_dict():
    """Test KnowledgeNode from_dict."""
    data = {
        "node_id": "node-001",
        "node_type": "paper",
        "title": "My Paper",
        "metadata": {"author": "Test"},
    }

    node = KnowledgeNode.from_dict(data)

    assert node.node_id == "node-001"
    assert node.node_type == "paper"
    assert node.title == "My Paper"
    assert node.metadata == {"author": "Test"}


# ── KnowledgeEdge Tests ───────────────────────────────────────────────────


def test_knowledge_edge_defaults():
    """Test KnowledgeEdge defaults."""
    edge = KnowledgeEdge()

    assert edge.edge_id != ""  # Auto-generated
    assert edge.source_id == ""
    assert edge.target_id == ""
    assert edge.relation_type == RelationType.RELATED_TO
    assert edge.weight == 1.0
    assert edge.confidence == 1.0
    assert edge.metadata == {}
    assert edge.created_at != ""


def test_knowledge_edge_to_dict():
    """Test KnowledgeEdge to_dict."""
    edge = KnowledgeEdge(
        source_id="node-001",
        target_id="node-002",
        relation_type=RelationType.CITES,
        weight=0.8,
        confidence=0.9,
    )

    data = edge.to_dict()

    assert data["source_id"] == "node-001"
    assert data["target_id"] == "node-002"
    assert data["relation_type"] == "cites"
    assert data["weight"] == 0.8
    assert data["confidence"] == 0.9


def test_knowledge_edge_from_dict():
    """Test KnowledgeEdge from_dict."""
    data = {
        "edge_id": "edge-001",
        "source_id": "node-001",
        "target_id": "node-002",
        "relation_type": "cites",
        "weight": 0.8,
        "confidence": 0.9,
        "metadata": {},
        "created_at": "2026-06-30T10:00:00Z",
    }

    edge = KnowledgeEdge.from_dict(data)

    assert edge.edge_id == "edge-001"
    assert edge.source_id == "node-001"
    assert edge.target_id == "node-002"
    assert edge.relation_type == RelationType.CITES
    assert edge.weight == 0.8
    assert edge.confidence == 0.9


# ── InMemoryKnowledgeGraph Tests ─────────────────────────────────────────


def test_graph_is_knowledge_graph_api():
    """Test that InMemoryKnowledgeGraph implements KnowledgeGraphAPI."""
    graph = InMemoryKnowledgeGraph()
    assert isinstance(graph, KnowledgeGraphAPI)


def test_graph_add_node():
    """Test adding a node to the graph."""
    graph = InMemoryKnowledgeGraph()
    graph.add_node(node_id="node-001", node_type="paper", title="My Paper")

    node = graph.get_node("node-001")
    assert node is not None
    assert node.node_id == "node-001"
    assert node.node_type == "paper"
    assert node.title == "My Paper"


def test_graph_add_node_updates_existing():
    """Test that adding a node with an existing ID updates it."""
    graph = InMemoryKnowledgeGraph()
    graph.add_node(node_id="node-001", node_type="paper", title="Original")
    graph.add_node(node_id="node-001", node_type="paper", title="Updated", metadata={"key": "value"})

    node = graph.get_node("node-001")
    assert node is not None
    assert node.title == "Updated"
    assert node.metadata == {"key": "value"}


def test_graph_get_node_nonexistent():
    """Test getting a non-existent node returns None."""
    graph = InMemoryKnowledgeGraph()
    assert graph.get_node("nonexistent") is None


def test_graph_add_edge():
    """Test adding an edge to the graph."""
    graph = InMemoryKnowledgeGraph()
    edge_id = graph.add_edge(
        source_id="node-001",
        target_id="node-002",
        relation_type=RelationType.CITES,
        weight=0.8,
        confidence=0.9,
    )

    assert edge_id != ""

    # Check outgoing edges
    outgoing = graph.get_outgoing("node-001")
    assert len(outgoing) == 1
    assert outgoing[0].relation_type == RelationType.CITES
    assert outgoing[0].weight == 0.8
    assert outgoing[0].confidence == 0.9

    # Check incoming edges
    incoming = graph.get_incoming("node-002")
    assert len(incoming) == 1
    assert incoming[0].relation_type == RelationType.CITES


def test_graph_add_edge_deduplicates():
    """Test that adding a duplicate edge updates it instead of creating a new one."""
    graph = InMemoryKnowledgeGraph()
    edge_id1 = graph.add_edge(
        source_id="node-001",
        target_id="node-002",
        relation_type=RelationType.CITES,
        weight=0.5,
    )
    edge_id2 = graph.add_edge(
        source_id="node-001",
        target_id="node-002",
        relation_type=RelationType.CITES,
        weight=0.9,
    )

    assert edge_id1 == edge_id2  # Same edge, updated

    outgoing = graph.get_outgoing("node-001")
    assert len(outgoing) == 1
    assert outgoing[0].weight == 0.9  # Updated weight


def test_graph_add_edge_creates_nodes():
    """Test that adding an edge with non-existent nodes creates them."""
    graph = InMemoryKnowledgeGraph()
    graph.add_edge(
        source_id="node-001",
        target_id="node-002",
        relation_type=RelationType.RELATED_TO,
    )

    assert graph.get_node("node-001") is not None
    assert graph.get_node("node-002") is not None


def test_graph_get_outgoing_empty():
    """Test getting outgoing edges for a node with no edges."""
    graph = InMemoryKnowledgeGraph()
    graph.add_node(node_id="node-001", node_type="paper")

    outgoing = graph.get_outgoing("node-001")
    assert len(outgoing) == 0


def test_graph_get_incoming_empty():
    """Test getting incoming edges for a node with no edges."""
    graph = InMemoryKnowledgeGraph()
    graph.add_node(node_id="node-001", node_type="paper")

    incoming = graph.get_incoming("node-001")
    assert len(incoming) == 0


def test_graph_get_neighbors():
    """Test getting neighbors of a node."""
    graph = InMemoryKnowledgeGraph()
    graph.add_edge(source_id="a", target_id="b", relation_type=RelationType.RELATED_TO)
    graph.add_edge(source_id="a", target_id="c", relation_type=RelationType.RELATED_TO)
    graph.add_edge(source_id="d", target_id="a", relation_type=RelationType.RELATED_TO)

    neighbors = graph.get_neighbors("a")
    assert set(neighbors) == {"b", "c", "d"}


def test_graph_get_related():
    """Test getting related nodes by relation type."""
    graph = InMemoryKnowledgeGraph()
    graph.add_edge(source_id="a", target_id="b", relation_type=RelationType.CITES)
    graph.add_edge(source_id="a", target_id="c", relation_type=RelationType.USES)
    graph.add_edge(source_id="d", target_id="a", relation_type=RelationType.CITES)

    cited = graph.get_related("a", RelationType.CITES)
    assert set(cited) == {"b", "d"}


def test_graph_find_path():
    """Test finding a path between two nodes."""
    graph = InMemoryKnowledgeGraph()
    # Create a chain: a → b → c → d
    graph.add_edge(source_id="a", target_id="b", relation_type=RelationType.RELATED_TO)
    graph.add_edge(source_id="b", target_id="c", relation_type=RelationType.RELATED_TO)
    graph.add_edge(source_id="c", target_id="d", relation_type=RelationType.RELATED_TO)

    path = graph.find_path("a", "d")
    assert path is not None
    assert path[0] == "a"
    assert path[-1] == "d"


def test_graph_find_path_no_path():
    """Test finding a path when no path exists."""
    graph = InMemoryKnowledgeGraph()
    graph.add_edge(source_id="a", target_id="b", relation_type=RelationType.RELATED_TO)
    # No connection to c

    path = graph.find_path("a", "c")
    assert path is None


def test_graph_find_path_same_node():
    """Test finding a path when source and target are the same."""
    graph = InMemoryKnowledgeGraph()
    graph.add_node(node_id="a", node_type="paper")

    path = graph.find_path("a", "a")
    assert path is not None
    assert path == ["a"]


def test_graph_query_by_node_type():
    """Test querying nodes by type."""
    graph = InMemoryKnowledgeGraph()
    graph.add_node(node_id="a", node_type="paper")
    graph.add_node(node_id="b", node_type="code")
    graph.add_node(node_id="c", node_type="paper")

    papers = graph.query(node_type="paper")
    assert len(papers) == 2
    paper_ids = {p.node_id for p in papers}
    assert paper_ids == {"a", "c"}


def test_graph_query_by_relation_type():
    """Test querying nodes by relation type."""
    graph = InMemoryKnowledgeGraph()
    graph.add_edge(source_id="a", target_id="b", relation_type=RelationType.CITES)
    graph.add_edge(source_id="c", target_id="d", relation_type=RelationType.USES)

    citing_nodes = graph.query(relation_type=RelationType.CITES)
    assert len(citing_nodes) == 2
    node_ids = {n.node_id for n in citing_nodes}
    assert node_ids == {"a", "b"}


def test_graph_query_no_filters():
    """Test querying all nodes."""
    graph = InMemoryKnowledgeGraph()
    graph.add_node(node_id="a", node_type="paper")
    graph.add_node(node_id="b", node_type="code")

    all_nodes = graph.query()
    assert len(all_nodes) == 2
