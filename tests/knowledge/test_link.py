"""Tests for the knowledge linking module (Phase 5)."""

from __future__ import annotations

from umrs.knowledge.link import (
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeLinker,
    KnowledgeNode,
    RelationType,
    link_knowledge,
)

# ── Model Tests ────────────────────────────────────────────────────────────


def test_knowledge_node_defaults():
    """Test KnowledgeNode defaults."""
    node = KnowledgeNode(node_id="paper-001", node_type="paper")

    assert node.node_id == "paper-001"
    assert node.node_type == "paper"
    assert node.title == ""
    assert node.metadata == {}


def test_knowledge_node_to_dict():
    """Test KnowledgeNode to_dict."""
    node = KnowledgeNode(
        node_id="paper-001",
        node_type="paper",
        title="My Paper",
        metadata={"author": "Test"},
    )

    data = node.to_dict()
    assert data["node_id"] == "paper-001"
    assert data["node_type"] == "paper"
    assert data["title"] == "My Paper"
    assert data["metadata"]["author"] == "Test"


def test_knowledge_edge_defaults():
    """Test KnowledgeEdge defaults."""
    edge = KnowledgeEdge(
        source_id="paper-001",
        target_id="paper-002",
        relation_type=RelationType.CITES,
    )

    assert edge.source_id == "paper-001"
    assert edge.target_id == "paper-002"
    assert edge.relation_type == RelationType.CITES
    assert edge.weight == 1.0
    assert edge.confidence == 1.0
    assert edge.metadata == {}


def test_knowledge_edge_to_dict():
    """Test KnowledgeEdge to_dict."""
    edge = KnowledgeEdge(
        source_id="paper-001",
        target_id="paper-002",
        relation_type=RelationType.CITES,
        weight=0.8,
        confidence=0.9,
    )

    data = edge.to_dict()
    assert data["source_id"] == "paper-001"
    assert data["target_id"] == "paper-002"
    assert data["relation_type"] == "cites"
    assert data["weight"] == 0.8
    assert data["confidence"] == 0.9


# ── KnowledgeGraph Tests ───────────────────────────────────────────────────


def test_knowledge_graph_add_node():
    """Test adding a node to the graph."""
    graph = KnowledgeGraph()
    node = KnowledgeNode(node_id="paper-001", node_type="paper")
    graph.add_node(node)

    assert "paper-001" in graph.nodes
    assert graph.get_node("paper-001") is node


def test_knowledge_graph_add_edge():
    """Test adding an edge to the graph."""
    graph = KnowledgeGraph()
    graph.add_node(KnowledgeNode(node_id="paper-001", node_type="paper"))
    graph.add_node(KnowledgeNode(node_id="paper-002", node_type="paper"))
    edge = KnowledgeEdge(
        source_id="paper-001",
        target_id="paper-002",
        relation_type=RelationType.CITES,
    )
    graph.add_edge(edge)

    assert len(graph.edges) == 1
    assert len(graph.get_outgoing("paper-001")) == 1
    assert len(graph.get_incoming("paper-002")) == 1


def test_knowledge_graph_no_duplicate_edges():
    """Test that duplicate edges are not added."""
    graph = KnowledgeGraph()
    edge1 = KnowledgeEdge(source_id="a", target_id="b", relation_type=RelationType.CITES)
    edge2 = KnowledgeEdge(source_id="a", target_id="b", relation_type=RelationType.CITES)
    graph.add_edge(edge1)
    graph.add_edge(edge2)

    assert len(graph.edges) == 1


def test_knowledge_graph_get_neighbors():
    """Test getting neighbors of a node."""
    graph = KnowledgeGraph()
    graph.add_edge(
        KnowledgeEdge(source_id="a", target_id="b", relation_type=RelationType.RELATED_TO)
    )
    graph.add_edge(
        KnowledgeEdge(source_id="a", target_id="c", relation_type=RelationType.RELATED_TO)
    )
    graph.add_edge(
        KnowledgeEdge(source_id="d", target_id="a", relation_type=RelationType.RELATED_TO)
    )

    neighbors = graph.get_neighbors("a")
    assert set(neighbors) == {"b", "c", "d"}


def test_knowledge_graph_find_path():
    """Test finding a path between two nodes."""
    graph = KnowledgeGraph()
    # Create a chain: a → b → c → d
    graph.add_edge(
        KnowledgeEdge(source_id="a", target_id="b", relation_type=RelationType.RELATED_TO)
    )
    graph.add_edge(
        KnowledgeEdge(source_id="b", target_id="c", relation_type=RelationType.RELATED_TO)
    )
    graph.add_edge(
        KnowledgeEdge(source_id="c", target_id="d", relation_type=RelationType.RELATED_TO)
    )

    path = graph.find_path("a", "d")
    assert path is not None
    assert path[0] == "a"
    assert path[-1] == "d"


def test_knowledge_graph_find_path_no_path():
    """Test finding a path when no path exists."""
    graph = KnowledgeGraph()
    graph.add_edge(
        KnowledgeEdge(source_id="a", target_id="b", relation_type=RelationType.RELATED_TO)
    )
    # No connection to c
    path = graph.find_path("a", "c")
    assert path is None


def test_knowledge_graph_to_dict():
    """Test converting graph to dictionary."""
    graph = KnowledgeGraph()
    graph.add_node(KnowledgeNode(node_id="a", node_type="paper"))
    graph.add_edge(KnowledgeEdge(source_id="a", target_id="b", relation_type=RelationType.CITES))

    data = graph.to_dict()
    assert "a" in data["nodes"]
    assert len(data["edges"]) == 1


# ── KnowledgeLinker Tests ─────────────────────────────────────────────────


def test_linker_init():
    """Test KnowledgeLinker initialization."""
    linker = KnowledgeLinker()

    assert len(linker._citation_patterns) > 0
    assert len(linker._mention_patterns) > 0


def test_linker_link_empty():
    """Test linking empty list of artifacts."""
    linker = KnowledgeLinker()
    graph = linker.link([])

    assert len(graph.nodes) == 0
    assert len(graph.edges) == 0


def test_linker_link_single_artifact():
    """Test linking a single artifact."""
    linker = KnowledgeLinker()
    artifacts = [
        {"artifact_id": "paper-001", "artifact_type": "paper", "filename": "paper.pdf"},
    ]
    graph = linker.link(artifacts)

    assert len(graph.nodes) == 1
    assert graph.get_node("paper-001") is not None


def test_linker_link_by_filename():
    """Test linking artifacts by filename similarity."""
    linker = KnowledgeLinker()
    artifacts = [
        {"artifact_id": "code-001", "artifact_type": "code", "filename": "transformer.py"},
        {
            "artifact_id": "notebook-001",
            "artifact_type": "notebook",
            "filename": "transformer.ipynb",
        },
    ]
    graph = linker.link(artifacts)

    # Should have edges linking the two artifacts
    assert len(graph.edges) >= 2  # At least one in each direction


def test_linker_link_with_extracted_knowledge():
    """Test linking with extracted knowledge."""
    linker = KnowledgeLinker()
    artifacts = [
        {
            "artifact_id": "paper-001",
            "artifact_type": "paper",
            "filename": "attention.pdf",
            "extracted_knowledge": {
                "datasets": ["QM9"],
                "algorithms": ["transformer"],
            },
        },
        {
            "artifact_id": "code-001",
            "artifact_type": "code",
            "filename": "transformer.py",
            "extracted_knowledge": {
                "algorithms": ["transformer"],
            },
        },
    ]
    graph = linker.link(artifacts)

    assert len(graph.nodes) == 2
    # Should have at least one edge
    assert len(graph.edges) >= 1


# ── Convenience Function Tests ─────────────────────────────────────────────


def test_link_knowledge_convenience_function():
    """Test the link_knowledge convenience function."""
    artifacts = [
        {"artifact_id": "a", "artifact_type": "paper", "filename": "test.pdf"},
        {"artifact_id": "b", "artifact_type": "code", "filename": "test.py"},
    ]
    graph = link_knowledge(artifacts)

    assert isinstance(graph, KnowledgeGraph)
    assert len(graph.nodes) == 2
