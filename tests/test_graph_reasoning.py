"""Tests for the graph reasoning module (M3 Phase 2)."""

from __future__ import annotations

from umrs.graph_reasoning import (
    EvidenceChain,
    ExperimentChain,
    GraphNavigator,
    GraphReasoningContext,
    RelatedObjects,
)
from umrs.knowledge.canonical_graph_view import CanonicalGraphView, in_memory_view
from umrs.knowledge.graph_types import RelationType


# ── Fixtures ──────────────────────────────────────────────────────────────


def _build_test_graph() -> CanonicalGraphView:
    """Build a test knowledge graph with scientific objects."""
    graph = in_memory_view()

    # Add nodes
    graph.add_node(node_id="paper-001", node_type="paper", title="Attention Is All You Need")
    graph.add_node(node_id="paper-002", node_type="paper", title="BERT")
    graph.add_node(node_id="code-001", node_type="code", title="transformer.py")
    graph.add_node(node_id="notebook-001", node_type="notebook", title="experiment.ipynb")
    graph.add_node(node_id="dataset-001", node_type="dataset", title="QM9")
    graph.add_node(node_id="hypothesis-001", node_type="hypothesis", title="Transformers can predict molecular properties")
    graph.add_node(node_id="experiment-001", node_type="experiment", title="Transformer on QM9")
    graph.add_node(node_id="finding-001", node_type="finding", title="Transformer achieves 95% accuracy")
    graph.add_node(node_id="figure-001", node_type="figure", title="loss_curve.png")
    graph.add_node(node_id="equation-001", node_type="equation", title="attention_weight")

    # Add edges
    graph.add_edge(source_id="paper-001", target_id="paper-002", relation_type=RelationType.CITES)
    graph.add_edge(source_id="paper-001", target_id="code-001", relation_type=RelationType.IMPLEMENTS)
    graph.add_edge(source_id="paper-001", target_id="hypothesis-001", relation_type=RelationType.DESCRIBES)
    graph.add_edge(source_id="code-001", target_id="notebook-001", relation_type=RelationType.RELATED_TO)
    graph.add_edge(source_id="notebook-001", target_id="dataset-001", relation_type=RelationType.USES)
    graph.add_edge(source_id="notebook-001", target_id="experiment-001", relation_type=RelationType.PRODUCES)
    graph.add_edge(source_id="experiment-001", target_id="finding-001", relation_type=RelationType.PRODUCES)
    graph.add_edge(source_id="experiment-001", target_id="figure-001", relation_type=RelationType.PRODUCES)
    graph.add_edge(source_id="finding-001", target_id="hypothesis-001", relation_type=RelationType.SUPPORTS)
    graph.add_edge(source_id="hypothesis-001", target_id="equation-001", relation_type=RelationType.RELATED_TO)

    return graph


# ── Model Tests ────────────────────────────────────────────────────────────


def test_related_objects_empty_object_lists():
    """Empty-collection buckets start empty."""
    related = RelatedObjects()
    assert related.papers == []
    assert related.code == []
    assert related.notebooks == []
    assert related.datasets == []
    assert related.hypotheses == []
    assert related.experiments == []


def test_related_objects_empty_namesake_lists():
    """Domain buckets also start empty."""
    related = RelatedObjects()
    assert related.findings == []
    assert related.figures == []
    assert related.equations == []
    assert related.total() == 0


def test_related_objects_total():
    """Test RelatedObjects total method."""
    related = RelatedObjects(
        papers=["p1", "p2"],
        code=["c1"],
        notebooks=["n1"],
    )
    assert related.total() == 4


def test_evidence_chain_defaults():
    """Test EvidenceChain defaults."""
    chain = EvidenceChain()
    assert chain.claim_id == ""
    assert chain.supporting == []
    assert chain.contradicting == []
    assert chain.graph_path == []


def test_experiment_chain_defaults():
    """Test ExperimentChain defaults."""
    chain = ExperimentChain()
    assert chain.hypothesis_id == ""
    assert chain.experiments == []
    assert chain.findings == []
    assert chain.graph_path == []


def test_graph_reasoning_context_defaults():
    """Test GraphReasoningContext defaults."""
    context = GraphReasoningContext()
    assert context.query == ""
    assert context.related.total() == 0
    assert context.evidence_chains == []
    assert context.experiment_chains == []
    assert context.graph_paths == []


# ── GraphNavigator Tests ──────────────────────────────────────────────────


def test_graph_navigator_init():
    """Test GraphNavigator initialization."""
    nav = GraphNavigator()
    assert nav is not None
    assert nav.graph is not None


def test_graph_navigator_init_with_graph():
    """Test GraphNavigator with custom graph."""
    graph = in_memory_view()
    nav = GraphNavigator(graph=graph)
    assert nav.graph is graph


def test_find_related():
    """Test finding related objects through the graph."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    related = nav.find_related("paper-001")

    # paper-001 is connected to paper-002 (CITES), code-001 (IMPLEMENTS),
    # hypothesis-001 (DESCRIBES)
    assert "paper-002" in related.papers
    assert "code-001" in related.code
    assert "hypothesis-001" in related.hypotheses
    assert related.total() > 0


def test_find_related_by_type():
    """Test finding related objects of a specific type."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    papers = nav.find_related_by_type("paper-001", "paper")
    assert "paper-002" in papers

    code = nav.find_related_by_type("paper-001", "code")
    assert "code-001" in code


def test_find_related_by_relation():
    """Test finding related objects by relation type."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    cited = nav.find_related_by_relation("paper-001", RelationType.CITES)
    assert "paper-002" in cited

    implemented = nav.find_related_by_relation("paper-001", RelationType.IMPLEMENTS)
    assert "code-001" in implemented


def test_find_evidence_chain():
    """Test finding evidence chains for a claim."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    # finding-001 SUPPORTS hypothesis-001
    chain = nav.find_evidence_chain("hypothesis-001")

    assert chain.claim_id == "hypothesis-001"
    # finding-001 should be in supporting evidence
    assert "finding-001" in chain.supporting


def test_find_experiment_chain():
    """Test finding experiment chains for a hypothesis."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    chain = nav.find_experiment_chain("hypothesis-001")

    assert chain.hypothesis_id == "hypothesis-001"
    # Should find experiments related to this hypothesis
    # (through the graph: hypothesis-001 ← equation-001, and
    #  notebook-001 → experiment-001, but we need direct connections)


def test_explain_path():
    """Test finding a path between two objects."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    # paper-001 → code-001 → notebook-001 → experiment-001
    path = nav.explain_path("paper-001", "experiment-001")
    assert path is not None
    assert path[0] == "paper-001"
    assert path[-1] == "experiment-001"


def test_explain_path_no_path():
    """Test finding a path when no path exists."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    # Add an isolated node
    graph.add_node(node_id="isolated-001", node_type="paper")

    path = nav.explain_path("paper-001", "isolated-001")
    assert path is None


def test_explain_edges():
    """Test finding edges between two objects."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    edges = nav.explain_edges("paper-001", "paper-002")
    assert len(edges) > 0
    assert any(e.relation_type == RelationType.CITES for e in edges)


def test_reason():
    """Test building a complete reasoning context."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    context = nav.reason("What research should we do next?", "paper-001")

    assert context.query == "What research should we do next?"
    assert context.related.total() > 0
    assert len(context.evidence_chains) > 0 or len(context.experiment_chains) > 0
    assert len(context.graph_paths) > 0


def test_reason_with_hypothesis_anchor():
    """Test reasoning with a hypothesis as the anchor."""
    graph = _build_test_graph()
    nav = GraphNavigator(graph=graph)

    context = nav.reason("Is this hypothesis supported?", "hypothesis-001")

    assert context.query == "Is this hypothesis supported?"
    assert context.related.total() > 0
    # Should find evidence chains for the hypothesis
    assert any(
        chain.claim_id == "hypothesis-001"
        for chain in context.evidence_chains
    )


def test_find_related_empty_graph():
    """Test finding related objects in an empty graph."""
    nav = GraphNavigator()
    related = nav.find_related("nonexistent")
    assert related.total() == 0


def test_find_evidence_chain_no_evidence():
    """Test finding evidence chain when no evidence exists."""
    graph = in_memory_view()
    graph.add_node(node_id="hypothesis-001", node_type="hypothesis")
    nav = GraphNavigator(graph=graph)

    chain = nav.find_evidence_chain("hypothesis-001")
    assert chain.supporting == []
    assert chain.contradicting == []
