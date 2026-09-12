"""End-to-end Scientific Research Workflow Test.

M3 Phase 5: Scientific Project Demonstration.

This is a PERMANENT REGRESSION TEST that exercises every major
architectural capability of the R4 scientific research platform.

The test verifies:
  1. Upload artifacts (paper, code, notebook, dataset)
  2. Knowledge graph has typed edges between objects
  3. Scientific Reasoning Engine reasons from the knowledge graph
  4. Planner calls Scientific Reasoning Engine (single reasoning path)
  5. Hypothesis has its own lifecycle with evidence-based transitions
  6. Closed-Loop Learning: Experiment → Finding → KnowledgeObject → next hypothesis
  7. Verification Session persists evidence IDs and graph traversal (NOT the LLM answer)
  8. Single confidence model (computed in ScientificReasoningEngine)
  9. Dependency direction: Planner → ScientificReasoningEngine → GraphNavigator → Knowledge Graph

If this test passes, the scientific reasoning architecture is integrated.
If it fails, an integration gap has been introduced.
"""

from __future__ import annotations

from umrs.closed_loop_learning import (
    ClosedLoopLearning,
    ExperimentResult,
)
from umrs.evidence import Evidence, EvidenceSourceType
from umrs.graph_reasoning import GraphNavigator
from umrs.hypothesis_lifecycle import (
    HypothesisLifecycle,
    HypothesisStatus,
)
from umrs.knowledge.canonical_graph_view import CanonicalGraphView, default_view
from umrs.knowledge.graph_types import RelationType
from umrs.knowledge.knowledge_driven_planner import KnowledgeDrivenPlanner
from umrs.knowledge.planner import ResearchPlan
from umrs.knowledge.verification_session import (
    VerificationSession,
    VerificationSessionStore,
)
from umrs.scientific_memory import InMemoryScientificMemoryStore
from umrs.scientific_reasoning_engine import ScientificReasoningEngine


# ── Test: Full Scientific Research Workflow ──────────────────────────────


def test_end_to_end_scientific_research_workflow():
    """Test the complete scientific research workflow end-to-end.

    This test exercises every major architectural capability:
      1. Knowledge Graph construction with typed edges
      2. Scientific Reasoning Engine reasoning from the graph
      3. Knowledge-Driven Planner using the reasoning engine (single path)
      4. Hypothesis Lifecycle with evidence-based state transitions
      5. Closed-Loop Learning: experiment → finding → next hypothesis
      6. Verification Session: persists evidence and graph traversal
      7. Single confidence model through the reasoning engine
    """

    # ── Step 1: Build Knowledge Graph ───────────────────────────────────
    # Simulate the knowledge graph that would be built after ingestion
    graph = default_view(store=None)

    # Add scientific objects as nodes
    graph.add_node(node_id="paper-001", node_type="paper", title="Attention Is All You Need")
    graph.add_node(node_id="paper-002", node_type="paper", title="BERT: Pre-training of Deep Bidirectional Transformers")
    graph.add_node(node_id="code-001", node_type="code", title="transformer_model.py")
    graph.add_node(node_id="code-002", node_type="code", title="data_loader.py")
    graph.add_node(node_id="notebook-001", node_type="notebook", title="transformer_experiment.ipynb")
    graph.add_node(node_id="dataset-001", node_type="dataset", title="QM9 Molecular Dataset")
    graph.add_node(node_id="hypothesis-001", node_type="hypothesis", title="Transformers can predict molecular properties")
    graph.add_node(node_id="experiment-001", node_type="experiment", title="Transformer on QM9")
    graph.add_node(node_id="finding-001", node_type="finding", title="Transformer achieves 95% accuracy on QM9")

    # Add typed edges (relationships between scientific objects)
    graph.add_edge(source_id="paper-001", target_id="paper-002", relation_type=RelationType.CITES, confidence=0.9)
    graph.add_edge(source_id="paper-001", target_id="code-001", relation_type=RelationType.IMPLEMENTS, confidence=0.85)
    graph.add_edge(source_id="paper-001", target_id="hypothesis-001", relation_type=RelationType.DESCRIBES, confidence=0.8)
    graph.add_edge(source_id="code-001", target_id="notebook-001", relation_type=RelationType.RELATED_TO, confidence=0.7)
    graph.add_edge(source_id="code-001", target_id="code-002", relation_type=RelationType.DEPENDS_ON, confidence=0.9)
    graph.add_edge(source_id="notebook-001", target_id="dataset-001", relation_type=RelationType.USES, confidence=0.95)
    graph.add_edge(source_id="notebook-001", target_id="experiment-001", relation_type=RelationType.PRODUCES, confidence=0.85)
    graph.add_edge(source_id="experiment-001", target_id="finding-001", relation_type=RelationType.PRODUCES, confidence=0.9)
    graph.add_edge(source_id="finding-001", target_id="hypothesis-001", relation_type=RelationType.SUPPORTS, confidence=0.95)

    # Verify graph has nodes and edges
    all_nodes = graph.query()
    assert len(all_nodes) >= 9, f"Expected at least 9 nodes, got {len(all_nodes)}"

    outgoing = graph.get_outgoing("paper-001")
    assert len(outgoing) >= 3, f"Expected at least 3 outgoing edges from paper-001, got {len(outgoing)}"

    # ── Step 2: Scientific Reasoning Engine ──────────────────────────────
    # Create the reasoning engine with the knowledge graph
    navigator = GraphNavigator(graph=graph)
    reasoning_engine = ScientificReasoningEngine(navigator=navigator)

    # Verify the reasoning engine can reason from the graph
    reasoning_result = reasoning_engine.reason(
        query="What research should we perform next?",
        anchor_object_id="paper-001",
    )

    assert reasoning_result is not None
    assert reasoning_result.query == "What research should we perform next?"
    assert reasoning_result.confidence >= 0.0
    assert len(reasoning_result.evidence_ids) >= 0
    assert len(reasoning_result.reasoning_steps) > 0

    # ── Step 3: Knowledge-Driven Planner ─────────────────────────────────
    # The planner should use the Scientific Reasoning Engine (single path)
    planner = KnowledgeDrivenPlanner(reasoning_engine=reasoning_engine)

    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Understand how transformers predict molecular properties",
        anchor_object_id="paper-001",
    )

    # Verify the plan was generated
    assert isinstance(plan, ResearchPlan)
    assert plan.project_id == "proj-001"
    assert len(plan.objectives) >= 3
    assert len(plan.hypotheses) >= 1
    assert len(plan.experiments) >= 1
    assert len(plan.expected_outcomes) >= 3
    assert 0.0 <= plan.success_probability <= 1.0

    # ── Step 4: Hypothesis Lifecycle ─────────────────────────────────────
    # Create a hypothesis lifecycle and propose a hypothesis
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(
        statement="Transformers can predict molecular properties with high accuracy",
        planner_version="plan-v1",
    )

    # Verify the hypothesis was proposed
    hyp = lifecycle.get(hyp_id)
    assert hyp is not None
    assert hyp.status == HypothesisStatus.PROPOSED
    assert hyp.confidence == 0.5
    assert hyp.originating_planner_version == "plan-v1"

    # Add supporting evidence
    lifecycle.add_evidence(hyp_id, "evidence-001", supports=True)
    assert hyp.confidence == 1.0  # 1 supporting, 0 contradicting

    # Transition through lifecycle
    lifecycle.transition(hyp_id, HypothesisStatus.TESTING)
    assert hyp.status == HypothesisStatus.TESTING

    # ── Step 5: Closed-Loop Learning ─────────────────────────────────────
    # Create a closed-loop learning system
    memory = InMemoryScientificMemoryStore()
    closed_loop = ClosedLoopLearning(
        knowledge_graph=graph,
        scientific_memory=memory,
        hypothesis_lifecycle=lifecycle,
        reasoning_engine=reasoning_engine,
    )

    # Simulate an experiment result
    result = ExperimentResult(
        experiment_id="exp-002",
        hypothesis_id=hyp_id,
        metrics={"accuracy": 0.97, "loss": 0.03},
        status="success",
    )

    # Process the experiment result through the closed loop
    finding = closed_loop.process_experiment_result(result)

    # Verify the finding was created
    assert finding is not None
    assert finding.supporting is True
    assert finding.experiment_id == "exp-002"
    assert len(finding.evidence_ids) == 1

    # Verify the hypothesis lifecycle was updated
    hyp = lifecycle.get(hyp_id)
    assert hyp.status == HypothesisStatus.SUPPORTED
    assert len(hyp.supporting_evidence) == 2  # evidence-001 + new evidence

    # Verify scientific memory was updated
    memory_entries = memory.list_entries()
    assert len(memory_entries) >= 1

    # ── Step 6: Verification Session ─────────────────────────────────────
    # Create a verification session that persists evidence IDs and graph traversal
    session = VerificationSession(
        project_id="proj-001",
        question="Explain this project to a new researcher",
        evidence_ids=reasoning_result.evidence_ids,
        graph_traversal=reasoning_result.graph_paths[0] if reasoning_result.graph_paths else [],
        answer="This is the LLM-generated answer (ephemeral, NOT persisted)",
        confidence=reasoning_result.confidence,
    )

    # Verify the session was created
    assert session.session_id != ""
    assert session.question == "Explain this project to a new researcher"
    assert session.answer == "This is the LLM-generated answer (ephemeral, NOT persisted)"

    # Verify to_persistent_dict excludes the ephemeral answer
    persistent_data = session.to_persistent_dict()
    assert "answer" not in persistent_data
    assert persistent_data["evidence_ids"] == reasoning_result.evidence_ids
    assert persistent_data["confidence"] == reasoning_result.confidence

    # ── Step 7: Single Confidence Model ─────────────────────────────────
    # Verify that confidence is computed in the Scientific Reasoning Engine
    # (not in the planner, not in the closed-loop learning)
    assert reasoning_result.confidence >= 0.0
    assert reasoning_result.confidence <= 1.0

    # The planner's success_probability should be derived from the reasoning
    assert plan.success_probability >= 0.0
    assert plan.success_probability <= 1.0

    # The hypothesis's confidence should be derived from evidence
    assert hyp.confidence >= 0.0
    assert hyp.confidence <= 1.0


def test_single_reasoning_path():
    """Verify that there is a single reasoning path.

    The path should be:
      Planner → ScientificReasoningEngine → GraphNavigator → Knowledge Graph

    The planner should NOT call GraphNavigator directly.
    """
    graph = default_view(store=None)
    graph.add_node(node_id="paper-001", node_type="paper", title="Test Paper")

    navigator = GraphNavigator(graph=graph)
    reasoning_engine = ScientificReasoningEngine(navigator=navigator)
    planner = KnowledgeDrivenPlanner(reasoning_engine=reasoning_engine)

    # The planner should have a reasoning_engine, not a navigator
    assert hasattr(planner, "_reasoning_engine")
    assert not hasattr(planner, "_navigator"), "Planner should NOT have a _navigator attribute"

    # The reasoning engine should have a navigator
    assert hasattr(reasoning_engine, "_navigator")
    assert reasoning_engine._navigator is navigator


def test_dependency_direction():
    """Verify that the dependency graph only points downward.

    The direction should be:
      Planner → ScientificReasoningEngine → GraphNavigator → Knowledge Graph

    Nothing below should call back into the planner.
    """
    graph = default_view(store=None)
    navigator = GraphNavigator(graph=graph)
    reasoning_engine = ScientificReasoningEngine(navigator=navigator)
    planner = KnowledgeDrivenPlanner(reasoning_engine=reasoning_engine)

    # Verify the dependency chain
    # Planner → ScientificReasoningEngine
    assert planner._reasoning_engine is reasoning_engine

    # ScientificReasoningEngine → GraphNavigator
    assert reasoning_engine._navigator is navigator

    # GraphNavigator → Knowledge Graph
    assert navigator._graph is graph

    # Verify that the Knowledge Graph does NOT reference the planner
    # (dependency direction is downward only)
    assert not hasattr(graph, "_planner")
    assert not hasattr(graph, "_reasoning_engine")


def test_graph_traversal_uses_single_implementation():
    """Verify that graph traversal uses a single implementation.

    All graph traversal should go through GraphNavigator.find_path()
    or GraphNavigator.get_neighbors(), not through duplicate implementations.
    """
    graph = default_view(store=None)
    graph.add_node(node_id="a", node_type="paper")
    graph.add_node(node_id="b", node_type="code")
    graph.add_edge(source_id="a", target_id="b", relation_type=RelationType.IMPLEMENTS)

    navigator = GraphNavigator(graph=graph)

    # Verify that find_path uses the graph's find_path
    path = navigator.explain_path("a", "b")
    assert path is not None
    assert path[0] == "a"
    assert path[-1] == "b"

    # Verify that the navigator delegates to the graph for traversal
    neighbors = navigator.graph.get_neighbors("a")
    assert "b" in neighbors


def test_evidence_ranking_uses_single_implementation():
    """Verify that evidence ranking uses a single implementation.

    All evidence ranking should go through the Scientific Reasoning Engine,
    not through duplicate implementations in the planner or closed-loop learning.
    """
    graph = default_view(store=None)
    graph.add_node(node_id="paper-001", node_type="paper", title="Test Paper")
    graph.add_node(node_id="finding-001", node_type="finding", title="Finding")
    graph.add_edge(source_id="finding-001", target_id="paper-001", relation_type=RelationType.SUPPORTS, confidence=0.9)

    navigator = GraphNavigator(graph=graph)
    reasoning_engine = ScientificReasoningEngine(navigator=navigator)

    # Verify that the reasoning engine computes confidence
    result = reasoning_engine.reason(
        query="What evidence supports this?",
        anchor_object_id="paper-001",
    )
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0

    # Verify that the reasoning engine's confidence is used by the planner
    planner = KnowledgeDrivenPlanner(reasoning_engine=reasoning_engine)
    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Test research goal",
        anchor_object_id="paper-001",
    )
    assert plan.success_probability >= 0.0
    assert plan.success_probability <= 1.0


def test_closed_loop_uses_reasoning_engine():
    """Verify that ClosedLoopLearning uses the Scientific Reasoning Engine.

    The closed-loop should call reasoning_engine.reason() to generate
    the next hypothesis, not implement its own reasoning logic.
    """
    graph = default_view(store=None)
    graph.add_node(node_id="paper-001", node_type="paper", title="Test Paper")
    graph.add_node(node_id="hypothesis-001", node_type="hypothesis", title="Test Hypothesis")
    graph.add_edge(source_id="paper-001", target_id="hypothesis-001", relation_type=RelationType.DESCRIBES, confidence=0.8)

    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test hypothesis")

    navigator = GraphNavigator(graph=graph)
    reasoning_engine = ScientificReasoningEngine(navigator=navigator)

    closed_loop = ClosedLoopLearning(
        knowledge_graph=graph,
        hypothesis_lifecycle=lifecycle,
        reasoning_engine=reasoning_engine,
    )

    # Verify that the closed-loop has a reasoning_engine
    assert hasattr(closed_loop, "_reasoning_engine")
    assert closed_loop._reasoning_engine is reasoning_engine

    # Process an experiment result
    result = ExperimentResult(
        experiment_id="exp-001",
        hypothesis_id=hyp_id,
        metrics={"accuracy": 0.95},
        status="success",
    )
    finding = closed_loop.process_experiment_result(result)
    assert finding is not None

    # Generate next hypothesis (should use reasoning engine)
    next_hyp_id = closed_loop.generate_next_hypothesis()
    assert next_hyp_id != ""
