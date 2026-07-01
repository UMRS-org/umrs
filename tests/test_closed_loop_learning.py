"""Tests for the closed-loop scientific learning (M3 Phase 4)."""

from __future__ import annotations

from umrs.closed_loop_learning import (
    ClosedLoopLearning,
    ExperimentResult,
    Finding,
)
from umrs.hypothesis_lifecycle import (
    HypothesisLifecycle,
    HypothesisStatus,
)
from umrs.knowledge_graph_api import InMemoryKnowledgeGraph, RelationType
from umrs.scientific_memory import InMemoryScientificMemoryStore


def test_experiment_result_defaults():
    """Test ExperimentResult defaults."""
    result = ExperimentResult(
        experiment_id="exp-001",
        hypothesis_id="hyp-001",
    )

    assert result.result_id != ""  # Auto-generated
    assert result.experiment_id == "exp-001"
    assert result.hypothesis_id == "hyp-001"
    assert result.metrics == {}
    assert result.status == "success"
    assert result.artifacts == []
    assert result.logs == ""
    assert result.created_at != ""


def test_finding_defaults():
    """Test Finding defaults."""
    finding = Finding(
        hypothesis_id="hyp-001",
        statement="Test finding",
    )

    assert finding.finding_id != ""  # Auto-generated
    assert finding.hypothesis_id == "hyp-001"
    assert finding.statement == "Test finding"
    assert finding.evidence_ids == []
    assert finding.confidence == 0.5
    assert finding.supporting is True
    assert finding.experiment_id == ""
    assert finding.created_at != ""


def test_closed_loop_init():
    """Test ClosedLoopLearning initialization."""
    loop = ClosedLoopLearning()
    assert loop is not None


def test_closed_loop_init_with_dependencies():
    """Test ClosedLoopLearning with custom dependencies."""
    graph = InMemoryKnowledgeGraph()
    memory = InMemoryScientificMemoryStore()
    lifecycle = HypothesisLifecycle()

    loop = ClosedLoopLearning(
        knowledge_graph=graph,
        scientific_memory=memory,
        hypothesis_lifecycle=lifecycle,
    )

    assert loop is not None


def test_process_experiment_result_success():
    """Test processing a successful experiment result."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test hypothesis")

    loop = ClosedLoopLearning(hypothesis_lifecycle=lifecycle)

    result = ExperimentResult(
        experiment_id="exp-001",
        hypothesis_id=hyp_id,
        metrics={"accuracy": 0.95},
        status="success",
    )

    finding = loop.process_experiment_result(result)

    assert isinstance(finding, Finding)
    assert finding.supporting is True
    assert finding.confidence == 0.8
    assert finding.experiment_id == "exp-001"
    assert len(finding.evidence_ids) == 1


def test_process_experiment_result_updates_hypothesis():
    """Test that processing a result updates the hypothesis lifecycle."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test hypothesis")

    loop = ClosedLoopLearning(hypothesis_lifecycle=lifecycle)

    result = ExperimentResult(
        experiment_id="exp-001",
        hypothesis_id=hyp_id,
        status="success",
    )

    loop.process_experiment_result(result)

    hyp = lifecycle.get(hyp_id)
    # After 1 supporting evidence (confidence=1.0 >= 0.8), transitions to SUPPORTED
    assert hyp.status == HypothesisStatus.SUPPORTED
    assert len(hyp.supporting_evidence) == 1


def test_process_experiment_result_creates_graph_edge():
    """Test that processing a result creates a knowledge graph edge."""
    graph = InMemoryKnowledgeGraph()
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test hypothesis")

    loop = ClosedLoopLearning(
        knowledge_graph=graph,
        hypothesis_lifecycle=lifecycle,
    )

    result = ExperimentResult(
        experiment_id="exp-001",
        hypothesis_id=hyp_id,
        status="success",
    )

    finding = loop.process_experiment_result(result)

    # Check that a SUPPORTS edge was created
    outgoing = graph.get_outgoing(finding.finding_id)
    assert len(outgoing) == 1
    assert outgoing[0].relation_type == RelationType.SUPPORTS
    assert outgoing[0].target_id == hyp_id


def test_process_experiment_result_updates_scientific_memory():
    """Test that processing a result updates scientific memory."""
    memory = InMemoryScientificMemoryStore()
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test hypothesis")

    loop = ClosedLoopLearning(
        scientific_memory=memory,
        hypothesis_lifecycle=lifecycle,
    )

    result = ExperimentResult(
        experiment_id="exp-001",
        hypothesis_id=hyp_id,
        status="success",
    )

    loop.process_experiment_result(result)

    entries = memory.list_entries()
    assert len(entries) == 1


def test_process_experiment_result_failure():
    """Test processing a failed experiment result."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test hypothesis")

    loop = ClosedLoopLearning(hypothesis_lifecycle=lifecycle)

    result = ExperimentResult(
        experiment_id="exp-001",
        hypothesis_id=hyp_id,
        status="failure",
    )

    finding = loop.process_experiment_result(result)

    assert finding.supporting is False
    assert finding.confidence == 0.2


def test_process_multiple_results_transitions_to_supported():
    """Test that multiple supporting results transition hypothesis to SUPPORTED."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test hypothesis")

    loop = ClosedLoopLearning(hypothesis_lifecycle=lifecycle)

    # Process multiple successful results
    for i in range(5):
        result = ExperimentResult(
            experiment_id=f"exp-{i:03d}",
            hypothesis_id=hyp_id,
            status="success",
        )
        loop.process_experiment_result(result)

    hyp = lifecycle.get(hyp_id)
    # First result transitions to SUPPORTED, subsequent results add evidence
    assert hyp.status == HypothesisStatus.SUPPORTED
    assert hyp.confidence == 1.0


def test_generate_next_hypothesis():
    """Test generating the next hypothesis."""
    loop = ClosedLoopLearning()
    hyp_id = loop.generate_next_hypothesis()

    assert hyp_id != ""


def test_closed_loop_full_cycle():
    """Test a full closed-loop cycle: experiment → finding → next hypothesis."""
    lifecycle = HypothesisLifecycle()
    graph = InMemoryKnowledgeGraph()
    memory = InMemoryScientificMemoryStore()

    loop = ClosedLoopLearning(
        knowledge_graph=graph,
        scientific_memory=memory,
        hypothesis_lifecycle=lifecycle,
    )

    # 1. Propose initial hypothesis
    hyp_id = lifecycle.propose(
        statement="Learning rate affects convergence",
        planner_version="v1",
    )

    # 2. Run experiment and process result
    result = ExperimentResult(
        experiment_id="exp-001",
        hypothesis_id=hyp_id,
        metrics={"accuracy": 0.92, "loss": 0.08},
        status="success",
    )
    finding = loop.process_experiment_result(result)

    # 3. Verify finding was created correctly
    assert finding.supporting is True
    assert finding.experiment_id == "exp-001"
    assert len(finding.evidence_ids) == 1

    # 4. Verify hypothesis was updated
    hyp = lifecycle.get(hyp_id)
    # After 1 supporting evidence (confidence=1.0 >= 0.8), transitions to SUPPORTED
    assert hyp.status == HypothesisStatus.SUPPORTED
    assert len(hyp.supporting_evidence) == 1

    # 5. Verify graph edge was created
    outgoing = graph.get_outgoing(finding.finding_id)
    assert len(outgoing) == 1
    assert outgoing[0].relation_type == RelationType.SUPPORTS

    # 6. Verify scientific memory was updated
    entries = memory.list_entries()
    assert len(entries) == 1

    # 7. Generate next hypothesis
    next_hyp_id = loop.generate_next_hypothesis()
    assert next_hyp_id != ""

    # 8. Verify the new hypothesis was proposed
    next_hyp = lifecycle.get(next_hyp_id)
    assert next_hyp is not None
    assert next_hyp.status == HypothesisStatus.PROPOSED
