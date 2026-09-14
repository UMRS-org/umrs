"""Tests for the hypothesis lifecycle (M3 Phase 4)."""

from __future__ import annotations

import pytest
from umrs.hypothesis_lifecycle import (
    Hypothesis,
    HypothesisLifecycle,
    HypothesisStatus,
)


def test_hypothesis_status_enum():
    """Test HypothesisStatus enum values."""
    assert HypothesisStatus.PROPOSED == "proposed"
    assert HypothesisStatus.TESTING == "testing"
    assert HypothesisStatus.SUPPORTED == "supported"
    assert HypothesisStatus.CONTRADICTED == "contradicted"
    assert HypothesisStatus.INCONCLUSIVE == "inconclusive"
    assert HypothesisStatus.REFINED == "refined"
    assert HypothesisStatus.ACCEPTED == "accepted"
    assert HypothesisStatus.REJECTED == "rejected"


def test_hypothesis_defaults():
    """Test Hypothesis defaults."""
    hyp = Hypothesis(statement="Test hypothesis")

    assert hyp.hypothesis_id != ""  # Auto-generated
    assert hyp.statement == "Test hypothesis"
    assert hyp.status == HypothesisStatus.PROPOSED
    assert hyp.confidence == 0.5
    assert hyp.supporting_evidence == []
    assert hyp.contradicting_evidence == []
    assert hyp.associated_experiments == []
    assert hyp.parent_hypothesis_id is None
    assert hyp.created_at != ""
    assert hyp.updated_at != ""


def test_hypothesis_to_dict():
    """Test Hypothesis to_dict."""
    hyp = Hypothesis(
        statement="Transformers work well",
        status=HypothesisStatus.SUPPORTED,
        confidence=0.9,
    )

    data = hyp.to_dict()

    assert data["statement"] == "Transformers work well"
    assert data["status"] == "supported"
    assert data["confidence"] == 0.9


def test_hypothesis_add_supporting_evidence():
    """Test adding supporting evidence."""
    hyp = Hypothesis(statement="Test")
    hyp.add_supporting_evidence("evidence-001")
    hyp.add_supporting_evidence("evidence-002")

    assert len(hyp.supporting_evidence) == 2
    assert hyp.confidence == 1.0  # 2 supporting, 0 contradicting


def test_hypothesis_add_contradicting_evidence():
    """Test adding contradicting evidence."""
    hyp = Hypothesis(statement="Test")
    hyp.add_contradicting_evidence("evidence-001")

    assert len(hyp.contradicting_evidence) == 1
    assert hyp.confidence == 0.0  # 0 supporting, 1 contradicting


def test_hypothesis_confidence_balanced():
    """Test confidence with balanced evidence."""
    hyp = Hypothesis(statement="Test")
    hyp.add_supporting_evidence("ev-001")
    hyp.add_contradicting_evidence("ev-002")

    assert hyp.confidence == 0.5  # 1 supporting, 1 contradicting


def test_hypothesis_add_experiment():
    """Test adding an experiment."""
    hyp = Hypothesis(statement="Test")
    hyp.add_experiment("experiment-001")

    assert len(hyp.associated_experiments) == 1


def test_hypothesis_add_duplicate_evidence():
    """Test that duplicate evidence is not added."""
    hyp = Hypothesis(statement="Test")
    hyp.add_supporting_evidence("ev-001")
    hyp.add_supporting_evidence("ev-001")

    assert len(hyp.supporting_evidence) == 1


def test_lifecycle_propose():
    """Test proposing a hypothesis."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(
        statement="Transformers can predict molecular properties",
        planner_version="planner-v1",
    )

    hyp = lifecycle.get(hyp_id)
    assert hyp is not None
    assert hyp.statement == "Transformers can predict molecular properties"
    assert hyp.status == HypothesisStatus.PROPOSED
    assert hyp.originating_planner_version == "planner-v1"


def test_lifecycle_transition_valid():
    """Test valid lifecycle transitions."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test")

    # PROPOSED → TESTING
    new_status = lifecycle.transition(hyp_id, HypothesisStatus.TESTING)
    assert new_status == HypothesisStatus.TESTING

    # TESTING → SUPPORTED
    new_status = lifecycle.transition(hyp_id, HypothesisStatus.SUPPORTED)
    assert new_status == HypothesisStatus.SUPPORTED

    # SUPPORTED → ACCEPTED
    new_status = lifecycle.transition(hyp_id, HypothesisStatus.ACCEPTED)
    assert new_status == HypothesisStatus.ACCEPTED


def test_lifecycle_transition_invalid():
    """Test that invalid transitions raise an error."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test")

    # PROPOSED → ACCEPTED is invalid
    with pytest.raises(ValueError, match="Invalid transition"):
        lifecycle.transition(hyp_id, HypothesisStatus.ACCEPTED)


def test_lifecycle_transition_nonexistent():
    """Test that transitioning a non-existent hypothesis raises KeyError."""
    lifecycle = HypothesisLifecycle()
    with pytest.raises(KeyError):
        lifecycle.transition("nonexistent", HypothesisStatus.TESTING)


def test_lifecycle_add_evidence_supporting():
    """Test adding supporting evidence through lifecycle."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test")

    lifecycle.add_evidence(hyp_id, "ev-001", supports=True)

    hyp = lifecycle.get(hyp_id)
    assert "ev-001" in hyp.supporting_evidence
    assert hyp.confidence == 1.0


def test_lifecycle_add_evidence_contradicting():
    """Test adding contradicting evidence through lifecycle."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test")

    lifecycle.add_evidence(hyp_id, "ev-001", supports=False)

    hyp = lifecycle.get(hyp_id)
    assert "ev-001" in hyp.contradicting_evidence
    assert hyp.confidence == 0.0


def test_lifecycle_get_history():
    """Test getting the lifecycle history."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(statement="Test")

    lifecycle.transition(hyp_id, HypothesisStatus.TESTING)
    lifecycle.transition(hyp_id, HypothesisStatus.SUPPORTED)

    history = lifecycle.get_history(hyp_id)
    assert len(history) == 3  # PROPOSED, TESTING, SUPPORTED
    assert history[0][0] == HypothesisStatus.PROPOSED
    assert history[1][0] == HypothesisStatus.TESTING
    assert history[2][0] == HypothesisStatus.SUPPORTED


def test_lifecycle_list_hypotheses():
    """Test listing hypotheses."""
    lifecycle = HypothesisLifecycle()
    h1 = lifecycle.propose(statement="H1")
    h2 = lifecycle.propose(statement="H2")

    all_hyps = lifecycle.list_hypotheses()
    assert len(all_hyps) == 2

    proposed = lifecycle.list_hypotheses(status=HypothesisStatus.PROPOSED)
    assert len(proposed) == 2


def test_lifecycle_refine():
    """Test refining a hypothesis."""
    lifecycle = HypothesisLifecycle()
    parent_id = lifecycle.propose(
        statement="Transformers work for all molecules",
        planner_version="v1",
    )

    # Refine into a more specific hypothesis
    child_id = lifecycle.refine(
        parent_hypothesis_id=parent_id,
        new_statement="Transformers work for small molecules only",
        planner_version="v2",
    )

    # Parent should be REFINED
    parent = lifecycle.get(parent_id)
    assert parent.status == HypothesisStatus.REFINED

    # Child should be PROPOSED
    child = lifecycle.get(child_id)
    assert child.status == HypothesisStatus.PROPOSED
    assert child.parent_hypothesis_id == parent_id
    assert child.originating_planner_version == "v2"


def test_lifecycle_full_cycle():
    """Test a full hypothesis lifecycle: propose → test → support → accept."""
    lifecycle = HypothesisLifecycle()
    hyp_id = lifecycle.propose(
        statement="Learning rate affects convergence",
        planner_version="v1",
    )

    # Add evidence
    lifecycle.add_evidence(hyp_id, "ev-001", supports=True)
    lifecycle.add_evidence(hyp_id, "ev-002", supports=True)

    # Transition through lifecycle
    lifecycle.transition(hyp_id, HypothesisStatus.TESTING)
    lifecycle.transition(hyp_id, HypothesisStatus.SUPPORTED)
    lifecycle.transition(hyp_id, HypothesisStatus.ACCEPTED)

    # Verify
    hyp = lifecycle.get(hyp_id)
    assert hyp.status == HypothesisStatus.ACCEPTED
    assert len(hyp.supporting_evidence) == 2
    assert hyp.confidence == 1.0

    # Check history
    history = lifecycle.get_history(hyp_id)
    assert len(history) == 4  # PROPOSED, TESTING, SUPPORTED, ACCEPTED
