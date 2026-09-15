"""Tests for the Scientific Reasoning Engine (M3 Phase 3)."""

from __future__ import annotations

from umrs.scientific_reasoning_engine import (
    ReasoningResult,
    ScientificReasoningEngine,
)


def test_reasoning_result_defaults():
    """Test ReasoningResult defaults."""
    result = ReasoningResult()

    assert result.reasoning_id != ""  # Auto-generated
    assert result.query == ""
    assert result.answer == ""
    assert result.evidence_ids == []
    assert result.graph_paths == []
    assert result.confidence == 0.0
    assert result.reasoning_context is None
    assert result.reasoning_steps == []
    assert result.timestamp != ""  # Auto-generated


def test_reasoning_result_to_persistent_dict():
    """Test that to_persistent_dict excludes the ephemeral answer."""
    result = ReasoningResult(
        query="What evidence supports this?",
        answer="The evidence shows...",
        evidence_ids=["paper-001", "code-001"],
        confidence=0.85,
    )

    data = result.to_persistent_dict()

    # Persistent data should be present
    assert data["query"] == "What evidence supports this?"
    assert data["evidence_ids"] == ["paper-001", "code-001"]
    assert data["confidence"] == 0.85

    # Ephemeral answer should NOT be in persistent dict
    assert "answer" not in data


def test_reasoning_result_to_dict_includes_answer():
    """Test that to_dict includes the ephemeral answer."""
    result = ReasoningResult(
        query="What evidence supports this?",
        answer="The evidence shows...",
    )

    data = result.to_dict()

    assert data["answer"] == "The evidence shows..."


def test_scientific_reasoning_engine_init():
    """Test ScientificReasoningEngine initialization."""
    engine = ScientificReasoningEngine()
    assert engine is not None
    assert engine.navigator is not None


def test_scientific_reasoning_engine_reason_no_anchor():
    """Test reasoning with no anchor (empty graph)."""
    engine = ScientificReasoningEngine()
    result = engine.reason("What should we research next?")

    assert isinstance(result, ReasoningResult)
    assert result.query == "What should we research next?"
    assert result.confidence == 0.0
    assert "No anchor object found" in result.reasoning_steps[0]


def test_scientific_reasoning_engine_reason_with_anchor():
    """Test reasoning with an anchor object."""
    engine = ScientificReasoningEngine()
    result = engine.reason(
        "What should we research next?",
        anchor_object_id="paper-001",
    )

    assert isinstance(result, ReasoningResult)
    assert result.query == "What should we research next?"
    # Without graph data, reasoning still produces a result
    assert len(result.reasoning_steps) > 0


def test_scientific_reasoning_engine_reason_about_hypothesis():
    """Test reasoning about a specific hypothesis."""
    engine = ScientificReasoningEngine()
    result = engine.reason_about_hypothesis("hypothesis-001")

    assert isinstance(result, ReasoningResult)
    assert "hypothesis-001" in result.query
    assert len(result.reasoning_steps) > 0
    assert 0.0 <= result.confidence <= 1.0


def test_scientific_reasoning_engine_reason_about_project():
    """Test reasoning about the entire project."""
    engine = ScientificReasoningEngine()
    result = engine.reason_about_project("paper-001")

    assert isinstance(result, ReasoningResult)
    assert "Explain this project" in result.query
    assert len(result.reasoning_steps) > 0
    assert len(result.answer) > 0


def test_scientific_reasoning_engine_confidence_range():
    """Test that confidence is always in [0, 1]."""
    engine = ScientificReasoningEngine()

    result1 = engine.reason("test query 1")
    result2 = engine.reason("test query 2", anchor_object_id="paper-001")
    result3 = engine.reason_about_hypothesis("hyp-001")
    result4 = engine.reason_about_project("paper-001")

    for result in [result1, result2, result3, result4]:
        assert 0.0 <= result.confidence <= 1.0
