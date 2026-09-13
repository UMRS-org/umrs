"""End-to-End Demonstration Test for M3.5.

This test exercises the complete Interactive Scientific Knowledge
Verification workflow:

1. Create project
2. Upload papers, code, notebooks, datasets
3. run_project_workflow()
4. Knowledge automatically built
5. User opens Knowledge Dashboard
6. User browses Knowledge Graph
7. User asks "What did you learn?"
8. Verification Chat explains
9. User corrects one mistake
10. Knowledge updated
11. Coverage updated
12. Research Readiness updated
13. User approves
14. Ready for Autonomous Research (M4)

If this test passes, the Interactive Scientific Knowledge Verification
milestone is complete.
"""

from __future__ import annotations

import pytest
from umrs.knowledge.knowledge_correction_controller import (
    KnowledgeCorrectionController,
)
from umrs.knowledge.knowledge_dashboard_controller import (
    KnowledgeDashboardController,
)
from umrs.knowledge.research_readiness_controller import (
    ResearchReadinessController,
)
from umrs.knowledge.verification_chat_controller import (
    VerificationChatController,
)
from umrs.knowledge.canonical_graph_view import in_memory_view
from umrs.knowledge.graph_types import RelationType
from umrs.knowledge_model import (
    InMemoryKnowledgeModelStore,
    KnowledgeObjectType,
)
from umrs.scientific_memory import InMemoryScientificMemoryStore, MemoryEntry


def _build_knowledge_base():
    """Build a knowledge base simulating post-ingestion state."""
    km = InMemoryKnowledgeModelStore()
    kg = in_memory_view()
    sm = InMemoryScientificMemoryStore()

    # Create scientific objects (simulating upload + ingestion)
    paper1_id = km.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "Attention Is All You Need", "authors": ["Vaswani et al."]},
        metadata={"doi": "10.48550/arXiv.1706.03762"},
    )
    paper2_id = km.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.PAPER,
        content={"title": "BERT: Pre-training of Deep Bidirectional Transformers"},
        metadata={"doi": "10.18653/v1/N19-1423"},
    )
    code_id = km.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.CODE,
        content={"title": "transformer_model.py", "language": "python"},
    )
    notebook_id = km.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.NOTEBOOK,
        content={"title": "transformer_experiment.ipynb", "cell_count": 15},
    )
    dataset_id = km.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.DATASET,
        content={"title": "QM9 Molecular Dataset", "size_mb": 120},
    )
    hypothesis_id = km.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.HYPOTHESIS,
        content={"title": "Transformers can predict molecular properties", "confidence": 0.7},
    )
    experiment_id = km.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.EXPERIMENT,
        content={"title": "Transformer on QM9", "status": "completed"},
    )
    finding_id = km.create_object(
        project_id="proj-001",
        object_type=KnowledgeObjectType.FINDING,
        content={"title": "Transformer achieves 95% accuracy on QM9"},
    )

    # Add graph nodes
    kg.add_node(node_id=paper1_id, node_type="paper", title="Attention Is All You Need")
    kg.add_node(node_id=paper2_id, node_type="paper", title="BERT")
    kg.add_node(node_id=code_id, node_type="code", title="transformer_model.py")
    kg.add_node(node_id=notebook_id, node_type="notebook", title="transformer_experiment.ipynb")
    kg.add_node(node_id=dataset_id, node_type="dataset", title="QM9 Molecular Dataset")
    kg.add_node(
        node_id=hypothesis_id,
        node_type="hypothesis",
        title="Transformers can predict molecular properties",
    )
    kg.add_node(node_id=experiment_id, node_type="experiment", title="Transformer on QM9")
    kg.add_node(node_id=finding_id, node_type="finding", title="Transformer achieves 95% accuracy")

    # Add typed edges (relationships)
    kg.add_edge(
        source_id=paper1_id, target_id=paper2_id, relation_type=RelationType.CITES, confidence=0.9
    )
    kg.add_edge(
        source_id=paper1_id,
        target_id=code_id,
        relation_type=RelationType.IMPLEMENTS,
        confidence=0.85,
    )
    kg.add_edge(
        source_id=paper1_id,
        target_id=hypothesis_id,
        relation_type=RelationType.DESCRIBES,
        confidence=0.8,
    )
    kg.add_edge(
        source_id=code_id,
        target_id=notebook_id,
        relation_type=RelationType.RELATED_TO,
        confidence=0.7,
    )
    kg.add_edge(
        source_id=notebook_id,
        target_id=dataset_id,
        relation_type=RelationType.USES,
        confidence=0.95,
    )
    kg.add_edge(
        source_id=notebook_id,
        target_id=experiment_id,
        relation_type=RelationType.PRODUCES,
        confidence=0.85,
    )
    kg.add_edge(
        source_id=experiment_id,
        target_id=finding_id,
        relation_type=RelationType.PRODUCES,
        confidence=0.9,
    )
    kg.add_edge(
        source_id=finding_id,
        target_id=hypothesis_id,
        relation_type=RelationType.SUPPORTS,
        confidence=0.95,
    )

    # Add scientific memory entries
    sm.add(
        MemoryEntry(
            project_id="proj-001",
            content="Transformer model was trained on QM9 dataset with 95% accuracy",
            summary="Training result",
        )
    )
    sm.add(
        MemoryEntry(
            project_id="proj-001",
            content="The transformer architecture uses self-attention mechanism",
            summary="Architecture description",
        )
    )

    return km, kg, sm


@pytest.mark.asyncio
async def test_end_to_end_knowledge_verification():
    """Test the complete Interactive Scientific Knowledge Verification workflow.

    Steps:
    1. Build knowledge base (simulating upload + ingestion)
    2. User opens Knowledge Dashboard
    3. User browses Knowledge Graph
    4. User asks "What did you learn?"
    5. Verification Chat explains with evidence
    6. User corrects one mistake
    7. Knowledge updated
    8. Coverage updated
    9. Research Readiness evaluated
    10. User approves
    11. Ready for Autonomous Research
    """

    # ── Step 1: Build knowledge base ────────────────────────────────────
    km, kg, sm = _build_knowledge_base()

    # ── Step 2: User opens Knowledge Dashboard ─────────────────────────
    dashboard_controller = KnowledgeDashboardController(
        knowledge_model=km,
        knowledge_graph=kg,
        scientific_memory=sm,
    )
    summary = dashboard_controller.get_knowledge_summary()

    assert summary.paper_count == 2, f"Expected 2 papers, got {summary.paper_count}"
    assert summary.code_count == 1, f"Expected 1 code, got {summary.code_count}"
    assert summary.notebook_count == 1, f"Expected 1 notebook, got {summary.notebook_count}"
    assert summary.dataset_count == 1, f"Expected 1 dataset, got {summary.dataset_count}"
    assert summary.hypothesis_count == 1, f"Expected 1 hypothesis, got {summary.hypothesis_count}"
    assert summary.experiment_count == 1, f"Expected 1 experiment, got {summary.experiment_count}"
    assert summary.finding_count == 1, f"Expected 1 finding, got {summary.finding_count}"
    assert summary.total_objects == 8, f"Expected 8 total objects, got {summary.total_objects}"
    assert summary.graph_node_count == 8, f"Expected 8 graph nodes, got {summary.graph_node_count}"
    assert summary.memory_entry_count == 2, (
        f"Expected 2 memory entries, got {summary.memory_entry_count}"
    )
    assert summary.coverage_score > 0.0, "Coverage should be > 0"

    # ── Step 3: User browses Knowledge Graph ────────────────────────────
    objects = km.list_objects(limit=100)
    paper1_id = [o for o in objects if o.content.get("title") == "Attention Is All You Need"][
        0
    ].object_id

    # Get object detail (should include related objects and provenance)
    detail = dashboard_controller.get_object_detail(paper1_id)
    assert detail is not None
    assert detail.object_id == paper1_id
    assert len(detail.related_object_ids) > 0, "Paper should have related objects"
    assert len(detail.provenance) > 0, "Paper should have provenance (graph edges)"

    # Navigate to neighbors (TUI uses existing KnowledgeGraph, never implements traversal)
    neighbors = dashboard_controller.get_graph_neighbors(paper1_id)
    assert len(neighbors) > 0, "Paper should have neighbors in the knowledge graph"

    # ── Step 4: User asks "What did you learn?" ────────────────────────
    verification_controller = VerificationChatController()
    answer = await verification_controller.ask(
        question="What did you learn from the uploaded papers?",
        anchor_object_id=paper1_id,
    )

    assert answer is not None
    assert answer.question == "What did you learn from the uploaded papers?"
    assert answer.answer != "", "Verification chat should return an answer"
    assert answer.confidence >= 0.0, "Confidence should be >= 0"
    assert answer.confidence <= 1.0, "Confidence should be <= 1"

    # ── Step 5: Verification Chat explains with evidence ──────────────
    # The answer should include evidence IDs and reasoning steps
    assert len(answer.reasoning_steps) > 0, "Answer should have reasoning steps"

    # ── Step 6: User corrects one mistake ─────────────────────────────
    correction_controller = KnowledgeCorrectionController(
        knowledge_graph=kg,
    )

    # User marks paper2 as unrelated to paper1
    objects = km.list_objects(limit=100)
    paper2_id = [
        o
        for o in objects
        if o.content.get("title") == "BERT: Pre-training of Deep Bidirectional Transformers"
    ][0].object_id

    correction = correction_controller.mark_unrelated(
        object_id_1=paper1_id,
        object_id_2=paper2_id,
        corrected_by="user-001",
        reason="These papers are not directly related to our research",
    )

    assert correction.correction_type == "unrelated"
    assert correction.corrected_by == "user-001"
    assert correction.reason != ""
    assert len(correction.affected_object_ids) == 2

    # ── Step 7: Knowledge updated ──────────────────────────────────────
    # Correction is recorded with provenance
    history = correction_controller.get_correction_history()
    assert len(history) == 1, "Correction history should have 1 entry"

    # ── Step 8: Coverage updated ───────────────────────────────────────
    summary_after = dashboard_controller.get_knowledge_summary()
    assert summary_after.total_objects == summary.total_objects, "Object count should not change"

    # ── Step 9: Research Readiness evaluated ──────────────────────────
    readiness_controller = ResearchReadinessController(
        knowledge_model=km,
        knowledge_graph=kg,
    )
    readiness_report = readiness_controller.evaluate_readiness()

    assert readiness_report.readiness_level in ["not_ready", "partially_ready", "ready", "blocked"]
    assert readiness_report.knowledge_completeness >= 0.0
    assert readiness_report.knowledge_completeness <= 1.0
    assert readiness_report.recommendation != ""

    # ── Step 10: User approves ─────────────────────────────────────────
    approval = readiness_controller.approve(
        user_id="user-001",
        readiness_report=readiness_report,
    )

    assert approval.approved is True
    assert approval.approved_by == "user-001"
    assert approval.approved_at != ""
    assert approval.readiness_level == readiness_report.readiness_level

    # ── Step 11: Ready for Autonomous Research ─────────────────────────
    assert readiness_controller.is_approved() is True, "Autonomous research should be approved"

    # Get the approval record
    approval_record = readiness_controller.get_approval()
    assert approval_record is not None
    assert approval_record.approved is True


def test_rejection_blocks_autonomous_research():
    """Test that rejection blocks autonomous research."""
    km = InMemoryKnowledgeModelStore()
    kg = in_memory_view()

    readiness_controller = ResearchReadinessController(
        knowledge_model=km,
        knowledge_graph=kg,
    )

    # User rejects
    rejection = readiness_controller.reject(
        user_id="user-001",
        reason="Knowledge is incomplete",
    )

    assert rejection.approved is False
    assert readiness_controller.is_approved() is False


def test_correction_provenance():
    """Test that corrections become provenance."""
    kg = in_memory_view()
    kg.add_node(node_id="paper-001", node_type="paper", title="Paper 1")
    kg.add_node(node_id="paper-002", node_type="paper", title="Paper 2")

    controller = KnowledgeCorrectionController(knowledge_graph=kg)

    # Add a relationship (user correction)
    record = controller.add_relationship(
        source_id="paper-001",
        target_id="paper-002",
        relation_type="cites",
        corrected_by="user-001",
        reason="Paper 1 cites Paper 2",
        confidence=0.95,
    )

    assert record.correction_type == "relates"
    assert record.corrected_by == "user-001"
    assert record.details["relation_type"] == "cites"
    assert record.details["confidence"] == 0.95

    # Verify the edge was added
    outgoing = kg.get_outgoing("paper-001")
    assert len(outgoing) == 1
    assert outgoing[0].target_id == "paper-002"
    assert outgoing[0].relation_type == RelationType.CITES


def test_readiness_levels():
    """Test that readiness levels are correctly determined."""
    km = InMemoryKnowledgeModelStore()
    kg = in_memory_view()

    # Empty knowledge base → BLOCKED
    controller = ResearchReadinessController(knowledge_model=km, knowledge_graph=kg)
    report = controller.evaluate_readiness()
    assert report.readiness_level == "blocked" or report.readiness_level == "not_ready"

    # Add some objects → should improve readiness
    km.create_object(
        project_id="p1", object_type=KnowledgeObjectType.PAPER, content={"title": "Paper 1"}
    )
    km.create_object(
        project_id="p1", object_type=KnowledgeObjectType.HYPOTHESIS, content={"title": "H1"}
    )
    km.create_object(
        project_id="p1", object_type=KnowledgeObjectType.EXPERIMENT, content={"title": "E1"}
    )
    km.create_object(
        project_id="p1", object_type=KnowledgeObjectType.FINDING, content={"title": "F1"}
    )
    km.create_object(
        project_id="p1", object_type=KnowledgeObjectType.EVIDENCE, content={"title": "EV1"}
    )

    report = controller.evaluate_readiness()
    assert report.knowledge_completeness > 0.0
