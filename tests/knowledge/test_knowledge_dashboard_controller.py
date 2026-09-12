"""Tests for M3.5 Knowledge Dashboard Controller."""

from __future__ import annotations

from umrs.knowledge.knowledge_dashboard_controller import (
    KnowledgeDashboardController,
    KnowledgeSummary,
    ObjectDetail,
    ObjectSummary,
)
from umrs.knowledge.canonical_graph_view import default_view
from umrs.knowledge.graph_types import RelationType
from umrs.knowledge_model import (
    InMemoryKnowledgeModelStore,
    KnowledgeObjectType,
)
from umrs.scientific_memory import InMemoryScientificMemoryStore, MemoryEntry


def _setup():
    km = InMemoryKnowledgeModelStore()
    kg = default_view(store=None)
    sm = InMemoryScientificMemoryStore()

    # Create some objects
    km.create_object(project_id="p1", object_type=KnowledgeObjectType.PAPER, content={"title": "Paper 1"})
    km.create_object(project_id="p1", object_type=KnowledgeObjectType.PAPER, content={"title": "Paper 2"})
    km.create_object(project_id="p1", object_type=KnowledgeObjectType.CODE, content={"title": "model.py"})
    km.create_object(project_id="p1", object_type=KnowledgeObjectType.HYPOTHESIS, content={"title": "H1"})

    # Add graph nodes and edges
    kg.add_node(node_id="paper-001", node_type="paper", title="Paper 1")
    kg.add_node(node_id="paper-002", node_type="paper", title="Paper 2")
    kg.add_node(node_id="code-001", node_type="code", title="model.py")
    kg.add_edge(source_id="paper-001", target_id="paper-002", relation_type=RelationType.CITES, confidence=0.9)
    kg.add_edge(source_id="paper-001", target_id="code-001", relation_type=RelationType.IMPLEMENTS, confidence=0.85)

    # Add memory entries
    sm.add(MemoryEntry(project_id="p1", content="Test memory entry"))

    return km, kg, sm


def test_knowledge_summary_defaults():
    s = KnowledgeSummary()
    assert s.paper_count == 0
    assert s.coverage_score == 0.0


def test_get_knowledge_summary():
    km, kg, sm = _setup()
    controller = KnowledgeDashboardController(
        knowledge_model=km, knowledge_graph=kg, scientific_memory=sm
    )
    summary = controller.get_knowledge_summary()

    assert summary.paper_count == 2
    assert summary.code_count == 1
    assert summary.hypothesis_count == 1
    assert summary.total_objects == 4
    assert summary.graph_node_count == 3
    assert summary.coverage_score > 0.0
    assert summary.memory_entry_count == 1


def test_list_objects():
    km, kg, sm = _setup()
    controller = KnowledgeDashboardController(knowledge_model=km)
    papers = controller.list_objects(object_type="paper")

    assert len(papers) == 2
    assert isinstance(papers[0], ObjectSummary)
    assert papers[0].object_type == "paper"


def test_list_objects_all_types():
    km, kg, sm = _setup()
    controller = KnowledgeDashboardController(knowledge_model=km)
    all_objects = controller.list_objects()

    assert len(all_objects) == 4


def test_list_objects_invalid_type():
    km, kg, sm = _setup()
    controller = KnowledgeDashboardController(knowledge_model=km)
    result = controller.list_objects(object_type="nonexistent")
    assert result == []


def test_get_object_detail():
    km, kg, sm = _setup()
    objects = km.list_objects(limit=10)
    first_id = objects[0].object_id

    controller = KnowledgeDashboardController(
        knowledge_model=km, knowledge_graph=kg
    )
    detail = controller.get_object_detail(first_id)

    assert detail is not None
    assert isinstance(detail, ObjectDetail)
    assert detail.object_id == first_id


def test_get_object_detail_not_found():
    km, kg, sm = _setup()
    controller = KnowledgeDashboardController(knowledge_model=km)
    detail = controller.get_object_detail("nonexistent")
    assert detail is None


def test_get_graph_neighbors():
    km, kg, sm = _setup()
    controller = KnowledgeDashboardController(
        knowledge_model=km, knowledge_graph=kg
    )
    neighbors = controller.get_graph_neighbors("paper-001")

    assert "paper-002" in neighbors
    assert "code-001" in neighbors
