"""R4 Knowledge Controllers — bridge between TUI and UMRS.

M3.5 Phase 1-2: Project Knowledge Dashboard and Knowledge Explorer.

These controllers are called by the R4 TUI to:
1. Get project knowledge summary (dashboard)
2. List and browse scientific objects (explorer)
3. Get object details (metadata, provenance, related objects)

The TUI never accesses UMRS directly. All interactions go through
these controllers.

Architecture:
  TUI → R4 Knowledge Controllers → UMRS APIs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from umrs.knowledge.canonical_graph_view import CanonicalGraphView, default_view
from umrs.knowledge_graph_api import InMemoryKnowledgeGraph
from umrs.knowledge_model import (
    InMemoryKnowledgeModelStore,
    KnowledgeObject,
    KnowledgeObjectType,
)
from umrs.scientific_memory import InMemoryScientificMemoryStore

__all__ = [
    "KnowledgeSummary",
    "ObjectSummary",
    "ObjectDetail",
    "KnowledgeDashboardController",
]


# ── Response Models ───────────────────────────────────────────────────────


@dataclass
class KnowledgeSummary:
    """Summary of project knowledge for the dashboard."""

    # Object counts by type
    paper_count: int = 0
    code_count: int = 0
    notebook_count: int = 0
    dataset_count: int = 0
    hypothesis_count: int = 0
    experiment_count: int = 0
    finding_count: int = 0
    figure_count: int = 0
    equation_count: int = 0
    evidence_count: int = 0

    # Graph statistics
    graph_node_count: int = 0
    graph_edge_count: int = 0

    # Memory statistics
    memory_entry_count: int = 0

    # Coverage
    total_objects: int = 0
    coverage_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_count": self.paper_count,
            "code_count": self.code_count,
            "notebook_count": self.notebook_count,
            "dataset_count": self.dataset_count,
            "hypothesis_count": self.hypothesis_count,
            "experiment_count": self.experiment_count,
            "finding_count": self.finding_count,
            "figure_count": self.figure_count,
            "equation_count": self.equation_count,
            "evidence_count": self.evidence_count,
            "graph_node_count": self.graph_node_count,
            "graph_edge_count": self.graph_edge_count,
            "memory_entry_count": self.memory_entry_count,
            "total_objects": self.total_objects,
            "coverage_score": self.coverage_score,
        }


@dataclass
class ObjectSummary:
    """Summary of a single scientific object for list display."""

    object_id: str = ""
    object_type: str = ""
    title: str = ""
    status: str = "active"
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "title": self.title,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class ObjectDetail:
    """Detailed view of a scientific object."""

    object_id: str = ""
    object_type: str = ""
    title: str = ""
    content: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "active"
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""

    # Related objects (via knowledge graph)
    related_object_ids: list[str] = field(default_factory=list)

    # Provenance
    provenance: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "title": self.title,
            "content": self.content,
            "metadata": self.metadata,
            "status": self.status,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "related_object_ids": self.related_object_ids,
            "provenance": self.provenance,
        }


# ── Controller ────────────────────────────────────────────────────────────


class KnowledgeDashboardController:
    """Controller for the Knowledge Dashboard and Explorer.

    Bridges between the TUI and UMRS APIs.

    Usage:
        controller = KnowledgeDashboardController(
            knowledge_model=knowledge_model,
            knowledge_graph=knowledge_graph,
            scientific_memory=scientific_memory,
        )
        summary = controller.get_knowledge_summary()
        objects = controller.list_objects(object_type="paper")
        detail = controller.get_object_detail(object_id="obj-001")
    """

    def __init__(
        self,
        *,
        knowledge_model: InMemoryKnowledgeModelStore | None = None,
        knowledge_graph: InMemoryKnowledgeGraph | CanonicalGraphView | None = None,
        scientific_memory: InMemoryScientificMemoryStore | None = None,
    ) -> None:
        self._knowledge_model = knowledge_model or InMemoryKnowledgeModelStore()
        self._knowledge_graph = knowledge_graph or default_view(store=None)
        self._scientific_memory = scientific_memory or InMemoryScientificMemoryStore()

    def get_knowledge_summary(self) -> KnowledgeSummary:
        """Get a summary of project knowledge for the dashboard.

        Returns:
            KnowledgeSummary with object counts, graph stats, and coverage.
        """
        summary = KnowledgeSummary()

        # Count objects by type
        for obj in self._knowledge_model.list_objects(limit=10000):
            self._bump_type_count(summary, obj.object_type)
            summary.total_objects += 1

        # Graph statistics
        all_nodes = self._knowledge_graph.query()
        summary.graph_node_count = len(all_nodes)

        # Count edges
        edge_count = 0
        for node in all_nodes:
            edge_count += len(self._knowledge_graph.get_outgoing(node.node_id))
        summary.graph_edge_count = edge_count

        # Memory statistics
        summary.memory_entry_count = len(self._scientific_memory.list_entries(limit=10000))

        summary.coverage_score = self._coverage_score(summary)
        return summary

    @staticmethod
    def _bump_type_count(summary: KnowledgeSummary, object_type: KnowledgeObjectType) -> None:
        """Increment the per-type count bucket (extracted for the gate)."""
        bump_map = {
            KnowledgeObjectType.PAPER: "paper_count",
            KnowledgeObjectType.CODE: "code_count",
            KnowledgeObjectType.NOTEBOOK: "notebook_count",
            KnowledgeObjectType.DATASET: "dataset_count",
            KnowledgeObjectType.HYPOTHESIS: "hypothesis_count",
            KnowledgeObjectType.EXPERIMENT: "experiment_count",
            KnowledgeObjectType.FINDING: "finding_count",
            KnowledgeObjectType.FIGURE: "figure_count",
            KnowledgeObjectType.EQUATION: "equation_count",
            KnowledgeObjectType.EVIDENCE: "evidence_count",
        }
        field = bump_map.get(object_type)
        if field:
            setattr(summary, field, getattr(summary, field) + 1)

    @staticmethod
    def _coverage_score(summary: KnowledgeSummary) -> float:
        """Coverage heuristic: presence-fraction across the five main kinds."""
        if summary.total_objects <= 0:
            return 0.0
        presence = sum(
            [
                summary.paper_count > 0,
                summary.code_count > 0,
                summary.hypothesis_count > 0,
                summary.experiment_count > 0,
                summary.finding_count > 0,
            ]
        )
        return presence / 5.0

    def list_objects(
        self,
        *,
        object_type: str | None = None,
        limit: int = 100,
    ) -> list[ObjectSummary]:
        """List scientific objects with optional type filter.

        Args:
            object_type: Filter by object type (paper, code, etc.)
            limit: Maximum number of objects to return

        Returns:
            List of ObjectSummary for display in the explorer.
        """
        obj_type: KnowledgeObjectType | None = None
        if object_type:
            try:
                obj_type = KnowledgeObjectType(object_type)
            except ValueError:
                return []

        objects = self._knowledge_model.list_objects(
            object_type=obj_type,
            limit=limit,
        )

        return [
            ObjectSummary(
                object_id=obj.object_id,
                object_type=obj.object_type.value,
                title=obj.content.get("title", obj.object_id),
                status=obj.status.value,
                created_at=obj.created_at,
            )
            for obj in objects
        ]

    def get_object_detail(self, object_id: str) -> ObjectDetail | None:
        """Get detailed view of a scientific object.

        Args:
            object_id: The object ID

        Returns:
            ObjectDetail with content, metadata, related objects, and provenance.
            None if the object is not found.
        """
        obj = self._knowledge_model.get_object(object_id)
        if not obj:
            return None

        # Get related objects via knowledge graph
        related_ids: list[str] = []
        for edge in self._knowledge_graph.get_outgoing(object_id):
            if edge.target_id not in related_ids:
                related_ids.append(edge.target_id)
        for edge in self._knowledge_graph.get_incoming(object_id):
            if edge.source_id not in related_ids:
                related_ids.append(edge.source_id)

        # Get provenance (edges to/from this object)
        provenance: list[dict[str, Any]] = []
        for edge in self._knowledge_graph.get_outgoing(object_id):
            provenance.append({
                "direction": "outgoing",
                "target_id": edge.target_id,
                "relation_type": edge.relation_type.value,
                "confidence": edge.confidence,
            })
        for edge in self._knowledge_graph.get_incoming(object_id):
            provenance.append({
                "direction": "incoming",
                "source_id": edge.source_id,
                "relation_type": edge.relation_type.value,
                "confidence": edge.confidence,
            })

        return ObjectDetail(
            object_id=obj.object_id,
            object_type=obj.object_type.value,
            title=obj.content.get("title", obj.object_id),
            content=obj.content,
            metadata=obj.metadata,
            status=obj.status.value,
            version=obj.version,
            created_at=obj.created_at,
            updated_at=obj.updated_at,
            created_by=obj.created_by,
            related_object_ids=related_ids,
            provenance=provenance,
        )

    def get_graph_neighbors(self, object_id: str) -> list[str]:
        """Get all neighbor object IDs for a given object.

        This uses the existing GraphNavigator/KnowledgeGraph.
        The TUI never implements graph traversal itself.

        Args:
            object_id: The object ID

        Returns:
            List of neighbor object IDs
        """
        return self._knowledge_graph.get_neighbors(object_id)
