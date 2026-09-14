"""Knowledge linking — automatically connect related knowledge artifacts.

Phase 5 of the Scientific Knowledge Build-up roadmap.

This module implements the knowledge linking layer that:
  1. Automatically connects related knowledge (Paper → Implementation →
     Notebook → Experiment → Generated Figure → Result → Conclusion)
  2. Builds cross references between knowledge artifacts
  3. Creates a knowledge graph of typed relationships

The knowledge graph uses typed edges between scientific knowledge objects:
  - CITES: Paper CITES Paper
  - DESCRIBES: Paper DESCRIBES Algorithm
  - IMPLEMENTS: Implementation IMPLEMENTS Algorithm
  - USES: Experiment USES Dataset/Implementation
  - PRODUCES: Experiment/Notebook PRODUCES Observation/Figure
  - DERIVES_FROM: Conclusion DERIVES_FROM Observation
  - SUPPORTS: Evidence SUPPORTS Hypothesis/Conclusion
  - CONTRADICTS: Evidence CONTRADICTS Hypothesis/Conclusion
  - VERIFIES: Conclusion VERIFIES Hypothesis
  - REFUTES: Conclusion REFUTES Hypothesis
  - ANSWERS: Conclusion ANSWERS Question
  - ASSUMES: Experiment ASSUMES Assumption
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Relationship Types ──────────────────────────────────────────────────────


class RelationType(str, Enum):
    """Typed relationships between scientific knowledge objects."""

    # Citation relationships
    CITES = "cites"
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"

    # Derivation relationships
    DERIVES_FROM = "derives_from"
    PRODUCES = "produces"
    PRODUCED_BY = "produced_by"

    # Evidence relationships
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EVIDENCE_FOR = "evidence_for"
    EVIDENCE_AGAINST = "evidence_against"

    # Implementation relationships
    IMPLEMENTS = "implements"
    IMPLEMENTED_BY = "implemented_by"
    DESCRIBES = "describes"
    DESCRIBED_BY = "described_by"

    # Composition relationships
    PART_OF = "part_of"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    USES = "uses"

    # Research flow relationships
    ANSWERS = "answers"
    ANSWERED_BY = "answered_by"
    ASSUMES = "assumes"
    VERIFIES = "verifies"
    REFUTES = "refutes"

    # Related (generic)
    RELATED_TO = "related_to"


# ── Knowledge Graph Models ──────────────────────────────────────────────────


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    node_id: str  # The artifact's id
    node_type: str  # "paper" | "code" | "notebook" | "experiment" | ...
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "title": self.title,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph (typed relationship)."""

    source_id: str  # Source node id
    target_id: str  # Target node id
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeGraph:
    """A knowledge graph of typed relationships between scientific artifacts."""

    nodes: dict[str, KnowledgeNode] = field(default_factory=dict)
    edges: list[KnowledgeEdge] = field(default_factory=list)
    # Index: source_id → list of edges
    _outgoing: dict[str, list[KnowledgeEdge]] = field(default_factory=dict)
    # Index: target_id → list of edges
    _incoming: dict[str, list[KnowledgeEdge]] = field(default_factory=dict)

    def add_node(self, node: KnowledgeNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: KnowledgeEdge) -> None:
        """Add an edge to the graph (if not already present)."""
        # Check for duplicate edge
        for existing in self.edges:
            if (
                existing.source_id == edge.source_id
                and existing.target_id == edge.target_id
                and existing.relation_type == edge.relation_type
            ):
                return  # Already exists

        self.edges.append(edge)
        self._outgoing.setdefault(edge.source_id, []).append(edge)
        self._incoming.setdefault(edge.target_id, []).append(edge)

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        """Get a node by id."""
        return self.nodes.get(node_id)

    def get_outgoing(self, node_id: str) -> list[KnowledgeEdge]:
        """Get all outgoing edges from a node."""
        return self._outgoing.get(node_id, [])

    def get_incoming(self, node_id: str) -> list[KnowledgeEdge]:
        """Get all incoming edges to a node."""
        return self._incoming.get(node_id, [])

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get all neighbor node ids (both outgoing and incoming)."""
        neighbors: set[str] = set()
        for edge in self.get_outgoing(node_id):
            neighbors.add(edge.target_id)
        for edge in self.get_incoming(node_id):
            neighbors.add(edge.source_id)
        return list(neighbors)

    def get_related(self, node_id: str, relation_type: RelationType) -> list[str]:
        """Get all node ids related to the given node with the given relation type."""
        related: list[str] = []
        for edge in self.get_outgoing(node_id):
            if edge.relation_type == relation_type:
                related.append(edge.target_id)
        for edge in self.get_incoming(node_id):
            if edge.relation_type == relation_type:
                related.append(edge.source_id)
        return related

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> list[str] | None:
        """Find a path between two nodes using BFS.

        Args:
            source_id: The source node id
            target_id: The target node id
            max_depth: Maximum search depth

        Returns:
            A list of node ids forming a path, or None if no path found.
        """
        if source_id == target_id:
            return [source_id]

        visited: set[str] = {source_id}
        queue: list[tuple[str, list[str]]] = [(source_id, [source_id])]

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            for neighbor in self.get_neighbors(current):
                if neighbor == target_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }


# ── Knowledge Linker ────────────────────────────────────────────────────────


class KnowledgeLinker:
    """Automatically connect related knowledge artifacts.

    Usage:
        linker = KnowledgeLinker()
        graph = linker.link(artifacts)
    """

    def __init__(self) -> None:
        # Patterns for extracting citations and references
        self._citation_patterns = [
            r"(?:cite|cited|citation)\s+(.+?)(?:\.|$)",
            r"\[(\d+)\]",  # Numeric citations [1], [2], etc.
            r"@(\w+)",  # BibTeX-style @citation_key
        ]
        # Patterns for extracting mentions of other artifacts
        self._mention_patterns = [
            r"(?:see|refer to|as described in)\s+(.+?)(?:\.|$)",
            r"(?:we use|using|we employ)\s+(.+?)(?:\.|$)",
            r"(?:based on|following|according to)\s+(.+?)(?:\.|$)",
        ]

    def link(
        self,
        artifacts: list[dict[str, Any]],
    ) -> KnowledgeGraph:
        """Link a list of artifacts into a knowledge graph.

        Args:
            artifacts: A list of artifact dictionaries with at least:
                - artifact_id: The artifact's id
                - artifact_type: The artifact's type
                - filename: The artifact's filename
                - extracted_knowledge: The extracted knowledge (optional)

        Returns:
            A ``KnowledgeGraph`` with nodes and edges representing the
            relationships between the artifacts.
        """
        graph = KnowledgeGraph()

        # Add all artifacts as nodes
        for artifact in artifacts:
            node = KnowledgeNode(
                node_id=artifact["artifact_id"],
                node_type=artifact["artifact_type"],
                title=artifact.get("filename", ""),
                metadata=artifact.get("metadata", {}),
            )
            graph.add_node(node)

        # Link artifacts based on extracted knowledge
        for i, artifact in enumerate(artifacts):
            knowledge = artifact.get("extracted_knowledge")
            if knowledge:
                self._link_from_knowledge(
                    graph,
                    artifact,
                    knowledge,
                    artifacts,
                )

            # Also link based on filename/content matching
            self._link_by_filename(graph, artifact, artifacts)

        return graph

    def _link_from_knowledge(
        self,
        graph: KnowledgeGraph,
        artifact: dict[str, Any],
        knowledge: Any,
        all_artifacts: list[dict[str, Any]],
    ) -> None:
        """Link an artifact to other artifacts based on extracted knowledge."""
        artifact_id = artifact["artifact_id"]
        artifact_type = artifact["artifact_type"]

        # Normalize knowledge to a dict-like accessor
        if isinstance(knowledge, dict):

            def get_field(name, default=None):
                return knowledge.get(name, default)
        else:

            def get_field(name, default=None):
                return getattr(knowledge, name, default)

        # Collect all name-like fields from extracted knowledge
        name_fields = [
            "datasets",
            "algorithms",
            "apis",
            "modules",
            "dependencies",
            "kernels",
            "keywords",
        ]
        all_names: list[str] = []
        for field_name in name_fields:
            values = get_field(field_name, [])
            if values:
                for v in values:
                    if isinstance(v, str):
                        all_names.append(v)
                    elif isinstance(v, list):
                        all_names.extend(x for x in v if isinstance(x, str))

        # Link to any other artifact whose filename matches one of our names
        for other in all_artifacts:
            if other["artifact_id"] == artifact_id:
                continue
            other_filename = other.get("filename", "")
            if not other_filename:
                continue
            for name in all_names:
                if self._name_matches(name, other_filename):
                    # Determine relation type based on artifact types
                    relation = self._infer_relation(
                        artifact_type,
                        other.get("artifact_type", ""),
                    )
                    graph.add_edge(
                        KnowledgeEdge(
                            source_id=artifact_id,
                            target_id=other["artifact_id"],
                            relation_type=relation,
                            confidence=0.7,
                        )
                    )
                    break  # One match per other artifact is enough

    def _infer_relation(
        self,
        source_type: str,
        target_type: str,
    ) -> RelationType:
        """Infer the relation type based on the artifact types."""
        if source_type == "paper" and target_type == "code":
            return RelationType.DESCRIBES
        if source_type == "code" and target_type == "paper":
            return RelationType.DESCRIBED_BY
        if source_type == "paper" and target_type == "notebook":
            return RelationType.REFERENCES
        if source_type == "notebook" and target_type == "code":
            return RelationType.USES
        if source_type == "notebook" and target_type == "paper":
            return RelationType.REFERENCES
        return RelationType.RELATED_TO

    def _link_by_filename(
        self,
        graph: KnowledgeGraph,
        artifact: dict[str, Any],
        all_artifacts: list[dict[str, Any]],
    ) -> None:
        """Link artifacts by filename similarity."""
        artifact_id = artifact["artifact_id"]
        filename = artifact.get("filename", "")
        if not filename:
            return

        # Extract the stem (filename without extension)
        import os

        stem = os.path.splitext(os.path.basename(filename))[0]

        # Find artifacts with similar filenames
        for other in all_artifacts:
            if other["artifact_id"] == artifact_id:
                continue
            other_filename = other.get("filename", "")
            if not other_filename:
                continue
            other_stem = os.path.splitext(os.path.basename(other_filename))[0]

            # If stems match exactly, link with high confidence
            if stem == other_stem and stem:
                graph.add_edge(
                    KnowledgeEdge(
                        source_id=artifact_id,
                        target_id=other["artifact_id"],
                        relation_type=RelationType.RELATED_TO,
                        confidence=0.9,
                    )
                )

    def _name_matches(self, name1: str, name2: str) -> bool:
        """Check if two names match (case-insensitive, partial match)."""
        if not name1 or not name2:
            return False
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        # Check for exact match or substring match
        return n1 == n2 or n1 in n2 or n2 in n1


# ── Convenience Function ────────────────────────────────────────────────────


def link_knowledge(
    artifacts: list[dict[str, Any]],
) -> KnowledgeGraph:
    """Link a list of artifacts into a knowledge graph (convenience function).

    Args:
        artifacts: A list of artifact dictionaries

    Returns:
        A ``KnowledgeGraph`` with nodes and edges.
    """
    linker = KnowledgeLinker()
    return linker.link(artifacts)
