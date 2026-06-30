"""Knowledge graph API — typed relationship graph between scientific objects.

Phase 1, Step 2.3 of the RC1 migration plan.

UMRS owns the knowledge graph. R4 consumes this API.

This module provides a typed relationship graph for scientific knowledge
objects. The graph supports:

  - Typed edges (CITES, IMPLEMENTS, USES, REFERENCES, SUPPORTS, CONTRADICTS,
    DERIVES_FROM, PRODUCES, ANSWERS, VERIFIES, REFUTES, ASSUMES, RELATED_TO)
  - Graph traversal (BFS, shortest path)
  - Neighbor discovery
  - Relationship queries
  - Edge deduplication
  - Edge confidence scores

Note: This is a new API-level module (`knowledge_graph_api`) that provides
the public interface for the knowledge graph. The existing `knowledge_graph/`
module contains the skeletal implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

__all__ = [
    "RelationType",
    "KnowledgeNode",
    "KnowledgeEdge",
    "KnowledgeGraphAPI",
    "InMemoryKnowledgeGraph",
]


# ── Relation Types ─────────────────────────────────────────────────────────


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


# ── Knowledge Node ─────────────────────────────────────────────────────────


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph.

    Attributes:
        node_id: Unique identifier for the node (typically an object_id)
        node_type: The type of node (paper, code, hypothesis, etc.)
        title: Human-readable title for the node
        metadata: Additional metadata
    """

    node_id: str = ""
    node_type: str = ""
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "title": self.title,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeNode":
        """Create from dictionary."""
        return cls(
            node_id=data.get("node_id", ""),
            node_type=data.get("node_type", ""),
            title=data.get("title", ""),
            metadata=data.get("metadata", {}),
        )


# ── Knowledge Edge ────────────────────────────────────────────────────────


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph (typed relationship).

    Attributes:
        edge_id: Unique identifier for the edge
        source_id: Source node ID
        target_id: Target node ID
        relation_type: Type of relationship
        weight: Edge weight (default 1.0)
        confidence: Confidence score (0.0–1.0)
        metadata: Additional metadata
        created_at: When the edge was created
    """

    edge_id: str = ""
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.RELATED_TO
    weight: float = 1.0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.edge_id:
            self.edge_id = f"edge-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeEdge":
        """Create from dictionary."""
        relation_type_raw = data.get("relation_type", "related_to")
        relation_type = (
            RelationType(relation_type_raw)
            if isinstance(relation_type_raw, str)
            else relation_type_raw
        )
        return cls(
            edge_id=data.get("edge_id", ""),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relation_type=relation_type,
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
        )


# ── Knowledge Graph API Protocol ──────────────────────────────────────────


class KnowledgeGraphAPI:
    """Protocol for the knowledge graph API.

    UMRS provides an implementation of this API. R4 consumes it.
    """

    def add_node(self, *, node_id: str, node_type: str, title: str = "", metadata: dict | None = None) -> None:
        raise NotImplementedError

    def add_edge(self, *, source_id: str, target_id: str, relation_type: RelationType, weight: float = 1.0, confidence: float = 1.0, metadata: dict | None = None) -> str:
        raise NotImplementedError

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        raise NotImplementedError

    def get_outgoing(self, node_id: str) -> list[KnowledgeEdge]:
        raise NotImplementedError

    def get_incoming(self, node_id: str) -> list[KnowledgeEdge]:
        raise NotImplementedError

    def get_neighbors(self, node_id: str) -> list[str]:
        raise NotImplementedError

    def get_related(self, node_id: str, relation_type: RelationType) -> list[str]:
        raise NotImplementedError

    def find_path(self, source_id: str, target_id: str, max_depth: int = 5) -> list[str] | None:
        raise NotImplementedError

    def query(self, *, node_type: str | None = None, relation_type: RelationType | None = None) -> list[KnowledgeNode]:
        raise NotImplementedError


# ── In-Memory Implementation ───────────────────────────────────────────────


class InMemoryKnowledgeGraph(KnowledgeGraphAPI):
    """In-memory implementation of the knowledge graph API.

    This is the reference implementation. The Postgres-backed implementation
    will follow the same interface.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, KnowledgeNode] = {}
        self._edges: dict[str, KnowledgeEdge] = {}
        self._outgoing: dict[str, list[str]] = {}  # node_id -> [edge_id, ...]
        self._incoming: dict[str, list[str]] = {}  # node_id -> [edge_id, ...]

    def add_node(
        self,
        *,
        node_id: str,
        node_type: str,
        title: str = "",
        metadata: dict | None = None,
    ) -> None:
        """Add a node to the graph."""
        if node_id in self._nodes:
            # Update existing node
            node = self._nodes[node_id]
            node.title = title or node.title
            node.metadata.update(metadata or {})
        else:
            self._nodes[node_id] = KnowledgeNode(
                node_id=node_id,
                node_type=node_type,
                title=title,
                metadata=metadata or {},
            )
            self._outgoing[node_id] = []
            self._incoming[node_id] = []

    def add_edge(
        self,
        *,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        weight: float = 1.0,
        confidence: float = 1.0,
        metadata: dict | None = None,
    ) -> str:
        """Add an edge to the graph.

        Returns the edge_id. If an edge with the same source, target, and
        relation_type already exists, it is updated.
        """
        # Ensure both nodes exist
        if source_id not in self._nodes:
            self.add_node(node_id=source_id, node_type="unknown")
        if target_id not in self._nodes:
            self.add_node(node_id=target_id, node_type="unknown")

        # Check for existing edge (deduplication)
        for edge_id in self._outgoing.get(source_id, []):
            edge = self._edges[edge_id]
            if (
                edge.target_id == target_id
                and edge.relation_type == relation_type
            ):
                # Update existing edge
                edge.weight = weight
                edge.confidence = confidence
                edge.metadata.update(metadata or {})
                return edge.edge_id

        # Create new edge
        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            weight=weight,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._edges[edge.edge_id] = edge
        self._outgoing.setdefault(source_id, []).append(edge.edge_id)
        self._incoming.setdefault(target_id, []).append(edge.edge_id)
        return edge.edge_id

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def get_outgoing(self, node_id: str) -> list[KnowledgeEdge]:
        """Get all outgoing edges from a node."""
        edge_ids = self._outgoing.get(node_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_incoming(self, node_id: str) -> list[KnowledgeEdge]:
        """Get all incoming edges to a node."""
        edge_ids = self._incoming.get(node_id, [])
        return [self._edges[eid] for eid in edge_ids if eid in self._edges]

    def get_neighbors(self, node_id: str) -> list[str]:
        """Get all neighbor node IDs (both outgoing and incoming)."""
        neighbors: set[str] = set()
        for edge in self.get_outgoing(node_id):
            neighbors.add(edge.target_id)
        for edge in self.get_incoming(node_id):
            neighbors.add(edge.source_id)
        return list(neighbors)

    def get_related(self, node_id: str, relation_type: RelationType) -> list[str]:
        """Get all node IDs related to the given node with the given relation type."""
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

        Returns a list of node IDs forming a path, or None if no path found.
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

    def query(
        self,
        *,
        node_type: str | None = None,
        relation_type: RelationType | None = None,
    ) -> list[KnowledgeNode]:
        """Query the graph for nodes matching the given criteria."""
        if node_type is not None:
            return [
                node for node in self._nodes.values()
                if node.node_type == node_type
            ]
        if relation_type is not None:
            # Return all nodes that have an edge of the given type
            matching_node_ids: set[str] = set()
            for edge in self._edges.values():
                if edge.relation_type == relation_type:
                    matching_node_ids.add(edge.source_id)
                    matching_node_ids.add(edge.target_id)
            return [
                self._nodes[nid] for nid in matching_node_ids
                if nid in self._nodes
            ]
        return list(self._nodes.values())
