"""Knowledge graph DTO home (ADR-026 wave B, 2026-09-11).

The shared type vocabulary for the knowledge graph layer
(`RelationType`, `KnowledgeNode`, `KnowledgeEdge`) moved out of the
deprecated ``umrs.knowledge_graph_api`` facade — that facade is wired to
this module until wave C closes (the types were ALWAYS a neutral DTO
layer; the facade only gave them a temporary address). Its own public
facade re-exports these names, so existing consumer imports keep
resolving — no aliasing needed inside consumer files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


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
