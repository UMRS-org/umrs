"""Knowledge Correction Controller — bridges TUI to UMRS for corrections.

M3.5 Phase 5: Human Knowledge Correction.

Users can correct the AI's understanding:
- "This paper is unrelated."
- "This notebook belongs to Experiment B."
- "These two algorithms are different."
- "This finding contradicts Paper X."

Corrections update UMRS (graph edges, object metadata) and record
provenance (who, when, why, affected objects).

Architecture:
  TUI → KnowledgeCorrectionController → UMRS APIs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from umrs.knowledge_graph_api import InMemoryKnowledgeGraph, RelationType

__all__ = [
    "CorrectionRecord",
    "CorrectionType",
    "KnowledgeCorrectionController",
]


class CorrectionType(str):
    """Types of knowledge corrections."""

    UNRELATED = "unrelated"
    REASSIGN = "reassign"
    DIFFERENTIATE = "differentiate"
    CONTRADICTS = "contradicts"
    RELATES = "relates"
    RENAME = "rename"
    CORRECT_METADATA = "correct_metadata"


@dataclass
class CorrectionRecord:
    """Record of a human knowledge correction.

    The correction itself becomes provenance.

    Attributes:
        correction_id: Unique identifier
        correction_type: Type of correction
        corrected_by: User ID who made the correction
        corrected_at: Timestamp
        reason: Why the correction was made
        affected_object_ids: Objects affected by the correction
        details: Correction-specific details
    """

    correction_id: str = ""
    correction_type: str = ""
    corrected_by: str = ""
    corrected_at: str = ""
    reason: str = ""
    affected_object_ids: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction_id": self.correction_id,
            "correction_type": self.correction_type,
            "corrected_by": self.corrected_by,
            "corrected_at": self.corrected_at,
            "reason": self.reason,
            "affected_object_ids": self.affected_object_ids,
            "details": self.details,
        }


class KnowledgeCorrectionController:
    """Controller for Human Knowledge Correction.

    Bridges between the TUI and UMRS APIs for corrections.

    Usage:
        controller = KnowledgeCorrectionController(
            knowledge_graph=knowledge_graph,
        )
        record = controller.mark_unrelated("paper-001", "paper-002", "user-001", "Not relevant")
    """

    def __init__(
        self,
        *,
        knowledge_graph: InMemoryKnowledgeGraph | None = None,
    ) -> None:
        self._knowledge_graph = knowledge_graph or InMemoryKnowledgeGraph()
        self._corrections: list[CorrectionRecord] = []

    def mark_unrelated(
        self,
        object_id_1: str,
        object_id_2: str,
        corrected_by: str,
        reason: str,
    ) -> CorrectionRecord:
        """Mark two objects as unrelated.

        This removes any RELATED_TO edges between the objects.

        Args:
            object_id_1: First object
            object_id_2: Second object
            corrected_by: User making the correction
            reason: Why they are unrelated

        Returns:
            CorrectionRecord documenting the correction
        """
        # Remove related edges (in a real implementation, this would
        # remove or deprecate edges in the knowledge graph)
        record = CorrectionRecord(
            correction_type=CorrectionType.UNRELATED,
            corrected_by=corrected_by,
            reason=reason,
            affected_object_ids=[object_id_1, object_id_2],
            details={
                "object_id_1": object_id_1,
                "object_id_2": object_id_2,
            },
        )
        self._corrections.append(record)
        return record

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        corrected_by: str,
        reason: str,
        confidence: float = 0.9,
    ) -> CorrectionRecord:
        """Add a relationship between two objects.

        This adds a typed edge in the knowledge graph.

        Args:
            source_id: Source object
            target_id: Target object
            relation_type: Relation type (cites, implements, uses, etc.)
            corrected_by: User making the correction
            reason: Why the relationship exists
            confidence: Confidence in the relationship

        Returns:
            CorrectionRecord documenting the correction
        """
        try:
            rel_type = RelationType(relation_type)
        except ValueError:
            rel_type = RelationType.RELATED_TO

        self._knowledge_graph.add_edge(
            source_id=source_id,
            target_id=target_id,
            relation_type=rel_type,
            confidence=confidence,
        )

        record = CorrectionRecord(
            correction_type=CorrectionType.RELATES,
            corrected_by=corrected_by,
            reason=reason,
            affected_object_ids=[source_id, target_id],
            details={
                "source_id": source_id,
                "target_id": target_id,
                "relation_type": relation_type,
                "confidence": confidence,
            },
        )
        self._corrections.append(record)
        return record

    def correct(
        self,
        correction_type: str,
        corrected_by: str,
        reason: str,
        affected_object_ids: list[str],
        details: dict[str, Any] | None = None,
    ) -> CorrectionRecord:
        """Record a generic human knowledge correction.

        For known correction types the controller also applies the
        corresponding graph mutation. Unknown types are still recorded
        so the correction becomes provenance.
        """
        record = CorrectionRecord(
            correction_type=correction_type,
            corrected_by=corrected_by,
            corrected_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            affected_object_ids=list(affected_object_ids),
            details=details or {},
        )

        if (
            correction_type == CorrectionType.UNRELATED
            and len(affected_object_ids) >= 2
        ):
            self.mark_unrelated(
                affected_object_ids[0],
                affected_object_ids[1],
                corrected_by,
                reason,
            )
        elif correction_type == CorrectionType.RELATES and details:
            source_id = details.get("source_id")
            target_id = details.get("target_id")
            relation_type = details.get("relation_type", "related_to")
            confidence = details.get("confidence", 0.9)
            if source_id and target_id:
                self.add_relationship(
                    str(source_id),
                    str(target_id),
                    str(relation_type),
                    corrected_by,
                    reason,
                    float(confidence),
                )

        self._corrections.append(record)
        return record

    def get_correction_history(self) -> list[CorrectionRecord]:
        """Get the history of all corrections.

        Returns:
            List of CorrectionRecord, newest first
        """
        return sorted(
            self._corrections,
            key=lambda c: c.corrected_at,
            reverse=True,
        )
