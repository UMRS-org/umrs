"""Research Readiness Controller — bridges TUI to UMRS for readiness.

M3.5 Phase 7-8: Research Readiness and User Approval Gate.

The controller provides:
1. Research readiness evaluation
2. User approval gate management
3. Readiness report generation

Architecture:
  TUI → ResearchReadinessController → UMRS APIs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from umrs.knowledge.canonical_graph_view import CanonicalGraphView, default_view
from umrs.knowledge_model import InMemoryKnowledgeModelStore

__all__ = [
    "ReadinessLevel",
    "ReadinessReport",
    "ApprovalRecord",
    "ResearchReadinessController",
]


class ReadinessLevel(str):
    """Research readiness levels."""

    NOT_READY = "not_ready"
    PARTIALLY_READY = "partially_ready"
    READY = "ready"
    BLOCKED = "blocked"


@dataclass
class ReadinessReport:
    """Research readiness report.

    Attributes:
        readiness_level: NOT_READY, PARTIALLY_READY, READY, BLOCKED
        knowledge_completeness: 0.0–1.0
        evidence_quality: low, medium, high
        contradictions: Number of contradictions in the knowledge graph
        unsupported_hypotheses: Number of unsupported hypotheses
        missing_experiments: Number of experiments that should be run
        recommendation: Human-readable recommendation
    """

    readiness_level: str = ReadinessLevel.NOT_READY
    knowledge_completeness: float = 0.0
    evidence_quality: str = "low"
    contradictions: int = 0
    unsupported_hypotheses: int = 0
    missing_experiments: int = 0
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "readiness_level": self.readiness_level,
            "knowledge_completeness": self.knowledge_completeness,
            "evidence_quality": self.evidence_quality,
            "contradictions": self.contradictions,
            "unsupported_hypotheses": self.unsupported_hypotheses,
            "missing_experiments": self.missing_experiments,
            "recommendation": self.recommendation,
        }


@dataclass
class ApprovalRecord:
    """Record of user approval for autonomous research.

    Attributes:
        approved: Whether the user approved
        approved_at: Timestamp
        approved_by: User ID
        readiness_level: Readiness level at time of approval
        knowledge_completeness: Completeness at time of approval
    """

    approved: bool = False
    approved_at: str = ""
    approved_by: str = ""
    readiness_level: str = ""
    knowledge_completeness: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "approved_at": self.approved_at,
            "approved_by": self.approved_by,
            "readiness_level": self.readiness_level,
            "knowledge_completeness": self.knowledge_completeness,
        }


class ResearchReadinessController:
    """Controller for Research Readiness and User Approval Gate.

    Bridges between the TUI and UMRS APIs for readiness evaluation.

    Usage:
        controller = ResearchReadinessController(
            knowledge_model=knowledge_model,
            knowledge_graph=knowledge_graph,
        )
        report = controller.evaluate_readiness()
        approved = controller.approve(user_id="user-001")
    """

    def __init__(
        self,
        *,
        knowledge_model: InMemoryKnowledgeModelStore | None = None,
        knowledge_graph: CanonicalGraphView | None = None,
    ) -> None:
        self._knowledge_model = knowledge_model or InMemoryKnowledgeModelStore()
        self._knowledge_graph = knowledge_graph or default_view(store=None)
        self._approval: ApprovalRecord | None = None

    def evaluate_readiness(self) -> ReadinessReport:
        """Evaluate research readiness.

        Returns:
            ReadinessReport with readiness level, completeness, and recommendation
        """
        report = ReadinessReport()
        counts = self._type_counts()
        report.knowledge_completeness = self._completeness(counts)

        report.evidence_quality = self._evidence_quality(counts["evidence"])

        report.contradictions = self._contradiction_count()

        # Unsupported hypotheses (hypotheses with no supporting evidence)
        report.unsupported_hypotheses = max(0, counts["hypothesis"] - counts["evidence"])

        # Missing experiments (hypotheses without experiments)
        report.missing_experiments = max(0, counts["hypothesis"] - counts["experiment"])

        level, recommendation = self._readiness_decision(report.knowledge_completeness, report.evidence_quality)
        report.readiness_level = level
        report.recommendation = recommendation

        return report

    def _type_counts(self) -> dict[str, int]:
        """Per-type object counts in one pass."""
        objects = self._knowledge_model.list_objects(limit=10000)
        keys = ("paper", "hypothesis", "experiment", "finding", "evidence")
        return {
            key: sum(1 for o in objects if o.object_type.value == key)
            for key in keys
        }

    @staticmethod
    def _completeness(counts: dict[str, int]) -> float:
        """Presence-fraction across the five readiness kinds."""
        kinds = ("paper", "hypothesis", "experiment", "finding", "evidence")
        presence = sum(1 for key in kinds if counts.get(key, 0) > 0)
        return presence / 5.0

    @staticmethod
    def _evidence_quality(evidence_count: int) -> str:
        """Evidence quality tier (high/medium/low by count)."""
        if evidence_count >= 5:
            return "high"
        if evidence_count >= 2:
            return "medium"
        return "low"

    def _contradiction_count(self) -> int:
        """Edges whose relation name reads 'contradict' (one pass, no double-count)."""
        all_nodes = self._knowledge_graph.query()
        contradiction_count = 0
        for node in all_nodes:
            for edge in self._knowledge_graph.get_outgoing(node.node_id):
                if "contradict" in edge.relation_type.value.lower():
                    contradiction_count += 1
        return contradiction_count

    @staticmethod
    def _readiness_decision(completeness: float, evidence_quality: str) -> tuple[ReadinessLevel, str]:
        """Level+recommendation table (one decision rule row each)."""
        if completeness >= 0.8 and evidence_quality == "high":
            return ReadinessLevel.READY, "Ready for autonomous research."
        if completeness >= 0.6:
            return ReadinessLevel.PARTIALLY_READY, "Partially ready. Consider adding more evidence."
        if completeness >= 0.4:
            return ReadinessLevel.NOT_READY, "Not ready. Upload more papers and datasets."
        return ReadinessLevel.BLOCKED, "Blocked. Insufficient knowledge base."

    def approve(
        self,
        user_id: str,
        *,
        readiness_report: ReadinessReport | None = None,
    ) -> ApprovalRecord:
        """Approve autonomous research.

        Args:
            user_id: The user approving
            readiness_report: The readiness report at time of approval

        Returns:
            ApprovalRecord documenting the approval
        """
        report = readiness_report or self.evaluate_readiness()

        self._approval = ApprovalRecord(
            approved=True,
            approved_at=datetime.now(timezone.utc).isoformat(),
            approved_by=user_id,
            readiness_level=report.readiness_level,
            knowledge_completeness=report.knowledge_completeness,
        )

        return self._approval

    def reject(
        self,
        user_id: str,
        reason: str = "",
    ) -> ApprovalRecord:
        """Reject autonomous research.

        Args:
            user_id: The user rejecting
            reason: Why the user rejected

        Returns:
            ApprovalRecord documenting the rejection
        """
        self._approval = ApprovalRecord(
            approved=False,
            approved_at=datetime.now(timezone.utc).isoformat(),
            approved_by=user_id,
        )

        return self._approval

    def is_approved(self) -> bool:
        """Check if autonomous research has been approved.

        Returns:
            True if the user has approved, False otherwise
        """
        return self._approval is not None and self._approval.approved

    def get_approval(self) -> ApprovalRecord | None:
        """Get the current approval record.

        Returns:
            The ApprovalRecord, or None if no approval has been made
        """
        return self._approval
