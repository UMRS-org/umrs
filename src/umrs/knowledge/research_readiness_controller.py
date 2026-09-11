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
from umrs.knowledge_graph_api import InMemoryKnowledgeGraph
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
        knowledge_graph: InMemoryKnowledgeGraph | CanonicalGraphView | None = None,
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

        # Count objects by type
        objects = self._knowledge_model.list_objects(limit=10000)
        paper_count = sum(1 for o in objects if o.object_type.value == "paper")
        hypothesis_count = sum(1 for o in objects if o.object_type.value == "hypothesis")
        experiment_count = sum(1 for o in objects if o.object_type.value == "experiment")
        finding_count = sum(1 for o in objects if o.object_type.value == "finding")
        evidence_count = sum(1 for o in objects if o.object_type.value == "evidence")

        # Knowledge completeness (simple heuristic)
        has_papers = paper_count > 0
        has_hypotheses = hypothesis_count > 0
        has_experiments = experiment_count > 0
        has_findings = finding_count > 0
        has_evidence = evidence_count > 0

        completeness = sum([
            has_papers, has_hypotheses, has_experiments,
            has_findings, has_evidence,
        ]) / 5.0

        report.knowledge_completeness = completeness

        # Evidence quality
        if evidence_count >= 5:
            report.evidence_quality = "high"
        elif evidence_count >= 2:
            report.evidence_quality = "medium"
        else:
            report.evidence_quality = "low"

        # Contradictions (edges with CONTRADICTS relation)
        all_nodes = self._knowledge_graph.query()
        contradiction_count = 0
        for node in all_nodes:
            for edge in self._knowledge_graph.get_outgoing(node.node_id):
                if "contradict" in edge.relation_type.value.lower():
                    contradiction_count += 1
        report.contradictions = contradiction_count

        # Unsupported hypotheses (hypotheses with no supporting evidence)
        report.unsupported_hypotheses = max(0, hypothesis_count - evidence_count)

        # Missing experiments (hypotheses without experiments)
        report.missing_experiments = max(0, hypothesis_count - experiment_count)

        # Determine readiness level
        if completeness >= 0.8 and report.evidence_quality == "high":
            report.readiness_level = ReadinessLevel.READY
            report.recommendation = "Ready for autonomous research."
        elif completeness >= 0.6:
            report.readiness_level = ReadinessLevel.PARTIALLY_READY
            report.recommendation = "Partially ready. Consider adding more evidence."
        elif completeness >= 0.4:
            report.readiness_level = ReadinessLevel.NOT_READY
            report.recommendation = "Not ready. Upload more papers and datasets."
        else:
            report.readiness_level = ReadinessLevel.BLOCKED
            report.recommendation = "Blocked. Insufficient knowledge base."

        return report

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
