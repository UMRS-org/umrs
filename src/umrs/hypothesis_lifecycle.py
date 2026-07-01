"""Hypothesis lifecycle — first-class scientific object with lifecycle.

M3 Phase 4: Closed-Loop Scientific Learning.

A hypothesis is not merely stored. It has its own lifecycle:
  Proposed → Testing → Supported → Refined → Accepted
  Proposed → Testing → Contradicted → Refined
  Proposed → Testing → Inconclusive → Refined

Each hypothesis knows:
  - supporting evidence
  - contradicting evidence
  - confidence
  - originating planner version
  - associated experiments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

__all__ = [
    "HypothesisStatus",
    "Hypothesis",
    "HypothesisLifecycle",
]


class HypothesisStatus(str, Enum):
    """Lifecycle states of a hypothesis."""

    PROPOSED = "proposed"           # Just proposed, not yet tested
    TESTING = "testing"             # Experiments are running
    SUPPORTED = "supported"         # Evidence supports the hypothesis
    CONTRADICTED = "contradicted"   # Evidence contradicts the hypothesis
    INCONCLUSIVE = "inconclusive"  # Evidence is inconclusive
    REFINED = "refined"             # Refined into a new hypothesis
    ACCEPTED = "accepted"           # Accepted as a finding
    REJECTED = "rejected"           # Rejected based on evidence


# Valid state transitions
_VALID_TRANSITIONS: dict[HypothesisStatus, set[HypothesisStatus]] = {
    HypothesisStatus.PROPOSED: {
        HypothesisStatus.TESTING,
        HypothesisStatus.REFINED,
    },
    HypothesisStatus.TESTING: {
        HypothesisStatus.SUPPORTED,
        HypothesisStatus.CONTRADICTED,
        HypothesisStatus.INCONCLUSIVE,
    },
    HypothesisStatus.SUPPORTED: {
        HypothesisStatus.REFINED,
        HypothesisStatus.ACCEPTED,
        HypothesisStatus.TESTING,
    },
    HypothesisStatus.CONTRADICTED: {
        HypothesisStatus.REFINED,
        HypothesisStatus.REJECTED,
    },
    HypothesisStatus.INCONCLUSIVE: {
        HypothesisStatus.REFINED,
        HypothesisStatus.TESTING,
    },
    HypothesisStatus.REFINED: {
        HypothesisStatus.TESTING,
    },
    HypothesisStatus.ACCEPTED: set(),
    HypothesisStatus.REJECTED: set(),
}


@dataclass
class Hypothesis:
    """A scientific hypothesis with lifecycle tracking.

    Attributes:
        hypothesis_id: Unique identifier
        statement: The hypothesis statement
        status: Current lifecycle status
        confidence: Confidence score (0.0–1.0)
        supporting_evidence: Evidence IDs that support this hypothesis
        contradicting_evidence: Evidence IDs that contradict this hypothesis
        originating_planner_version: Which planner version proposed this
        associated_experiments: Experiment IDs that test this hypothesis
        parent_hypothesis_id: If refined, the parent hypothesis
        metadata: Additional metadata
        created_at: When the hypothesis was proposed
        updated_at: When the hypothesis was last updated
    """

    hypothesis_id: str = ""
    statement: str = ""
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = 0.5
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    originating_planner_version: str = ""
    associated_experiments: list[str] = field(default_factory=list)
    parent_hypothesis_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            self.hypothesis_id = f"hyp-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "status": self.status.value,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "contradicting_evidence": self.contradicting_evidence,
            "originating_planner_version": self.originating_planner_version,
            "associated_experiments": self.associated_experiments,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def add_supporting_evidence(self, evidence_id: str) -> None:
        """Add supporting evidence to this hypothesis."""
        if evidence_id not in self.supporting_evidence:
            self.supporting_evidence.append(evidence_id)
            self._update_confidence()
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_contradicting_evidence(self, evidence_id: str) -> None:
        """Add contradicting evidence to this hypothesis."""
        if evidence_id not in self.contradicting_evidence:
            self.contradicting_evidence.append(evidence_id)
            self._update_confidence()
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_experiment(self, experiment_id: str) -> None:
        """Associate an experiment with this hypothesis."""
        if experiment_id not in self.associated_experiments:
            self.associated_experiments.append(experiment_id)
            self.updated_at = datetime.now(timezone.utc).isoformat()

    def _update_confidence(self) -> None:
        """Update confidence based on evidence."""
        total = len(self.supporting_evidence) + len(self.contradicting_evidence)
        if total == 0:
            self.confidence = 0.5
        else:
            self.confidence = len(self.supporting_evidence) / total


class HypothesisLifecycle:
    """Manages the lifecycle of hypotheses.

    Enforces valid state transitions and tracks the history of
    each hypothesis's lifecycle.
    """

    def __init__(self) -> None:
        self._hypotheses: dict[str, Hypothesis] = {}
        self._history: dict[str, list[tuple[HypothesisStatus, str]]] = {}

    def propose(
        self,
        statement: str,
        *,
        planner_version: str = "",
        parent_hypothesis_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Propose a new hypothesis.

        Args:
            statement: The hypothesis statement
            planner_version: Which planner version proposed this
            parent_hypothesis_id: If refined, the parent hypothesis
            metadata: Additional metadata

        Returns:
            The hypothesis_id
        """
        hyp = Hypothesis(
            statement=statement,
            status=HypothesisStatus.PROPOSED,
            originating_planner_version=planner_version,
            parent_hypothesis_id=parent_hypothesis_id,
            metadata=metadata or {},
        )
        self._hypotheses[hyp.hypothesis_id] = hyp
        self._history[hyp.hypothesis_id] = [
            (HypothesisStatus.PROPOSED, hyp.created_at)
        ]
        return hyp.hypothesis_id

    def transition(
        self,
        hypothesis_id: str,
        new_status: HypothesisStatus,
    ) -> HypothesisStatus:
        """Transition a hypothesis to a new status.

        Args:
            hypothesis_id: The hypothesis ID
            new_status: The new status

        Returns:
            The new status

        Raises:
            KeyError: If hypothesis not found
            ValueError: If transition is invalid
        """
        hyp = self._hypotheses.get(hypothesis_id)
        if not hyp:
            raise KeyError(f"Hypothesis {hypothesis_id} not found")

        valid_targets = _VALID_TRANSITIONS.get(hyp.status, set())
        if new_status not in valid_targets:
            raise ValueError(
                f"Invalid transition: {hyp.status.value} → {new_status.value}"
            )

        hyp.status = new_status
        hyp.updated_at = datetime.now(timezone.utc).isoformat()
        self._history[hypothesis_id].append((new_status, hyp.updated_at))
        return new_status

    def add_evidence(
        self,
        hypothesis_id: str,
        evidence_id: str,
        supports: bool,
    ) -> None:
        """Add evidence to a hypothesis.

        Args:
            hypothesis_id: The hypothesis ID
            evidence_id: The evidence ID
            supports: True if supporting, False if contradicting
        """
        hyp = self._hypotheses.get(hypothesis_id)
        if not hyp:
            raise KeyError(f"Hypothesis {hypothesis_id} not found")

        if supports:
            hyp.add_supporting_evidence(evidence_id)
        else:
            hyp.add_contradicting_evidence(evidence_id)

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        """Get a hypothesis by ID."""
        return self._hypotheses.get(hypothesis_id)

    def get_history(
        self,
        hypothesis_id: str,
    ) -> list[tuple[HypothesisStatus, str]]:
        """Get the lifecycle history of a hypothesis."""
        return self._history.get(hypothesis_id, [])

    def list_hypotheses(
        self,
        *,
        status: HypothesisStatus | None = None,
    ) -> list[Hypothesis]:
        """List hypotheses with optional status filter."""
        results = list(self._hypotheses.values())
        if status is not None:
            results = [h for h in results if h.status == status]
        return sorted(results, key=lambda h: h.created_at)

    def refine(
        self,
        parent_hypothesis_id: str,
        new_statement: str,
        *,
        planner_version: str = "",
    ) -> str:
        """Refine a hypothesis into a new hypothesis.

        The parent hypothesis transitions to REFINED status.
        A new hypothesis is proposed with the parent_hypothesis_id set.

        Args:
            parent_hypothesis_id: The parent hypothesis to refine
            new_statement: The refined hypothesis statement
            planner_version: Which planner version proposed this

        Returns:
            The new hypothesis_id
        """
        self.transition(
            parent_hypothesis_id, HypothesisStatus.REFINED
        )
        return self.propose(
            statement=new_statement,
            planner_version=planner_version,
            parent_hypothesis_id=parent_hypothesis_id,
        )
