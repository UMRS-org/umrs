"""Reflective memory inspection (MEM-D4).

Aggregates all reflective memory stores (outcome history, routing feedback,
planning heuristics) into a unified inspection interface.  Operators can
view, audit, and reset entries.

This module lives in ``umrs.jobs`` because it reads from other ``umrs.jobs``
stores and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@dataclass
class ReflectiveMemoryEntry:
    """A single reflective memory entry for inspection."""

    entry_id: str
    tier: str  # outcome | routing | heuristic
    source_store: str  # which store it came from
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "tier": self.tier,
            "source_store": self.source_store,
            "summary": self.summary,
            "data": self.data,
            "created_at": self.created_at,
        }


@dataclass
class ReflectiveMemorySnapshot:
    """Full snapshot of all reflective memory stores."""

    entries: list[ReflectiveMemoryEntry] = field(default_factory=list)
    total_outcomes: int = 0
    total_routing_feedback: int = 0
    total_heuristics: int = 0
    active_heuristics: int = 0
    generated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "total_outcomes": self.total_outcomes,
            "total_routing_feedback": self.total_routing_feedback,
            "total_heuristics": self.total_heuristics,
            "active_heuristics": self.active_heuristics,
            "generated_at": _iso(self.generated_at),
        }


@dataclass
class _EntryBatch:
    entries: list[ReflectiveMemoryEntry]
    total: int
    active: int = 0


class ReflectiveMemoryInspector:
    """Unified inspection interface for reflective memory.

    Aggregates data from:
    - WorkflowOutcomeStore (MEM-D1)
    - RoutingFeedbackStore (MEM-D2)
    - AdaptivePlanningEngine (MEM-D3)

    All data is read-only through this interface. Deletion goes through
    individual stores with explicit IDs.
    """

    def __init__(
        self,
        outcome_store: Any | None = None,
        routing_store: Any | None = None,
        planning_engine: Any | None = None,
    ) -> None:
        self._outcomes = outcome_store
        self._routing = routing_store
        self._planning = planning_engine

    def inspect(
        self,
        tier: str | None = None,
        workflow_name: str | None = None,
        model_name: str | None = None,
        limit: int = 50,
    ) -> ReflectiveMemorySnapshot:
        """Inspect reflective memory with optional filters.

        Args:
            tier: Filter by tier ('outcome', 'routing', 'heuristic').
            workflow_name: Filter by workflow name.
            model_name: Filter by model name.
            limit: Maximum entries per tier.

        Returns:
            ReflectiveMemorySnapshot with all matching entries.
        """
        entries: list[ReflectiveMemoryEntry] = []
        total_outcomes = 0
        total_routing = 0
        total_heuristics = 0
        active_heuristics = 0

        # Outcomes (MEM-D1)
        if _include_tier(tier, "outcome") and self._outcomes:
            batch = self._outcome_entries(workflow_name, model_name, limit)
            entries.extend(batch.entries)
            total_outcomes = batch.total

        # Routing feedback (MEM-D2)
        if _include_tier(tier, "routing") and self._routing:
            batch = self._routing_entries(model_name, limit)
            entries.extend(batch.entries)
            total_routing = batch.total

        # Heuristics (MEM-D3)
        if _include_tier(tier, "heuristic") and self._planning:
            batch = self._heuristic_entries(workflow_name, limit)
            entries.extend(batch.entries)
            total_heuristics = batch.total
            active_heuristics = batch.active

        return ReflectiveMemorySnapshot(
            entries=entries,
            total_outcomes=total_outcomes,
            total_routing_feedback=total_routing,
            total_heuristics=total_heuristics,
            active_heuristics=active_heuristics,
        )

    def _outcome_entries(
        self,
        workflow_name: str | None,
        model_name: str | None,
        limit: int,
    ) -> _EntryBatch:
        outcomes = self._outcomes.list_outcomes(
            workflow_name=workflow_name,
            model_used=model_name,
            limit=limit,
        )
        return _EntryBatch(
            entries=[_outcome_entry(outcome) for outcome in outcomes],
            total=len(outcomes),
        )

    def _routing_entries(self, model_name: str | None, limit: int) -> _EntryBatch:
        feedbacks = self._routing.list_feedback(model_name=model_name, limit=limit)
        return _EntryBatch(
            entries=[_routing_entry(feedback) for feedback in feedbacks],
            total=len(feedbacks),
        )

    def _heuristic_entries(self, workflow_name: str | None, limit: int) -> _EntryBatch:
        heuristics = self._planning.list_heuristics(workflow_name=workflow_name)
        return _EntryBatch(
            entries=[_heuristic_entry(heuristic) for heuristic in heuristics[:limit]],
            total=len(heuristics),
            active=sum(1 for heuristic in heuristics if heuristic.is_active),
        )

    def delete_entry(self, entry_id: str, tier: str) -> bool:
        """Delete a specific reflective memory entry.

        Routes to the appropriate store based on tier.
        """
        if tier == "outcome" and self._outcomes:
            # WorkflowOutcomeStore doesn't have delete yet — log warning
            logger.warning("Outcome deletion not yet supported: %s", entry_id)
            return False
        elif tier == "routing" and self._routing:
            return self._routing.delete_feedback(entry_id)
        elif tier == "heuristic" and self._planning:
            return self._planning.delete_heuristic(entry_id)
        return False

    def reset_tier(self, tier: str) -> int:
        """Reset (deactivate) all entries in a tier.

        Returns the count of entries affected.
        """
        count = 0
        if tier == "heuristic" and self._planning:
            for h in self._planning.list_heuristics():
                if self._planning.deactivate_heuristic(h.heuristic_id):
                    count += 1
        return count


def _include_tier(selected: str | None, candidate: str) -> bool:
    return selected is None or selected == candidate


def _outcome_entry(outcome: Any) -> ReflectiveMemoryEntry:
    return ReflectiveMemoryEntry(
        entry_id=outcome.outcome_id,
        tier="outcome",
        source_store="WorkflowOutcomeStore",
        summary=f"[{outcome.status}] {outcome.workflow_name} (model={outcome.model_used or 'N/A'})",
        data=outcome.to_dict(),
        created_at=_iso(outcome.created_at) or "",
    )


def _routing_entry(feedback: Any) -> ReflectiveMemoryEntry:
    return ReflectiveMemoryEntry(
        entry_id=feedback.feedback_id,
        tier="routing",
        source_store="RoutingFeedbackStore",
        summary=f"[{feedback.outcome}] {feedback.model_selected} → {_accepted_label(feedback.user_accepted)} (step={feedback.step_name})",
        data=feedback.to_dict(),
        created_at=_iso(feedback.created_at) or "",
    )


def _accepted_label(user_accepted: bool | None) -> str:
    return {True: "accepted", False: "rejected"}.get(user_accepted, "pending")


def _heuristic_entry(heuristic: Any) -> ReflectiveMemoryEntry:
    return ReflectiveMemoryEntry(
        entry_id=heuristic.heuristic_id,
        tier="heuristic",
        source_store="AdaptivePlanningEngine",
        summary=f"[{heuristic.rule_type}] {heuristic.condition} → {heuristic.action} (conf={heuristic.confidence:.2f})",
        data=heuristic.to_dict(),
        created_at=_iso(heuristic.created_at) or "",
    )
