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
        if (tier is None or tier == "outcome") and self._outcomes:
            outcomes = self._outcomes.list_outcomes(
                workflow_name=workflow_name,
                model_used=model_name,
                limit=limit,
            )
            total_outcomes = len(outcomes)
            for o in outcomes:
                entries.append(ReflectiveMemoryEntry(
                    entry_id=o.outcome_id,
                    tier="outcome",
                    source_store="WorkflowOutcomeStore",
                    summary=f"[{o.status}] {o.workflow_name} (model={o.model_used or 'N/A'})",
                    data=o.to_dict(),
                    created_at=_iso(o.created_at) or "",
                ))

        # Routing feedback (MEM-D2)
        if (tier is None or tier == "routing") and self._routing:
            feedbacks = self._routing.list_feedback(
                model_name=model_name,
                limit=limit,
            )
            total_routing = len(feedbacks)
            for f in feedbacks:
                accepted_str = "pending"
                if f.user_accepted is True:
                    accepted_str = "accepted"
                elif f.user_accepted is False:
                    accepted_str = "rejected"
                entries.append(ReflectiveMemoryEntry(
                    entry_id=f.feedback_id,
                    tier="routing",
                    source_store="RoutingFeedbackStore",
                    summary=f"[{f.outcome}] {f.model_selected} → {accepted_str} (step={f.step_name})",
                    data=f.to_dict(),
                    created_at=_iso(f.created_at) or "",
                ))

        # Heuristics (MEM-D3)
        if (tier is None or tier == "heuristic") and self._planning:
            heuristics = self._planning.list_heuristics(workflow_name=workflow_name)
            total_heuristics = len(heuristics)
            active_heuristics = sum(1 for h in heuristics if h.is_active)
            for h in heuristics[:limit]:
                entries.append(ReflectiveMemoryEntry(
                    entry_id=h.heuristic_id,
                    tier="heuristic",
                    source_store="AdaptivePlanningEngine",
                    summary=f"[{h.rule_type}] {h.condition} → {h.action} (conf={h.confidence:.2f})",
                    data=h.to_dict(),
                    created_at=_iso(h.created_at) or "",
                ))

        return ReflectiveMemorySnapshot(
            entries=entries,
            total_outcomes=total_outcomes,
            total_routing_feedback=total_routing,
            total_heuristics=total_heuristics,
            active_heuristics=active_heuristics,
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
