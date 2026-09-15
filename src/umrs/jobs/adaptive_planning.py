"""Adaptive planning heuristics (MEM-D3).

Based on workflow outcome history, suggests parameter adjustments or model
selections for new runs.  Presented as recommendations, never auto-applied.

This module lives in ``umrs.jobs`` because it reads from outcome and routing
feedback stores and does not import from ``umrs.api`` or ``umrs.server``.
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
class Recommendation:
    """A single adaptive recommendation for a workflow template."""

    recommendation_id: str
    category: str  # model_switch | param_adjust | resource_adjust | retry_config
    confidence: float  # 0.0-1.0
    description: str
    current_value: Any = None
    suggested_value: Any = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "category": self.category,
            "confidence": self.confidence,
            "description": self.description,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "evidence": self.evidence,
        }


@dataclass
class PlanningHeuristic:
    """A stored heuristic derived from outcome data."""

    heuristic_id: str
    workflow_name: str
    rule_type: str  # model_preference | param_default | failure_avoidance
    condition: str  # human-readable condition
    action: str  # what to recommend
    confidence: float = 0.0
    sample_count: int = 0
    is_active: bool = True
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "heuristic_id": self.heuristic_id,
            "workflow_name": self.workflow_name,
            "rule_type": self.rule_type,
            "condition": self.condition,
            "action": self.action,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "is_active": self.is_active,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
        }


@dataclass
class SuggestionResult:
    """Result of running adaptive planning for a workflow template."""

    workflow_name: str
    recommendations: list[Recommendation] = field(default_factory=list)
    heuristics_used: list[str] = field(default_factory=list)
    outcome_count: int = 0
    generated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "heuristics_used": self.heuristics_used,
            "outcome_count": self.outcome_count,
            "generated_at": _iso(self.generated_at),
        }


VALID_CATEGORIES = {"model_switch", "param_adjust", "resource_adjust", "retry_config"}
VALID_RULE_TYPES = {"model_preference", "param_default", "failure_avoidance"}


class AdaptivePlanningEngine:
    """Generates adaptive recommendations from outcome and routing data.

    Reads from WorkflowOutcomeStore and optionally RoutingFeedbackStore
    to produce suggestions. Never auto-applies — all output is advisory.
    """

    def __init__(
        self,
        outcome_store: Any,
        routing_store: Any | None = None,
    ) -> None:
        self._outcomes = outcome_store
        self._routing = routing_store
        # In-memory heuristic storage
        self._heuristics: dict[str, PlanningHeuristic] = {}
        self._counter = 0

    def suggest(self, workflow_name: str) -> SuggestionResult:
        """Generate recommendations for a workflow template.

        Analyzes outcome history and routing feedback to produce
        actionable suggestions. Returns recommendations sorted by
        confidence (highest first).
        """
        outcomes = self._outcomes.list_outcomes(workflow_name=workflow_name, limit=1000)
        if not outcomes:
            return SuggestionResult(workflow_name=workflow_name, outcome_count=0)

        recommendations: list[Recommendation] = []
        heuristics_used: list[str] = []

        # 1. Model selection recommendation
        model_rec = self._analyze_model_performance(workflow_name, outcomes)
        if model_rec:
            recommendations.append(model_rec)

        # 2. Failure pattern avoidance
        failure_recs = self._analyze_failure_patterns(workflow_name, outcomes)
        recommendations.extend(failure_recs)

        # 3. Resource adjustment
        resource_rec = self._analyze_resource_usage(workflow_name, outcomes)
        if resource_rec:
            recommendations.append(resource_rec)

        # 4. Apply stored heuristics
        for h in self.list_heuristics(workflow_name):
            if h.is_active:
                heuristics_used.append(h.heuristic_id)

        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        return SuggestionResult(
            workflow_name=workflow_name,
            recommendations=recommendations,
            heuristics_used=heuristics_used,
            outcome_count=len(outcomes),
        )

    # =========================================================================
    # Heuristic CRUD
    # =========================================================================

    def store_heuristic(
        self,
        workflow_name: str,
        rule_type: str,
        condition: str,
        action: str,
        confidence: float = 0.0,
        sample_count: int = 0,
    ) -> PlanningHeuristic:
        """Store a planning heuristic."""
        if rule_type not in VALID_RULE_TYPES:
            raise ValueError(
                f"Invalid rule_type: {rule_type!r}; must be one of {VALID_RULE_TYPES}"
            )

        self._counter += 1
        h = PlanningHeuristic(
            heuristic_id=f"heur-{self._counter:04d}",
            workflow_name=workflow_name,
            rule_type=rule_type,
            condition=condition,
            action=action,
            confidence=confidence,
            sample_count=sample_count,
        )
        self._heuristics[h.heuristic_id] = h
        return h

    def get_heuristic(self, heuristic_id: str) -> PlanningHeuristic | None:
        return self._heuristics.get(heuristic_id)

    def list_heuristics(
        self, workflow_name: str | None = None
    ) -> list[PlanningHeuristic]:
        results = list(self._heuristics.values())
        if workflow_name:
            results = [h for h in results if h.workflow_name == workflow_name]
        return sorted(results, key=lambda h: h.confidence, reverse=True)

    def deactivate_heuristic(self, heuristic_id: str) -> bool:
        h = self._heuristics.get(heuristic_id)
        if h:
            h.is_active = False
            h.updated_at = _now()
            return True
        return False

    def delete_heuristic(self, heuristic_id: str) -> bool:
        return self._heuristics.pop(heuristic_id, None) is not None

    # =========================================================================
    # Analysis helpers
    # =========================================================================

    def _analyze_model_performance(
        self,
        workflow_name: str,
        outcomes: list[Any],
    ) -> Recommendation | None:
        """Recommend best-performing model based on outcome history."""
        model_stats: dict[str, dict[str, int]] = {}
        for o in outcomes:
            model = o.model_used or "unknown"
            if model not in model_stats:
                model_stats[model] = {"total": 0, "succeeded": 0}
            model_stats[model]["total"] += 1
            if o.status == "succeeded":
                model_stats[model]["succeeded"] += 1

        if len(model_stats) < 2:
            return None

        # Find best model by success rate
        best_model = None
        best_rate = 0.0
        current_model = None
        for model, stats in model_stats.items():
            rate = stats["succeeded"] / stats["total"] if stats["total"] > 0 else 0
            if rate > best_rate:
                best_rate = rate
                best_model = model
            # Most recent outcome's model is "current"
            if current_model is None:
                current_model = model

        if best_model and best_model != current_model and best_rate > 0.7:
            self._counter += 1
            return Recommendation(
                recommendation_id=f"rec-{self._counter:04d}",
                category="model_switch",
                confidence=min(best_rate, 0.95),
                description=f"Model '{best_model}' has {best_rate:.0%} success rate vs current",
                current_value=current_model,
                suggested_value=best_model,
                evidence={"model_stats": model_stats},
            )
        return None

    def _analyze_failure_patterns(
        self,
        workflow_name: str,
        outcomes: list[Any],
    ) -> list[Recommendation]:
        """Identify recurring failure patterns and suggest avoidance."""
        recs: list[Recommendation] = []
        category_counts: dict[str, int] = {}
        total = len(outcomes)

        for o in outcomes:
            if o.failure_category:
                category_counts[o.failure_category] = (
                    category_counts.get(o.failure_category, 0) + 1
                )

        for cat, count in category_counts.items():
            rate = count / total if total > 0 else 0
            if rate >= 0.2 and count >= 2:
                self._counter += 1
                recs.append(
                    Recommendation(
                        recommendation_id=f"rec-{self._counter:04d}",
                        category="retry_config",
                        confidence=min(rate, 0.9),
                        description=f"Failure category '{cat}' occurs in {rate:.0%} of runs ({count}/{total})",
                        current_value=cat,
                        suggested_value=f"Add retry/mitigation for {cat}",
                        evidence={
                            "failure_category": cat,
                            "count": count,
                            "total": total,
                        },
                    )
                )

        return recs

    def _analyze_resource_usage(
        self,
        workflow_name: str,
        outcomes: list[Any],
    ) -> Recommendation | None:
        """Suggest resource adjustments based on outcome data."""
        durations = [
            o.execution_duration_sec for o in outcomes if o.execution_duration_sec
        ]
        if len(durations) < 3:
            return None

        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)

        # If max is 3x average, suggest resource increase
        if max_duration > avg_duration * 3 and avg_duration > 0:
            self._counter += 1
            return Recommendation(
                recommendation_id=f"rec-{self._counter:04d}",
                category="resource_adjust",
                confidence=0.6,
                description=f"High duration variance (avg={avg_duration:.1f}s, max={max_duration:.1f}s)",
                current_value={"avg_duration_sec": round(avg_duration, 1)},
                suggested_value={"timeout_sec": round(max_duration * 1.5, 0)},
                evidence={"durations": [round(d, 1) for d in durations[-10:]]},
            )
        return None
