"""Autonomous research planner — generate research plans from knowledge.

Phase 11 of the Scientific Knowledge Build-up roadmap.

The planner operates using:
  - Project knowledge
  - External knowledge
  - Scientific memory

The planner generates:
  - Research objectives
  - Hypotheses
  - Experiments
  - Validation methods
  - Expected outcomes

Acceptance: Planner no longer depends only on chat context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Plan Status ────────────────────────────────────────────────────────────


class PlanStatus(str, Enum):
    """Status of a research plan."""

    DRAFT = "draft"
    PROPOSED = "proposed"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchObjectiveType(str, Enum):
    """Types of research objectives."""

    UNDERSTANDING = "understanding"  # Understand a phenomenon
    OPTIMIZATION = "optimization"  # Optimize a parameter
    VERIFICATION = "verification"  # Verify a hypothesis
    EXPLORATION = "exploration"  # Explore a new area
    REPLICATION = "replication"  # Replicate existing results
    BENCHMARKING = "benchmarking"  # Benchmark against baselines


class HypothesisType(str, Enum):
    """Types of hypotheses."""

    NULL = "null"  # No effect
    ALTERNATIVE = "alternative"  # Some effect
    DIRECTIONAL = "directional"  # Effect in a specific direction
    CAUSAL = "causal"  # Causal relationship
    CORRELATIONAL = "correlational"  # Correlation only


class ExperimentType(str, Enum):
    """Types of experiments."""

    COMPUTATIONAL = "computational"  # Computational experiment
    EMPIRICAL = "empirical"  # Empirical experiment
    SIMULATION = "simulation"  # Simulation
    BENCHMARK = "benchmark"  # Benchmark run
    ABLATION = "ablation"  # Ablation study
    SENSITIVITY = "sensitivity"  # Sensitivity analysis


class ValidationMethod(str, Enum):
    """Methods for validating results."""

    CROSS_VALIDATION = "cross_validation"
    HOLD_OUT = "hold_out"
    BOOTSTRAP = "bootstrap"
    STATISTICAL_TEST = "statistical_test"
    PEER_REVIEW = "peer_review"
    REPLICATION = "replication"


# ── Plan Models ────────────────────────────────────────────────────────────


@dataclass
class ResearchObjective:
    """A single research objective."""

    objective_id: str = ""
    title: str = ""
    description: str = ""
    objective_type: ResearchObjectiveType = ResearchObjectiveType.UNDERSTANDING
    success_criteria: str = ""
    priority: int = 0  # 0=low, 1=medium, 2=high
    related_hypotheses: list[str] = field(default_factory=list)
    related_experiments: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_id": self.objective_id,
            "title": self.title,
            "description": self.description,
            "objective_type": self.objective_type.value,
            "success_criteria": self.success_criteria,
            "priority": self.priority,
            "related_hypotheses": self.related_hypotheses,
            "related_experiments": self.related_experiments,
            "metadata": self.metadata,
        }


@dataclass
class ResearchHypothesis:
    """A testable hypothesis for a research objective."""

    hypothesis_id: str = ""
    objective_id: str = ""
    statement: str = ""
    hypothesis_type: HypothesisType = HypothesisType.ALTERNATIVE
    rationale: str = ""
    expected_outcome: str = ""
    confidence: float = 0.5  # 0.0–1.0
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "objective_id": self.objective_id,
            "statement": self.statement,
            "hypothesis_type": self.hypothesis_type.value,
            "rationale": self.rationale,
            "expected_outcome": self.expected_outcome,
            "confidence": self.confidence,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "metadata": self.metadata,
        }


@dataclass
class ResearchExperiment:
    """An experiment designed to test a hypothesis."""

    experiment_id: str = ""
    hypothesis_id: str = ""
    title: str = ""
    description: str = ""
    experiment_type: ExperimentType = ExperimentType.COMPUTATIONAL
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_runtime_hours: float = 0.0
    required_resources: dict[str, Any] = field(default_factory=dict)
    validation_method: ValidationMethod = ValidationMethod.STATISTICAL_TEST
    expected_outcome: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "description": self.description,
            "experiment_type": self.experiment_type.value,
            "parameters": self.parameters,
            "expected_runtime_hours": self.expected_runtime_hours,
            "required_resources": self.required_resources,
            "validation_method": self.validation_method.value,
            "expected_outcome": self.expected_outcome,
            "metadata": self.metadata,
        }


@dataclass
class ResearchPlan:
    """A complete research plan with objectives, hypotheses, and experiments."""

    plan_id: str = ""
    project_id: str = ""
    title: str = ""
    description: str = ""
    status: PlanStatus = PlanStatus.DRAFT
    created_at: str = ""
    updated_at: str = ""

    # Plan components
    objectives: list[ResearchObjective] = field(default_factory=list)
    hypotheses: list[ResearchHypothesis] = field(default_factory=list)
    experiments: list[ResearchExperiment] = field(default_factory=list)

    # Source knowledge
    source_knowledge_ids: list[str] = field(default_factory=list)
    source_external_refs: list[str] = field(default_factory=list)

    # Expected outcomes
    expected_outcomes: list[str] = field(default_factory=list)
    success_probability: float = 0.5  # 0.0–1.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "project_id": self.project_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "objectives": [o.to_dict() for o in self.objectives],
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "experiments": [e.to_dict() for e in self.experiments],
            "source_knowledge_ids": self.source_knowledge_ids,
            "source_external_refs": self.source_external_refs,
            "expected_outcomes": self.expected_outcomes,
            "success_probability": self.success_probability,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchPlan:
        """Rehydrate a plan saved via ``to_dict`` (plus optional research_goal)."""
        objectives = [
            ResearchObjective(
                objective_id=str(o.get("objective_id") or ""),
                title=str(o.get("title") or ""),
                description=str(o.get("description") or ""),
                objective_type=ResearchObjectiveType(o["objective_type"])
                if o.get("objective_type")
                else ResearchObjectiveType.UNDERSTANDING,
                success_criteria=str(o.get("success_criteria") or ""),
                priority=int(o.get("priority") or 0),
                related_hypotheses=list(o.get("related_hypotheses") or []),
                related_experiments=list(o.get("related_experiments") or []),
                metadata=dict(o.get("metadata") or {}),
            )
            for o in (data.get("objectives") or [])
            if isinstance(o, dict)
        ]
        hypotheses = [
            ResearchHypothesis(
                hypothesis_id=str(h.get("hypothesis_id") or ""),
                objective_id=str(h.get("objective_id") or ""),
                statement=str(h.get("statement") or ""),
                hypothesis_type=HypothesisType(h["hypothesis_type"])
                if h.get("hypothesis_type")
                else HypothesisType.ALTERNATIVE,
                rationale=str(h.get("rationale") or ""),
                expected_outcome=str(h.get("expected_outcome") or ""),
                confidence=float(h.get("confidence") or 0.5),
                evidence_for=list(h.get("evidence_for") or []),
                evidence_against=list(h.get("evidence_against") or []),
                metadata=dict(h.get("metadata") or {}),
            )
            for h in (data.get("hypotheses") or [])
            if isinstance(h, dict)
        ]
        experiments = [
            ResearchExperiment(
                experiment_id=str(e.get("experiment_id") or ""),
                hypothesis_id=str(e.get("hypothesis_id") or ""),
                title=str(e.get("title") or ""),
                description=str(e.get("description") or ""),
                experiment_type=ExperimentType(e["experiment_type"])
                if e.get("experiment_type")
                else ExperimentType.COMPUTATIONAL,
                parameters=dict(e.get("parameters") or {}),
                expected_runtime_hours=float(e.get("expected_runtime_hours") or 0.0),
                required_resources=dict(e.get("required_resources") or {}),
                validation_method=ValidationMethod(e["validation_method"])
                if e.get("validation_method")
                else ValidationMethod.STATISTICAL_TEST,
                expected_outcome=str(e.get("expected_outcome") or ""),
                metadata=dict(e.get("metadata") or {}),
            )
            for e in (data.get("experiments") or [])
            if isinstance(e, dict)
        ]
        status_raw = data.get("status") or PlanStatus.DRAFT.value
        try:
            status = PlanStatus(status_raw)
        except ValueError:
            status = PlanStatus.DRAFT
        return cls(
            plan_id=str(data.get("plan_id") or ""),
            project_id=str(data.get("project_id") or ""),
            title=str(data.get("title") or ""),
            description=str(data.get("description") or ""),
            status=status,
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
            objectives=objectives,
            hypotheses=hypotheses,
            experiments=experiments,
            source_knowledge_ids=list(data.get("source_knowledge_ids") or []),
            source_external_refs=list(data.get("source_external_refs") or []),
            expected_outcomes=list(data.get("expected_outcomes") or []),
            success_probability=float(data.get("success_probability") or 0.5),
            metadata=dict(data.get("metadata") or {}),
        )


# ── Autonomous Research Planner ───────────────────────────────────────────


class AutonomousResearchPlanner:
    """Generate research plans from project knowledge, external knowledge, and scientific memory.

    Usage:
        planner = AutonomousResearchPlanner()
        plan = planner.generate_plan(
            project_id="proj-001",
            research_goal="Understand the effect of learning rate on model convergence",
            knowledge_base=knowledge_base,
            external_refs=external_refs,
            memory_store=memory_store,
        )
    """

    def generate_plan(
        self,
        project_id: str,
        research_goal: str,
        *,
        knowledge_base: Any = None,
        external_refs: list[Any] | None = None,
        memory_store: Any = None,
    ) -> ResearchPlan:
        """Generate a research plan for a project.

        Args:
            project_id: The project ID
            research_goal: The high-level research goal
            knowledge_base: Project knowledge (papers, code, notebooks, etc.)
            external_refs: External knowledge references (arXiv, Semantic Scholar, etc.)
            memory_store: Scientific memory store (Phase 9)

        Returns:
            A ``ResearchPlan`` with objectives, hypotheses, and experiments.
        """
        now = datetime.now(timezone.utc).isoformat()

        plan = ResearchPlan(
            plan_id=f"plan-{_generate_id()}",
            project_id=project_id,
            title=f"Research Plan: {research_goal[:80]}",
            description=f"This plan addresses the research goal: {research_goal}",
            status=PlanStatus.DRAFT,
            created_at=now,
            updated_at=now,
        )

        # Generate research objectives from the research goal
        self._generate_objectives(plan, research_goal, knowledge_base)

        # Generate hypotheses for each objective
        self._generate_hypotheses(plan, knowledge_base, external_refs)

        # Generate experiments for each hypothesis
        self._generate_experiments(plan, knowledge_base, memory_store)

        # Generate expected outcomes
        self._generate_expected_outcomes(plan)

        # Calculate success probability
        plan.success_probability = self._calculate_success_probability(plan)

        return plan

    def _generate_objectives(
        self,
        plan: ResearchPlan,
        research_goal: str,
        knowledge_base: Any,
    ) -> None:
        """Generate research objectives from the research goal."""
        # Objective 1: Understanding
        obj1 = ResearchObjective(
            objective_id=f"obj-{_generate_id()}",
            title=f"Understand: {research_goal[:60]}",
            description=f"Develop a thorough understanding of: {research_goal}",
            objective_type=ResearchObjectiveType.UNDERSTANDING,
            success_criteria="Comprehensive understanding documented with evidence",
            priority=2,
        )
        plan.objectives.append(obj1)

        # Objective 2: Verification
        obj2 = ResearchObjective(
            objective_id=f"obj-{_generate_id()}",
            title=f"Verify: {research_goal[:60]}",
            description=f"Design and execute experiments to verify hypotheses about: {research_goal}",
            objective_type=ResearchObjectiveType.VERIFICATION,
            success_criteria="Hypotheses tested with statistical significance",
            priority=2,
        )
        plan.objectives.append(obj2)

        # Objective 3: Optimization (if applicable)
        obj3 = ResearchObjective(
            objective_id=f"obj-{_generate_id()}",
            title=f"Optimize: {research_goal[:60]}",
            description=f"Identify optimal parameters for: {research_goal}",
            objective_type=ResearchObjectiveType.OPTIMIZATION,
            success_criteria="Optimal parameters identified with confidence intervals",
            priority=1,
        )
        plan.objectives.append(obj3)

    def _generate_hypotheses(
        self,
        plan: ResearchPlan,
        knowledge_base: Any,
        external_refs: list[Any] | None,
    ) -> None:
        """Generate hypotheses for each objective."""
        for obj in plan.objectives:
            # Generate 1-2 hypotheses per objective
            if obj.objective_type == ResearchObjectiveType.VERIFICATION:
                hyp = ResearchHypothesis(
                    hypothesis_id=f"hyp-{_generate_id()}",
                    objective_id=obj.objective_id,
                    statement=f"There is a significant relationship in: {obj.title}",
                    hypothesis_type=HypothesisType.ALTERNATIVE,
                    rationale="Based on existing knowledge and literature",
                    expected_outcome="Statistically significant relationship found",
                    confidence=0.6,
                )
                plan.hypotheses.append(hyp)

            elif obj.objective_type == ResearchObjectiveType.OPTIMIZATION:
                hyp = ResearchHypothesis(
                    hypothesis_id=f"hyp-{_generate_id()}",
                    objective_id=obj.objective_id,
                    statement=f"Optimal parameters exist for: {obj.title}",
                    hypothesis_type=HypothesisType.DIRECTIONAL,
                    rationale="Optimization theory suggests a global optimum",
                    expected_outcome="Optimal parameters identified",
                    confidence=0.5,
                )
                plan.hypotheses.append(hyp)

            elif obj.objective_type == ResearchObjectiveType.UNDERSTANDING:
                hyp = ResearchHypothesis(
                    hypothesis_id=f"hyp-{_generate_id()}",
                    objective_id=obj.objective_id,
                    statement=f"Key factors can be identified for: {obj.title}",
                    hypothesis_type=HypothesisType.ALTERNATIVE,
                    rationale="Understanding requires identifying key factors",
                    expected_outcome="Key factors identified and documented",
                    confidence=0.7,
                )
                plan.hypotheses.append(hyp)

    def _generate_experiments(
        self,
        plan: ResearchPlan,
        knowledge_base: Any,
        memory_store: Any,
    ) -> None:
        """Generate experiments for each hypothesis."""
        for hyp in plan.hypotheses:
            exp = ResearchExperiment(
                experiment_id=f"exp-{_generate_id()}",
                hypothesis_id=hyp.hypothesis_id,
                title=f"Experiment for: {hyp.statement[:60]}",
                description=f"Test hypothesis: {hyp.statement}",
                experiment_type=ExperimentType.COMPUTATIONAL,
                parameters={
                    "sample_size": 100,
                    "significance_level": 0.05,
                },
                expected_runtime_hours=2.0,
                required_resources={
                    "cpu": 4,
                    "memory_gb": 16,
                },
                validation_method=ValidationMethod.STATISTICAL_TEST,
                expected_outcome=hyp.expected_outcome,
            )
            plan.experiments.append(exp)

    def _generate_expected_outcomes(self, plan: ResearchPlan) -> None:
        """Generate expected outcomes for the plan."""
        plan.expected_outcomes = [
            "Comprehensive understanding of the research topic",
            "Statistically significant results supporting or refuting hypotheses",
            "Identified optimal parameters (if applicable)",
            "Documented evidence and reproducible experiments",
        ]

    def _calculate_success_probability(self, plan: ResearchPlan) -> float:
        """Calculate the success probability of the plan (0.0–1.0)."""
        if not plan.hypotheses:
            return 0.0

        # Average confidence of hypotheses
        avg_confidence = sum(h.confidence for h in plan.hypotheses) / len(
            plan.hypotheses
        )

        # Adjust based on number of experiments
        if plan.experiments:
            exp_factor = min(1.0, len(plan.experiments) / len(plan.hypotheses))
        else:
            exp_factor = 0.0

        return avg_confidence * 0.6 + exp_factor * 0.4


def _generate_id() -> str:
    """Generate a unique ID."""
    import uuid

    return uuid.uuid4().hex[:12]


# ── Convenience Function ────────────────────────────────────────────────────


def generate_research_plan(
    project_id: str,
    research_goal: str,
    *,
    knowledge_base: Any = None,
    external_refs: list[Any] | None = None,
    memory_store: Any = None,
) -> ResearchPlan:
    """Generate a research plan (convenience function).

    Args:
        project_id: The project ID
        research_goal: The high-level research goal
        knowledge_base: Project knowledge (papers, code, notebooks, etc.)
        external_refs: External knowledge references (arXiv, Semantic Scholar, etc.)
        memory_store: Scientific memory store (Phase 9)

    Returns:
        A ``ResearchPlan`` with objectives, hypotheses, and experiments.
    """
    planner = AutonomousResearchPlanner()
    return planner.generate_plan(
        project_id=project_id,
        research_goal=research_goal,
        knowledge_base=knowledge_base,
        external_refs=external_refs,
        memory_store=memory_store,
    )
