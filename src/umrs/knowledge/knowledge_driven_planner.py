"""Knowledge-driven planner — planner reasons via ScientificReasoningEngine.

M3 Phase 3: Knowledge-Driven Planning.

Architecture (single reasoning path):

  Planner (decides WHAT)
    ↓
  ScientificReasoningEngine (determines WHY and HOW)
    ↓
  GraphNavigator (traverses Knowledge Graph)
    ↓
  Knowledge Graph → Evidence → Memory → Experiments → Catalog → RAG
    ↓
  LLM (generate plan/answer)

The planner calls ScientificReasoningEngine.reason() to get a
ReasoningResult, then generates a ResearchPlan from that result.

ResearchPlan is R4 orchestration state (NOT a KnowledgeObject).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from umrs.graph_reasoning import GraphReasoningContext
from umrs.scientific_reasoning_engine import (
    ReasoningResult,
    ScientificReasoningEngine,
)

from .planner import (
    AutonomousResearchPlanner,
    PlanStatus,
    ResearchExperiment,
    ResearchHypothesis,
    ResearchObjective,
    ResearchObjectiveType,
    ResearchPlan,
    _generate_id,
)

__all__ = [
    "KnowledgeDrivenPlanner",
]


class KnowledgeDrivenPlanner:
    """Planner that reasons via the Scientific Reasoning Engine.

    Single reasoning path:
      Planner → ScientificReasoningEngine → GraphNavigator → Knowledge Graph

    The planner does NOT call GraphNavigator directly.
    The planner calls ScientificReasoningEngine.reason() and generates
    a ResearchPlan from the ReasoningResult.

    ResearchPlan is R4 orchestration state (NOT a KnowledgeObject).
    """

    def __init__(
        self,
        reasoning_engine: ScientificReasoningEngine | None = None,
    ) -> None:
        self._reasoning_engine = reasoning_engine or ScientificReasoningEngine()
        self._fallback_planner = AutonomousResearchPlanner()

    def plan_research(
        self,
        project_id: str,
        research_goal: str,
        *,
        anchor_object_id: str | None = None,
    ) -> ResearchPlan:
        """Generate a research plan grounded in the knowledge graph.

        The planner calls ScientificReasoningEngine.reason() to get
        a ReasoningResult, then generates a ResearchPlan from that result.

        Args:
            project_id: The project ID
            research_goal: The high-level research goal
            anchor_object_id: The anchor object to start reasoning from

        Returns:
            A ResearchPlan with objectives, hypotheses, and experiments
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

        # If no anchor specified, try to find one from the graph
        if anchor_object_id is None:
            all_nodes = self._reasoning_engine.navigator.graph.query()
            for node in all_nodes:
                if node.node_type == "paper":
                    anchor_object_id = node.node_id
                    break

        if anchor_object_id is None:
            # No anchor found — fall back to basic planner
            return self._fallback_planner.generate_plan(
                project_id=project_id,
                research_goal=research_goal,
            )

        # SINGLE REASONING PATH: Planner → ScientificReasoningEngine → Graph
        reasoning_result = self._reasoning_engine.reason(
            query=research_goal,
            anchor_object_id=anchor_object_id,
        )

        # Generate plan components from the reasoning result
        self._generate_objectives(plan, research_goal)
        self._generate_hypotheses(plan, reasoning_result)
        self._generate_experiments(plan, reasoning_result)
        self._generate_expected_outcomes(plan)
        plan.success_probability = self._calculate_success_probability(plan)

        # Record source knowledge from the reasoning context
        if reasoning_result.reasoning_context:
            ctx = reasoning_result.reasoning_context
            plan.source_knowledge_ids = (
                ctx.related.papers
                + ctx.related.code
                + ctx.related.notebooks
                + ctx.related.datasets
            )

        return plan

    def _generate_objectives(
        self,
        plan: ResearchPlan,
        research_goal: str,
    ) -> None:
        """Generate research objectives."""
        plan.objectives.append(ResearchObjective(
            objective_id=f"obj-{_generate_id()}",
            title=f"Understand: {research_goal[:60]}",
            description=f"Develop a thorough understanding of: {research_goal}",
            objective_type=ResearchObjectiveType.UNDERSTANDING,
            success_criteria="Comprehensive understanding documented with evidence",
            priority=2,
        ))
        plan.objectives.append(ResearchObjective(
            objective_id=f"obj-{_generate_id()}",
            title=f"Verify: {research_goal[:60]}",
            description=f"Design and execute experiments to verify hypotheses about: {research_goal}",
            objective_type=ResearchObjectiveType.VERIFICATION,
            success_criteria="Hypotheses tested with statistical significance",
            priority=2,
        ))
        plan.objectives.append(ResearchObjective(
            objective_id=f"obj-{_generate_id()}",
            title=f"Optimize: {research_goal[:60]}",
            description=f"Identify optimal parameters for: {research_goal}",
            objective_type=ResearchObjectiveType.OPTIMIZATION,
            success_criteria="Optimal parameters identified with confidence intervals",
            priority=1,
        ))

    def _generate_hypotheses(
        self,
        plan: ResearchPlan,
        reasoning_result: ReasoningResult,
    ) -> None:
        """Generate hypotheses from the reasoning result."""
        for obj in plan.objectives:
            if obj.objective_type == ResearchObjectiveType.VERIFICATION:
                # Use the reasoning engine's confidence (single confidence model)
                confidence = reasoning_result.confidence
                evidence_for = reasoning_result.evidence_ids

                hyp = ResearchHypothesis(
                    hypothesis_id=f"hyp-{_generate_id()}",
                    objective_id=obj.objective_id,
                    statement=f"There is a significant relationship in: {obj.title}",
                    rationale="Based on evidence from the knowledge graph",
                    expected_outcome="Statistically significant relationship found",
                    confidence=confidence,
                    evidence_for=evidence_for,
                )
                plan.hypotheses.append(hyp)
                obj.related_hypotheses.append(hyp.hypothesis_id)

            elif obj.objective_type == ResearchObjectiveType.UNDERSTANDING:
                hyp = ResearchHypothesis(
                    hypothesis_id=f"hyp-{_generate_id()}",
                    objective_id=obj.objective_id,
                    statement=f"Key factors can be identified for: {obj.title}",
                    rationale="Understanding requires identifying key factors",
                    expected_outcome="Key factors identified and documented",
                    confidence=0.7,
                )
                plan.hypotheses.append(hyp)
                obj.related_hypotheses.append(hyp.hypothesis_id)

    def _generate_experiments(
        self,
        plan: ResearchPlan,
        reasoning_result: ReasoningResult,
    ) -> None:
        """Generate experiments from the reasoning result."""
        for hyp in plan.hypotheses:
            exp = ResearchExperiment(
                experiment_id=f"exp-{_generate_id()}",
                hypothesis_id=hyp.hypothesis_id,
                title=f"Experiment for: {hyp.statement[:60]}",
                description=f"Test hypothesis: {hyp.statement}",
                parameters={"sample_size": 100, "significance_level": 0.05},
                expected_runtime_hours=2.0,
                expected_outcome=hyp.expected_outcome,
            )
            plan.experiments.append(exp)

    def _generate_expected_outcomes(self, plan: ResearchPlan) -> None:
        """Generate expected outcomes."""
        plan.expected_outcomes = [
            "Comprehensive understanding of the research topic",
            "Statistically significant results supporting or refuting hypotheses",
            "Identified optimal parameters (if applicable)",
            "Documented evidence and reproducible experiments",
        ]

    def _calculate_success_probability(self, plan: ResearchPlan) -> float:
        """Calculate success probability from plan hypotheses."""
        if not plan.hypotheses:
            return 0.0
        avg_confidence = sum(h.confidence for h in plan.hypotheses) / len(plan.hypotheses)
        total_evidence = sum(len(h.evidence_for) for h in plan.hypotheses)
        evidence_factor = min(1.0, total_evidence / 10.0)
        return avg_confidence * 0.6 + evidence_factor * 0.4
