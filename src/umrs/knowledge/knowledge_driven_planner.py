"""Knowledge-driven planner — planner reasons from UMRS Knowledge Graph.

M3 Phase 3: Knowledge-Driven Planning.

The planner uses GraphNavigator to reason from the knowledge graph.
The planner priority order (M3 Decision 4) is:

  1. Knowledge Graph    — what is related?
  2. Evidence           — what supports this?
  3. Scientific Memory  — what has been done?
  4. Experiments        — what has been tested?
  5. Catalog           — what artifacts exist?
  6. RAG               — what is in the literature?
  7. LLM               — generate plan/answer

ResearchPlan is R4 orchestration state (NOT a KnowledgeObject).
The plan is mutable, versioned, and regenerated frequently.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from umrs.graph_reasoning import GraphNavigator, GraphReasoningContext

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
    """Planner that reasons from the UMRS Knowledge Graph.

    This planner uses GraphNavigator to:
      1. Find related objects through the knowledge graph
      2. Build evidence chains for claims
      3. Build experiment chains for hypotheses
      4. Generate a research plan grounded in the knowledge graph

    The planner follows the M3 Decision 4 priority order:
      Graph → Evidence → Memory → Experiments → Catalog → RAG → LLM

    ResearchPlan is R4 orchestration state (NOT a KnowledgeObject).
    """

    def __init__(
        self,
        navigator: GraphNavigator | None = None,
    ) -> None:
        self._navigator = navigator or GraphNavigator()
        self._fallback_planner = AutonomousResearchPlanner()

    def plan_research(
        self,
        project_id: str,
        research_goal: str,
        *,
        anchor_object_id: str | None = None,
    ) -> ResearchPlan:
        """Generate a research plan grounded in the knowledge graph.

        Args:
            project_id: The project ID
            research_goal: The high-level research goal
            anchor_object_id: The anchor object to start reasoning from
                (if None, uses the first paper in the graph)

        Returns:
            A ResearchPlan with objectives, hypotheses, and experiments,
            grounded in the knowledge graph.

        The plan generation follows the priority order:
          1. Knowledge Graph — find related objects
          2. Evidence — find evidence chains
          3. Experiments — find experiment chains
          4. Generate plan from reasoning context
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

        # If no anchor specified, try to find one
        if anchor_object_id is None:
            # Use the first paper node as anchor
            all_nodes = self._navigator.graph.query()
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

        # 1. Knowledge Graph — find related objects through the graph
        reasoning_context = self._navigator.reason(
            query=research_goal,
            anchor_id=anchor_object_id,
        )

        # 2. Generate objectives from reasoning context
        self._generate_objectives_from_context(
            plan, research_goal, reasoning_context
        )

        # 3. Generate hypotheses from reasoning context
        self._generate_hypotheses_from_context(
            plan, reasoning_context
        )

        # 4. Generate experiments from reasoning context
        self._generate_experiments_from_context(
            plan, reasoning_context
        )

        # 5. Generate expected outcomes
        self._generate_expected_outcomes(plan)

        # 6. Calculate success probability
        plan.success_probability = self._calculate_success_probability(plan)

        # 7. Record source knowledge from the graph
        plan.source_knowledge_ids = (
            reasoning_context.related.papers
            + reasoning_context.related.code
            + reasoning_context.related.notebooks
            + reasoning_context.related.datasets
        )

        return plan

    def _generate_objectives_from_context(
        self,
        plan: ResearchPlan,
        research_goal: str,
        context: GraphReasoningContext,
    ) -> None:
        """Generate research objectives from the reasoning context."""
        # Understanding objective
        obj1 = ResearchObjective(
            objective_id=f"obj-{_generate_id()}",
            title=f"Understand: {research_goal[:60]}",
            description=f"Develop a thorough understanding of: {research_goal}",
            objective_type=ResearchObjectiveType.UNDERSTANDING,
            success_criteria="Comprehensive understanding documented with evidence",
            priority=2,
            related_hypotheses=[],
        )
        plan.objectives.append(obj1)

        # Verification objective
        obj2 = ResearchObjective(
            objective_id=f"obj-{_generate_id()}",
            title=f"Verify: {research_goal[:60]}",
            description=f"Design and execute experiments to verify hypotheses about: {research_goal}",
            objective_type=ResearchObjectiveType.VERIFICATION,
            success_criteria="Hypotheses tested with statistical significance",
            priority=2,
        )
        plan.objectives.append(obj2)

        # Optimization objective
        obj3 = ResearchObjective(
            objective_id=f"obj-{_generate_id()}",
            title=f"Optimize: {research_goal[:60]}",
            description=f"Identify optimal parameters for: {research_goal}",
            objective_type=ResearchObjectiveType.OPTIMIZATION,
            success_criteria="Optimal parameters identified with confidence intervals",
            priority=1,
        )
        plan.objectives.append(obj3)

    def _generate_hypotheses_from_context(
        self,
        plan: ResearchPlan,
        context: GraphReasoningContext,
    ) -> None:
        """Generate hypotheses from the reasoning context."""
        for obj in plan.objectives:
            if obj.objective_type == ResearchObjectiveType.VERIFICATION:
                # Generate hypothesis grounded in evidence chains
                evidence_count = sum(
                    len(chain.supporting) for chain in context.evidence_chains
                )
                confidence = min(0.9, 0.3 + evidence_count * 0.1)

                hyp = ResearchHypothesis(
                    hypothesis_id=f"hyp-{_generate_id()}",
                    objective_id=obj.objective_id,
                    statement=f"There is a significant relationship in: {obj.title}",
                    rationale="Based on evidence from the knowledge graph",
                    expected_outcome="Statistically significant relationship found",
                    confidence=confidence,
                    evidence_for=[
                        eid
                        for chain in context.evidence_chains
                        for eid in chain.supporting
                    ],
                    evidence_against=[
                        eid
                        for chain in context.evidence_chains
                        for eid in chain.contradicting
                    ],
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

    def _generate_experiments_from_context(
        self,
        plan: ResearchPlan,
        context: GraphReasoningContext,
    ) -> None:
        """Generate experiments from the reasoning context."""
        for hyp in plan.hypotheses:
            # Generate experiment for each hypothesis
            # Use existing experiments as reference
            existing_experiments = context.related.experiments

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
        """Generate expected outcomes for the plan."""
        plan.expected_outcomes = [
            "Comprehensive understanding of the research topic",
            "Statistically significant results supporting or refuting hypotheses",
            "Identified optimal parameters (if applicable)",
            "Documented evidence and reproducible experiments",
        ]

    def _calculate_success_probability(self, plan: ResearchPlan) -> float:
        """Calculate the success probability of the plan."""
        if not plan.hypotheses:
            return 0.0

        # Average confidence of hypotheses
        avg_confidence = sum(h.confidence for h in plan.hypotheses) / len(
            plan.hypotheses
        )

        # Factor in evidence
        total_evidence = sum(
            len(h.evidence_for) for h in plan.hypotheses
        )
        evidence_factor = min(1.0, total_evidence / 10.0)

        # Weighted combination
        return avg_confidence * 0.6 + evidence_factor * 0.4
