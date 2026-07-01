"""Tests for the knowledge-driven planner (M3 Phase 3)."""

from __future__ import annotations

from umrs.knowledge.knowledge_driven_planner import KnowledgeDrivenPlanner
from umrs.knowledge.planner import PlanStatus, ResearchPlan


def test_knowledge_driven_planner_init():
    """Test KnowledgeDrivenPlanner initialization."""
    planner = KnowledgeDrivenPlanner()
    assert planner is not None


def test_plan_research_generates_plan():
    """Test that plan_research generates a ResearchPlan."""
    planner = KnowledgeDrivenPlanner()
    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Understand transformer attention",
    )

    assert isinstance(plan, ResearchPlan)
    assert plan.project_id == "proj-001"
    assert plan.status == PlanStatus.DRAFT
    assert len(plan.objectives) > 0
    assert len(plan.hypotheses) > 0
    assert len(plan.experiments) > 0


def test_plan_research_has_objectives():
    """Test that the plan has research objectives."""
    planner = KnowledgeDrivenPlanner()
    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Understand transformer attention",
    )

    assert len(plan.objectives) >= 3  # Understanding, Verification, Optimization


def test_plan_research_has_hypotheses():
    """Test that the plan has research hypotheses."""
    planner = KnowledgeDrivenPlanner()
    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Understand transformer attention",
    )

    assert len(plan.hypotheses) > 0
    # Each hypothesis should have a confidence score
    for hyp in plan.hypotheses:
        assert 0.0 <= hyp.confidence <= 1.0


def test_plan_research_has_experiments():
    """Test that the plan has research experiments."""
    planner = KnowledgeDrivenPlanner()
    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Understand transformer attention",
    )

    assert len(plan.experiments) > 0


def test_plan_research_has_expected_outcomes():
    """Test that the plan has expected outcomes."""
    planner = KnowledgeDrivenPlanner()
    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Understand transformer attention",
    )

    assert len(plan.expected_outcomes) > 0


def test_plan_research_has_success_probability():
    """Test that the plan has a success probability."""
    planner = KnowledgeDrivenPlanner()
    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Understand transformer attention",
    )

    assert 0.0 <= plan.success_probability <= 1.0


def test_plan_research_has_source_knowledge_ids():
    """Test that the plan records source knowledge IDs from the graph."""
    planner = KnowledgeDrivenPlanner()
    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Understand transformer attention",
    )

    # source_knowledge_ids may be empty if no graph data exists
    assert isinstance(plan.source_knowledge_ids, list)


def test_plan_research_to_dict():
    """Test that the plan can be serialized to a dictionary."""
    planner = KnowledgeDrivenPlanner()
    plan = planner.plan_research(
        project_id="proj-001",
        research_goal="Understand transformer attention",
    )

    data = plan.to_dict()
    assert data["project_id"] == "proj-001"
    assert "objectives" in data
    assert "hypotheses" in data
    assert "experiments" in data
