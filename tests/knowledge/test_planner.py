"""Tests for the autonomous research planner module (Phase 11)."""

from __future__ import annotations

from umrs.knowledge.planner import (
    AutonomousResearchPlanner,
    ExperimentType,
    HypothesisType,
    PlanStatus,
    ResearchExperiment,
    ResearchHypothesis,
    ResearchObjective,
    ResearchObjectiveType,
    ResearchPlan,
    ValidationMethod,
    generate_research_plan,
)


# ── Enum Tests ─────────────────────────────────────────────────────────────


def test_plan_status_enum():
    """Test PlanStatus enum values."""
    assert PlanStatus.DRAFT == "draft"
    assert PlanStatus.PROPOSED == "proposed"
    assert PlanStatus.APPROVED == "approved"
    assert PlanStatus.EXECUTING == "executing"
    assert PlanStatus.COMPLETED == "completed"
    assert PlanStatus.FAILED == "failed"
    assert PlanStatus.CANCELLED == "cancelled"


def test_research_objective_type_enum():
    """Test ResearchObjectiveType enum values."""
    assert ResearchObjectiveType.UNDERSTANDING == "understanding"
    assert ResearchObjectiveType.OPTIMIZATION == "optimization"
    assert ResearchObjectiveType.VERIFICATION == "verification"
    assert ResearchObjectiveType.EXPLORATION == "exploration"
    assert ResearchObjectiveType.REPLICATION == "replication"
    assert ResearchObjectiveType.BENCHMARKING == "benchmarking"


def test_hypothesis_type_enum():
    """Test HypothesisType enum values."""
    assert HypothesisType.NULL == "null"
    assert HypothesisType.ALTERNATIVE == "alternative"
    assert HypothesisType.DIRECTIONAL == "directional"
    assert HypothesisType.CAUSAL == "causal"
    assert HypothesisType.CORRELATIONAL == "correlational"


def test_experiment_type_enum():
    """Test ExperimentType enum values."""
    assert ExperimentType.COMPUTATIONAL == "computational"
    assert ExperimentType.EMPIRICAL == "empirical"
    assert ExperimentType.SIMULATION == "simulation"
    assert ExperimentType.BENCHMARK == "benchmark"
    assert ExperimentType.ABLATION == "ablation"
    assert ExperimentType.SENSITIVITY == "sensitivity"


def test_validation_method_enum():
    """Test ValidationMethod enum values."""
    assert ValidationMethod.CROSS_VALIDATION == "cross_validation"
    assert ValidationMethod.HOLD_OUT == "hold_out"
    assert ValidationMethod.BOOTSTRAP == "bootstrap"
    assert ValidationMethod.STATISTICAL_TEST == "statistical_test"
    assert ValidationMethod.PEER_REVIEW == "peer_review"
    assert ValidationMethod.REPLICATION == "replication"


# ── Model Tests ────────────────────────────────────────────────────────────


def test_research_objective_defaults():
    """Test ResearchObjective defaults."""
    obj = ResearchObjective()

    assert obj.objective_id == ""
    assert obj.title == ""
    assert obj.description == ""
    assert obj.objective_type == ResearchObjectiveType.UNDERSTANDING
    assert obj.success_criteria == ""
    assert obj.priority == 0
    assert obj.related_hypotheses == []
    assert obj.related_experiments == []


def test_research_objective_to_dict():
    """Test ResearchObjective to_dict."""
    obj = ResearchObjective(
        objective_id="obj-001",
        title="Understand X",
        description="Understand the effect of X",
        objective_type=ResearchObjectiveType.VERIFICATION,
        success_criteria="p < 0.05",
        priority=2,
    )

    data = obj.to_dict()
    assert data["objective_id"] == "obj-001"
    assert data["title"] == "Understand X"
    assert data["objective_type"] == "verification"
    assert data["priority"] == 2


def test_research_hypothesis_defaults():
    """Test ResearchHypothesis defaults."""
    hyp = ResearchHypothesis()

    assert hyp.hypothesis_id == ""
    assert hyp.objective_id == ""
    assert hyp.statement == ""
    assert hyp.hypothesis_type == HypothesisType.ALTERNATIVE
    assert hyp.confidence == 0.5
    assert hyp.evidence_for == []
    assert hyp.evidence_against == []


def test_research_experiment_defaults():
    """Test ResearchExperiment defaults."""
    exp = ResearchExperiment()

    assert exp.experiment_id == ""
    assert exp.hypothesis_id == ""
    assert exp.title == ""
    assert exp.experiment_type == ExperimentType.COMPUTATIONAL
    assert exp.validation_method == ValidationMethod.STATISTICAL_TEST
    assert exp.expected_runtime_hours == 0.0


def test_research_plan_defaults():
    """Test ResearchPlan defaults."""
    plan = ResearchPlan()

    assert plan.plan_id == ""
    assert plan.project_id == ""
    assert plan.status == PlanStatus.DRAFT
    assert plan.objectives == []
    assert plan.hypotheses == []
    assert plan.experiments == []
    assert plan.expected_outcomes == []
    assert plan.success_probability == 0.5


def test_research_plan_to_dict():
    """Test ResearchPlan to_dict."""
    plan = ResearchPlan(
        plan_id="plan-001",
        project_id="proj-001",
        title="Test Plan",
        status=PlanStatus.DRAFT,
    )

    data = plan.to_dict()
    assert data["plan_id"] == "plan-001"
    assert data["project_id"] == "proj-001"
    assert data["status"] == "draft"
    assert data["objectives"] == []
    assert data["hypotheses"] == []
    assert data["experiments"] == []


# ── AutonomousResearchPlanner Tests ────────────────────────────────────────


def test_planner_init():
    """Test AutonomousResearchPlanner initialization."""
    planner = AutonomousResearchPlanner()
    assert planner is not None


def test_planner_generate_plan_basic():
    """Test generating a basic research plan."""
    planner = AutonomousResearchPlanner()
    plan = planner.generate_plan(
        project_id="proj-001",
        research_goal="Understand the effect of learning rate on convergence",
    )

    assert plan.project_id == "proj-001"
    assert plan.status == PlanStatus.DRAFT
    assert len(plan.objectives) >= 3  # Understanding, Verification, Optimization
    assert len(plan.hypotheses) >= 3  # At least one per objective
    assert len(plan.experiments) >= 3  # At least one per hypothesis
    assert len(plan.expected_outcomes) >= 4
    assert 0.0 <= plan.success_probability <= 1.0


def test_planner_generate_plan_with_knowledge_base():
    """Test generating a plan with a knowledge base."""
    planner = AutonomousResearchPlanner()
    knowledge_base = {"papers": [], "code": [], "notebooks": []}

    plan = planner.generate_plan(
        project_id="proj-001",
        research_goal="Test hypothesis about X",
        knowledge_base=knowledge_base,
    )

    assert plan.project_id == "proj-001"
    assert len(plan.objectives) >= 1


def test_planner_generate_plan_with_external_refs():
    """Test generating a plan with external references."""
    planner = AutonomousResearchPlanner()
    external_refs = [
        {"title": "Paper 1", "arxiv_id": "2401.12345"},
        {"title": "Paper 2", "doi": "10.1234/test"},
    ]

    plan = planner.generate_plan(
        project_id="proj-001",
        research_goal="Replicate results from Paper 1",
        external_refs=external_refs,
    )

    assert plan.project_id == "proj-001"
    assert len(plan.objectives) >= 1


def test_planner_generate_plan_with_memory_store():
    """Test generating a plan with a memory store."""
    planner = AutonomousResearchPlanner()

    # Mock memory store
    memory_store = type("Store", (), {"search": lambda self, q: []})()

    plan = planner.generate_plan(
        project_id="proj-001",
        research_goal="Understand phenomenon Y",
        memory_store=memory_store,
    )

    assert plan.project_id == "proj-001"
    assert len(plan.objectives) >= 1


def test_planner_objectives_have_different_types():
    """Test that generated objectives have different types."""
    planner = AutonomousResearchPlanner()
    plan = planner.generate_plan(
        project_id="proj-001",
        research_goal="Optimize parameter Z",
    )

    obj_types = {obj.objective_type for obj in plan.objectives}
    assert ResearchObjectiveType.UNDERSTANDING in obj_types
    assert ResearchObjectiveType.VERIFICATION in obj_types
    assert ResearchObjectiveType.OPTIMIZATION in obj_types


def test_planner_hypotheses_link_to_objectives():
    """Test that hypotheses are linked to objectives."""
    planner = AutonomousResearchPlanner()
    plan = planner.generate_plan(
        project_id="proj-001",
        research_goal="Understand X",
    )

    for hyp in plan.hypotheses:
        assert hyp.objective_id != ""
        assert any(obj.objective_id == hyp.objective_id for obj in plan.objectives)


def test_planner_experiments_link_to_hypotheses():
    """Test that experiments are linked to hypotheses."""
    planner = AutonomousResearchPlanner()
    plan = planner.generate_plan(
        project_id="proj-001",
        research_goal="Understand X",
    )

    for exp in plan.experiments:
        assert exp.hypothesis_id != ""
        assert any(hyp.hypothesis_id == exp.hypothesis_id for hyp in plan.hypotheses)


def test_planner_success_probability_in_range():
    """Test that success probability is in valid range."""
    planner = AutonomousResearchPlanner()
    plan = planner.generate_plan(
        project_id="proj-001",
        research_goal="Understand X",
    )

    assert 0.0 <= plan.success_probability <= 1.0


def test_planner_generate_plan_creates_unique_ids():
    """Test that generated IDs are unique."""
    planner = AutonomousResearchPlanner()
    plan = planner.generate_plan(
        project_id="proj-001",
        research_goal="Understand X",
    )

    obj_ids = [obj.objective_id for obj in plan.objectives]
    hyp_ids = [hyp.hypothesis_id for hyp in plan.hypotheses]
    exp_ids = [exp.experiment_id for exp in plan.experiments]

    assert len(obj_ids) == len(set(obj_ids))
    assert len(hyp_ids) == len(set(hyp_ids))
    assert len(exp_ids) == len(set(exp_ids))


# ── Convenience Function Tests ─────────────────────────────────────────────


def test_generate_research_plan_convenience_function():
    """Test the generate_research_plan convenience function."""
    plan = generate_research_plan(
        project_id="proj-001",
        research_goal="Understand the effect of X on Y",
    )

    assert isinstance(plan, ResearchPlan)
    assert plan.project_id == "proj-001"
    assert len(plan.objectives) >= 1
    assert len(plan.hypotheses) >= 1
    assert len(plan.experiments) >= 1
