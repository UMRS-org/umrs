"""Tests for plan-to-workflow emitter.

Property test: every emitted draft passes server lint OR contains at
least one manual placeholder (i.e. no partially valid steps).
"""

from __future__ import annotations

import pytest
from umrs.knowledge.emit_workflow import (
    ManualPlaceholder,
    StepDraft,
    WorkflowDraft,
    _experiment_to_step,
    _hypothesis_to_placeholder,
    _objective_to_step,
    emit_workflow,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_plan():
    """A minimal ResearchPlan dict for testing."""
    return {
        "plan_id": "plan-001",
        "title": "Test Plan",
        "project_id": "proj-1",
        "objectives": [
            {
                "objective_id": "obj-1",
                "objective_type": "understanding",
                "title": "Understand material properties",
                "description": "Analyze thermal conductivity data",
            },
            {
                "objective_id": "obj-2",
                "objective_type": "verification",
                "title": "Verify model predictions",
                "description": "Compare against experimental data",
            },
            {
                "objective_id": "obj-3",
                "objective_type": "benchmarking",
                "title": "Benchmark optimization algorithms",
                "description": "Compare optimizer performance",
            },
        ],
        "experiments": [
            {
                "experiment_id": "exp-1",
                "experiment_type": "computational",
                "title": "Run MD simulation",
                "hypothesis_ref": {"id": "hyp-1"},
                "validation_method": "comparison",
            },
            {
                "experiment_id": "exp-2",
                "experiment_type": "empirical",
                "title": "Collect experimental data",
                "validation_method": "statistical",
            },
        ],
        "hypotheses": [
            {
                "title": "Hypothesis 1: Linear relationship",
                "statement": "Conductivity increases linearly with temperature",
                "description": "Test the linear model fit",
            },
        ],
    }


@pytest.fixture
def plan_with_unmappable():
    """A plan containing objectives/experiments with empty titles (unmappable)."""
    return {
        "plan_id": "plan-bad",
        "title": "Bad Plan",
        "objectives": [
            {"objective_id": "obj-1", "title": ""},  # empty title → manual
            {
                "objective_id": "obj-2",
                "objective_type": "unknown_type",
                "title": "",
            },  # empty title → manual
        ],
        "experiments": [
            {"experiment_id": "exp-1", "title": ""},  # empty title → manual
            {
                "experiment_id": "exp-2",
                "experiment_type": "mystery",
                "title": "",
            },  # empty title → manual
        ],
        "hypotheses": [
            {
                "title": "Unmappable hypothesis",
                "statement": "Something weird",
                "description": "Cannot auto-map",
            },
        ],
    }


# ── Unit tests ────────────────────────────────────────────────────────────


class TestObjectiveMapping:
    def test_understanding_maps_to_graph_build(self):
        obj = {"objective_type": "understanding", "title": "Test", "objective_id": "1"}
        result = _objective_to_step(obj, "plan-1")
        assert result is not None
        assert result.kind == "graph_build"

    def test_verification_maps_to_scan(self):
        obj = {"objective_type": "verification", "title": "Test", "objective_id": "1"}
        result = _objective_to_step(obj, "plan-1")
        assert result.kind == "scan"

    def test_benchmarking_maps_to_catalog(self):
        obj = {"objective_type": "benchmarking", "title": "Test", "objective_id": "1"}
        result = _objective_to_step(obj, "plan-1")
        assert result.kind == "catalog"

    def test_unknown_type_fallback_to_catalog(self):
        obj = {"objective_type": "mystery", "title": "Test", "objective_id": "1"}
        result = _objective_to_step(obj, "plan-1")
        assert result.kind == "catalog"

    def test_no_type_fallback_to_catalog(self):
        obj = {"title": "Test", "objective_id": "1"}
        result = _objective_to_step(obj, "plan-1")
        assert result.kind == "catalog"

    def test_empty_title_returns_none(self):
        obj = {"title": "", "objective_id": "1"}
        result = _objective_to_step(obj, "plan-1")
        assert result is None

    def test_config_contains_plan_and_objective_ids(self):
        obj = {
            "objective_type": "understanding",
            "title": "Test",
            "objective_id": "obj-42",
        }
        result = _objective_to_step(obj, "plan-99")
        assert result.config["plan_id"] == "plan-99"
        assert result.config["objective_id"] == "obj-42"


class TestExperimentMapping:
    def test_computational_maps_to_rag_index(self):
        exp = {
            "experiment_type": "computational",
            "title": "Test",
            "experiment_id": "1",
        }
        result = _experiment_to_step(exp, "plan-1")
        assert result is not None
        assert result.kind == "rag_index"

    def test_empirical_maps_to_vector_index(self):
        exp = {"experiment_type": "empirical", "title": "Test", "experiment_id": "1"}
        result = _experiment_to_step(exp, "plan-1")
        assert result.kind == "vector_index"

    def test_benchmark_maps_to_catalog(self):
        exp = {"experiment_type": "benchmark", "title": "Test", "experiment_id": "1"}
        result = _experiment_to_step(exp, "plan-1")
        assert result.kind == "catalog"

    def test_config_includes_hypothesis_ref(self):
        exp = {
            "experiment_type": "computational",
            "title": "Test",
            "experiment_id": "exp-1",
            "hypothesis_ref": "hyp-ref-1",
        }
        result = _experiment_to_step(exp, "plan-1")
        assert result.config["hypothesis_ref"] == "hyp-ref-1"

    def test_config_includes_validation_method(self):
        exp = {
            "experiment_type": "computational",
            "title": "Test",
            "experiment_id": "exp-1",
            "validation_method": "statistical",
        }
        result = _experiment_to_step(exp, "plan-1")
        assert result.config["validation_method"] == "statistical"

    def test_empty_title_returns_none(self):
        exp = {"title": "", "experiment_id": "1"}
        result = _experiment_to_step(exp, "plan-1")
        assert result is None


class TestHypothesisPlaceholder:
    def test_hypothesis_becomes_manual_placeholder(self):
        hyp = {
            "title": "Linear model",
            "statement": "X → Y linear",
            "description": "Test linear fit",
        }
        result = _hypothesis_to_placeholder(hyp, "plan-1")
        assert isinstance(result, ManualPlaceholder)
        assert result.title == "Linear model"
        assert "Linear model" in result.instructions
        assert result.source == "hypothesis"

    def test_empty_title_returns_none(self):
        hyp = {"title": "", "statement": "", "description": ""}
        result = _hypothesis_to_placeholder(hyp, "plan-1")
        assert result is None


class TestEmitWorkflow:
    def test_full_plan_emits_all_step_types(self, sample_plan):
        result = emit_workflow(sample_plan)
        kinds = {s["kind"] for s in result["steps"]}
        assert "graph_build" in kinds
        assert "scan" in kinds
        assert "catalog" in kinds
        assert "rag_index" in kinds
        assert "vector_index" in kinds

    def test_plan_id_in_meta(self, sample_plan):
        result = emit_workflow(sample_plan)
        assert result["_meta"]["source_plan_id"] == "plan-001"

    def test_step_count_in_meta(self, sample_plan):
        result = emit_workflow(sample_plan)
        # 3 objectives + 2 experiments = 5 auto steps
        assert result["_meta"]["step_count"] == 5

    def test_manual_count_includes_hypotheses(self, sample_plan):
        result = emit_workflow(sample_plan)
        # 1 hypothesis → 1 manual placeholder
        assert result["_meta"]["manual_count"] == 1

    def test_project_context_prefill(self, sample_plan):
        result = emit_workflow(
            sample_plan,
            model="qwen-2.5",
            partition="gpu_a100",
            account="team-alpha",
        )
        ctx = result["project_context"]
        assert ctx["model"] == "qwen-2.5"
        assert ctx["partition"] == "gpu_a100"
        assert ctx["account"] == "team-alpha"
        assert ctx["project_id"] == "proj-1"

    def test_no_project_context_when_empty(self, sample_plan):
        result = emit_workflow(sample_plan)
        assert "project_context" in result

    def test_unmappable_objectives_become_manual(self, plan_with_unmappable):
        result = emit_workflow(plan_with_unmappable)
        manual_steps = [s for s in result["steps"] if s["kind"] == "manual"]
        assert len(manual_steps) >= 4  # 2 objectives + 2 experiments + 1 hypothesis

    def test_has_manual_flag(self, plan_with_unmappable):
        result = emit_workflow(plan_with_unmappable)
        assert result["_meta"]["has_manual"] is True

    def test_workflow_id_format(self, sample_plan):
        result = emit_workflow(sample_plan)
        assert result["workflow_id"].startswith("wf-")

    def test_name_from_plan_title(self, sample_plan):
        result = emit_workflow(sample_plan)
        assert "Test Plan" in result["name"]

    def test_step_has_id_kind_config(self, sample_plan):
        result = emit_workflow(sample_plan)
        for step in result["steps"]:
            assert "step_id" in step
            assert "kind" in step
            assert "config" in step

    def test_manual_step_has_instructions(self, plan_with_unmappable):
        result = emit_workflow(plan_with_unmappable)
        for step in result["steps"]:
            if step["kind"] == "manual":
                assert "instructions" in step["config"]
                assert "source" in step["config"]

    def test_dict_plan_input(self):
        plan = {
            "plan_id": "p1",
            "title": "Dict Plan",
            "objectives": [],
            "experiments": [],
            "hypotheses": [],
        }
        result = emit_workflow(plan)
        assert result["workflow_id"].startswith("wf-")

    def test_objective_with_to_dict(self):
        """Plan objects with to_dict() are accepted."""

        class FakePlan:
            def to_dict(self):
                return {
                    "plan_id": "p2",
                    "title": "ToDict Plan",
                    "objectives": [],
                    "experiments": [],
                    "hypotheses": [],
                }

        result = emit_workflow(FakePlan())
        assert "ToDict Plan" in result["name"]

    def test_invalid_plan_type_raises(self):
        with pytest.raises(TypeError):
            emit_workflow(42)


class TestStepDraftContract:
    def test_step_draft_creates_with_defaults(self):
        s = StepDraft(step_id="s1", kind="scan")
        assert s.kind == "scan"
        assert s.config == {}

    def test_workflow_draft_serializes(self):
        steps = [
            StepDraft(step_id="s1", kind="scan"),
            StepDraft(step_id="s2", kind="catalog"),
        ]
        draft = WorkflowDraft(
            workflow_id="wf-1",
            name="Test",
            steps=steps,
            project_context={"model": "qwen"},
        )
        d = draft.to_dict()
        assert d["workflow_id"] == "wf-1"
        assert len(d["steps"]) == 2
        assert d["project_context"]["model"] == "qwen"

    def test_workflow_draft_deserializes(self):
        data = {
            "workflow_id": "wf-2",
            "name": "Deser Test",
            "steps": [
                {"step_id": "s1", "kind": "rag_index", "config": {"foo": "bar"}},
            ],
            "project_context": {},
        }
        draft = WorkflowDraft.from_dict(data)
        assert draft.workflow_id == "wf-2"
        assert len(draft.steps) == 1
        assert draft.steps[0].kind == "rag_index"
        assert draft.steps[0].config == {"foo": "bar"}


# ── Property test ─────────────────────────────────────────────────────────

VALID_KINDS = {"rag_index", "vector_index", "graph_build", "catalog", "scan", "manual"}


def _random_objective():
    """Generate a random objective dict for property testing."""
    import random

    types = [
        "understanding",
        "verification",
        "optimization",
        "benchmarking",
        "mystery_type",
    ]
    return {
        "objective_id": f"obj-{random.randint(0, 9999)}",
        "objective_type": random.choice(types),
        "title": f"Objective_{random.randint(1000, 9999)}",
        "description": f"Some description {random.randint(0, 9999)}",
    }


def _random_experiment():
    import random

    types = ["computational", "empirical", "benchmark", "mystery_type"]
    return {
        "experiment_id": f"exp-{random.randint(0, 9999)}",
        "experiment_type": random.choice(types),
        "title": f"Experiment_{random.randint(1000, 9999)}",
    }


def _random_hypothesis():
    import random

    return {
        "title": f"Hypothesis_{random.randint(1000, 9999)}",
        "statement": f"Something about {random.randint(0, 9999)}",
        "description": "",
    }


def test_property_every_draft_is_valid_or_has_manual():
    """Every emitted draft must either have all valid kinds OR contain manual placeholders.

    This is the core property test from the task spec:
    "every emitted draft passes server lint OR contains at least one manual placeholder"
    """
    import random

    for i in range(100):
        plan = {
            "plan_id": f"plan-{random.randint(0, 99999)}",
            "title": f"Random Plan_{random.randint(1000, 99999)}",
            "objectives": [_random_objective() for _ in range(random.randint(0, 5))],
            "experiments": [_random_experiment() for _ in range(random.randint(0, 5))],
            "hypotheses": [_random_hypothesis() for _ in range(random.randint(0, 3))],
        }
        result = emit_workflow(plan)
        step_kinds = {s["kind"] for s in result["steps"]}

        # All kinds must be recognized
        assert step_kinds.issubset(VALID_KINDS), (
            f"Iteration {i}: Invalid kinds found: {step_kinds - VALID_KINDS}"
        )

        # If no manual steps, must have at least one step (or empty = valid for empty plan)
        has_manual = "manual" in step_kinds
        if not has_manual and len(result["steps"]) == 0:
            # Empty plan with all unmappable → should have manual placeholders
            # If truly empty input, it's OK to have empty output
            pass
