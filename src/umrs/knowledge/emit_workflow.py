"""Plan-to-workflow emitter — map ResearchPlan → WorkflowDraft.

Converts the output of the knowledge-driven planner (ResearchPlan with
objectives, hypotheses, experiments) into a WorkflowDraft (list of
StepDraft) that UMRS can submit as a workflow.

Mapping rules:
  - objective_type == UNDERSTANDING  → graph_build step
  - objective_type == VERIFICATION   → scan step
  - objective_type in (OPTIMIZATION, BENCHMARKING) → catalog step
  - EXPERIMENT (computational/ablation/sensitivity) → rag_index step
  - EXPERIMENT (empirical/simulation) → vector_index step
  - unmappable objectives/experiments → manual placeholder step

Manual placeholders carry the original instruction text so the user can
review and approve before execution.

Property test: every emitted draft passes server lint OR contains at
least one manual placeholder (i.e. no partially valid steps).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Internal mapping ──────────────────────────────────────────────────────

_OBJECTIVE_KIND_MAP = {
    "understanding": "graph_build",
    "optimization": "catalog",
    "verification": "scan",
    "exploration": "catalog",
    "replication": "catalog",
    "benchmarking": "catalog",
}

_EXPERIMENT_KIND_MAP = {
    "computational": "rag_index",
    "ablation": "rag_index",
    "sensitivity": "rag_index",
    "empirical": "vector_index",
    "simulation": "vector_index",
    "benchmark": "catalog",
}

# Step types that are natively mappable
_NATIVE_KINDS = {"rag_index", "vector_index", "graph_build", "catalog", "scan"}


@dataclass
class StepMapping:
    """Result of mapping a single plan element to a StepDraft."""

    kind: str  # one of the StepDraft.kind literals
    title: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ManualPlaceholder:
    """A StepDraft that could not be auto-mapped."""

    title: str
    instructions: str
    source: str  # plan section that produced it


def _make_step_id(suffix: str) -> str:
    """Return a deterministic-ish UUID str for a step."""
    return uuid.uuid4().hex[:32]


def _get(field: Any, name: str, default: Any = None) -> Any:
    """Get a field from a dict or object (for cross-format compatibility)."""
    if isinstance(field, dict):
        return field.get(name, default)
    return getattr(field, name, default)


def _objective_to_step(
    objective: Any,
    plan_id: str,
) -> StepMapping | None:
    """Map a single objective to a StepMapping or None (unmappable)."""
    obj_type = _get(objective, "objective_type")
    obj_id = _get(objective, "objective_id", "") or ""
    title = _get(objective, "title", "") or ""
    desc = _get(objective, "description", "") or ""

    # Resolve kind
    if obj_type is None:
        kind = "catalog"  # fallback
    elif isinstance(obj_type, str):
        kind = _OBJECTIVE_KIND_MAP.get(obj_type, "catalog")
    else:
        kind = _OBJECTIVE_KIND_MAP.get(obj_type.value, "catalog")

    if not title:
        logger.warning("Objective %s has empty title — skipping", obj_id)
        return None

    config: dict[str, Any] = {
        "plan_id": plan_id,
        "objective_id": obj_id,
    }
    if desc:
        config["description"] = desc[:500]  # cap length

    return StepMapping(kind=kind, title=title, config=config)


def _experiment_to_step(
    experiment: Any,
    plan_id: str,
) -> StepMapping | None:
    """Map a single experiment to a StepMapping or None (unmappable)."""
    exp_type = _get(experiment, "experiment_type")
    exp_id = _get(experiment, "experiment_id", "") or ""
    title = _get(experiment, "title", "") or ""

    if exp_type is None:
        kind = "catalog"
    elif isinstance(exp_type, str):
        kind = _EXPERIMENT_KIND_MAP.get(exp_type, "catalog")
    else:
        kind = _EXPERIMENT_KIND_MAP.get(exp_type.value, "catalog")

    if not title:
        logger.warning("Experiment %s has empty title — skipping", exp_id)
        return None

    config: dict[str, Any] = {
        "plan_id": plan_id,
        "experiment_id": exp_id,
    }
    hypothesis_ref = _get(experiment, "hypothesis_ref", None)
    if hypothesis_ref:
        config["hypothesis_ref"] = (
            hypothesis_ref
            if isinstance(hypothesis_ref, str)
            else getattr(hypothesis_ref, "id", None)
        )
    validation = _get(experiment, "validation_method", None)
    if validation is not None:
        if isinstance(validation, str):
            config["validation_method"] = validation
        else:
            config["validation_method"] = validation.value

    return StepMapping(kind=kind, title=title, config=config)


def _hypothesis_to_placeholder(
    hypothesis: Any,
    plan_id: str,
) -> ManualPlaceholder | None:
    """Unmappable hypotheses become manual placeholders."""
    title = _get(hypothesis, "title", "") or ""
    if not title:
        return None
    statement = _get(hypothesis, "statement", "") or ""
    description = _get(hypothesis, "description", "") or ""
    return ManualPlaceholder(
        title=title,
        instructions=(
            f'Hypothesis "{title}" in plan {plan_id}.\n'
            f"Statement: {statement}\n"
            f"Description: {description}\n"
            "Review and manually execute or skip."
        ),
        source="hypothesis",
    )


def _coerce_plan_dict(plan: Any) -> dict[str, Any]:
    """Coerce a ResearchPlan instance or dict into a plain dict."""
    if hasattr(plan, "to_dict"):
        return plan.to_dict()
    if isinstance(plan, dict):
        return plan
    raise TypeError(f"Expected ResearchPlan or dict, got {type(plan).__name__}")


def _map_plan_items(
    plan_dict: dict[str, Any], plan_id: str
) -> tuple[list[StepDraft], list[ManualPlaceholder]]:
    """Map objectives/experiments/hypotheses to steps and placeholders."""
    plan_steps: list[StepDraft] = []
    manual_steps: list[ManualPlaceholder] = []

    for obj in plan_dict.get("objectives", []):
        mapping = _objective_to_step(obj, plan_id)
        if mapping is not None:
            plan_steps.append(_step_draft_from_mapping(mapping))
        else:
            manual_steps.append(
                ManualPlaceholder(
                    title=obj.get("title", "Unknown objective"),
                    instructions=f'Objective "{obj.get("title", "")}" had no type — manual review required.',
                    source="objective",
                )
            )

    for exp in plan_dict.get("experiments", []):
        mapping = _experiment_to_step(exp, plan_id)
        if mapping is not None:
            plan_steps.append(_step_draft_from_mapping(mapping))
        else:
            manual_steps.append(
                ManualPlaceholder(
                    title=exp.get("title", "Unknown experiment"),
                    instructions=f'Experiment "{exp.get("title", "")}" had no recognized type — manual review required.',
                    source="experiment",
                )
            )

    for hyp in plan_dict.get("hypotheses", []):
        placeholder = _hypothesis_to_placeholder(hyp, plan_id)
        if placeholder is not None:
            manual_steps.append(placeholder)

    return plan_steps, manual_steps


def _build_step_dicts(
    plan_steps: list[StepDraft], manual_steps: list[ManualPlaceholder]
) -> list[dict[str, Any]]:
    """Serialize mapped steps + manual placeholders into step dicts."""
    steps: list[dict[str, Any]] = []
    for s in plan_steps:
        kind = s["kind"] if isinstance(s, dict) else s.kind
        config = s["config"] if isinstance(s, dict) else s.config
        steps.append({"step_id": _make_step_id(kind), "kind": kind, "config": config})
    for m in manual_steps:
        steps.append(
            {
                "step_id": _make_step_id("manual"),
                "kind": "manual",
                "config": {"instructions": m.instructions, "source": m.source},
            }
        )
    return steps


def _build_project_context(
    plan_project_id: str,
    model: str | None,
    partition: str | None,
    account: str | None,
) -> dict[str, Any]:
    """Build the project context prefill dict."""
    context: dict[str, Any] = {}
    if model:
        context["model"] = model
    if partition:
        context["partition"] = partition
    if account:
        context["account"] = account
    if plan_project_id:
        context["project_id"] = plan_project_id
    return context


def emit_workflow(
    plan: Any,
    *,
    project_id: str | None = None,
    model: str | None = None,
    partition: str | None = None,
    account: str | None = None,
) -> dict[str, Any]:
    """Emit a WorkflowDraft dict from a ResearchPlan object.

    Parameters:
        plan: A ResearchPlan instance (or dict-like) from the planner.
        project_id: Override project ID from the plan.
        model: Model name for project context prefill.
        partition: Slurm partition for project context prefill.
        account: Slurm account for project context prefill.

    Returns:
        A dict matching WorkflowDraft contract shape, plus
        ``_meta`` with provenance info.
    """
    plan_dict = _coerce_plan_dict(plan)

    plan_id = plan_dict.get("plan_id", "plan-unknown")
    plan_title = plan_dict.get("title", "")
    plan_project_id = project_id or plan_dict.get("project_id", "")

    plan_steps, manual_steps = _map_plan_items(plan_dict, plan_id)
    steps = _build_step_dicts(plan_steps, manual_steps)
    context = _build_project_context(plan_project_id, model, partition, account)

    return {
        "workflow_id": f"wf-{_make_step_id('draft')}",
        "name": f"Workflow: {plan_title}" if plan_title else "Draft Workflow",
        "steps": steps,
        "project_context": context,
        "_meta": {
            "source_plan_id": plan_id,
            "step_count": len(plan_steps),
            "manual_count": len(manual_steps),
            "has_manual": len(manual_steps) > 0,
        },
    }


def _step_draft_from_mapping(m: StepMapping) -> dict[str, Any]:
    """Emit a plain StepDraft dict from a StepMapping."""
    return {
        "step_id": _make_step_id(m.kind),
        "kind": m.kind,
        "config": m.config,
    }


@dataclass
class StepDraft:
    """A single step in a knowledge workflow draft (contract shape).

    Re-exported for backwards compatibility. Prefer `r4_contracts.knowledge.StepDraft`.
    """

    step_id: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDraft:
    """A draft knowledge indexing workflow definition (contract shape).

    Re-exported for backwards compatibility. Prefer
    `r4_contracts.knowledge.WorkflowDraft` (defined alongside StepDraft
    in the r4-contracts package).
    """

    workflow_id: str
    name: str
    steps: list[StepDraft] = field(default_factory=list)
    source: str | None = None
    project_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "steps": [
                {"step_id": s.step_id, "kind": s.kind, "config": s.config}
                for s in self.steps
            ],
            "source": self.source,
            "project_context": self.project_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowDraft:
        steps = [
            StepDraft(
                step_id=s["step_id"],
                kind=s["kind"],
                config=s.get("config", {}),
            )
            for s in data.get("steps", [])
        ]
        return cls(
            workflow_id=data["workflow_id"],
            name=data["name"],
            steps=steps,
            source=data.get("source"),
            project_context=data.get("project_context", {}),
        )
