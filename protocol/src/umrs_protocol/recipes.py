"""Recipe schema and pre-flight validation.

A recipe is a declarative YAML workflow consumed by UMRS. This module
defines the *client-visible* shape (so r4 / CLI / tooling can validate
recipes locally before submitting them) without any UMRS runtime deps.

Mirror of the public DTO portion of `umrs.recipes`. UMRS remains the
implementation owner of recipe registration, scheduling, and execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "StepType",
    "RecipeScope",
    "RecipeParam",
    "RecipeStep",
    "RecipeDefinition",
]


class StepType(Enum):
    """Types of recipe steps."""

    SHELL = "shell"
    PROMPT = "prompt"
    TOOL_CALL = "tool_call"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SUB_RECIPE = "sub_recipe"


class RecipeScope(Enum):
    SYSTEM = "system"
    PROJECT = "project"
    USER = "user"


@dataclass
class RecipeParam:
    """A parameter declared by a recipe."""

    name: str
    param_type: str = "string"  # string, int, float, bool, path
    description: str = ""
    required: bool = True
    default: Any | None = None


@dataclass
class RecipeStep:
    """A single step in a recipe."""

    step_type: StepType
    name: str = ""
    command: str = ""  # For SHELL steps
    prompt: str = ""  # For PROMPT steps
    tool: str = ""  # For TOOL_CALL steps
    args: dict[str, Any] = field(default_factory=dict)
    condition: str = ""  # For CONDITION steps (Python expression)
    steps: list[RecipeStep] = field(default_factory=list)  # For PARALLEL/CONDITION
    timeout_seconds: int = 300
    retry_count: int = 0
    on_error: str = "fail"  # fail | skip | retry


@dataclass
class RecipeDefinition:
    """A complete recipe definition."""

    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    params: list[RecipeParam] = field(default_factory=list)
    steps: list[RecipeStep] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecipeDefinition:
        """Parse a recipe from a dict (loaded from YAML)."""
        params = [
            RecipeParam(
                name=p.get("name", ""),
                param_type=p.get("type", "string"),
                description=p.get("description", ""),
                required=p.get("required", True),
                default=p.get("default"),
            )
            for p in data.get("params", [])
        ]

        steps = [_parse_step(s) for s in data.get("steps", [])]

        return cls(
            name=data.get("name", ""),
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            params=params,
            steps=steps,
            tags=data.get("tags", []),
        )

    def validate(self) -> list[str]:
        """Validate the recipe. Returns list of errors (empty if valid)."""
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.steps:
            errors.append("at least one step is required")
        for i, step in enumerate(self.steps):
            errors.extend(_validate_step(step, i))
        return errors


def _parse_step(data: dict[str, Any]) -> RecipeStep:
    """Parse a single step from dict."""
    step_type = StepType(data.get("type", "shell"))
    sub_steps = [_parse_step(s) for s in data.get("steps", [])]
    return RecipeStep(
        step_type=step_type,
        name=data.get("name", ""),
        command=data.get("command", ""),
        prompt=data.get("prompt", ""),
        tool=data.get("tool", ""),
        args=data.get("args", {}),
        condition=data.get("condition", ""),
        steps=sub_steps,
        timeout_seconds=data.get("timeout", 300),
        retry_count=data.get("retry", 0),
        on_error=data.get("on_error", "fail"),
    )


def _validate_step(step: RecipeStep, index: int) -> list[str]:
    errors: list[str] = []
    prefix = f"step[{index}]"
    if step.step_type == StepType.SHELL and not step.command:
        errors.append(f"{prefix}: shell step requires 'command'")
    if step.step_type == StepType.PROMPT and not step.prompt:
        errors.append(f"{prefix}: prompt step requires 'prompt'")
    if step.step_type == StepType.TOOL_CALL and not step.tool:
        errors.append(f"{prefix}: tool_call step requires 'tool'")
    return errors
