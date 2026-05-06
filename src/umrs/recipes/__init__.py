"""UMRS Recipe system — goose-style YAML recipes.

A recipe is a declarative YAML workflow that:
- Lists ordered steps (shell, prompt, tool-call, etc.)
- Declares required inputs (params)
- Can be executed as a durable UMRS workflow

Recipes are registered in the UMRS recipe registry and can be
invoked from any client (r4 CLI, n8n, API).

No dependency on r4.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    default: Optional[Any] = None


@dataclass
class RecipeStep:
    """A single step in a recipe."""

    step_type: StepType
    name: str = ""
    command: str = ""  # For SHELL steps
    prompt: str = ""  # For PROMPT steps
    tool: str = ""  # For TOOL_CALL steps
    args: Dict[str, Any] = field(default_factory=dict)
    condition: str = ""  # For CONDITION steps (Python expression)
    steps: List[RecipeStep] = field(default_factory=list)  # For PARALLEL/CONDITION
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
    params: List[RecipeParam] = field(default_factory=list)
    steps: List[RecipeStep] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RecipeDefinition:
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

    def validate(self) -> List[str]:
        """Validate the recipe. Returns list of errors."""
        errors: List[str] = []
        if not self.name:
            errors.append("name is required")
        if not self.steps:
            errors.append("at least one step is required")
        for i, step in enumerate(self.steps):
            step_errors = _validate_step(step, i)
            errors.extend(step_errors)
        return errors


def _parse_step(data: Dict[str, Any]) -> RecipeStep:
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


def _validate_step(step: RecipeStep, index: int) -> List[str]:
    errors: List[str] = []
    prefix = f"step[{index}]"
    if step.step_type == StepType.SHELL and not step.command:
        errors.append(f"{prefix}: shell step requires 'command'")
    if step.step_type == StepType.PROMPT and not step.prompt:
        errors.append(f"{prefix}: prompt step requires 'prompt'")
    if step.step_type == StepType.TOOL_CALL and not step.tool:
        errors.append(f"{prefix}: tool_call step requires 'tool'")
    return errors


@dataclass
class RegisteredRecipe:
    """A recipe stored in the UMRS registry."""

    recipe_id: str = ""
    definition: RecipeDefinition = field(
        default_factory=lambda: RecipeDefinition(name="")
    )
    scope: RecipeScope = RecipeScope.USER
    owner_id: str = ""
    project_id: Optional[str] = None
    registered_at: Optional[datetime] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.recipe_id and self.definition.name:
            self.recipe_id = hashlib.sha256(
                f"{self.scope.value}:{self.owner_id}:{self.definition.name}".encode()
            ).hexdigest()[:16]


class RecipeRegistry:
    """Registry for recipe definitions."""

    def __init__(self) -> None:
        self._recipes: Dict[str, RegisteredRecipe] = {}

    def register(
        self,
        definition: RecipeDefinition,
        scope: RecipeScope = RecipeScope.USER,
        owner_id: str = "",
        project_id: Optional[str] = None,
    ) -> RegisteredRecipe:
        errors = definition.validate()
        if errors:
            raise ValueError(f"Invalid recipe: {'; '.join(errors)}")

        recipe = RegisteredRecipe(
            definition=definition,
            scope=scope,
            owner_id=owner_id,
            project_id=project_id,
            registered_at=datetime.utcnow(),
        )
        self._recipes[recipe.recipe_id] = recipe
        logger.info("Registered recipe: %s (id=%s)", definition.name, recipe.recipe_id)
        return recipe

    def get(self, recipe_id: str) -> Optional[RegisteredRecipe]:
        return self._recipes.get(recipe_id)

    def list_recipes(
        self,
        scope: Optional[RecipeScope] = None,
        owner_id: Optional[str] = None,
    ) -> List[RegisteredRecipe]:
        results: List[RegisteredRecipe] = []
        for r in self._recipes.values():
            if not r.enabled:
                continue
            if scope and r.scope != scope:
                continue
            if owner_id and r.owner_id != owner_id:
                continue
            results.append(r)
        return results

    def unregister(self, recipe_id: str) -> bool:
        if recipe_id in self._recipes:
            del self._recipes[recipe_id]
            return True
        return False
