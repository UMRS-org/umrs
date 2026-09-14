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
from typing import Any, Dict, List, Optional

from r4_contracts.recipes import (
    RecipeDefinition,
    RecipeParam,
    RecipeScope,
    RecipeStep,
    StepType,
)

logger = logging.getLogger(__name__)


@dataclass
class RegisteredRecipe:
    """A recipe stored in the UMRS registry."""

    recipe_id: str = ""
    definition: RecipeDefinition = field(
        default_factory=lambda: RecipeDefinition(name="")
    )
    scope: RecipeScope = RecipeScope.USER
    owner_id: str = ""
    project_id: str | None = None
    registered_at: datetime | None = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.recipe_id and self.definition.name:
            self.recipe_id = hashlib.sha256(
                f"{self.scope.value}:{self.owner_id}:{self.definition.name}".encode()
            ).hexdigest()[:16]


class RecipeRegistry:
    """Registry for recipe definitions."""

    def __init__(self) -> None:
        self._recipes: dict[str, RegisteredRecipe] = {}

    def register(
        self,
        definition: RecipeDefinition,
        scope: RecipeScope = RecipeScope.USER,
        owner_id: str = "",
        project_id: str | None = None,
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

    def get(self, recipe_id: str) -> RegisteredRecipe | None:
        return self._recipes.get(recipe_id)

    def list_recipes(
        self,
        scope: RecipeScope | None = None,
        owner_id: str | None = None,
    ) -> list[RegisteredRecipe]:
        results: list[RegisteredRecipe] = []
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
