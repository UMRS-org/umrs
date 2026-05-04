"""Access control enforcement for model operations (U-803)."""

from __future__ import annotations

from dataclasses import dataclass

from ..models import ModelEntry, ModelScope
from .errors import PermissionDeniedError


@dataclass
class AccessContext:
    """Context for access control decisions."""

    user_id: str
    is_admin: bool = False
    project_memberships: set[str] = None  # Projects user belongs to
    admin_project_ids: set[str] = None    # Projects where user is admin (U-815)

    def __post_init__(self):
        if self.project_memberships is None:
            self.project_memberships = set()
        if self.admin_project_ids is None:
            self.admin_project_ids = set()


class ACLEnforcer:
    """
    Enforces access control for model operations.

    Rules:
    - System models: readable by all, writable by admins only
    - Project models: readable/writable by project members
    - User models: readable/writable by owner only
    """

    def can_read(self, model: ModelEntry, ctx: AccessContext) -> bool:
        """Check if user can read/use a model."""
        if ctx.is_admin:
            return True
        if model.scope == ModelScope.SYSTEM:
            return True
        if model.scope == ModelScope.PROJECT:
            return model.project_id in ctx.project_memberships
        if model.scope == ModelScope.USER:
            return model.owner_id == ctx.user_id
        return False

    def can_write(self, model: ModelEntry, ctx: AccessContext) -> bool:
        """Check if user can modify/delete a model."""
        if ctx.is_admin:
            return True
        if model.scope == ModelScope.SYSTEM:
            return False
        if model.scope == ModelScope.PROJECT:
            return model.project_id in ctx.project_memberships
        if model.scope == ModelScope.USER:
            return model.owner_id == ctx.user_id
        return False

    def can_create(
        self,
        scope: ModelScope,
        ctx: AccessContext,
        project_id: str | None = None,
    ) -> bool:
        """Check if user can create a model in the given scope."""
        if ctx.is_admin:
            return True
        if scope == ModelScope.SYSTEM:
            return False
        if scope == ModelScope.PROJECT:
            if not project_id:
                return False
            return project_id in ctx.project_memberships
        return scope == ModelScope.USER

    def enforce_read(self, model: ModelEntry, ctx: AccessContext) -> None:
        """Raise PermissionDeniedError if user cannot read model."""
        if not self.can_read(model, ctx):
            raise PermissionDeniedError(
                ctx.user_id,
                model.model_id,
                "read",
                f"Not a member of {model.scope.value} scope",
            )

    def enforce_write(self, model: ModelEntry, ctx: AccessContext) -> None:
        """Raise PermissionDeniedError if user cannot write model."""
        if not self.can_write(model, ctx):
            raise PermissionDeniedError(
                ctx.user_id,
                model.model_id,
                "modify",
                f"Not authorized to modify {model.scope.value} models",
            )

    def enforce_create(
        self,
        scope: ModelScope,
        ctx: AccessContext,
        project_id: str | None = None,
    ) -> None:
        """Raise PermissionDeniedError if user cannot create in scope."""
        if not self.can_create(scope, ctx, project_id):
            raise PermissionDeniedError(
                ctx.user_id,
                "new_model",
                "create",
                f"Not authorized to create {scope.value} models",
            )
