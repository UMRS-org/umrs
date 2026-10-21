"""Access control enforcement for model operations (U-803).

Fetch-path authorization (ADR-041): when the static scope tier denies a
read, an optional grant reader is consulted — a live, version-matching,
unexpired asset grant authorizes the read. Writes are never grant-authorized
(ownership stays with the static tier); no grant reader attached behaves
exactly as the legacy static ladder (fail-closed default is unchanged).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..models import ModelEntry, ModelScope
from .errors import HiddenModelError, PermissionDeniedError

if TYPE_CHECKING:  # avoids a hard import; grants stays a soft dependency
    from umrs.grants import AssetGrant

# (asset_kind, asset_id) -> live grants for that asset ([] when none/unwired)
GrantReader = Callable[[str, str], "list[AssetGrant]"]


@dataclass
class AccessContext:
    """Context for access control decisions."""

    user_id: str
    is_admin: bool = False
    project_memberships: set[str] = None  # Projects user belongs to
    admin_project_ids: set[str] = None  # Projects where user is admin (U-815)

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
    - Any scope: readable with a live matching asset grant (ADR-041)
    """

    def __init__(self, grant_reader: GrantReader | None = None) -> None:
        self._grant_reader = grant_reader

    def _grant_allows(self, model: ModelEntry, ctx: AccessContext) -> bool:
        """True when a live grant authorizes this read (deny path only)."""
        if self._grant_reader is None:
            return False
        from umrs.grants import grant_authorizes

        now = datetime.now(timezone.utc)
        grants = self._grant_reader("model", model.model_id)
        for grant in grants:
            if grant_authorizes(
                grant,
                asset_kind="model",
                asset_id=model.model_id,
                user_id=ctx.user_id,
                project_id="",
                now=now,
            ):
                return True
            for project_id in ctx.project_memberships:
                if grant_authorizes(
                    grant,
                    asset_kind="model",
                    asset_id=model.model_id,
                    user_id=ctx.user_id,
                    project_id=project_id,
                    now=now,
                ):
                    return True
        return False

    def read_reason(self, model: ModelEntry, ctx: AccessContext) -> str:
        """WHY safe reading holds (WP-06 r2 console badge source).

        Honest one-token surface consumed verbatim by UI badges. Returns the
        grant chip exactly when a *live grant* is what authorizes the read —
        scope ownership is the normal case. 'none' when no read holds.
        """
        if ctx.is_admin:
            return "admin"
        if model.scope == ModelScope.SYSTEM:
            return "system"
        if (
            model.scope == ModelScope.PROJECT
            and model.project_id in ctx.project_memberships
        ):
            return "project"
        if model.scope == ModelScope.USER and model.owner_id == ctx.user_id:
            return "user"
        if self._grant_allows(model, ctx):
            return "grant"
        return "none"

    def can_read(self, model: ModelEntry, ctx: AccessContext) -> bool:
        """Check if user can read/use a model."""
        if ctx.is_admin:
            return True
        if model.scope == ModelScope.SYSTEM:
            return True
        if model.scope == ModelScope.PROJECT:
            if model.project_id in ctx.project_memberships:
                return True
            return self._grant_allows(model, ctx)
        if model.scope == ModelScope.USER:
            if model.owner_id == ctx.user_id:
                return True
            return self._grant_allows(model, ctx)
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

    def enforce_read(
        self, model: ModelEntry, ctx: AccessContext, lookup_key: str | None = None
    ) -> None:
        """Raise HiddenModelError (reads as not found) if user cannot read model.

        ``lookup_key`` is what the caller looked the model up by (its id by
        default), so the refusal names exactly what an unknown key names.
        """
        if not self.can_read(model, ctx):
            raise HiddenModelError(lookup_key or model.model_id, ctx.user_id)

    def enforce_write(self, model: ModelEntry, ctx: AccessContext) -> None:
        """Raise PermissionDeniedError if user cannot write model.

        A model the caller cannot even read stays hidden (HiddenModelError);
        only a model the caller can see is refused with a 403-style error.
        """
        self.enforce_read(model, ctx)
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
