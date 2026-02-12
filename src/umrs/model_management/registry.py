"""
Model Registry Service (MM-001).

Tracks available models across the platform with metadata,
access control, and lifecycle management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..model_registry.models import (
    ExecutionAuthority,
    ModelEntry,
    ModelRef,
    ModelScope,
    ModelStatus,
    ModelType,
)
from ..model_registry.store import AccessContext, ModelStore

logger = logging.getLogger(__name__)


class ModelVisibility(Enum):
    """Model visibility in the registry."""

    PUBLIC = "public"  # Visible to all users
    PROJECT = "project"  # Visible to project members
    PRIVATE = "private"  # Visible only to owner


@dataclass
class ModelRegistration:
    """Model registration request."""

    name: str
    scope: ModelScope
    model_type: ModelType
    storage_ref: str

    # Optional metadata
    description: Optional[str] = None
    huggingface_id: Optional[str] = None
    version: Optional[str] = None
    tags: Optional[List[str]] = None

    # Ownership
    owner_id: Optional[str] = None
    project_id: Optional[str] = None

    # Serving configuration
    serving_profile: Optional[Dict[str, Any]] = None
    execution_authority: Optional[ExecutionAuthority] = None
    model_ref: Optional[ModelRef] = None


@dataclass
class ModelSearchCriteria:
    """Search criteria for finding models."""

    name_pattern: Optional[str] = None
    scope: Optional[ModelScope] = None
    model_type: Optional[ModelType] = None
    status: Optional[ModelStatus] = None
    tags: Optional[List[str]] = None
    huggingface_id: Optional[str] = None
    owner_id: Optional[str] = None
    project_id: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None


class ModelRegistryService:
    """
    Model Registry Service (MM-001).

    Provides high-level model management operations:
    - Registration and discovery
    - Access control checks
    - Lifecycle management
    - Search and filtering
    """

    def __init__(
        self,
        store: ModelStore,
        default_execution_authority: ExecutionAuthority = ExecutionAuthority.SLURM,
    ):
        """
        Initialize registry service.

        Args:
            store: Underlying model store
            default_execution_authority: Default execution backend
        """
        self.store = store
        self.default_execution_authority = default_execution_authority

    async def register_model(
        self,
        registration: ModelRegistration,
        ctx: AccessContext,
    ) -> ModelEntry:
        """
        Register a new model in the registry.

        Args:
            registration: Model registration details
            ctx: Access context for authorization

        Returns:
            Registered model entry
        """
        logger.info(
            "Registering model '%s' in scope %s by user %s",
            registration.name,
            registration.scope.value,
            ctx.user_id,
        )

        # Validate scope-specific requirements
        if registration.scope == ModelScope.PROJECT and not registration.project_id:
            raise ValueError("project_id required for project-scoped models")
        if registration.scope == ModelScope.USER and not registration.owner_id:
            registration.owner_id = ctx.user_id

        # Build model ref if not provided
        model_ref = registration.model_ref
        if model_ref is None and registration.storage_ref:
            model_ref = ModelRef.filesystem(registration.storage_ref)

        # Register via store
        entry = self.store.register_model(
            name=registration.name,
            scope=registration.scope,
            storage_ref=registration.storage_ref,
            model_type=registration.model_type,
            project_id=registration.project_id,
            owner_id=registration.owner_id or ctx.user_id,
            huggingface_id=registration.huggingface_id,
            description=registration.description,
            serving_profile=registration.serving_profile,
            model_ref=model_ref,
            execution_authority=registration.execution_authority or self.default_execution_authority,
            ctx=ctx,
        )

        logger.info(
            "Model registered: %s (id=%s)",
            registration.name,
            entry.model_id,
        )
        return entry

    async def get_model(
        self,
        model_id: str,
        ctx: AccessContext,
    ) -> ModelEntry:
        """
        Get model by ID with access control.

        Args:
            model_id: Model identifier
            ctx: Access context for authorization

        Returns:
            Model entry if accessible
        """
        return self.store.get_model(model_id, ctx)

    async def get_model_by_name(
        self,
        name: str,
        scope: ModelScope,
        project_id: Optional[str] = None,
        ctx: Optional[AccessContext] = None,
    ) -> Optional[ModelEntry]:
        """
        Get model by name within a scope.

        Args:
            name: Model name
            scope: Model scope
            project_id: Project ID (for project scope)
            ctx: Access context

        Returns:
            Model entry if found, None otherwise
        """
        return self.store.get_model_by_name(name, scope, project_id, ctx)

    async def list_models(
        self,
        ctx: AccessContext,
        criteria: Optional[ModelSearchCriteria] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ModelEntry]:
        """
        List models matching criteria.

        Args:
            ctx: Access context for filtering
            criteria: Optional search criteria
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of accessible models
        """
        criteria = criteria or ModelSearchCriteria()

        return self.store.list_models(
            ctx=ctx,
            scope=criteria.scope,
            model_type=criteria.model_type,
            status=criteria.status,
            project_id=criteria.project_id,
            limit=limit,
            offset=offset,
        )

    async def search_models(
        self,
        query: str,
        ctx: AccessContext,
        limit: int = 20,
    ) -> List[ModelEntry]:
        """
        Search models by name or description.

        Args:
            query: Search query
            ctx: Access context
            limit: Maximum results

        Returns:
            Matching models
        """
        # Get all accessible models and filter
        all_models = self.store.list_models(ctx=ctx, limit=500)

        query_lower = query.lower()
        matches = []

        for model in all_models:
            if query_lower in model.name.lower():
                matches.append(model)
            elif model.description and query_lower in model.description.lower():
                matches.append(model)
            elif model.huggingface_id and query_lower in model.huggingface_id.lower():
                matches.append(model)

            if len(matches) >= limit:
                break

        return matches

    async def update_model_status(
        self,
        model_id: str,
        status: ModelStatus,
        ctx: AccessContext,
    ) -> ModelEntry:
        """
        Update model availability status.

        Args:
            model_id: Model identifier
            status: New status
            ctx: Access context

        Returns:
            Updated model entry
        """
        logger.info(
            "Updating model %s status to %s by user %s",
            model_id,
            status.value,
            ctx.user_id,
        )
        return self.store.update_model(model_id, status=status, ctx=ctx)

    async def deprecate_model(
        self,
        model_id: str,
        ctx: AccessContext,
        replacement_id: Optional[str] = None,
    ) -> ModelEntry:
        """
        Mark model as deprecated.

        Args:
            model_id: Model to deprecate
            ctx: Access context
            replacement_id: Optional replacement model

        Returns:
            Updated model entry
        """
        logger.info(
            "Deprecating model %s by user %s (replacement: %s)",
            model_id,
            ctx.user_id,
            replacement_id,
        )
        return await self.update_model_status(model_id, ModelStatus.DEPRECATED, ctx)

    async def delete_model(
        self,
        model_id: str,
        ctx: AccessContext,
        force: bool = False,
    ) -> bool:
        """
        Delete model from registry.

        Args:
            model_id: Model to delete
            ctx: Access context
            force: Delete even if in use

        Returns:
            True if deleted
        """
        logger.info(
            "Deleting model %s by user %s (force=%s)",
            model_id,
            ctx.user_id,
            force,
        )
        self.store.delete_model(model_id, ctx)
        return True

    async def get_system_models(
        self,
        model_type: Optional[ModelType] = None,
    ) -> List[ModelEntry]:
        """
        Get all system-level models.

        Args:
            model_type: Optional type filter

        Returns:
            List of system models
        """
        # System models are publicly accessible
        ctx = AccessContext(user_id="__system__", is_admin=True)
        return self.store.list_models(
            ctx=ctx,
            scope=ModelScope.SYSTEM,
            model_type=model_type,
            status=ModelStatus.AVAILABLE,
        )

    async def get_project_models(
        self,
        project_id: str,
        ctx: AccessContext,
        include_system: bool = True,
    ) -> List[ModelEntry]:
        """
        Get models available to a project.

        Args:
            project_id: Project identifier
            ctx: Access context
            include_system: Include system models

        Returns:
            List of available models
        """
        models = []

        # Add project-specific models
        project_models = self.store.list_models(
            ctx=ctx,
            scope=ModelScope.PROJECT,
            project_id=project_id,
            status=ModelStatus.AVAILABLE,
        )
        models.extend(project_models)

        # Add system models if requested
        if include_system:
            system_models = await self.get_system_models()
            models.extend(system_models)

        return models
