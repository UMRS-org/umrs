"""
Model Registry Store (U-803, U-804, U-805).

Provides:
- CRUD operations for model catalog entries
- CRUD operations for model policies (U-805)
- Access control (ACL) enforcement
- Path validation for secure serving
- Duplicate detection
- ModelRef resolution (U-808)

See DEVELOPMENT_OBJECTIVES.md Section 1.9 for design details.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import psycopg2
from psycopg2 import sql

from .models import (
    ExecutionAuthority,
    ModelEntry,
    ModelPolicy,
    ModelRef,
    ModelScope,
    ModelStatus,
    ModelType,
    ServingProfile,
    generate_model_id,
    generate_policy_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class ModelNotFoundError(Exception):
    """Raised when a model is not found."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model not found: {model_id}")


class DuplicateModelError(Exception):
    """Raised when a model with the same name already exists in scope."""

    def __init__(self, name: str, scope: ModelScope, project_id: Optional[str] = None):
        self.name = name
        self.scope = scope
        self.project_id = project_id
        if scope == ModelScope.PROJECT:
            super().__init__(
                f"Model '{name}' already exists in project '{project_id}'"
            )
        else:
            super().__init__(f"Model '{name}' already exists in {scope.value} scope")


class PermissionDeniedError(Exception):
    """Raised when user lacks permission to access/modify a model."""

    def __init__(
        self,
        user_id: str,
        model_id: str,
        action: str = "access",
        reason: Optional[str] = None,
    ):
        self.user_id = user_id
        self.model_id = model_id
        self.action = action
        msg = f"User '{user_id}' cannot {action} model '{model_id}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class PathNotAllowedError(Exception):
    """Raised when a storage path is not in the allowed directories."""

    def __init__(self, path: str, allowed_patterns: List[str]):
        self.path = path
        self.allowed_patterns = allowed_patterns
        super().__init__(
            f"Path '{path}' is not allowed. "
            f"Allowed patterns: {', '.join(allowed_patterns)}"
        )


# =============================================================================
# Path Validation (U-804)
# =============================================================================

# Default allowed path patterns for model storage
# These can be overridden via environment variable MODEL_ALLOWED_PATHS
DEFAULT_ALLOWED_PATHS = [
    "/shared/models/system/*",       # System models
    "/shared/projects/*/models/*",   # Project models
    "/shared/users/*/models/*",      # User models
    "/models/*",                     # Development/Docker paths
    "/home/*/models/*",              # Home directory models (dev)
]


@dataclass
class PathValidator:
    """
    Validates that storage paths are within allowed directories.

    This prevents users from serving arbitrary filesystem paths.
    """

    allowed_patterns: List[str]

    def __init__(self, allowed_patterns: Optional[List[str]] = None):
        """
        Initialize path validator.

        Args:
            allowed_patterns: List of glob-like patterns (uses * for wildcard).
                              If None, uses DEFAULT_ALLOWED_PATHS or
                              MODEL_ALLOWED_PATHS environment variable.
        """
        if allowed_patterns is not None:
            self.allowed_patterns = allowed_patterns
        else:
            # Check environment variable
            env_paths = os.environ.get("MODEL_ALLOWED_PATHS")
            if env_paths:
                self.allowed_patterns = [p.strip() for p in env_paths.split(":")]
            else:
                self.allowed_patterns = DEFAULT_ALLOWED_PATHS.copy()

    def is_allowed(self, path: str) -> bool:
        """
        Check if a path matches any allowed pattern.

        Args:
            path: Absolute filesystem path

        Returns:
            True if path is allowed
        """
        # Normalize path
        path = os.path.normpath(path)

        for pattern in self.allowed_patterns:
            if self._match_pattern(path, pattern):
                return True
        return False

    def validate(self, path: str) -> None:
        """
        Validate a path, raising PathNotAllowedError if not allowed.

        Args:
            path: Absolute filesystem path

        Raises:
            PathNotAllowedError: If path is not in allowed directories
        """
        if not self.is_allowed(path):
            raise PathNotAllowedError(path, self.allowed_patterns)

    def _match_pattern(self, path: str, pattern: str) -> bool:
        """
        Match path against a glob-like pattern.

        Supports:
        - * matches any single path component
        - ** not supported (use * for each level)
        """
        # Convert glob pattern to regex
        # Escape special regex chars except *
        regex_pattern = re.escape(pattern)
        # Replace escaped \* with regex for "any path component"
        regex_pattern = regex_pattern.replace(r"\*", r"[^/]+")
        # Anchor pattern
        regex_pattern = f"^{regex_pattern}$"

        return bool(re.match(regex_pattern, path))

    def validate_scope_path(
        self,
        path: str,
        scope: ModelScope,
        owner_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """
        Validate that path matches the claimed scope.

        For example, a user model should be in /shared/users/<owner_id>/models/,
        not in /shared/models/system/.

        Args:
            path: Storage path
            scope: Claimed model scope
            owner_id: Owner user ID (for user scope)
            project_id: Project ID (for project scope)

        Raises:
            PathNotAllowedError: If path doesn't match scope
        """
        # First check general allowlist
        self.validate(path)

        # Additional scope-specific validation
        normalized = os.path.normpath(path)

        if scope == ModelScope.SYSTEM:
            # System models should be in system directories
            valid_prefixes = ["/shared/models/system", "/models"]
            if not any(normalized.startswith(p) for p in valid_prefixes):
                raise PathNotAllowedError(
                    path,
                    [f"{p}/*" for p in valid_prefixes],
                )

        elif scope == ModelScope.PROJECT:
            if not project_id:
                raise ValueError("project_id required for project scope")
            # Project models should be in project directory
            expected_prefix = f"/shared/projects/{project_id}/models"
            if not normalized.startswith(expected_prefix):
                # Also allow generic project patterns for dev
                if not normalized.startswith("/shared/projects/"):
                    raise PathNotAllowedError(path, [f"{expected_prefix}/*"])

        elif scope == ModelScope.USER:
            if not owner_id:
                raise ValueError("owner_id required for user scope")
            # User models should be in user directory
            valid_prefixes = [
                f"/shared/users/{owner_id}/models",
                f"/home/{owner_id}/models",
            ]
            if not any(normalized.startswith(p) for p in valid_prefixes):
                raise PathNotAllowedError(
                    path,
                    [f"{p}/*" for p in valid_prefixes],
                )


# =============================================================================
# ACL Enforcement (U-803)
# =============================================================================


@dataclass
class AccessContext:
    """Context for access control decisions."""

    user_id: str
    is_admin: bool = False
    project_memberships: Set[str] = None  # Projects user belongs to
    admin_project_ids: Set[str] = None    # Projects where user is admin (U-815)

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
            return True  # All users can read system models

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
            return False  # Only admins can write system models

        if model.scope == ModelScope.PROJECT:
            # Project members can write (could add project_admin check later)
            return model.project_id in ctx.project_memberships

        if model.scope == ModelScope.USER:
            return model.owner_id == ctx.user_id

        return False

    def can_create(
        self,
        scope: ModelScope,
        ctx: AccessContext,
        project_id: Optional[str] = None,
    ) -> bool:
        """Check if user can create a model in the given scope."""
        if ctx.is_admin:
            return True

        if scope == ModelScope.SYSTEM:
            return False  # Only admins can create system models

        if scope == ModelScope.PROJECT:
            if not project_id:
                return False
            return project_id in ctx.project_memberships

        if scope == ModelScope.USER:
            return True  # Users can create their own models

        return False

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
        project_id: Optional[str] = None,
    ) -> None:
        """Raise PermissionDeniedError if user cannot create in scope."""
        if not self.can_create(scope, ctx, project_id):
            raise PermissionDeniedError(
                ctx.user_id,
                "new_model",
                "create",
                f"Not authorized to create {scope.value} models",
            )


# =============================================================================
# Model Store
# =============================================================================


class ModelStore:
    """
    Model Registry data access layer.

    Provides CRUD operations with ACL enforcement and path validation.
    """

    def __init__(
        self,
        conn: psycopg2.extensions.connection,
        path_validator: Optional[PathValidator] = None,
        acl_enforcer: Optional[ACLEnforcer] = None,
    ):
        """
        Initialize model store.

        Args:
            conn: PostgreSQL connection
            path_validator: Path validator instance (uses default if None)
            acl_enforcer: ACL enforcer instance (uses default if None)
        """
        self.conn = conn
        self.path_validator = path_validator or PathValidator()
        self.acl = acl_enforcer or ACLEnforcer()

    # -------------------------------------------------------------------------
    # Create
    # -------------------------------------------------------------------------

    def create_model(
        self,
        name: str,
        scope: ModelScope,
        storage_ref: str,
        model_type: ModelType = ModelType.LLM,
        owner_id: Optional[str] = None,
        project_id: Optional[str] = None,
        huggingface_id: Optional[str] = None,
        description: Optional[str] = None,
        serving_profile: Optional[ServingProfile] = None,
        status: ModelStatus = ModelStatus.AVAILABLE,
        model_ref: Optional[ModelRef] = None,
        execution_authority_default: ExecutionAuthority = ExecutionAuthority.COMPOSE,
        ctx: Optional[AccessContext] = None,
    ) -> ModelEntry:
        """
        Create a new model entry.

        Args:
            name: Model name (unique within scope)
            scope: Visibility scope
            storage_ref: Filesystem path to model weights (legacy, prefer model_ref)
            model_type: LLM or embedding
            owner_id: Owner user ID (required for user scope)
            project_id: Project ID (required for project scope)
            huggingface_id: Source HuggingFace model ID
            description: Human-readable description
            serving_profile: vLLM/LiteLLM configuration
            status: Initial status
            model_ref: ModelRef abstraction (U-808) - preferred over storage_ref
            execution_authority_default: Default execution authority (U-807)
            ctx: Access context for authorization

        Returns:
            Created ModelEntry

        Raises:
            PermissionDeniedError: If user cannot create in scope
            PathNotAllowedError: If storage path is not allowed
            DuplicateModelError: If model name already exists in scope
        """
        # Enforce ACL
        if ctx:
            self.acl.enforce_create(scope, ctx, project_id)

        # Validate storage path
        self.path_validator.validate_scope_path(
            storage_ref, scope, owner_id, project_id
        )

        # Generate ID
        model_id = generate_model_id()
        serving_profile = serving_profile or ServingProfile()

        # Create entry
        model = ModelEntry(
            model_id=model_id,
            name=name,
            scope=scope,
            owner_id=owner_id,
            project_id=project_id,
            storage_ref=storage_ref,
            model_ref=model_ref,
            model_type=model_type,
            huggingface_id=huggingface_id,
            description=description,
            serving_profile=serving_profile,
            execution_authority_default=execution_authority_default,
            status=status,
        )

        # Insert into database
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_catalog_entries (
                        model_id, name, scope, owner_id, project_id,
                        storage_ref, model_type, huggingface_id, description,
                        serving_profile, status, model_ref_json, execution_authority_default
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    RETURNING created_at, updated_at
                    """,
                    (
                        model.model_id,
                        model.name,
                        model.scope.value,
                        model.owner_id,
                        model.project_id,
                        model.storage_ref,
                        model.model_type.value,
                        model.huggingface_id,
                        model.description,
                        json.dumps(model.serving_profile.to_dict()),
                        model.status.value,
                        json.dumps(model.model_ref.to_dict()) if model.model_ref else None,
                        model.execution_authority_default.value,
                    ),
                )
                row = cur.fetchone()
                model.created_at = row[0]
                model.updated_at = row[1]
                self.conn.commit()

        except psycopg2.errors.UniqueViolation:
            self.conn.rollback()
            raise DuplicateModelError(name, scope, project_id)

        logger.info(
            "Created model %s (%s) in %s scope",
            model.model_id,
            name,
            scope.value,
        )
        return model

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    def get_model(
        self,
        model_id: str,
        ctx: Optional[AccessContext] = None,
    ) -> ModelEntry:
        """
        Get a model by ID.

        Args:
            model_id: Model identifier
            ctx: Access context for authorization

        Returns:
            ModelEntry

        Raises:
            ModelNotFoundError: If model doesn't exist
            PermissionDeniedError: If user cannot access model
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT model_id, name, scope, owner_id, project_id,
                       storage_ref, model_type, huggingface_id, description,
                       serving_profile, status, model_ref_json, execution_authority_default,
                       created_at, updated_at
                FROM model_catalog_entries
                WHERE model_id = %s
                """,
                (model_id,),
            )
            row = cur.fetchone()

        if row is None:
            raise ModelNotFoundError(model_id)

        model = ModelEntry.from_row(row)

        # Enforce ACL
        if ctx:
            self.acl.enforce_read(model, ctx)

        return model

    def get_model_by_name(
        self,
        name: str,
        scope: ModelScope,
        project_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        ctx: Optional[AccessContext] = None,
    ) -> ModelEntry:
        """
        Get a model by name within a scope.

        Args:
            name: Model name
            scope: Model scope
            project_id: Project ID (for project scope)
            owner_id: Owner ID (for user scope)
            ctx: Access context

        Returns:
            ModelEntry

        Raises:
            ModelNotFoundError: If model doesn't exist
            PermissionDeniedError: If user cannot access model
        """
        with self.conn.cursor() as cur:
            if scope == ModelScope.SYSTEM:
                cur.execute(
                    """
                    SELECT model_id, name, scope, owner_id, project_id,
                           storage_ref, model_type, huggingface_id, description,
                           serving_profile, status, model_ref_json, execution_authority_default,
                           created_at, updated_at
                    FROM model_catalog_entries
                    WHERE name = %s AND scope = 'system'
                    """,
                    (name,),
                )
            elif scope == ModelScope.PROJECT:
                cur.execute(
                    """
                    SELECT model_id, name, scope, owner_id, project_id,
                           storage_ref, model_type, huggingface_id, description,
                           serving_profile, status, model_ref_json, execution_authority_default,
                           created_at, updated_at
                    FROM model_catalog_entries
                    WHERE name = %s AND scope = 'project' AND project_id = %s
                    """,
                    (name, project_id),
                )
            else:  # USER
                cur.execute(
                    """
                    SELECT model_id, name, scope, owner_id, project_id,
                           storage_ref, model_type, huggingface_id, description,
                           serving_profile, status, model_ref_json, execution_authority_default,
                           created_at, updated_at
                    FROM model_catalog_entries
                    WHERE name = %s AND scope = 'user' AND owner_id = %s
                    """,
                    (name, owner_id),
                )

            row = cur.fetchone()

        if row is None:
            raise ModelNotFoundError(f"{scope.value}:{name}")

        model = ModelEntry.from_row(row)

        if ctx:
            self.acl.enforce_read(model, ctx)

        return model

    def list_models(
        self,
        ctx: AccessContext,
        scope: Optional[ModelScope] = None,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        project_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ModelEntry]:
        """
        List models visible to the user.

        Args:
            ctx: Access context (required for filtering)
            scope: Filter by scope
            model_type: Filter by type (llm/embedding)
            status: Filter by status
            project_id: Filter by project (also limits to project scope)
            limit: Max results
            offset: Pagination offset

        Returns:
            List of ModelEntry visible to user
        """
        conditions = []
        params: List[Any] = []

        # Build visibility filter based on user context
        # Users can see:
        # - All system models
        # - Project models for their projects
        # - Their own user models
        if not ctx.is_admin:
            visibility_parts = ["scope = 'system'"]

            if ctx.project_memberships:
                placeholders = ", ".join(["%s"] * len(ctx.project_memberships))
                visibility_parts.append(
                    f"(scope = 'project' AND project_id IN ({placeholders}))"
                )
                params.extend(ctx.project_memberships)

            visibility_parts.append("(scope = 'user' AND owner_id = %s)")
            params.append(ctx.user_id)

            conditions.append(f"({' OR '.join(visibility_parts)})")

        # Apply filters
        if scope:
            conditions.append("scope = %s")
            params.append(scope.value)

        if model_type:
            conditions.append("model_type = %s")
            params.append(model_type.value)

        if status:
            conditions.append("status = %s")
            params.append(status.value)

        if project_id:
            conditions.append("project_id = %s")
            params.append(project_id)

        # Build query
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT model_id, name, scope, owner_id, project_id,
                   storage_ref, model_type, huggingface_id, description,
                   serving_profile, status, model_ref_json, execution_authority_default,
                   created_at, updated_at
            FROM model_catalog_entries
            WHERE {where_clause}
            ORDER BY scope, name
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [ModelEntry.from_row(row) for row in rows]

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------

    def update_model(
        self,
        model_id: str,
        ctx: AccessContext,
        description: Optional[str] = None,
        serving_profile: Optional[ServingProfile] = None,
        status: Optional[ModelStatus] = None,
    ) -> ModelEntry:
        """
        Update model metadata.

        Args:
            model_id: Model to update
            ctx: Access context
            description: New description
            serving_profile: New serving profile
            status: New status

        Returns:
            Updated ModelEntry

        Raises:
            ModelNotFoundError: If model doesn't exist
            PermissionDeniedError: If user cannot modify model
        """
        # Get current model and check permissions
        model = self.get_model(model_id)
        self.acl.enforce_write(model, ctx)

        # Build update
        updates = []
        params: List[Any] = []

        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if serving_profile is not None:
            updates.append("serving_profile = %s")
            params.append(json.dumps(serving_profile.to_dict()))

        if status is not None:
            updates.append("status = %s")
            params.append(status.value)

        if not updates:
            return model  # Nothing to update

        updates.append("updated_at = NOW()")
        params.append(model_id)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE model_catalog_entries
                SET {', '.join(updates)}
                WHERE model_id = %s
                RETURNING updated_at
                """,
                params,
            )
            row = cur.fetchone()
            self.conn.commit()

        # Refresh model
        return self.get_model(model_id)

    def update_status(
        self,
        model_id: str,
        status: ModelStatus,
        ctx: Optional[AccessContext] = None,
    ) -> None:
        """
        Update model status (convenience method).

        This is commonly used after download jobs complete.
        """
        if ctx:
            model = self.get_model(model_id)
            self.acl.enforce_write(model, ctx)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE model_catalog_entries
                SET status = %s, updated_at = NOW()
                WHERE model_id = %s
                """,
                (status.value, model_id),
            )
            if cur.rowcount == 0:
                raise ModelNotFoundError(model_id)
            self.conn.commit()

        logger.info("Updated model %s status to %s", model_id, status.value)

    # -------------------------------------------------------------------------
    # Delete
    # -------------------------------------------------------------------------

    def delete_model(
        self,
        model_id: str,
        ctx: AccessContext,
    ) -> None:
        """
        Delete a model entry.

        Note: This only removes the catalog entry, not the actual model files.

        Args:
            model_id: Model to delete
            ctx: Access context

        Raises:
            ModelNotFoundError: If model doesn't exist
            PermissionDeniedError: If user cannot delete model
        """
        model = self.get_model(model_id)
        self.acl.enforce_write(model, ctx)

        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM model_catalog_entries WHERE model_id = %s",
                (model_id,),
            )
            self.conn.commit()

        logger.info("Deleted model %s (%s)", model_id, model.name)

    # -------------------------------------------------------------------------
    # Publish (U-815)
    # -------------------------------------------------------------------------

    def publish_model(
        self,
        model_id: str,
        target_project_id: str,
        ctx: AccessContext,
        new_name: Optional[str] = None,
    ) -> ModelEntry:
        """
        Publish a user model to project scope (U-815).

        This promotes a user-owned model to be accessible by all project members.
        Only the model owner can publish, and they must be a project admin.

        Args:
            model_id: Model to publish
            target_project_id: Project to publish to
            ctx: Access context (must be project admin)
            new_name: Optional new name for the published model

        Returns:
            Updated model entry with project scope

        Raises:
            ModelNotFoundError: If model doesn't exist
            PermissionDeniedError: If user cannot publish this model
            DuplicateModelError: If model name already exists in project
        """
        model = self.get_model(model_id)

        # Must be owner of the model
        if model.owner_id != ctx.user_id:
            raise PermissionDeniedError(
                ctx.user_id,
                model_id,
                "publish",
                "Only the model owner can publish",
            )

        # Model must be user-scoped
        if model.scope != ModelScope.USER:
            raise PermissionDeniedError(
                ctx.user_id,
                model_id,
                "publish",
                f"Can only publish user models, not {model.scope.value}",
            )

        # User must be admin for target project
        if not ctx.is_admin and target_project_id not in ctx.admin_project_ids:
            raise PermissionDeniedError(
                ctx.user_id,
                model_id,
                "publish",
                f"Must be admin of project {target_project_id} to publish",
            )

        # Determine final name
        publish_name = new_name if new_name else model.name

        # Check for duplicate in project scope
        existing = self._find_model_by_name(publish_name, ModelScope.PROJECT, target_project_id)
        if existing is not None:
            raise DuplicateModelError(publish_name, ModelScope.PROJECT, target_project_id)

        # Update model to project scope
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE model_catalog_entries
                SET scope = %s,
                    project_id = %s,
                    name = %s,
                    owner_id = NULL,
                    updated_at = NOW()
                WHERE model_id = %s
                """,
                (ModelScope.PROJECT.value, target_project_id, publish_name, model_id),
            )
            self.conn.commit()

        logger.info(
            "Published model %s (%s) to project %s as %s",
            model_id, model.name, target_project_id, publish_name,
        )

        return self.get_model(model_id)

    def _find_model_by_name(
        self,
        name: str,
        scope: ModelScope,
        project_id: Optional[str] = None,
    ) -> Optional[ModelEntry]:
        """Find a model by name in scope (helper for duplicate checking)."""
        with self.conn.cursor() as cur:
            if scope == ModelScope.PROJECT:
                cur.execute(
                    """
                    SELECT * FROM model_catalog_entries
                    WHERE name = %s AND scope = %s AND project_id = %s
                    """,
                    (name, scope.value, project_id),
                )
            elif scope == ModelScope.SYSTEM:
                cur.execute(
                    """
                    SELECT * FROM model_catalog_entries
                    WHERE name = %s AND scope = %s
                    """,
                    (name, scope.value),
                )
            else:
                return None

            row = cur.fetchone()
            if row:
                return self._row_to_entry(row, cur.description)
        return None

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def resolve_model_path(
        self,
        model_id: str,
        ctx: AccessContext,
    ) -> str:
        """
        Resolve model ID to validated storage path.

        This is used by endpoint provisioning to get the model path
        from a model_id, with ACL enforcement.

        Args:
            model_id: Model identifier
            ctx: Access context

        Returns:
            Validated storage_ref path

        Raises:
            ModelNotFoundError: If model doesn't exist
            PermissionDeniedError: If user cannot access model
        """
        model = self.get_model(model_id, ctx)

        if model.status != ModelStatus.AVAILABLE:
            raise PermissionDeniedError(
                ctx.user_id,
                model_id,
                "use",
                f"Model status is {model.status.value}, not available",
            )

        return model.storage_ref

    def model_exists(self, model_id: str) -> bool:
        """Check if a model exists."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM model_catalog_entries WHERE model_id = %s",
                (model_id,),
            )
            return cur.fetchone() is not None

    # -------------------------------------------------------------------------
    # Model Policies (U-805)
    # -------------------------------------------------------------------------

    def create_policy(
        self,
        model_name: str,
        allowed_projects: Optional[List[str]] = None,
        denied_projects: Optional[List[str]] = None,
        max_concurrent_endpoints: Optional[int] = None,
        max_tokens_per_day: Optional[int] = None,
        priority: int = 0,
        warm_pool_min: int = 0,
        warm_pool_max: int = 0,
        warm_pool_ttl_seconds: int = 3600,
        enable_warm_pool: bool = False,
        is_system_model: bool = False,
        default_serving_profile: Optional[ServingProfile] = None,
        ctx: Optional[AccessContext] = None,
    ) -> ModelPolicy:
        """
        Create a model policy (U-805).

        Args:
            model_name: Model name this policy applies to
            allowed_projects: Projects allowed to use this model
            denied_projects: Projects denied access (takes precedence)
            max_concurrent_endpoints: Max concurrent endpoints
            max_tokens_per_day: Daily token quota
            priority: Scheduling priority
            warm_pool_min: Minimum warm endpoints
            warm_pool_max: Maximum warm endpoints
            warm_pool_ttl_seconds: Warm endpoint TTL
            enable_warm_pool: Whether to enable warm pooling
            is_system_model: Mark as system-level shared model
            default_serving_profile: Default serving configuration
            ctx: Access context (admin only)

        Returns:
            Created ModelPolicy

        Raises:
            PermissionDeniedError: If non-admin tries to create policy
        """
        if ctx and not ctx.is_admin:
            raise PermissionDeniedError(
                ctx.user_id, "policy", "create", "Admin only"
            )

        policy_id = generate_policy_id()
        default_serving_profile = default_serving_profile or ServingProfile()

        policy = ModelPolicy(
            policy_id=policy_id,
            model_name=model_name,
            allowed_projects=allowed_projects or [],
            denied_projects=denied_projects or [],
            max_concurrent_endpoints=max_concurrent_endpoints,
            max_tokens_per_day=max_tokens_per_day,
            priority=priority,
            warm_pool_min=warm_pool_min,
            warm_pool_max=warm_pool_max,
            warm_pool_ttl_seconds=warm_pool_ttl_seconds,
            enable_warm_pool=enable_warm_pool,
            is_system_model=is_system_model,
            default_serving_profile=default_serving_profile,
        )

        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO model_policies (
                    policy_id, model_name, allowed_projects, denied_projects,
                    max_concurrent_endpoints, max_tokens_per_day, priority,
                    warm_pool_min, warm_pool_max, warm_pool_ttl_seconds, enable_warm_pool,
                    is_system_model, default_serving_profile
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING created_at, updated_at
                """,
                (
                    policy.policy_id,
                    policy.model_name,
                    policy.allowed_projects,
                    policy.denied_projects,
                    policy.max_concurrent_endpoints,
                    policy.max_tokens_per_day,
                    policy.priority,
                    policy.warm_pool_min,
                    policy.warm_pool_max,
                    policy.warm_pool_ttl_seconds,
                    policy.enable_warm_pool,
                    policy.is_system_model,
                    json.dumps(policy.default_serving_profile.to_dict()),
                ),
            )
            row = cur.fetchone()
            policy.created_at = row[0]
            policy.updated_at = row[1]
            self.conn.commit()

        logger.info("Created policy %s for model %s", policy_id, model_name)
        return policy

    def get_policy(self, policy_id: str) -> ModelPolicy:
        """Get a policy by ID."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT policy_id, model_name, allowed_projects, denied_projects,
                       max_concurrent_endpoints, max_tokens_per_day, priority,
                       warm_pool_min, warm_pool_max, warm_pool_ttl_seconds, enable_warm_pool,
                       is_system_model, default_serving_profile, created_at, updated_at
                FROM model_policies
                WHERE policy_id = %s
                """,
                (policy_id,),
            )
            row = cur.fetchone()

        if row is None:
            raise ModelNotFoundError(f"policy:{policy_id}")

        return ModelPolicy.from_row(row)

    def get_policy_by_model_name(self, model_name: str) -> Optional[ModelPolicy]:
        """Get policy for a model by name. Returns None if no policy exists."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT policy_id, model_name, allowed_projects, denied_projects,
                       max_concurrent_endpoints, max_tokens_per_day, priority,
                       warm_pool_min, warm_pool_max, warm_pool_ttl_seconds, enable_warm_pool,
                       is_system_model, default_serving_profile, created_at, updated_at
                FROM model_policies
                WHERE model_name = %s
                """,
                (model_name,),
            )
            row = cur.fetchone()

        if row is None:
            return None

        return ModelPolicy.from_row(row)

    def list_policies(
        self,
        is_system_model: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ModelPolicy]:
        """List model policies."""
        conditions = []
        params: List[Any] = []

        if is_system_model is not None:
            conditions.append("is_system_model = %s")
            params.append(is_system_model)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT policy_id, model_name, allowed_projects, denied_projects,
                   max_concurrent_endpoints, max_tokens_per_day, priority,
                   warm_pool_min, warm_pool_max, warm_pool_ttl_seconds, enable_warm_pool,
                   is_system_model, default_serving_profile, created_at, updated_at
            FROM model_policies
            WHERE {where_clause}
            ORDER BY model_name
            LIMIT %s OFFSET %s
        """
        params.extend([limit, offset])

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [ModelPolicy.from_row(row) for row in rows]

    def update_policy(
        self,
        policy_id: str,
        allowed_projects: Optional[List[str]] = None,
        denied_projects: Optional[List[str]] = None,
        max_concurrent_endpoints: Optional[int] = None,
        enable_warm_pool: Optional[bool] = None,
        warm_pool_min: Optional[int] = None,
        warm_pool_max: Optional[int] = None,
        ctx: Optional[AccessContext] = None,
    ) -> ModelPolicy:
        """
        Update a model policy.

        Args:
            policy_id: Policy to update
            allowed_projects: New allowed projects list
            denied_projects: New denied projects list
            max_concurrent_endpoints: New max concurrent endpoints
            enable_warm_pool: Enable/disable warm pooling
            warm_pool_min: New minimum warm endpoints
            warm_pool_max: New maximum warm endpoints
            ctx: Access context (admin only)

        Returns:
            Updated ModelPolicy

        Raises:
            ModelNotFoundError: If policy doesn't exist
            PermissionDeniedError: If non-admin tries to update
        """
        if ctx and not ctx.is_admin:
            raise PermissionDeniedError(
                ctx.user_id, policy_id, "update", "Admin only"
            )

        # Check exists
        self.get_policy(policy_id)

        updates = []
        params: List[Any] = []

        if allowed_projects is not None:
            updates.append("allowed_projects = %s")
            params.append(allowed_projects)

        if denied_projects is not None:
            updates.append("denied_projects = %s")
            params.append(denied_projects)

        if max_concurrent_endpoints is not None:
            updates.append("max_concurrent_endpoints = %s")
            params.append(max_concurrent_endpoints)

        if enable_warm_pool is not None:
            updates.append("enable_warm_pool = %s")
            params.append(enable_warm_pool)

        if warm_pool_min is not None:
            updates.append("warm_pool_min = %s")
            params.append(warm_pool_min)

        if warm_pool_max is not None:
            updates.append("warm_pool_max = %s")
            params.append(warm_pool_max)

        if not updates:
            return self.get_policy(policy_id)

        updates.append("updated_at = NOW()")
        params.append(policy_id)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE model_policies
                SET {', '.join(updates)}
                WHERE policy_id = %s
                """,
                params,
            )
            self.conn.commit()

        return self.get_policy(policy_id)

    def delete_policy(self, policy_id: str, ctx: Optional[AccessContext] = None) -> None:
        """Delete a model policy."""
        if ctx and not ctx.is_admin:
            raise PermissionDeniedError(
                ctx.user_id, policy_id, "delete", "Admin only"
            )

        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM model_policies WHERE policy_id = %s",
                (policy_id,),
            )
            if cur.rowcount == 0:
                raise ModelNotFoundError(f"policy:{policy_id}")
            self.conn.commit()

        logger.info("Deleted policy %s", policy_id)

    def update_policy_by_model_name(
        self,
        model_name: str,
        allowed_projects: Optional[List[str]] = None,
        denied_projects: Optional[List[str]] = None,
        max_concurrent_endpoints: Optional[int] = None,
        enable_warm_pool: Optional[bool] = None,
        warm_pool_min: Optional[int] = None,
        warm_pool_max: Optional[int] = None,
        ctx: Optional[AccessContext] = None,
    ) -> ModelPolicy:
        """
        Update a model policy by model name (U-820).

        Args:
            model_name: Model name whose policy to update
            allowed_projects: New allowed projects list
            denied_projects: New denied projects list
            max_concurrent_endpoints: New max concurrent endpoints
            enable_warm_pool: Enable/disable warm pooling
            warm_pool_min: New minimum warm endpoints
            warm_pool_max: New maximum warm endpoints
            ctx: Access context (admin only)

        Returns:
            Updated ModelPolicy

        Raises:
            ModelNotFoundError: If no policy exists for this model
            PermissionDeniedError: If non-admin tries to update
        """
        if ctx and not ctx.is_admin:
            raise PermissionDeniedError(
                ctx.user_id, model_name, "update_policy", "Admin only"
            )

        policy = self.get_policy_by_model_name(model_name)
        if policy is None:
            raise ModelNotFoundError(f"policy for model: {model_name}")

        return self.update_policy(
            policy_id=policy.policy_id,
            allowed_projects=allowed_projects,
            denied_projects=denied_projects,
            max_concurrent_endpoints=max_concurrent_endpoints,
            enable_warm_pool=enable_warm_pool,
            warm_pool_min=warm_pool_min,
            warm_pool_max=warm_pool_max,
            ctx=ctx,
        )

    def create_system_model(
        self,
        name: str,
        storage_ref: str,
        model_type: ModelType = ModelType.LLM,
        huggingface_id: Optional[str] = None,
        description: Optional[str] = None,
        serving_profile: Optional[ServingProfile] = None,
        execution_authority_default: ExecutionAuthority = ExecutionAuthority.COMPOSE,
        warm_pool_min: int = 0,
        warm_pool_max: int = 1,
        warm_pool_ttl_seconds: int = 3600,
        enable_warm_pool: bool = False,
        allowed_projects: Optional[List[str]] = None,
        denied_projects: Optional[List[str]] = None,
        max_concurrent_endpoints: Optional[int] = None,
        priority: int = 0,
        ctx: Optional[AccessContext] = None,
    ) -> tuple[ModelEntry, ModelPolicy]:
        """
        Register a system-level model with an associated policy (U-817).

        This is an atomic operation that creates both the model entry
        and its policy in a single transaction.

        Args:
            name: Model name (unique in system scope)
            storage_ref: Filesystem path to model weights
            model_type: Model type (llm or embedding)
            huggingface_id: Source HuggingFace model ID
            description: Model description
            serving_profile: Serving configuration
            execution_authority_default: Default execution authority
            warm_pool_min: Minimum warm endpoints
            warm_pool_max: Maximum warm endpoints
            warm_pool_ttl_seconds: Warm endpoint TTL
            enable_warm_pool: Enable warm pooling
            allowed_projects: Projects allowed to use model
            denied_projects: Projects denied access
            max_concurrent_endpoints: Max concurrent endpoints
            priority: Scheduling priority
            ctx: Access context (admin only)

        Returns:
            Tuple of (ModelEntry, ModelPolicy)

        Raises:
            PermissionDeniedError: If non-admin tries to create
            DuplicateModelError: If model already exists
            PathNotAllowedError: If storage_ref not in allowed paths
        """
        if ctx and not ctx.is_admin:
            raise PermissionDeniedError(
                ctx.user_id, name, "create_system_model", "Admin only"
            )

        # Validate path
        self.path_validator.validate(storage_ref)

        # Check for duplicates
        if self._check_duplicate(name, ModelScope.SYSTEM, None, ctx.user_id if ctx else None):
            raise DuplicateModelError(name, ModelScope.SYSTEM)

        model_id = generate_model_id()
        policy_id = generate_policy_id()

        profile = serving_profile or ServingProfile()

        try:
            with self.conn.cursor() as cur:
                # Create model entry
                cur.execute(
                    """
                    INSERT INTO model_catalog_entries (
                        model_id, name, scope, storage_ref, model_type,
                        huggingface_id, description, serving_profile,
                        execution_authority_default, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        model_id,
                        name,
                        ModelScope.SYSTEM.value,
                        storage_ref,
                        model_type.value,
                        huggingface_id,
                        description,
                        json.dumps(profile.to_dict()),
                        execution_authority_default.value,
                        ModelStatus.AVAILABLE.value,
                    ),
                )

                # Create associated policy
                cur.execute(
                    """
                    INSERT INTO model_policies (
                        policy_id, model_name, allowed_projects, denied_projects,
                        max_concurrent_endpoints, priority,
                        warm_pool_min, warm_pool_max, warm_pool_ttl_seconds,
                        enable_warm_pool, is_system_model, default_serving_profile
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        policy_id,
                        name,
                        allowed_projects or [],
                        denied_projects or [],
                        max_concurrent_endpoints,
                        priority,
                        warm_pool_min,
                        warm_pool_max,
                        warm_pool_ttl_seconds,
                        enable_warm_pool,
                        True,  # is_system_model
                        json.dumps(profile.to_dict()),
                    ),
                )

                self.conn.commit()

            logger.info(
                "Created system model %s (%s) with policy %s",
                model_id, name, policy_id
            )

            return (
                self.get_model(model_id, ctx),
                self.get_policy(policy_id),
            )

        except psycopg2.Error as e:
            self.conn.rollback()
            raise

    def check_policy_access(
        self,
        model_name: str,
        project_id: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a project has access to a model based on policy.

        Args:
            model_name: Model name to check
            project_id: Project requesting access

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        policy = self.get_policy_by_model_name(model_name)

        if policy is None:
            # No policy = allowed by default
            return True, None

        if not policy.is_project_allowed(project_id):
            if project_id in policy.denied_projects:
                return False, f"Project {project_id} is denied access to model {model_name}"
            return False, f"Project {project_id} is not in allowed list for model {model_name}"

        return True, None
