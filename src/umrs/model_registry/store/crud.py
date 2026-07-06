"""
Model Registry data access layer — ModelStore (U-803, U-804, U-805).

CRUD operations for model catalog entries and model policies,
with ACL enforcement and path validation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import psycopg2

from umrs.model_registry.capabilities import resolve_capabilities

from ..models import (
    ExecutionAuthority,
    ModelCapabilities,
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
from ..runtime_config import RuntimeConfig
from .acl import AccessContext, ACLEnforcer
from .errors import DuplicateModelError, ModelNotFoundError, PermissionDeniedError
from .path_validator import PathValidator
from umrs.metrics import METRICS

logger = logging.getLogger(__name__)


class ModelStore:
    """
    Model Registry data access layer.

    Provides CRUD operations with ACL enforcement and path validation.
    """

    def __init__(
        self,
        conn: psycopg2.extensions.connection,
        path_validator: PathValidator | None = None,
        acl_enforcer: ACLEnforcer | None = None,
        pool: "psycopg2.pool.AbstractConnectionPool | None" = None,
    ):
        self._conn = conn
        self._pool = pool
        self.path_validator = path_validator or PathValidator()
        self.acl = acl_enforcer or ACLEnforcer()

    @property
    def conn(self) -> psycopg2.extensions.connection:
        """Return a live connection, reconnecting if stale.

        psycopg2 connections held for long periods (e.g. the model store's
        persistent connection) can go stale when Postgres drops idle
        sessions or the network interrupts.  Before every cursor() call
        we check ``conn.closed`` and, if the connection is dead, borrow a
        fresh one from the pool (if available) or reconnect directly.
        """
        if self._conn is None or self._conn.closed:
            if self._pool is not None:
                # Return the dead connection to the pool (closed) and get a new one
                if self._conn is not None:
                    try:
                        self._pool.putconn(self._conn, close=True)
                    except Exception:
                        pass
                self._conn = self._pool.getconn()
            else:
                # No pool — reconnect with the same DSN
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                self._conn = psycopg2.connect(
                    dbname=self._conn.info.dbname if self._conn else None,
                )
        return self._conn

    # -------------------------------------------------------------------------
    # Create
    # -------------------------------------------------------------------------

    def create_model(
        self,
        name: str,
        scope: ModelScope,
        storage_ref: str,
        model_type: ModelType = ModelType.LLM,
        owner_id: str | None = None,
        project_id: str | None = None,
        huggingface_id: str | None = None,
        description: str | None = None,
        serving_profile: ServingProfile | None = None,
        capabilities: ModelCapabilities | None = None,
        runtime_config: RuntimeConfig | None = None,
        status: ModelStatus = ModelStatus.AVAILABLE,
        model_ref: ModelRef | None = None,
        execution_authority_default: ExecutionAuthority = ExecutionAuthority.COMPOSE,
        ctx: AccessContext | None = None,
    ) -> ModelEntry:
        """
        Create a new model entry.

        Raises:
            PermissionDeniedError: If user cannot create in scope
            PathNotAllowedError: If storage path is not allowed
            DuplicateModelError: If model name already exists in scope
        """
        if ctx:
            self.acl.enforce_create(scope, ctx, project_id)

        self.path_validator.validate_scope_path(
            storage_ref, scope, owner_id, project_id
        )

        model_id = generate_model_id()
        serving_profile = serving_profile or ServingProfile()
        explicit_capabilities = capabilities is not None
        capabilities = resolve_capabilities(name, capabilities or ModelCapabilities())
        if storage_ref and not explicit_capabilities:
            from ..detect import merge_detected_capabilities

            capabilities = merge_detected_capabilities(
                capabilities, storage_ref, model_name=name
            )
        runtime_config = runtime_config or RuntimeConfig()

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
            capabilities=capabilities,
            execution_authority_default=execution_authority_default,
            status=status,
        )

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_catalog_entries (
                        model_id, name, scope, owner_id, project_id,
                        storage_ref, model_type, huggingface_id, description,
                        serving_profile, status, model_ref_json, execution_authority_default,
                        capabilities, runtime_config
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
                        json.dumps(model.model_ref.to_dict())
                        if model.model_ref
                        else None,
                        model.execution_authority_default.value,
                        json.dumps(model.capabilities.to_dict()),
                        json.dumps(model.runtime_config.to_dict()),
                    ),
                )
                row = cur.fetchone()
                model.created_at = row[0]
                model.updated_at = row[1]
                self.conn.commit()

        except psycopg2.errors.UniqueViolation:
            self.conn.rollback()
            raise DuplicateModelError(name, scope, project_id) from None

        logger.info(
            "Created model %s (%s) in %s scope",
            model.model_id,
            name,
            scope.value,
        )
        METRICS.model_event("registered")
        return model

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    def get_model(
        self,
        model_id: str,
        ctx: AccessContext | None = None,
    ) -> ModelEntry:
        """Get a model by ID."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT model_id, name, scope, owner_id, project_id,
                       storage_ref, model_type, huggingface_id, description,
                       serving_profile, status, model_ref_json, execution_authority_default,
                       created_at, updated_at, capabilities, runtime_config
                FROM model_catalog_entries
                WHERE model_id = %s
                """,
                (model_id,),
            )
            row = cur.fetchone()

        if row is None:
            raise ModelNotFoundError(model_id)

        model = ModelEntry.from_row(row)

        if ctx:
            self.acl.enforce_read(model, ctx)

        return model

    def get_model_by_name(
        self,
        name: str,
        scope: ModelScope,
        project_id: str | None = None,
        owner_id: str | None = None,
        ctx: AccessContext | None = None,
    ) -> ModelEntry:
        """Get a model by name within a scope."""
        with self.conn.cursor() as cur:
            if scope == ModelScope.SYSTEM:
                cur.execute(
                    """
                    SELECT model_id, name, scope, owner_id, project_id,
                           storage_ref, model_type, huggingface_id, description,
                           serving_profile, status, model_ref_json, execution_authority_default,
                           created_at, updated_at, capabilities, runtime_config
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
                           created_at, updated_at, capabilities, runtime_config
                    FROM model_catalog_entries
                    WHERE name = %s AND scope = 'project' AND project_id = %s
                    """,
                    (name, project_id),
                )
            else:
                cur.execute(
                    """
                    SELECT model_id, name, scope, owner_id, project_id,
                           storage_ref, model_type, huggingface_id, description,
                           serving_profile, status, model_ref_json, execution_authority_default,
                           created_at, updated_at, capabilities, runtime_config
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
        scope: ModelScope | None = None,
        model_type: ModelType | None = None,
        status: ModelStatus | None = None,
        project_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelEntry]:
        """List models visible to the user."""
        conditions = []
        params: list[Any] = []

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

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT model_id, name, scope, owner_id, project_id,
                   storage_ref, model_type, huggingface_id, description,
                   serving_profile, status, model_ref_json, execution_authority_default,
                   created_at, updated_at, capabilities, runtime_config
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
        description: str | None = None,
        serving_profile: ServingProfile | None = None,
        status: ModelStatus | None = None,
        capabilities: ModelCapabilities | None = None,
    ) -> ModelEntry:
        """Update model metadata."""
        model = self.get_model(model_id)
        self.acl.enforce_write(model, ctx)

        updates = []
        params: list[Any] = []

        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if serving_profile is not None:
            updates.append("serving_profile = %s")
            params.append(json.dumps(serving_profile.to_dict()))

        if status is not None:
            updates.append("status = %s")
            params.append(status.value)

        if capabilities is not None:
            updates.append("capabilities = %s")
            params.append(json.dumps(capabilities.to_dict()))

        if not updates:
            return model

        updates.append("updated_at = NOW()")
        params.append(model_id)

        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE model_catalog_entries
                SET {", ".join(updates)}
                WHERE model_id = %s
                RETURNING updated_at
                """,
                params,
            )
            cur.fetchone()
            self.conn.commit()

        return self.get_model(model_id)

    def update_status(
        self,
        model_id: str,
        status: ModelStatus,
        ctx: AccessContext | None = None,
    ) -> None:
        """Update model status (convenience method)."""
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
        METRICS.model_event("status_updated")

    def update_capabilities(
        self,
        model_id: str,
        capabilities: ModelCapabilities,
        ctx: AccessContext | None = None,
    ) -> None:
        """Update model capabilities."""
        if ctx:
            model = self.get_model(model_id)
            self.acl.enforce_write(model, ctx)

        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE model_catalog_entries
                SET capabilities = %s, updated_at = NOW()
                WHERE model_id = %s
                """,
                (json.dumps(capabilities.to_dict()), model_id),
            )
            if cur.rowcount == 0:
                raise ModelNotFoundError(model_id)
            self.conn.commit()

        logger.info("Updated model %s capabilities", model_id)

    # -------------------------------------------------------------------------
    # Delete
    # -------------------------------------------------------------------------

    def delete_model(
        self,
        model_id: str,
        ctx: AccessContext,
    ) -> ModelEntry:
        """Delete a model catalog entry and return the deleted entry."""
        model = self.get_model(model_id)
        self.acl.enforce_write(model, ctx)

        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM model_catalog_entries WHERE model_id = %s",
                (model_id,),
            )
            self.conn.commit()

        logger.info("Deleted model %s (%s)", model_id, model.name)
        METRICS.model_event("deleted")
        return model

    # -------------------------------------------------------------------------
    # Publish (U-815)
    # -------------------------------------------------------------------------

    def publish_model(
        self,
        model_id: str,
        target_project_id: str,
        ctx: AccessContext,
        new_name: str | None = None,
    ) -> ModelEntry:
        """Publish a user model to project scope (U-815)."""
        model = self.get_model(model_id)

        if model.owner_id != ctx.user_id:
            raise PermissionDeniedError(
                ctx.user_id,
                model_id,
                "publish",
                "Only the model owner can publish",
            )

        if model.scope != ModelScope.USER:
            raise PermissionDeniedError(
                ctx.user_id,
                model_id,
                "publish",
                f"Can only publish user models, not {model.scope.value}",
            )

        if not ctx.is_admin and target_project_id not in ctx.admin_project_ids:
            raise PermissionDeniedError(
                ctx.user_id,
                model_id,
                "publish",
                f"Must be admin of project {target_project_id} to publish",
            )

        publish_name = new_name if new_name else model.name

        existing = self._find_model_by_name(
            publish_name, ModelScope.PROJECT, target_project_id
        )
        if existing is not None:
            raise DuplicateModelError(
                publish_name, ModelScope.PROJECT, target_project_id
            )

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
            model_id,
            model.name,
            target_project_id,
            publish_name,
        )
        METRICS.model_event("published")

        return self.get_model(model_id)

    def _find_model_by_name(
        self,
        name: str,
        scope: ModelScope,
        project_id: str | None = None,
    ) -> ModelEntry | None:
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
        """Resolve model ID to validated storage path."""
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
        allowed_projects: list[str] | None = None,
        denied_projects: list[str] | None = None,
        max_concurrent_endpoints: int | None = None,
        max_tokens_per_day: int | None = None,
        priority: int = 0,
        warm_pool_min: int = 0,
        warm_pool_max: int = 0,
        warm_pool_ttl_seconds: int = 3600,
        enable_warm_pool: bool = False,
        is_system_model: bool = False,
        default_serving_profile: ServingProfile | None = None,
        ctx: AccessContext | None = None,
    ) -> ModelPolicy:
        """Create a model policy (U-805)."""
        if ctx and not ctx.is_admin:
            raise PermissionDeniedError(ctx.user_id, "policy", "create", "Admin only")

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

    def get_policy_by_model_name(self, model_name: str) -> ModelPolicy | None:
        """Get policy for a model by name."""
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
        is_system_model: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ModelPolicy]:
        """List model policies."""
        conditions = []
        params: list[Any] = []

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
        allowed_projects: list[str] | None = None,
        denied_projects: list[str] | None = None,
        max_concurrent_endpoints: int | None = None,
        enable_warm_pool: bool | None = None,
        warm_pool_min: int | None = None,
        warm_pool_max: int | None = None,
        ctx: AccessContext | None = None,
    ) -> ModelPolicy:
        """Update a model policy."""
        if ctx and not ctx.is_admin:
            raise PermissionDeniedError(ctx.user_id, policy_id, "update", "Admin only")

        self.get_policy(policy_id)

        updates = []
        params: list[Any] = []

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
                SET {", ".join(updates)}
                WHERE policy_id = %s
                """,
                params,
            )
            self.conn.commit()

        return self.get_policy(policy_id)

    def delete_policy(self, policy_id: str, ctx: AccessContext | None = None) -> None:
        """Delete a model policy."""
        if ctx and not ctx.is_admin:
            raise PermissionDeniedError(ctx.user_id, policy_id, "delete", "Admin only")

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
        allowed_projects: list[str] | None = None,
        denied_projects: list[str] | None = None,
        max_concurrent_endpoints: int | None = None,
        enable_warm_pool: bool | None = None,
        warm_pool_min: int | None = None,
        warm_pool_max: int | None = None,
        ctx: AccessContext | None = None,
    ) -> ModelPolicy:
        """Update a model policy by model name (U-820)."""
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
        huggingface_id: str | None = None,
        description: str | None = None,
        serving_profile: ServingProfile | None = None,
        capabilities: ModelCapabilities | None = None,
        execution_authority_default: ExecutionAuthority = ExecutionAuthority.COMPOSE,
        warm_pool_min: int = 0,
        warm_pool_max: int = 1,
        warm_pool_ttl_seconds: int = 3600,
        enable_warm_pool: bool = False,
        allowed_projects: list[str] | None = None,
        denied_projects: list[str] | None = None,
        max_concurrent_endpoints: int | None = None,
        priority: int = 0,
        ctx: AccessContext | None = None,
    ) -> tuple[ModelEntry, ModelPolicy]:
        """
        Register a system-level model with an associated policy (U-817).

        Atomic operation that creates both model entry and policy.
        """
        if ctx and not ctx.is_admin:
            raise PermissionDeniedError(
                ctx.user_id, name, "create_system_model", "Admin only"
            )

        self.path_validator.validate(storage_ref)

        if self._check_duplicate(
            name, ModelScope.SYSTEM, None, ctx.user_id if ctx else None
        ):
            raise DuplicateModelError(name, ModelScope.SYSTEM)

        model_id = generate_model_id()
        policy_id = generate_policy_id()

        profile = serving_profile or ServingProfile()
        explicit_capabilities = capabilities is not None
        caps = resolve_capabilities(name, capabilities or ModelCapabilities())
        if storage_ref and not explicit_capabilities:
            from ..detect import merge_detected_capabilities

            caps = merge_detected_capabilities(caps, storage_ref, model_name=name)

        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_catalog_entries (
                        model_id, name, scope, storage_ref, model_type,
                        huggingface_id, description, serving_profile,
                        execution_authority_default, status, capabilities
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        json.dumps(caps.to_dict()),
                    ),
                )

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
                        True,
                        json.dumps(profile.to_dict()),
                    ),
                )

                self.conn.commit()

            logger.info(
                "Created system model %s (%s) with policy %s", model_id, name, policy_id
            )

            return (
                self.get_model(model_id, ctx),
                self.get_policy(policy_id),
            )

        except psycopg2.Error:
            self.conn.rollback()
            raise

    def check_policy_access(
        self,
        model_name: str,
        project_id: str,
    ) -> tuple[bool, str | None]:
        """Check if a project has access to a model based on policy."""
        policy = self.get_policy_by_model_name(model_name)

        if policy is None:
            return True, None

        if not policy.is_project_allowed(project_id):
            if project_id in policy.denied_projects:
                return (
                    False,
                    f"Project {project_id} is denied access to model {model_name}",
                )
            return (
                False,
                f"Project {project_id} is not in allowed list for model {model_name}",
            )

        return True, None
