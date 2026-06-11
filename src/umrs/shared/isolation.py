"""
Multi-user isolation and access control (U-005).

Provides:
- UserContext: Current user/project information
- AccessControl: Resource access validation
- IsolationValidator: Cross-user data isolation checks

Design Principles:
- All resources are scoped to (user_id, project_id)
- Default-deny: resources are private unless explicitly shared
- Hierarchical: admins can access all projects, users only their own
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class UserRole(Enum):
    """User roles for authorization."""

    ANONYMOUS = "anonymous"  # No authentication
    USER = "user"  # Standard authenticated user
    ADMIN = "admin"  # Platform administrator
    SERVICE = "service"  # Service account (inter-service)


class ResourceType(Enum):
    """Types of resources subject to access control."""

    FILE = "file"
    DOCUMENT = "document"
    PIPELINE = "pipeline"
    RAG_PROFILE = "rag_profile"
    MODEL = "model"
    ENDPOINT = "endpoint"
    JOB = "job"
    MCP_SERVER = "mcp_server"


class AccessLevel(Enum):
    """Access levels for resources."""

    NONE = 0
    READ = 1
    WRITE = 2
    ADMIN = 3

    def __ge__(self, other: AccessLevel) -> bool:
        return self.value >= other.value

    def __gt__(self, other: AccessLevel) -> bool:
        return self.value > other.value

    def __le__(self, other: AccessLevel) -> bool:
        return self.value <= other.value

    def __lt__(self, other: AccessLevel) -> bool:
        return self.value < other.value


# =============================================================================
# User Context
# =============================================================================


@dataclass
class UserContext:
    """Current user context for request processing."""

    user_id: str
    project_id: str
    role: UserRole = UserRole.USER
    groups: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Derived from JWT claims or headers
    display_name: str | None = None
    email: str | None = None

    # Slurm accounting name used when this user's jobs are submitted via sbatch.
    # Populated from the user record by the auth layer. ``None`` falls back to
    # the site default (see ``site.yaml scheduler.account``).
    slurm_account: str | None = None

    @classmethod
    def anonymous(cls, project_id: str = "default") -> UserContext:
        """Create anonymous context (no authentication)."""
        return cls(
            user_id="anonymous",
            project_id=project_id,
            role=UserRole.ANONYMOUS,
        )

    @classmethod
    def from_headers(
        cls,
        x_user_id: str | None,
        x_project_id: str | None,
        x_user_role: str | None = None,
        x_user_groups: str | None = None,
        x_slurm_account: str | None = None,
    ) -> UserContext:
        """Create context from HTTP headers."""
        user_id = x_user_id or "anonymous"
        project_id = x_project_id or "default"

        # Parse role
        role = UserRole.ANONYMOUS
        if x_user_role:
            try:
                role = UserRole(x_user_role.lower())
            except ValueError:
                role = UserRole.USER if user_id != "anonymous" else UserRole.ANONYMOUS

        # Parse groups (comma-separated)
        groups = []
        if x_user_groups:
            groups = [g.strip() for g in x_user_groups.split(",") if g.strip()]

        return cls(
            user_id=user_id,
            project_id=project_id,
            role=role,
            groups=groups,
            slurm_account=(x_slurm_account.strip() if x_slurm_account else None)
            or None,
        )

    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated."""
        return self.role != UserRole.ANONYMOUS

    @property
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == UserRole.ADMIN

    @property
    def is_service(self) -> bool:
        """Check if user is a service account."""
        return self.role == UserRole.SERVICE

    def can_access_project(self, target_project_id: str) -> bool:
        """Check if user can access a specific project."""
        if self.is_admin or self.is_service:
            return True
        return self.project_id == target_project_id

    def can_access_user_data(self, target_user_id: str) -> bool:
        """Check if user can access another user's data."""
        if self.is_admin or self.is_service:
            return True
        return self.user_id == target_user_id


# =============================================================================
# Resource Ownership
# =============================================================================


@dataclass
class ResourceOwnership:
    """Ownership information for a resource."""

    resource_type: ResourceType
    resource_id: str
    owner_id: str
    project_id: str
    is_shared: bool = False
    shared_with: set[str] = field(default_factory=set)  # user_ids or "project:{id}"

    def is_accessible_by(self, ctx: UserContext) -> bool:
        """Check if resource is accessible by the given user context."""
        # Admins and services can access everything
        if ctx.is_admin or ctx.is_service:
            return True

        # Owner can always access
        if ctx.user_id == self.owner_id:
            return True

        # Check project scope
        if ctx.project_id != self.project_id:
            return False

        # Check explicit sharing
        if ctx.user_id in self.shared_with:
            return True

        # Check project-level sharing
        project_key = f"project:{ctx.project_id}"
        return project_key in self.shared_with


# =============================================================================
# Access Control
# =============================================================================


class AccessControl:
    """Central access control validator."""

    # Patterns that indicate cross-user access attempts
    SUSPICIOUS_PATTERNS = [
        r"\.\./",  # Path traversal
        r"^/",  # Absolute paths
        r"[<>|]",  # Shell metacharacters
    ]

    def __init__(self):
        self._compiled_patterns = [re.compile(p) for p in self.SUSPICIOUS_PATTERNS]

    def validate_resource_access(
        self,
        ctx: UserContext,
        ownership: ResourceOwnership,
        required_level: AccessLevel = AccessLevel.READ,
    ) -> bool:
        """Validate access to a resource.

        Returns True if access is allowed, False otherwise.
        """
        if not ownership.is_accessible_by(ctx):
            logger.warning(
                f"Access denied: user={ctx.user_id} resource={ownership.resource_id} "
                f"owner={ownership.owner_id}"
            )
            return False

        # For write access, user must be owner or admin
        if required_level >= AccessLevel.WRITE:
            if not (ctx.is_admin or ctx.user_id == ownership.owner_id):
                logger.warning(
                    f"Write access denied: user={ctx.user_id} resource={ownership.resource_id}"
                )
                return False

        return True

    def validate_path(self, path: str, ctx: UserContext) -> bool:
        """Validate a file path for suspicious patterns.

        Returns True if path is safe, False otherwise.
        """
        for pattern in self._compiled_patterns:
            if pattern.search(path):
                logger.warning(
                    f"Suspicious path detected: user={ctx.user_id} path={path}"
                )
                return False
        return True

    def validate_project_access(
        self,
        ctx: UserContext,
        target_project_id: str,
    ) -> bool:
        """Validate access to a project."""
        if not ctx.can_access_project(target_project_id):
            logger.warning(
                f"Project access denied: user={ctx.user_id} "
                f"user_project={ctx.project_id} target_project={target_project_id}"
            )
            return False
        return True


# =============================================================================
# Isolation Validator
# =============================================================================


class IsolationValidator:
    """Validates multi-user isolation invariants."""

    def __init__(self):
        self._access_control = AccessControl()

    def validate_request_isolation(
        self,
        ctx: UserContext,
        request_data: dict[str, Any],
    ) -> list[str]:
        """Validate that request data doesn't violate isolation.

        Returns list of violation messages (empty if valid).
        """
        violations = []

        # Check for project_id mismatch
        if "project_id" in request_data:
            req_project = request_data["project_id"]
            if not ctx.can_access_project(req_project):
                violations.append(
                    f"Project mismatch: request specifies {req_project}, "
                    f"user has access to {ctx.project_id}"
                )

        # Check for user_id override attempts
        if "user_id" in request_data:
            req_user = request_data["user_id"]
            if not ctx.can_access_user_data(req_user):
                violations.append(
                    f"User override attempt: request specifies {req_user}, "
                    f"current user is {ctx.user_id}"
                )

        # Check for path traversal in any path-like fields
        path_fields = ["path", "file_path", "model_path", "output_path"]
        for path_field in path_fields:
            if path_field in request_data:
                path_val = request_data[path_field]
                if isinstance(path_val, str):
                    if not self._access_control.validate_path(path_val, ctx):
                        violations.append(
                            f"Suspicious path in {path_field}: {path_val}"
                        )

        return violations

    def validate_response_scoping(
        self,
        ctx: UserContext,
        response_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Scope response data to user/project context.

        Removes or filters data that shouldn't be visible to the user.
        """
        # If admin or service, return full response
        if ctx.is_admin or ctx.is_service:
            return response_data

        # Filter list responses
        if isinstance(response_data, list):
            return [
                item for item in response_data if self._item_visible_to_user(item, ctx)
            ]

        # For single items, check visibility
        if isinstance(response_data, dict):
            if not self._item_visible_to_user(response_data, ctx):
                return {}

        return response_data

    def _item_visible_to_user(
        self,
        item: dict[str, Any],
        ctx: UserContext,
    ) -> bool:
        """Check if an item should be visible to the user."""
        # Check project_id match
        if "project_id" in item:
            if item["project_id"] != ctx.project_id and not ctx.is_admin:
                return False

        # Check owner_id match
        if "owner_id" in item and item["owner_id"] != ctx.user_id and not ctx.is_admin:
            # Check if shared
            if "shared_with" in item:
                if ctx.user_id not in item.get("shared_with", []):
                    return False
            else:
                return False

        return True


# =============================================================================
# Configuration Isolation
# =============================================================================


class ConfigIsolation:
    """Isolates configuration access for multi-user scenarios."""

    def __init__(self, user_config_map: dict[str, str] | None = None):
        """Initialize with user-to-config-path mapping."""
        self._user_config_map: dict[str, str] = user_config_map or {}
        self._config_cache: dict[str, dict[str, Any]] = {}

    def register_user_config(self, user_id: str, config_path: str) -> None:
        """Register a configuration file path for a user."""
        self._user_config_map[user_id] = config_path
        # Invalidate cache
        self._config_cache.pop(user_id, None)

    def get_user_config_path(self, ctx: UserContext) -> str | None:
        """Get configuration file path for user, respecting isolation."""
        # Admins can use any config (for debugging)
        if ctx.is_admin and "target_user" in ctx.metadata:
            target_user = ctx.metadata["target_user"]
            return self._user_config_map.get(target_user)

        return self._user_config_map.get(ctx.user_id)

    def get_user_config(self, ctx: UserContext) -> dict[str, Any] | None:
        """Get parsed configuration for user."""
        if ctx.user_id in self._config_cache:
            return self._config_cache[ctx.user_id]

        config_path = self.get_user_config_path(ctx)
        if not config_path:
            return None

        # Load and cache (implementation would parse the config file)
        # For now, return empty dict as placeholder
        config = {}
        self._config_cache[ctx.user_id] = config
        return config

    def list_user_buckets(self, ctx: UserContext) -> list[str]:
        """List S3 buckets accessible to user.

        Ensures users can only see their own buckets.
        """
        # This would be populated from BucketsCfg filtered by user
        return []


# =============================================================================
# Global Instances
# =============================================================================

_access_control: AccessControl | None = None
_isolation_validator: IsolationValidator | None = None
_config_isolation: ConfigIsolation | None = None


def get_access_control() -> AccessControl:
    """Get global access control instance."""
    global _access_control
    if _access_control is None:
        _access_control = AccessControl()
    return _access_control


def get_isolation_validator() -> IsolationValidator:
    """Get global isolation validator instance."""
    global _isolation_validator
    if _isolation_validator is None:
        _isolation_validator = IsolationValidator()
    return _isolation_validator


def get_config_isolation() -> ConfigIsolation:
    """Get global config isolation instance."""
    global _config_isolation
    if _config_isolation is None:
        _config_isolation = ConfigIsolation()
    return _config_isolation


def set_config_isolation(isolation: ConfigIsolation) -> None:
    """Set global config isolation (for testing)."""
    global _config_isolation
    _config_isolation = isolation
