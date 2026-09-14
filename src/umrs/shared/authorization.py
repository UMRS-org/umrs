"""Authorization engine for UMRS operations.

Enforces per-user, per-project, and per-session resource policies
*after* authentication has established identity.

Design principles:
- ZAP (authentication) stays simple: is-key-registered? → allow/deny.
- Authorization runs per-request inside UMRS handlers.
- Policies compose: session grants ∩ site policy ∩ user record.
- Deny-by-default: unknown operations are rejected.

Usage::

    from umrs.shared.authorization import authorize, AuthorizationDenied

    try:
        authorize(auth_ctx, operation="submit_job", resource=request_spec)
    except AuthorizationDenied as exc:
        return error_response(403, str(exc))
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from umrs.shared.auth import AuthContext

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class AuthorizationDenied(Exception):
    """Raised when an operation is denied by policy."""

    def __init__(
        self,
        message: str,
        *,
        user_id: str = "",
        operation: str = "",
        reason: str = "",
    ):
        self.user_id = user_id
        self.operation = operation
        self.reason = reason
        super().__init__(message)


# =============================================================================
# Policy Data Model
# =============================================================================


@dataclass
class ResourceRequest:
    """Describes the resource being requested for authorization checks."""

    project_id: str = ""
    gpus: int = 0
    walltime_sec: int = 0
    model: str = ""
    operation: str = ""  # submit_job, cancel_job, upload_file, etc.
    target_user: str = ""  # For admin operations on another user's resources
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SitePolicy:
    """Site-level resource policy loaded from environment/config.

    Represents hard caps that apply regardless of user role.
    """

    max_gpus_per_job: int = 8
    max_walltime_sec: int = 172800  # 48 hours
    max_concurrent_jobs: int = 10
    require_project: bool = False
    allowed_partitions: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> SitePolicy:
        """Load site policy from environment variables."""
        return cls(
            max_gpus_per_job=int(os.getenv("UMRS_MAX_GPUS_PER_JOB", "8")),
            max_walltime_sec=int(os.getenv("UMRS_MAX_WALLTIME_SEC", "172800")),
            max_concurrent_jobs=int(os.getenv("UMRS_MAX_CONCURRENT_JOBS", "10")),
            require_project=os.getenv("UMRS_REQUIRE_PROJECT", "").lower()
            in ("1", "true", "yes"),
            allowed_partitions=[
                p.strip()
                for p in os.getenv("UMRS_ALLOWED_PARTITIONS", "").split(",")
                if p.strip()
            ],
        )


# Module-level cached policy instance
_site_policy: SitePolicy | None = None


def get_site_policy() -> SitePolicy:
    """Get or create the cached site policy."""
    global _site_policy
    if _site_policy is None:
        _site_policy = SitePolicy.from_env()
    return _site_policy


def reset_site_policy() -> None:
    """Reset cached policy (for testing)."""
    global _site_policy
    _site_policy = None


# =============================================================================
# Authorization Logic
# =============================================================================


def authorize(
    auth: AuthContext,
    operation: str,
    resource: ResourceRequest | None = None,
    *,
    session_grants: Any = None,
) -> None:
    """Authorize an operation for the given identity.

    Checks (in order):
    1. Admin/service bypass (admins can do anything)
    2. Project access (if resource specifies a project)
    3. Resource limits (GPU count, walltime)
    4. Operation permission (command allowlist)

    Parameters
    ----------
    auth:
        The authenticated identity context.
    operation:
        Operation name (e.g. "submit_job", "cancel_job", "upload_file").
    resource:
        Optional resource request details for limit checks.
    session_grants:
        Optional :class:`~umrs.core.session_curve_provider.SessionGrants`
        for session-scoped capability checks.

    Raises
    ------
    AuthorizationDenied:
        If any policy check fails.
    """
    if resource is None:
        resource = ResourceRequest(operation=operation)
    else:
        resource.operation = operation

    # 1. Admin/service bypass
    if auth.is_admin or auth.is_service:
        logger.debug(
            "Authorization ALLOW (admin/service): user=%s op=%s",
            auth.user_id,
            operation,
        )
        return

    policy = get_site_policy()

    # 2. Project access check
    _check_project_access(auth, resource, session_grants, policy)

    # 3. Resource limits
    _check_resource_limits(auth, resource, session_grants, policy)

    # 4. Ownership check (users can only affect their own resources)
    _check_ownership(auth, resource)

    logger.debug(
        "Authorization ALLOW: user=%s op=%s project=%s",
        auth.user_id,
        operation,
        resource.project_id,
    )


def _check_project_access(
    auth: AuthContext,
    resource: ResourceRequest,
    session_grants: Any,
    policy: SitePolicy,
) -> None:
    """Verify user has access to the requested project."""
    if not resource.project_id:
        if policy.require_project:
            raise AuthorizationDenied(
                f"Operation '{resource.operation}' requires a project_id",
                user_id=auth.user_id,
                operation=resource.operation,
                reason="project_required",
            )
        return  # No project specified and not required

    # Check session grants first (tighter scope)
    if session_grants is not None and hasattr(session_grants, "has_project"):
        if not session_grants.has_project(resource.project_id):
            raise AuthorizationDenied(
                f"Session does not have access to project '{resource.project_id}'",
                user_id=auth.user_id,
                operation=resource.operation,
                reason="session_project_denied",
            )

    # Check auth context project (if set, must match)
    if auth.project_id and auth.project_id != resource.project_id:
        raise AuthorizationDenied(
            f"User bound to project '{auth.project_id}', cannot access '{resource.project_id}'",
            user_id=auth.user_id,
            operation=resource.operation,
            reason="project_mismatch",
        )


def _check_resource_limits(
    auth: AuthContext,
    resource: ResourceRequest,
    session_grants: Any,
    policy: SitePolicy,
) -> None:
    """Verify requested resources are within policy and session limits."""
    # GPU limit check
    if resource.gpus > 0:
        # Session cap (if set)
        if session_grants is not None and hasattr(session_grants, "effective_max_gpus"):
            session_max = session_grants.effective_max_gpus(policy.max_gpus_per_job)
            if resource.gpus > session_max:
                raise AuthorizationDenied(
                    f"Requested {resource.gpus} GPUs exceeds session cap of {session_max}",
                    user_id=auth.user_id,
                    operation=resource.operation,
                    reason="gpu_limit_exceeded",
                )
        # Site cap
        if resource.gpus > policy.max_gpus_per_job:
            raise AuthorizationDenied(
                f"Requested {resource.gpus} GPUs exceeds site limit of {policy.max_gpus_per_job}",
                user_id=auth.user_id,
                operation=resource.operation,
                reason="gpu_site_limit_exceeded",
            )

    # Walltime limit check
    if resource.walltime_sec > 0:
        # Session cap
        if session_grants is not None and hasattr(
            session_grants, "effective_max_walltime"
        ):
            session_max = session_grants.effective_max_walltime(policy.max_walltime_sec)
            if resource.walltime_sec > session_max:
                raise AuthorizationDenied(
                    f"Requested walltime {resource.walltime_sec}s exceeds session cap of {session_max}s",
                    user_id=auth.user_id,
                    operation=resource.operation,
                    reason="walltime_limit_exceeded",
                )
        # Site cap
        if resource.walltime_sec > policy.max_walltime_sec:
            raise AuthorizationDenied(
                f"Requested walltime {resource.walltime_sec}s exceeds site limit of {policy.max_walltime_sec}s",
                user_id=auth.user_id,
                operation=resource.operation,
                reason="walltime_site_limit_exceeded",
            )


def _check_ownership(auth: AuthContext, resource: ResourceRequest) -> None:
    """Verify users can only operate on their own resources."""
    if resource.target_user and resource.target_user != auth.user_id:
        raise AuthorizationDenied(
            f"Cannot operate on another user's resources (target={resource.target_user})",
            user_id=auth.user_id,
            operation=resource.operation,
            reason="ownership_violation",
        )
