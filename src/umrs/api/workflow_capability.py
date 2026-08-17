"""Authoritative UMRS workflow-upload authorization.

IdP/JWT roles arrive as HMAC-covered ``X-Verified-Workflow-Author`` /
``X-Verified-Is-Admin``. Local R4 policy files are never consulted.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, status

from ._verified_headers import CallerIdentity

WORKFLOW_SCOPES = frozenset({"user", "system"})


def normalize_workflow_scope(scope: str | None) -> str:
    resolved = (scope or "user").strip() or "user"
    if resolved not in WORKFLOW_SCOPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="scope must be 'user' or 'system'",
        )
    return resolved


def authorize_workflow_upload(identity: CallerIdentity, scope: str) -> None:
    """Reject unauthorized executable workflow uploads before archive I/O."""
    resolved_scope = normalize_workflow_scope(scope)
    if resolved_scope == "system":
        if not identity.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="system-scoped workflow upload requires admin",
            )
        return
    if identity.is_admin or identity.is_workflow_author:
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="user-scoped workflow upload requires workflow_author",
    )


def capabilities_payload(identity: CallerIdentity) -> dict[str, Any]:
    """Read-only effective capabilities for the verified caller."""
    user_upload = identity.is_admin or identity.is_workflow_author
    system_upload = identity.is_admin
    authorized_scopes = []
    if user_upload:
        authorized_scopes.append("user")
    if system_upload:
        authorized_scopes.append("system")
    denial_reason = ""
    if not user_upload:
        denial_reason = (
            "UMRS upload requires the workflow_author role from the identity "
            "provider. Local Companion policy cannot grant this."
        )
    return {
        "available": True,
        "is_admin": identity.is_admin,
        "is_workflow_author": identity.is_workflow_author,
        "authorized_scopes": authorized_scopes,
        "operations": {
            "list": True,
            "describe": True,
            "validate": True,
            "status": True,
            "results": True,
            "wait": True,
            "run": True,
            "cancel": True,
            "upload": {"user": user_upload, "system": system_upload},
        },
        "denial_reason": denial_reason,
    }
