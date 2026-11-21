"""Verified caller identity: the public API (U3 S1 item 10, ADR-117).

Routes outside this package (and the ``umrs_r4`` extension after U3 S5) need
the strict-mode identity checks. They were reached through private names:
``workflows._require_verified_workflow_identity``, ``_verified_headers`` and
``_scope_identity``. This module is the one public import point; the objects
are the same.
"""

from __future__ import annotations

from fastapi import Header

from ._scope_identity import request_caller_scope
from ._verified_headers import resolve_caller
from .workflows import require_verified_workflow_identity


def strict_mode() -> bool:
    """True when ``UMRS_AUTH_MODE`` is strict (the default). Public since U3
    S5a, for routes outside core that branch on the auth mode."""
    from ._verified_headers import _strict_mode_env

    return _strict_mode_env()


def strict_user_header(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
) -> str:
    """Gate + return the caller's verified user id."""
    require_verified_workflow_identity(
        x_user_id, x_verified_project_id, x_verified_is_admin, x_verified_hmac
    )
    return (x_user_id or "").strip()


__all__ = [
    "request_caller_scope",
    "require_verified_workflow_identity",
    "resolve_caller",
    "strict_mode",
    "strict_user_header",
]
