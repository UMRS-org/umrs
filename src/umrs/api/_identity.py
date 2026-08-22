"""Canonical caller-identity resolution for all UMRS route families (R1).

Single merger point for the historical identity variants ({chat,llm,rag}.py
plain-tuple, ``skills.py`` Header-based class, ``jobs.py`` HMAC-verified +
admin, ``_scope_identity.py`` tiered scope):

Resolution tiers (unchanged semantics, one implementation):

1. ``X-Verified-Project-ID`` (+ optional ``X-User-ID``,
   ``X-Verified-Is-Admin``, ``X-Verified-HMAC``) → verified identity. When
   ``verify_hmac=True`` (jobs hardening) the tuple must carry a valid
   gateway HMAC signature; without that flag the strict-mode contract is
   honored (missing verified tuple → 401) but the signature itself is not
   rechecked — NGINX ``auth_request`` already validated the token upstream.
2. No verified project but ``X-Project-ID`` present → legacy project-level
   identity (mvp mode only; strict mode → 401).
3. Neither:
   - ``require_project=True`` (chat/llm/rag surfaces, headers sidecar-set) → 401.
   - ``require_project=False`` (read/scoped surfaces) → anonymous scope
     ``(project_id="", no hypotheses)`` in mvp mode; 401 in strict mode.

``request_identity`` is the single resolver; ``request_caller`` adapts it to
the three-tier ``CallerScope`` universe (read/write predicates in
``umrs.knowledge.access_scope``). Pre-existing modules delegate here
(``_scope_identity``, ``rag``, ``chat``, ``llm``, ``skills``, ``jobs``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import HTTPException, Request, status


def _strict_mode() -> bool:
    return os.environ.get("UMRS_AUTH_MODE", "mvp").strip().lower() == "strict"


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


@dataclass(frozen=True)
class CallerIdentity:
    """Resolved request identity (project_id empty only when anonymous)."""

    user_id: str = ""
    project_id: str = ""
    is_admin: bool = False
    anonymous: bool = False
    legacy: bool = False  # X-Project-ID fallback (mvp compat), not HMAC-trusted


_LEGACY_PARAM_WARN_ONCE: set[str] = set()


def warn_legacy_params_once(feature: str) -> None:
    """Deprecation warning for query-param identity (R1, drifts R1/16).

    Query-param identity (caller_user_id / caller_project_id / is_admin) is
    MVP-compat only: strict mode already rejects it, and it is scheduled
    for removal in the next release.
    """
    import logging

    if feature in _LEGACY_PARAM_WARN_ONCE:
        return
    _LEGACY_PARAM_WARN_ONCE.add(feature)
    logging.getLogger("umrs.api.identity").warning(
        "DEPRECATED: query-param identity (%s) — strict mode rejects it; "
        "removal scheduled for the next release.",
        feature,
    )


def _reject_401(detail: str) -> None:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


def _verify_handshake(
    request: Request,
    user_id: str,
    project_id: str,
    is_admin: bool,
) -> None:
    """Gateway HMAC signature check (jobs.py hardening, preserved verbatim)."""
    from ._verified_headers import handshake_secret, verified_headers_trusted

    if not verified_headers_trusted(
        strict=_strict_mode(),
        secret=handshake_secret(),
        presented_hmac=_clean(request.headers.get("x-verified-hmac")) or None,
        user_id=user_id,
        project_id=project_id,
        is_admin=is_admin,
    ):
        _reject_401("Gateway-verified identity signature rejected")


def request_identity(
    request: Request,
    *,
    require_project: bool = True,
    verify_hmac: bool = False,
) -> CallerIdentity:
    """Resolve the caller identity for one request (see module doc)."""
    headers = request.headers
    user_id = _clean(headers.get("x-user-id"))
    verified_project = _clean(headers.get("x-verified-project-id"))
    admin = _clean(headers.get("x-verified-is-admin")) in ("true", "1", "yes")
    legacy_project = _clean(headers.get("x-project-id"))

    if verified_project:
        if verify_hmac:
            _verify_handshake(request, user_id, verified_project, admin)
        return CallerIdentity(
            user_id=user_id, project_id=verified_project, is_admin=admin
        )

    if _strict_mode():
        _reject_401("Missing X-Verified-Project-ID header (Gateway must set this)")

    if legacy_project:
        return CallerIdentity(user_id=user_id, project_id=legacy_project, legacy=True)

    if require_project:
        _reject_401("Missing X-Project-ID header (auth sidecar must set this)")

    return CallerIdentity(user_id=user_id, anonymous=True)


def request_caller(request: Request):
    """Adapt the canonical identity for three-tier access predicates."""
    from umrs.knowledge.access_scope import CallerScope

    identity = request_identity(request, require_project=False, verify_hmac=False)
    if identity.anonymous:
        return CallerScope(anonymous=True)
    return CallerScope(
        user_id=identity.user_id,
        project_id=identity.project_id,
        is_admin=identity.is_admin,
    )


def verified_identity(
    request: Request,
    *,
    require_project: bool = True,
    verify_hmac: bool = False,
) -> tuple[str, str]:
    """Legacy plain-tuple adapter for chat/llm/rag surfaces."""
    identity = request_identity(
        request, require_project=require_project, verify_hmac=verify_hmac
    )
    return identity.user_id, identity.project_id
