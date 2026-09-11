"""Caller-scope resolution for three-tier read/write boundaries (read routes).

Resolution tiers, mirroring ``jobs.get_caller_identity`` semantics:

1. ``X-User-ID`` + ``X-Verified-Project-ID`` present → HMAC-verified scope
   (rejected when the gateway signature does not validate; in strict mode a
   missing tuple is a 401).
2. No user identity but ``X-Project-ID`` present → legacy project-level scope
   (preserves pre-boundary routes in mvp mode; user-owned rows stay hidden).
3. Neither → anonymous scope (system/global reads only) in mvp mode;
   401 in strict mode.

The pure visibility logic lives in ``umrs.knowledge.access_scope`` — this
module is only the HTTP/identity adapter for read-side route wiring.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Header, HTTPException, Request

from umrs.knowledge.access_scope import CallerScope


def _strict_mode() -> bool:
    # Default strict (audit B12): mvp is an explicit dev opt-in, never the
    # silent default — an unconfigured process must not run wide open.
    return os.environ.get("UMRS_AUTH_MODE", "strict").strip().lower() == "strict"


def _clean(value: Any) -> str:
    return str(value).strip() if isinstance(value, str) else ""


def _verify(headers: Any, uid: str, pid: str, admin: bool) -> None:
    """HMAC trust check — shared-secret class closure (same as jobs.py)."""
    from ._verified_headers import handshake_secret, verified_headers_trusted

    if not verified_headers_trusted(
        strict=_strict_mode(),
        secret=handshake_secret(),
        presented_hmac=_clean(headers.get("x-verified-hmac")) or None,
        user_id=uid,
        project_id=pid,
        is_admin=admin,
    ):
        raise_typed(
            PlatformError.scopeidentity(
                "unauthorized",
                "Gateway-verified identity signature rejected"
            ),
        )


def request_caller_scope(request: Request) -> CallerScope:
    """Resolve the caller's access scope from request headers (see module doc).

    Delegates to the canonical ``_identity.request_caller`` (R1) — single
    tier chain shared by rag/chat/llm/jobs/skills. HMAC verification is
    included for the verified tier.
    """
    headers = request.headers
    uid = _clean(headers.get("x-user-id"))
    pid = _clean(headers.get("x-verified-project-id"))
    if not uid or not pid:
        legacy = _clean(headers.get("x-project-id"))
        if legacy:
            # Legacy project-level trust (pre-boundary routes, mvp mode).
            if _strict_mode():
                raise_typed(
                    PlatformError.scopeidentity(
                        "unauthorized",
                        "Missing gateway-verified caller identity"
                    ),
                )
            return CallerScope(user_id="", project_id=legacy)
        if _strict_mode():
            raise_typed(
                PlatformError.scopeidentity(
                    "unauthorized",
                    "Missing gateway-verified caller identity"
                ),
            )
        return CallerScope(anonymous=True)
    admin = _clean(headers.get("x-verified-is-admin")) in ("true", "1")
    _verify(headers, uid, pid, admin)
    return CallerScope(user_id=uid, project_id=pid, is_admin=admin)


def get_caller_scope(
    request: Request,
    x_user_id: str | None = Header(None),
) -> CallerScope:
    """FastAPI dependency: scope for read-side routes (caller optional)."""
    return request_caller_scope(request)
