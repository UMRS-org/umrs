"""Trust gate for gateway-issued ``X-Verified-*`` identity headers.

Pre-fix (live PoC 2026-08-13): UMRS strict-mode identity resolution trusted
``X-User-ID`` + ``X-Verified-Project-ID`` + ``X-Verified-Is-Admin: true``
without any gateway-origin proof. Any caller that could reach the UMRS
sidecar port directly (shell users on the frontend, co-tenant workloads)
self-issued an admin identity.

Contract (shared with ``r4_contracts.handshake_hmac``): a gateway that has
``UMRS_HANDSHAKE_HMAC_SECRET`` signs the identity tuple and adds
``X-Verified-HMAC``. UMRS strict mode honors verified headers only when the
signature validates. MVP mode keeps the legacy trust to avoid breaking the
compat deployment while the secret roll-out completes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fastapi import Header, HTTPException

from r4_contracts.handshake_hmac import ENV_VAR, presented_version, verify_presented


def handshake_secret() -> str | None:
    """Shared sign/verify secret from env (empty ⇒ unset)."""
    return os.environ.get(ENV_VAR, "").strip() or None


def verified_headers_trusted(
    *,
    strict: bool,
    secret: str | None,
    presented_hmac: str | None,
    user_id: str | None,
    project_id: str | None,
    is_admin: bool,
    is_workflow_author: bool = False,
) -> bool:
    """Decide whether ``X-Verified-*`` identity may be honored this request.

    strict=True  → honored ONLY with a valid HMAC signature (secret present).
    strict=False → legacy trust (mvp compatibility).
    Missing/absent secret in strict mode always denies: verified headers are
    treated as if a non-gateway caller set them.

    v1 signatures ignore ``is_workflow_author``. v2 signatures include it.
    """
    if not strict:
        return True
    uid = (user_id or "").strip()
    pid = (project_id or "").strip()
    if not uid or not pid:
        return False
    return verify_presented(
        secret,
        presented_hmac,
        uid,
        pid,
        is_admin,
        is_workflow_author,
    )


def verify_admin_flag(
    *,
    strict: bool,
    secret: str | None,
    presented_hmac: str | None,
    user_id: str | None,
    project_id: str | None,
    presented_is_admin: bool,
    presented_is_workflow_author: bool = False,
) -> bool:
    """Validate that an admin-claiming header set is signed for admin=TRUE.

    The base trust gate above uses the ASKED-FOR is_admin value, which
    would not catch a forged admin flag. For the admin path specifically we
    verify the signature with the caller's ASKED admin flag so a tampered
    ``X-Verified-Is-Admin: true`` rejects. v2 signatures also cover the
    presented workflow-author bit.
    """
    return verified_headers_trusted(
        strict=strict,
        secret=secret,
        presented_hmac=presented_hmac,
        user_id=user_id,
        project_id=project_id,
        is_admin=presented_is_admin,
        is_workflow_author=presented_is_workflow_author,
    )


# ---------------------------------------------------------------------------
# Wave-6 shared caller dependency (F2-F6: tenant-binding closure)
# ---------------------------------------------------------------------------


def _strict_mode_env() -> bool:
    """Strict-mode check shared with workflow_auth (avoids import cycle)."""
    return os.environ.get("UMRS_AUTH_MODE", "strict").strip().lower() == "strict"


@dataclass(frozen=True)
class CallerIdentity:
    """Resolved, trust-gated caller identity for tenant-bound routes.

    Produced only after the HMAC trust gate accepts the gateway headers
    (strict mode) or when legacy trust applies (mvp mode).
    """

    user_id: str
    project_id: str
    is_admin: bool = False
    is_workflow_author: bool = False

    def scope_filter(self, requested: str | None) -> str | None:
        """Binds an optional project filter for LIST/READ queries.

        - Admin callers keep the requested filter unchanged (None = global).
        - Non-admin callers are forced to their own verified project:
          requesting another project is a loud 403, never a silent rewrite.
        """
        if self.is_admin:
            return requested
        if requested and requested.strip() and requested.strip() != self.project_id:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"project filter '{requested.strip()}' is outside the "
                    "caller's verified project"
                ),
            )
        return self.project_id

    def bind_for_write(self, declared: str | None) -> str:
        """Binds the project for WRITE operations.

        - Admin: declared project honored (must be non-empty).
        - Non-admin: declared must equal the caller's verified project;
          a mismatch or foreign target is 403.
        """
        declared_clean = (declared or "").strip()
        if self.is_admin:
            if not declared_clean:
                raise HTTPException(status_code=400, detail="project_id is required")
            return declared_clean
        if declared_clean and declared_clean != self.project_id:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"project_id '{declared_clean}' is outside the "
                    "caller's verified project"
                ),
            )
        return self.project_id

    def assert_can_access(self, target_project: str | None) -> None:
        """404-hide check for row-level reads/mutations.

        Hiding (404) rather than 403 avoids confirming the existence of a
        foreign-tenant row id, matching the artifact-routes pattern.
        """
        if self.is_admin:
            return
        if target_project and target_project != self.project_id:
            raise HTTPException(status_code=404, detail="resource not found")


def _missing_caller(strict: bool) -> CallerIdentity | None:
    """Absent identity headers: reject under strict, anonymous under mvp."""
    if strict:
        raise HTTPException(
            status_code=401,
            detail="Missing Gateway-verified caller identity",
        )
    return None


def _signature_ok(
    strict: bool,
    uid: str,
    pid: str,
    x_verified_is_admin: str | None,
    x_verified_hmac: str | None,
    x_verified_workflow_author: str | None = None,
) -> bool:
    """HMAC trust decision for a present identity tuple."""
    return verified_headers_trusted(
        strict=strict,
        secret=handshake_secret(),
        presented_hmac=(x_verified_hmac or "").strip() or None,
        user_id=uid,
        project_id=pid,
        is_admin=(x_verified_is_admin or "").strip() == "true",
        is_workflow_author=(x_verified_workflow_author or "").strip().lower() == "true",
    )


def resolve_caller(
    x_user_id: str | None,
    x_verified_project_id: str | None,
    x_verified_is_admin: str | None,
    x_verified_hmac: str | None,
    x_verified_workflow_author: str | None = None,
) -> CallerIdentity | None:
    """Core resolution shared by the dependency + direct callers.

    Returns ``None`` for anonymous (no identity headers) under mvp mode —
    legacy route behavior is preserved there. Returns ``None`` never in
    strict mode: unsigned or absent identity raises 401.
    """
    uid = (x_user_id or "").strip()
    pid = (x_verified_project_id or "").strip()
    strict = _strict_mode_env()
    if not uid or not pid:
        return _missing_caller(strict)
    if not _signature_ok(
        strict,
        uid,
        pid,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    ):
        raise HTTPException(
            status_code=401,
            detail="Gateway-verified identity signature rejected",
        )
    presented_author = (x_verified_workflow_author or "").strip().lower() == "true"
    hmac_version = presented_version((x_verified_hmac or "").strip() or None)
    return CallerIdentity(
        user_id=uid,
        project_id=pid,
        is_admin=(x_verified_is_admin or "").strip() == "true",
        is_workflow_author=presented_author and hmac_version == "2",
    )


def get_caller(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_verified_workflow_author: str | None = Header(None),
) -> CallerIdentity | None:
    """FastAPI dependency: trust-gated caller for tenant-bound routers."""
    return resolve_caller(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )


def get_verified_caller(
    x_user_id: str | None = Header(None),
    x_verified_project_id: str | None = Header(None),
    x_verified_is_admin: str | None = Header(None),
    x_verified_hmac: str | None = Header(None),
    x_verified_workflow_author: str | None = Header(None),
) -> CallerIdentity:
    """FastAPI dependency: caller identity is REQUIRED (even in mvp mode).

    Use on mutation routes where an anonymous write is never acceptable.
    """
    caller = resolve_caller(
        x_user_id,
        x_verified_project_id,
        x_verified_is_admin,
        x_verified_hmac,
        x_verified_workflow_author,
    )
    if caller is None:
        raise HTTPException(
            status_code=401,
            detail="Caller identity required for this operation",
        )
    return caller
