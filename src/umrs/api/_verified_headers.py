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

from r4_contracts.handshake_hmac import ENV_VAR, verify_claims


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
) -> bool:
    """Decide whether ``X-Verified-*`` identity may be honored this request.

    strict=True  → honored ONLY with a valid HMAC signature (secret present).
    strict=False → legacy trust (mvp compatibility).
    Missing/absent secret in strict mode always denies: verified headers are
    treated as if a non-gateway caller set them.

    Note: is_admin defaults to False for the signature when deriving the
    base trust tuple; consumers that require the admin flag must verify
    it separatively via ``verify_admin_flag`` so that flipping the flag
    invalidates the signature.
    """
    if not strict:
        return True
    uid = (user_id or "").strip()
    pid = (project_id or "").strip()
    if not uid or not pid:
        return False
    return verify_claims(secret, presented_hmac, uid, pid, is_admin)


def verify_admin_flag(
    *,
    strict: bool,
    secret: str | None,
    presented_hmac: str | None,
    user_id: str | None,
    project_id: str | None,
    presented_is_admin: bool,
) -> bool:
    """Validate that an admin-claiming header set is signed for admin=TRUE.

    The base trust gate above uses the ASKED-FOR is_admin value, which
    would not catch a forged admin flag. For the admin path specifically we
    verify the signature with the caller's ASKED admin flag so a tampered
    ``X-Verified-Is-Admin: true`` rejects.
    """
    return verified_headers_trusted(
        strict=strict,
        secret=secret,
        presented_hmac=presented_hmac,
        user_id=user_id,
        project_id=project_id,
        is_admin=presented_is_admin,
    )
