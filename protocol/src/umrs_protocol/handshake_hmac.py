"""Verified-handshake HMAC contract (gateway/sidecar → UMRS).

Problem: UMRS strict-mode identity resolution treats ``X-User-ID`` /
``X-Verified-Project-ID`` / ``X-Verified-Is-Admin: true`` headers as proof.
Those headers are plain text and client-settable, so any caller that can
reach UMRS directly (internal network, localhost, compute co-tenant) can
self-issue an admin identity (live PoC 2026-08-13).

Contract: the gateway/sidecar signs the identity tuple with a shared secret
(HMAC-SHA256) and adds ``X-Verified-HMAC``. UMRS strict mode requires a
valid signature before trusting ANY ``X-Verified-*`` identity header.
Without a configured secret (UMRS_HANDSHAKE_HMAC_SECRET) strict mode treats
verified headers as absent → falls back to JWT or deny.

Memberships (H-1, 2026-10-18): ``X-Verified-Project-Memberships`` is signed
separately (``sign_memberships`` → ``X-Verified-Memberships-HMAC``) over a
canonical sorted, deduplicated list bound to the same identity tuple, on its
own domain prefix. The identity tuple above is unchanged; UMRS strict mode
ignores a list whose signature does not verify.

Zero dependencies — stdlib only, per ADR-005 (r4-contracts is zero-dep).
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable

ENV_VAR = "UMRS_HANDSHAKE_HMAC_SECRET"

HEADER_HMAC = "X-Verified-HMAC"
HEADER_USER_ID = "X-User-ID"
HEADER_PROJECT_ID = "X-Verified-Project-ID"
HEADER_IS_ADMIN = "X-Verified-Is-Admin"
HEADER_WORKFLOW_AUTHOR = "X-Verified-Workflow-Author"
HEADER_WORKFLOW_AUTHOR_HMAC = "X-Verified-Workflow-Author-HMAC"
HEADER_PROJECT_MEMBERSHIPS = "X-Verified-Project-Memberships"
HEADER_MEMBERSHIPS_HMAC = "X-Verified-Memberships-HMAC"

# H-1 (2026-10-18): the memberships list gets its own signature, on its own
# domain, bound to the identity tuple. Additive: sign_claims is unchanged,
# so gateways and UMRS that predate it keep verifying the identity tuple.
MEMBERSHIPS_DOMAIN = "memberships-v1|"


def _canonical(user_id: str, project_id: str, is_admin: bool) -> bytes:
    """Canonical claim string — MUST be byte-identical on both sides."""
    return f"v1\n{user_id}\n{project_id}\n{1 if is_admin else 0}".encode()


def _canonical_workflow_author(user_id: str, project_id: str, author: bool) -> bytes:
    """ADR-027 wf-claim canonical — a SEPARATE signature from the identity
    tuple. Splitting bases keeps every pre-ADR route byte-identical (a
    shared 5-line tuple would have desynced routes that never forward the
    author header); deploys stay rollable."""
    return f"v1wf\n{user_id}\n{project_id}\n{1 if author else 0}".encode()


def sign_claims(
    secret: str,
    user_id: str,
    project_id: str,
    is_admin: bool,
) -> str:
    """Return the hex HMAC-SHA256 over the canonical identity tuple."""
    key = secret.encode("utf-8")
    return hmac.new(key, _canonical(user_id, project_id, is_admin), hashlib.sha256).hexdigest()


def verify_claims(
    secret: str | None,
    presented: str | None,
    user_id: str,
    project_id: str,
    is_admin: bool,
) -> bool:
    """Constant-time verification; ANY missing input is invalid."""
    if not secret or not presented or not user_id or not project_id:
        return False
    expected = sign_claims(secret, user_id, project_id, is_admin)
    try:
        return hmac.compare_digest(expected, presented)
    except (TypeError, ValueError):
        return False


def sign_workflow_author_claim(
    secret: str,
    user_id: str,
    project_id: str,
    author: bool,
) -> str:
    """ADR-027: hex HMAC-SHA256 over the wf-claim canonical (own basis)."""
    key = secret.encode("utf-8")
    return hmac.new(
        key,
        _canonical_workflow_author(user_id, project_id, author),
        hashlib.sha256,
    ).hexdigest()


def verify_workflow_author_claim(
    secret: str | None,
    presented: str | None,
    user_id: str,
    project_id: str,
    author: bool,
) -> bool:
    """Constant-time wf-claim verification; ANY missing input is invalid."""
    if not secret or not presented or not user_id or not project_id:
        return False
    expected = sign_workflow_author_claim(secret, user_id, project_id, author)
    try:
        return hmac.compare_digest(expected, presented)
    except (TypeError, ValueError):
        return False


def canonical_memberships(memberships: Iterable[str] | str | None) -> str:
    """Canonical memberships list: comma-split, trimmed, deduplicated, sorted,
    comma-joined. A header string and the claim list it was made from give
    the same value, so both sides sign and verify the same bytes."""
    if memberships is None:
        return ""
    items = [memberships] if isinstance(memberships, str) else list(memberships)
    parts = {part.strip() for item in items for part in str(item).split(",")}
    return ",".join(sorted(part for part in parts if part))


def _canonical_memberships_claim(
    user_id: str, project_id: str, is_admin: bool, memberships: Iterable[str] | str | None
) -> bytes:
    """Memberships canonical: own domain prefix, then the identity tuple and
    the canonical list (a separate basis from sign_claims)."""
    listed = canonical_memberships(memberships)
    return f"{MEMBERSHIPS_DOMAIN}{user_id}\n{project_id}\n{1 if is_admin else 0}\n{listed}".encode()


def sign_memberships(
    secret: str,
    user_id: str,
    project_id: str,
    is_admin: bool,
    memberships: Iterable[str] | str | None,
) -> str:
    """Hex HMAC-SHA256 over the memberships canonical (``X-Verified-Memberships-HMAC``)."""
    key = secret.encode("utf-8")
    body = _canonical_memberships_claim(user_id, project_id, is_admin, memberships)
    return hmac.new(key, body, hashlib.sha256).hexdigest()


def verify_memberships(
    secret: str | None,
    presented: str | None,
    user_id: str,
    project_id: str,
    is_admin: bool,
    memberships: Iterable[str] | str | None,
) -> bool:
    """Constant-time memberships verification; ANY missing input (an empty
    list included) is invalid."""
    if not secret or not presented or not user_id or not project_id:
        return False
    if not canonical_memberships(memberships):
        return False
    expected = sign_memberships(secret, user_id, project_id, is_admin, memberships)
    try:
        return hmac.compare_digest(expected, presented)
    except (TypeError, ValueError):
        return False
