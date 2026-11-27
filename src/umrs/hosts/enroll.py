"""Host enrollment with a one-time token (ADR-098 D-5; slice SH-H1b).

    admin   umrs-server admin host enroll <name>   → prints a token once
    host    generate_host_key()                    → ed25519 keypair, kept local
    host    enrollment_request(token, private)     → {token, public_key}
    server  redeem_enrollment(store, token, key)   → the host's enrolled key

The server stores only the SHA-256 of the token and the host's public key.
A token is used at most once (compare-and-set) and expires (default one
hour, at most a day); a reused, expired or unknown token is refused by name
(``enroll.token_used``, ``enroll.token_expired``, ``enroll.token_invalid``).
The inline host (``local``) never enrolls: its identity is the UMRS process.

The HTTP route an agent redeems through arrives with ``umrs-hostd`` (SH-H3);
this module is the whole server-side rule, so the route stays a thin shell.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from umrs.hosts.identity import (
    EnrollmentError,
    HostKey,
    HostKeyStore,
    b64url,
    unb64url,
)
from umrs.hosts.runtime import INLINE_HOST_ID

ENROLL_TOKEN_PREFIX = "umrs-enroll-"
DEFAULT_ENROLL_TTL_S = 3600
MAX_ENROLL_TTL_S = 86400
_HOST_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")


def enrollment_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def check_host_id(host_id: str) -> str:
    """A host id an admin may enroll (lower-case DNS-ish, not ``local``)."""
    if host_id == INLINE_HOST_ID:
        raise EnrollmentError(
            "enroll.host_reserved",
            f"{INLINE_HOST_ID!r} is the inline host: it never enrolls",
        )
    if not _HOST_ID.match(host_id or ""):
        raise EnrollmentError(
            "enroll.host_invalid",
            "host id: 1-63 of a-z 0-9 . _ - starting with a letter or digit",
        )
    return host_id


def _check_ttl(ttl_s: int) -> int:
    if not 0 < int(ttl_s) <= MAX_ENROLL_TTL_S:
        raise EnrollmentError(
            "enroll.ttl_invalid", f"ttl must be 1-{MAX_ENROLL_TTL_S} seconds"
        )
    return int(ttl_s)


@dataclass(frozen=True)
class IssuedEnrollment:
    """The plaintext ``token`` exists only here: print it once, never log it."""

    host_id: str
    venue: str
    token: str
    expires_at: datetime


def issue_enrollment(
    store: HostKeyStore,
    host_id: str,
    *,
    ttl_s: int = DEFAULT_ENROLL_TTL_S,
    created_by: str = "",
    rekey: bool = False,
) -> IssuedEnrollment:
    """Admin side: a one-time token for ``host_id``.

    A host that already has a key is refused (``enroll.already_enrolled``)
    unless ``rekey``: the new key replaces the old one when redeemed."""
    check_host_id(host_id)
    ttl = _check_ttl(ttl_s)
    if not rekey and store.public_key(host_id) is not None:
        raise EnrollmentError(
            "enroll.already_enrolled",
            f"host {host_id!r} is enrolled: pass --rekey to replace its key",
        )
    token = ENROLL_TOKEN_PREFIX + secrets.token_urlsafe(32)
    expires = store.add_enrollment(
        enrollment_token_hash(token), host_id, ttl, created_by
    )
    return IssuedEnrollment(host_id, store.venue, token, expires)


def parse_public_key(text: str) -> bytes:
    """A raw ed25519 public key (32 bytes, base64url) or ``enroll.key_invalid``."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        raw = unb64url(str(text or ""))
        Ed25519PublicKey.from_public_bytes(raw)
    except ValueError as exc:
        raise EnrollmentError(
            "enroll.key_invalid", "public key: raw ed25519, base64url"
        ) from exc
    return raw


def redeem_enrollment(store: HostKeyStore, token: str, public_key: str) -> HostKey:
    """Server side: use the token once and enroll the host's public key."""
    raw = parse_public_key(public_key)
    if not str(token or "").startswith(ENROLL_TOKEN_PREFIX):
        raise EnrollmentError("enroll.token_invalid", "enrollment token refused")
    return store.consume_enrollment(enrollment_token_hash(token), raw)


# ── host side ─────────────────────────────────────────────────────────────


def generate_host_key() -> Any:
    """A fresh ed25519 private key; it never leaves the host."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519PrivateKey.generate()


def public_key_text(private_key: Any) -> str:
    """The base64url raw public key a host enrolls with."""
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    raw = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return b64url(raw)


def enrollment_request(token: str, private_key: Any) -> dict[str, str]:
    """The body a host sends to enroll: the token and its public key only."""
    return {"token": token, "public_key": public_key_text(private_key)}


__all__ = [
    "DEFAULT_ENROLL_TTL_S",
    "ENROLL_TOKEN_PREFIX",
    "MAX_ENROLL_TTL_S",
    "EnrollmentError",
    "IssuedEnrollment",
    "check_host_id",
    "enrollment_request",
    "enrollment_token_hash",
    "generate_host_key",
    "issue_enrollment",
    "parse_public_key",
    "public_key_text",
    "redeem_enrollment",
]
