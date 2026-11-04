"""Signed scope tokens of work levels (ADR-108 D-3, W2b). Pure: no I/O.

A loop, or a teamup run, acts for a user for hours. Before this, the only
mark of an agent call was the self-declared ``x-r4-actor`` header. A scope
token is issued by UMRS on the caller's verified identity. It names the
user, the project, the level's path (``s-ab12/loop-7/iter-3``), its parent,
the git branch and commit the level works on, what the level may do
(grants), and an expiry.

- **Format:** ``base64url(json payload) "." base64url(HMAC-SHA256)``.
- **Key:** HKDF-SHA256 of ``UMRS_HANDSHAKE_HMAC_SECRET`` with its own info
  string, so no new secret is provisioned and the gateway's handshake
  signature can never be replayed as a scope token (or the reverse).
- **A child** narrows its parent: its path extends the parent's, its grants
  are a subset, and it never outlives the parent.
- **Grants:** ``ingest``, ``upload``, ``web``, ``workflow:<name>`` and
  ``workflow:*`` (any workflow).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

VERSION = 1
DEFAULT_TTL_SEC = 12 * 3600
MAX_TTL_SEC = 72 * 3600
MAX_DEPTH = 8
HEADER = "x-r4-scope"
_INFO = b"r4 scope token v1"
_SALT = b"r4-adr108-scope"
_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_PLAIN_GRANTS = frozenset({"ingest", "upload", "web"})
_WORKFLOW = re.compile(r"^workflow:(\*|[A-Za-z0-9][A-Za-z0-9._-]{0,127})$")


class ScopeError(ValueError):
    """A token or request that cannot stand; ``code`` is the scopes.* name."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ScopeClaims:
    user: str
    project: str
    path: str
    grants: tuple[str, ...]
    iat: int
    exp: int
    jti: str
    parent: str = ""
    repo: str = ""
    branch: str = ""
    commit: str = ""
    v: int = VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        out = asdict(self)
        out["grants"] = list(self.grants)
        out.pop("extra", None)
        return out


# ── key and encoding ───────────────────────────────────────────────────────


def derive_key(secret: str) -> bytes:
    """HKDF-SHA256 (one block) of the handshake secret for scope tokens."""
    if not secret:
        raise ScopeError("unavailable", "scope tokens need UMRS_HANDSHAKE_HMAC_SECRET")
    prk = hmac.new(_SALT, secret.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(prk, _INFO + b"\x01", hashlib.sha256).digest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def encode(claims: ScopeClaims, secret: str) -> str:
    payload = json.dumps(claims.record(), sort_keys=True, separators=(",", ":")).encode()
    signature = hmac.new(derive_key(secret), payload, hashlib.sha256).digest()
    return f"{_b64(payload)}.{_b64(signature)}"


def _payload(token: str, secret: str) -> dict[str, Any]:
    """The verified payload, or ScopeError("invalid")."""
    try:
        body, sig = token.strip().split(".", 1)
        payload, signature = _unb64(body), _unb64(sig)
    except (ValueError, AttributeError) as exc:
        raise ScopeError("invalid", "malformed scope token") from exc
    expected = hmac.new(derive_key(secret), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ScopeError("invalid", "scope token signature does not match")
    try:
        data = json.loads(payload)
    except ValueError as exc:
        raise ScopeError("invalid", "malformed scope token") from exc
    if not isinstance(data, dict) or data.get("v") != VERSION:
        raise ScopeError("invalid", "unknown scope token version")
    return data


def decode(token: str, secret: str, *, now: float | None = None) -> ScopeClaims:
    """The claims of a valid, unexpired token (revocation is the caller's)."""
    data = _payload(token, secret)
    try:
        claims = ScopeClaims(**{**data, "grants": tuple(data.get("grants") or ())})
    except TypeError as exc:
        raise ScopeError("invalid", "malformed scope token") from exc
    if (now if now is not None else time.time()) >= claims.exp:
        raise ScopeError("invalid", f"scope token for {claims.path} expired")
    return claims


# ── paths and grants ───────────────────────────────────────────────────────


def check_path(path: str) -> str:
    segments = (path or "").split("/")
    if not path or len(segments) > MAX_DEPTH or not all(_SEGMENT.match(s) for s in segments):
        raise ScopeError(
            "bad_request",
            f"scope path must be 1-{MAX_DEPTH} segments of letters, digits, '.', '_', '-'",
        )
    return path


def check_grants(grants: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for grant in grants:
        text = str(grant).strip()
        if text not in _PLAIN_GRANTS and not _WORKFLOW.match(text):
            raise ScopeError("bad_request", f"unknown grant {text!r}")
        if text not in out:
            out.append(text)
    return tuple(out)


def allows(grants: Iterable[str], wanted: str) -> bool:
    """True when ``grants`` permit ``wanted`` (``workflow:*`` covers any)."""
    held = set(grants)
    if wanted in held:
        return True
    return wanted.startswith("workflow:") and "workflow:*" in held


def _within(child: str, parent: str) -> bool:
    return child.startswith(parent + "/")


def revoked_by(path: str, prefix: str) -> bool:
    """True when a revocation of ``prefix`` covers ``path``."""
    return path == prefix or _within(path, prefix)


# ── issuing ────────────────────────────────────────────────────────────────


def _ttl(ttl: int | None) -> int:
    value = DEFAULT_TTL_SEC if ttl is None else int(ttl)
    if value <= 0 or value > MAX_TTL_SEC:
        raise ScopeError("bad_request", f"ttl must be 1..{MAX_TTL_SEC} seconds")
    return value


def _narrow(parent: ScopeClaims, user: str, project: str, path: str, grants: tuple[str, ...]) -> None:
    if (parent.user, parent.project) != (user, project):
        raise ScopeError("forbidden", "a child scope belongs to its parent's user and project")
    if not _within(path, parent.path):
        raise ScopeError("forbidden", f"{path} is not beneath {parent.path}")
    extra = [g for g in grants if not allows(parent.grants, g)]
    if extra:
        raise ScopeError("forbidden", f"grants beyond the parent's: {', '.join(extra)}")


def _window(parent: ScopeClaims | None, ttl: int | None, now: float | None) -> tuple[int, int]:
    """``(iat, exp)``: a child never outlives its parent."""
    start = int(now if now is not None else time.time())
    exp = start + _ttl(ttl)
    return start, (exp if parent is None else min(exp, parent.exp))


def issue(
    *,
    user: str,
    project: str,
    path: str,
    grants: Iterable[str],
    secret: str,
    parent: ScopeClaims | None = None,
    ttl: int | None = None,
    repo: str = "",
    branch: str = "",
    commit: str = "",
    now: float | None = None,
) -> tuple[str, ScopeClaims]:
    """A new token; a child never exceeds its parent (path, grants, expiry)."""
    if not user or not project:
        raise ScopeError("forbidden", "a scope needs a verified user and project")
    path, wanted = check_path(path), check_grants(grants)
    if parent is not None:
        _narrow(parent, user, project, path, wanted)
    start, exp = _window(parent, ttl, now)
    claims = ScopeClaims(
        user=user,
        project=project,
        path=path,
        grants=wanted,
        iat=start,
        exp=exp,
        jti=secrets.token_hex(8),
        parent=parent.path if parent is not None else "",
        repo=str(repo or "")[:512],
        branch=str(branch or "")[:255],
        commit=str(commit or "")[:64],
    )
    return encode(claims, secret), claims
