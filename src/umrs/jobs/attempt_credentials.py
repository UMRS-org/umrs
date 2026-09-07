"""Attempt-scoped short-lived S3 credentials (WP-11, G-P5-3).

Until now every presigned URL derived from ambient site credentials with a
flat one-hour TTL and no binding to the job/attempt asking for it — any
attempt could ask for any key. This module mints an **attempt-keyed
credential record**: a key-space scoped to ``jobs/<job_id>/attempt_<n>/``,
a short TTL (default 900s), and an auditable minted/expiry timestamp. Down-
stream presign verbs REFUSE keys outside the recorded prefix.

The signing party is still the site's ambient credentials in round 1 —
true per-attempt identity needs STS federation (named round-2 in the ADR,
not masqueraded). What round 1 *removes*: unscoped keyspace, one-hour
flat TTLs, and silent issuing (env keys never appear in records; only the
scope, TTL and expiry are stored — nothing secret ever lands in metrics).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

DEFAULT_ATTEMPT_TTL_SEC = int(os.environ.get("UMRS_ATTEMPT_CRED_TTL_SEC", "900"))


@dataclass(frozen=True)
class AttemptCredentials:
    """A minted attempt-scoped credential record (nothing secret stored)."""

    job_id: str
    attempt_id: str
    attempt_no: int
    scope_prefixes: tuple[str, ...]
    ttl_sec: int
    minted_at: str
    expires_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "attempt_no": self.attempt_no,
            "scope_prefixes": list(self.scope_prefixes),
            "ttl_sec": self.ttl_sec,
            "minted_at": self.minted_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttemptCredentials:
        scope = data.get("scope_prefixes") or []
        return cls(
            job_id=str(data.get("job_id") or ""),
            attempt_id=str(data.get("attempt_id") or ""),
            attempt_no=int(data.get("attempt_no") or 0),
            scope_prefixes=tuple(str(p) for p in scope),
            ttl_sec=int(data.get("ttl_sec") or DEFAULT_ATTEMPT_TTL_SEC),
            minted_at=str(data.get("minted_at") or ""),
            expires_at=str(data.get("expires_at") or ""),
        )


def scope_prefix_for(job_id: str, attempt_no: int) -> str:
    """The attempt's S3 key space (durable convention: attempt_<n>)."""
    return f"jobs/{job_id}/attempt_{attempt_no}/"


def mint_attempt_credentials(
    job_id: str,
    attempt_id: str,
    attempt_no: int,
    *,
    ttl_sec: int = DEFAULT_ATTEMPT_TTL_SEC,
    extra_prefixes: tuple[str, ...] = (),
    now: datetime | None = None,
) -> AttemptCredentials:
    """Mint the credential record for one attempt.

    Scope = the attempt's durable output key space plus any explicitly
    declared extra prefixes (never wildcards).
    """
    if ttl_sec <= 0:
        raise ValueError("ttl_sec must be positive")
    moment = now or datetime.now(timezone.utc)
    prefixes = (scope_prefix_for(job_id, attempt_no),)
    if extra_prefixes:
        prefixes = prefixes + tuple(extra_prefixes)
    return AttemptCredentials(
        job_id=str(job_id),
        attempt_id=str(attempt_id),
        attempt_no=int(attempt_no),
        scope_prefixes=tuple(prefixes),
        ttl_sec=int(ttl_sec),
        minted_at=moment.isoformat(),
        expires_at=(moment + timedelta(seconds=int(ttl_sec))).isoformat(),
    )


def key_in_scope(
    key: str, credentials: AttemptCredentials, *, now: datetime | None = None
) -> bool:
    """May the credential mint a URL for this key RIGHT NOW?

    A minted URL is refused when: the record is expired, or the key is not
    under a recorded prefix.
    """
    moment = now or datetime.now(timezone.utc)
    expires = datetime.fromisoformat(credentials.expires_at)
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    if moment >= expires:
        return False
    norm = key.lstrip("/")
    return any(norm.startswith(prefix) for prefix in credentials.scope_prefixes)
