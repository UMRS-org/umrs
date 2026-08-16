"""Durable companion-worker primitives (wave T3a, venue primitives).

Per-(user, project) long-lived Companion worker registered with UMRS as a
durable job. Scope of this module is narrow: the job-type carrier spec for
worker registration plus session-token fingerprinting. Execution execution
venue (spawn supervision), request routing, and the /gw/companion/* surface
arrive with T3b/T3c per docs/development/wave-companion-t3-venue-design.md.

Security note: session resume tokens are NEVER persisted — only their
SHA-256 fingerprint prefix lands in the durable spec (compare-on-hold on
reattach; token value travels only over the session transport).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


WORKER_KIND = "companion_worker"

# How the worker's liveness is governed — mirrors the design doc §4 policy
# and the default attempt policy in umrs.jobs.models.DEFAULT_BACKOFFS.
WORKER_RESTART_DELAY_SEC = (30, 60, 300, 600)
WORKER_MAX_ATTEMPTS = 6


@dataclass(frozen=True)
class CompanionWorkerSpec:
    """(user, project) frozen-contract spec for a companion_worker job."""

    owner_user_id: str
    project_id: str
    session_token_fingerprint: str = ""
    metadata: dict = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.owner_user_id.strip():
            raise ValueError("owner_user_id required")
        if not self.project_id.strip():
            raise ValueError("project_id required")

    def registry_key(self) -> str:
        """Stable per-(user, project) key for the active-worker registry."""
        return f"{self.owner_user_id}@{self.project_id}"

    def to_dict(self) -> dict:
        return {
            "worker_kind": WORKER_KIND,
            "owner_user_id": self.owner_user_id,
            "project_id": self.project_id,
            "session_token_fingerprint": self.session_token_fingerprint,
            "registry_key": self.registry_key(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompanionWorkerSpec":
        return cls(
            owner_user_id=str(data.get("owner_user_id") or ""),
            project_id=str(data.get("project_id") or ""),
            session_token_fingerprint=str(data.get("session_token_fingerprint") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


def token_fingerprint(session_token: str) -> str:
    """Persistable non-reversible fingerprint for a resume token.

    Collisions against different tokens are fine for a same-spec comparison
    (256-bit sha2 prefix over ~2^80 candidate space is Way beyond needs);
    raw tokens NEVER land in the durable spec.
    """
    if not session_token.strip():
        return ""
    return hashlib.sha256(session_token.encode("utf-8")).hexdigest()[:24]


def fingerprint_matches(spec: CompanionWorkerSpec, session_token: str) -> bool:
    """Constant-shape token check used at reattach; empty fingerprint fails closed."""
    if spec.session_token_fingerprint == "":
        return False
    return token_fingerprint(session_token) == spec.session_token_fingerprint


def worker_job_spec(spec: CompanionWorkerSpec) -> dict:
    """Build the jobs-layer spec payload for a companion_worker durable job."""
    payload = spec.to_dict()
    payload["restart_policy"] = "on_failure"
    payload["max_attempts"] = WORKER_MAX_ATTEMPTS
    payload["restart_delays_sec"] = list(WORKER_RESTART_DELAY_SEC)
    return payload
