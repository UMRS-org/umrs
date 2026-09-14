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
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

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
    def from_dict(cls, data: dict) -> CompanionWorkerSpec:
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


# ── Manager + typed errors (T3a venue: authority + heartbeat push) ──────────


class WorkerNotFound(KeyError):
    """No worker row for the registry key (or evicted)."""


class WorkerExhausted(RuntimeError):
    """Liveness gate: job FAILED, draining grace over, or attempts exhausted."""


class InvalidResumeToken(RuntimeError):
    """Resume-token fingerprint does not match the registry row."""


class WorkerFetchError(RuntimeError):
    """Registry row missing after a successful mutation (pool/store inconsistency)."""


@dataclass(frozen=True)
class TokenIssue:
    """Outcome of issue_token: raw token (call-side transport only) + row."""

    token: str
    registry_key: str
    session_token_fingerprint: str


class CompanionWorkerManager:
    """Authority over the companion_workers registry (wave T3a manager).

    Liveness model (design doc §1/§6): the attempt loop PUSHES heartbeats with
    attempt_index; the manager computes exhaustion locally — the registry is
    never a polling client of its own platform. Eviction gates on job state
    FIRST (FAILED/draining set by heartbeat/attempt, not by idle time), then
    on idle TTL; TTL alone must never mark a live worker evicted.
    """

    def __init__(self, db_pool: Any, *, token_factory: Any = None) -> None:
        self._pool = db_pool
        self._make_token = token_factory or (lambda: secrets.token_urlsafe(32))

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = self._pool.getconn()
        try:
            with conn:
                yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        from psycopg2.extras import RealDictCursor

        with self._conn() as conn:
            yield conn.cursor(cursor_factory=RealDictCursor)

    # -- registration / lookup ----------------------------------------------

    def register_active(self, spec: CompanionWorkerSpec, *, job_id: str) -> dict:
        """(Re)bind a worker job as active; reset the attempt counter.

        Job identity beats row identity: a re-deployed worker takes the same
        registry_key with a clean restart budget.
        """
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO companion_workers "
                "(registry_key, owner_user_id, project_id, job_id, "
                " session_token_fingerprint, state) "
                "VALUES (%s, %s, %s, %s, %s, 'active') "
                "ON CONFLICT (registry_key) DO UPDATE "
                "SET job_id = EXCLUDED.job_id, state = 'active', "
                "    attempt_index = 0, attempts_max = EXCLUDED.attempts_max, "
                "    session_token_fingerprint = EXCLUDED.session_token_fingerprint, "
                "    last_active_at = now(), updated_at = now()",
                (
                    spec.registry_key(),
                    spec.owner_user_id,
                    spec.project_id,
                    job_id,
                    spec.session_token_fingerprint,
                ),
            )
        return self._fetch_required(spec.registry_key())

    def get(self, registry_key: str) -> dict | None:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM companion_workers WHERE registry_key = %s",
                (registry_key,),
            )
            return cur.fetchone()

    def _fetch_required(self, registry_key: str) -> dict:
        row = self.get(registry_key)
        if row is None:
            raise WorkerFetchError(f"registry row missing for {registry_key!r}")
        return row

    # -- heartbeat (push) ----------------------------------------------------

    def heartbeat(
        self,
        registry_key: str,
        *,
        attempt_index: int,
        attempts_max: int = WORKER_MAX_ATTEMPTS,
    ) -> dict | None:
        """Register one attempt-loop heartbeat; degrade on the last attempt.

        attempt_index >= attempts_max - 1 → state='draining' (graceful reconnect
        window for the in-flight session; new token issuance is refused).
        FAILED/EVICTED rows ignore further heartbeats.
        """
        with self._cursor() as cur:
            cur.execute(
                "UPDATE companion_workers "
                "SET attempt_index = %s, attempts_max = %s, "
                "    last_heartbeat_at = now(), last_active_at = now(), "
                "    updated_at = now(), "
                "    state = CASE "
                "      WHEN %s >= %s - 1 AND state = 'active' THEN 'draining' "
                "      ELSE state END "
                "WHERE registry_key = %s AND state IN ('active', 'draining')",
                (
                    attempt_index,
                    attempts_max,
                    registry_key,
                    attempt_index,
                    attempts_max,
                ),
            )
        return self.get(registry_key)

    def mark_failed(self, registry_key: str) -> None:
        """Attempt loop closed FAILED: the restart budget is gone for good."""
        self._set_state(registry_key, "failed")

    def _set_state(self, registry_key: str, state: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE companion_workers SET state = %s, updated_at = now() "
                "WHERE registry_key = %s",
                (state, registry_key),
            )

    # -- token issuance (liveness gate, not idle-time) -----------------------

    def issue_token(self, spec: CompanionWorkerSpec) -> TokenIssue:
        """Mint a fresh resume token — fail closed on any liveness signal.

        Checks = token ≠ auth and token ≠ liveness:
          (1) row must exist and be 'active',
          (2) attempts_remaining (attempts_max - attempt_index) >= 1.
        """
        key = spec.registry_key()
        row = self._fetch_required(key)
        self._assert_liveness(row)
        raw = self._make_token()
        fingerprint = token_fingerprint(raw)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE companion_workers "
                "SET session_token_fingerprint = %s, "
                "    last_active_at = now(), updated_at = now() "
                "WHERE registry_key = %s AND state = 'active'",
                (fingerprint, key),
            )
        return TokenIssue(
            token=raw, registry_key=key, session_token_fingerprint=fingerprint
        )

    @staticmethod
    def _assert_liveness(row: dict) -> None:
        state = str(row.get("state") or "")
        if state == "failed":
            raise WorkerExhausted("worker restart budget exhausted (job failed)")
        if state == "evicted":
            raise WorkerExhausted("worker evicted")
        if state == "draining":
            raise WorkerExhausted("worker draining: reconnect grace window only")
        if state != "active":
            raise WorkerExhausted(f"worker not active (state={state!r})")
        remaining = int(row.get("attempts_max", 0)) - int(row.get("attempt_index", 0))
        if remaining < 1:
            raise WorkerExhausted("worker attempts exhausted")

    # -- resume (reattach) ----------------------------------------------------

    def resolve(self, spec: CompanionWorkerSpec, session_token: str) -> dict:
        """Resume-path lookup: token + liveness verified, returns the row."""
        row = self.get(spec.registry_key())
        if row is None:
            raise WorkerNotFound(spec.registry_key())
        if not fingerprint_matches(
            CompanionWorkerSpec(
                owner_user_id=spec.owner_user_id,
                project_id=spec.project_id,
                session_token_fingerprint=str(
                    row.get("session_token_fingerprint") or ""
                ),
            ),
            session_token,
        ):
            raise InvalidResumeToken(spec.registry_key())
        self._assert_liveness(row)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE companion_workers SET last_active_at = now() "
                "WHERE registry_key = %s",
                (spec.registry_key(),),
            )
        return row

    # -- retirement -----------------------------------------------------------

    def evict(self, registry_key: str) -> None:
        """Explicit retirement (user stop / admin)."""
        self._set_state(registry_key, "evicted")

    def evict_idle(self, idle_seconds: int) -> int:
        """evict rows inactive longer than the TTL — active and draining only.

        Returns rows evicted. Draining rows expire too: the grace window is
        measured by TTL, not by liveness signals alone.
        """
        with self._cursor() as cur:
            cur.execute(
                "UPDATE companion_workers SET state = 'evicted', updated_at = now() "
                "WHERE state IN ('active', 'draining') "
                "  AND last_active_at < now() - make_interval(secs => %s) "
                "RETURNING registry_key",
                (idle_seconds,),
            )
            return len(cur.fetchall())


# ── Turn-attempt registry (T3b m2/m3, ADR-028) ──────────────────────────────


class TurnAttemptNotFound(KeyError):
    """attempt_id unknown to the turn registry."""


@dataclass(frozen=True)
class TurnAttempt:
    """One companion turn attempt row as the bridge to the identity contract.

    T3c content plane (migration 0074): the venue persists what it
    executes — ``user_message`` (the m5 hash alone could not be
    executed), ``response_content`` (ADR-028 replay body), and the
    per-attempt permission ``mode`` ('ask' | 'auto').
    """

    attempt_id: str
    job_id: str
    registry_key: str
    session_id: str
    request_token: str
    user_message_hash: str
    state: str
    committed_response_ref: str | None
    parent_attempt_id: str | None
    research_session_id: str | None
    user_message: str = ""
    response_content: str | None = None
    mode: str = "ask"

    @classmethod
    def from_row(cls, row: dict) -> TurnAttempt:
        return cls(
            attempt_id=str(row["attempt_id"]),
            job_id=str(row.get("job_id") or ""),
            registry_key=str(row["registry_key"]),
            session_id=str(row.get("session_id") or ""),
            request_token=str(row.get("request_token") or ""),
            user_message_hash=str(row.get("user_message_hash") or ""),
            state=str(row.get("state") or "in_flight"),
            committed_response_ref=row.get("committed_response_ref"),
            parent_attempt_id=row.get("parent_attempt_id"),
            research_session_id=row.get("research_session_id"),
            user_message=str(row.get("user_message") or ""),
            response_content=row.get("response_content"),
            mode=str(row.get("mode") or "ask"),
        )


def user_attempt_token_hash(message: str) -> str:
    """Deterministic content hash for a turn's user input; empty-safe."""
    return hashlib.sha256(message.strip().encode("utf-8")).hexdigest()[:16]


def generate_attempt_id() -> str:
    """Platform attempt identifier; compact, URL-safe."""
    return f"at-{secrets.token_hex(6)}"


class CompanionTurnAttemptRegistry:
    """Turn-attempt authority: begin/complete/fail/get under ADR-028 semantics.

    Exactly-once semantics: begin inserts ON CONFLICT DO NOTHING and returns
    the present row; complete marks only 'in_flight' (a failure or commit is
    terminal); fail marks only 'in_flight'. Re-issuing a committed request
    returns the committed row without a second commit.
    """

    def __init__(self, db_pool: Any) -> None:
        self._pool = db_pool

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = self._pool.getconn()
        try:
            with conn:
                yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        from psycopg2.extras import RealDictCursor

        with self._conn() as conn:
            yield conn.cursor(cursor_factory=RealDictCursor)

    _SELECT_COLS = (
        "attempt_id, job_id, registry_key, session_id, request_token, "
        "user_message_hash, state, committed_response_ref, parent_attempt_id, "
        "research_session_id, user_message, response_content, mode"
    )

    def get(self, attempt_id: str) -> TurnAttempt | None:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {self._SELECT_COLS} FROM companion_turns WHERE attempt_id = %s",
                (attempt_id,),
            )
            row = cur.fetchone()
        return TurnAttempt.from_row(row) if row else None

    def _required(self, attempt_id: str) -> TurnAttempt:
        attempt = self.get(attempt_id)
        if attempt is None:
            raise TurnAttemptNotFound(attempt_id)
        return attempt

    def begin(
        self,
        *,
        job_id: str,
        registry_key: str,
        session_id: str,
        request_token: str,
        user_message: str,
        parent_attempt_id: str | None = None,
        research_session_id: str | None = None,
        mode: str = "ask",
    ) -> TurnAttempt:
        """start or reuse the attempt row for a request token (idempotent).

        A re-issue over an existing (committed or in-flight) request token
        returns the existing row; a failed request is not restarted by begin —
        callers create a REATTEMPT (parent_attempt_id pointing at the failed
        row), which yields a fresh attempt with a new request token.
        """
        attempt_id = generate_attempt_id()
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO companion_turns "
                "(attempt_id, job_id, registry_key, session_id, request_token, "
                " user_message_hash, state, parent_attempt_id, research_session_id, "
                " user_message, mode) "
                "VALUES (%s, %s, %s, %s, %s, %s, 'in_flight', %s, %s, %s, %s) "
                "ON CONFLICT (registry_key, request_token) DO NOTHING",
                (
                    attempt_id,
                    job_id,
                    registry_key,
                    session_id,
                    request_token,
                    user_attempt_token_hash(user_message),
                    parent_attempt_id,
                    research_session_id,
                    user_message,
                    mode,
                ),
            )
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {self._SELECT_COLS} FROM companion_turns "
                "WHERE registry_key = %s AND request_token = %s",
                (registry_key, request_token),
            )
            row = cur.fetchone()
        if row is None:  # pragma: no cover -
            raise TurnAttemptNotFound(request_token)
        return TurnAttempt.from_row(row)

    def complete(
        self,
        attempt_id: str,
        *,
        committed_response_ref: str,
        response_content: str | None = None,
    ) -> None:
        """Commit the buffered response exactly-once; idempotent per attempt."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE companion_turns "
                "SET state = 'committed', committed_response_ref = %s, "
                "    response_content = COALESCE(%s, response_content), "
                "    updated_at = now() "
                "WHERE attempt_id = %s AND state = 'in_flight'",
                (committed_response_ref, response_content, attempt_id),
            )

    def fail(self, attempt_id: str) -> None:
        """Mark an attempt failed; exactly-once (already-committed no-ops)."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE companion_turns SET state = 'failed', updated_at = now() "
                "WHERE attempt_id = %s AND state = 'in_flight'",
                (attempt_id,),
            )

    def fail_if_stale(self, attempt_id: str, ttl_seconds: int) -> bool:
        """Fail an in_flight attempt older than ``ttl_seconds`` (the reaper).

        Attempts orphaned by a sidecar reload would otherwise poll as
        in_flight forever (T3c follow-up). Idempotent — a committed or
        already-failed attempt never matches the WHERE guard. Returns True
        only when the row was actually reaped.
        """
        with self._cursor() as cur:
            cur.execute(
                "UPDATE companion_turns SET state = 'failed', updated_at = now() "
                "WHERE attempt_id = %s AND state = 'in_flight' "
                "AND updated_at < now() - make_interval(secs => %s)",
                (attempt_id, int(ttl_seconds)),
            )
            return getattr(cur, "rowcount", 0) > 0

    def list_recent(self, registry_key: str, *, limit: int = 50) -> list[TurnAttempt]:
        """Recent attempts in replay order (newest first)."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {self._SELECT_COLS} FROM companion_turns "
                "WHERE registry_key = %s ORDER BY created_at DESC LIMIT %s",
                (registry_key, limit),
            )
            rows = cur.fetchall()
        return [TurnAttempt.from_row(row) for row in rows]
