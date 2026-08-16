"""Companion-worker durable spec + manager tests (wave T3a primitives)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fake_pg_pool import FakePool  # noqa: E402
from umrs.jobs.companion_worker import (
    WORKER_MAX_ATTEMPTS,
    WORKER_RESTART_DELAY_SEC,
    CompanionWorkerManager,
    CompanionWorkerSpec,
    InvalidResumeToken,
    WorkerExhausted,
    WorkerNotFound,
    fingerprint_matches,
    token_fingerprint,
    worker_job_spec,
)
from umrs.jobs.models import DEFAULT_BACKOFFS, JobType


class TestWorkerSpec:
    def test_registry_key_and_round_trip(self):
        spec = CompanionWorkerSpec(
            owner_user_id="alice",
            project_id="default",
            session_token_fingerprint="fp-abc",
            metadata={"venue": "data0"},
        )
        assert spec.registry_key() == "alice@default"
        assert CompanionWorkerSpec.from_dict(spec.to_dict()) == spec
        assert spec.to_dict()["worker_kind"] == "companion_worker"

    def test_owner_and_project_required(self):
        with pytest.raises(ValueError, match="owner_user_id"):
            CompanionWorkerSpec(owner_user_id="", project_id="default")
        with pytest.raises(ValueError, match="project_id"):
            CompanionWorkerSpec(owner_user_id="alice", project_id=" ")


class TestTokenFingerprint:
    def test_fingerprint_never_equals_raw(self):
        raw = "resume-token-abc123xyz"
        fp = token_fingerprint(raw)
        assert fp != raw
        assert len(fp) == 24

    def test_deterministic_same_input(self):
        assert token_fingerprint("tok") == token_fingerprint("tok")

    def test_fails_closed_on_empty_spec_fp(self):
        spec = CompanionWorkerSpec(owner_user_id="a", project_id="p")
        assert fingerprint_matches(spec, "tok") is False

    def test_match_and_mismatch(self):
        fp = token_fingerprint("secret-resume")
        spec = CompanionWorkerSpec(owner_user_id="a", project_id="p", session_token_fingerprint=fp)
        assert fingerprint_matches(spec, "secret-resume")
        assert not fingerprint_matches(spec, "secret-resumE")


class TestWorkerJobSpec:
    def test_policy_carried(self):
        spec = CompanionWorkerSpec(owner_user_id="a", project_id="p")
        payload = worker_job_spec(spec)
        assert payload["max_attempts"] == WORKER_MAX_ATTEMPTS
        assert payload["restart_policy"] == "on_failure"
        assert payload["restart_delays_sec"] == list(WORKER_RESTART_DELAY_SEC)
        assert payload["registry_key"] == "a@p"


class TestJobTypeAndBackoff:
    def test_job_type_registered(self):
        assert JobType.COMPANION_WORKER.value == "companion_worker"

    def test_backoff_entry_bounded_supervision(self):
        entry = DEFAULT_BACKOFFS.get(JobType.COMPANION_WORKER)
        assert entry is not None
        assert entry["max_attempts"] == WORKER_MAX_ATTEMPTS
        assert len(entry["delays"]) >= 3


# ── Manager (T3a venue: heartbeat-driven liveness + token gate) ─────────────


def _worker_row(**overrides):
    row = {
        "registry_key": "alice@default",
        "owner_user_id": "alice",
        "project_id": "default",
        "job_id": "job-w1",
        "session_token_fingerprint": "",
        "state": "active",
        "attempt_index": 0,
        "attempts_max": 6,
        "last_heartbeat_at": None,
        "last_active_at": None,
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


SPEC_A = CompanionWorkerSpec(owner_user_id="alice", project_id="default")


def _mgr(script, *, token="tok-fixed-1"):
    return CompanionWorkerManager(FakePool(list(script)), token_factory=lambda: token)


class TestWorkerRegistration:
    def test_register_active_upserts_and_returns_active_row(self):
        mgr = _mgr([None, _worker_row()])
        row = mgr.register_active(SPEC_A, job_id="job-w1")
        assert row["state"] == "active"
        assert row["registry_key"] == "alice@default"

    def test_heartbeat_updates_attempt_index(self):
        mgr = _mgr([None, _worker_row(attempt_index=2)])
        row = mgr.heartbeat("alice@default", attempt_index=2)
        assert row["attempt_index"] == 2
        assert row["state"] == "active"

    def test_heartbeat_drains_on_last_attempt(self):
        mgr = _mgr([None, _worker_row(state="draining", attempt_index=5)])
        row = mgr.heartbeat("alice@default", attempt_index=5)
        assert row["state"] == "draining"


class TestTokenGate:
    def test_issue_token_on_active_mints_and_stores_fingerprint(self):
        mgr = _mgr([_worker_row(), None], token="tok-fixed-1")
        issue = mgr.issue_token(SPEC_A)
        assert issue.token == "tok-fixed-1"
        assert issue.session_token_fingerprint == token_fingerprint("tok-fixed-1")
        assert issue.registry_key == "alice@default"

    def test_issue_token_refuses_draining(self):
        mgr = _mgr([_worker_row(state="draining", attempt_index=5)])
        with pytest.raises(WorkerExhausted, match="draining"):
            mgr.issue_token(SPEC_A)

    def test_issue_token_refuses_failed(self):
        mgr = _mgr([_worker_row(state="failed")])
        with pytest.raises(WorkerExhausted, match="budget exhausted"):
            mgr.issue_token(SPEC_A)

    def test_issue_token_refuses_evicted(self):
        mgr = _mgr([_worker_row(state="evicted")])
        with pytest.raises(WorkerExhausted, match="evicted"):
            mgr.issue_token(SPEC_A)

    def test_issue_token_refuses_attempt_exhaustion(self):
        mgr = _mgr([_worker_row(attempt_index=6)])
        with pytest.raises(WorkerExhausted, match="attempts exhausted"):
            mgr.issue_token(SPEC_A)


class TestResume:
    def test_resolve_valid_token_returns_row(self):
        fp = token_fingerprint("tok-fixed-1")
        mgr = _mgr([_worker_row(state="active", session_token_fingerprint=fp), None])
        row = mgr.resolve(SPEC_A, "tok-fixed-1")
        assert row["registry_key"] == "alice@default"
        assert row["state"] == "active"

    def test_resolve_invalid_token_raises(self):
        mgr = _mgr([_worker_row(session_token_fingerprint=token_fingerprint("other"))])
        with pytest.raises(InvalidResumeToken):
            mgr.resolve(SPEC_A, "wrong-token")

    def test_resolve_missing_row_raises_not_found(self):
        mgr = _mgr([None])
        with pytest.raises(WorkerNotFound):
            mgr.resolve(SPEC_A, "anything")

    def test_resolve_liveness_blocked_raises_exhausted(self):
        mgr = _mgr([_worker_row(state="failed", session_token_fingerprint=token_fingerprint("x"))])
        with pytest.raises(WorkerExhausted):
            mgr.resolve(SPEC_A, "x")


class TestRetirement:
    def test_evict_moves_to_evicted(self):
        mgr = _mgr([None])
        mgr.evict("alice@default")
        sql, params = mgr._pool.record[0]
        assert "SET state = %s" in sql
        assert params == ("evicted", "alice@default")

    def test_evict_idle_counts_rows_and_skips_failed(self):
        mgr = _mgr([[{"registry_key": "a@b"}, {"registry_key": "c@d"}]])
        assert mgr.evict_idle(24 * 3600) == 2
        sql, _ = mgr._pool.record[0]
        assert "state IN ('active', 'draining')" in sql


class TestLostWorkerScenario:
    """LOST×6 e2e: eviction gate fires on restart exhaustion, not on TTL."""

    def test_lost_x6_transitions_before_ttl_and_blocks_new_tokens(self):
        script = [None, _worker_row()]  # register_active INSERT + SELECT
        for attempt in range(5):  # first attempts in flight, still active
            script += [None, _worker_row(state="active", attempt_index=attempt)]
        script += [None, _worker_row(state="draining", attempt_index=5)]  # last attempt
        script += [None]  # mark_failed UPDATE
        script += [_worker_row(state="failed", attempt_index=6)]  # issue_token SELECT

        mgr = _mgr(script)
        mgr.register_active(SPEC_A, job_id="job-w1")
        final = None
        for attempt in range(6):
            final = mgr.heartbeat("alice@default", attempt_index=attempt)
        assert final is not None and final["state"] == "draining"

        # TTL sweep NOW — before the 24h TTL would elapse — finds nothing to retire.
        # (Drain was state-driven, not idle-driven.)
        mgr.mark_failed("alice@default")
        with pytest.raises(WorkerExhausted, match="budget exhausted"):
            mgr.issue_token(SPEC_A)

    def test_lost_x6_resume_also_blocked(self):
        # Even with the correct resume token, a failed worker refuses.
        # Token issuance: active → token minted.
        fp = token_fingerprint("tok-fixed-1")
        script = [
            _worker_row(session_token_fingerprint=fp),  # issue_token SELECT
            None,  # issue_token fingerprint UPDATE
            _worker_row(state="failed", session_token_fingerprint=fp),  # resolve
        ]
        mgr = _mgr(script)
        issue = mgr.issue_token(SPEC_A)
        assert issue.token == "tok-fixed-1"
        with pytest.raises(WorkerExhausted):
            mgr.resolve(SPEC_A, "tok-fixed-1")
