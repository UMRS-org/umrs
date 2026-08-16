"""Companion-worker durable spec tests (wave T3a primitives)."""

from __future__ import annotations

import pytest
from umrs.jobs.companion_worker import (
    WORKER_MAX_ATTEMPTS,
    WORKER_RESTART_DELAY_SEC,
    CompanionWorkerSpec,
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
        spec = CompanionWorkerSpec(
            owner_user_id="a", project_id="p", session_token_fingerprint=fp
        )
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
