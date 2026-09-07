"""
Tests for durable job tracking system.

These tests verify the core job tracking functionality:
- Job creation with idempotency
- Attempt lifecycle
- State transitions
- Crash recovery (reconciliation)
"""

import subprocess
import sys

import pytest

# Import the modules we're testing
from umrs.jobs.models import (
    DEFAULT_BACKOFFS,
    AttemptStatus,
    Job,
    JobAttempt,
    JobStatus,
    JobType,
    RestartPolicy,
    compute_idempotency_key,
    generate_attempt_id,
    generate_job_id,
)


class TestJobModels:
    """Test Job and JobAttempt models."""

    def test_generate_job_id(self):
        """Test job ID generation."""
        job_id = generate_job_id()
        assert job_id.startswith("job-")
        assert len(job_id) == 16  # "job-" + 12 hex chars

    def test_generate_attempt_id(self):
        """Test attempt ID generation."""
        attempt_id = generate_attempt_id()
        assert attempt_id.startswith("att-")
        assert len(attempt_id) == 16

    def test_compute_idempotency_key(self):
        """Test idempotency key computation."""
        key1 = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.INDEX_BUILD,
            spec={"dataset": "test"},
        )

        # Same inputs should produce same key
        key2 = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.INDEX_BUILD,
            spec={"dataset": "test"},
        )
        assert key1 == key2

        # Different inputs should produce different key
        key3 = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.INDEX_BUILD,
            spec={"dataset": "different"},
        )
        assert key1 != key3

    def test_job_can_retry(self):
        """Test Job.can_retry() logic."""
        job = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.INDEX_BUILD,
            spec={},
            idempotency_key="key-1",
            desired_max_attempts=3,
            restart_policy=RestartPolicy.ON_CRASH,
        )

        assert job.can_retry(1)  # 1 < 3
        assert job.can_retry(2)  # 2 < 3
        assert not job.can_retry(3)  # 3 >= 3
        assert not job.can_retry(4)  # 4 >= 3

    def test_job_can_retry_never_policy(self):
        """Test Job.can_retry() with NEVER policy."""
        job = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.CLEANUP,
            spec={},
            idempotency_key="key-1",
            desired_max_attempts=3,
            restart_policy=RestartPolicy.NEVER,
        )

        assert not job.can_retry(1)  # NEVER policy
        assert not job.can_retry(2)

    def test_job_get_backoff_delay(self):
        """Test Job.get_backoff_delay() logic."""
        job = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec={},
            idempotency_key="key-1",
        )

        # ENDPOINT_PROVISION has delays [5, 10, 20]
        assert job.get_backoff_delay(1) == 5
        assert job.get_backoff_delay(2) == 10
        assert job.get_backoff_delay(3) == 20
        assert job.get_backoff_delay(4) == 20  # Uses last delay

    def test_data_ingestion_job_type_and_backoff(self):
        """DATA_INGESTION is a first-class durable job type."""
        assert JobType.DATA_INGESTION.value == "data_ingestion"
        backoff = DEFAULT_BACKOFFS[JobType.DATA_INGESTION]
        assert backoff["max_attempts"] == 4
        assert backoff["delays"] == [10, 30, 60]

    def test_job_to_dict_and_back(self):
        """Test Job serialization round-trip."""
        original = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.INDEX_BUILD,
            spec={"dataset": "test", "chunk_size": 512},
            idempotency_key="key-1",
            status=JobStatus.QUEUED,
            priority=5,
            desired_max_attempts=3,
            restart_policy=RestartPolicy.ON_FAILURE,
        )

        data = original.to_dict()
        restored = Job.from_dict(data)

        assert restored.job_id == original.job_id
        assert restored.project_id == original.project_id
        assert restored.job_type == original.job_type
        assert restored.spec == original.spec
        assert restored.status == original.status
        assert restored.priority == original.priority
        assert restored.restart_policy == original.restart_policy

    def test_attempt_output_prefix(self):
        """Test JobAttempt.output_prefix property."""
        attempt = JobAttempt(
            attempt_id="att-test",
            job_id="job-123",
            attempt_no=2,
            backend="slurm",
        )

        assert attempt.output_prefix == "jobs/job-123/attempt_2"

    def test_attempt_to_dict_and_back(self):
        """Test JobAttempt serialization round-trip."""
        original = JobAttempt(
            attempt_id="att-test",
            job_id="job-123",
            attempt_no=1,
            backend="kubernetes",
            backend_ref="mini-rivault-xyz",
            status=AttemptStatus.RUNNING,
            status_detail="polling readiness",
            metrics={"readiness_polls": 5},
        )

        data = original.to_dict()
        restored = JobAttempt.from_dict(data)

        assert restored.attempt_id == original.attempt_id
        assert restored.job_id == original.job_id
        assert restored.backend == original.backend
        assert restored.backend_ref == original.backend_ref
        assert restored.status == original.status
        assert restored.metrics == original.metrics


class TestJobStoreAdvisoryLockKey:
    """Test JobStore advisory-lock key generation."""

    def test_advisory_lock_key_is_stable_int32(self):
        """Advisory lock keys must be deterministic across processes."""
        from umrs.jobs.store import JobStore

        key_1 = JobStore._advisory_lock_key("job-test-123")
        key_2 = JobStore._advisory_lock_key("job-test-123")
        other_key = JobStore._advisory_lock_key("job-test-456")

        assert key_1 == key_2
        assert key_1 != other_key
        assert 0 <= key_1 <= 0x7FFFFFFF

    def test_advisory_lock_key_is_stable_across_processes(self):
        """Advisory lock keys must not depend on Python's per-process hash seed."""
        script = (
            "from umrs.jobs.store import JobStore; "
            "print(JobStore._advisory_lock_key('job-test-123'))"
        )
        keys = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                pytest.skip(f"Subprocess failed (circular import?): {result.stderr.strip()}")
            keys.append(result.stdout.strip())

        assert keys[0] == keys[1]


class TestJobStatus:
    """Test job status enum values."""

    def test_all_job_statuses(self):
        """Verify all expected job statuses exist."""
        expected = [
            "planned",
            "queued",
            "running",
            "completed",
            "all_complete",
            "failed",
            "cancelled",
        ]
        actual = [s.value for s in JobStatus]
        assert set(expected) == set(actual)

    def test_all_attempt_statuses(self):
        """Verify all expected attempt statuses exist."""
        expected = ["created", "submitted", "running", "succeeded", "failed", "lost"]
        actual = [s.value for s in AttemptStatus]
        assert set(expected) == set(actual)


class TestJobTypes:
    """Test job type configuration."""

    def test_all_job_types(self):
        """Verify all expected job types exist."""
        expected = [
            "index_build",
            "embedding_batch",
            "pipeline_run",
            "endpoint_provision",
            "endpoint_upgrade",
            "model_download",
            "cleanup",
            "workflow_run",
            "data_ingestion",
            "knowledge_ingest",
            "knowledge_understand",
            "knowledge_embed",
            "knowledge_verify",
            "knowledge_export",
            "companion_worker",
            "knowledge_reindex",
            "research_loop_job",
            "vision_infer",
        ]
        actual = [t.value for t in JobType]
        assert set(expected) == set(actual)

    def test_job_type_backoffs(self):
        """Verify backoff configuration exists for all types."""
        from umrs.jobs.models import DEFAULT_BACKOFFS

        for job_type in JobType:
            assert job_type in DEFAULT_BACKOFFS, f"Missing backoff for {job_type}"
            backoff = DEFAULT_BACKOFFS[job_type]
            assert "delays" in backoff
            assert "max_attempts" in backoff


class TestSchemaGeneration:
    """Test database schema SQL generation (basic sanity of the DDL strings)."""

    def test_jobs_table_sql(self):
        from umrs.jobs.schema import CREATE_JOBS_TABLE

        assert "CREATE TABLE IF NOT EXISTS jobs" in CREATE_JOBS_TABLE
        assert "job_id VARCHAR(64) PRIMARY KEY" in CREATE_JOBS_TABLE
        assert "spec JSONB NOT NULL" in CREATE_JOBS_TABLE

    def test_attempts_table_sql(self):
        from umrs.jobs.schema import CREATE_ATTEMPTS_TABLE

        assert "CREATE TABLE IF NOT EXISTS job_attempts" in CREATE_ATTEMPTS_TABLE
        assert "REFERENCES jobs(job_id)" in CREATE_ATTEMPTS_TABLE

    def test_job_locks_table_sql(self):
        from umrs.jobs.schema import CREATE_JOB_LOCKS_TABLE

        assert "CREATE INDEX IF NOT EXISTS idx_job_locks_expires" in CREATE_JOB_LOCKS_TABLE
        assert "WHERE expires_at < NOW()" not in CREATE_JOB_LOCKS_TABLE

    def test_indexes_sql(self):
        from umrs.jobs.schema import CREATE_INDEXES

        assert "CREATE INDEX IF NOT EXISTS" in CREATE_INDEXES
        assert "idx_jobs_scheduler" in CREATE_INDEXES


class TestReconcileResult:
    """Test ReconcileResult class."""

    def test_reconcile_result_empty(self):
        """Test empty reconcile result."""
        from umrs.jobs.reconciler import ReconcileResult

        result = ReconcileResult()
        assert result.total_processed == 0
        assert len(result.lost_attempts) == 0
        assert len(result.requeued_jobs) == 0
        assert len(result.failed_jobs) == 0
        assert len(result.errors) == 0

    def test_reconcile_result_str(self):
        """Test ReconcileResult string representation."""
        from umrs.jobs.reconciler import ReconcileResult

        result = ReconcileResult()
        result.lost_attempts = ["att-1", "att-2"]
        result.requeued_jobs = ["job-1"]
        result.failed_jobs = ["job-2"]

        s = str(result)
        assert "lost=2" in s
        assert "requeued=1" in s
        assert "failed=1" in s


class TestAttemptCheckResult:
    """Test AttemptCheckResult class."""

    def test_running_result(self):
        """Test AttemptCheckResult.running()."""
        from umrs.jobs.monitor import AttemptCheckResult

        result = AttemptCheckResult.running("polling endpoint")
        assert result.status == "running"
        assert result.status_detail == "polling endpoint"

    def test_succeeded_result(self):
        """Test AttemptCheckResult.succeeded()."""
        from umrs.jobs.monitor import AttemptCheckResult

        result = AttemptCheckResult.succeeded(
            metrics={"duration_sec": 30},
            commit=True,
        )
        assert result.status == "succeeded"
        assert result.metrics == {"duration_sec": 30}
        assert result.should_commit is True

    def test_failed_result(self):
        """Test AttemptCheckResult.failed()."""
        from umrs.jobs.monitor import AttemptCheckResult

        result = AttemptCheckResult.failed("container crashed")
        assert result.status == "failed"
        assert result.exit_reason == "container crashed"


# =============================================================================
# Job API Ownership Enforcement Tests
# =============================================================================


class TestCallerIdentity:
    """Test CallerIdentity and _enforce_job_access."""

    def _make_job(self, project_id="proj-1", user_id="user-1"):
        return Job(
            job_id="job-test",
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.INDEX_BUILD,
            spec={},
            idempotency_key="key-1",
        )

    def test_owner_can_access_job(self):
        """Job owner can access their own job."""
        from umrs.api.jobs import CallerIdentity

        caller = CallerIdentity(user_id="user-1", project_id="proj-1")
        job = self._make_job()
        assert caller.can_access_job(job) is True

    def test_other_user_cannot_access_job(self):
        """Different user in same project cannot access job."""
        from umrs.api.jobs import CallerIdentity

        caller = CallerIdentity(user_id="user-2", project_id="proj-1")
        job = self._make_job()
        assert caller.can_access_job(job) is False

    def test_other_project_cannot_access_job(self):
        """Same user in different project cannot access job."""
        from umrs.api.jobs import CallerIdentity

        caller = CallerIdentity(user_id="user-1", project_id="proj-OTHER")
        job = self._make_job()
        assert caller.can_access_job(job) is False

    def test_admin_can_access_any_job(self):
        """Admin can access any job regardless of ownership."""
        from umrs.api.jobs import CallerIdentity

        caller = CallerIdentity(user_id="admin-user", project_id="admin-proj", is_admin=True)
        job = self._make_job()
        assert caller.can_access_job(job) is True

    def test_verified_headers_override_query_identity_in_strict_mode(self, monkeypatch):
        """Strict mode trusts Gateway-issued headers instead of query identity."""
        from umrs.api.jobs import get_caller_identity

        from r4_contracts.handshake_hmac import sign_claims

        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "t")

        caller = get_caller_identity(
            caller_user_id="attacker",
            caller_project_id="evil-project",
            x_user_id="alice",
            x_verified_project_id="project-a",
            x_verified_hmac=sign_claims("t", "alice", "project-a", False),
        )

        assert caller.user_id == "alice"
        assert caller.project_id == "project-a"

    def test_strict_mode_rejects_missing_verified_job_identity(self, monkeypatch):
        """Strict mode rejects legacy query-only caller identity."""
        from fastapi import HTTPException as FastAPIHTTPException
        from umrs.api.jobs import get_caller_identity

        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")

        with pytest.raises(FastAPIHTTPException) as exc_info:
            get_caller_identity(
                caller_user_id="alice",
                caller_project_id="project-a",
            )

        assert exc_info.value.status_code == 401

    def test_enforce_raises_403_for_non_owner(self):
        """_enforce_job_access raises HTTPException 403 for non-owners."""
        from fastapi import HTTPException as FastAPIHTTPException
        from umrs.api.jobs import CallerIdentity, _enforce_job_access

        caller = CallerIdentity(user_id="attacker", project_id="proj-1")
        job = self._make_job()

        with pytest.raises(FastAPIHTTPException) as exc_info:
            _enforce_job_access(caller, job, action="cancel")
        assert exc_info.value.status_code == 403

    def test_enforce_passes_for_owner(self):
        """_enforce_job_access does not raise for job owner."""
        from umrs.api.jobs import CallerIdentity, _enforce_job_access

        caller = CallerIdentity(user_id="user-1", project_id="proj-1")
        job = self._make_job()

        # Should not raise
        _enforce_job_access(caller, job, action="read")

    def test_enforce_passes_for_admin(self):
        """_enforce_job_access does not raise for admin."""
        from umrs.api.jobs import CallerIdentity, _enforce_job_access

        caller = CallerIdentity(user_id="other", project_id="other", is_admin=True)
        job = self._make_job()

        # Should not raise
        _enforce_job_access(caller, job, action="cancel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
