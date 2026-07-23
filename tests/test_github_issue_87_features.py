"""Tests for Job TTL and download notification features (GitHub Issue #87)."""

import pytest


class TestJobTTL:
    """Test Job TTL field in Job model."""

    def test_job_ttl_default_none(self):
        from umrs.jobs.models import Job, JobType

        job = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.CLEANUP,
            spec={},
            idempotency_key="test-key",
        )
        assert job.ttl_seconds is None

    def test_job_ttl_positive(self):
        from umrs.jobs.models import Job, JobType

        job = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec={},
            idempotency_key="test-key",
            ttl_seconds=300,
        )
        assert job.ttl_seconds == 300

    def test_job_ttl_partition_limit(self):
        from umrs.jobs.models import Job, JobType

        job = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec={},
            idempotency_key="test-key",
            ttl_seconds=-1,
        )
        assert job.ttl_seconds == -1

    def test_job_to_dict_includes_ttl(self):
        from umrs.jobs.models import Job, JobType

        job = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.CLEANUP,
            spec={"model": "test"},
            idempotency_key="test-key",
            ttl_seconds=120,
        )
        d = job.to_dict()
        assert d["ttl_seconds"] == 120

    def test_job_from_dict_with_ttl(self):
        from umrs.jobs.models import Job

        data = {
            "job_id": "job-test",
            "project_id": "proj-1",
            "user_id": "user-1",
            "job_type": "cleanup",
            "spec": {},
            "idempotency_key": "test-key",
            "status": "planned",
            "priority": 0,
            "desired_max_attempts": 3,
            "restart_policy": "on_crash",
            "ttl_seconds": 60,
            "next_run_at": None,
            "active_attempt_id": None,
            "commit_attempt_id": None,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "last_state_change_at": None,
            "last_heartbeat_at": None,
        }
        job = Job.from_dict(data)
        assert job.ttl_seconds == 60

    def test_job_from_dict_without_ttl(self):
        from umrs.jobs.models import Job

        data = {
            "job_id": "job-test",
            "project_id": "proj-1",
            "user_id": "user-1",
            "job_type": "cleanup",
            "spec": {},
            "idempotency_key": "test-key",
            "status": "planned",
            "priority": 0,
            "desired_max_attempts": 3,
            "restart_policy": "on_crash",
            "next_run_at": None,
            "active_attempt_id": None,
            "commit_attempt_id": None,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "last_state_change_at": None,
            "last_heartbeat_at": None,
        }
        job = Job.from_dict(data)
        assert job.ttl_seconds is None


class TestJobTTLSchemaVersion:
    """Verify schema version was bumped."""

    def test_schema_version_bumped(self):
        from umrs.jobs.schema import SCHEMA_VERSION

        assert SCHEMA_VERSION >= 3

    def test_schema_has_ttl_column(self):
        from umrs.jobs.schema import CREATE_JOBS_TABLE, SCHEMA_MIGRATIONS

        assert "ttl_seconds" in CREATE_JOBS_TABLE
        assert "ttl_seconds" in SCHEMA_MIGRATIONS[3]
        assert "fan_out_child_job_ids" in SCHEMA_MIGRATIONS[4]
        assert "# NULL" not in CREATE_JOBS_TABLE


class TestJobSpecTTLField:
    """Test JobSpec and JobCreateRequest TTL fields (integration test).

    Note: These tests require r4_contracts to be installed.
    Skip if not available.
    """

    @pytest.mark.skipif(
        True,  # Temporarily skip — requires full PYTHONPATH setup
        reason="Requires r4_contracts import path setup",
    )
    def test_job_create_request_ttl_fields(self):
        """Test JobCreateRequest has ttl_seconds field."""
        from umrs.api.jobs import JobCreateRequest, JobSpec

        req = JobCreateRequest(
            project_id="p",
            user_id="u",
            job_type="cleanup",
            spec=JobSpec(),
            ttl_seconds=300,
        )
        assert req.ttl_seconds == 300


class TestDownloadNotification:
    """Test download notification and blocking mode."""

    def test_check_download_ready_succeeded(self):
        from umrs.shared.warnings import check_download_ready

        result = check_download_ready("job-1", "succeeded", blocking=False)
        assert result["ready"] is True

    def test_check_download_ready_completed(self):
        from umrs.shared.warnings import check_download_ready

        result = check_download_ready("job-1", "completed", blocking=False)
        assert result["ready"] is True

    def test_check_download_ready_running(self):
        from umrs.shared.warnings import check_download_ready

        result = check_download_ready("job-1", "running", blocking=False)
        assert result["ready"] is False
        assert "error" in result

    def test_check_download_ready_queued(self):
        from umrs.shared.warnings import check_download_ready

        result = check_download_ready("job-1", "queued", blocking=False)
        assert result["ready"] is False

    def test_check_download_ready_blocking_mode(self):
        from umrs.shared.warnings import check_download_ready

        result = check_download_ready("job-1", "running", blocking=True)
        assert result["ready"] is False
        assert result.get("blocking") is True


def test_dashboard_static_ui_and_health_route():
    from fastapi.testclient import TestClient
    from umrs.server.http_sidecar import _create_app

    app = _create_app()
    with TestClient(app) as client:
        page = client.get("/umrs/web/")
        health = client.get("/umrs/web/dashboard/health")
    assert page.status_code == 200
    assert "UMRS Operations" in page.text
    assert health.status_code == 200
    assert health.json()["status"] in {"healthy", "degraded", "unhealthy"}
