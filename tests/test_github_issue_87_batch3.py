"""Tests for GitHub Issue #87: BATCH 3 features.

TASK-015-004: Model Version Management
TASK-015-005: Flexible Job Orchestration (fan-out/fan-in)
"""

import pytest


class TestModelVersionManagement:
    """Test model version management (TASK-015-004)."""

    def test_version_status_enum(self):
        from umrs.model_registry.versions import VersionStatus

        assert hasattr(VersionStatus, "DRAFT")
        assert hasattr(VersionStatus, "RELEASED")
        assert hasattr(VersionStatus, "DEPRECATED")
        assert hasattr(VersionStatus, "ARCHIVED")
        assert VersionStatus.DRAFT.value == "draft"
        assert VersionStatus.RELEASED.value == "released"

    def test_model_version_to_dict(self):
        from datetime import datetime

        from umrs.model_registry.versions import ModelVersion, VersionStatus

        ver = ModelVersion(
            version_id=ModelVersion.generate_version_id(),
            model_id="model-1",
            version="1.0.0",
            storage_ref="/models/test",
            status=VersionStatus.RELEASED,
            created_at=datetime(2026, 1, 1),
        )
        d = ver.to_dict()
        assert d["version"] == "1.0.0"
        assert d["status"] == "released"

    def test_version_history_latest(self):
        from datetime import datetime

        from umrs.model_registry.versions import ModelVersion, VersionHistory, VersionStatus

        v1 = ModelVersion(
            version_id="v1",
            model_id="m1",
            version="1.0.0",
            storage_ref="/m1",
            status=VersionStatus.RELEASED,
            created_at=datetime(2026, 1, 1),
        )
        v2 = ModelVersion(
            version_id="v2",
            model_id="m1",
            version="2.0.0",
            storage_ref="/m2",
            status=VersionStatus.RELEASED,
            created_at=datetime(2026, 6, 1),
        )

        history = VersionHistory(model_id="m1", versions=[v1, v2])
        assert history.latest is v2
        assert history.latest.version == "2.0.0"

    def test_version_is_newer(self):
        from umrs.model_registry.versions import VersionComparator

        assert VersionComparator.is_newer("2.0.0", "1.0.0") is True
        assert VersionComparator.is_newer("1.1.0", "1.0.0") is True
        assert VersionComparator.is_newer("1.0.1", "1.0.0") is True
        assert VersionComparator.is_newer("1.0.0", "2.0.0") is False

    def test_version_is_newer_with_suffixes(self):
        from umrs.model_registry.versions import VersionComparator

        # 1.0.0 > 1.0.0-beta (no suffix is newer)
        assert VersionComparator.is_newer("1.0.0", "1.0.0-beta") is True
        assert VersionComparator.is_newer("1.0.0-beta", "1.0.0") is False

    def test_version_comparison_returns_zero(self):
        from umrs.model_registry.versions import VersionComparator

        assert VersionComparator.compare("1.0.0", "1.0.0") == 0
        assert VersionComparator.compare("1.0.0", "2.0.0") == -1
        assert VersionComparator.compare("2.0.0", "1.0.0") == 1

    def test_version_history_get_version(self):
        from umrs.model_registry.versions import ModelVersion, VersionHistory, VersionStatus

        v = ModelVersion(
            version_id="v1",
            model_id="m1",
            version="1.0.0",
            storage_ref="/m1",
            status=VersionStatus.RELEASED,
        )
        history = VersionHistory(model_id="m1", versions=[v])
        found = history.get_version("1.0.0")
        assert found is v

    def test_version_history_get_version_not_found(self):
        from umrs.model_registry.versions import VersionHistory

        history = VersionHistory(model_id="m1", versions=[])
        assert history.get_version("1.0.0") is None

    def test_version_history_to_dict(self):
        from datetime import datetime

        from umrs.model_registry.versions import ModelVersion, VersionHistory, VersionStatus

        v = ModelVersion(
            version_id="v1",
            model_id="m1",
            version="1.0.0",
            storage_ref="/m1",
            status=VersionStatus.RELEASED,
            created_at=datetime(2026, 1, 1),
        )
        history = VersionHistory(model_id="m1", versions=[v])
        d = history.to_dict()
        assert d["model_id"] == "m1"
        assert len(d["versions"]) == 1


class TestJobFanOut:
    """Test fan-out job orchestration (TASK-015-005)."""

    def test_job_fan_out_default_empty(self):
        from umrs.jobs.models import Job, JobType

        job = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.CLEANUP,
            spec={},
            idempotency_key="test",
        )
        assert job.fan_out_child_job_ids == []

    def test_job_fan_out_with_children(self):
        from umrs.jobs.models import Job, JobType

        job = Job(
            job_id="job-parent",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.PIPELINE_RUN,
            spec={"fan_out": True},
            idempotency_key="test",
            fan_out_child_job_ids=["job-child-1", "job-child-2"],
        )
        assert len(job.fan_out_child_job_ids) == 2
        assert "job-child-1" in job.fan_out_child_job_ids

    def test_job_to_dict_includes_fan_out(self):
        from umrs.jobs.models import Job, JobType

        job = Job(
            job_id="job-test",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.PIPELINE_RUN,
            spec={},
            idempotency_key="test",
            fan_out_child_job_ids=["child-1"],
        )
        d = job.to_dict()
        assert d["fan_out_child_job_ids"] == ["child-1"]

    def test_job_from_dict_with_fan_out(self):
        from umrs.jobs.models import Job

        data = {
            "job_id": "job-test",
            "project_id": "proj-1",
            "user_id": "user-1",
            "job_type": "pipeline_run",
            "spec": {},
            "idempotency_key": "test",
            "status": "planned",
            "priority": 0,
            "desired_max_attempts": 3,
            "restart_policy": "on_crash",
            "ttl_seconds": None,
            "fan_out_child_job_ids": ["child-1", "child-2"],
            "next_run_at": None,
            "active_attempt_id": None,
            "commit_attempt_id": None,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "last_state_change_at": None,
            "last_heartbeat_at": None,
        }
        job = Job.from_dict(data)
        assert len(job.fan_out_child_job_ids) == 2

    def test_job_from_dict_without_fan_out(self):
        from umrs.jobs.models import Job

        data = {
            "job_id": "job-test",
            "project_id": "proj-1",
            "user_id": "user-1",
            "job_type": "cleanup",
            "spec": {},
            "idempotency_key": "test",
            "status": "planned",
            "priority": 0,
            "desired_max_attempts": 3,
            "restart_policy": "on_crash",
            "ttl_seconds": None,
            "next_run_at": None,
            "active_attempt_id": None,
            "commit_attempt_id": None,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "last_state_change_at": None,
            "last_heartbeat_at": None,
        }
        job = Job.from_dict(data)
        assert job.fan_out_child_job_ids == []

    def test_fan_out_cycle_is_rejected_before_database_access(self):
        from unittest.mock import MagicMock

        from umrs.jobs.models import JobType
        from umrs.jobs.store import JobStore

        store = JobStore.__new__(JobStore)
        store._get_conn = MagicMock()
        with pytest.raises(ValueError, match="cycle"):
            store.create_fan_out_graph(
                project_id="project",
                user_id="user",
                parent_job_type=JobType.PIPELINE_RUN,
                parent_spec={"data_source": "s3://input"},
                children=[
                    {"name": "first", "job_type": "cleanup", "depends_on": ["second"]},
                    {"name": "second", "job_type": "cleanup", "depends_on": ["first"]},
                ],
            )
        store._get_conn.assert_not_called()


class TestFanOutSchemaVersion:
    """Verify schema version was bumped."""

    def test_schema_version_bumped(self):
        from umrs.jobs.schema import SCHEMA_VERSION

        assert SCHEMA_VERSION >= 4

    def test_schema_has_fan_out_column(self):
        from umrs.jobs.schema import CREATE_JOBS_TABLE

        assert "fan_out_child_job_ids" in CREATE_JOBS_TABLE
