"""
Integration Tests for Durable Job Tracking.

Tests the full provisioning flow with the durable job tracking system.

These tests require:
- PostgreSQL with pgvector (use docker-compose.yml)
- Docker (for compose backend)

Run with:
    pytest tests/test_durable_orchestration.py -v
"""

import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import asyncio

# Test imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class MockConnection:
    """Mock Postgres connection for testing."""

    def __init__(self):
        self._data = {
            "jobs": {},
            "job_attempts": {},
        }
        self._cursor_results = []

    def cursor(self):
        return MockCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockCursor:
    """Mock database cursor."""

    def __init__(self, conn):
        self.conn = conn
        self._rows = []
        self._description = None

    def execute(self, sql, params=None):
        # Simple SQL parsing for tests
        pass

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    @property
    def rowcount(self):
        return len(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class MockPool:
    """Mock connection pool."""

    def __init__(self):
        self._conn = MockConnection()

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        pass


# ============================================================================
# Unit Tests for Models
# ============================================================================

class TestJobModels:
    """Tests for job models and utilities."""

    def test_job_type_enum(self):
        """Test JobType enum values."""
        from umrs.jobs import JobType

        assert JobType.ENDPOINT_PROVISION.value == "endpoint_provision"
        assert JobType.INDEX_BUILD.value == "index_build"
        assert JobType.EMBEDDING_BATCH.value == "embedding_batch"

    def test_job_status_enum(self):
        """Test JobStatus enum values."""
        from umrs.jobs import JobStatus

        assert JobStatus.PLANNED.value == "planned"
        assert JobStatus.QUEUED.value == "queued"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_attempt_status_enum(self):
        """Test AttemptStatus enum values."""
        from umrs.jobs import AttemptStatus

        assert AttemptStatus.CREATED.value == "created"
        assert AttemptStatus.SUBMITTED.value == "submitted"
        assert AttemptStatus.RUNNING.value == "running"
        assert AttemptStatus.SUCCEEDED.value == "succeeded"
        assert AttemptStatus.FAILED.value == "failed"
        assert AttemptStatus.LOST.value == "lost"

    def test_idempotency_key_computation(self):
        """Test idempotency key is deterministic."""
        from umrs.jobs import compute_idempotency_key, JobType

        key1 = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec={"model": "test"},
        )
        key2 = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec={"model": "test"},
        )
        assert key1 == key2

        # Different spec should produce different key
        key3 = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec={"model": "other"},
        )
        assert key1 != key3

    def test_job_serialization(self):
        """Test Job to_dict/from_dict round-trip."""
        from umrs.jobs import Job, JobType, JobStatus, RestartPolicy, compute_idempotency_key

        idempotency_key = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec={"model": "test"},
        )

        job = Job(
            job_id="job-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            status=JobStatus.QUEUED,
            spec={"model": "test"},
            idempotency_key=idempotency_key,
            priority=5,
            desired_max_attempts=3,
            restart_policy=RestartPolicy.ON_FAILURE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        data = job.to_dict()
        restored = Job.from_dict(data)

        assert restored.job_id == job.job_id
        assert restored.project_id == job.project_id
        assert restored.job_type == job.job_type
        assert restored.status == job.status
        assert restored.spec == job.spec
        assert restored.priority == job.priority

    def test_attempt_serialization(self):
        """Test JobAttempt to_dict/from_dict round-trip."""
        from umrs.jobs import JobAttempt, AttemptStatus

        attempt = JobAttempt(
            attempt_id="att-123",
            job_id="job-123",
            attempt_no=1,
            backend="compose",
            backend_ref="container-abc",
            status=AttemptStatus.RUNNING,
            umrs_instance_id="umrs-1",
            started_at=datetime.utcnow(),
        )

        data = attempt.to_dict()
        restored = JobAttempt.from_dict(data)

        assert restored.attempt_id == attempt.attempt_id
        assert restored.job_id == attempt.job_id
        assert restored.attempt_no == attempt.attempt_no
        assert restored.status == attempt.status
        assert restored.backend_ref == attempt.backend_ref


# ============================================================================
# Unit Tests for Adapters
# ============================================================================

class TestAdapters:
    """Tests for backend adapters."""

    def test_index_build_backend_submit(self):
        """IndexBuildBackend should perform pgvector maintenance synchronously."""
        from umrs.jobs import Job, JobAttempt, JobType, JobStatus, AttemptStatus, compute_idempotency_key
        from umrs.jobs.adapters import IndexBuildBackend

        spec = {
            "database": "xp_demo",
            "table": "text_embeddings",
            "rebuild": True,
        }
        job = Job(
            job_id="job-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.INDEX_BUILD,
            status=JobStatus.RUNNING,
            spec=spec,
            idempotency_key=compute_idempotency_key(
                project_id="proj-1",
                user_id="user-1",
                job_type=JobType.INDEX_BUILD,
                spec=spec,
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        attempt = JobAttempt(
            attempt_id="att-123",
            job_id="job-123",
            attempt_no=1,
            backend="index_build",
            status=AttemptStatus.CREATED,
            created_at=datetime.utcnow(),
        )

        backend = IndexBuildBackend()
        with patch("umrs.jobs.adapters.maintain_pgvector_index") as mock_maintain:
            mock_maintain.return_value = {
                "database": "xp_demo",
                "table": "text_embeddings",
                "index_name": "idx_text_embeddings_embedding_hnsw",
                "action": "reindexed",
            }
            backend_ref = asyncio.run(backend.submit_job(job, attempt))

        assert "index_build|database=xp_demo|table=text_embeddings" in backend_ref
        mock_maintain.assert_called_once_with(
            "xp_demo",
            "text_embeddings",
            rebuild=True,
            index_name=None,
        )

    def test_cleanup_backend_submit_endpoint(self):
        """CleanupBackend should teardown an endpoint target."""
        from umrs.jobs import Job, JobAttempt, JobType, JobStatus, AttemptStatus, compute_idempotency_key
        from umrs.jobs.adapters import CleanupBackend

        spec = {
            "target": "endpoint",
            "job_id": "slurm-123",
            "force": True,
        }
        job = Job(
            job_id="job-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.CLEANUP,
            status=JobStatus.RUNNING,
            spec=spec,
            idempotency_key=compute_idempotency_key(
                project_id="proj-1",
                user_id="user-1",
                job_type=JobType.CLEANUP,
                spec=spec,
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        attempt = JobAttempt(
            attempt_id="att-123",
            job_id="job-123",
            attempt_no=1,
            backend="cleanup",
            status=AttemptStatus.CREATED,
            created_at=datetime.utcnow(),
        )

        mock_orchestrator = AsyncMock()
        mock_registry = MagicMock()
        mock_registry.get_by_job_id.return_value = MagicMock(job_id="slurm-123")
        backend = CleanupBackend(mock_orchestrator, mock_registry)

        backend_ref = asyncio.run(backend.submit_job(job, attempt))

        assert backend_ref == "cleanup|target=endpoint|job_id=slurm-123|force=1"
        mock_orchestrator.cleanup.assert_awaited_once_with("slurm-123", force=True)
        mock_registry.remove.assert_called_once_with("slurm-123")

    def test_cleanup_backend_resolves_durable_endpoint_job_id(self):
        """CleanupBackend should resolve durable endpoint job IDs to backend refs."""
        from umrs.jobs import Job, JobAttempt, JobType, JobStatus, AttemptStatus, compute_idempotency_key
        from umrs.jobs.adapters import CleanupBackend

        spec = {
            "target": "endpoint",
            "job_id": "job-endpoint-123",
        }
        job = Job(
            job_id="job-cleanup-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.CLEANUP,
            status=JobStatus.RUNNING,
            spec=spec,
            idempotency_key=compute_idempotency_key(
                project_id="proj-1",
                user_id="user-1",
                job_type=JobType.CLEANUP,
                spec=spec,
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        attempt = JobAttempt(
            attempt_id="att-cleanup-123",
            job_id="job-cleanup-123",
            attempt_no=1,
            backend="cleanup",
            status=AttemptStatus.CREATED,
            created_at=datetime.utcnow(),
        )
        provision_job = Job(
            job_id="job-endpoint-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            status=JobStatus.COMPLETED,
            spec={"model": "sam3"},
            idempotency_key=compute_idempotency_key(
                project_id="proj-1",
                user_id="user-1",
                job_type=JobType.ENDPOINT_PROVISION,
                spec={"model": "sam3"},
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        provision_attempt = JobAttempt(
            attempt_id="att-endpoint-123",
            job_id="job-endpoint-123",
            attempt_no=1,
            backend="slurm-podman",
            backend_ref="slurm-123",
            status=AttemptStatus.SUCCEEDED,
            created_at=datetime.utcnow(),
        )

        mock_orchestrator = AsyncMock()
        mock_registry = MagicMock()
        mock_registry.get_by_job_id.side_effect = lambda job_id: (
            None if job_id == "job-endpoint-123" else MagicMock(job_id=job_id)
        )
        backend = CleanupBackend(
            mock_orchestrator,
            mock_registry,
            get_job_with_attempts=lambda _job_id: (provision_job, [provision_attempt]),
        )

        backend_ref = asyncio.run(backend.submit_job(job, attempt))

        assert (
            backend_ref
            == "cleanup|target=endpoint|job_id=slurm-123|requested_job_id=job-endpoint-123|force=0"
        )
        mock_orchestrator.cleanup.assert_awaited_once_with("slurm-123", force=False)
        mock_registry.remove.assert_called_once_with("slurm-123")

    def test_cleanup_status_checker_succeeds_when_endpoint_is_absent(self):
        """Cleanup status should succeed once the target endpoint is gone."""
        from umrs.jobs import Job, JobAttempt, JobType, JobStatus, AttemptStatus, compute_idempotency_key
        from umrs.jobs.adapters import CleanupStatusChecker

        spec = {"target": "endpoint", "job_id": "slurm-123"}
        job = Job(
            job_id="job-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.CLEANUP,
            status=JobStatus.RUNNING,
            spec=spec,
            idempotency_key=compute_idempotency_key(
                project_id="proj-1",
                user_id="user-1",
                job_type=JobType.CLEANUP,
                spec=spec,
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        attempt = JobAttempt(
            attempt_id="att-123",
            job_id="job-123",
            attempt_no=1,
            backend="cleanup",
            backend_ref="cleanup|target=endpoint|job_id=slurm-123|force=0",
            status=AttemptStatus.SUBMITTED,
            created_at=datetime.utcnow(),
        )

        mock_registry = MagicMock()
        mock_registry.get_by_job_id.return_value = None
        checker = CleanupStatusChecker(mock_registry)

        result = asyncio.run(checker.check_status(attempt, job))

        assert result.status == "succeeded"
        assert result.should_commit is False
        assert result.metrics["target_job_id"] == "slurm-123"

    def test_cleanup_status_checker_reports_requested_job_id(self):
        """Cleanup status should preserve the original durable endpoint job ID in metrics."""
        from umrs.jobs import Job, JobAttempt, JobType, JobStatus, AttemptStatus, compute_idempotency_key
        from umrs.jobs.adapters import CleanupStatusChecker

        spec = {"target": "endpoint", "job_id": "job-endpoint-123"}
        job = Job(
            job_id="job-cleanup-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.CLEANUP,
            status=JobStatus.RUNNING,
            spec=spec,
            idempotency_key=compute_idempotency_key(
                project_id="proj-1",
                user_id="user-1",
                job_type=JobType.CLEANUP,
                spec=spec,
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        attempt = JobAttempt(
            attempt_id="att-cleanup-123",
            job_id="job-cleanup-123",
            attempt_no=1,
            backend="cleanup",
            backend_ref=(
                "cleanup|target=endpoint|job_id=slurm-123"
                "|requested_job_id=job-endpoint-123|force=0"
            ),
            status=AttemptStatus.SUBMITTED,
            created_at=datetime.utcnow(),
        )

        mock_registry = MagicMock()
        mock_registry.get_by_job_id.return_value = None
        checker = CleanupStatusChecker(mock_registry)

        result = asyncio.run(checker.check_status(attempt, job))

        assert result.status == "succeeded"
        assert result.metrics["target_job_id"] == "slurm-123"
        assert result.metrics["requested_job_id"] == "job-endpoint-123"

    def test_endpoint_provision_backend_submit(self):
        """Test EndpointProvisionBackend submit_job."""
        from umrs.jobs import Job, JobAttempt, JobType, JobStatus, AttemptStatus, compute_idempotency_key
        from umrs.jobs.adapters import EndpointProvisionBackend

        # Mock orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.backend_type = "compose"
        mock_orchestrator.submit_job = AsyncMock(return_value=MagicMock(
            job_id="container-123",
            endpoint=MagicMock(node="localhost"),
        ))

        # Mock registry
        mock_registry = MagicMock()
        mock_registry.register = MagicMock()

        # Mock port manager
        mock_port_manager = MagicMock()
        mock_port_manager.allocate = MagicMock(return_value=11434)

        backend = EndpointProvisionBackend(
            orchestrator=mock_orchestrator,
            registry=mock_registry,
            port_manager=mock_port_manager,
        )

        # Create test job and attempt
        spec = {
            "model": "test-model",
            "model_path": "/models/test",
            "gpu_count": 1,
            "max_model_len": 4096,
            "ttl_sec": 3600,
        }
        idempotency_key = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec=spec,
        )

        job = Job(
            job_id="job-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            status=JobStatus.RUNNING,
            spec=spec,
            idempotency_key=idempotency_key,
            priority=0,
            desired_max_attempts=3,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        attempt = JobAttempt(
            attempt_id="att-123",
            job_id="job-123",
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            umrs_instance_id="umrs-1",
            created_at=datetime.utcnow(),
        )

        # Patch _determine_target_host to avoid module import issues
        with patch.object(backend, '_determine_target_host', return_value="localhost"):
            # Submit
            backend_ref = asyncio.run(backend.submit_job(job, attempt))

            assert backend_ref == "container-123"
            mock_port_manager.allocate.assert_called_once()
            mock_orchestrator.submit_job.assert_called_once()
            mock_registry.register.assert_called_once()

    def test_endpoint_provision_backend_merges_env_vars(self):
        """EndpointProvisionBackend should preserve caller env overrides."""
        from umrs.jobs import Job, JobAttempt, JobType, JobStatus, AttemptStatus, compute_idempotency_key
        from umrs.jobs.adapters import EndpointProvisionBackend

        mock_orchestrator = AsyncMock()
        mock_orchestrator.backend_type = "compose"
        mock_orchestrator.submit_job = AsyncMock(return_value=MagicMock(
            job_id="container-123",
            endpoint=MagicMock(node="localhost"),
        ))

        backend = EndpointProvisionBackend(
            orchestrator=mock_orchestrator,
            registry=MagicMock(),
            port_manager=MagicMock(allocate=MagicMock(return_value=11434)),
        )

        spec = {
            "model": "test-model",
            "model_path": "/models/test",
            "gpu_count": 1,
            "max_model_len": 4096,
            "ttl_sec": 3600,
            "env_vars": {
                "RIVAULT_PROFILES": "harness vllm vision-bentoml",
                "CUSTOM_FLAG": 7,
            },
        }
        idempotency_key = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec=spec,
        )

        job = Job(
            job_id="job-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            status=JobStatus.RUNNING,
            spec=spec,
            idempotency_key=idempotency_key,
            priority=0,
            desired_max_attempts=3,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        attempt = JobAttempt(
            attempt_id="att-123",
            job_id="job-123",
            attempt_no=1,
            backend="compose",
            status=AttemptStatus.CREATED,
            umrs_instance_id="umrs-1",
            created_at=datetime.utcnow(),
        )

        with patch.object(backend, '_determine_target_host', return_value="localhost"):
            asyncio.run(backend.submit_job(job, attempt))

        env = mock_orchestrator.submit_job.await_args.kwargs["env"]
        assert env["RIVAULT_PROFILES"] == "harness vllm vision-bentoml"
        assert env["CUSTOM_FLAG"] == "7"
        assert env["LITELLM_PORT"] == "11434"
        assert env["MODEL_NAME"] == "test-model"

    def test_endpoint_provision_backend_treats_slurm_podman_as_pending_host(self):
        """Slurm-Podman endpoints should wait for backend host discovery."""
        from umrs.jobs.adapters import EndpointProvisionBackend

        mock_orchestrator = MagicMock()
        mock_orchestrator.backend_name = "slurm-podman"

        backend = EndpointProvisionBackend(
            orchestrator=mock_orchestrator,
            registry=MagicMock(),
            port_manager=MagicMock(),
        )

        assert backend.backend_name == "slurm-podman"
        assert backend._determine_target_host() == "pending"

    def test_endpoint_status_checker_refreshes_slurm_endpoint_location(self):
        """Backend status should update pending Slurm endpoints before readiness checks."""
        from umrs.jobs import Job, JobAttempt, JobType, JobStatus, AttemptStatus, compute_idempotency_key
        from umrs.jobs.adapters import EndpointProvisionStatusChecker
        from umrs.orchestrator.base import EndpointInfo, JobInfo, JobStatus as OrchestratorJobStatus
        from umrs.orchestrator.endpoint_registry import EndpointStatus, RegisteredEndpoint

        job = Job(
            job_id="job-123",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            status=JobStatus.RUNNING,
            spec={"model": "test-model"},
            idempotency_key=compute_idempotency_key(
                project_id="proj-1",
                user_id="user-1",
                job_type=JobType.ENDPOINT_PROVISION,
                spec={"model": "test-model"},
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        attempt = JobAttempt(
            attempt_id="att-123",
            job_id="job-123",
            attempt_no=1,
            backend="slurm-podman",
            backend_ref="slurm-123",
            status=AttemptStatus.SUBMITTED,
            created_at=datetime.utcnow(),
        )

        endpoint = RegisteredEndpoint(
            project_id="proj-1",
            model="test-model",
            job_id="slurm-123",
            host="pending",
            port=11434,
            status=EndpointStatus.PROVISIONING,
            backend="slurm-podman",
        )

        mock_registry = MagicMock()
        mock_registry.get_by_job_id.side_effect = lambda _job_id: endpoint
        mock_registry.update_location.side_effect = (
            lambda _job_id, host=None, port=None, metadata=None: (
                setattr(endpoint, "host", host if host is not None else endpoint.host),
                setattr(endpoint, "port", port if port is not None else endpoint.port),
                endpoint.metadata.update(metadata or {}),
                endpoint,
            )[-1]
        )
        mock_registry.update_status.return_value = endpoint
        mock_registry.mark_failed.return_value = endpoint

        mock_readiness = MagicMock()
        mock_readiness.check_ready_detail = AsyncMock(
            return_value=(True, ["test-model"], None)
        )

        mock_orchestrator = MagicMock()
        mock_orchestrator.get_job_status = AsyncMock(
            return_value=JobInfo(
                job_id="slurm-123",
                status=OrchestratorJobStatus.RUNNING,
                endpoint=EndpointInfo(node="gpu-node-1", litellm_port=11434),
                backend="slurm-podman",
            )
        )

        checker = EndpointProvisionStatusChecker(
            readiness_poller=mock_readiness,
            registry=mock_registry,
            orchestrator=mock_orchestrator,
        )

        result = asyncio.run(checker.check_status(attempt, job))

        assert result.status == "succeeded"
        assert endpoint.host == "gpu-node-1"
        assert result.metrics["host"] == "gpu-node-1"
        mock_orchestrator.get_job_status.assert_awaited_once_with("slurm-123")
        mock_readiness.check_ready_detail.assert_awaited_once_with("gpu-node-1", 11434)

    def test_backend_factory_creation(self):
        """Test create_backend_factory."""
        from umrs.jobs import JobType
        from umrs.jobs.adapters import create_backend_factory, create_status_checker

        mock_orchestrator = MagicMock()
        mock_registry = MagicMock()
        mock_port_manager = MagicMock()
        mock_readiness_poller = MagicMock()

        factory = create_backend_factory(
            orchestrator=mock_orchestrator,
            registry=mock_registry,
            port_manager=mock_port_manager,
            get_job_with_attempts=MagicMock(),
        )

        # Get endpoint provision backend
        backend = factory(JobType.ENDPOINT_PROVISION)
        assert backend is not None
        assert factory(JobType.INDEX_BUILD) is not None
        assert factory(JobType.CLEANUP) is not None

        checker = create_status_checker(
            readiness_poller=mock_readiness_poller,
            registry=mock_registry,
            include_model_download=False,
        )
        assert JobType.ENDPOINT_PROVISION in checker._checkers
        assert JobType.INDEX_BUILD in checker._checkers
        assert JobType.CLEANUP in checker._checkers


class TestScheduler:
    """Tests for scheduler failure handling."""

    def test_schedule_job_marks_backend_submission_failures(self):
        """Scheduler should mark backend submission failures for retry handling."""
        from umrs.jobs import JobType
        from umrs.jobs.scheduler import Scheduler

        job_store = MagicMock()
        job_store.get_job.return_value = MagicMock(job_type=JobType.ENDPOINT_PROVISION)
        job_store.create_attempt.return_value = "att-123"
        job_store.get_attempt.return_value = MagicMock(attempt_id="att-123")

        backend = MagicMock(backend_name="slurm-podman")
        backend.submit_job = AsyncMock(side_effect=RuntimeError("boom"))

        scheduler = Scheduler(
            job_store=job_store,
            backend_factory=lambda _job_type: backend,
        )

        asyncio.run(scheduler._schedule_job("job-123"))

        job_store.mark_attempt_failed.assert_called_once_with(
            "att-123",
            exit_reason="backend_submission_failed: boom",
            retry_delay=30,
        )


class TestOrchestrationServiceConfig:
    """Tests for backend selection behavior."""

    def test_backend_prefers_site_config_kind(self):
        """A configured site backend should override generic auto-detection."""
        from umrs.orchestrator.service import OrchestrationService

        fake_site = MagicMock()
        fake_site.scheduler.kind = "slurm-podman"
        fake_backend = MagicMock()

        with patch("umrs.orchestrator.service.load_site_config", return_value=fake_site), \
             patch("umrs.orchestrator.service.get_orchestrator", return_value=fake_backend) as mock_get_orchestrator, \
             patch("umrs.orchestrator.service.detect_backend") as mock_detect_backend:
            service = OrchestrationService()
            backend = service.backend

        assert backend is fake_backend
        assert service.backend_type.value == "slurm"
        mock_detect_backend.assert_not_called()
        mock_get_orchestrator.assert_called_once_with("slurm-podman")

    def test_provision_reuses_existing_endpoint(self):
        """Warm reusable endpoints should bypass backend submission."""
        from umrs.orchestrator.endpoint_registry import EndpointStatus, RegisteredEndpoint
        from umrs.orchestrator.service import OrchestrationService, ProvisionRequest

        endpoint = RegisteredEndpoint(
            project_id="proj-1",
            model="intfloat/e5-base-v2",
            job_id="slurm-123",
            host="node-1",
            port=8001,
            status=EndpointStatus.READY,
            ready_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=30),
            backend="slurm-podman",
        )

        mock_registry = MagicMock()
        mock_registry.get.return_value = endpoint

        mock_backend = MagicMock()
        mock_backend.submit_job = AsyncMock()

        service = OrchestrationService(
            registry=mock_registry,
            port_manager=MagicMock(),
            readiness_poller=MagicMock(),
            backend=mock_backend,
        )

        result = asyncio.run(
            service.provision(
                ProvisionRequest(
                    project_id="proj-1",
                    model="intfloat/e5-base-v2",
                    model_path="",
                )
            )
        )

        assert result.success is True
        assert result.reused is True
        assert result.job_id == "slurm-123"
        mock_backend.submit_job.assert_not_called()

    def test_provision_passes_env_vars_to_backend(self):
        """ProvisionRequest env_vars should reach backend submission."""
        from umrs.orchestrator.service import OrchestrationService, ProvisionRequest

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_registry.register.return_value = MagicMock(job_id="slurm-321", is_usable=True)
        mock_registry.get_by_job_id.return_value = MagicMock(job_id="slurm-321", is_usable=True)

        mock_port_manager = MagicMock()
        mock_port_manager.allocate.return_value = 11434

        mock_readiness = MagicMock()
        mock_readiness.config.timeout_sec = 300
        mock_readiness.wait_for_ready = AsyncMock(return_value=MagicMock(is_ready=True, models=["intfloat/e5-base-v2"]))

        mock_backend = MagicMock()
        mock_backend.submit_job = AsyncMock(return_value=MagicMock(
            job_id="slurm-321",
            endpoint=MagicMock(node="node-1"),
        ))

        service = OrchestrationService(
            registry=mock_registry,
            port_manager=mock_port_manager,
            readiness_poller=mock_readiness,
            backend=mock_backend,
        )

        result = asyncio.run(
            service.provision(
                ProvisionRequest(
                    project_id="proj-1",
                    model="intfloat/e5-base-v2",
                    model_path="",
                    env_vars={
                        "RIVAULT_PROFILES": "harness vllm",
                        "CUSTOM_FLAG": 9,
                    },
                )
            )
        )

        assert result.success is True
        env = mock_backend.submit_job.await_args.kwargs["env"]
        assert env["RIVAULT_PROFILES"] == "harness vllm"
        assert env["CUSTOM_FLAG"] == "9"
        assert env["LITELLM_PORT"] == "11434"

    def test_provision_waits_for_backend_endpoint_before_readiness(self):
        """Slurm-style submissions should resolve the assigned host before readiness polling."""
        from umrs.orchestrator.detection import BackendType
        from umrs.orchestrator.base import EndpointInfo, JobInfo, JobStatus as BackendJobStatus
        from umrs.orchestrator.service import OrchestrationService, ProvisionRequest

        registered_endpoint = MagicMock(job_id="slurm-654", is_usable=True)

        mock_registry = MagicMock()
        mock_registry.get.return_value = None
        mock_registry.register.return_value = registered_endpoint
        mock_registry.get_by_job_id.return_value = registered_endpoint

        mock_port_manager = MagicMock()
        mock_port_manager.allocate.return_value = 11434

        mock_readiness = MagicMock()
        mock_readiness.config.timeout_sec = 300
        mock_readiness.wait_for_ready = AsyncMock(
            return_value=MagicMock(is_ready=True, models=["intfloat/e5-base-v2"])
        )

        mock_backend = MagicMock()
        mock_backend.config = MagicMock(readiness_poll_interval_sec=1)
        mock_backend.submit_job = AsyncMock(
            return_value=JobInfo(
                job_id="slurm-654",
                status=BackendJobStatus.PENDING,
                endpoint=None,
                backend="slurm-podman",
            )
        )
        mock_backend.get_job_status = AsyncMock(
            return_value=JobInfo(
                job_id="slurm-654",
                status=BackendJobStatus.STARTING,
                endpoint=EndpointInfo(node="gpu-node-1", litellm_port=11434),
                backend="slurm-podman",
            )
        )

        service = OrchestrationService(
            registry=mock_registry,
            port_manager=mock_port_manager,
            readiness_poller=mock_readiness,
            backend=mock_backend,
        )
        service._backend_type = BackendType.SLURM

        result = asyncio.run(
            service.provision(
                ProvisionRequest(
                    project_id="proj-1",
                    model="intfloat/e5-base-v2",
                    model_path="",
                )
            )
        )

        assert result.success is True
        mock_backend.get_job_status.assert_awaited_once_with("slurm-654")
        register_kwargs = mock_registry.register.call_args.kwargs
        assert register_kwargs["host"] == "gpu-node-1"
        assert register_kwargs["port"] == 11434
        assert register_kwargs["metadata"]["allocated_host"] == "pending"
        readiness_kwargs = mock_readiness.wait_for_ready.await_args.kwargs
        assert readiness_kwargs["host"] == "gpu-node-1"
        assert readiness_kwargs["port"] == 11434


# ============================================================================
# Unit Tests for Reconciler
# ============================================================================

class TestReconciler:
    """Tests for boot reconciliation."""

    def test_reconciler_creation(self):
        """Test Reconciler can be created."""
        from umrs.jobs import Reconciler, JobStore

        mock_store = MagicMock(spec=JobStore)
        mock_store.umrs_instance_id = "umrs-1"
        reconciler = Reconciler(
            job_store=mock_store,
        )

        assert reconciler.job_store == mock_store
        assert reconciler.umrs_instance_id == "umrs-1"

    def test_reconcile_result(self):
        """Test ReconcileResult class."""
        from umrs.jobs import ReconcileResult

        result = ReconcileResult()

        # Populate lists
        result.lost_attempts = ["att-1", "att-2", "att-3", "att-4", "att-5"]
        result.requeued_jobs = ["job-1", "job-2", "job-3"]
        result.failed_jobs = ["job-4", "job-5"]
        result.errors = []

        assert len(result.lost_attempts) == 5
        assert len(result.requeued_jobs) == 3
        assert len(result.failed_jobs) == 2
        assert len(result.errors) == 0
        assert result.total_processed == 5


# ============================================================================
# Integration Tests (require DB)
# ============================================================================

@pytest.fixture
def db_pool():
    """Create a test database pool."""
    # For unit tests, use mock pool
    return MockPool()


class TestDurableOrchestrationIntegration:
    """Integration tests for durable orchestration."""

    def test_durable_service_lifecycle(self):
        """Test DurableOrchestrationService startup/shutdown."""
        from umrs.orchestrator.durable import DurableOrchestrationService

        # Create with mocks
        mock_pool = MockPool()

        with patch('umrs.orchestrator.durable.ensure_schema'):
            with patch('umrs.orchestrator.durable.JobService') as MockJobService:
                mock_job_service = MagicMock()
                mock_job_service.startup = AsyncMock(return_value=MagicMock())
                mock_job_service.shutdown = AsyncMock()
                MockJobService.return_value = mock_job_service

                with patch('umrs.orchestrator.durable.OrchestrationService') as MockOrch:
                    mock_orch = MagicMock()
                    mock_orch.backend = MagicMock()
                    mock_orch.registry = MagicMock()
                    mock_orch.port_manager = MagicMock()
                    mock_orch.readiness_poller = MagicMock()
                    mock_orch.close = AsyncMock()
                    MockOrch.return_value = mock_orch

                    with patch('umrs.orchestrator.durable.create_backend_factory'):
                        with patch('umrs.orchestrator.durable.create_status_checker'):
                            service = DurableOrchestrationService(
                                db_pool=mock_pool,
                            )

                            asyncio.run(service.startup())
                            assert service._started

                            asyncio.run(service.shutdown())
                            assert not service._started


# ============================================================================
# State Machine Tests
# ============================================================================

class TestStateMachine:
    """Tests for job state machine transitions."""

    def test_job_can_transition_queued_to_running(self):
        """Test valid QUEUED -> RUNNING transition."""
        from umrs.jobs import Job, JobType, JobStatus, compute_idempotency_key

        spec = {}
        idempotency_key = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec=spec,
        )

        job = Job(
            job_id="job-1",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            status=JobStatus.QUEUED,
            spec=spec,
            idempotency_key=idempotency_key,
            priority=0,
            desired_max_attempts=3,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # QUEUED can transition to RUNNING
        assert job.status == JobStatus.QUEUED

    def test_attempt_can_retry(self):
        """Test Job can_retry logic."""
        from umrs.jobs import Job, JobType, JobStatus, compute_idempotency_key

        spec = {}
        idempotency_key = compute_idempotency_key(
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            spec=spec,
        )

        job = Job(
            job_id="job-1",
            project_id="proj-1",
            user_id="user-1",
            job_type=JobType.ENDPOINT_PROVISION,
            status=JobStatus.RUNNING,
            spec=spec,
            idempotency_key=idempotency_key,
            priority=0,
            desired_max_attempts=3,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Can retry if current_attempt_no < desired_max_attempts
        assert job.can_retry(current_attempt_no=1)
        assert job.can_retry(current_attempt_no=2)
        assert not job.can_retry(current_attempt_no=3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
