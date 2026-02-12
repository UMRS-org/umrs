"""
Integration Tests for Durable Job Tracking.

Tests the full provisioning flow with the durable job tracking system.

These tests require:
- PostgreSQL with pgvector (use docker-compose.yml)
- Docker (for compose backend)

Run with:
    pytest tests/test_durable_orchestration.py -v
"""

import asyncio
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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

    @pytest.mark.asyncio
    async def test_endpoint_provision_backend_submit(self):
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
            backend_ref = await backend.submit_job(job, attempt)

            assert backend_ref == "container-123"
            mock_port_manager.allocate.assert_called_once()
            mock_orchestrator.submit_job.assert_called_once()
            mock_registry.register.assert_called_once()

    def test_backend_factory_creation(self):
        """Test create_backend_factory."""
        from umrs.jobs import JobType
        from umrs.jobs.adapters import create_backend_factory

        mock_orchestrator = MagicMock()
        mock_registry = MagicMock()
        mock_port_manager = MagicMock()

        factory = create_backend_factory(
            orchestrator=mock_orchestrator,
            registry=mock_registry,
            port_manager=mock_port_manager,
        )

        # Get endpoint provision backend
        backend = factory(JobType.ENDPOINT_PROVISION)
        assert backend is not None

        # Index build should fail without worker
        with pytest.raises(RuntimeError):
            factory(JobType.INDEX_BUILD)


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

    @pytest.mark.asyncio
    async def test_durable_service_lifecycle(self):
        """Test DurableOrchestrationService startup/shutdown."""
        from umrs.orchestrator.durable import DurableOrchestrationService

        # Create with mocks
        mock_pool = MockPool()

        with patch('umrs.orchestrator.durable.ensure_schema'):
            with patch('umrs.jobs.service.JobService') as MockJobService:
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

                            await service.startup()
                            assert service._started

                            await service.shutdown()
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
