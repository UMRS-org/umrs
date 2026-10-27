"""
Unit tests for Model Download Backend (U-821 - U-825).

Tests:
- DownloadProgress dataclass
- DownloadTaskContext dataclass
- DownloadTaskRegistry
- ModelDownloadBackend job submission
- ModelDownloadStatusChecker
- HuggingFace download (mocked)
- S3 download (mocked)
- HTTP download (mocked)
- Auto-registration on completion
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from umrs.jobs.model_download import (
    DownloadProgress,
    DownloadSource,
    DownloadTaskContext,
    DownloadTaskRegistry,
    ModelDownloadBackend,
    ModelDownloadStatusChecker,
    get_download_registry,
)
from umrs.jobs.models import (
    AttemptStatus,
    Job,
    JobAttempt,
    JobStatus,
    JobType,
    generate_attempt_id,
    generate_job_id,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_job() -> Job:
    """Create a sample model download job."""
    return Job(
        job_id=generate_job_id(),
        project_id="proj-test",
        user_id="user-test",
        job_type=JobType.MODEL_DOWNLOAD,
        spec={
            "source": "huggingface",
            "source_id": "microsoft/DialoGPT-small",
            "target_path": "/shared/models/system/dialogpt-small",
            "model_name": "dialogpt-small",
            "scope": "system",
        },
        idempotency_key="test-key-123",
        status=JobStatus.QUEUED,
    )


@pytest.fixture
def sample_attempt(sample_job: Job) -> JobAttempt:
    """Create a sample job attempt."""
    return JobAttempt(
        attempt_id=generate_attempt_id(),
        job_id=sample_job.job_id,
        attempt_no=1,
        backend="model_download",
        status=AttemptStatus.CREATED,
    )


@pytest.fixture
def mock_model_store():
    """Create a mock model store."""
    store = MagicMock()
    store.create_model = MagicMock(return_value=None)
    return store


@pytest.fixture
def mock_path_validator():
    """Create a mock path validator that allows all paths."""
    validator = MagicMock()
    validator.is_allowed = MagicMock(return_value=True)
    validator.allowed_patterns = ["/shared/models/*", "/models/*"]
    return validator


@pytest.fixture
def download_registry():
    """Create a fresh download registry."""
    return DownloadTaskRegistry()


# =============================================================================
# Test DownloadProgress
# =============================================================================


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        progress = DownloadProgress()
        assert progress.total_files == 0
        assert progress.downloaded_files == 0
        assert progress.total_bytes == 0
        assert progress.downloaded_bytes == 0
        assert progress.status == "pending"
        assert progress.error_message is None

    def test_progress_percent_zero_bytes(self):
        """Test progress percentage when total is zero."""
        progress = DownloadProgress()
        assert progress.progress_percent == 0.0

    def test_progress_percent_calculation(self):
        """Test progress percentage calculation."""
        progress = DownloadProgress(
            total_bytes=1000,
            downloaded_bytes=500,
        )
        assert progress.progress_percent == 50.0

    def test_progress_percent_cap_at_100(self):
        """Test progress doesn't exceed 100%."""
        progress = DownloadProgress(
            total_bytes=100,
            downloaded_bytes=150,  # Over-reported
        )
        assert progress.progress_percent == 100.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(timezone.utc)
        progress = DownloadProgress(
            total_files=10,
            downloaded_files=5,
            total_bytes=1000,
            downloaded_bytes=500,
            current_file="model.bin",
            started_at=now,
            updated_at=now,
            status="downloading",
        )
        data = progress.to_dict()
        assert data["total_files"] == 10
        assert data["downloaded_files"] == 5
        assert data["progress_percent"] == 50.0
        assert data["status"] == "downloading"
        assert data["started_at"] == now.isoformat()

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "total_files": 10,
            "downloaded_files": 5,
            "total_bytes": 1000,
            "downloaded_bytes": 500,
            "current_file": "model.bin",
            "started_at": now.isoformat(),
            "status": "downloading",
        }
        progress = DownloadProgress.from_dict(data)
        assert progress.total_files == 10
        assert progress.downloaded_files == 5
        assert progress.status == "downloading"


# =============================================================================
# Test DownloadTaskContext
# =============================================================================


class TestDownloadTaskContext:
    """Tests for DownloadTaskContext dataclass."""

    def test_creation(self):
        """Test task context creation."""
        ctx = DownloadTaskContext(
            job_id="job-123",
            attempt_id="att-456",
            source=DownloadSource.HUGGINGFACE,
            source_id="meta-llama/Llama-3.2-8B",
            target_path="/models/llama",
            model_name="llama-8b",
            scope="user",
            owner_id="user-test",
        )
        assert ctx.source == DownloadSource.HUGGINGFACE
        assert ctx.source_id == "meta-llama/Llama-3.2-8B"
        assert ctx.hf_token is None

    def test_to_dict_no_token(self):
        """Test serialization doesn't include actual token."""
        ctx = DownloadTaskContext(
            job_id="job-123",
            attempt_id="att-456",
            source=DownloadSource.HUGGINGFACE,
            source_id="meta-llama/Llama-3.2-8B",
            target_path="/models/llama",
            model_name="llama-8b",
            scope="user",
            owner_id="user-test",
            hf_token="secret-token-123",
        )
        data = ctx.to_dict()
        assert "hf_token" not in data
        assert data["hf_token_set"] is True


# =============================================================================
# Test DownloadTaskRegistry
# =============================================================================


class TestDownloadTaskRegistry:
    """Tests for DownloadTaskRegistry."""

    @pytest.mark.asyncio
    async def test_register_and_get(self, download_registry):
        """Test registering and retrieving tasks."""
        ctx = DownloadTaskContext(
            job_id="job-1",
            attempt_id="att-1",
            source=DownloadSource.HUGGINGFACE,
            source_id="test/model",
            target_path="/models/test",
            model_name="test-model",
            scope="user",
            owner_id="user-1",
        )

        await download_registry.register(ctx)
        retrieved = await download_registry.get("att-1")

        assert retrieved is not None
        assert retrieved.job_id == "job-1"
        assert retrieved.source_id == "test/model"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, download_registry):
        """Test getting a nonexistent task."""
        result = await download_registry.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_progress(self, download_registry):
        """Test updating progress for a task."""
        ctx = DownloadTaskContext(
            job_id="job-1",
            attempt_id="att-1",
            source=DownloadSource.HUGGINGFACE,
            source_id="test/model",
            target_path="/models/test",
            model_name="test-model",
            scope="user",
            owner_id="user-1",
        )
        await download_registry.register(ctx)

        new_progress = DownloadProgress(
            status="downloading",
            total_bytes=1000,
            downloaded_bytes=500,
        )
        await download_registry.update_progress("att-1", new_progress)

        retrieved = await download_registry.get("att-1")
        assert retrieved.progress.downloaded_bytes == 500

    @pytest.mark.asyncio
    async def test_remove(self, download_registry):
        """Test removing a task."""
        ctx = DownloadTaskContext(
            job_id="job-1",
            attempt_id="att-1",
            source=DownloadSource.HUGGINGFACE,
            source_id="test/model",
            target_path="/models/test",
            model_name="test-model",
            scope="user",
            owner_id="user-1",
        )
        await download_registry.register(ctx)

        removed = await download_registry.remove("att-1")
        assert removed is not None
        assert removed.job_id == "job-1"

        # Should be gone now
        result = await download_registry.get("att-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_tasks(self, download_registry):
        """Test listing all tasks."""
        for i in range(3):
            ctx = DownloadTaskContext(
                job_id=f"job-{i}",
                attempt_id=f"att-{i}",
                source=DownloadSource.HUGGINGFACE,
                source_id=f"test/model-{i}",
                target_path=f"/models/test-{i}",
                model_name=f"test-model-{i}",
                scope="user",
                owner_id="user-1",
            )
            await download_registry.register(ctx)

        tasks = await download_registry.list_tasks()
        assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_submit_job_validates_path(
        self,
        mock_model_store,
        download_registry,
        sample_job,
        sample_attempt,
    ):
        """Test that submit_job validates the target path."""
        validator = MagicMock()
        validator.is_allowed = MagicMock(return_value=False)
        validator.allowed_patterns = ["/allowed/*"]

        backend = ModelDownloadBackend(
            model_store=mock_model_store,
            path_validator=validator,
            download_registry=download_registry,
        )

        with pytest.raises(Exception) as exc_info:
            await backend.submit_job(sample_job, sample_attempt)

        assert "not allowed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_submit_job_requires_source_id(
        self,
        mock_model_store,
        mock_path_validator,
        download_registry,
        sample_attempt,
    ):
        """Test that source_id is required."""
        job = Job(
            job_id=generate_job_id(),
            project_id="proj-test",
            user_id="user-test",
            job_type=JobType.MODEL_DOWNLOAD,
            spec={
                "source": "huggingface",
                # Missing source_id
                "target_path": "/models/test",
            },
            idempotency_key="test-key",
        )

        backend = ModelDownloadBackend(
            model_store=mock_model_store,
            path_validator=mock_path_validator,
            download_registry=download_registry,
        )

        with pytest.raises(ValueError, match="source_id is required"):
            await backend.submit_job(job, sample_attempt)

    @pytest.mark.asyncio
    async def test_submit_job_default_model_name(
        self,
        mock_model_store,
        mock_path_validator,
        download_registry,
        sample_attempt,
    ):
        """Test default model name from source_id."""
        job = Job(
            job_id=generate_job_id(),
            project_id="proj-test",
            user_id="user-test",
            job_type=JobType.MODEL_DOWNLOAD,
            spec={
                "source": "huggingface",
                "source_id": "meta-llama/Llama-3.2-8B",
                "target_path": "/models/test",
                # No model_name specified
            },
            idempotency_key="test-key",
        )

        backend = ModelDownloadBackend(
            model_store=mock_model_store,
            path_validator=mock_path_validator,
            download_registry=download_registry,
        )

        with patch.object(backend, "_execute_download", new_callable=AsyncMock):
            await backend.submit_job(job, sample_attempt)

        task = await download_registry.get(sample_attempt.attempt_id)
        assert task.model_name == "Llama-3.2-8B"  # Last part of source_id


# =============================================================================
# Test ModelDownloadStatusChecker
# =============================================================================


class TestModelDownloadStatusChecker:
    """Tests for ModelDownloadStatusChecker."""

    @pytest.mark.asyncio
    async def test_check_attempt_not_found(self, download_registry, sample_job, sample_attempt):
        """Test checking status when task not found."""
        checker = ModelDownloadStatusChecker(download_registry=download_registry)

        result = await checker.check_attempt(sample_attempt, sample_job)

        assert result.status == "failed"
        assert "not found" in result.exit_reason.lower()

    @pytest.mark.asyncio
    async def test_check_attempt_completed(self, download_registry, sample_job, sample_attempt):
        """Test checking status when download completed."""
        # Register a completed task
        ctx = DownloadTaskContext(
            job_id=sample_job.job_id,
            attempt_id=sample_attempt.attempt_id,
            source=DownloadSource.HUGGINGFACE,
            source_id="test/model",
            target_path="/models/test",
            model_name="test-model",
            scope="user",
            owner_id="user-1",
            progress=DownloadProgress(
                status="completed",
                total_files=10,
                downloaded_files=10,
                total_bytes=1000,
                downloaded_bytes=1000,
            ),
        )
        await download_registry.register(ctx)

        checker = ModelDownloadStatusChecker(download_registry=download_registry)
        result = await checker.check_attempt(sample_attempt, sample_job)

        assert result.status == "succeeded"
        assert result.metrics["status"] == "completed"
        assert result.metrics["progress_percent"] == 100.0

    @pytest.mark.asyncio
    async def test_check_attempt_failed(self, download_registry, sample_job, sample_attempt):
        """Test checking status when download failed."""
        ctx = DownloadTaskContext(
            job_id=sample_job.job_id,
            attempt_id=sample_attempt.attempt_id,
            source=DownloadSource.HUGGINGFACE,
            source_id="test/model",
            target_path="/models/test",
            model_name="test-model",
            scope="user",
            owner_id="user-1",
            progress=DownloadProgress(
                status="failed",
                error_message="Rate limit exceeded",
            ),
        )
        await download_registry.register(ctx)

        checker = ModelDownloadStatusChecker(download_registry=download_registry)
        result = await checker.check_attempt(sample_attempt, sample_job)

        assert result.status == "failed"
        assert "Rate limit" in result.exit_reason

    @pytest.mark.asyncio
    async def test_check_attempt_running(self, download_registry, sample_job, sample_attempt):
        """Test checking status when download in progress."""
        ctx = DownloadTaskContext(
            job_id=sample_job.job_id,
            attempt_id=sample_attempt.attempt_id,
            source=DownloadSource.HUGGINGFACE,
            source_id="test/model",
            target_path="/models/test",
            model_name="test-model",
            scope="user",
            owner_id="user-1",
            progress=DownloadProgress(
                status="downloading",
                total_bytes=1000,
                downloaded_bytes=500,
            ),
        )
        await download_registry.register(ctx)

        checker = ModelDownloadStatusChecker(download_registry=download_registry)
        result = await checker.check_attempt(sample_attempt, sample_job)

        assert result.status == "running"
        assert result.metrics["progress_percent"] == 50.0

    def test_job_type(self, download_registry):
        """Test that checker handles MODEL_DOWNLOAD job type."""
        checker = ModelDownloadStatusChecker(download_registry=download_registry)
        assert checker.job_type == JobType.MODEL_DOWNLOAD


# =============================================================================
# Test DownloadSource Enum
# =============================================================================


class TestDownloadSource:
    """Tests for DownloadSource enum."""

    def test_all_sources(self):
        """Test all download sources exist."""
        assert DownloadSource.HUGGINGFACE.value == "huggingface"
        assert DownloadSource.S3.value == "s3"
        assert DownloadSource.HTTP.value == "http"
        assert DownloadSource.LOCAL.value == "local"

    def test_from_string(self):
        """Test creating from string value."""
        assert DownloadSource("huggingface") == DownloadSource.HUGGINGFACE
        assert DownloadSource("s3") == DownloadSource.S3


# =============================================================================
# Test Global Registry
# =============================================================================


class TestGlobalRegistry:
    """Tests for global registry singleton."""

    def test_get_same_instance(self):
        """Test that get_download_registry returns same instance."""
        reg1 = get_download_registry()
        reg2 = get_download_registry()
        assert reg1 is reg2


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_model_download_backend(self, mock_model_store, mock_path_validator):
        """Test create_model_download_backend factory."""
        from umrs.jobs.model_download import create_model_download_backend

        backend = create_model_download_backend(
            model_store=mock_model_store,
            path_validator=mock_path_validator,
        )

        assert isinstance(backend, ModelDownloadBackend)
        assert backend.model_store is mock_model_store

    def test_create_model_download_status_checker(self):
        """Test create_model_download_status_checker factory."""
        from umrs.jobs.model_download import create_model_download_status_checker

        checker = create_model_download_status_checker()
        assert isinstance(checker, ModelDownloadStatusChecker)


# =============================================================================
# Integration with adapters.py
# =============================================================================


class TestAdaptersIntegration:
    """Tests for integration with adapters.py factory."""

    def test_factory_includes_model_download(
        self,
        mock_model_store,
        mock_path_validator,
    ):
        """Test that create_backend_factory includes MODEL_DOWNLOAD."""
        from umrs.jobs.adapters import create_backend_factory

        # Create minimal mocks for other backends
        mock_orchestrator = MagicMock()
        mock_registry = MagicMock()
        mock_port_manager = MagicMock()

        factory = create_backend_factory(
            orchestrator=mock_orchestrator,
            registry=mock_registry,
            port_manager=mock_port_manager,
            model_store=mock_model_store,
            path_validator=mock_path_validator,
        )

        # Should be able to get MODEL_DOWNLOAD backend
        backend = factory(JobType.MODEL_DOWNLOAD)
        assert isinstance(backend, ModelDownloadBackend)

    def test_factory_raises_without_model_store(self):
        """Test that factory raises when model_store not provided."""
        from umrs.jobs.adapters import create_backend_factory

        mock_orchestrator = MagicMock()
        mock_registry = MagicMock()
        mock_port_manager = MagicMock()

        factory = create_backend_factory(
            orchestrator=mock_orchestrator,
            registry=mock_registry,
            port_manager=mock_port_manager,
            # No model_store or path_validator
        )

        with pytest.raises(RuntimeError, match="not configured"):
            factory(JobType.MODEL_DOWNLOAD)


# =============================================================================
# Staging rollback on failed/cancelled downloads (M6B)
# =============================================================================


class TestStagedModelRollback:
    """Failed/cancelled downloads must not leave permanent staging rows."""

    def _ctx(self, model_id="model-1") -> DownloadTaskContext:
        return DownloadTaskContext(
            job_id="job-rb",
            attempt_id="att-rb",
            source=DownloadSource.HUGGINGFACE,
            source_id="Qwen/Qwen-AgentWorld-35B-A3B",
            target_path="/models/Qwen-AgentWorld-35B-A3B",
            model_name="Qwen-AgentWorld-35B-A3B",
            scope="system",
            owner_id=None,
            metadata={"model_id": model_id},
        )

    def _staged_entry(self):
        from umrs.model_registry.models import ModelStatus

        entry = MagicMock()
        entry.model_id = "model-1"
        entry.name = "Qwen-AgentWorld-35B-A3B"
        entry.status = ModelStatus.STAGING
        return entry

    def test_failed_download_deletes_staged_entry(self, mock_model_store, mock_path_validator):
        backend = ModelDownloadBackend(
            model_store=mock_model_store,
            path_validator=mock_path_validator,
            download_registry=DownloadTaskRegistry(),
        )
        ctx = self._ctx()
        ctx.progress.status = "failed"
        mock_model_store.get_model.return_value = self._staged_entry()

        backend._rollback_staged_model(ctx)

        mock_model_store.delete_model.assert_called_once_with("model-1")

    def test_available_entry_is_not_rolled_back(self, mock_model_store, mock_path_validator):
        from umrs.model_registry.models import ModelStatus

        backend = ModelDownloadBackend(
            model_store=mock_model_store,
            path_validator=mock_path_validator,
            download_registry=DownloadTaskRegistry(),
        )
        ctx = self._ctx()
        ctx.progress.status = "interrupted"
        entry = self._staged_entry()
        entry.status = ModelStatus.AVAILABLE
        mock_model_store.get_model.return_value = entry

        backend._rollback_staged_model(ctx)

        mock_model_store.delete_model.assert_not_called()

    def test_fallback_lookup_by_name_and_scope(self, mock_model_store, mock_path_validator):
        backend = ModelDownloadBackend(
            model_store=mock_model_store,
            path_validator=mock_path_validator,
            download_registry=DownloadTaskRegistry(),
        )
        ctx = self._ctx(model_id=None)
        ctx.metadata = {}
        ctx.progress.status = "failed"
        mock_model_store.get_model_by_name.return_value = self._staged_entry()

        backend._rollback_staged_model(ctx)

        mock_model_store.delete_model.assert_called_once_with("model-1")

    def test_store_errors_are_swallowed(self, mock_model_store, mock_path_validator):
        backend = ModelDownloadBackend(
            model_store=mock_model_store,
            path_validator=mock_path_validator,
            download_registry=DownloadTaskRegistry(),
        )
        ctx = self._ctx()
        ctx.progress.status = "failed"
        mock_model_store.get_model.side_effect = RuntimeError("db down")

        backend._rollback_staged_model(ctx)  # must not raise

    @pytest.mark.asyncio
    async def test_execute_download_rolls_back_on_failure(
        self, mock_model_store, mock_path_validator, sample_job, sample_attempt, monkeypatch
    ):
        # Hermetic: the venue check reads the HOST's free space for the target
        # (here "/"); on the CI runner (data0, root fs under the 20 GiB floor,
        # 2026-10-27) it refused before the mocked download ever ran.
        monkeypatch.setenv("UMRS_DOWNLOAD_MIN_FREE_BYTES", "0")
        backend = ModelDownloadBackend(
            model_store=mock_model_store,
            path_validator=mock_path_validator,
            download_registry=DownloadTaskRegistry(),
        )
        ctx = backend._create_download_context(sample_job, sample_attempt)
        mock_model_store.get_model_by_name.return_value = self._staged_entry()
        failed = DownloadProgress(status="failed", error_message="writer closed")

        with patch(
            "umrs.jobs.model_download.download_from_huggingface",
            new=AsyncMock(return_value=failed),
        ):
            await backend._execute_download(ctx)

        mock_model_store.delete_model.assert_called_once_with("model-1")

    @pytest.mark.asyncio
    async def test_a_refused_venue_rolls_back_the_staged_entry(
        self, mock_model_store, mock_path_validator, sample_job, sample_attempt
    ):
        """The venue refusal (overlay target, free space under the floor) is a
        failed download: it must not leave a permanent staging row either."""
        from umrs.jobs.download_safety import DownloadVenueError

        backend = ModelDownloadBackend(
            model_store=mock_model_store,
            path_validator=mock_path_validator,
            download_registry=DownloadTaskRegistry(),
        )
        ctx = backend._create_download_context(sample_job, sample_attempt)
        mock_model_store.get_model_by_name.return_value = self._staged_entry()

        with patch(
            "umrs.jobs.download_safety.assert_download_safe",
            side_effect=DownloadVenueError("only 18 GiB free; floor 20 GiB"),
        ):
            await backend._execute_download(ctx)

        assert ctx.progress.status == "failed"
        mock_model_store.delete_model.assert_called_once_with("model-1")


# =============================================================================
# HF xet gating (M6B) — writer-channel fix default-off
# =============================================================================


class TestHfXetGating:
    """Plain-LFS is the default; xet is operator opt-in."""

    @pytest.mark.asyncio
    async def test_xet_disabled_by_default(self, monkeypatch):
        from umrs.jobs import model_download_sources as src

        monkeypatch.delenv("UMRS_HF_ENABLE_XET", raising=False)
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
        deps = MagicMock()
        monkeypatch.setattr(src, "_load_huggingface_deps", lambda: deps)

        await src._run_huggingface_snapshot(
            deps,
            repo_id="Qwen/x",
            target_path="/models/x",
            token=None,
            revision=None,
        )

        import os

        assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
        deps.snapshot_download.assert_called_once()

    @pytest.mark.asyncio
    async def test_xet_opt_in_keeps_xet(self, monkeypatch):
        from umrs.jobs import model_download_sources as src

        monkeypatch.setenv("UMRS_HF_ENABLE_XET", "1")
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
        deps = MagicMock()
        monkeypatch.setattr(src, "_load_huggingface_deps", lambda: deps)

        await src._run_huggingface_snapshot(
            deps,
            repo_id="Qwen/x",
            target_path="/models/x",
            token=None,
            revision=None,
        )

        import os

        assert "HF_HUB_DISABLE_XET" not in os.environ
