"""
Unit tests for UMRS Orchestrator backends (U-601).

Tests orchestrator lifecycle, job management, and backend abstraction.
"""

import importlib.util
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from umrs.orchestrator.base import (
    EndpointInfo,
    JobInfo,
    JobStatus,
    OrchestratorBackend,
    OrchestratorConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Create test orchestrator configuration."""
    return OrchestratorConfig(
        model_path="/models/test-model",
        litellm_master_key="sk-test-key",
        litellm_port=11434,
        vllm_port=8000,
        vllm_max_model_len=4096,
        startup_timeout_sec=10,
        idle_timeout_sec=60,
        readiness_poll_interval_sec=1,
    )


@pytest.fixture
def endpoint_info():
    """Create test endpoint info."""
    return EndpointInfo(
        node="test-node",
        litellm_port=11434,
        vllm_port=8000,
        model_name="rivault-gpt-oss-static",
    )


@pytest.fixture
def job_info(endpoint_info):
    """Create test job info."""
    return JobInfo(
        job_id="test-job-123",
        status=JobStatus.READY,
        endpoint=endpoint_info,
        model_path="/models/test-model",
        created_at=datetime.now(timezone.utc),
        ready_at=datetime.now(timezone.utc),
        backend="test",
    )


# =============================================================================
# Test EndpointInfo
# =============================================================================


class TestEndpointInfo:
    """Tests for EndpointInfo dataclass."""

    def test_litellm_url(self, endpoint_info):
        """Test LiteLLM URL generation."""
        assert endpoint_info.litellm_url == "http://test-node:11434/v1"

    def test_health_url(self, endpoint_info):
        """Test health URL generation."""
        assert endpoint_info.health_url == "http://test-node:11434/health"

    def test_models_url(self, endpoint_info):
        """Test models URL generation."""
        assert endpoint_info.models_url == "http://test-node:11434/v1/models"

    def test_custom_port(self):
        """Test with custom port."""
        endpoint = EndpointInfo(node="192.168.1.100", litellm_port=8080)
        assert endpoint.litellm_url == "http://192.168.1.100:8080/v1"


# =============================================================================
# Test JobInfo
# =============================================================================


class TestJobInfo:
    """Tests for JobInfo dataclass."""

    def test_is_active_ready(self, job_info):
        """Test is_active for ready job."""
        assert job_info.is_active() is True

    def test_is_active_pending(self, endpoint_info):
        """Test is_active for pending job."""
        job = JobInfo(
            job_id="test",
            status=JobStatus.PENDING,
            endpoint=endpoint_info,
        )
        assert job.is_active() is True

    def test_is_active_stopped(self, endpoint_info):
        """Test is_active for stopped job."""
        job = JobInfo(
            job_id="test",
            status=JobStatus.STOPPED,
            endpoint=endpoint_info,
        )
        assert job.is_active() is False

    def test_is_active_failed(self, endpoint_info):
        """Test is_active for failed job."""
        job = JobInfo(
            job_id="test",
            status=JobStatus.FAILED,
            endpoint=endpoint_info,
            error_message="GPU allocation failed",
        )
        assert job.is_active() is False

    def test_is_ready_true(self, job_info):
        """Test is_ready for ready job."""
        assert job_info.is_ready() is True

    def test_is_ready_running(self, endpoint_info):
        """Test is_ready for running job."""
        job = JobInfo(
            job_id="test",
            status=JobStatus.RUNNING,
            endpoint=endpoint_info,
        )
        assert job.is_ready() is True

    def test_is_ready_pending(self, endpoint_info):
        """Test is_ready for pending job."""
        job = JobInfo(
            job_id="test",
            status=JobStatus.PENDING,
            endpoint=endpoint_info,
        )
        assert job.is_ready() is False


# =============================================================================
# Test OrchestratorConfig
# =============================================================================


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OrchestratorConfig()
        assert config.litellm_port == 11434
        assert config.vllm_port == 8000
        assert config.vllm_max_model_len == 4096
        assert config.vllm_enable_prefix_caching is False
        assert config.startup_timeout_sec == 600
        assert config.idle_timeout_sec == 3600
        assert config.k8s_image == "mini-rivault:latest"

    def test_custom_values(self, config):
        """Test custom configuration values."""
        assert config.model_path == "/models/test-model"
        assert config.startup_timeout_sec == 10

    @patch.dict(
        "os.environ",
        {
            "VLLM_MODEL_PATH": "/custom/model",
            "LITELLM_PORT": "9000",
            "VLLM_MAX_MODEL_LEN": "8192",
            "VLLM_ENABLE_PREFIX_CACHING": "true",
            "SLURM_PARTITION": "a100",
        },
    )
    def test_from_env(self):
        """Test loading config from environment."""
        config = OrchestratorConfig.from_env()
        assert config.model_path == "/custom/model"
        assert config.litellm_port == 9000
        assert config.vllm_max_model_len == 8192
        assert config.vllm_enable_prefix_caching is True
        assert config.slurm_partition == "a100"

    @patch.dict(
        "os.environ",
        {
            "RIVAULT_HTTP_PROXY": "",
            "EGRESS_GUARD_PROXY_URL": "",
            "RIVAULT_NO_PROXY": "",
        },
    )
    def test_blank_egress_proxy_env_falls_back_to_guard_defaults(self):
        """Blank proxy variables should not disable mini-RiVault EgressGuard."""
        config = OrchestratorConfig.from_env()

        assert config.egress_proxy_url == "http://egress-guard:8888"
        assert "egress-guard" in config.egress_no_proxy


# =============================================================================
# Test OrchestratorBackend (Abstract)
# =============================================================================


class MockBackend(OrchestratorBackend):
    """Mock backend for testing."""

    def __init__(self, config=None):
        super().__init__(config)
        self._jobs: dict[str, JobInfo] = {}
        self._job_counter = 0

    @property
    def backend_name(self) -> str:
        return "mock"

    async def submit_job(self, model_path=None, **kwargs):
        self._job_counter += 1
        job_id = f"mock-job-{self._job_counter}"
        job = JobInfo(
            job_id=job_id,
            status=JobStatus.PENDING,
            model_path=model_path or self.config.model_path,
            created_at=datetime.now(timezone.utc),
            backend="mock",
        )
        self._jobs[job_id] = job
        return job

    async def get_job_status(self, job_id: str):
        if job_id not in self._jobs:
            return JobInfo(
                job_id=job_id,
                status=JobStatus.UNKNOWN,
                backend="mock",
            )
        return self._jobs[job_id]

    async def wait_for_ready(self, job_id: str, timeout_sec=None):
        job = self._jobs.get(job_id)
        if not job:
            raise RuntimeError(f"Job not found: {job_id}")

        # Simulate immediate readiness for testing
        job.status = JobStatus.READY
        job.endpoint = EndpointInfo(
            node="mock-node",
            litellm_port=self.config.litellm_port,
        )
        job.ready_at = datetime.now(timezone.utc)
        return job.endpoint

    async def cleanup(self, job_id: str, force: bool = False):
        if job_id in self._jobs:
            self._jobs[job_id].status = JobStatus.STOPPED
            self._jobs[job_id].stopped_at = datetime.now(timezone.utc)

    async def list_jobs(self, status_filter=None):
        jobs = list(self._jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j.status in status_filter]
        return jobs

    async def cleanup_orphans(self):
        orphan_count = 0
        for job_id, job in list(self._jobs.items()):
            if job.status == JobStatus.UNKNOWN:
                del self._jobs[job_id]
                orphan_count += 1
        return orphan_count


class TestOrchestratorBackend:
    """Tests for OrchestratorBackend interface."""

    @pytest.fixture
    def backend(self, config):
        """Create mock backend."""
        return MockBackend(config)

    @pytest.mark.asyncio
    async def test_submit_job(self, backend):
        """Test job submission."""
        job = await backend.submit_job()
        assert job.job_id == "mock-job-1"
        assert job.status == JobStatus.PENDING
        assert job.backend == "mock"

    @pytest.mark.asyncio
    async def test_submit_job_with_model_path(self, backend):
        """Test job submission with custom model path."""
        job = await backend.submit_job(model_path="/custom/model")
        assert job.model_path == "/custom/model"

    @pytest.mark.asyncio
    async def test_get_job_status(self, backend):
        """Test getting job status."""
        job = await backend.submit_job()
        status = await backend.get_job_status(job.job_id)
        assert status.job_id == job.job_id
        assert status.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_job_status_unknown(self, backend):
        """Test getting status of unknown job."""
        status = await backend.get_job_status("nonexistent-job")
        assert status.status == JobStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_wait_for_ready(self, backend):
        """Test waiting for job readiness."""
        job = await backend.submit_job()
        endpoint = await backend.wait_for_ready(job.job_id)
        assert endpoint.node == "mock-node"
        assert endpoint.litellm_port == 11434

    @pytest.mark.asyncio
    async def test_wait_for_ready_unknown_job(self, backend):
        """Test waiting for unknown job raises error."""
        with pytest.raises(RuntimeError, match="Job not found"):
            await backend.wait_for_ready("nonexistent-job")

    @pytest.mark.asyncio
    async def test_cleanup(self, backend):
        """Test job cleanup."""
        job = await backend.submit_job()
        await backend.cleanup(job.job_id)

        status = await backend.get_job_status(job.job_id)
        assert status.status == JobStatus.STOPPED
        assert status.stopped_at is not None

    @pytest.mark.asyncio
    async def test_list_jobs(self, backend):
        """Test listing jobs."""
        await backend.submit_job()
        await backend.submit_job()

        jobs = await backend.list_jobs()
        assert len(jobs) == 2

    @pytest.mark.asyncio
    async def test_list_jobs_with_filter(self, backend):
        """Test listing jobs with status filter."""
        job1 = await backend.submit_job()
        await backend.submit_job()

        await backend.wait_for_ready(job1.job_id)

        pending_jobs = await backend.list_jobs(status_filter=[JobStatus.PENDING])
        assert len(pending_jobs) == 1

        ready_jobs = await backend.list_jobs(status_filter=[JobStatus.READY])
        assert len(ready_jobs) == 1

    @pytest.mark.asyncio
    async def test_cleanup_orphans(self, backend):
        """Test cleaning up orphaned resources."""
        backend._jobs["orphan-1"] = JobInfo(
            job_id="orphan-1",
            status=JobStatus.UNKNOWN,
            backend="mock",
        )
        backend._jobs["orphan-2"] = JobInfo(
            job_id="orphan-2",
            status=JobStatus.UNKNOWN,
            backend="mock",
        )

        count = await backend.cleanup_orphans()
        assert count == 2
        assert len(backend._jobs) == 0

    @pytest.mark.asyncio
    async def test_check_readiness_success(self, backend, endpoint_info):
        """Test readiness check with successful response."""
        import aiohttp

        mock_response = MagicMock()
        mock_response.status = 200

        mock_get = AsyncMock()
        mock_get.__aenter__.return_value = mock_response
        mock_get.__aexit__.return_value = None

        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_get

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(aiohttp, "ClientSession", return_value=mock_session):
            result = await backend.check_readiness(endpoint_info)
            assert result is True

    @pytest.mark.asyncio
    async def test_check_readiness_failure(self, backend, endpoint_info):
        """Test readiness check with failed response."""
        import aiohttp

        mock_response = MagicMock()
        mock_response.status = 503

        mock_get = AsyncMock()
        mock_get.__aenter__.return_value = mock_response
        mock_get.__aexit__.return_value = None

        mock_session_instance = MagicMock()
        mock_session_instance.get.return_value = mock_get

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(aiohttp, "ClientSession", return_value=mock_session):
            result = await backend.check_readiness(endpoint_info)
            assert result is False

    @pytest.mark.asyncio
    async def test_check_readiness_exception(self, backend, endpoint_info):
        """Test readiness check with connection error."""
        import aiohttp

        mock_session_instance = MagicMock()
        mock_session_instance.get.side_effect = Exception("Connection refused")

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(aiohttp, "ClientSession", return_value=mock_session):
            result = await backend.check_readiness(endpoint_info)
            assert result is False

    @pytest.mark.asyncio
    async def test_health_check(self, backend):
        """Test backend health check."""
        health = await backend.health_check()
        assert health["backend"] == "mock"
        assert health["healthy"] is True


# =============================================================================
# Test Job Lifecycle
# =============================================================================


class TestJobLifecycle:
    """Integration tests for complete job lifecycle."""

    @pytest.fixture
    def backend(self, config):
        """Create mock backend."""
        return MockBackend(config)

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, backend):
        """Test complete job lifecycle: submit -> ready -> cleanup."""
        # Submit job
        job = await backend.submit_job()
        assert job.status == JobStatus.PENDING

        # Wait for ready
        endpoint = await backend.wait_for_ready(job.job_id)
        assert endpoint is not None

        # Verify job is ready
        status = await backend.get_job_status(job.job_id)
        assert status.status == JobStatus.READY
        assert status.endpoint is not None

        # Cleanup
        await backend.cleanup(job.job_id)
        status = await backend.get_job_status(job.job_id)
        assert status.status == JobStatus.STOPPED

    @pytest.mark.asyncio
    async def test_multiple_jobs(self, backend):
        """Test managing multiple jobs."""
        jobs = []
        for _ in range(3):
            job = await backend.submit_job()
            jobs.append(job)

        all_jobs = await backend.list_jobs()
        assert len(all_jobs) == 3

        # Make first job ready
        await backend.wait_for_ready(jobs[0].job_id)

        ready_jobs = await backend.list_jobs(status_filter=[JobStatus.READY])
        pending_jobs = await backend.list_jobs(status_filter=[JobStatus.PENDING])

        assert len(ready_jobs) == 1
        assert len(pending_jobs) == 2

        # Cleanup all
        for job in jobs:
            await backend.cleanup(job.job_id)

        active_jobs = await backend.list_jobs(status_filter=[JobStatus.READY, JobStatus.PENDING])
        assert len(active_jobs) == 0


# =============================================================================
# Test Slurm Backend (Mocked)
# =============================================================================


class TestSlurmBackendMocked:
    """Tests for Slurm backend with mocked subprocess."""

    @pytest.fixture
    def slurm_config(self):
        """Slurm-specific config."""
        return OrchestratorConfig(
            model_path="/models/test",
            slurm_partition="gpu",
            slurm_time="01:00:00",
            slurm_gpus=1,
            singularity_image="/opt/mini-rivault.sif",
        )

    @pytest.mark.asyncio
    async def test_slurm_submit_job_mocked(self, slurm_config):
        """Test Slurm job submission (mocked)."""
        # Import the actual backend
        try:
            from umrs.orchestrator.slurm_backend import SlurmBackend
        except ImportError:
            pytest.skip("SlurmBackend not available")

        backend = SlurmBackend(slurm_config)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            # Mock sbatch --parsable output (returns "JOBID" or "JOBID;CLUSTER")
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"12345\n", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            job = await backend.submit_job()
            assert job.job_id == "12345"
            assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_slurm_cleanup_mocked(self, slurm_config):
        """Test Slurm job cleanup (mocked)."""
        try:
            from umrs.orchestrator.slurm_backend import SlurmBackend
        except ImportError:
            pytest.skip("SlurmBackend not available")

        backend = SlurmBackend(slurm_config)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            await backend.cleanup("12345")
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_slurm_cleanup_logs_error_when_scancel_fails(self, slurm_config, caplog):
        """2026-09-18 finding: a still-provisioning registry id is not the
        slurm job number scancel needs; the silent no-op masked it. The
        backend must now log the scancel failure at ERROR."""
        try:
            from umrs.orchestrator.slurm_backend import SlurmBackend
        except ImportError:
            pytest.skip("SlurmBackend not available")

        backend = SlurmBackend(slurm_config)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (
                b"",
                b"scancel: error: Invalid job id specified\n",
            )
            mock_process.returncode = 1
            mock_exec.return_value = mock_process

            with caplog.at_level("ERROR", logger="umrs.orchestrator.slurm_backend"):
                await backend.cleanup("5434")

            assert any(
                rec.levelno == 40
                and "scancel --signal=TERM 5434 exited rc=1" in rec.getMessage()
                for rec in caplog.records
            )

    @pytest.mark.asyncio
    async def test_slurm_cleanup_success_logs_no_error(self, slurm_config, caplog):
        try:
            from umrs.orchestrator.slurm_backend import SlurmBackend
        except ImportError:
            pytest.skip("SlurmBackend not available")

        backend = SlurmBackend(slurm_config)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            with caplog.at_level("ERROR", logger="umrs.orchestrator.slurm_backend"):
                await backend.cleanup("384999")

            assert not any(rec.levelno == 40 for rec in caplog.records)


# =============================================================================
# Test Kubernetes Backend (Mocked)
# =============================================================================


@pytest.mark.skipif(
    not importlib.util.find_spec("kubernetes"),
    reason="kubernetes package not installed",
)
class TestKubernetesBackendMocked:
    """Tests for Kubernetes backend with mocked client."""

    @pytest.fixture
    def k8s_config(self):
        """Kubernetes-specific config."""
        return OrchestratorConfig(
            model_path="/models/test",
            k8s_namespace="rivault-test",
            k8s_image="ghcr.io/rccs/mini-rivault:test",
        )

    @pytest.mark.asyncio
    async def test_k8s_submit_job_mocked(self, k8s_config):
        """Test Kubernetes job submission (mocked)."""
        try:
            from umrs.orchestrator.kubernetes_backend import KubernetesBackend
        except ImportError:
            pytest.skip("KubernetesBackend not available")

        with (
            patch("kubernetes.client.BatchV1Api") as mock_batch,
            patch("kubernetes.client.CoreV1Api"),
            patch("kubernetes.client.NetworkingV1Api") as mock_networking,
            patch("kubernetes.config.load_incluster_config"),
        ):
            mock_job_result = MagicMock()
            mock_job_result.metadata.name = "rivault-test-123"
            creation_order = []

            def create_network_policy(**_kwargs):
                creation_order.append("network_policy")

            def create_job(**_kwargs):
                creation_order.append("job")
                return mock_job_result

            mock_networking.return_value.create_namespaced_network_policy.side_effect = (
                create_network_policy
            )
            mock_batch.return_value.create_namespaced_job.side_effect = create_job

            backend = KubernetesBackend(k8s_config)
            job = await backend.submit_job()

            # Job ID is generated by KubernetesBackend, starts with mini-rivault-
            assert job.job_id.startswith("mini-rivault-")
            assert job.status == JobStatus.PENDING
            mock_networking.return_value.create_namespaced_network_policy.assert_called_once()
            assert creation_order == ["network_policy", "job"]

    def test_k8s_manifest_includes_prefix_caching_env(self, k8s_config):
        try:
            from umrs.orchestrator.kubernetes_backend import KubernetesBackend
        except ImportError:
            pytest.skip("KubernetesBackend not available")

        k8s_config.vllm_enable_prefix_caching = True
        manifest = KubernetesBackend(k8s_config)._build_job_manifest(
            "mini-rivault-test",
            "/models/test",
        )

        env = {
            item["name"]: item.get("value")
            for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
            if "value" in item
        }
        assert env["VLLM_ENABLE_PREFIX_CACHING"] == "true"


# =============================================================================
# Test Compose Backend (Mocked)
# =============================================================================


class TestComposeBackendMocked:
    """Tests for Docker Compose backend with mocked subprocess."""

    @pytest.fixture
    def compose_config(self):
        """Compose-specific config."""
        return OrchestratorConfig(
            model_path="/models/test",
            compose_project_name="test-rivault",
            compose_file="/path/to/compose.yaml",
        )

    @pytest.mark.asyncio
    async def test_compose_submit_job_mocked(self, compose_config):
        """Test Compose job submission (mocked)."""
        try:
            from umrs.orchestrator.compose_backend import DockerComposeBackend
        except ImportError:
            pytest.skip("DockerComposeBackend not available")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            backend = DockerComposeBackend(compose_config)
            job = await backend.submit_job()

            # Job ID is the compose project name
            assert job.job_id == compose_config.compose_project_name
            assert job.status == JobStatus.STARTING
            assert job.backend == "docker-compose"
            assert mock_exec.call_args.kwargs["env"]["VLLM_ENABLE_PREFIX_CACHING"] == "false"

    @pytest.mark.asyncio
    async def test_compose_cleanup_mocked(self, compose_config):
        """Test Compose cleanup (mocked)."""
        try:
            from umrs.orchestrator.compose_backend import DockerComposeBackend
        except ImportError:
            pytest.skip("DockerComposeBackend not available")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            backend = DockerComposeBackend(compose_config)
            await backend.cleanup(compose_config.compose_project_name)

            # Verify compose down was called
            mock_exec.assert_called()
            call_args = mock_exec.call_args_list[0][0]
            assert "down" in call_args

    @pytest.mark.asyncio
    async def test_compose_get_job_status_ready(self, compose_config):
        """Test Compose job status when containers running and ready."""
        try:
            from umrs.orchestrator.compose_backend import DockerComposeBackend
        except ImportError:
            pytest.skip("DockerComposeBackend not available")

        # JSONL format - one JSON object per line
        compose_output = (
            '{"Name": "vllm", "State": "running"}\n{"Name": "litellm", "State": "running"}'
        )

        with (
            patch("asyncio.create_subprocess_exec") as mock_exec,
            patch.object(DockerComposeBackend, "check_readiness", return_value=True),
        ):
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (compose_output.encode(), b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            backend = DockerComposeBackend(compose_config)
            job = await backend.get_job_status(compose_config.compose_project_name)

            # Compose goes from STARTING → READY (not RUNNING)
            assert job.status == JobStatus.READY
            assert job.endpoint is not None

    @pytest.mark.asyncio
    async def test_compose_get_job_status_stopped(self, compose_config):
        """Test Compose job status when stopped."""
        try:
            from umrs.orchestrator.compose_backend import DockerComposeBackend
        except ImportError:
            pytest.skip("DockerComposeBackend not available")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            # No containers running - empty output
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            backend = DockerComposeBackend(compose_config)
            job = await backend.get_job_status(compose_config.compose_project_name)

            assert job.status == JobStatus.STOPPED


# =============================================================================
# Test Kubernetes Backend Extended (Mocked)
# =============================================================================


@pytest.mark.skipif(
    not importlib.util.find_spec("kubernetes"),
    reason="kubernetes package not installed",
)
class TestKubernetesBackendExtended:
    """Extended tests for Kubernetes backend with mocked K8s API."""

    @pytest.fixture
    def k8s_config(self):
        """Kubernetes-specific config."""
        return OrchestratorConfig(
            model_path="/models/test",
            k8s_namespace="test-ns",
        )

    @pytest.mark.asyncio
    async def test_k8s_cleanup_mocked(self, k8s_config):
        """Test Kubernetes cleanup (mocked)."""
        try:
            from umrs.orchestrator.kubernetes_backend import KubernetesBackend
        except ImportError:
            pytest.skip("KubernetesBackend not available")

        with (
            patch("kubernetes.client.BatchV1Api") as mock_batch,
            patch("kubernetes.client.CoreV1Api"),
            patch("kubernetes.client.NetworkingV1Api"),
            patch("kubernetes.config.load_incluster_config"),
        ):
            backend = KubernetesBackend(k8s_config)
            await backend.cleanup("mini-rivault-test-123")

            # Verify job deletion was called
            mock_batch.return_value.delete_namespaced_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_k8s_get_job_status_ready(self, k8s_config):
        """Test Kubernetes job status when pod is ready."""
        try:
            from umrs.orchestrator.kubernetes_backend import KubernetesBackend
        except ImportError:
            pytest.skip("KubernetesBackend not available")

        with (
            patch("kubernetes.client.BatchV1Api") as mock_batch,
            patch("kubernetes.client.CoreV1Api") as mock_core,
            patch("kubernetes.config.load_incluster_config"),
        ):
            mock_job = MagicMock()
            mock_job.status.active = 1
            mock_job.status.succeeded = None
            mock_job.status.failed = None
            mock_batch.return_value.read_namespaced_job.return_value = mock_job

            # Pod is running and ready
            mock_condition = MagicMock()
            mock_condition.type = "Ready"
            mock_condition.status = "True"
            mock_pod = MagicMock()
            mock_pod.status.phase = "Running"
            mock_pod.status.conditions = [mock_condition]
            mock_pod_list = MagicMock()
            mock_pod_list.items = [mock_pod]
            mock_core.return_value.list_namespaced_pod.return_value = mock_pod_list

            backend = KubernetesBackend(k8s_config)
            job = await backend.get_job_status("mini-rivault-test-123")

            assert job.status == JobStatus.STARTING
            assert job.endpoint is not None

    @pytest.mark.asyncio
    async def test_k8s_get_job_status_failed(self, k8s_config):
        """Test Kubernetes job status when failed."""
        try:
            from umrs.orchestrator.kubernetes_backend import KubernetesBackend
        except ImportError:
            pytest.skip("KubernetesBackend not available")

        with (
            patch("kubernetes.client.BatchV1Api") as mock_batch,
            patch("kubernetes.client.CoreV1Api"),
            patch("kubernetes.config.load_incluster_config"),
        ):
            mock_job = MagicMock()
            mock_job.status.active = None
            mock_job.status.succeeded = None
            mock_job.status.failed = 1
            mock_batch.return_value.read_namespaced_job.return_value = mock_job

            backend = KubernetesBackend(k8s_config)
            job = await backend.get_job_status("mini-rivault-test-123")

            assert job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_k8s_list_jobs_mocked(self, k8s_config):
        """Test listing Kubernetes jobs."""
        try:
            from umrs.orchestrator.kubernetes_backend import KubernetesBackend
        except ImportError:
            pytest.skip("KubernetesBackend not available")

        with (
            patch("kubernetes.client.BatchV1Api") as mock_batch,
            patch("kubernetes.client.CoreV1Api"),
            patch("kubernetes.config.load_incluster_config"),
        ):
            # Jobs for list_namespaced_job
            mock_job1 = MagicMock()
            mock_job1.metadata.name = "mini-rivault-job-1"
            mock_job1.status.active = 1
            mock_job1.status.succeeded = 0
            mock_job1.status.failed = 0

            mock_job2 = MagicMock()
            mock_job2.metadata.name = "mini-rivault-job-2"
            mock_job2.status.active = 0
            mock_job2.status.succeeded = 1
            mock_job2.status.failed = 0

            mock_job_list = MagicMock()
            mock_job_list.items = [mock_job1, mock_job2]
            mock_batch.return_value.list_namespaced_job.return_value = mock_job_list

            # Mock read_namespaced_job for get_job_status calls
            def read_job_side_effect(name, namespace):
                """Return appropriate job based on name."""
                if name == "mini-rivault-job-1":
                    return mock_job1
                return mock_job2

            mock_batch.return_value.read_namespaced_job.side_effect = read_job_side_effect

            backend = KubernetesBackend(k8s_config)
            jobs = await backend.list_jobs()

            assert len(jobs) == 2
            assert jobs[0].job_id == "mini-rivault-job-1"
            assert jobs[1].job_id == "mini-rivault-job-2"
