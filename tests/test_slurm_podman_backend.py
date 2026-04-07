"""
Unit tests for the Slurm + Podman-Compose orchestrator backend.
"""

import asyncio
import os
import textwrap
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from umrs.orchestrator.base import EndpointInfo, JobInfo, JobStatus, OrchestratorConfig
from umrs.orchestrator.site_config import SiteConfig, SchedulerConfig, RuntimeConfig, NetworkConfig, reset_site_config
from umrs.orchestrator.slurm_podman_backend import SlurmPodmanBackend


def _run(coro):
    """Helper to run async test functions synchronously."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _clean_singleton():
    reset_site_config()
    yield
    reset_site_config()


@pytest.fixture
def site():
    return SiteConfig(
        name="test-site",
        scheduler=SchedulerConfig(partition="test-gpu", time_limit="02:00:00", gpus=1),
        runtime=RuntimeConfig(rivault_dir="/opt/rivault"),
        network=NetworkConfig(litellm_port=8001, litellm_master_key="sk-test"),
    )


@pytest.fixture
def slurm_script(tmp_path):
    script = tmp_path / "serve-model.slurm"
    script.write_text("#!/bin/bash\necho test\n")
    return str(script)


@pytest.fixture
def output_dir(tmp_path):
    d = tmp_path / "jobs"
    d.mkdir()
    return d


@pytest.fixture
def backend(site, slurm_script, output_dir):
    b = SlurmPodmanBackend(site=site, slurm_script=slurm_script)
    b._output_dir = output_dir
    return b


# =============================================================================
# Tests — properties
# =============================================================================


class TestBackendProperties:

    def test_backend_name(self, backend):
        assert backend.backend_name == "slurm-podman"

    def test_slurm_script_explicit(self, backend, slurm_script):
        assert backend.slurm_script == slurm_script

    def test_slurm_script_resolves_from_repo_dir_and_site_name(self, tmp_path):
        repo_root = tmp_path / "repo"
        wrapper = repo_root / "deploy" / "test-site" / "serve-model.slurm"
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/bash\necho test\n")

        site = SiteConfig(
            name="test-site",
            scheduler=SchedulerConfig(partition="test-gpu"),
            runtime=RuntimeConfig(
                repo_dir=str(repo_root),
                rivault_dir=str(repo_root / "src" / "umrs" / "mini-rivault-llm"),
            ),
            network=NetworkConfig(litellm_port=8001, litellm_master_key="sk-test"),
        )

        backend = SlurmPodmanBackend(site=site)
        assert backend.slurm_script == str(wrapper)

    def test_slurm_script_resolves_from_rivault_dir_when_repo_dir_missing(self, tmp_path):
        repo_root = tmp_path / "repo"
        rivault_dir = repo_root / "src" / "umrs" / "mini-rivault-llm"
        wrapper = repo_root / "deploy" / "test-site" / "serve-model.slurm"
        rivault_dir.mkdir(parents=True)
        wrapper.parent.mkdir(parents=True)
        wrapper.write_text("#!/bin/bash\necho test\n")

        site = SiteConfig(
            name="test-site",
            scheduler=SchedulerConfig(partition="test-gpu"),
            runtime=RuntimeConfig(rivault_dir=str(rivault_dir)),
            network=NetworkConfig(litellm_port=8001, litellm_master_key="sk-test"),
        )

        backend = SlurmPodmanBackend(site=site)
        assert backend.slurm_script == str(wrapper)

    def test_slurm_script_falls_back_to_env(self, tmp_path):
        env_wrapper = tmp_path / "env-serve-model.slurm"
        env_wrapper.write_text("#!/bin/bash\necho env\n")

        site = SiteConfig(
            name="missing-site",
            scheduler=SchedulerConfig(partition="test-gpu"),
            runtime=RuntimeConfig(repo_dir=str(tmp_path / "missing-repo")),
            network=NetworkConfig(litellm_port=8001, litellm_master_key="sk-test"),
        )

        with patch.dict(os.environ, {"RIVAULT_SLURM_SCRIPT": str(env_wrapper)}):
            backend = SlurmPodmanBackend(site=site)
            assert backend.slurm_script == str(env_wrapper)

    def test_site_property(self, backend, site):
        assert backend.site is site


# =============================================================================
# Tests — sbatch env
# =============================================================================


class TestSbatchEnv:

    def test_env_includes_site_values(self, backend):
        env = backend._sbatch_env()
        assert env["SLURM_PARTITION"] == "test-gpu"
        assert env["NGINX_LITELLM_DOCKER_PORT"] == "8001"
        assert env["LITELLM_MASTER_KEY"] == "sk-test"

    def test_env_includes_overrides(self, backend):
        env = backend._sbatch_env(VLLM_MODEL_PATH="/models/custom")
        assert env["VLLM_MODEL_PATH"] == "/models/custom"

    def test_env_overrides_win(self, backend):
        env = backend._sbatch_env(LITELLM_MASTER_KEY="override-key")
        assert env["LITELLM_MASTER_KEY"] == "override-key"


# =============================================================================
# Tests — submit_job
# =============================================================================


class TestSubmitJob:

    def test_submit_success(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"12345\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            job = _run(backend.submit_job(model_path="/models/test"))

        assert job.job_id == "12345"
        assert job.status == JobStatus.PENDING
        assert job.backend == "slurm-podman"

        # Verify sbatch was called with the right script
        call_args = mock_exec.call_args
        args = call_args.args if call_args.args else call_args[0]
        assert args[0] == "sbatch"
        assert "--parsable" in args
        assert backend.slurm_script in args

    def test_submit_passes_partition(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"99999\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            _run(backend.submit_job())

        args = mock_exec.call_args.args if mock_exec.call_args.args else mock_exec.call_args[0]
        assert "--partition=test-gpu" in args

    def test_submit_failure_raises(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"sbatch: error: Batch job submission failed\n")
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="sbatch failed"):
                _run(backend.submit_job())

    def test_submit_missing_script_raises(self, site):
        b = SlurmPodmanBackend(site=site, slurm_script="/nonexistent.slurm")
        with pytest.raises(FileNotFoundError, match="Slurm script not found"):
            _run(b.submit_job())

    def test_submit_cluster_format_jobid(self, backend):
        """Handle sbatch --parsable returning 'JOBID;CLUSTER' format."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"67890;mycluster\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            job = _run(backend.submit_job())
        assert job.job_id == "67890"


# =============================================================================
# Tests — get_job_status
# =============================================================================


class TestGetJobStatus:

    def test_pending_job(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"(null) PENDING\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            job = _run(backend.get_job_status("12345"))

        assert job.status == JobStatus.PENDING
        assert job.endpoint is None

    def test_running_job_with_node(self, backend, output_dir):
        # Write a fake output file with node
        (output_dir / "job-12345.out").write_text("Node: compute-01\n")

        # squeue returns running
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"compute-01 RUNNING\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            job = _run(backend.get_job_status("12345"))

        assert job.status == JobStatus.STARTING  # RUNNING maps to STARTING
        assert job.endpoint is not None
        assert job.endpoint.node == "compute-01"
        assert job.endpoint.litellm_port == 8001

    def test_completed_job_via_sacct(self, backend):
        # squeue returns nothing (job left queue)
        squeue_proc = AsyncMock()
        squeue_proc.communicate.return_value = (b"", b"")
        squeue_proc.returncode = 0

        # sacct returns COMPLETED
        sacct_proc = AsyncMock()
        sacct_proc.communicate.return_value = (b"COMPLETED\n", b"")
        sacct_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", side_effect=[squeue_proc, sacct_proc]):
            job = _run(backend.get_job_status("12345"))

        assert job.status == JobStatus.STOPPED


# =============================================================================
# Tests — cleanup
# =============================================================================


class TestCleanup:

    def test_cleanup_calls_scancel(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            _run(backend.cleanup("12345"))

        args = mock_exec.call_args.args if mock_exec.call_args.args else mock_exec.call_args[0]
        assert "scancel" in args
        assert "--signal=TERM" in args
        assert "12345" in args

    def test_cleanup_force_uses_kill(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            _run(backend.cleanup("12345", force=True))

        args = mock_exec.call_args.args if mock_exec.call_args.args else mock_exec.call_args[0]
        assert "--signal=KILL" in args

    def test_cleanup_removes_output_files(self, backend, output_dir):
        (output_dir / "job-12345.out").write_text("output")
        (output_dir / "job-12345.err").write_text("errors")

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            _run(backend.cleanup("12345"))

        assert not (output_dir / "job-12345.out").exists()
        assert not (output_dir / "job-12345.err").exists()


# =============================================================================
# Tests — _map_slurm_state
# =============================================================================


class TestMapSlurmState:

    @pytest.mark.parametrize("slurm_state,expected", [
        ("PENDING", JobStatus.PENDING),
        ("CONFIGURING", JobStatus.PENDING),
        ("RUNNING", JobStatus.STARTING),
        ("COMPLETING", JobStatus.RUNNING),
        ("COMPLETED", JobStatus.STOPPED),
        ("FAILED", JobStatus.FAILED),
        ("CANCELLED", JobStatus.STOPPED),
        ("TIMEOUT", JobStatus.FAILED),
        ("NODE_FAIL", JobStatus.FAILED),
        ("PREEMPTED", JobStatus.FAILED),
        ("UNKNOWN_STATE", JobStatus.UNKNOWN),
        ("", JobStatus.UNKNOWN),
    ])
    def test_state_mapping(self, slurm_state, expected):
        assert SlurmPodmanBackend._map_slurm_state(slurm_state) == expected


# =============================================================================
# Tests — _node_from_output
# =============================================================================


class TestNodeFromOutput:

    def test_extracts_node(self, backend, output_dir):
        (output_dir / "job-123.out").write_text(
            "Starting job...\nNode: compute-42\nDone.\n"
        )
        node = _run(backend._node_from_output("123"))
        assert node == "compute-42"

    def test_returns_none_if_no_file(self, backend):
        node = _run(backend._node_from_output("999"))
        assert node is None

    def test_returns_none_if_no_match(self, backend, output_dir):
        (output_dir / "job-123.out").write_text("Some other output\n")
        node = _run(backend._node_from_output("123"))
        assert node is None


# =============================================================================
# Tests — factory registration
# =============================================================================


class TestFactoryRegistration:

    def test_slurm_podman_registered(self):
        from umrs.orchestrator.factory import get_orchestrator, _lazy_register_builtins, _BACKEND_REGISTRY
        _lazy_register_builtins()
        assert "slurm-podman" in _BACKEND_REGISTRY


# =============================================================================
# Tests — health_check
# =============================================================================


class TestHealthCheck:

    def test_healthy_when_script_exists(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"test-gpu up 1\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = _run(backend.health_check())

        assert result["healthy"] is True
        assert result["backend"] == "slurm-podman"
        assert result["site"] == "test-site"

    def test_unhealthy_when_script_missing(self, site):
        b = SlurmPodmanBackend(site=site, slurm_script="/nonexistent.slurm")

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"test-gpu up 1\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = _run(b.health_check())

        assert result["healthy"] is False
        assert "missing" in result["message"].lower()
