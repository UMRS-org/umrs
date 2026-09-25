"""
Unit tests for the Slurm + Podman-Compose orchestrator backend.
"""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest
from umrs.orchestrator.base import JobStatus
from umrs.orchestrator.site_config import (
    NetworkConfig,
    RuntimeConfig,
    SchedulerConfig,
    SiteConfig,
    reset_site_config,
)
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
        with patch.dict(os.environ, {}, clear=True):
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

    def test_process_env_overrides_site_defaults(self, backend):
        with patch.dict(os.environ, {"RIVAULT_PROFILES": "harness vllm"}, clear=False):
            env = backend._sbatch_env()

        assert env["RIVAULT_PROFILES"] == "harness vllm"

    def test_env_exports_partition_candidates_safely(self, slurm_script, output_dir):
        site = SiteConfig(
            name="test-site",
            scheduler=SchedulerConfig(partition="gpu-a,gpu-b", time_limit="02:00:00", gpus=1),
            runtime=RuntimeConfig(rivault_dir="/opt/rivault"),
            network=NetworkConfig(litellm_port=8001, litellm_master_key="sk-test"),
        )
        backend = SlurmPodmanBackend(site=site, slurm_script=slurm_script)
        backend._output_dir = output_dir

        with patch.dict(os.environ, {}, clear=True):
            env = backend._sbatch_env()

        assert env["SLURM_PARTITION"] == "gpu-a"
        assert env["SLURM_PARTITIONS"] == "gpu-a,gpu-b"

    def test_env_includes_egress_proxy_site_values(self, slurm_script, output_dir):
        site = SiteConfig(
            name="test-site",
            scheduler=SchedulerConfig(partition="test-gpu"),
            runtime=RuntimeConfig(rivault_dir="/opt/rivault"),
            network=NetworkConfig(
                litellm_port=8001,
                litellm_master_key="sk-test",
                egress_proxy_url="http://egress-guard:8888",
                egress_no_proxy="localhost,vllm,litellm",
            ),
        )
        backend = SlurmPodmanBackend(site=site, slurm_script=slurm_script)
        backend._output_dir = output_dir

        with patch.dict(os.environ, {}, clear=True):
            env = backend._sbatch_env()

        assert env["RIVAULT_HTTP_PROXY"] == "http://egress-guard:8888"
        assert env["RIVAULT_HTTPS_PROXY"] == "http://egress-guard:8888"
        assert env["EGRESS_GUARD_PROXY_URL"] == "http://egress-guard:8888"
        assert env["RIVAULT_NO_PROXY"] == "localhost,vllm,litellm"


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

    def test_submit_flattens_env_overrides(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"99999\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            _run(
                backend.submit_job(
                    env={
                        "RIVAULT_PROFILES": "harness vision-bentoml",
                        "CUSTOM_FLAG": 7,
                    }
                )
            )

        env = mock_exec.call_args.kwargs["env"]
        assert env["RIVAULT_PROFILES"] == "harness vision-bentoml"
        assert env["CUSTOM_FLAG"] == "7"

    def test_submit_separates_model_path_from_served_name(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"77777\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            _run(
                backend.submit_job(
                    model_path="/models/qwen3-8b",
                    env={"MODEL_NAME": "Qwen/Qwen3-8B"},
                )
            )

        env = mock_exec.call_args.kwargs["env"]
        assert env["VLLM_MODEL_PATH"] == "/models/qwen3-8b"
        assert env["VLLM_SERVED_MODEL_NAME"] == "Qwen/Qwen3-8B"
        assert env["VLLM_MODEL"] == "Qwen/Qwen3-8B"

    def test_submit_selects_one_active_partition_from_candidates(self, slurm_script, output_dir):
        site = SiteConfig(
            name="test-site",
            scheduler=SchedulerConfig(partition="gpu-a,gpu-b", time_limit="02:00:00", gpus=1),
            runtime=RuntimeConfig(rivault_dir="/opt/rivault"),
            network=NetworkConfig(litellm_port=8001, litellm_master_key="sk-test"),
        )
        backend = SlurmPodmanBackend(site=site, slurm_script=slurm_script)
        backend._output_dir = output_dir

        sinfo_proc = AsyncMock()
        sinfo_proc.communicate.return_value = (b"gpu-a drain\ngpu-b idle\n", b"")
        sinfo_proc.returncode = 0

        sbatch_proc = AsyncMock()
        sbatch_proc.communicate.return_value = (b"54321\n", b"")
        sbatch_proc.returncode = 0

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[sinfo_proc, sbatch_proc],
        ) as mock_exec:
            job = _run(backend.submit_job())

        assert job.job_id == "54321"
        args = mock_exec.call_args.args if mock_exec.call_args.args else mock_exec.call_args[0]
        assert "--partition=gpu-b" in args
        assert "--partition=gpu-a,gpu-b" not in args

        env = mock_exec.call_args.kwargs["env"]
        assert env["SLURM_PARTITION"] == "gpu-b"

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
        (output_dir / "slurm-12345.out").write_text("Running on node: compute-01\n")

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

        wait_for_exit = AsyncMock(return_value=False)
        with patch.object(backend, "_wait_for_job_exit", wait_for_exit):
            with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
                _run(backend.cleanup("12345"))

        args = mock_exec.call_args.args if mock_exec.call_args.args else mock_exec.call_args[0]
        assert args == ("scancel", "-Q", "12345")

    def test_cleanup_force_uses_kill(self, backend):
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            _run(backend.cleanup("12345", force=True))

        args = mock_exec.call_args.args if mock_exec.call_args.args else mock_exec.call_args[0]
        assert args[0:2] == ("scancel", "-Q")
        assert "--full" in args
        assert "--signal=KILL" in args
        assert args[-1] == "12345"

    def test_cleanup_preserves_retained_slurm_logs(self, backend, output_dir):
        (output_dir / "slurm-12345.out").write_text("output")
        (output_dir / "slurm-12345.err").write_text("errors")

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            _run(backend.cleanup("12345"))

        assert (output_dir / "slurm-12345.out").exists()
        assert (output_dir / "slurm-12345.err").exists()


# =============================================================================
# Tests — _map_slurm_state
# =============================================================================


class TestMapSlurmState:
    @pytest.mark.parametrize(
        "slurm_state,expected",
        [
            ("PENDING", JobStatus.PENDING),
            ("CONFIGURING", JobStatus.PENDING),
            ("RUNNING", JobStatus.STARTING),
            ("COMPLETING", JobStatus.STOPPED),
            ("COMPLETED", JobStatus.STOPPED),
            ("FAILED", JobStatus.FAILED),
            ("CANCELLED", JobStatus.STOPPED),
            ("CANCELLED+", JobStatus.STOPPED),
            ("TIMEOUT", JobStatus.FAILED),
            ("NODE_FAIL", JobStatus.FAILED),
            ("PREEMPTED", JobStatus.FAILED),
            ("UNKNOWN_STATE", JobStatus.UNKNOWN),
            ("", JobStatus.UNKNOWN),
        ],
    )
    def test_state_mapping(self, slurm_state, expected):
        assert SlurmPodmanBackend._map_slurm_state(slurm_state) == expected


class TestAccountingStateMapping:
    """`_status_from_accounting_state` must not report a live job as UNKNOWN.

    Regression: a still-RUNNING job queried via ``sacct`` (squeue miss under
    load) previously mapped to ``JobStatus.UNKNOWN``, which the active-endpoint
    liveness sweep treated as dead and removed the healthy endpoint.
    """

    @pytest.mark.parametrize(
        "state,expected",
        [
            ("RUNNING", JobStatus.RUNNING),
            ("PENDING", JobStatus.PENDING),
            ("COMPLETED", JobStatus.STOPPED),
            ("FAILED", JobStatus.FAILED),
            ("CANCELLED", JobStatus.FAILED),
            ("CANCELLED by 1000", JobStatus.FAILED),
            ("TIMEOUT", JobStatus.FAILED),
            ("NODE_FAIL", JobStatus.FAILED),
            ("OUT_OF_MEMORY", JobStatus.FAILED),
            ("PREEMPTED", JobStatus.FAILED),
            ("", JobStatus.UNKNOWN),
        ],
    )
    def test_accounting_state_mapping(self, backend, state, expected):
        from umrs.orchestrator.slurm_backend import SlurmBackend

        # ``_status_from_accounting_state`` is a pure mapper (ignores ``self``)
        # defined on ``SlurmBackend`` and inherited by ``RemoteSlurmBackend``
        # (the AI4S remote-Slurm path).
        result = SlurmBackend._status_from_accounting_state(
            SlurmBackend.__new__(SlurmBackend), state
        )
        assert result == expected


# =============================================================================
# Tests — _node_from_output
# =============================================================================


class TestNodeFromOutput:
    def test_extracts_node(self, backend, output_dir):
        (output_dir / "slurm-123.out").write_text(
            "Starting job...\nFully qualified name: compute-42\nDone.\n"
        )
        node = _run(backend._node_from_output("123"))
        assert node == "compute-42"

    def test_prefers_fqdn_running_on_node_over_short_name(self, backend, output_dir):
        """The wrapper prints both a FQDN line and a short name; the fqdn must
        win because short names only resolve in the site's search domain."""
        (output_dir / "slurm-124.out").write_text(
            "Allocated nodes: dgx-spark-0\n"
            "Running on node: spark-atom-0.r-ccs10.riken.jp\n"
            "Fully qualified name: spark-atom-0\n"
        )
        node = _run(backend._node_from_output("124"))
        assert node == "spark-atom-0.r-ccs10.riken.jp"

    def test_reads_err_file_as_well(self, backend, output_dir):
        """Job header lands in both .out and .err streams of the wrapper."""
        (output_dir / "slurm-125.err").write_text(
            "Running on node: spark-atom-1.r-ccs10.riken.jp\n"
        )
        node = _run(backend._node_from_output("125"))
        assert node == "spark-atom-1.r-ccs10.riken.jp"

    def test_status_prefers_output_node_over_squeue_token(self, backend, output_dir):
        """Regression (bdrt): squeue reports dgx-spark-0 (no DNS on the login
        node); the wrapper's FQDN must replace it for the endpoint probe."""
        (output_dir / "slurm-12345.out").write_text(
            "Allocated nodes: dgx-spark-0\nRunning on node: spark-atom-0.r-ccs10.riken.jp\n"
        )
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"dgx-spark-0 RUNNING\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            job = _run(backend.get_job_status("12345"))

        assert job.endpoint is not None
        assert job.endpoint.node == "spark-atom-0.r-ccs10.riken.jp"

    def test_returns_none_if_no_file(self, backend):
        node = _run(backend._node_from_output("999"))
        assert node is None

    def test_returns_none_if_no_match(self, backend, output_dir):
        (output_dir / "slurm-123.out").write_text("Some other output\n")
        node = _run(backend._node_from_output("123"))
        assert node is None


# =============================================================================
# Tests — factory registration
# =============================================================================


class TestFactoryRegistration:
    def test_slurm_podman_registered(self):
        from umrs.orchestrator.factory import _BACKEND_REGISTRY, _lazy_register_builtins

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


# =============================================================================
# Regression — JOB_INFO_DIR propagation (orphan fix, 2026-08-31)
# =============================================================================


class TestJobInfoDirPropagation:
    def test_submit_sets_job_info_dir_by_default(self, backend):
        """Without an explicit override, JOB_INFO_DIR must land in the sbatch
        env pointing at the backend's output dir — otherwise serve-model.slurm
        falls back to ${WRAPPER_DIR}/jobs on the ephemeral staged copy and the
        60s orphan reaper marks every valid sbatch as orphaned."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"12345\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            _run(backend.submit_job(model_path="/models/test"))

        env = mock_exec.call_args.kwargs["env"]
        assert env["JOB_INFO_DIR"] == str(backend._output_dir)

    def test_caller_override_wins(self, backend):
        """An explicit JOB_INFO_DIR override must always win over the default
        (relative missions with custom write targets)."""
        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"12345\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            _run(
                backend.submit_job(
                    model_path="/models/test",
                    env={"JOB_INFO_DIR": "/custom/info-dir"},
                )
            )

        env = mock_exec.call_args.kwargs["env"]
        assert env["JOB_INFO_DIR"] == "/custom/info-dir"

    def test_submit_sets_wrapper_dir_for_the_spooled_shim(self, backend):
        """sbatch spools the script, so bash-relative discovery of site.yaml
        fails; RIVAULT_WRAPPER_DIR must point at the real wrapper dir (same
        contract as remote_slurm_backend; BDRT hotfix upstreamed 2026-09-25)."""
        import os

        mock_proc = AsyncMock()
        mock_proc.communicate.return_value = (b"12345\n", b"")
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            _run(backend.submit_job(model_path="/models/test"))

        env = mock_exec.call_args.kwargs["env"]
        assert env["RIVAULT_WRAPPER_DIR"] == os.path.dirname(backend.slurm_script)
