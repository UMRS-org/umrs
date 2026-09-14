"""
Tests for Phase 1 sandbox launcher fixes.

Covers:
- slurm_runner: SINGULARITYENV_ prefix, source bind mount, metrics path
- runner: _generate_container_env UMRS_* passthrough
- plugin_runner.entrypoint: env-var parsing, step import, manifest emission
- user_workflow: _read_step_manifest, async invocation pattern
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from umrs.pipeline_sdk.workflow import (
    ExecutionMode,
    OutputSpec,
    SourceSpec,
    SourceType,
    StepSpec,
    WorkflowDefinition,
)
from umrs.plugins.models import PluginResult, PluginSourceType, PluginSpec, PluginStatus
from umrs.plugins.runner import PluginRunner
from umrs.plugins.slurm_runner import SlurmPluginRunner
from umrs.plugins.step_adapter import step_to_plugin_spec
from umrs.plugins.user_workflow import UserWorkflow, UserWorkflowRunner

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_plugin_spec(**overrides) -> PluginSpec:
    defaults = dict(
        source_type=PluginSourceType.LOCAL_PATH,
        source_uri="/tmp/test_scripts",
        entrypoint="mod:MyStep",
        env_vars={
            "UMRS_STEP_NAME": "test_step",
            "UMRS_STEP_ENTRYPOINT": "mod:MyStep",
            "UMRS_STEP_CONFIG": "{}",
            "UMRS_INPUT_BUCKET": "in-bucket",
            "UMRS_INPUT_KEYS": '["a.npy"]',
            "UMRS_OUTPUT_BUCKET": "out-bucket",
            "UMRS_RUN_ID": "run123",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        timeout_seconds=3600,
    )
    defaults.update(overrides)
    return PluginSpec(**defaults)


def _source_spec(path="/tmp/test") -> SourceSpec:
    return SourceSpec(type=SourceType.LOCAL_PATH, path=path)


def _make_wf_def(**overrides) -> WorkflowDefinition:
    defaults = dict(
        name="test_wf",
        source=_source_spec(),
        steps=[StepSpec(name="step_a", entrypoint="mod:StepA", config={"k": "v"})],
        version="1.0",
        description="test",
        output=OutputSpec(return_type="vector", embedding_vector_size=128),
    )
    defaults.update(overrides)
    return WorkflowDefinition(**defaults)


# =========================================================================
# 1. runner.py — _generate_container_env env-var prefix fix
# =========================================================================


class TestGenerateContainerEnv:
    """Verify that UMRS_* vars pass through without USER_ prefix."""

    def _make_runner(self):
        """Create a concrete runner subclass for testing the base method."""

        class _TestRunner(PluginRunner):
            async def _execute_backend(self, plugin_id, spec):
                pass

            async def cancel(self, plugin_id):
                return False

            async def get_status(self, plugin_id):
                return None

        return _TestRunner()

    def test_umrs_vars_pass_through_unprefixed(self):
        runner = self._make_runner()
        spec = _make_plugin_spec()
        env = runner._generate_container_env(spec)

        # UMRS_* vars should pass through as-is
        assert env["UMRS_STEP_NAME"] == "test_step"
        assert env["UMRS_STEP_ENTRYPOINT"] == "mod:MyStep"
        assert env["UMRS_RUN_ID"] == "run123"

        # They should NOT have USER_ prefix
        assert "USER_UMRS_STEP_NAME" not in env
        assert "USER_UMRS_STEP_ENTRYPOINT" not in env
        assert "USER_UMRS_RUN_ID" not in env

    def test_non_umrs_non_plugin_vars_get_user_prefix(self):
        runner = self._make_runner()
        spec = _make_plugin_spec()
        env = runner._generate_container_env(spec)

        # Non-UMRS, non-PLUGIN vars get USER_ prefix
        assert env["USER_CUDA_VISIBLE_DEVICES"] == "0"
        assert "CUDA_VISIBLE_DEVICES" not in env

    def test_plugin_vars_are_blocked(self):
        runner = self._make_runner()
        spec = _make_plugin_spec(
            env_vars={
                "PLUGIN_EVIL": "should_not_appear",
                "UMRS_STEP_NAME": "ok",
            }
        )
        env = runner._generate_container_env(spec)

        assert "PLUGIN_EVIL" not in env
        assert "USER_PLUGIN_EVIL" not in env
        # But the system PLUGIN_* vars set by the runner itself are present
        assert "PLUGIN_ENTRYPOINT" in env

    def test_system_plugin_vars_always_set(self):
        runner = self._make_runner()
        spec = _make_plugin_spec()
        env = runner._generate_container_env(spec)

        assert env["PLUGIN_ENTRYPOINT"] == "mod:MyStep"
        assert env["PLUGIN_SOURCE_TYPE"] == "local_path"
        assert env["PLUGIN_SOURCE_URI"] == "/tmp/test_scripts"
        assert env["PLUGIN_TIMEOUT"] == "3600"


class TestStepAdapterReservedEnv:
    """Verify user env cannot override reserved step-executor keys."""

    def test_reserved_step_env_keys_are_ignored(self):
        spec = step_to_plugin_spec(
            step_name="real-step",
            entrypoint="real.mod:RealStep",
            script_dir="/scripts",
            step_config={},
            input_bucket="real-in",
            input_keys=["a.npy"],
            output_bucket="real-out",
            run_id="run-123",
            env_vars={
                "UMRS_STEP_ENTRYPOINT": "evil.mod:EvilStep",
                "UMRS_INPUT_BUCKET": "evil-in",
                "PLUGIN_SOURCE_URI": "/evil",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        )

        assert spec.env_vars["UMRS_STEP_ENTRYPOINT"] == "real.mod:RealStep"
        assert spec.env_vars["UMRS_INPUT_BUCKET"] == "real-in"
        assert "PLUGIN_SOURCE_URI" not in spec.env_vars
        assert spec.env_vars["CUDA_VISIBLE_DEVICES"] == "0"


# =========================================================================
# 2. slurm_runner — sbatch script generation
# =========================================================================


class TestSlurmSbatchScript:
    """Verify sbatch script fixes: SINGULARITYENV_, source bind, metrics path."""

    def test_env_vars_use_singularityenv_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            runner = SlurmPluginRunner(work_dir=td)
            spec = _make_plugin_spec()
            plugin_id = "test-abc123"

            job_dir = Path(td) / "plugin_jobs" / plugin_id
            job_dir.mkdir(parents=True)
            script_path = job_dir / "job.sh"

            runner._generate_sbatch_script(script_path, plugin_id, spec)
            script = script_path.read_text()

            # All env exports should use SINGULARITYENV_ prefix
            assert "SINGULARITYENV_PLUGIN_ENTRYPOINT" in script
            assert "SINGULARITYENV_UMRS_STEP_NAME" in script
            # Old-style plain exports should NOT appear
            lines = [l for l in script.splitlines() if l.startswith("export ")]
            for line in lines:
                assert "SINGULARITYENV_" in line, f"Missing prefix: {line}"

    def test_source_dir_bound_for_local_path(self):
        with tempfile.TemporaryDirectory() as td:
            runner = SlurmPluginRunner(work_dir=td)
            spec = _make_plugin_spec(source_uri="/opt/user_scripts")
            plugin_id = "test-src123"

            job_dir = Path(td) / "plugin_jobs" / plugin_id
            job_dir.mkdir(parents=True)
            script_path = job_dir / "job.sh"

            runner._generate_sbatch_script(script_path, plugin_id, spec)
            script = script_path.read_text()

            assert "/opt/user_scripts:/source:ro" in script

    def test_metrics_path_is_host_side(self):
        with tempfile.TemporaryDirectory() as td:
            runner = SlurmPluginRunner(work_dir=td)
            spec = _make_plugin_spec()
            plugin_id = "test-met456"

            job_dir = Path(td) / "plugin_jobs" / plugin_id
            job_dir.mkdir(parents=True)
            script_path = job_dir / "job.sh"

            runner._generate_sbatch_script(script_path, plugin_id, spec)
            script = script_path.read_text()

            # Metrics should be written to the host-side path, not /workspace
            host_metrics = str(job_dir / "metrics.txt")
            assert host_metrics in script
            assert "/workspace/metrics.txt" not in script

    def test_entrypoint_is_umrs_module(self):
        with tempfile.TemporaryDirectory() as td:
            runner = SlurmPluginRunner(work_dir=td)
            spec = _make_plugin_spec()
            plugin_id = "test-ep789"

            job_dir = Path(td) / "plugin_jobs" / plugin_id
            job_dir.mkdir(parents=True)
            script_path = job_dir / "job.sh"

            runner._generate_sbatch_script(script_path, plugin_id, spec)
            script = script_path.read_text()

            assert "python -m umrs.plugins.plugin_runner.entrypoint" in script
            assert "python -m plugin_runner.entrypoint" not in script

    def test_gpu_line_present_when_gpu_requested(self):
        with tempfile.TemporaryDirectory() as td:
            runner = SlurmPluginRunner(work_dir=td)
            spec = _make_plugin_spec(gpu_count=2)
            plugin_id = "test-gpu"

            job_dir = Path(td) / "plugin_jobs" / plugin_id
            job_dir.mkdir(parents=True)
            script_path = job_dir / "job.sh"

            runner._generate_sbatch_script(script_path, plugin_id, spec)
            script = script_path.read_text()

            assert "#SBATCH --gpus=2" in script


# =========================================================================
# 3. plugin_runner.entrypoint — env-var parsing and manifest emission
# =========================================================================


class TestPluginRunnerEntrypoint:
    """Test the in-container step executor."""

    def test_successful_execution_writes_manifest(self, tmp_path):
        # Create a step script
        step_dir = tmp_path / "steps"
        step_dir.mkdir()
        (step_dir / "my_step.py").write_text("""
from umrs.pipeline_sdk.step import Step, StepContext, StepResult

class MyStep(Step):
    name = "my_step"
    def setup(self, config):
        pass
    def run(self, ctx):
        return StepResult(
            output_keys=["out/result.npy"],
            checksums=["abc123"],
            metadata=[{"processed": True}],
        )
    def teardown(self):
        pass
""")

        workspace = tmp_path / "workspace"
        workspace.mkdir()

        env = {
            "UMRS_STEP_NAME": "my_step",
            "UMRS_STEP_ENTRYPOINT": "my_step:MyStep",
            "UMRS_STEP_CONFIG": "{}",
            "UMRS_INPUT_BUCKET": "test-in",
            "UMRS_INPUT_KEYS": '["input/a.npy"]',
            "UMRS_OUTPUT_BUCKET": "test-out",
            "UMRS_RUN_ID": "run-test-001",
            "PLUGIN_SOURCE_URI": str(step_dir),
            "PLUGIN_WORKSPACE": str(workspace),
            "PLUGIN_SOURCE_DIR": str(step_dir),
        }

        with patch.dict(os.environ, env, clear=False):
            # Patch module-level constants
            import umrs.plugins.plugin_runner.entrypoint as ep_mod
            from umrs.plugins.plugin_runner.entrypoint import main

            orig_ws = ep_mod.WORKSPACE
            orig_sd = ep_mod.SOURCE_DIR
            ep_mod.WORKSPACE = workspace
            ep_mod.SOURCE_DIR = step_dir
            try:
                exit_code = main()
            finally:
                ep_mod.WORKSPACE = orig_ws
                ep_mod.SOURCE_DIR = orig_sd

        assert exit_code == 0

        manifest_path = workspace / "result_manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "succeeded"
        assert manifest["step_name"] == "my_step"
        assert manifest["run_id"] == "run-test-001"
        assert manifest["output_keys"] == ["out/result.npy"]
        assert manifest["checksums"] == ["abc123"]
        assert manifest["metadata"] == [{"processed": True}]

    def test_missing_env_var_writes_failed_manifest(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Don't set required UMRS_STEP_ENTRYPOINT
        env = {
            "UMRS_STEP_NAME": "broken",
            "UMRS_STEP_ENTRYPOINT": "",  # empty → required fails
            "PLUGIN_WORKSPACE": str(workspace),
        }

        with patch.dict(os.environ, env, clear=False):
            import umrs.plugins.plugin_runner.entrypoint as ep_mod

            orig_ws = ep_mod.WORKSPACE
            ep_mod.WORKSPACE = workspace
            try:
                exit_code = ep_mod.main()
            finally:
                ep_mod.WORKSPACE = orig_ws

        assert exit_code == 1

        manifest_path = workspace / "result_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["status"] == "failed"
        assert "error" in manifest


# =========================================================================
# 4. user_workflow — _read_step_manifest
# =========================================================================


class TestReadStepManifest:
    def test_reads_valid_manifest(self, tmp_path):
        work_dir = tmp_path
        plugin_id = "plugin-abc"
        job_dir = work_dir / plugin_id
        job_dir.mkdir()

        manifest_data = {
            "status": "succeeded",
            "output_keys": ["out/x.npy"],
            "embeddings": [[0.1, 0.2]],
            "checksums": ["hash1"],
            "metadata": [{"k": "v"}],
        }
        (job_dir / "result_manifest.json").write_text(json.dumps(manifest_data))

        result = UserWorkflowRunner._read_step_manifest(work_dir, plugin_id)
        assert result is not None
        assert result["output_keys"] == ["out/x.npy"]
        assert result["embeddings"] == [[0.1, 0.2]]

    def test_returns_none_for_missing_manifest(self, tmp_path):
        result = UserWorkflowRunner._read_step_manifest(tmp_path, "nonexistent")
        assert result is None

    def test_returns_none_for_corrupt_json(self, tmp_path):
        work_dir = tmp_path
        plugin_id = "plugin-bad"
        job_dir = work_dir / plugin_id
        job_dir.mkdir()
        (job_dir / "result_manifest.json").write_text("not json{{{")

        result = UserWorkflowRunner._read_step_manifest(work_dir, plugin_id)
        assert result is None


# =========================================================================
# 5. user_workflow — Slurm mode uses asyncio.run
# =========================================================================


class TestSlurmModeAsync:
    """Verify that _run_slurm properly awaits runner.execute()."""

    def test_slurm_mode_calls_asyncio_run(self):
        wf = _make_wf_def(execution_mode=ExecutionMode.SLURM)
        uw = UserWorkflow.from_workflow_definition(
            wf,
            env_vars={"MY_VAR": "1"},
            user_id="alice",
        )
        runner = UserWorkflowRunner(uw)
        mock_s3 = MagicMock()

        # Create a mock PluginResult
        mock_result = PluginResult(
            plugin_id="plugin-test123",
            status=PluginStatus.SUCCEEDED,
            exit_code=0,
        )

        # Create a manifest file
        with tempfile.TemporaryDirectory() as td:
            work_dir = Path(td) / "plugin_jobs"
            work_dir.mkdir()
            manifest_dir = work_dir / "plugin-test123"
            manifest_dir.mkdir()
            manifest = {
                "status": "succeeded",
                "output_keys": ["out/result.npy"],
                "embeddings": [[1.0, 2.0]],
                "checksums": ["h1"],
                "metadata": [{"done": True}],
            }
            (manifest_dir / "result_manifest.json").write_text(json.dumps(manifest))

            with (
                patch("umrs.plugins.user_workflow._resolve_source", return_value=Path("/tmp/src")),
                patch("umrs.plugins.slurm_runner.SlurmPluginRunner") as MockSlurm,
                patch("asyncio.run", return_value=mock_result) as mock_asyncio_run,
            ):
                mock_runner_instance = MockSlurm.return_value
                mock_runner_instance.work_dir = work_dir
                mock_runner_instance.execute.return_value = "coroutine_placeholder"

                result = runner._run_slurm(mock_s3, ["key1.npy"], "in-bkt", "out-bkt")

                # Verify asyncio.run was called (not raw coroutine)
                mock_asyncio_run.assert_called_once()

            # Result should have manifest data
            embeddings, checksums, metadata, saved_keys = result
            assert embeddings == [[1.0, 2.0]]
            assert checksums == ["h1"]
            assert metadata == [{"done": True}]
            assert saved_keys == ["out/result.npy"]


class TestK8sModeManifestHandling:
    """Verify K8s mode does not require a Slurm-style work_dir."""

    def test_k8s_mode_skips_manifest_when_runner_has_no_work_dir(self):
        wf = _make_wf_def(execution_mode=ExecutionMode.K8S)
        uw = UserWorkflow.from_workflow_definition(
            wf,
            env_vars={"MY_VAR": "1"},
            user_id="alice",
        )
        runner = UserWorkflowRunner(uw)
        mock_s3 = MagicMock()

        mock_result = PluginResult(
            plugin_id="plugin-test123",
            status=PluginStatus.SUCCEEDED,
            exit_code=0,
        )

        class _FakeK8sRunner:
            def execute(self, spec):
                return "coroutine_placeholder"

        with (
            patch(
                "umrs.plugins.user_workflow._resolve_source",
                return_value=Path("/tmp/src"),
            ),
            patch(
                "umrs.plugins.k8s_runner.K8sPluginRunner",
                return_value=_FakeK8sRunner(),
            ),
            patch(
                "asyncio.run",
                return_value=mock_result,
            ) as mock_asyncio_run,
        ):
            result = runner._run_k8s(
                mock_s3,
                ["key1.npy"],
                "in-bkt",
                "out-bkt",
            )

            mock_asyncio_run.assert_called_once()

        embeddings, checksums, metadata, saved_keys = result
        assert embeddings == [None]
        assert checksums == []
        assert metadata == []
        assert saved_keys == [None]
