"""
Phase 3 tests — Unified Plugin + Step model.

Covers:
- ExecutionMode enum serialization / backward compat
- UserWorkflow dataclass (creation, to_dict, from_dict, round-trip)
- UserWorkflowRunner in-process mode (mock S3 + dummy Steps)
- step_to_plugin_spec adapter
- WorkflowPipeline delegation for non-IN_PROCESS modes
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.pipeline_sdk.workflow import (
    ExecutionMode,
    OutputSpec,
    SourceSpec,
    SourceType,
    StepSpec,
    WorkflowDefinition,
)
from umrs.plugins.step_adapter import step_to_plugin_spec
from umrs.plugins.user_workflow import UserWorkflow, UserWorkflowRunner

# ── Helpers ──────────────────────────────────────────────────────────────


def _source_spec(path: str = "/tmp/test_scripts") -> SourceSpec:
    return SourceSpec(type=SourceType.LOCAL_PATH, path=path)


def _make_step_spec(name: str = "step_a", entrypoint: str = "mod:StepA"):
    return StepSpec(name=name, entrypoint=entrypoint, config={"k": "v"})


def _make_wf_def(**overrides) -> WorkflowDefinition:
    defaults = dict(
        name="test_wf",
        source=_source_spec(),
        steps=[_make_step_spec()],
        version="1.0",
        description="test",
        output=OutputSpec(return_type="vector", embedding_vector_size=128),
    )
    defaults.update(overrides)
    return WorkflowDefinition(**defaults)


class DummyStep(Step):
    """Minimal step for testing the runner."""

    name = "dummy"

    def setup(self, config):
        pass

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(
            output_keys=ctx.input_keys,
            embeddings=[[0.1, 0.2]] * len(ctx.input_keys),
            checksums=["abc123"] * len(ctx.input_keys),
            metadata=[{"src": k} for k in ctx.input_keys],
        )

    def teardown(self):
        pass


# =========================================================================
# 1. ExecutionMode serialization
# =========================================================================


class TestExecutionMode:
    def test_enum_values(self):
        assert ExecutionMode.IN_PROCESS.value == "in_process"
        assert ExecutionMode.SLURM.value == "slurm"
        assert ExecutionMode.K8S.value == "k8s"
        assert ExecutionMode.SUBPROCESS.value == "subprocess"

    def test_round_trip(self):
        for mode in ExecutionMode:
            assert ExecutionMode(mode.value) is mode

    def test_default_in_workflow_definition(self):
        wf = _make_wf_def()
        assert wf.execution_mode == ExecutionMode.IN_PROCESS

    def test_to_dict_omits_in_process(self):
        wf = _make_wf_def()
        d = wf.to_dict()
        assert "execution_mode" not in d

    def test_to_dict_includes_non_default(self):
        wf = _make_wf_def(execution_mode=ExecutionMode.SLURM)
        d = wf.to_dict()
        assert d["execution_mode"] == "slurm"

    def test_from_dict_default(self):
        wf = _make_wf_def()
        d = wf.to_dict()
        wf2 = WorkflowDefinition.from_dict(d)
        assert wf2.execution_mode == ExecutionMode.IN_PROCESS

    def test_from_dict_explicit(self):
        wf = _make_wf_def(execution_mode=ExecutionMode.K8S)
        d = wf.to_dict()
        wf2 = WorkflowDefinition.from_dict(d)
        assert wf2.execution_mode == ExecutionMode.K8S


# =========================================================================
# 2. UserWorkflow dataclass
# =========================================================================


class TestUserWorkflow:
    def test_from_workflow_definition_defaults(self):
        wf = _make_wf_def()
        uw = UserWorkflow.from_workflow_definition(wf)
        assert uw.name == wf.name
        assert uw.execution_mode == ExecutionMode.IN_PROCESS
        assert uw.env_vars == {}
        assert uw.output_destinations == []
        assert uw.user_id is None
        assert uw.project_id is None

    def test_from_workflow_definition_with_extras(self):
        wf = _make_wf_def(execution_mode=ExecutionMode.SLURM)
        uw = UserWorkflow.from_workflow_definition(
            wf,
            env_vars={"CUDA": "1"},
            output_destinations=["s3://results"],
            user_id="alice",
            project_id="proj42",
        )
        assert uw.execution_mode == ExecutionMode.SLURM
        assert uw.env_vars == {"CUDA": "1"}
        assert uw.output_destinations == ["s3://results"]
        assert uw.user_id == "alice"
        assert uw.project_id == "proj42"

    def test_to_workflow_definition_round_trip(self):
        wf = _make_wf_def(execution_mode=ExecutionMode.K8S)
        uw = UserWorkflow.from_workflow_definition(wf, user_id="bob")
        wf2 = uw.to_workflow_definition()
        assert wf2.name == wf.name
        assert wf2.execution_mode == ExecutionMode.K8S
        # PluginSpec-like fields are stripped
        assert not hasattr(wf2, "user_id") or wf2.__class__ is WorkflowDefinition

    def test_to_dict_round_trip(self):
        wf = _make_wf_def(execution_mode=ExecutionMode.SLURM)
        uw = UserWorkflow.from_workflow_definition(
            wf,
            env_vars={"A": "1"},
            user_id="carol",
        )
        d = uw.to_dict()
        assert d["execution_mode"] == "slurm"
        assert d["env_vars"] == {"A": "1"}
        assert d["user_id"] == "carol"

        uw2 = UserWorkflow.from_dict(d)
        assert uw2.execution_mode == ExecutionMode.SLURM
        assert uw2.env_vars == {"A": "1"}
        assert uw2.user_id == "carol"

    def test_to_dict_omits_empty_optional_fields(self):
        wf = _make_wf_def()
        uw = UserWorkflow.from_workflow_definition(wf)
        d = uw.to_dict()
        assert "env_vars" not in d
        assert "output_destinations" not in d
        assert "user_id" not in d
        assert "project_id" not in d


# =========================================================================
# 3. UserWorkflowRunner — in-process mode
# =========================================================================


class TestUserWorkflowRunnerInProcess:
    """Test in-process execution with mock S3 and dummy steps."""

    @pytest.fixture()
    def mock_s3(self):
        s3 = MagicMock()
        s3.head_object.return_value = {
            "ETag": '"abc123"',
            "Metadata": {"src": "test"},
        }
        # No staging objects to clean up
        paginator = MagicMock()
        paginator.paginate.return_value = []
        s3.get_paginator.return_value = paginator
        return s3

    @pytest.fixture()
    def runner(self, mock_s3):
        wf = _make_wf_def()
        uw = UserWorkflow.from_workflow_definition(wf)
        r = UserWorkflowRunner(uw, s3=mock_s3)
        # Inject DummyStep directly to avoid script loading
        r._step_instances = [DummyStep()]
        return r

    def test_run_returns_tuple_of_four(self, runner, mock_s3):
        result = runner.run(mock_s3, ["key1.npy"], "input-bkt", "output-bkt")
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_run_embeddings(self, runner, mock_s3):
        embeddings, checksums, metadata, saved_keys = runner.run(
            mock_s3,
            ["key1.npy", "key2.npy"],
            "input-bkt",
            "output-bkt",
        )
        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2]
        assert len(checksums) == 2
        assert checksums[0] == "abc123"
        assert len(metadata) == 2
        assert len(saved_keys) == 2

    def test_run_single_key(self, runner, mock_s3):
        result = runner.run(mock_s3, ["single.bin"], "in", "out")
        assert len(result[0]) == 1

    def test_mode_is_in_process(self, runner):
        assert runner.workflow.execution_mode == ExecutionMode.IN_PROCESS

    def test_subprocess_raises(self, mock_s3):
        wf = _make_wf_def(execution_mode=ExecutionMode.SUBPROCESS)
        uw = UserWorkflow.from_workflow_definition(wf)
        r = UserWorkflowRunner(uw)
        with pytest.raises(NotImplementedError, match="Subprocess"):
            r.run(mock_s3, ["k"], "in", "out")


# =========================================================================
# 4. step_to_plugin_spec adapter
# =========================================================================


class TestStepToPluginSpec:
    def test_basic_fields(self):
        spec = step_to_plugin_spec(
            step_name="embed",
            entrypoint="my_mod:EmbedStep",
            script_dir="/scripts",
            step_config={"dim": 128},
            input_bucket="raw",
            input_keys=["a.npy", "b.npy"],
            output_bucket="processed",
            run_id="run123",
        )
        assert spec.source_uri == "/scripts"
        assert spec.entrypoint == "my_mod:EmbedStep"
        assert spec.env_vars["UMRS_STEP_NAME"] == "embed"
        assert spec.env_vars["UMRS_STEP_ENTRYPOINT"] == "my_mod:EmbedStep"
        assert json.loads(spec.env_vars["UMRS_STEP_CONFIG"]) == {"dim": 128}
        assert json.loads(spec.env_vars["UMRS_INPUT_KEYS"]) == ["a.npy", "b.npy"]
        assert spec.env_vars["UMRS_RUN_ID"] == "run123"
        assert spec.metadata["adapter"] == "StepPluginAdapter"

    def test_resource_limits(self):
        spec = step_to_plugin_spec(
            step_name="train",
            entrypoint="m:S",
            script_dir="/s",
            step_config={},
            input_bucket="b",
            input_keys=[],
            output_bucket="b",
            run_id="r",
            cpu_limit="4",
            memory_limit="8Gi",
            gpu_count=2,
            timeout_seconds=7200,
        )
        assert spec.cpu_limit == "4"
        assert spec.memory_limit == "8Gi"
        assert spec.gpu_count == 2
        assert spec.timeout_seconds == 7200

    def test_extra_env_vars_merged(self):
        spec = step_to_plugin_spec(
            step_name="x",
            entrypoint="m:C",
            script_dir="/s",
            step_config={},
            input_bucket="b",
            input_keys=[],
            output_bucket="b",
            run_id="r",
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1"},
        )
        assert spec.env_vars["CUDA_VISIBLE_DEVICES"] == "0,1"
        # Core env vars still present
        assert "UMRS_STEP_NAME" in spec.env_vars


# =========================================================================
# 5. WorkflowPipeline delegation
# =========================================================================


class TestWorkflowPipelineDelegation:
    """Verify that non-IN_PROCESS modes delegate to UserWorkflowRunner."""

    def test_run_as_embeddings_delegates_for_slurm(self):
        """run_as_embeddings should call UserWorkflowRunner for SLURM."""
        from umrs.pipeline_sdk.workflow import WorkflowPipeline

        wf = _make_wf_def(execution_mode=ExecutionMode.SLURM)
        pipeline = WorkflowPipeline.__new__(WorkflowPipeline)
        pipeline.workflow = wf
        pipeline._s3 = MagicMock()
        pipeline._s3_keys = ["k1"]
        pipeline._input_bucket = "in"
        pipeline._output_bucket = "out"
        pipeline._step_instances = []
        pipeline._log = MagicMock()

        with patch(
            "umrs.plugins.user_workflow.UserWorkflowRunner.run",
            return_value=([1], ["cs"], [{}], [None]),
        ) as mock_run:
            result = pipeline.run_as_embeddings(
                pipeline._s3,
                ["k1"],
                "in",
                "out",
            )
            mock_run.assert_called_once()
            assert result == ([1], ["cs"], [{}], [None])

    def test_run_as_embeddings_in_process_no_delegation(self):
        """IN_PROCESS should NOT delegate — runs original step chain."""
        from umrs.pipeline_sdk.workflow import WorkflowPipeline

        wf = _make_wf_def(execution_mode=ExecutionMode.IN_PROCESS)
        pipeline = WorkflowPipeline.__new__(WorkflowPipeline)
        pipeline.workflow = wf
        pipeline._s3 = MagicMock()
        pipeline._s3_keys = ["k1"]
        pipeline._input_bucket = "in"
        pipeline._output_bucket = "out"
        pipeline._step_instances = [DummyStep()]
        pipeline._log = MagicMock()

        # Mock S3 helpers
        pipeline._s3.head_object.return_value = {
            "ETag": '"x"',
            "Metadata": {},
        }
        paginator = MagicMock()
        paginator.paginate.return_value = []
        pipeline._s3.get_paginator.return_value = paginator

        result = pipeline.run_as_embeddings(
            pipeline._s3,
            ["k1"],
            "in",
            "out",
        )
        assert len(result) == 4
        assert result[0] == [[0.1, 0.2]]


# =========================================================================
# 6. Backward compatibility
# =========================================================================


class TestBackwardCompat:
    """Existing workflow.json without execution_mode should still work."""

    def test_old_workflow_json_round_trip(self):
        data = {
            "name": "legacy",
            "source": {"type": "local_path", "path": "/old/scripts"},
            "steps": [{"name": "s", "entrypoint": "m:C"}],
        }
        wf = WorkflowDefinition.from_dict(data)
        assert wf.execution_mode == ExecutionMode.IN_PROCESS
        d = wf.to_dict()
        assert "execution_mode" not in d

    def test_user_workflow_from_old_dict(self):
        data = {
            "name": "legacy",
            "source": {"type": "local_path", "path": "/old/scripts"},
            "steps": [{"name": "s", "entrypoint": "m:C"}],
        }
        uw = UserWorkflow.from_dict(data)
        assert uw.execution_mode == ExecutionMode.IN_PROCESS
        assert uw.env_vars == {}
        assert uw.user_id is None
