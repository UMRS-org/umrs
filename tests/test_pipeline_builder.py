"""Tests for B8 — Pipeline Builder API."""

import json

import pytest
from umrs.pipeline_sdk.builder import (
    PipelineBuilder,
    PipelineBuilderError,
    StepBuilder,
)


class TestStepBuilder:
    def test_basic_creation(self):
        sb = StepBuilder("embed", "steps/embed.py:Embed")
        d = sb.to_dict()
        assert d["name"] == "embed"
        assert d["entrypoint"] == "steps/embed.py:Embed"
        assert d["depends_on"] == []  # explicit: the upload contract needs every edge

    def test_with_config(self):
        sb = StepBuilder("s1", "mod:Class")
        sb.with_config("batch_size", 32)
        d = sb.to_dict()
        assert d["config"]["batch_size"] == 32

    def test_with_retry(self):
        sb = StepBuilder("s1", "mod:Class")
        sb.with_retry(max_retries=5, delay=10)
        d = sb.to_dict()
        assert d["max_retries"] == 5
        assert d["retry_delay"] == 10

    def test_extending(self):
        sb = StepBuilder("s1", "mod:Class", depends_on=["preprocess"])
        d = sb.to_dict()
        assert "preprocess" in d["depends_on"]


class TestPipelineBuilder:
    def test_minimal_build(self):
        wf = PipelineBuilder("simple").set_description("One step.").add_step("step1", "mod:Step1").build()
        assert wf["name"] == "simple"
        assert len(wf["steps"]) == 1
        assert wf["steps"][0]["name"] == "step1"

    def test_multi_step_with_deps(self):
        wf = (
            PipelineBuilder("multi")
            .set_description("Load, process, save.")
            .add_step("load", "mod:Load")
            .add_step("process", "mod:Process", depends_on=["load"])
            .add_step("save", "mod:Save", depends_on=["process"])
            .build()
        )
        assert len(wf["steps"]) == 3
        proc_step = [s for s in wf["steps"] if s["name"] == "process"][0]
        assert "load" in proc_step["depends_on"]

    def test_set_execution_mode(self):
        wf = PipelineBuilder("p1").set_description("Slurm run.").add_step("s1", "m:S").set_execution_mode("SLURM").build()
        assert wf["execution_mode"] == "slurm"  # the enum value the runner reads

    def test_set_source(self):
        wf = (
            PipelineBuilder("p1")
            .set_description("Local source.")
            .add_step("s1", "m:S")
            .set_source("/path/to/steps", source_type="LOCAL_PATH")
            .build()
        )
        assert wf["source"]["path"] == "/path/to/steps"
        assert wf["source"]["type"] == "local_path"

    def test_set_version_description(self):
        wf = (
            PipelineBuilder("p1")
            .add_step("s1", "m:S")
            .set_version("2.0")
            .set_description("A test pipeline")
            .build()
        )
        assert wf["version"] == "2.0"
        assert wf["description"] == "A test pipeline"

    def test_set_output(self):
        wf = (
            PipelineBuilder("p1")
            .set_description("Vector output.")
            .add_step("s1", "m:S")
            .set_output(return_type="vector", embedding_vector_size=768)
            .build()
        )
        assert wf["output"]["return_type"] == "vector"
        assert wf["output"]["embedding_vector_size"] == 768

    def test_validate_no_steps_raises(self):
        builder = PipelineBuilder("empty")
        with pytest.raises(PipelineBuilderError, match="at least one step"):
            builder.build()

    def test_validate_duplicate_step_names(self):
        builder = PipelineBuilder("dup").add_step("s1", "m:S").add_step("s1", "m:S2")
        with pytest.raises(PipelineBuilderError, match="Duplicate"):
            builder.build()

    def test_validate_invalid_dependency(self):
        builder = PipelineBuilder("bad-dep").add_step("s1", "m:S", depends_on=["nonexistent"])
        with pytest.raises(PipelineBuilderError, match="nonexistent"):
            builder.build()

    def test_validate_cycle_detection(self):
        builder = (
            PipelineBuilder("cycle")
            .add_step("a", "m:A", depends_on=["b"])
            .add_step("b", "m:B", depends_on=["a"])
        )
        with pytest.raises(PipelineBuilderError, match="[Cc]ycle"):
            builder.build()

    def test_to_json(self):
        builder = PipelineBuilder("json-test").set_description("JSON round trip.").add_step("s1", "m:S")
        j = builder.to_json()
        parsed = json.loads(j)
        assert parsed["name"] == "json-test"

    def test_fluent_chaining(self):
        """Verify all fluent methods return the builder."""
        b = PipelineBuilder("chain")
        result = (
            b.add_step("s1", "m:S")
            .set_execution_mode("IN_PROCESS")
            .set_source(".")
            .set_version("1.0")
            .set_description("test")
            .add_requirement("numpy")
            .set_env("KEY", "VAL")
            .set_resources(gpu_count=1)
            .set_output("file")
        )
        assert result is b

    def test_complex_dag(self):
        """Build a realistic multi-branch DAG."""
        wf = (
            PipelineBuilder("complex")
            .set_description("Ingest, embed text and images, merge, export.")
            .add_step("ingest", "steps:Ingest")
            .add_step("text_embed", "steps:TextEmbed", depends_on=["ingest"])
            .add_step("image_embed", "steps:ImageEmbed", depends_on=["ingest"])
            .add_step("merge", "steps:Merge", depends_on=["text_embed", "image_embed"])
            .add_step("export", "steps:Export", depends_on=["merge"])
            .set_execution_mode("IN_PROCESS")
            .build()
        )
        assert len(wf["steps"]) == 5
        merge = [s for s in wf["steps"] if s["name"] == "merge"][0]
        assert set(merge["depends_on"]) == {"text_embed", "image_embed"}


class TestUploadContract:
    """2026-10-02: what the builder emits meets the upload contract
    (r4_contracts.workflow_package): a description and every step's
    depends_on, [] for a step that starts the run."""

    def test_description_is_required(self):
        builder = PipelineBuilder("undescribed").add_step("s1", "m:S")
        with pytest.raises(PipelineBuilderError, match="description is required"):
            builder.build()

    def test_a_built_workflow_passes_the_upload_check(self):
        from r4_contracts.workflow_package import errors_of, workflow_package_problems

        wf = (
            PipelineBuilder("fan")
            .set_description("Ingest once, embed twice, merge.")
            .add_step("ingest", "steps:Ingest")
            .add_step("a", "steps:A", depends_on=["ingest"])
            .add_step("b", "steps:B", depends_on=["ingest"])
            .add_step("merge", "steps:Merge", depends_on=["a", "b"])
            .build()
        )
        assert wf["steps"][0]["depends_on"] == []
        assert errors_of(workflow_package_problems(wf)) == []

    def test_the_built_workflow_parses_as_the_runner_reads_it(self):
        from umrs.pipeline_sdk.workflow import ExecutionMode, SourceType, WorkflowDefinition

        wf = (
            PipelineBuilder("p")
            .set_description("One step on Slurm.")
            .set_execution_mode("slurm")
            .add_step("s1", "m:S")
            .build()
        )
        parsed = WorkflowDefinition.from_dict(wf)
        assert parsed.execution_mode is ExecutionMode.SLURM
        assert parsed.source.type is SourceType.LOCAL_PATH
