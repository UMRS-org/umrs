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
        assert "depends_on" not in d  # empty deps omitted

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
        wf = (
            PipelineBuilder("simple")
            .add_step("step1", "mod:Step1")
            .build()
        )
        assert wf["name"] == "simple"
        assert len(wf["steps"]) == 1
        assert wf["steps"][0]["name"] == "step1"

    def test_multi_step_with_deps(self):
        wf = (
            PipelineBuilder("multi")
            .add_step("load", "mod:Load")
            .add_step("process", "mod:Process", depends_on=["load"])
            .add_step("save", "mod:Save", depends_on=["process"])
            .build()
        )
        assert len(wf["steps"]) == 3
        proc_step = [s for s in wf["steps"] if s["name"] == "process"][0]
        assert "load" in proc_step["depends_on"]

    def test_set_execution_mode(self):
        wf = (
            PipelineBuilder("p1")
            .add_step("s1", "m:S")
            .set_execution_mode("SLURM")
            .build()
        )
        assert wf["execution_mode"] == "SLURM"

    def test_set_source(self):
        wf = (
            PipelineBuilder("p1")
            .add_step("s1", "m:S")
            .set_source("/path/to/steps", source_type="LOCAL_PATH")
            .build()
        )
        assert wf["source"]["path"] == "/path/to/steps"
        assert wf["source"]["type"] == "LOCAL_PATH"

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
        builder = (
            PipelineBuilder("dup")
            .add_step("s1", "m:S")
            .add_step("s1", "m:S2")
        )
        with pytest.raises(PipelineBuilderError, match="Duplicate"):
            builder.build()

    def test_validate_invalid_dependency(self):
        builder = (
            PipelineBuilder("bad-dep")
            .add_step("s1", "m:S", depends_on=["nonexistent"])
        )
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
        builder = (
            PipelineBuilder("json-test")
            .add_step("s1", "m:S")
        )
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
