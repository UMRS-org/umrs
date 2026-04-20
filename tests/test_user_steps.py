"""
Tests for the unified Pipeline SDK — Step + WorkflowPipeline.

Covers:
  - WorkflowDefinition parsing & validation
  - Script loading & step import
  - WorkflowPipeline step chaining & S3 I/O
  - Step ABC contract enforcement
  - StepContext S3 helpers
  - EmbeddingAdapter contract
"""

from __future__ import annotations

import io
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.pipeline_sdk.workflow import (
    OutputSpec,
    ResourceSpec,
    ScriptLoadError,
    SourceSpec,
    SourceType,
    StepSpec,
    WorkflowDefinition,
    WorkflowPipeline,
    load_steps,
    load_workflow,
)
from umrs.pipeline_sdk.adapter import EmbeddingAdapter


# =========================================================================
# Fixtures
# =========================================================================

def _minimal_workflow_dict(**overrides: Any) -> Dict[str, Any]:
    """Return the smallest valid workflow.json dict."""
    d = {
        "name": "test-pipeline",
        "source": {"type": "local_path", "path": "/tmp/fake"},
        "steps": [
            {"name": "step1", "entrypoint": "my_mod:MyStep", "config": {"k": 1}},
        ],
    }
    d.update(overrides)
    return d


class PassthroughStep(Step):
    """Minimal concrete step: copies input keys to output keys."""
    name = "passthrough"

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(output_keys=list(ctx.input_keys))


class UppercaseStep(Step):
    """Reads text from S3, uppercases it, writes to output bucket."""
    name = "uppercase"

    def setup(self, config: dict) -> None:
        self.suffix = config.get("suffix", "_upper")

    def run(self, ctx: StepContext) -> StepResult:
        out_keys = []
        for key in ctx.input_keys:
            data = ctx.download_bytes(key)
            out_key = f"{key}{self.suffix}"
            ctx.upload_bytes(out_key, data.upper())
            out_keys.append(out_key)
        return StepResult(output_keys=out_keys)


class EmbeddingStep(Step):
    """Produces dummy embeddings."""
    name = "embed"

    def run(self, ctx: StepContext) -> StepResult:
        n = len(ctx.input_keys)
        return StepResult(
            output_keys=ctx.input_keys,
            embeddings=np.ones((n, 4), dtype=np.float32),
            checksums=["abc"] * n,
            metadata=[{"step": "embed"}] * n,
        )


# =========================================================================
# Mock S3
# =========================================================================

def _mock_s3():
    """Build a mock S3 client with basic get/put/head support."""
    store: Dict[str, bytes] = {}
    s3 = MagicMock()

    def put_object(Bucket, Key, Body, **kw):
        store[f"{Bucket}/{Key}"] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(Bucket, Key):
        full = f"{Bucket}/{Key}"
        if full not in store:
            raise Exception(f"NoSuchKey: {full}")
        data = store[full]
        body = MagicMock()
        body.read.return_value = data
        return {"Body": body, "ETag": '"abc123"', "ContentLength": len(data),
                "Metadata": {}}

    def head_object(Bucket, Key):
        full = f"{Bucket}/{Key}"
        if full not in store:
            raise Exception(f"NoSuchKey: {full}")
        return {"ETag": '"abc123"', "ContentLength": len(store[full]),
                "Metadata": {}}

    s3.put_object = MagicMock(side_effect=put_object)
    s3.get_object = MagicMock(side_effect=get_object)
    s3.head_object = MagicMock(side_effect=head_object)
    s3._store = store  # expose for assertions
    return s3


# =========================================================================
# WorkflowDefinition
# =========================================================================

class TestWorkflowDefinition:

    def test_from_dict_minimal(self):
        wf = WorkflowDefinition.from_dict(_minimal_workflow_dict())
        assert wf.name == "test-pipeline"
        assert len(wf.steps) == 1
        assert wf.steps[0].entrypoint == "my_mod:MyStep"
        assert wf.version == "1.0"

    def test_from_dict_full(self):
        d = _minimal_workflow_dict(
            version="2.0",
            description="a pipeline",
            requirements=["numpy"],
            env={"FOO": "bar"},
            resources={"gpu_count": 1, "memory_limit": "4Gi"},
            output={"return_type": "vector", "embedding_vector_size": 768},
        )
        wf = WorkflowDefinition.from_dict(d)
        assert wf.version == "2.0"
        assert wf.resources.gpu_count == 1
        assert wf.output.embedding_vector_size == 768

    def test_roundtrip(self):
        wf = WorkflowDefinition.from_dict(_minimal_workflow_dict())
        d2 = wf.to_dict()
        wf2 = WorkflowDefinition.from_dict(d2)
        assert wf2.name == wf.name
        assert len(wf2.steps) == len(wf.steps)

    def test_validate_empty_name(self):
        d = _minimal_workflow_dict(name="")
        wf = WorkflowDefinition.from_dict(d)
        with pytest.raises(ValueError, match="name is required"):
            wf.validate()

    def test_validate_no_steps(self):
        d = _minimal_workflow_dict(steps=[])
        wf = WorkflowDefinition.from_dict(d)
        with pytest.raises(ValueError, match="at least one step"):
            wf.validate()

    def test_validate_duplicate_step_names(self):
        d = _minimal_workflow_dict(steps=[
            {"name": "dup", "entrypoint": "a:B"},
            {"name": "dup", "entrypoint": "c:D"},
        ])
        wf = WorkflowDefinition.from_dict(d)
        with pytest.raises(ValueError, match="Duplicate step name"):
            wf.validate()

    def test_validate_bad_entrypoint(self):
        d = _minimal_workflow_dict(steps=[
            {"name": "s1", "entrypoint": "no_colon_here"},
        ])
        wf = WorkflowDefinition.from_dict(d)
        with pytest.raises(ValueError, match="module:ClassName"):
            wf.validate()

    def test_from_file(self, tmp_path: Path):
        d = _minimal_workflow_dict()
        d["source"]["path"] = str(tmp_path)
        p = tmp_path / "workflow.json"
        p.write_text(json.dumps(d))
        wf = WorkflowDefinition.from_file(p)
        assert wf.name == "test-pipeline"

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            WorkflowDefinition.from_file("/nonexistent/workflow.json")

    def test_from_dict_execution_parallel_roots(self):
        wf = WorkflowDefinition.from_dict(_minimal_workflow_dict(
            steps=[
                {"name": "segment", "entrypoint": "segment:SegmentStep"},
                {"name": "features", "entrypoint": "features:FeatureStep"},
                {"name": "thumbnails", "entrypoint": "thumbs:ThumbnailStep"},
            ],
            execution={"parallel": ["segment", "features", "thumbnails"]},
        ))

        assert [step.depends_on for step in wf.steps] == [[], [], []]

    def test_from_dict_execution_parallel_fans_back_into_next_step(self):
        wf = WorkflowDefinition.from_dict(_minimal_workflow_dict(
            steps=[
                {"name": "load", "entrypoint": "load:LoadStep"},
                {"name": "segment", "entrypoint": "segment:SegmentStep"},
                {"name": "features", "entrypoint": "features:FeatureStep"},
                {"name": "merge", "entrypoint": "merge:MergeStep"},
            ],
            execution={"parallel": ["segment", "features"]},
        ))

        assert wf.steps[0].depends_on == []
        assert wf.steps[1].depends_on == ["load"]
        assert wf.steps[2].depends_on == ["load"]
        assert wf.steps[3].depends_on == ["segment", "features"]


# =========================================================================
# SourceSpec validation
# =========================================================================

class TestSourceSpec:

    def test_local_path_valid(self):
        s = SourceSpec(type=SourceType.LOCAL_PATH, path="/tmp/scripts")
        s.validate()  # should not raise

    def test_local_path_relative(self):
        s = SourceSpec(type=SourceType.LOCAL_PATH, path="relative/path")
        with pytest.raises(ValueError, match="absolute"):
            s.validate()

    def test_git_without_ref(self):
        s = SourceSpec(type=SourceType.GIT, path="https://github.com/org/repo.git")
        with pytest.raises(ValueError, match="commit hash"):
            s.validate()

    def test_s3_invalid_prefix(self):
        s = SourceSpec(type=SourceType.S3, path="/not/s3")
        with pytest.raises(ValueError, match="s3://"):
            s.validate()


# =========================================================================
# Script loading (load_workflow + load_steps)
# =========================================================================

class TestScriptLoading:

    def test_load_workflow_local(self, tmp_path: Path):
        d = _minimal_workflow_dict()
        d["source"]["path"] = str(tmp_path)
        p = tmp_path / "workflow.json"
        p.write_text(json.dumps(d))
        wf = load_workflow(str(p))
        assert wf.name == "test-pipeline"

    def test_load_steps_local(self, tmp_path: Path):
        (tmp_path / "my_step.py").write_text(textwrap.dedent("""\
            from umrs.pipeline_sdk import Step, StepContext, StepResult

            class HelloStep(Step):
                name = "hello"
                def setup(self, config):
                    self.greeting = config.get("greeting", "hi")
                def run(self, ctx):
                    return StepResult(output_keys=ctx.input_keys)
        """))

        wf = WorkflowDefinition(
            name="test",
            source=SourceSpec(type=SourceType.LOCAL_PATH, path=str(tmp_path)),
            steps=[StepSpec(name="hello", entrypoint="my_step:HelloStep",
                            config={"greeting": "howdy"})],
        )
        instances = load_steps(wf)
        assert len(instances) == 1
        assert instances[0].name == "hello"
        assert instances[0].greeting == "howdy"  # type: ignore[attr-defined]

    def test_load_steps_bad_entrypoint_format(self, tmp_path: Path):
        wf = WorkflowDefinition(
            name="test",
            source=SourceSpec(type=SourceType.LOCAL_PATH, path=str(tmp_path)),
            steps=[StepSpec(name="s", entrypoint="bad format!")],
        )
        with pytest.raises(ScriptLoadError, match="invalid entrypoint"):
            load_steps(wf)

    def test_load_steps_module_not_found(self, tmp_path: Path):
        wf = WorkflowDefinition(
            name="test",
            source=SourceSpec(type=SourceType.LOCAL_PATH, path=str(tmp_path)),
            steps=[StepSpec(name="s", entrypoint="nonexistent_mod:Cls")],
        )
        with pytest.raises(ScriptLoadError, match="cannot import"):
            load_steps(wf)

    def test_load_steps_class_not_step(self, tmp_path: Path):
        (tmp_path / "bad.py").write_text("class NotAStep:\n    pass\n")
        wf = WorkflowDefinition(
            name="test",
            source=SourceSpec(type=SourceType.LOCAL_PATH, path=str(tmp_path)),
            steps=[StepSpec(name="s", entrypoint="bad:NotAStep")],
        )
        with pytest.raises(ScriptLoadError, match="does not inherit"):
            load_steps(wf)

    def test_load_steps_class_missing(self, tmp_path: Path):
        (tmp_path / "empty_mod.py").write_text("x = 1\n")
        wf = WorkflowDefinition(
            name="test",
            source=SourceSpec(type=SourceType.LOCAL_PATH, path=str(tmp_path)),
            steps=[StepSpec(name="s", entrypoint="empty_mod:Missing")],
        )
        with pytest.raises(ScriptLoadError, match="not found"):
            load_steps(wf)


# =========================================================================
# WorkflowPipeline (step chaining via run_as_embeddings)
# =========================================================================

class TestWorkflowPipeline:

    def _make_pipeline(self, steps: List[Step], **wf_kw) -> WorkflowPipeline:
        step_specs = [StepSpec(name=s.name, entrypoint=f"fake:{s.__class__.__name__}")
                      for s in steps]
        wf = WorkflowDefinition(
            name=wf_kw.get("name", "test-wf"),
            source=SourceSpec(type=SourceType.LOCAL_PATH, path="/tmp"),
            steps=step_specs,
            output=wf_kw.get("output", OutputSpec()),
        )
        return WorkflowPipeline.from_workflow(wf, steps)

    def test_single_passthrough_step(self):
        s3 = _mock_s3()
        s3._store["in-bkt/file1.txt"] = b"hello"
        s3._store["in-bkt/file2.txt"] = b"world"

        pipeline = self._make_pipeline([PassthroughStep()])
        embs, chk, _meta, _saved = pipeline.run_as_embeddings(
            s3, ["file1.txt", "file2.txt"], "in-bkt", "out-bkt"
        )
        assert len(embs) == 2
        assert len(chk) == 2

    def test_single_embedding_step(self):
        s3 = _mock_s3()
        s3._store["bkt/a.txt"] = b"data"

        pipeline = self._make_pipeline([EmbeddingStep()])
        embs, chk, meta, _saved = pipeline.run_as_embeddings(
            s3, ["a.txt"], "bkt", "out"
        )
        assert len(embs) == 1
        np.testing.assert_array_equal(embs[0], np.ones(4, dtype=np.float32))
        assert chk == ["abc"]
        assert meta == [{"step": "embed"}]

    def test_multi_step_chaining(self):
        """Uppercase step chains into passthrough step."""
        s3 = _mock_s3()
        s3._store["in/hello.txt"] = b"hello world"

        upper = UppercaseStep()
        upper.setup({"suffix": "_UP"})
        passth = PassthroughStep()

        pipeline = self._make_pipeline([upper, passth])
        _embs, _chk, _meta, saved = pipeline.run_as_embeddings(
            s3, ["hello.txt"], "in", "out"
        )
        assert len(saved) == 1

    def test_teardown_called(self):
        s3 = _mock_s3()
        s3._store["bkt/f.txt"] = b"x"

        step = PassthroughStep()
        step.teardown = MagicMock()  # type: ignore[assignment]

        pipeline = self._make_pipeline([step])
        pipeline.run_as_embeddings(s3, ["f.txt"], "bkt", "out")
        step.teardown.assert_called_once()


# =========================================================================
# StepContext helpers
# =========================================================================

class TestStepContext:

    def test_download_upload_bytes(self):
        s3 = _mock_s3()
        s3._store["bkt/in.bin"] = b"\x00\x01\x02"

        ctx = StepContext(s3=s3, input_bucket="bkt", input_keys=["in.bin"],
                          output_bucket="out")
        data = ctx.download_bytes("in.bin")
        assert data == b"\x00\x01\x02"

        uri = ctx.upload_bytes("out.bin", b"\x03\x04")
        assert uri == "s3://out/out.bin"
        assert s3._store["out/out.bin"] == b"\x03\x04"

    def test_download_upload_numpy(self):
        arr = np.array([1.0, 2.0, 3.0])
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        npy_bytes = buf.getvalue()

        s3 = _mock_s3()
        s3._store["bkt/vec.npy"] = npy_bytes

        ctx = StepContext(s3=s3, input_bucket="bkt", input_keys=["vec.npy"],
                          output_bucket="out")
        loaded = ctx.download_numpy("vec.npy")
        np.testing.assert_array_equal(loaded, arr)

        ctx.upload_numpy("out.npy", loaded)
        assert "out/out.npy" in s3._store


# =========================================================================
# Step ABC enforcement
# =========================================================================

class TestStepContract:

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Step()  # type: ignore[abstract]

    def test_concrete_step_ok(self):
        step = PassthroughStep()
        assert step.name == "passthrough"


# =========================================================================
# EmbeddingAdapter
# =========================================================================

class TestEmbeddingAdapter:

    def test_adapter_returns_4_tuple(self):
        s3 = _mock_s3()
        s3._store["bkt/a.txt"] = b"data"

        step = EmbeddingStep()
        spec = StepSpec(name=step.name, entrypoint="fake:EmbeddingStep")
        wf = WorkflowDefinition(
            name="adapter-test",
            source=SourceSpec(type=SourceType.LOCAL_PATH, path="/tmp"),
            steps=[spec],
            output=OutputSpec(return_type="vector", embedding_vector_size=4),
        )
        pipeline = WorkflowPipeline.from_workflow(wf, [step])
        adapter = EmbeddingAdapter(pipeline)

        result = adapter.compute_embeddings(s3, ["a.txt"], "bkt", "out")
        assert isinstance(result, tuple)
        assert len(result) == 4

        embs, _, _, _ = result
        assert len(embs) == 1
        np.testing.assert_array_equal(embs[0], np.ones(4, dtype=np.float32))


# =========================================================================
# ResourceSpec.extra field
# =========================================================================

class TestResourceSpecExtra:

    def test_extra_field_default(self):
        rs = ResourceSpec()
        assert rs.extra == {}

    def test_extra_roundtrip(self):
        rs = ResourceSpec(extra={"execution_mode": "slurm", "custom": True})
        d = rs.to_dict()
        assert d["extra"]["execution_mode"] == "slurm"
        rs2 = ResourceSpec.from_dict(d)
        assert rs2.extra == rs.extra

    def test_extra_omitted_when_empty(self):
        rs = ResourceSpec()
        d = rs.to_dict()
        assert "extra" not in d