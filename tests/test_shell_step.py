"""D-08 leg: ShellStep — command over materialized inputs, declared outputs."""

from __future__ import annotations

import pytest
from umrs.pipeline_sdk.step import StepContext
from umrs.pipeline_sdk.templates.shell_step import ShellStep


class _FakeS3:
    def __init__(self, store):
        self._store = store

    def get_object(self, Bucket, Key):
        class _Body:
            def read(self_inner, _k=Key, _b=Bucket):
                return self._store[_k]

        return {"Body": _Body()}

    def put_object(self, Bucket=None, Key=None, Body=None, **_kw):
        self._store[Key] = Body


def _ctx(store, *, run_id="r1", parent=False):
    return StepContext(
        s3=_FakeS3(store),
        input_bucket="in",
        input_keys=["reads/a.txt"],
        output_bucket="out",
        config={},
        run_id=run_id,
        parent_output_bucket="pin" if parent else "",
    )


def test_shell_command_runs_and_uploads_declared_outputs():
    store = {"reads/a.txt": b"hello"}
    ctx = _ctx(store)
    step = ShellStep()
    step.setup(
        {
            "command": "cat $UMRS_INPUTS_DIR/reads/a.txt > $UMRS_OUTPUTS_DIR/out/copy.txt",
            "output_globs": ["outputs/out/copy.txt"],
        }
    )
    result = step.run(ctx)
    assert result.output_keys == ["shell/r1/out/copy.txt"]
    assert store["shell/r1/out/copy.txt"] == b"hello"
    assert result.extra["step"] == "shell"


def test_child_step_reads_parent_artifacts():
    store = {"reads/a.txt": b"from-parent"}
    ctx = _ctx(store, parent=True)
    step = ShellStep()
    step.setup(
        {
            "command": "cat $UMRS_INPUTS_DIR/reads/a.txt > $UMRS_OUTPUTS_DIR/out/x.txt",
            "output_globs": ["outputs/out/x.txt"],
        }
    )
    result = step.run(ctx)
    assert result.output_keys and store[result.output_keys[0]] == b"from-parent"


def test_failed_command_raises_with_stderr():
    ctx = StepContext(
        s3=_FakeS3({}), input_bucket="in", input_keys=[], output_bucket="out", run_id="r3"
    )
    step = ShellStep()
    step.setup({"command": "echo boom >&2; exit 3", "output_globs": ["outputs/*"]})
    with pytest.raises(RuntimeError, match="exit 3"):
        step.run(ctx)


def test_setup_refuses_empty_command_and_globs():
    step = ShellStep()
    with pytest.raises(ValueError, match="command"):
        step.setup({"command": "", "output_globs": ["a"]})
    with pytest.raises(ValueError, match="output_globs"):
        step.setup({"command": "true", "output_globs": []})


def test_input_aliases_materialize_stable_names():
    """DAG contract: run-scoped upstream keys get stable aliases (importers)."""
    store = {"shell/r1/mapped/a.bam": b"upstream"}
    ctx = StepContext(
        s3=_FakeS3(store),
        input_bucket="in",
        input_keys=["shell/r1/mapped/a.bam"],
        output_bucket="out",
        config={},
        run_id="r2",
        parent_output_bucket="pin",
    )
    step = ShellStep()
    step.setup(
        {
            "command": "cat $UMRS_INPUTS_DIR/mapped/a.bam > $UMRS_OUTPUTS_DIR/out/x",
            "output_globs": ["outputs/out/x"],
            "input_aliases": {"mapped/a.bam": "**/mapped/a.bam"},
        }
    )
    result = step.run(ctx)
    assert store[result.output_keys[0]] == b"upstream"


def test_alias_with_no_match_refuses_by_name():
    ctx = StepContext(
        s3=_FakeS3({}), input_bucket="in", input_keys=[], output_bucket="out", run_id="r3"
    )
    step = ShellStep()
    step.setup(
        {
            "command": "true",
            "output_globs": ["outputs/x"],
            "input_aliases": {"a.txt": "**/a.txt"},
        }
    )
    with pytest.raises(RuntimeError, match="input alias 'a.txt' matched nothing"):
        step.run(ctx)
