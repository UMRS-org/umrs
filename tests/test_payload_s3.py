"""Slice 11 tests: object-store payload externalization.

Tests cover:
* S3 key helper layout
* PodmanExecutor S3 upload/download path (mocked S3 + podman)
* Payload size hard cap (setup error)
* Threshold gating (small payloads stay local even with S3 enabled)
* Routing log annotations (payload_mode, bucket, prefix)
* Worker S3 read/write (unit-level, no podman)
* Metadata stripping (__payload_s3__ not leaked to worker)
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.server.step_executor import (
    EXECUTOR_IMAGE_METADATA_KEY,
    EXECUTOR_PAYLOAD_S3_METADATA_KEY,
    PodmanExecutor,
    StepExecutionError,
    StepExecutor,
    _payload_s3_key,
)
from umrs.server.step_router import reset_feedback_store
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_runner import run_workflow_record


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_execution_registry_for_tests()
    reset_feedback_store()
    yield
    reset_execution_registry_for_tests()
    reset_feedback_store()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_ctx(
    *,
    metadata: dict[str, Any] | None = None,
    s3: Any = None,
    run_id: str = "run-1",
) -> StepContext:
    return StepContext(
        s3=s3,
        input_bucket="b",
        input_keys=[],
        output_bucket="b",
        config={"step_name": "test-step"},
        metadata=metadata or {},
        run_id=run_id,
    )


class _FakeS3:
    """In-memory S3 stub supporting put_object/get_object."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.puts: list[dict[str, Any]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        self.objects[f"{Bucket}/{Key}"] = Body
        self.puts.append({"Bucket": Bucket, "Key": Key, "size": len(Body)})

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        data = self.objects.get(f"{Bucket}/{Key}")
        if data is None:
            raise Exception(f"NoSuchKey: {Bucket}/{Key}")
        return {"Body": SimpleNamespace(read=lambda: data)}


class _FakePodman:
    """Captures the podman invocation and writes a synthetic reply."""

    def __init__(
        self,
        reply: dict[str, Any],
        *,
        write_output: bool = True,
        fake_s3: _FakeS3 | None = None,
    ) -> None:
        self.reply = reply
        self.write_output = write_output
        self.fake_s3 = fake_s3
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))

        # If S3 mode, write reply to S3 using the output key from env.
        output_s3_ref = None
        for i, arg in enumerate(cmd):
            if arg == "-e" and i + 1 < len(cmd):
                val = cmd[i + 1]
                if val.startswith("UMRS_STEP_OUTPUT_S3="):
                    output_s3_ref = val.split("=", 1)[1]

        if output_s3_ref and self.fake_s3 and self.write_output:
            bucket, _, key = output_s3_ref.partition("/")
            self.fake_s3.put_object(Bucket=bucket, Key=key, Body=pickle.dumps(self.reply))
        elif self.write_output:
            # Local file mode: find the workdir bind and write output.pkl.
            workdir = None
            for i, arg in enumerate(cmd):
                if arg == "-v" and i + 1 < len(cmd):
                    workdir = cmd[i + 1].split(":")[0]
                    break
            if workdir is not None:
                with open(os.path.join(workdir, "output.pkl"), "wb") as fh:
                    fh.write(pickle.dumps(self.reply))

        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")


# ── S3 key helper ───────────────────────────────────────────────────────


def test_payload_s3_key_layout():
    key = _payload_s3_key("umrs", "run-42", "vision", "input.pkl")
    assert key == "umrs/run-42/vision/0/input.pkl"


def test_payload_s3_key_custom_attempt():
    key = _payload_s3_key("pfx", "r1", "ocr", "output.pkl", attempt=3)
    assert key == "pfx/r1/ocr/3/output.pkl"


# ── Max-bytes guard ─────────────────────────────────────────────────────


def test_payload_exceeds_max_bytes_raises_setup_error(monkeypatch):
    """A payload larger than the hard cap triggers a setup error."""
    # Set the cap very low so we can trigger it easily.
    monkeypatch.setattr("umrs.server.step_executor._PAYLOAD_MAX_BYTES", 100)
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)

    ctx = _make_ctx()
    # The payload will be > 100 bytes once pickled.
    with pytest.raises(StepExecutionError) as exc_info:
        PodmanExecutor(image="img").execute(
            entrypoint="pkg.mod:Cls",
            config={"step_name": "s", "big": "x" * 200},
            ctx=ctx,
        )
    assert exc_info.value.phase == "setup"
    assert (
        "hard cap" in exc_info.value.message.lower() or "exceeds" in exc_info.value.message.lower()
    )
    # Podman was never invoked.
    assert fake.calls == []


# ── S3 payload transport (PodmanExecutor) ────────────────────────────────


def test_podman_s3_payload_upload_and_download(monkeypatch):
    """When S3 config is present and payload exceeds threshold, the executor
    uploads to S3 and reads the reply from S3."""
    monkeypatch.setattr("umrs.server.step_executor._PAYLOAD_THRESHOLD_BYTES", 10)
    fake_s3 = _FakeS3()
    fake = _FakePodman(
        {"status": "ok", "result": StepResult(extra={"s3": True})},
        fake_s3=fake_s3,
    )
    monkeypatch.setattr(subprocess, "run", fake)

    ctx = _make_ctx(
        metadata={
            EXECUTOR_PAYLOAD_S3_METADATA_KEY: {
                "bucket": "test-bucket",
                "prefix": "umrs",
            },
        },
        s3=fake_s3,
        run_id="run-99",
    )
    result = PodmanExecutor(image="img").execute(
        entrypoint="pkg.mod:Cls",
        config={"step_name": "vision"},
        ctx=ctx,
    )

    assert isinstance(result, StepResult)
    assert result.extra == {"s3": True}

    # Verify S3 was used (input uploaded).
    assert len(fake_s3.puts) >= 1
    input_key = fake_s3.puts[0]["Key"]
    assert "run-99" in input_key
    assert "vision" in input_key
    assert "input.pkl" in input_key

    # Verify podman command uses S3 env vars, not local file paths.
    cmd = fake.calls[0]
    s3_input_env = [a for a in cmd if a.startswith("UMRS_STEP_INPUT_S3=")]
    s3_output_env = [a for a in cmd if a.startswith("UMRS_STEP_OUTPUT_S3=")]
    assert len(s3_input_env) == 1
    assert len(s3_output_env) == 1
    # And local file envs are NOT present.
    local_envs = [a for a in cmd if a.startswith("UMRS_STEP_INPUT=/")]
    assert local_envs == []


def test_podman_small_payload_stays_local_even_with_s3_enabled(monkeypatch):
    """Below-threshold payloads stay on local file even with S3 config."""
    # Set threshold very high so the payload stays local.
    monkeypatch.setattr("umrs.server.step_executor._PAYLOAD_THRESHOLD_BYTES", 10_000_000)
    fake_s3 = _FakeS3()
    fake = _FakePodman(
        {"status": "ok", "result": StepResult()},
    )
    monkeypatch.setattr(subprocess, "run", fake)

    ctx = _make_ctx(
        metadata={
            EXECUTOR_PAYLOAD_S3_METADATA_KEY: {
                "bucket": "test-bucket",
                "prefix": "umrs",
            },
        },
        s3=fake_s3,
    )
    PodmanExecutor(image="img").execute(
        entrypoint="pkg.mod:Cls",
        config={"step_name": "tiny"},
        ctx=ctx,
    )

    # S3 was NOT used for input upload.
    assert fake_s3.puts == []

    # Podman command uses local file env vars.
    cmd = fake.calls[0]
    local_envs = [a for a in cmd if "UMRS_STEP_INPUT=/umrs-step" in a]
    assert len(local_envs) == 1


def test_podman_s3_upload_failure_is_setup_error(monkeypatch):
    """If the S3 upload fails, it's a setup-phase error."""
    monkeypatch.setattr("umrs.server.step_executor._PAYLOAD_THRESHOLD_BYTES", 10)

    class _FailS3:
        def put_object(self, **kwargs):
            raise RuntimeError("S3 is down")

    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)

    ctx = _make_ctx(
        metadata={
            EXECUTOR_PAYLOAD_S3_METADATA_KEY: {
                "bucket": "test-bucket",
                "prefix": "umrs",
            },
        },
        s3=_FailS3(),
    )
    with pytest.raises(StepExecutionError) as exc_info:
        PodmanExecutor(image="img").execute(
            entrypoint="pkg.mod:Cls",
            config={"step_name": "fail-upload"},
            ctx=ctx,
        )
    assert exc_info.value.phase == "setup"
    assert "upload" in exc_info.value.message.lower()
    assert fake.calls == []


# ── Metadata stripping ──────────────────────────────────────────────────


def test_payload_s3_metadata_stripped_from_worker_payload(monkeypatch):
    """``__payload_s3__`` must not leak into the worker's StepContext.metadata."""
    captured: dict[str, Any] = {}

    def _capturing_run(cmd, **kwargs):
        workdir = None
        for i, arg in enumerate(cmd):
            if arg == "-v" and i + 1 < len(cmd):
                workdir = cmd[i + 1].split(":")[0]
                break
        with open(os.path.join(workdir, "input.pkl"), "rb") as fh:
            payload = pickle.load(fh)
        captured["payload"] = payload
        with open(os.path.join(workdir, "output.pkl"), "wb") as fh:
            fh.write(pickle.dumps({"status": "ok", "result": StepResult()}))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _capturing_run)

    # S3 config present but payload below threshold → local mode.
    # The metadata key should still be stripped from the payload.
    ctx = _make_ctx(
        metadata={
            EXECUTOR_PAYLOAD_S3_METADATA_KEY: {
                "bucket": "b",
                "prefix": "p",
            },
        },
    )
    PodmanExecutor(image="img").execute(
        entrypoint="pkg.mod:Cls",
        config={"step_name": "strip-test"},
        ctx=ctx,
    )

    ctx_metadata = captured["payload"]["ctx"]["metadata"]
    assert EXECUTOR_PAYLOAD_S3_METADATA_KEY not in ctx_metadata
    assert EXECUTOR_IMAGE_METADATA_KEY not in ctx_metadata


# ── Runner: routing log annotations ─────────────────────────────────────


class _RecordingExecutor(StepExecutor):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(self, *, entrypoint, config, ctx) -> StepResult:
        self.calls.append({"metadata": dict(ctx.metadata)})
        return StepResult(extra={"ok": True})


class _OkStep(Step):
    name = "probe"

    def setup(self, config):
        pass

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(extra={"ok": True})


_PROBE_MODULE = types.ModuleType("umrs_test_payload_s3_module")
_PROBE_MODULE._OkStep = _OkStep  # type: ignore[attr-defined]
sys.modules["umrs_test_payload_s3_module"] = _PROBE_MODULE


def _workflow() -> dict[str, Any]:
    return {
        "name": "wf",
        "steps": [
            {
                "name": "probe",
                "entrypoint": "umrs_test_payload_s3_module:_OkStep",
                "config": {},
            },
        ],
    }


def test_runner_routing_log_shows_payload_mode_local(tmp_path):
    """Without UMRS_STEP_PAYLOAD_S3, routing log says payload_mode=local."""
    executor = _RecordingExecutor()
    record = get_execution_registry().create(workflow_name="wf", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow(),
        executor=executor,
    )
    assert record.status == "succeeded", record.error
    routing = record.results.get("_routing", {})
    assert routing["probe"]["payload_mode"] == "local"


def test_runner_routing_log_shows_payload_mode_s3(tmp_path, monkeypatch):
    """With UMRS_STEP_PAYLOAD_S3=1 + UMRS_S3_BUCKET, routing log says
    payload_mode=s3 and includes bucket/prefix."""
    monkeypatch.setenv("UMRS_STEP_PAYLOAD_S3", "1")
    monkeypatch.setenv("UMRS_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("UMRS_S3_PREFIX", "jobs")

    executor = _RecordingExecutor()
    record = get_execution_registry().create(workflow_name="wf", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow(),
        executor=executor,
    )
    assert record.status == "succeeded", record.error
    routing = record.results.get("_routing", {})
    assert routing["probe"]["payload_mode"] == "s3"
    assert routing["probe"]["payload_s3_bucket"] == "my-bucket"
    assert routing["probe"]["payload_s3_prefix"] == "jobs"


def test_runner_threads_payload_s3_metadata(tmp_path, monkeypatch):
    """The runner passes the structured S3 config in ctx.metadata when
    UMRS_STEP_PAYLOAD_S3=1."""
    monkeypatch.setenv("UMRS_STEP_PAYLOAD_S3", "1")
    monkeypatch.setenv("UMRS_S3_BUCKET", "my-bucket")
    monkeypatch.setenv("UMRS_S3_PREFIX", "pfx")

    executor = _RecordingExecutor()
    record = get_execution_registry().create(workflow_name="wf", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow(),
        executor=executor,
    )
    assert record.status == "succeeded", record.error
    meta = executor.calls[0]["metadata"]
    assert meta[EXECUTOR_PAYLOAD_S3_METADATA_KEY] == {
        "bucket": "my-bucket",
        "prefix": "pfx",
    }


def test_runner_no_payload_s3_when_env_unset(tmp_path, monkeypatch):
    """Without UMRS_STEP_PAYLOAD_S3, no metadata key is set."""
    monkeypatch.delenv("UMRS_STEP_PAYLOAD_S3", raising=False)

    executor = _RecordingExecutor()
    record = get_execution_registry().create(workflow_name="wf", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow(),
        executor=executor,
    )
    assert record.status == "succeeded", record.error
    meta = executor.calls[0]["metadata"]
    assert EXECUTOR_PAYLOAD_S3_METADATA_KEY not in meta


# ── Worker S3 read/write ────────────────────────────────────────────────


def test_worker_reads_from_s3_when_env_set(monkeypatch, tmp_path):
    """Worker prefers UMRS_STEP_INPUT_S3 over UMRS_STEP_INPUT."""
    from umrs.server import step_worker

    fake_s3 = _FakeS3()
    payload = {
        "entrypoint": "pkg:Cls",
        "config": {},
        "ctx": {
            "input_bucket": "b",
            "input_keys": [],
            "output_bucket": "b",
            "config": {},
            "run_id": "r",
        },
        "s3_config": {},
    }
    fake_s3.put_object(Bucket="bkt", Key="pfx/input.pkl", Body=pickle.dumps(payload))

    monkeypatch.setenv("UMRS_STEP_INPUT_S3", "bkt/pfx/input.pkl")
    monkeypatch.setattr(step_worker, "_build_s3_client", lambda _: fake_s3)

    data = step_worker._read_payload()
    assert len(data) > 0
    decoded = pickle.loads(data)
    assert decoded["entrypoint"] == "pkg:Cls"


def test_worker_write_to_s3_when_env_set(monkeypatch, tmp_path):
    """Worker writes reply to S3 when UMRS_STEP_OUTPUT_S3 is set."""
    from umrs.server import step_worker

    fake_s3 = _FakeS3()
    monkeypatch.setenv("UMRS_STEP_OUTPUT_S3", "bkt/pfx/output.pkl")
    monkeypatch.setattr(step_worker, "_build_s3_client", lambda _: fake_s3)

    step_worker._emit({"status": "ok", "result": "test"})
    assert "bkt/pfx/output.pkl" in fake_s3.objects
    reply = pickle.loads(fake_s3.objects["bkt/pfx/output.pkl"])
    assert reply["status"] == "ok"
