"""Slurm step executor payload must match step_worker / Podman shape."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest
from umrs.pipeline_sdk.step import StepContext, StepResult
from umrs.server.slurm_step_executor import SlurmStepExecutor
from umrs.server.step_executor import EXECUTOR_GPU_METADATA_KEY, StepExecutionError


def _run_with_fake_sbatch(executor: SlurmStepExecutor, ctx: StepContext, **kwargs):
    captured: dict = {}

    def fake_submit(sbatch_path, workdir):
        captured["workdir"] = workdir
        with open(Path(workdir) / "input.pkl", "rb") as handle:
            captured["payload"] = pickle.load(handle)
        reply = kwargs.get(
            "reply",
            {
                "status": "ok",
                "result": StepResult(output_keys=["outputs/summary/all_metrics.csv"]),
            },
        )
        with open(Path(workdir) / "output.pkl", "wb") as handle:
            pickle.dump(reply, handle)
        return "12345"

    with (
        patch.object(executor, "_submit_sbatch", side_effect=fake_submit),
        patch.object(executor, "_check_job_status", return_value="COMPLETED"),
    ):
        result = executor.execute(
            entrypoint=kwargs.get("entrypoint", "aggregate_step:AggregateStep"),
            config=kwargs.get("config", {"step_name": "aggregate"}),
            ctx=ctx,
        )
    return result, captured


def test_slurm_payload_includes_nested_ctx_and_parent_results():
    parent = StepResult(output_keys=["outputs/yolo/metrics.csv"], extra={"task": "yolo"})
    ctx = StepContext(
        s3=None,
        input_bucket="in",
        input_keys=["outputs/yolo/metrics.csv"],
        output_bucket="out",
        parent_output_bucket="out",
        config={"step_name": "aggregate"},
        metadata={},
        run_id="run-1",
        parent_results={"yolo": parent},
    )
    executor = SlurmStepExecutor(image="video-ai-yolo:latest", keep_workdir=True)
    result, captured = _run_with_fake_sbatch(executor, ctx)

    payload = captured["payload"]
    assert "ctx" in payload
    assert payload["ctx"]["input_keys"] == ["outputs/yolo/metrics.csv"]
    assert payload["ctx"]["parent_output_bucket"] == "out"
    assert "yolo" in payload["ctx"]["parent_results"]
    assert result.output_keys == ["outputs/summary/all_metrics.csv"]

    script = Path(captured["workdir"], "run_step.sbatch").read_text(encoding="utf-8")
    assert "UMRS_STEP_INPUT=" in script
    assert "UMRS_STEP_OUTPUT=" in script
    assert "--network=host" in script


def test_slurm_sbatch_forwards_credentials_and_gpu(monkeypatch):
    # S3_ENDPOINT_URL is canonical (ADR-055) and wins over the legacy name; CI
    # exports its own S3_ENDPOINT_URL for the MinIO service, so pin both here.
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://minio:9000")
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    ctx = StepContext(
        s3=None,
        input_bucket="in",
        input_keys=[],
        output_bucket="out",
        metadata={EXECUTOR_GPU_METADATA_KEY: True},
        run_id="run-gpu",
    )
    executor = SlurmStepExecutor(
        image="video-ai-yolo:latest",
        keep_workdir=True,
        timeout_seconds=7200,
    )
    _result, captured = _run_with_fake_sbatch(executor, ctx)
    script = Path(captured["workdir"], "run_step.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --time=02:00:00" in script
    assert "--device nvidia.com/gpu=all" in script
    assert "S3_ENDPOINT_URL=http://minio:9000" in script  # canonical (ADR-055)
    assert "MINIO_ENDPOINT=http://minio:9000" in script  # legacy name still forwarded
    assert "AWS_ACCESS_KEY_ID=test-key" in script
    assert "AWS_SECRET_ACCESS_KEY=test-secret" in script


def test_slurm_rejects_oversized_payload(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_PAYLOAD_MAX_BYTES", "64")
    # Re-import constant path used by executor via module attribute.
    import umrs.server.slurm_step_executor as mod

    monkeypatch.setattr(mod, "_PAYLOAD_MAX_BYTES", 64)
    ctx = StepContext(
        s3=None,
        input_bucket="in",
        input_keys=[],
        output_bucket="out",
        config={"blob": "x" * 200},
        parent_results={"p": StepResult(extra={"big": "y" * 200})},
        run_id="run-big",
    )
    executor = SlurmStepExecutor(image="video-ai-yolo:latest", keep_workdir=True)
    with pytest.raises(StepExecutionError, match="payload size"):
        executor.execute(
            entrypoint="aggregate_step:AggregateStep",
            config={"step_name": "aggregate"},
            ctx=ctx,
        )


def test_slurm_error_reply_raises():
    ctx = StepContext(
        s3=None,
        input_bucket="in",
        input_keys=[],
        output_bucket="out",
        run_id="run-err",
    )
    executor = SlurmStepExecutor(image="video-ai-yolo:latest", keep_workdir=True)
    with pytest.raises(StepExecutionError, match="boom"):
        _run_with_fake_sbatch(
            executor,
            ctx,
            reply={"status": "error", "phase": "run", "message": "boom"},
        )


def test_slurm_refuses_an_image_that_is_not_a_plain_reference():
    """routing_hints.image landed inside `echo "Image: ..."` in the sbatch
    script: $(...) ran on the compute node as the service account."""
    from umrs.server.step_executor import EXECUTOR_IMAGE_METADATA_KEY

    ctx = StepContext(
        s3=None,
        input_bucket="in",
        input_keys=[],
        output_bucket="out",
        metadata={EXECUTOR_IMAGE_METADATA_KEY: "img$(touch /tmp/pwned)"},
        run_id="run-inj",
    )
    executor = SlurmStepExecutor(image="video-ai-yolo:latest")
    with pytest.raises(StepExecutionError, match="plain image reference"):
        executor._resolve_image(ctx)


def test_slurm_script_quotes_interpolations_and_never_echoes_credentials(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s3cr3t with space")
    ctx = StepContext(
        s3=None, input_bucket="in", input_keys=[], output_bucket="out", run_id="run-q"
    )
    executor = SlurmStepExecutor(image="registry.example/team/step:1.0", keep_workdir=True)
    _result, captured = _run_with_fake_sbatch(executor, ctx)
    script = Path(captured["workdir"], "run_step.sbatch").read_text(encoding="utf-8")
    assert "echo Image: registry.example/team/step:1.0" in script
    assert 'echo "Running: ' not in script
    assert "'AWS_SECRET_ACCESS_KEY=s3cr3t with space'" in script
    assert script.count("s3cr3t") == 1  # only in the podman argv, never echoed
