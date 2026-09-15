"""Tests for :class:`PodmanExecutor` (Slice 9).

By default these tests stub out :func:`subprocess.run` so they can
run on any machine without podman installed. A gated integration
path runs a real podman invocation when
``UMRS_PODMAN_INTEGRATION=1`` is set and ``podman`` is on ``PATH``;
that path is opt-in and skipped in normal CI.
"""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from umrs.pipeline_sdk.step import StepContext, StepResult
from umrs.server.step_executor import (
    InProcessExecutor,
    PodmanExecutor,
    StepExecutionError,
    SubprocessExecutor,
    build_step_executor,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_ctx() -> StepContext:
    return StepContext(
        s3=None,
        input_bucket="b",
        input_keys=[],
        output_bucket="b",
        config={},
        run_id="t",
    )


class _FakePodman:
    """Captures the podman invocation and writes a synthetic reply.

    The ``reply`` argument is what the worker would have written to
    ``output.pkl``. Set ``write_output=False`` to simulate a worker
    crash that produces no payload.
    """

    def __init__(
        self,
        reply: Dict[str, Any],
        *,
        write_output: bool = True,
        returncode: int = 0,
        stderr: bytes = b"",
        raises: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.write_output = write_output
        self.returncode = returncode
        self.stderr = stderr
        self.raises = raises
        self.calls: List[List[str]] = []

    def __call__(self, cmd, **kwargs):  # noqa: ANN001
        self.calls.append(list(cmd))
        if self.raises is not None:
            raise self.raises
        # Locate the workdir mount: ``-v <host>:/umrs-step:Z``.
        workdir = None
        for i, arg in enumerate(cmd):
            if arg == "-v" and i + 1 < len(cmd):
                bind = cmd[i + 1]
                workdir = bind.split(":")[0]
                break
        if self.write_output and workdir is not None:
            with open(os.path.join(workdir, "output.pkl"), "wb") as fh:
                fh.write(pickle.dumps(self.reply))
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=b"",
            stderr=self.stderr,
        )


# ----------------------------------------------------------------------
# Command shape
# ----------------------------------------------------------------------


def test_podman_command_shape(monkeypatch):
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)

    executor = PodmanExecutor(image="example.com/umrs-step:latest")
    executor.execute(
        entrypoint="pkg.mod:Cls",
        config={"k": 1},
        ctx=_make_ctx(),
    )

    assert len(fake.calls) == 1
    cmd = fake.calls[0]
    assert cmd[0] == "podman"
    assert cmd[1] == "run"
    assert "--rm" in cmd
    assert "--security-opt=no-new-privileges" in cmd
    assert "--network=host" in cmd
    # Image is the last positional before the entry-command.
    assert "example.com/umrs-step:latest" in cmd
    img_idx = cmd.index("example.com/umrs-step:latest")
    assert cmd[img_idx + 1 : img_idx + 4] == [
        "python",
        "-m",
        "umrs.server.step_worker",
    ]
    # The container name is unique per call.
    name_idx = cmd.index("--name")
    assert cmd[name_idx + 1].startswith("umrs-step-")
    # Worker reads the input file via env.
    assert "UMRS_STEP_INPUT=/umrs-step/input.pkl" in cmd
    assert "UMRS_STEP_OUTPUT=/umrs-step/output.pkl" in cmd


def test_podman_gpu_flag_off_by_default(monkeypatch):
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)
    PodmanExecutor(image="img").execute(entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx())
    assert "--device" not in fake.calls[0]


def test_podman_gpu_flag_on(monkeypatch):
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)
    PodmanExecutor(image="img", gpu=True).execute(
        entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx()
    )
    cmd = fake.calls[0]
    assert "--device" in cmd
    dev_idx = cmd.index("--device")
    assert cmd[dev_idx + 1] == "nvidia.com/gpu=all"


def test_podman_extra_args_injected_before_image(monkeypatch):
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)
    PodmanExecutor(
        image="img",
        extra_args=["--pull=missing", "--label", "team=ai4s"],
    ).execute(entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx())
    cmd = fake.calls[0]
    img_idx = cmd.index("img")
    assert "--pull=missing" in cmd[:img_idx]
    assert "--label" in cmd[:img_idx]


def test_podman_custom_network(monkeypatch):
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)
    PodmanExecutor(image="img", network="none").execute(
        entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx()
    )
    assert "--network=none" in fake.calls[0]


def test_podman_forwards_minio_env(monkeypatch):
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setenv("MINIO_ENDPOINT", "http://minio:9000")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    PodmanExecutor(image="img").execute(entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx())
    cmd = fake.calls[0]
    assert "MINIO_ENDPOINT=http://minio:9000" in cmd
    assert "AWS_REGION=us-east-1" in cmd


# ----------------------------------------------------------------------
# Reply handling
# ----------------------------------------------------------------------


def test_podman_returns_step_result(monkeypatch):
    expected = StepResult(extra={"hello": "podman"})
    fake = _FakePodman({"status": "ok", "result": expected})
    monkeypatch.setattr(subprocess, "run", fake)
    out = PodmanExecutor(image="img").execute(entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx())
    assert isinstance(out, StepResult)
    assert out.extra == {"hello": "podman"}


def test_podman_propagates_run_phase_error(monkeypatch):
    fake = _FakePodman({"status": "error", "phase": "run", "message": "step boom"})
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(StepExecutionError) as excinfo:
        PodmanExecutor(image="img").execute(entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx())
    assert excinfo.value.phase == "run"
    assert "step boom" in excinfo.value.message


def test_podman_setup_phase_error(monkeypatch):
    fake = _FakePodman({"status": "error", "phase": "setup", "message": "bad import"})
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(StepExecutionError) as excinfo:
        PodmanExecutor(image="img").execute(entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx())
    assert excinfo.value.phase == "setup"


def test_podman_missing_output_classified_as_worker(monkeypatch):
    fake = _FakePodman(
        {"status": "ok", "result": StepResult()},
        write_output=False,
        returncode=139,
        stderr=b"segfault in container\n",
    )
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(StepExecutionError) as excinfo:
        PodmanExecutor(image="img").execute(entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx())
    assert excinfo.value.phase == "worker"
    assert "without writing" in excinfo.value.message


def test_podman_timeout_classified_as_worker(monkeypatch):
    fake = _FakePodman(
        {"status": "ok", "result": StepResult()},
        raises=subprocess.TimeoutExpired(cmd=["podman"], timeout=0.5),
    )
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(StepExecutionError) as excinfo:
        PodmanExecutor(image="img", timeout_seconds=0.5).execute(
            entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx()
        )
    assert excinfo.value.phase == "worker"
    assert "timed out" in excinfo.value.message


def test_podman_binary_missing_classified_as_worker(monkeypatch):
    fake = _FakePodman(
        {"status": "ok", "result": StepResult()},
        raises=FileNotFoundError("podman not on PATH"),
    )
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(StepExecutionError) as excinfo:
        PodmanExecutor(image="img").execute(entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx())
    assert excinfo.value.phase == "worker"


def test_podman_workdir_cleaned_up_by_default(monkeypatch):
    captured: List[str] = []

    def _capture_then_succeed(cmd, **kwargs):
        for i, arg in enumerate(cmd):
            if arg == "-v" and i + 1 < len(cmd):
                workdir = cmd[i + 1].split(":")[0]
                captured.append(workdir)
                with open(os.path.join(workdir, "output.pkl"), "wb") as fh:
                    fh.write(pickle.dumps({"status": "ok", "result": StepResult()}))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _capture_then_succeed)
    PodmanExecutor(image="img").execute(entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx())
    assert captured
    assert not os.path.isdir(captured[0])


def test_podman_keep_workdir_preserves_dir(monkeypatch, tmp_path):
    captured: List[str] = []

    def _ok(cmd, **kwargs):
        for i, arg in enumerate(cmd):
            if arg == "-v" and i + 1 < len(cmd):
                workdir = cmd[i + 1].split(":")[0]
                captured.append(workdir)
                with open(os.path.join(workdir, "output.pkl"), "wb") as fh:
                    fh.write(pickle.dumps({"status": "ok", "result": StepResult()}))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _ok)
    try:
        PodmanExecutor(image="img", keep_workdir=True).execute(
            entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx()
        )
        assert captured
        assert os.path.isdir(captured[0])
    finally:
        for d in captured:
            shutil.rmtree(d, ignore_errors=True)


def test_podman_timeout_triggers_force_remove(monkeypatch):
    rm_calls: List[List[str]] = []

    def _run(cmd, **kwargs):
        if cmd[:3] == ["podman", "rm", "-f"]:
            rm_calls.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=0.5)

    monkeypatch.setattr(subprocess, "run", _run)
    with pytest.raises(StepExecutionError):
        PodmanExecutor(image="img", timeout_seconds=0.5).execute(
            entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx()
        )
    assert rm_calls, "expected best-effort 'podman rm -f' on timeout"


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def test_factory_dispatches_to_podman(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "podman")
    monkeypatch.setenv("UMRS_STEP_IMAGE", "example.com/umrs-step:tag")
    monkeypatch.delenv("UMRS_STEP_PODMAN_EXTRA_ARGS", raising=False)
    executor = build_step_executor()
    assert isinstance(executor, PodmanExecutor)
    assert executor.image == "example.com/umrs-step:tag"
    assert executor.network == "host"
    assert executor.gpu is False
    assert executor.extra_args == []


def test_factory_podman_requires_image(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "podman")
    monkeypatch.delenv("UMRS_STEP_IMAGE", raising=False)
    with pytest.raises(RuntimeError, match="UMRS_STEP_IMAGE"):
        build_step_executor()


def test_factory_podman_parses_knobs(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "podman")
    monkeypatch.setenv("UMRS_STEP_IMAGE", "img")
    monkeypatch.setenv("UMRS_STEP_PODMAN_BIN", "/usr/local/bin/podman")
    monkeypatch.setenv("UMRS_STEP_PODMAN_NETWORK", "none")
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    monkeypatch.setenv(
        "UMRS_STEP_PODMAN_EXTRA_ARGS",
        "--pull=missing --label foo=bar",
    )
    monkeypatch.setenv("UMRS_STEP_KEEP_WORKDIR", "1")
    monkeypatch.setenv("UMRS_STEP_EXECUTOR_TIMEOUT", "30")
    executor = build_step_executor()
    assert isinstance(executor, PodmanExecutor)
    assert executor.podman_bin == "/usr/local/bin/podman"
    assert executor.network == "none"
    assert executor.gpu is True
    assert executor.extra_args == ["--pull=missing", "--label", "foo=bar"]
    assert executor.keep_workdir is True
    assert executor.timeout_seconds == 30.0


def test_factory_other_modes_unaffected(monkeypatch):
    """The new podman branch must not change inproc/subprocess dispatch."""
    monkeypatch.delenv("UMRS_STEP_EXECUTOR", raising=False)
    assert isinstance(build_step_executor(), InProcessExecutor)
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "subprocess")
    assert isinstance(build_step_executor(), SubprocessExecutor)


# ----------------------------------------------------------------------
# Runner integration — failure feedback (Slice 7) still fires
# ----------------------------------------------------------------------


def test_podman_run_failure_propagated_through_runner(monkeypatch):
    """A podman ``run``-phase failure must surface as a failed step.

    The runner's existing Slice-7 feedback wiring (``record_failure``
    on non-setup phases) is verified directly by the runner test
    suite; here we just guarantee that swapping in a Podman-style
    executor does not bypass the runner's failure path.
    """
    from umrs.server import step_router
    from umrs.server.workflow_execution_registry import (
        get_execution_registry,
        reset_execution_registry_for_tests,
    )
    from umrs.server.workflow_runner import run_workflow_record

    reset_execution_registry_for_tests()
    step_router.reset_feedback_store()

    calls: List[Dict[str, Any]] = []

    class _FailingPodman:
        name = "podman-stub"

        def execute(self, *, entrypoint, config, ctx):
            calls.append({"entrypoint": entrypoint, "run_id": ctx.run_id})
            raise StepExecutionError("run", "step crashed in container")

    record = get_execution_registry().create(workflow_name="podman-fail", user_id="u")
    run_workflow_record(
        record,
        source_path="/tmp",
        workflow_json={
            "name": "podman-fail",
            "steps": [
                {
                    "name": "only",
                    # Real importable Step class — early validation in
                    # the runner imports the entrypoint before the
                    # executor is invoked. ``_FailingPodman`` ignores
                    # the class and raises a run-phase error.
                    "entrypoint": "test_step_executor:_OkStep",
                    "config": {},
                },
            ],
        },
        executor=_FailingPodman(),
    )

    reset_execution_registry_for_tests()
    step_router.reset_feedback_store()

    assert record.status == "failed"
    assert len(calls) == 1
    assert calls[0]["entrypoint"].endswith(":_OkStep")
    # Error message preserves the StepExecutionError phase + message.
    step_state = record.steps.get("only")
    assert step_state is not None
    assert "run() raised" in (step_state.error or "")
    assert "step crashed in container" in (step_state.error or "")


# ----------------------------------------------------------------------
# Slice 10: per-step image override via ctx.metadata
# ----------------------------------------------------------------------


def _ctx_with_image(image_value: Any) -> StepContext:
    from umrs.server.step_executor import EXECUTOR_IMAGE_METADATA_KEY

    ctx = _make_ctx()
    ctx.metadata[EXECUTOR_IMAGE_METADATA_KEY] = image_value
    return ctx


def test_podman_per_step_image_override_used(monkeypatch):
    """A non-empty string override replaces the executor's default image."""
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)
    PodmanExecutor(image="default-img:latest").execute(
        entrypoint="pkg.mod:Cls",
        config={},
        ctx=_ctx_with_image("override:cuda"),
    )
    cmd = fake.calls[0]
    assert "override:cuda" in cmd
    # Default must not also appear (one image per run).
    assert "default-img:latest" not in cmd
    img_idx = cmd.index("override:cuda")
    assert cmd[img_idx + 1 : img_idx + 4] == [
        "python",
        "-m",
        "umrs.server.step_worker",
    ]


def test_podman_per_step_image_default_fallback(monkeypatch):
    """Absent override → executor's default image is used."""
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)
    PodmanExecutor(image="default-img:latest").execute(
        entrypoint="pkg.mod:Cls",
        config={},
        ctx=_make_ctx(),  # no override
    )
    assert "default-img:latest" in fake.calls[0]


def test_podman_per_step_image_empty_string_is_setup_error(monkeypatch):
    """Empty/whitespace override → setup-phase error, not silent fallback."""
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(StepExecutionError) as exc_info:
        PodmanExecutor(image="default-img").execute(
            entrypoint="pkg.mod:Cls",
            config={},
            ctx=_ctx_with_image("   "),
        )
    assert exc_info.value.phase == "setup"
    assert "empty" in exc_info.value.message.lower()
    # Podman was never invoked.
    assert fake.calls == []


def test_podman_per_step_image_non_string_is_setup_error(monkeypatch):
    """Non-string override → setup-phase error."""
    fake = _FakePodman({"status": "ok", "result": StepResult()})
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(StepExecutionError) as exc_info:
        PodmanExecutor(image="default-img").execute(
            entrypoint="pkg.mod:Cls",
            config={},
            ctx=_ctx_with_image(42),
        )
    assert exc_info.value.phase == "setup"
    assert "string" in exc_info.value.message.lower()
    assert fake.calls == []


def test_podman_per_step_image_metadata_stripped_from_payload(monkeypatch):
    """``__executor_image__`` is an executor-internal contract — it must not
    leak into the worker's StepContext.metadata."""
    captured: Dict[str, Any] = {}

    def _capturing_run(cmd, **kwargs):  # noqa: ANN001
        # Find the workdir bind and read input.pkl that the executor wrote.
        workdir = None
        for i, arg in enumerate(cmd):
            if arg == "-v" and i + 1 < len(cmd):
                workdir = cmd[i + 1].split(":")[0]
                break
        with open(os.path.join(workdir, "input.pkl"), "rb") as fh:
            payload = pickle.load(fh)
        captured["payload"] = payload
        # Now write a synthetic reply so execute() returns cleanly.
        with open(os.path.join(workdir, "output.pkl"), "wb") as fh:
            fh.write(pickle.dumps({"status": "ok", "result": StepResult()}))
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _capturing_run)
    PodmanExecutor(image="default-img").execute(
        entrypoint="pkg.mod:Cls",
        config={},
        ctx=_ctx_with_image("override:tag"),
    )
    from umrs.server.step_executor import EXECUTOR_IMAGE_METADATA_KEY

    payload = captured["payload"]
    # The dunder key must not be visible to the step.
    ctx_metadata = payload["ctx"]["metadata"]
    assert EXECUTOR_IMAGE_METADATA_KEY not in ctx_metadata
