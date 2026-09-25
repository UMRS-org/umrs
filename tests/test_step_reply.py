"""Data-only step replies (ADR-074 D-6, 2026-09-25).

The step writes its own reply, so the server must never ``pickle.loads`` it:
the Slurm lane reads a JSON reply, and an image that predates JSON replies is
read through a loader that resolves StepResult and builtin containers only.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from umrs.pipeline_sdk.step import StepContext, StepResult
from umrs.server.slurm_step_executor import SlurmStepExecutor
from umrs.server.step_executor import StepExecutionError
from umrs.server.step_reply import (
    check_reply_size,
    reply_to_json,
    result_from_json,
    result_from_legacy_pickle,
)

_RAN: list[str] = []


def _record_ran(tag: str) -> None:
    _RAN.append(tag)


class _Payload:
    """Pickles to a call of an arbitrary global — what a hostile step writes."""

    def __reduce__(self):
        return (_record_ran, ("pwned",))


def _ok(result: StepResult) -> dict:
    return {"status": "ok", "result": result}


def test_json_round_trip_keeps_fields_and_packed_embeddings():
    emb = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = StepResult(output_keys=["a/b"], extra={"n": 3, (1, 2): "tuple-key"}, embeddings=emb)
    back = result_from_json(reply_to_json(_ok(result)))
    assert back.output_keys == ["a/b"]
    assert back.extra == {"n": 3, "(1, 2)": "tuple-key"}
    assert back.embeddings.dtype == np.float32
    assert np.array_equal(back.embeddings, emb)


def test_json_error_reply_raises_with_its_phase():
    blob = json.dumps({"status": "error", "phase": "run", "message": "boom"}).encode()
    with pytest.raises(StepExecutionError, match="boom") as info:
        result_from_json(blob)
    assert info.value.phase == "run"


@pytest.mark.parametrize(
    "fields",
    [
        {"output_keys": [1]},
        {"extra": []},
        {"embeddings": {"dtype": "|O", "shape": [1], "b64": ""}},
        {"embeddings": {"dtype": "<f4", "shape": [4], "b64": "AAAA"}},
        {"embeddings": {"dtype": "<f4", "shape": [-1], "b64": ""}},
    ],
)
def test_json_reply_fields_are_checked(fields):
    with pytest.raises(StepExecutionError):
        result_from_json(json.dumps({"status": "ok", "result": fields}).encode())


def test_legacy_pickle_of_a_plain_result_still_loads():
    result = result_from_legacy_pickle(pickle.dumps(_ok(StepResult(output_keys=["k"]))))
    assert result.output_keys == ["k"]


def test_legacy_pickle_never_calls_a_global():
    _RAN.clear()
    blob = pickle.dumps(_ok(StepResult(extra={"x": _Payload()})))
    with pytest.raises(StepExecutionError, match="refused"):
        result_from_legacy_pickle(blob)
    assert _RAN == []


def test_legacy_numpy_reply_is_refused_by_name():
    blob = pickle.dumps(_ok(StepResult(embeddings=np.zeros(3))))
    with pytest.raises(StepExecutionError, match="numpy"):
        result_from_legacy_pickle(blob)


def test_legacy_shared_containers_are_refused():
    shared = [1, 2]
    blob = pickle.dumps(_ok(StepResult(extra={"a": shared, "b": shared})))
    with pytest.raises(StepExecutionError, match="shares containers"):
        result_from_legacy_pickle(blob)


def test_reply_size_cap():
    with pytest.raises(StepExecutionError, match="exceeds 4 bytes"):
        check_reply_size(b"12345", "/w/output.json", 4)


def test_worker_writes_the_json_reply_when_asked(tmp_path, monkeypatch):
    from umrs.server import step_worker

    out, js = tmp_path / "output.pkl", tmp_path / "output.json"
    monkeypatch.setenv("UMRS_STEP_OUTPUT", str(out))
    monkeypatch.setenv("UMRS_STEP_OUTPUT_JSON", str(js))
    step_worker._emit(_ok(StepResult(output_keys=["k"])))
    assert result_from_json(js.read_bytes()).output_keys == ["k"]
    assert out.exists()


def _slurm_run(tmp_path, writer):
    ctx = StepContext(s3=None, input_bucket="in", input_keys=[], output_bucket="out", run_id="r")
    executor = SlurmStepExecutor(image="step:1", keep_workdir=True)
    seen: dict = {}

    def fake_submit(_sbatch_path, workdir):
        seen["workdir"] = workdir
        writer(Path(workdir))
        return "1"

    with (
        patch.object(executor, "_submit_sbatch", side_effect=fake_submit),
        patch.object(executor, "_check_job_status", return_value="COMPLETED"),
    ):
        result = executor.execute(entrypoint="m:S", config={"step_name": "s"}, ctx=ctx)
    return result, Path(seen["workdir"], "run_step.sbatch").read_text(encoding="utf-8")


def test_slurm_lane_asks_for_and_reads_the_json_reply(tmp_path):
    def writer(workdir: Path) -> None:
        (workdir / "output.json").write_bytes(reply_to_json(_ok(StepResult(output_keys=["j"]))))
        # a hostile output.pkl next to it is never read
        (workdir / "output.pkl").write_bytes(pickle.dumps(_Payload()))

    _RAN.clear()
    result, script = _slurm_run(tmp_path, writer)
    assert "UMRS_STEP_OUTPUT_JSON=/work/output.json" in script
    assert result.output_keys == ["j"]
    assert _RAN == []


def test_slurm_lane_refuses_a_hostile_legacy_reply(tmp_path):
    _RAN.clear()
    with pytest.raises(StepExecutionError, match="refused"):
        _slurm_run(tmp_path, lambda w: (w / "output.pkl").write_bytes(pickle.dumps(_Payload())))
    assert _RAN == []
    assert not os.environ.get("UMRS_STEP_OUTPUT_JSON")
