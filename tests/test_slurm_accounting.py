"""D-36 leg B pins: Slurm accounting primitives + the executor stamp.

ADR-067's never-approximate rule: only MEASURED sacct numbers count —
garbage parses to None/0, unmeasured jobs carry nothing.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from umrs.server.slurm_step_executor import measure_slurm_accounting, stamp_slurm_accounting
from umrs.shared.slurm_accounting import (
    measured_gpu_seconds,
    parse_alloc_tres_gpus,
    parse_slurm_elapsed,
)


class TestParseSlurmElapsed:
    def test_plain_minutes(self):
        assert parse_slurm_elapsed("05:30") == 330.0

    def test_hours(self):
        assert parse_slurm_elapsed("2:05:30") == 7530.0

    def test_days(self):
        assert parse_slurm_elapsed("3-02:05:30") == 266730.0

    def test_slurm_garbage_is_none(self):
        assert parse_slurm_elapsed("INVALID") is None
        assert parse_slurm_elapsed("") is None
        assert parse_slurm_elapsed("n/a") is None
        assert parse_slurm_elapsed("12:34:56:78") is None


class TestParseAllocTresGpus:
    def test_gres_colon_form(self):
        assert parse_alloc_tres_gpus("gres:gpu=2") == 2

    def test_gres_slash_form(self):
        assert parse_alloc_tres_gpus("billing=4,gres/gpu=1,mem=8G") == 1

    def test_bare_form(self):
        assert parse_alloc_tres_gpus("gpu=4") == 4

    def test_no_gpus_is_zero(self):
        assert parse_alloc_tres_gpus("billing=2,cpu=2,mem=4G") == 0
        assert parse_alloc_tres_gpus("") == 0


class TestMeasuredGpuSeconds:
    def test_measured(self):
        assert measured_gpu_seconds(3600.0, 2) == 7200.0

    def test_no_elapsed_is_zero(self):
        assert measured_gpu_seconds(None, 2) == 0.0

    def test_no_gpus_is_zero(self):
        assert measured_gpu_seconds(3600.0, 0) == 0.0


def _sacct_run(stdout="", returncode=0):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")

    return fake_run


class TestMeasureSlurmAccounting:
    def test_measures_elapsed_times_gpus(self):
        with patch(
            "umrs.server.slurm_step_executor.subprocess.run",
            _sacct_run(stdout="00:10:00|gres:gpu=2\n"),
        ):
            acct = measure_slurm_accounting("42")
        assert acct == {
            "job_id": "42",
            "elapsed_seconds": 600.0,
            "gpus": 2,
            "gpu_seconds": 1200.0,
        }

    def test_cpu_job_measures_zero_gpus(self):
        with patch(
            "umrs.server.slurm_step_executor.subprocess.run",
            _sacct_run(stdout="00:10:00|billing=2,cpu=2\n"),
        ):
            acct = measure_slurm_accounting("42")
        assert acct["gpus"] == 0
        assert acct["gpu_seconds"] == 0.0

    def test_garbage_elapsed_is_unmeasured(self):
        with patch(
            "umrs.server.slurm_step_executor.subprocess.run",
            _sacct_run(stdout="INVALID|gres:gpu=2\n"),
        ):
            assert measure_slurm_accounting("42") == {}

    def test_sacct_failure_is_unmeasured(self):
        with patch(
            "umrs.server.slurm_step_executor.subprocess.run",
            _sacct_run(returncode=1),
        ):
            assert measure_slurm_accounting("42") == {}

    def test_sacct_absent_is_unmeasured(self):
        def boom(cmd, **kwargs):
            raise FileNotFoundError("no sacct")

        with patch("umrs.server.slurm_step_executor.subprocess.run", boom):
            assert measure_slurm_accounting("42") == {}


class TestStampSlurmAccounting:
    def _result(self):
        from umrs.pipeline_sdk.step import StepResult

        return StepResult(output_keys=["k"])

    def test_stamp_lands_in_extra(self):
        result = self._result()
        stamp_slurm_accounting(result, {"gpu_seconds": 12.0, "gpus": 1})
        assert result.extra["slurm"] == {"gpu_seconds": 12.0, "gpus": 1}

    def test_unmeasured_never_stamps(self):
        result = self._result()
        stamp_slurm_accounting(result, {})
        assert "slurm" not in result.extra
