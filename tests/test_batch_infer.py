"""WP-09 EXEC-1 pins: BatchInferBackend contract (fail-without-change)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from umrs.jobs.batch_infer import (
    BatchInferBackend,
    BatchInferStatusChecker,
    BatchTaskContext,
    BatchTaskRegistry,
    _build_context,
    get_batch_registry,
)
from umrs.jobs.models import (
    AttemptStatus,
    Job,
    JobAttempt,
    JobStatus,
    JobType,
    RestartPolicy,
)


def _job(spec: dict, job_id: str = "job-b1") -> Job:
    return Job(
        job_id=job_id,
        job_type=JobType.BATCH_INFER,
        project_id="proj-9",
        user_id="u-9",
        spec=spec,
        idempotency_key="",
        status=JobStatus.QUEUED,
        restart_policy=RestartPolicy.ON_CRASH,
        desired_max_attempts=1,
    )


def _attempt(attempt_id: str = "att-b1") -> JobAttempt:
    return JobAttempt(
        attempt_id=attempt_id,
        job_id="job-b1",
        attempt_no=1,
        backend="batch_infer",
        status=AttemptStatus.RUNNING,
    )


def _endpoint(model: str = "qwen3-8b", host: str = "compute-1", port: int = 11434):
    ep = MagicMock()
    ep.model = model
    ep.host = host
    ep.port = port
    ep.variant = None
    ep.metadata = {"served_model_name": model, "tier": "user", "owner_type": "user"}
    return ep


def _ctx(**kw) -> BatchTaskContext:
    params = dict(
        job_id="job-b1",
        attempt_id="att-b1",
        attempt_no=1,
        project_id="proj-9",
        model="qwen3-8b",
        prompts_ref="",
        prompts=["p1", "p2", "p3"],
        max_content_len=8000,
        continue_on_error=True,
        endpoint_host="compute-1",
        endpoint_port=11434,
        endpoint_model="qwen3-8b",
        endpoint_tier="user",
    )
    params.update(kw)
    return BatchTaskContext(**params)


class TestBuildContext:
    def test_requires_model(self):
        with pytest.raises(ValueError, match="requires spec.model"):
            _build_context(_job({"prompts": ["a"]}), _attempt())

    def test_prompt_xor_ref(self):
        with pytest.raises(ValueError, match="XOR"):
            _build_context(
                _job({"model": "m", "prompts": ["a"], "prompts_ref": "s3://b/k"}),
                _attempt(),
            )

    def test_inline_cap_enforced(self):
        from umrs.jobs.batch_infer import _MAX_INLINE_PROMPTS

        with pytest.raises(ValueError, match="capped"):
            _build_context(
                _job({"model": "m", "prompts": ["x"] * (_MAX_INLINE_PROMPTS + 1)}),
                _attempt(),
            )

    def test_ready_endpoint_discovery(self):
        ep = _endpoint()
        registry = MagicMock()
        registry.list_all.return_value = [ep]
        with patch("umrs.jobs.batch_infer.get_registry", return_value=registry):
            ctx = _build_context(_job({"model": "qwen3-8b", "prompts": ["a"]}), _attempt())
        assert ctx.endpoint_host == "compute-1"
        assert ctx.endpoint_port == 11434
        assert ctx.endpoint_tier == "user"

    def test_no_endpoint_fails_honestly(self):
        registry = MagicMock()
        registry.list_all.return_value = []
        with (
            patch("umrs.jobs.batch_infer.get_registry", return_value=registry),
            pytest.raises(ValueError, match="no READY/DEGRADED endpoint"),
        ):
            _build_context(_job({"model": "m", "prompts": ["a"]}), _attempt())


class TestExecuteLoop:
    def test_continue_on_error_completes_with_partial_failure(self):
        backend = BatchInferBackend(batch_registry=BatchTaskRegistry())
        ctx = _ctx()
        rows = [
            {"ok": True, "prompt": "p1", "output": "out1", "usage": {}},
            {"ok": False, "error": "upstream_status_500", "prompt": "p2"},
            {"ok": True, "prompt": "p3", "output": "out3", "usage": {}},
        ]
        with (
            patch("umrs.jobs.batch_infer._iter_s3_prompts", return_value=["p1", "p2", "p3"]),
            patch("umrs.jobs.batch_infer._run_one_row", new=AsyncMock(side_effect=rows)),
            patch("umrs.jobs.batch_infer._upload_results", new=AsyncMock(return_value="jobs/job-b1/attempt_1/results.jsonl")),
        ):
            asyncio.run(backend._execute(ctx))
        assert ctx.progress["status"] == "completed"
        assert ctx.progress["rows_failed"] == 1
        assert ctx.progress["rows_ok"] == 2
        assert ctx.progress["s3_key"].endswith("results.jsonl")

    def test_fail_fast_stops_at_first_error(self):
        backend = BatchInferBackend(batch_registry=BatchTaskRegistry())
        ctx = _ctx(continue_on_error=False)
        rows = [
            {"ok": True, "prompt": "p1", "output": "out1", "usage": {}},
            {"ok": False, "error": "upstream_status_500", "prompt": "p2"},
        ]
        with (
            patch("umrs.jobs.batch_infer._iter_s3_prompts", return_value=["p1", "p2", "p3"]),
            patch("umrs.jobs.batch_infer._run_one_row", new=AsyncMock(side_effect=rows)),
            patch("umrs.jobs.batch_infer._upload_results", new=AsyncMock()),
        ):
            asyncio.run(backend._execute(ctx))
        assert ctx.progress["status"] == "failed"
        assert "upstream_status_500" in ctx.progress["error"]

    def test_cancel_marks_cancelled_mid_loop(self):
        backend = BatchInferBackend(batch_registry=BatchTaskRegistry())
        ctx = _ctx()
        backend._cancelled.add(ctx.attempt_id)
        with patch("umrs.jobs.batch_infer._iter_s3_prompts", return_value=["p1", "p2"]):
            asyncio.run(backend._execute(ctx))
        assert ctx.progress["status"] == "cancelled"
        assert "cancelled" in ctx.progress["cancel_reason"]


class TestStatusChecker:
    def test_unknown_stays_running(self):
        checker = BatchInferStatusChecker()
        result = asyncio.run(checker.check_status(_attempt(), _job({})))
        assert result.status == "running"
        asyncio.run(get_batch_registry().remove(_attempt().attempt_id))

    def test_completed_status_passes_through_metrics(self):
        ctx = _ctx()
        ctx.progress = {"status": "completed", "rows_ok": 3, "s3_key": "jobs/job-b1/attempt_1/results.jsonl"}
        asyncio.run(get_batch_registry().register(ctx))
        try:
            checker = BatchInferStatusChecker()
            result = asyncio.run(checker.check_status(_attempt(), _job({})))
        finally:
            asyncio.run(get_batch_registry().remove(ctx.attempt_id))
        assert result.status == "succeeded"
        assert result.metrics["rows_ok"] == 3
