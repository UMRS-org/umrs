"""WP-09 EXEC-2 pins: ModelEvalBackend contract (fail-without-change)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from umrs.jobs.batch_infer import BatchTaskContext
from umrs.jobs.models import AttemptStatus, Job, JobAttempt, JobStatus, JobType, RestartPolicy
from umrs.jobs.model_eval import (
    ModelEvalBackend,
    ModelEvalStatusChecker,
    _build_context,
    _score,
    get_eval_registry,
)


def _job(spec: dict, job_id: str = "job-e1") -> Job:
    return Job(
        job_id=job_id,
        job_type=JobType.MODEL_EVAL,
        project_id="proj-9",
        user_id="u-9",
        spec=spec,
        idempotency_key="",
        status=JobStatus.QUEUED,
        restart_policy=RestartPolicy.ON_CRASH,
        desired_max_attempts=1,
    )


def _attempt() -> JobAttempt:
    return JobAttempt(
        attempt_id="att-e1",
        job_id="job-e1",
        attempt_no=1,
        backend="model_eval",
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


class TestScore:
    def test_exact_trim(self):
        assert _score("exact_trim", " tell ", "tell") == 1.0
        assert _score("exact_trim", " tell ", "tell me more") == 0.0

    def test_contains(self):
        assert _score("contains", "qwen", "the qwen3 model") == 1.0
        assert _score("contains", "qwen2", "the qwen3 model") == 0.0

    def test_regex(self):
        assert _score("regex", "t.*l", "tell") == 1.0
        assert _score("regex", "^q", "tell") == 0.0
        assert _score("regex", "(", "anything") == 0.0  # invalid pattern never scores


class TestBuildContext:
    def test_requires_target_model(self):
        with pytest.raises(ValueError, match="requires spec.target_model"):
            _build_context(_job({"bench": []}), _attempt())

    def test_bench_xor_file(self):
        with pytest.raises(ValueError, match="XOR"):
            _build_context(
                _job(
                    {
                        "target_model": "m",
                        "bench": [{"prompt": "a", "expected": "b"}],
                        "bench_file": "s3://b/k.jsonl",
                    }
                ),
                _attempt(),
            )

    def test_inline_bench_row_shape_gate(self):
        with pytest.raises(ValueError, match="prompt"):
            _build_context(
                _job({"target_model": "m", "bench": [{"prompt": "", "expected": "x"}]}),
                _attempt(),
            )

    def test_default_metric_normalized(self):
        registry = MagicMock()
        registry.list_all.return_value = [_endpoint()]
        with patch("umrs.jobs.model_eval.get_registry", return_value=registry):
            ctx = _build_context(
                _job(
                    {
                        "target_model": "qwen3-8b",
                        "bench": [{"prompt": "a", "expected": "b"}],
                        "metric_default": "contains",
                    }
                ),
                _attempt(),
            )
        assert ctx.metric_default == "contains"
        assert ctx.endpoint_host == "compute-1"

    def test_bad_default_metric_fails_gate(self):
        registry = MagicMock()
        registry.list_all.return_value = [_endpoint()]
        with (
            patch("umrs.jobs.model_eval.get_registry", return_value=registry),
            pytest.raises(ValueError, match="unknown metric"),
        ):
            _build_context(
                _job(
                    {
                        "target_model": "m",
                        "bench": [{"prompt": "a", "expected": "b"}],
                        "metric_default": "fuzzy_addr",
                    }
                ),
                _attempt(),
            )

    def test_no_endpoint_fails_honestly(self):
        registry = MagicMock()
        registry.list_all.return_value = []
        with (
            patch("umrs.jobs.model_eval.get_registry", return_value=registry),
            pytest.raises(ValueError, match="no READY/DEGRADED endpoint"),
        ):
            _build_context(
                _job({"target_model": "m", "bench": [{"prompt": "a", "expected": "b"}]}),
                _attempt(),
            )


def _ctx(**kw) -> __import__("umrs.jobs.model_eval", fromlist=["EvalTaskContext"]).EvalTaskContext:
    from umrs.jobs.model_eval import EvalTaskContext

    params = dict(
        job_id="job-e1",
        attempt_id="att-e1",
        attempt_no=1,
        project_id="proj-9",
        target_model="qwen3-8b",
        bench=[],
        bench_file="",
        metric_default="exact_trim",
        endpoint_host="compute-1",
        endpoint_port=11434,
        endpoint_model="qwen3-8b",
        endpoint_tier="user",
    )
    params.update(kw)
    return EvalTaskContext(**params)


class TestExecuteLoop:
    def test_rows_collect_metrics_and_counts(self):
        backend = ModelEvalBackend(eval_registry=get_eval_registry())
        ctx = _ctx()
        ctx.bench = [
            {"prompt": "p1", "expected": "tell", "metric": "exact_trim"},
            {"prompt": "p2", "expected": "qwen", "metric": "contains"},
            {"prompt": "p3", "expected": "other", "metric": "exact_trim"},
        ]

        outputs = iter([" tell ", "the qwen3 model", "wrong answer"])
        driver = backend._driver_context(ctx)  # shared handle into the same socket

        async def _row(_, _ctx, prompt):  # mock driver call
            nxt = next(outputs)
            return {"ok": True, "output": nxt, "usage": {}}

        with (
            patch("umrs.jobs.model_eval._run_one_row", new=_row),
            patch(
                "umrs.jobs.model_eval._upload_eval_results",
                new=AsyncMock(return_value="jobs/job-e1/attempt_1/eval_results.jsonl"),
            ),
        ):
            asyncio.run(backend._execute(ctx))
        assert ctx.progress["status"] == "completed"
        assert ctx.progress["rows_total"] == 3
        assert ctx.progress["rows_failed"] == 0
        assert ctx.progress["pass_count"] == 2
        assert ctx.progress["fail_count"] == 1
        assert ctx.progress["s3_key"].endswith("eval_results.jsonl")

    def test_failed_rows_increment_counters_not_performance(self):
        backend = ModelEvalBackend(eval_registry=get_eval_registry())
        ctx = _ctx()
        ctx.bench = [{"prompt": "p1", "expected": "x"}]

        with (
            patch(
                "umrs.jobs.model_eval._run_one_row",
                new=AsyncMock(return_value={"ok": False, "error": "transport: slap", "output": ""}),
            ),
            patch(
                "umrs.jobs.model_eval._upload_eval_results",
                new=AsyncMock(return_value="jobs/job-e1/attempt_1/eval_results.jsonl"),
            ),
        ):
            asyncio.run(backend._execute(ctx))
        assert ctx.progress["status"] == "completed"
        assert ctx.progress["rows_failed"] == 1
        assert ctx.progress["fail_count"] == 0  # broken rows never award nor deduct points

    def test_cancel_marks_cancelled_mid_loop(self):
        backend = ModelEvalBackend(eval_registry=get_eval_registry())
        ctx = _ctx()
        ctx.bench = [{"prompt": "p1", "expected": "x"}]
        backend._cancelled.add(ctx.attempt_id)
        asyncio.run(backend._execute(ctx))
        assert ctx.progress["status"] == "cancelled"
        assert "cancelled" in ctx.progress["cancel_reason"]


class TestStatusChecker:
    def test_unknown_stays_running(self):
        asyncio.run(get_eval_registry().remove("att-e1"))
        checker = ModelEvalStatusChecker()
        result = asyncio.run(checker.check_status(_attempt(), _job({})))
        assert result.status == "running"

    def test_completed_passes_counts(self):
        ctx = _ctx()
        ctx.progress = {
            "status": "completed",
            "pass_count": 8,
            "fail_count": 2,
            "s3_key": "jobs/job-e1/attempt_1/eval_results.jsonl",
        }
        asyncio.run(get_eval_registry().register(ctx))
        try:
            result = asyncio.run(ModelEvalStatusChecker().check_status(_attempt(), _job({})))
        finally:
            asyncio.run(get_eval_registry().remove(ctx.attempt_id))
        assert result.status == "succeeded"
        assert result.metrics["pass_count"] == 8
