"""ModelEvalBackend (WP-09 round 2, ADR-047 EXEC-2).

Runs ``model_eval`` durable jobs: a bench suite (``.jsonl`` with rows
``{prompt, expected, metric?}``) driven against the target READY/DEGRADED
endpoint (the same discovery pattern as every other executor on this host).
One attempt per bench; terminal maps completed/cancelled/failed with
counter evidence — pass/fail COUNTS, never a synthesized verdict.

Spec contract (strictly validated):

    target_model: string    — catalog endpoint name (required)
    bench_ref:              — one of:
        bench: [ {prompt, expected, metric?}, ... ] (inline cap 500 rows)
        bench_file: "s3://bucket/key" → .jsonl rows (dicts with a `prompt`
                    and `expected` string; optional per-row `metric` key)
    metric_default?: string — fallback metric when a row has none:
        `exact_trim` (default), `contains`, or `regex` — each named family
        computes what it says: equality after strip/case, substring match,
        full-match; score is always 1.0 on hit and 0.0 on miss (never
        partiality masqueraded as a percentage).

Output: ``jobs/<job_id>/attempt_<n>/eval_results.jsonl`` — one row per bench
item with prompt echo, expected echo, model output, metric used, and a
per-row ``score`` field; attempt metrics carry the counters
(rows_total/rows_ok/rows_failed + pass_count/fail_count + no judgment
sentiment text.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..api.files import get_s3_config
from ..orchestrator.endpoint_registry import EndpointStatus, get_registry
from .batch_infer import (
    _read_prompts_from_s3,
    _run_one_row,
)
from .download_safety import DownloadCancelledError, register_cancel_listener
from .models import Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker

logger = logging.getLogger(__name__)

_MAX_INLINE_ROWS = 500
_VALID_METRICS = ("exact_trim", "contains", "regex")


@dataclass
class EvalTaskContext:
    """Per-attempt eval context."""

    job_id: str
    attempt_id: str
    attempt_no: int
    project_id: str
    target_model: str
    bench: list[dict[str, str]]
    bench_file: str
    metric_default: str
    endpoint_host: str
    endpoint_port: int
    endpoint_model: str
    endpoint_tier: str
    progress: dict[str, Any] = field(default_factory=dict)


def _normalize_metric(name: Any, default: str) -> str:
    candidate = str(name or "").strip().lower() or default
    if candidate not in _VALID_METRICS:
        raise ValueError(
            f"unknown metric {candidate!r}; valid: {', '.join(_VALID_METRICS)}"
        )
    return candidate


def _score(metric: str, expected: str, output: str) -> float:
    """Deterministic row-score. Never a fuzzy score beyond [0,1] boundary mark."""
    if metric == "exact_trim":
        return 1.0 if expected.strip() == output.strip() else 0.0
    if metric == "contains":
        return 1.0 if expected.strip() in output else 0.0
    if metric == "regex":
        try:
            return 1.0 if re.match(expected.strip(), output.strip()) else 0.0
        except re.error:
            return 0.0
    return 0.0


def _validate_bench_row(row: Any) -> dict[str, str]:
    """Single bench-row shape validation (errors detail their row when broken)."""
    if not isinstance(row, dict):
        raise ValueError("bench row must be a dict")
    prompt = str(row.get("prompt") or "")
    expected = str(row.get("expected") or "")
    if not prompt or not expected:
        raise ValueError("bench row requires non-empty `prompt` and `expected` strings")
    return {
        "prompt": prompt,
        "expected": expected,
        "metric": str(row.get("metric") or ""),
    }


def _validate_inline_bench(bench: Any) -> list[dict[str, str]]:
    """Inline bench lane: shape + cap + per-row well-formedness."""
    if not isinstance(bench, list):
        raise ValueError("spec.bench must be a list of rows")
    if len(bench) > _MAX_INLINE_ROWS:
        raise ValueError(f"spec.bench capped at {_MAX_INLINE_ROWS} inline rows")
    return [_validate_bench_row(row) for row in bench]


def _validate_input(
    spec: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    """Strict input lane validation → (bench_file, inline_rows)."""
    bench = spec.get("bench")
    bench_file = str(spec.get("bench_file") or "").strip()
    if (not bench_file) and (not bench):
        raise ValueError("model_eval requires spec.bench OR spec.bench_file")
    if bench_file and bench:
        raise ValueError(
            "model_eval accepts only one input lane (bench_file XOR bench)"
        )
    if bench is None:
        return bench_file, []
    return bench_file, _validate_inline_bench(bench)


def _load_bench_inline(ctx: EvalTaskContext) -> list[dict[str, str]]:
    """Context bench rows — inline rows OR rows from the s3 bench file."""
    if ctx.bench:
        return list(ctx.bench)
    client = get_s3_config().get_client()
    lines = _read_prompts_from_s3(ctx.bench_file, client)
    rows: list[dict[str, str]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"bench file row not json: {exc}") from exc
        rows.append(_validate_bench_row(parsed))
    return rows


def _resolve_ready_endpoint(project_id: str, target_model: str) -> Any:
    registry = get_registry()
    candidates = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )
    for ep in candidates:
        served = ep.metadata.get("served_model_name") or ep.model
        if ep.model == target_model or served == target_model:
            return ep
    raise ValueError(
        f"no READY/DEGRADED endpoint for model {target_model!r} under project {project_id!r}"
    )


def _build_context(job: Job, attempt: JobAttempt) -> EvalTaskContext:
    """Strict validation chain → execution context."""
    spec = job.spec or {}
    target_model = str(spec.get("target_model") or "").strip()
    if not target_model:
        raise ValueError(
            "model_eval requires spec.target_model (endpoint catalog name)"
        )
    project_id = job.project_id or spec.get("project_id") or ""
    bench_file, bench = _validate_input(spec)
    metric_default = _normalize_metric(spec.get("metric_default") or "", "exact_trim")
    endpoint = _resolve_ready_endpoint(project_id, target_model)
    return EvalTaskContext(
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
        attempt_no=attempt.attempt_no,
        project_id=project_id,
        target_model=target_model,
        bench=bench,
        bench_file=bench_file,
        metric_default=metric_default,
        endpoint_host=endpoint.host,
        endpoint_port=endpoint.port,
        endpoint_model=endpoint.model,
        endpoint_tier=str(
            endpoint.metadata.get("tier") or endpoint.metadata.get("owner_type") or ""
        ),
    )


class ModelEvalTaskRegistry:
    """In-flight eval context registry (same shape as the other executors)."""

    def __init__(self) -> None:
        self._tasks: dict[str, EvalTaskContext] = {}
        self._lock = asyncio.Lock()

    async def register(self, ctx: EvalTaskContext) -> None:
        async with self._lock:
            self._tasks[ctx.attempt_id] = ctx

    async def get(self, attempt_id: str) -> EvalTaskContext | None:
        async with self._lock:
            return self._tasks.get(attempt_id)

    async def remove(self, attempt_id: str) -> EvalTaskContext | None:
        async with self._lock:
            return self._tasks.pop(attempt_id, None)


_eval_registry: ModelEvalTaskRegistry | None = None


def get_eval_registry() -> ModelEvalTaskRegistry:
    """Get or create the global eval registry."""
    global _eval_registry
    if _eval_registry is None:
        _eval_registry = ModelEvalTaskRegistry()
    return _eval_registry


async def _upload_eval_results(ctx: EvalTaskContext, rows: list[dict[str, Any]]) -> str:
    """Write eval_results.jsonl under the attempt scope. Returns the key."""
    from .attempt_credentials import scope_prefix_for

    config = get_s3_config()
    client = config.get_client()
    key = f"{scope_prefix_for(ctx.job_id, ctx.attempt_no)}eval_results.jsonl"
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + (
        "\n" if rows else ""
    )
    client.put_object(
        Bucket=config.bucket,
        Key=key,
        Body=payload.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    return key


class ModelEvalBackend:
    """Executor for model_eval jobs (ADR-047 EXEC-2)."""

    def __init__(
        self,
        eval_registry: ModelEvalTaskRegistry | None = None,
        model: str = "model_eval",
    ) -> None:
        self.eval_registry = eval_registry or get_eval_registry()
        self._model_tag = model
        self._handles: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        register_cancel_listener(self._on_durable_cancel)

    @property
    def backend_name(self) -> str:
        return self._model_tag

    def _on_durable_cancel(self, job_id: str) -> None:
        """Cancel-bus listener (6b-3b shape): our eval attempts get interrupted."""
        for attempt_id, ctx in list(self.eval_registry._tasks.items()):
            if ctx.job_id != job_id:
                continue
            self._cancelled.add(attempt_id)
            handle = self._handles.get(attempt_id)
            if handle is not None and not handle.done():
                handle.cancel()

    def cancel_attempt(self, attempt_id: str) -> bool:
        """Explicit per-attempt cancel (public; mirrors the durable path)."""
        self._cancelled.add(attempt_id)
        handle = self._handles.get(attempt_id)
        if handle is None or handle.done():
            return False
        handle.cancel()
        return True

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        """Submit one eval attempt (inline asyncio; the loop drives the rows)."""
        ctx = _build_context(job, attempt)
        await self.eval_registry.register(ctx)
        self._handles[attempt.attempt_id] = asyncio.create_task(self._execute(ctx))
        logger.info(
            "Submitted model_eval: job=%s attempt=%s target=%s backend=%s",
            job.job_id,
            attempt.attempt_id,
            ctx.target_model,
            self.backend_name,
        )
        return attempt.attempt_id

    def _driver_context(self, ctx: EvalTaskContext) -> Any:
        """LLM-driver shape consumed by `_run_one_row` (holds endpoint fields only)."""
        from .batch_infer import BatchTaskContext

        return BatchTaskContext(
            job_id=ctx.job_id,
            attempt_id=ctx.attempt_id,
            attempt_no=ctx.attempt_no,
            project_id=ctx.project_id,
            model=ctx.target_model,
            prompts_ref="",
            prompts=[],
            max_content_len=8000,
            continue_on_error=True,
            endpoint_host=ctx.endpoint_host,
            endpoint_port=ctx.endpoint_port,
            endpoint_model=ctx.endpoint_model,
            endpoint_tier=ctx.endpoint_tier,
        )

    async def _drive_rows(
        self, ctx: EvalTaskContext
    ) -> tuple[list[dict[str, Any]], str]:
        """Bench-row loop (cancel-aware, progress counters only)."""
        import httpx

        rows: list[dict[str, Any]] = []
        bench_rows = _load_bench_inline(ctx)
        total = len(bench_rows)
        driver_ctx = self._driver_context(ctx)
        async with httpx.AsyncClient() as client:
            for row in bench_rows:
                if ctx.attempt_id in self._cancelled:
                    raise DownloadCancelledError(f"attempt {ctx.attempt_id} cancelled")
                metric = _normalize_metric(row.get("metric") or "", ctx.metric_default)
                out = await _run_one_row(client, driver_ctx, row["prompt"])
                score = None
                error_text = out.get("error", "")
                if out["ok"]:
                    score = _score(metric, row["expected"], out["output"])
                rows.append(
                    {
                        "ok": out["ok"],
                        "error": error_text,
                        "prompt": row["prompt"],
                        "expected": row["expected"],
                        "output": out.get("output", ""),
                        "metric": metric,
                        "score": score,
                    }
                )
                self._refresh_progress(ctx, rows, total)
        s3_key = await _upload_eval_results(ctx, rows)
        return rows, s3_key

    async def _execute(self, ctx: EvalTaskContext) -> None:
        """Run the bench rows; never fabrication a score from a broken row."""
        try:
            _, s3_key = await self._drive_rows(ctx)
            ctx.progress = {**ctx.progress, "status": "completed", "s3_key": s3_key}
        except DownloadCancelledError as exc:
            ctx.progress = {
                **ctx.progress,
                "status": "cancelled",
                "cancel_reason": str(exc),
            }
        except Exception as exc:
            logger.exception("model_eval failed for %s", ctx.attempt_id)
            ctx.progress = {**ctx.progress, "status": "failed", "error": str(exc)}
        finally:
            self._handles.pop(ctx.attempt_id, None)
            await self.eval_registry.remove(ctx.attempt_id)

    @staticmethod
    def _refresh_progress(
        ctx: EvalTaskContext, rows: list[dict[str, Any]], total: int
    ) -> None:
        """Per-row progress payload (counters only — never judgment text)."""
        ctx.progress = {
            "rows_total": total,
            "rows_done": len(rows),
            "rows_ok": sum(1 for r in rows if r["ok"]),
            "rows_failed": sum(1 for r in rows if not r["ok"]),
            "pass_count": sum(1 for r in rows if r.get("score") == 1.0),
            "fail_count": sum(1 for r in rows if r.get("score") == 0.0),
        }


class ModelEvalStatusChecker(BackendStatusChecker):
    """Read the eval registry into terminal-or-running verdicts."""

    @property
    def job_type(self) -> JobType:
        return JobType.MODEL_EVAL

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        registry = get_eval_registry()
        ctx = await registry.get(attempt.attempt_id)
        progress = (ctx.progress or {}) if ctx else {}
        state = progress.get("status")
        if state == "completed":
            return AttemptCheckResult.succeeded(metrics=dict(progress))
        if state == "cancelled":
            reason = progress.get("cancel_reason", "")
            return AttemptCheckResult.failed(f"cancelled: {reason}")
        if state == "failed":
            return AttemptCheckResult.failed(progress.get("error") or "")
        if state in (None, "running", ""):
            return AttemptCheckResult.running()
        return AttemptCheckResult.failed(str(state))


def create_model_eval_backend() -> ModelEvalBackend:
    """Factory hook for composition composition."""
    return ModelEvalBackend()


def create_model_eval_status_checker() -> ModelEvalStatusChecker:
    """Factory hook for the composite status checker."""
    return ModelEvalStatusChecker()
