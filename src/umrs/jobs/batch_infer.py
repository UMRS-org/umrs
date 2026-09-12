"""BatchInferBackend (WP-09 round 2, ADR-047 EXEC-1).

Runs ``batch_infer`` durable jobs against an ALREADY-DEPLOYED readiness
endpoint (litellm on mini-rivault or the UMRS-side proxy pattern, same
discovery as api/llm.py — never a second dialect). Executor-driven lane from
ADR-044: scheduler owns the attempt; a failed row draws per-row evidence
(never burned silently); cancel during execution = checkpoint-terminate at
the row boundary.

Submission-side spec contract (strictly validated, WP-09 Phase-EXEC-1):

    model: string           — catalog endpoint name (discover-ready chain via
                              EndpointRegistry; empty // unknown → 503-honest)
    input:                  — one of:
        prompts: [str, ...]  (max 500; inline list, pre-materialized)
        prompts_ref: "s3://bucket/key" → a .jsonl file, each row an object
                        with a "prompt" string field (fallback plain-text)
    max_content_len?: int    (truncate per-row prompt input; default 8000)
    continue_on_error?: bool (default true)

Output writes via the attempt-scoped credentialed scope only:
``jobs/<job_id>/attempt_<n>/results.jsonl`` (WP-11 mechanism — one row per
prompt with prompt-echo, output text, usage tokens, and a per-row ``ok`` +
error name when failed).

Honesty rules (same as the broader platform, stated publicly):

- Zero-success rows stay zero-rows: ``ok=false`` with a per-row error string;
  a batch is NOT "completed" unless its independent errors stay under the
  spec-published border post. `Continue_on_error` shut means the first error
  fails the attempt; never arrives there by accident.
- Cancel during ≤ loop N: the
  loop surfaces "rows_done" to the manifest even when partially complete —
  a cancelled batch is counted, not netted as "completed but slower".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..api.files import get_s3_config
from ..orchestrator.endpoint_registry import (
    EndpointStatus,
    get_registry,
)
from .download_safety import DownloadCancelledError
from .models import Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker

logger = logging.getLogger(__name__)

_MAX_INLINE_PROMPTS = 500
_PER_ROW_TIMEOUT_SEC = float(os.environ.get("UMRS_BATCH_ROW_TIMEOUT_SEC", "300"))


@dataclass
class BatchTaskContext:
    """Per-attempt batch context — nothing secret inside."""

    job_id: str
    attempt_id: str
    attempt_no: int
    project_id: str
    model: str
    prompts_ref: str
    prompts: list[str]
    max_content_len: int
    continue_on_error: bool
    endpoint_host: str
    endpoint_port: int
    endpoint_model: str
    endpoint_tier: str
    progress: dict[str, Any] = field(default_factory=dict)


class BatchTaskRegistry:
    """In-memory registry of in-flight attempts (matches the platform shape)."""

    def __init__(self) -> None:
        self._tasks: dict[str, BatchTaskContext] = {}
        self._lock = asyncio.Lock()

    async def register(self, ctx: BatchTaskContext) -> None:
        async with self._lock:
            self._tasks[ctx.attempt_id] = ctx

    async def get(self, attempt_id: str) -> BatchTaskContext | None:
        async with self._lock:
            return self._tasks.get(attempt_id)

    async def remove(self, attempt_id: str) -> BatchTaskContext | None:
        async with self._lock:
            return self._tasks.pop(attempt_id, None)

    async def list_tasks(self) -> list[BatchTaskContext]:
        async with self._lock:
            return list(self._tasks.values())


_registry: BatchTaskRegistry | None = None


def get_batch_registry() -> BatchTaskRegistry:
    """Get or create the global batch registry."""
    global _registry
    if _registry is None:
        _registry = BatchTaskRegistry()
    return _registry


def _read_prompts_from_s3(uri: str, client: Any) -> list[str]:
    """Download a prompts .jsonl and return prompt strings (plain fallback)."""
    if not uri.startswith("s3://"):
        raise ValueError("prompts_ref must be an s3:// URI")
    bucket_key = uri[len("s3://"):]
    bucket, _, key = bucket_key.partition("/")
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    prompts: list[str] = []
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("prompt"), str):
                prompts.append(row["prompt"])
                continue
        except json.JSONDecodeError:
            pass
        prompts.append(line)  # plain-text fallback, recorded as such
    return prompts


def _validate_inline(prompt_inline: Any) -> list[str]:
    """Inline lane shape validation (or error — never silent casting)."""
    if not isinstance(prompt_inline, list):
        raise ValueError("spec.prompts must be a list of strings")
    if not all(isinstance(p, str) for p in prompt_inline):
        raise ValueError("spec.prompts must be a list of strings")
    if len(prompt_inline) > _MAX_INLINE_PROMPTS:
        raise ValueError(f"spec.prompts capped at {_MAX_INLINE_PROMPTS} inline rows")
    return list(prompt_inline)


def _validate_input_lane(
    spec: dict[str, Any],
) -> tuple[str, list[str]]:
    """Prompt lane validation → (prompts_ref, inline_prompts); honest errors."""
    prompt_inline = spec.get("prompts")
    prompts_ref = str(spec.get("prompts_ref") or "").strip()
    if (not prompts_ref) and (not prompt_inline):
        raise ValueError("batch_infer requires spec.prompts OR spec.prompts_ref")
    if prompts_ref and prompt_inline:
        raise ValueError("batch_infer accepts only one input lane (prompts_ref XOR prompts)")
    if prompt_inline is None:
        return prompts_ref, []
    return prompts_ref, _validate_inline(prompt_inline)


def _resolve_ready_endpoint(
    project_id: str, model: str
) -> Any:
    """Discover the READY/DEGRADED endpoint for (project, model) or raise."""
    registry = get_registry()
    candidates = registry.list_all(
        project_id=project_id,
        status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
    )
    for ep in candidates:
        served_name = ep.metadata.get("served_model_name") or ep.model
        if ep.model == model or served_name == model:
            return ep
    raise ValueError(
        f"no READY/DEGRADED endpoint for model {model!r} under project {project_id!r}"
    )


def _build_context(job: Job, attempt: JobAttempt) -> BatchTaskContext:
    """Strict spec validation → execution context (never fabricate defaults)."""
    spec = job.spec or {}
    model = str(spec.get("model") or "").strip()
    if not model:
        raise ValueError("batch_infer requires spec.model (endpoint catalog name)")
    project_id = job.project_id or spec.get("project_id") or ""
    prompts_ref, prompt_inline = _validate_input_lane(spec)
    endpoint = _resolve_ready_endpoint(project_id, model)
    return BatchTaskContext(
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
        attempt_no=attempt.attempt_no,
        project_id=project_id,
        model=model,
        prompts_ref=prompts_ref,
        prompts=prompt_inline,
        max_content_len=int(spec.get("max_content_len", 8000)),
        continue_on_error=bool(spec.get("continue_on_error", True)),
        endpoint_host=endpoint.host,
        endpoint_port=endpoint.port,
        endpoint_model=endpoint.model,
        endpoint_tier=str(
            endpoint.metadata.get("tier") or endpoint.metadata.get("owner_type") or ""
        ),
    )


def _iter_s3_prompts(ctx: BatchTaskContext, config: Any) -> list[str]:
    if ctx.prompts:
        return list(ctx.prompts)
    client = config.get_client()
    return _read_prompts_from_s3(ctx.prompts_ref, client)


async def _run_one_row(
    client: httpx.AsyncClient,
    ctx: BatchTaskContext,
    prompt_text: str,
) -> dict[str, Any]:
    """One prompt → one output record. Honest row shape, never exception-swallowed."""
    prompt_text = prompt_text[: ctx.max_content_len]
    url = f"http://{ctx.endpoint_host}:{ctx.endpoint_port}/v1/chat/completions"
    payload = {
        "model": ctx.endpoint_model,
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
    }
    try:
        response = await client.post(url, json=payload, timeout=_PER_ROW_TIMEOUT_SEC)
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "error": f"transport: {exc}",
            "prompt": prompt_text,
        }
    if response.status_code != 200:
        return {
            "ok": False,
            "error": f"upstream_status_{response.status_code}: {response.text[:200]}",
            "prompt": prompt_text,
        }
    try:
        body = response.json()
        choices = body.get("choices") or []
        text = (choices[0].get("message", {}) or {}).get("content", "") if choices else ""
        usage = body.get("usage", {}) or {}
    except (ValueError, KeyError) as exc:
        return {"ok": False, "error": f"shape_mismatch: {exc}", "prompt": prompt_text}
    return {
        "ok": True,
        "prompt": prompt_text,
        "output": text,
        "usage": usage,
    }


async def _upload_results(ctx: BatchTaskContext, rows: list[dict[str, Any]]) -> str:
    """Write results.jsonl under the attempt-scoped S3 key; return the key."""
    from .attempt_credentials import scope_prefix_for

    config = get_s3_config()
    client = config.get_client()
    key = f"{scope_prefix_for(ctx.job_id, ctx.attempt_no)}results.jsonl"
    payload = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
    client.put_object(
        Bucket=config.bucket,
        Key=key,
        Body=payload.encode("utf-8"),
        ContentType="application/x-ndjson",
    )
    return key


class BatchInferBackend:
    """Extract-later executor for batch inference (WP-09 EXEC-1)."""

    def __init__(
        self,
        batch_registry: BatchTaskRegistry | None = None,
        model: str = "batch_infer",
    ) -> None:
        self.batch_registry = batch_registry or get_batch_registry()
        self._model_tag = model
        self._handles: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        from .download_safety import register_cancel_listener

        register_cancel_listener(self._on_durable_cancel)

    @property
    def backend_name(self) -> str:
        return self._model_tag

    def _on_durable_cancel(self, job_id: str) -> None:
        """Cancel-bus listener (6b-3b symmetry): our jobs get interrupted."""
        for attempt_id, ctx in list(self.batch_registry._tasks.items()):
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
        """Submit one batch-inference attempt (inline; the task handles the loop)."""
        ctx = _build_context(job, attempt)
        await self.batch_registry.register(ctx)
        self._handles[attempt.attempt_id] = asyncio.create_task(self._execute(ctx))
        logger.info(
            "Submitted batch_infer: job=%s attempt=%s model=%s backend=%s",
            job.job_id,
            attempt.attempt_id,
            ctx.model,
            self.backend_name,
        )
        return attempt.attempt_id

    async def _drive_rows(self, ctx: BatchTaskContext) -> tuple[list[dict[str, Any]], str]:
        """Run the per-row loop (cancel-aware, progress-shaped) → (rows, s3_key)."""
        rows: list[dict[str, Any]] = []
        config = get_s3_config()
        prompts = _iter_s3_prompts(ctx, config)
        total = len(prompts)
        async with httpx.AsyncClient() as client:
            for idx, prompt in enumerate(prompts):
                if ctx.attempt_id in self._cancelled:
                    raise DownloadCancelledError(f"attempt {ctx.attempt_id} cancelled")
                row = await _run_one_row(client, ctx, prompt)
                rows.append(row)
                if not row["ok"] and not ctx.continue_on_error:
                    raise RuntimeError(
                        f"batch_infer stopped at row {idx + 1}/{total}: {row['error']}"
                    )
                self._refresh_progress(ctx, rows, total, idx + 1)
        s3_key = await _upload_results(ctx, rows)
        return rows, s3_key

    @staticmethod
    def _refresh_progress(
        ctx: BatchTaskContext, rows: list[dict[str, Any]], total: int, done: int
    ) -> None:
        """Cheap per-row progress payload — never anything but the counters."""
        ctx.progress = {
            "rows_total": total,
            "rows_done": done,
            "rows_ok": sum(1 for r in rows if r["ok"]),
            "rows_failed": sum(1 for r in rows if not r["ok"]),
            "lane_width": "jsonl",
        }

    async def _execute(self, ctx: BatchTaskContext) -> None:
        """Run the loop with cancel-bus + row-boundary honesty."""
        try:
            rows, s3_key = await self._drive_rows(ctx)
            ctx.progress = {
                **ctx.progress,
                "status": "completed",
                "s3_key": s3_key,
            }
        except DownloadCancelledError as exc:
            ctx.progress = {
                **ctx.progress,
                "status": "cancelled",
                "cancel_reason": str(exc),
            }
        except Exception as exc:
            logger.exception("batch_infer failed for %s", ctx.attempt_id)
            ctx.progress = {
                **ctx.progress,
                "status": "failed",
                "error": str(exc),
            }
        finally:
            self._handles.pop(ctx.attempt_id, None)
            await self.batch_registry.remove(ctx.attempt_id)


class BatchInferStatusChecker(BackendStatusChecker):

    """Read the batch registry's progress into terminal-or-running verdicts."""

    @property
    def job_type(self) -> JobType:
        return JobType.BATCH_INFER

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        registry = get_batch_registry()
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
            # In-flight or not yet told its first progress row.
            return AttemptCheckResult.running()
        return AttemptCheckResult.failed(str(state))


def create_batch_infer_backend() -> BatchInferBackend:
    """Factory hook for the durable runtime (matches parent-pattern naming)."""
    return BatchInferBackend()


def create_batch_infer_status_checker() -> BatchInferStatusChecker:
    """Factory hook for the composite status checker."""
    return BatchInferStatusChecker()
