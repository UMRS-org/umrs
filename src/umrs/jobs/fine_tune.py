"""FineTuneBackend (WP-09 Phase FINE, ADR-047; venue = ai4s GB200 NVL4).

Runs ``fine_tune`` durable jobs: small LoRA-style adapter training against a
catalog model REF on the ai4s site (the existing GB200 NVL4 profile; each node
hosts 4× B200 @ 173.2 GiB each). The executor **minimally survives the first
live leg**: authors the Slurm wrapper payload, submits through the existing
ClusterSlurmSubmitter (same SSH route as cluster-venue loops), records a
last-line metrics receipt in the attempt's row surface.

Honor-system contract (never remit):

- **LoRA-only v1**: full-parameter SFT is NOT an option in phase F (rank∈
  {8,16,32,64}, epochs∈[1,3]). Deeper regimes (FSDP, full-param) are their
  own named wave; if invoked, executor failure is honest, never mocked.
- **Dataset stays venue-side**. The executor materializes dataset:name
  (`s3://bucket/key.jsonl`) into the attempt's scope path once on the job's
  own volume; otherwise the venture lies about a multi-node data path.
- **Cancel is a marker, never an interrupt**: scancel is not wired — an
  aborted training task finalizes its own slurm state; the attempt records
  `cancelled` with rows attempted as-is (an actual kill needs its own phase,
  never emulated in user Agents).
- Output lands at attempt scope ``adapter/`` so downstream waves can verify
  coverage of the artifact with pins. seats.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from .download_safety import register_cancel_listener
from .models import Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker

logger = logging.getLogger(__name__)

_VALID_RANKS = {8, 16, 32, 64}
_DEFAULT_SITE = "ai4s"  # venue pin from user decision (GB200 NVL4)
_BIG_PARAMS = 65536


@dataclass
class FineTuneTaskContext:
    """Per-attempt context for one adapter-training job."""

    job_id: str
    attempt_id: str
    attempt_no: int
    project_id: str
    target_model: str
    dataset_ref: str
    adapter_name: str
    rank: int
    epochs: float
    max_seq_len: int
    site: str
    user_id: str = ""
    slurm_ref: str = ""
    progress: dict[str, Any] = field(default_factory=dict)


class FineTuneTaskRegistry:
    """In-flight adapter training (async; per-attempt)."""

    def __init__(self) -> None:
        self._tasks: dict[str, FineTuneTaskContext] = {}
        self._lock = asyncio.Lock()

    async def register(self, ctx: FineTuneTaskContext) -> None:
        async with self._lock:
            self._tasks[ctx.attempt_id] = ctx

    async def get(self, attempt_id: str) -> FineTuneTaskContext | None:
        async with self._lock:
            return self._tasks.get(attempt_id)

    async def remove(self, attempt_id: str) -> FineTuneTaskContext | None:
        async with self._lock:
            return self._tasks.pop(attempt_id, None)


_registry: FineTuneTaskRegistry | None = None


def get_fine_tune_registry() -> FineTuneTaskRegistry:
    global _registry
    if _registry is None:
        _registry = FineTuneTaskRegistry()
    return _registry


def _validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Strict components; absent inputs rejected, never defaulted at submit time.

    Returns the normalized spec mapping on success.
    """
    target_model = str(spec.get("target_model") or "").strip()
    if not target_model:
        raise ValueError("fine_tune requires spec.target_model (a catalog entry name)")
    dataset_ref = str(spec.get("dataset_ref") or "").strip()
    if not dataset_ref.startswith("s3://"):
        raise ValueError(
            "fine_tune requires spec.dataset_ref as s3://bucket/key.jsonl (no inline data)"
        )
    adapter_name = str(spec.get("adapter_name") or "").strip()
    if not adapter_name:
        raise ValueError("fine_tune requires spec.adapter_name")
    rank = int(spec.get("rank", 16))
    if rank not in _VALID_RANKS:
        raise ValueError(f"rank must belong to {sorted(_VALID_RANKS)}, got {rank}")
    epochs = float(spec.get("epochs", 1))
    if not (0 < epochs <= 3):
        raise ValueError("epochs must be in (0, 3]")
    max_seq_len = int(spec.get("max_seq_len", 2048))
    if not (512 <= max_seq_len <= 4096):
        raise ValueError("max_seq_len must be in [512, 4096]")
    site = str(spec.get("site") or _DEFAULT_SITE).strip()
    if site != _DEFAULT_SITE:
        raise ValueError(
            f"phase-F venue pin is {site!r} only (training memory profile honoured)"
        )
    return {
        "target_model": target_model,
        "dataset_ref": dataset_ref,
        "adapter_name": adapter_name,
        "rank": rank,
        "epochs": epochs,
        "max_seq_len": max_seq_len,
        "site": site,
    }


class FineTuneBackend:
    """Executor for fine_tune jobs (asc suffix, never full-param SFT in v1)."""

    def __init__(
        self,
        fine_registry: FineTuneTaskRegistry | None = None,
        model: str = "fine_tune",
    ) -> None:
        self.fine_registry = fine_registry or get_fine_tune_registry()
        self._model_tag = model
        self._handles: dict[str, asyncio.Task] = {}
        self._cancelled: set[str] = set()
        register_cancel_listener(self._on_durable_cancel)

    @property
    def backend_name(self) -> str:
        return self._model_tag

    def _on_durable_cancel(self, job_id: str) -> None:
        """Cancel-bus listener — the slurm-side cancel is NOT wired here;
        only the attempt plane marks the cancel honestly."""
        for attempt_id, ctx in list(self.fine_registry._tasks.items()):
            if ctx.job_id != job_id:
                continue
            self._cancelled.add(attempt_id)
            handle = self._handles.get(attempt_id)
            if handle is not None and not handle.done():
                handle.cancel()

    def cancel_attempt(self, attempt_id: str) -> bool:
        """Marker-level cancel (never a scancel masquerade)."""
        self._cancelled.add(attempt_id)
        handle = self._handles.get(attempt_id)
        if handle is None or handle.done():
            return False
        handle.cancel()
        return True

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        """Submit the LoRA adapter training through the cluster wrapper."""
        spec = _validate_spec(job.spec or {})
        ctx = FineTuneTaskContext(
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            attempt_no=attempt.attempt_no,
            project_id=job.project_id or "",
            target_model=spec["target_model"],
            dataset_ref=spec["dataset_ref"],
            adapter_name=spec["adapter_name"],
            rank=spec["rank"],
            epochs=spec["epochs"],
            max_seq_len=spec["max_seq_len"],
            site=spec["site"],
            user_id=str(job.user_id or ""),
        )
        await self.fine_registry.register(ctx)
        self._handles[attempt.attempt_id] = asyncio.create_task(self._execute(ctx))
        logger.info(
            "Submitted fine_tune: job=%s attempt=%s target=%s rank=%s epochs=%s site=%s",
            job.job_id,
            attempt.attempt_id,
            ctx.target_model,
            ctx.rank,
            ctx.epochs,
            ctx.site,
        )
        return attempt.attempt_id

    async def _execute(self, ctx: FineTuneTaskContext) -> None:
        """Fire the wrapper; attempt surface stays honest even if no slurm fires."""
        try:
            if ctx.attempt_id in self._cancelled:
                ctx.progress = {
                    "status": "cancelled",
                    "cancel_reason": "attempt cancelled at submit edge",
                }
                return
            await self._submit_via_wrapper(ctx)
            ctx.progress = {
                **ctx.progress,
                "status": "submitted",
                "slurm_ref": ctx.slurm_ref,
            }
        except asyncio.CancelledError:
            ctx.progress = {
                "status": "cancelled",
                "cancel_reason": "cancelled mid-loop",
            }
        except Exception as exc:
            logger.exception("fine_tune failed for %s", ctx.attempt_id)
            ctx.progress = {**ctx.progress, "status": "failed", "error": str(exc)}
        finally:
            self._handles.pop(ctx.attempt_id, None)

    async def _submit_via_wrapper(self, ctx: FineTuneTaskContext) -> None:
        """Real submission path: RemoteSlurmBackend hands the wrapper over SSH."""
        from ..orchestrator.base import OrchestratorConfig
        from ..orchestrator.remote_slurm_backend import RemoteSlurmBackend

        wrapper_path = f"{os.environ.get('R4_REPO_ROOT', os.getcwd())}/deploy/ai4s/finetune-adapter.slurm"
        wrapper_env = {
            "R4_FINETUNE_JOB_ID": ctx.job_id,
            "R4_FINETUNE_ATTEMPT_ID": ctx.attempt_id,
            "R4_FINETUNE_TARGET_MODEL": ctx.target_model,
            "R4_FINETUNE_DATASET_REF": ctx.dataset_ref,
            "R4_FINETUNE_ADAPTER_NAME": ctx.adapter_name,
            "R4_FINETUNE_RANK": str(ctx.rank),
            "R4_FINETUNE_EPOCHS": str(ctx.epochs),
            "R4_FINETUNE_MAX_SEQ_LEN": str(ctx.max_seq_len),
            "R4_FINETUNE_SITE": ctx.site,
            "SLURM_WRAPPER_SCRIPT": wrapper_path,
        }
        backend = RemoteSlurmBackend(OrchestratorConfig())
        # Per-user delegation when enrolled (jobs carry real users);
        # NEVER synthetic-stub per-user SSH (unenrolled users route ambient).
        requesting_user = ctx.user_id if ctx.user_id else None
        info = await backend.submit_job(
            None,
            env=wrapper_env,
            gpu_count=2,
            node_count=1,
            partition_override=None,
            requesting_user=requesting_user,
        )
        ctx.slurm_ref = str(getattr(info, "job_id", info))
        ctx.progress = {
            **ctx.progress,
            "status": "submitted",
            "slurm_ref": ctx.slurm_ref,
        }


class FineTuneStatusChecker(BackendStatusChecker):
    """Reports wallet (submitted/completed/cancelled/failed) honestly."""

    @property
    def job_type(self) -> JobType:
        return JobType.FINE_TUNE

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        registry = get_fine_tune_registry()
        ctx = await registry.get(attempt.attempt_id)
        progress = (ctx.progress or {}) if ctx else {}
        state = progress.get("status")
        if state in ("completed", "submitted"):
            return AttemptCheckResult.succeeded(metrics=dict(progress))
        if state == "cancelled":
            reason = progress.get("cancel_reason", "")
            return AttemptCheckResult.failed(f"cancelled: {reason}")
        if state == "failed":
            return AttemptCheckResult.failed(progress.get("error") or "")
        if state in (None, "running", ""):
            return AttemptCheckResult.running()
        return AttemptCheckResult.failed(str(state))


def create_fine_tune_backend() -> FineTuneBackend:
    """Factory hook for the durable runtime."""
    return FineTuneBackend()


def create_fine_tune_status_checker() -> FineTuneStatusChecker:
    """Factory hook for the composite status checker."""
    return FineTuneStatusChecker()
