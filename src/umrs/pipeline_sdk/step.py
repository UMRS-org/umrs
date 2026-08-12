"""
Step interface — abstract base class for user-defined processing steps.

Users implement ``Step`` subclasses in their own Python scripts, then
reference them from a ``workflow.json`` file.  UMRS discovers, loads, and
executes these steps via :class:`WorkflowPipeline` without any internal
code changes.

Typical usage in a user script::

    from umrs.pipeline_sdk import Step, StepContext, StepResult

    class MyPreprocess(Step):
        name = "preprocess"

        def setup(self, config: dict) -> None:
            self.size = config.get("target_size", 1024)

        def run(self, ctx: StepContext) -> StepResult:
            out_keys = []
            for key in ctx.input_keys:
                data = ctx.download_bytes(key)
                processed = self._do_work(data)
                out_key = f"preprocessed/{key}"
                ctx.upload_bytes(out_key, processed)
                out_keys.append(out_key)
            return StepResult(output_keys=out_keys)
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepPolicy:
    """Per-step execution policy shared by all executors (ADR-021).

    Members are normalized by ``from_mapping`` so executor-duplicated
    parsing semantics (invalid values fall back with a warning, clamps)
    live in exactly one place. Gate steps are NOT policy — detection of
    manual-approval steps stays at the executor boundary.
    """

    max_retries: int = 0
    retry_delay: float = 1.0
    timeout_seconds: float | None = None

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any]) -> "StepPolicy":
        """Parse a raw step spec (workflow.json dict) into a StepPolicy."""
        return cls(
            max_retries=_int_field(spec, "max_retries", 0, minimum=0),
            retry_delay=_float_field(spec, "retry_delay", 1.0, minimum=0.0),
            timeout_seconds=_timeout_field(spec),
        )


def _int_field(
    spec: Mapping[str, Any], name: str, default: int, *, minimum: int | None
) -> int:
    """Coerce an int policy field; invalid values fall back with a warning."""
    raw = spec.get(name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Step %s: invalid %s=%r; using %r",
            spec.get("name", "?"),
            name,
            raw,
            default,
        )
        return default
    if minimum is not None and value < minimum:
        logger.warning(
            "Step %s: %s=%r below minimum %r; clamped",
            spec.get("name", "?"),
            name,
            raw,
            minimum,
        )
        return minimum
    return value


def _float_field(
    spec: Mapping[str, Any], name: str, default: float, *, minimum: float | None
) -> float:
    """Coerce a float policy field; invalid values fall back with a warning."""
    raw = spec.get(name, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Step %s: invalid %s=%r; using %r",
            spec.get("name", "?"),
            name,
            raw,
            default,
        )
        return default
    if minimum is not None and value < minimum:
        logger.warning(
            "Step %s: %s=%r below minimum %r; clamped",
            spec.get("name", "?"),
            name,
            raw,
            minimum,
        )
        return minimum
    return value


def _timeout_field(spec: Mapping[str, Any]) -> float | None:
    """Coerce timeout_seconds; non-positive/invalid → None with a warning."""
    raw = spec.get("timeout_seconds")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Step %s: invalid timeout_seconds=%r; ignoring",
            spec.get("name", "?"),
            raw,
        )
        return None
    if value <= 0:
        logger.warning(
            "Step %s: timeout_seconds=%r non-positive; ignoring",
            spec.get("name", "?"),
            raw,
        )
        return None
    return value


def call_with_deadline(
    fn: Callable[[], Any],
    *,
    timeout_seconds: float | None,
    label: str = "step",
) -> Any:
    """Run ``fn()`` with an optional soft deadline.

    ``timeout_seconds=None``/``<=0`` runs inline with no thread
    overhead.  A breach raises :class:`TimeoutError`.  The underlying
    work is **not** interrupted (threads cannot be killed safely in
    CPython); the late result is discarded and the worker unwinds on
    its own.  Treat this as an advisory timeout: failure is reported
    at the deadline, background completion is best-effort.
    """
    if not timeout_seconds or timeout_seconds <= 0:
        return fn()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="umrs-step-deadline")
    future = pool.submit(fn)
    try:
        return future.result(timeout=float(timeout_seconds))
    except FutureTimeoutError as exc:
        raise TimeoutError(f"{label} exceeded deadline of {timeout_seconds}s") from exc
    finally:
        pool.shutdown(wait=False)


# ── Step execution context ───────────────────────────────────────────────────


@dataclass
class StepContext:
    """Execution context passed to :meth:`Step.run`.

    Attributes:
        s3: A boto3-compatible S3 client (MinIO).
        input_bucket: Source/trigger bucket (inventory, videos, plans).
            Always the workflow input bucket — never rewritten to a
            parent step's output bucket.
        input_keys: Upstream artifact keys (parent ``output_keys`` for
            child steps; empty for root steps).
        output_bucket: Bucket where this step writes outputs.
        parent_output_bucket: Bucket that holds parent ``output_keys``
            for the common ``{input_bucket}-output`` pattern. Empty for
            root steps. Use :meth:`download_parent_bytes` / this field
            for upstream artifacts; keep :meth:`download_bytes` for
            source-bucket objects.
        config: Step-specific configuration from *workflow.json*.
        metadata: Mutable dict carried across steps for the same run.
            Note: under the parallel workflow runner (independent steps
            in the same DAG wave run concurrently) this dict is **not**
            safe for cross-step mutation. Use ``parent_results`` to pass
            data forward in fan-in scenarios; reserve ``metadata`` for
            within-step bookkeeping.
        run_id: Unique identifier for the current workflow execution.
        parent_results: Results from parent steps in DAG mode (keyed by step name).
    """

    s3: Any  # boto3 S3 client
    input_bucket: str
    input_keys: list[str]
    output_bucket: str
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    parent_results: dict[str, StepResult] = field(default_factory=dict)
    parent_output_bucket: str = ""

    # ── Progress reporting ───────────────────────────────────────────────

    def report_progress(self, percent: float, message: str = "") -> None:
        """Emit a progress update for monitoring.

        Args:
            percent: Completion percentage (0.0 – 100.0).
            message: Optional human-readable status message.
        """
        pct = max(0.0, min(100.0, percent))
        logger.info(
            "[run=%s] progress %.1f%% %s",
            self.run_id,
            pct,
            message,
        )

    # ── S3 download helpers ──────────────────────────────────────────────

    def download_bytes(self, key: str, bucket: str | None = None) -> bytes:
        """Download an S3 object and return its raw bytes.

        Defaults to :attr:`input_bucket` (source/trigger objects).
        """
        bkt = bucket or self.input_bucket
        obj = self.s3.get_object(Bucket=bkt, Key=key)
        return obj["Body"].read()

    def download_parent_bytes(self, key: str, bucket: str | None = None) -> bytes:
        """Download a parent-step artifact.

        Defaults to :attr:`parent_output_bucket`, falling back to
        :attr:`output_bucket` (the shared ``{input}-output`` pattern)
        when the parent bucket was not set explicitly.
        """
        bkt = bucket or self.parent_output_bucket or self.output_bucket
        if not bkt:
            raise ValueError(
                "No parent_output_bucket/output_bucket available for parent artifact download"
            )
        obj = self.s3.get_object(Bucket=bkt, Key=key)
        return obj["Body"].read()

    def download_numpy(self, key: str, bucket: str | None = None) -> np.ndarray:
        """Download a ``.npy`` file from S3 and return a NumPy array."""
        import numpy as np

        raw = self.download_bytes(key, bucket)
        return np.load(io.BytesIO(raw), allow_pickle=False)

    # ── S3 upload helpers ────────────────────────────────────────────────

    def upload_bytes(
        self,
        key: str,
        data: bytes,
        bucket: str | None = None,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload raw bytes to S3 and return the full ``s3://`` URI."""
        bkt = bucket or self.output_bucket
        extra: dict[str, Any] = {"ContentType": content_type}
        if metadata:
            extra["Metadata"] = metadata
        self.s3.put_object(Bucket=bkt, Key=key, Body=data, **extra)
        return f"s3://{bkt}/{key}"

    def upload_numpy(
        self,
        key: str,
        arr: np.ndarray,
        bucket: str | None = None,
    ) -> str:
        """Save a NumPy array as ``.npy`` to S3."""
        import numpy as np

        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return self.upload_bytes(
            key, buf.getvalue(), bucket, content_type="application/x-npy"
        )

    # ── S3 metadata helpers ──────────────────────────────────────────────

    def head_object(self, key: str, bucket: str | None = None) -> dict[str, Any]:
        """Return S3 HEAD metadata for *key*."""
        bkt = bucket or self.input_bucket
        return self.s3.head_object(Bucket=bkt, Key=key)


@dataclass
class StepResult:
    """Value returned by :meth:`Step.run`.

    Attributes:
        output_keys: S3 keys produced by this step (becomes the next
            step's ``input_keys``).
        embeddings: Optional embedding matrix (*N* × *dim*); only the
            **last** step to produce embeddings is stored in pgvector.
        checksums: Per-key checksums for idempotency tracking.
        metadata: Per-key metadata dicts.
        extra: Arbitrary key-value bag (logging, metrics, …).
    """

    output_keys: list[str] = field(default_factory=list)
    embeddings: Any | None = None  # numpy.ndarray when set
    checksums: list[str] = field(default_factory=list)
    metadata: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON storage in traces.

        The ``embeddings`` field is excluded (too large / not JSON-safe).
        """
        return {
            "output_keys": self.output_keys,
            "checksums": self.checksums,
            "metadata": self.metadata,
            "extra": self.extra,
        }


# ── Step abstract base class ────────────────────────────────────────────────


class Step(ABC):
    """Abstract base class every user-defined step must inherit."""

    # Subclasses MUST set a unique name (used as key in workflow.json).
    name: str = ""

    def setup(self, config: dict) -> None:
        """Optional one-time initialisation (load weights, parse config, …).

        Called once after instantiation, before any ``run()`` calls.
        Override this instead of ``__init__`` so the loader can instantiate
        the class without arguments.
        """

    @abstractmethod
    def run(self, ctx: StepContext) -> StepResult:
        """Execute the step on one batch of S3 objects.

        Args:
            ctx: A :class:`StepContext` that provides S3 helpers,
                 ``input_keys``, ``output_bucket``, and a shared
                 ``metadata`` dict for passing state between steps.

        Returns:
            A :class:`StepResult` describing what was produced.
        """

    def teardown(self) -> None:
        """Optional cleanup (release GPU memory, close files, …).

        Called once after **all** batches have been processed for this step.
        """
