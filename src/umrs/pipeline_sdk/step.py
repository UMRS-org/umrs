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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# ── Step execution context ───────────────────────────────────────────────────

@dataclass
class StepContext:
    """Execution context passed to :meth:`Step.run`.

    Attributes:
        s3: A boto3-compatible S3 client (MinIO).
        input_bucket: Bucket that contains the objects to process.
        input_keys: S3 keys to process in this batch.
        output_bucket: Bucket where outputs should be written.
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
    input_keys: List[str]
    output_bucket: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    parent_results: Dict[str, "StepResult"] = field(default_factory=dict)

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
            self.run_id, pct, message,
        )

    # ── S3 download helpers ──────────────────────────────────────────────

    def download_bytes(self, key: str, bucket: Optional[str] = None) -> bytes:
        """Download an S3 object and return its raw bytes."""
        bkt = bucket or self.input_bucket
        obj = self.s3.get_object(Bucket=bkt, Key=key)
        return obj["Body"].read()

    def download_numpy(self, key: str, bucket: Optional[str] = None) -> "np.ndarray":
        """Download a ``.npy`` file from S3 and return a NumPy array."""
        import numpy as np
        raw = self.download_bytes(key, bucket)
        return np.load(io.BytesIO(raw), allow_pickle=False)

    # ── S3 upload helpers ────────────────────────────────────────────────

    def upload_bytes(
        self,
        key: str,
        data: bytes,
        bucket: Optional[str] = None,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Upload raw bytes to S3 and return the full ``s3://`` URI."""
        bkt = bucket or self.output_bucket
        extra: Dict[str, Any] = {"ContentType": content_type}
        if metadata:
            extra["Metadata"] = metadata
        self.s3.put_object(Bucket=bkt, Key=key, Body=data, **extra)
        return f"s3://{bkt}/{key}"

    def upload_numpy(
        self,
        key: str,
        arr: "np.ndarray",
        bucket: Optional[str] = None,
    ) -> str:
        """Save a NumPy array as ``.npy`` to S3."""
        import numpy as np
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return self.upload_bytes(key, buf.getvalue(), bucket,
                                content_type="application/x-npy")

    # ── S3 metadata helpers ──────────────────────────────────────────────

    def head_object(self, key: str, bucket: Optional[str] = None) -> Dict[str, Any]:
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

    output_keys: List[str] = field(default_factory=list)
    embeddings: Optional[Any] = None  # numpy.ndarray when set
    checksums: List[str] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


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
