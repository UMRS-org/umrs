"""
Step execution context and result types.

StepContext wraps the S3 connection and exposes convenient helpers so that
user scripts never have to manipulate boto3 directly.

StepResult carries the outputs of a single step execution back to the
WorkflowExecutor.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StepContext:
    """Execution context passed to :meth:`UMRSStep.run`.

    Attributes:
        s3: A boto3-compatible S3 client (MinIO).
        input_bucket: Bucket that contains the objects to process.
        input_keys: S3 keys to process in this batch.
        output_bucket: Bucket where outputs should be written.
        config: Step-specific configuration from *workflow.json*.
        metadata: Mutable dict carried across steps for the same run.
        run_id: Unique identifier for the current workflow execution.
    """

    s3: Any  # boto3 S3 client
    input_bucket: str
    input_keys: List[str]
    output_bucket: str
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    run_id: str = ""

    # ---- S3 download helpers ------------------------------------------------

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

    # ---- S3 upload helpers ---------------------------------------------------

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

    # ---- S3 metadata helpers -------------------------------------------------

    def head_object(self, key: str, bucket: Optional[str] = None) -> Dict[str, Any]:
        """Return S3 HEAD metadata for *key*."""
        bkt = bucket or self.input_bucket
        return self.s3.head_object(Bucket=bkt, Key=key)


@dataclass
class StepResult:
    """Value returned by :meth:`UMRSStep.run`.

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
