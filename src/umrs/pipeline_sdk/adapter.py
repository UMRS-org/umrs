"""
EmbeddingAdapter — wraps any Pipeline into the ``compute_embeddings`` contract.

This adapter bridges the unified Pipeline SDK with the UMRS server's
``EmbeddingModel.compute_embeddings`` interface, so that ``load_model()``
in ``models_loading.py`` can bind a workflow-backed Pipeline as a
standard embedding model.

Usage in ``models_loading.py``::

    from umrs.pipeline_sdk.workflow import WorkflowPipeline
    from umrs.pipeline_sdk.adapter import EmbeddingAdapter

    pipeline = WorkflowPipeline(source_uri)
    compute_fn = EmbeddingAdapter(pipeline).compute_embeddings
    # → bind to EmbeddingModel
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from umrs.pipeline_sdk.workflow import WorkflowPipeline

logger = logging.getLogger(__name__)


class EmbeddingAdapter:
    """Adapt a :class:`WorkflowPipeline` to the ``compute_embeddings`` signature.

    The adapter simply delegates to :meth:`WorkflowPipeline.run_as_embeddings`,
    which returns the 4-tuple expected by ``compute_and_store_embedding()``
    in ``embeddings.py``.
    """

    def __init__(self, pipeline: WorkflowPipeline) -> None:
        self.pipeline = pipeline

    def compute_embeddings(
        self,
        s3: Any,
        s3_keys: List[str],
        path_to_bucket: str,
        output_bucket: str,
    ) -> Tuple[List[Any], List[str], List[Dict[str, Any]], List[Optional[str]]]:
        """Drop-in replacement for ``EmbeddingModel.compute_embeddings``.

        Args:
            s3: boto3-compatible S3 client (MinIO).
            s3_keys: Input S3 keys to process.
            path_to_bucket: Source bucket name.
            output_bucket: Destination bucket for final results.

        Returns:
            ``(embeddings, checksums, metadata_list, saved_keys)``
        """
        return self.pipeline.run_as_embeddings(
            s3, s3_keys, path_to_bucket, output_bucket,
        )
