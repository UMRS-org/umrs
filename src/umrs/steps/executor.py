"""
Workflow executor — sequences user-defined steps with S3 I/O hand-off (U-204).

The executor is the bridge between the UMRS ``compute_embeddings`` contract
(used by ``embeddings.py``) and the user's chain of ``UMRSStep`` instances.

Execution flow
--------------
1.  For each step *i* in order:
    a. Build a :class:`StepContext` whose ``input_keys`` are either the
       original S3 keys (step 0) or the previous step's ``output_keys``.
    b. Call ``step.run(ctx)``.
    c. Collect the :class:`StepResult`.
2.  After all steps, collect the final ``output_keys``, embeddings,
    checksums, and metadata and return them in the shape expected by
    ``compute_and_store_embedding()``.

Staging
-------
Intermediate outputs land under a namespaced prefix inside the output
bucket so they don't collide with other runs::

    s3://<output_bucket>/_staging/<run_id>/step_<i>_<name>/…

The final step writes directly to the output bucket root (or a
user-chosen prefix).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from umrs.steps.context import StepContext, StepResult
from umrs.steps.interface import UMRSStep
from umrs.steps.models import WorkflowDefinition
from umrs.steps.sandbox import SandboxConfig, SubprocessSandbox

logger = logging.getLogger(__name__)


class WorkflowExecutor:
    """Runs an ordered list of :class:`UMRSStep` instances."""

    def __init__(
        self,
        workflow: WorkflowDefinition,
        steps: List[UMRSStep],
        sandbox: Optional[SandboxConfig] = None,
        s3_params: Optional[Dict[str, str]] = None,
    ) -> None:
        if len(steps) != len(workflow.steps):
            raise ValueError(
                f"Step count mismatch: workflow declares {len(workflow.steps)} "
                f"steps but {len(steps)} instances were provided"
            )
        self.workflow = workflow
        self.steps = steps
        self.sandbox = sandbox
        self.s3_params = s3_params  # needed for subprocess mode
        self._subprocess_sandbox: Optional[SubprocessSandbox] = None
        if sandbox and sandbox.enabled:
            self._subprocess_sandbox = SubprocessSandbox(sandbox)

    # ------------------------------------------------------------------
    # Public: compute_embeddings-compatible method
    # ------------------------------------------------------------------

    def run(
        self,
        s3: Any,
        s3_keys: List[str],
        path_to_bucket: str,
        output_bucket: str,
    ) -> Tuple[List[Any], List[str], List[Dict[str, Any]], List[Optional[str]]]:
        """Execute the full workflow — drop-in for ``EmbeddingModel.compute_embeddings``.

        Args:
            s3: boto3-compatible S3 client (MinIO).
            s3_keys: Input S3 keys to process.
            path_to_bucket: Source bucket name.
            output_bucket: Destination bucket for final results.

        Returns:
            ``(embeddings, checksums, metadata_list, saved_keys)`` matching
            the contract of ``compute_and_store_embedding`` in
            ``embeddings.py``.
        """
        run_id = uuid.uuid4().hex[:12]
        logger.info(
            "Workflow '%s' run=%s: %d steps, %d input keys",
            self.workflow.name, run_id, len(self.steps), len(s3_keys),
        )

        current_keys = list(s3_keys)
        current_bucket = path_to_bucket
        shared_metadata: Dict[str, Any] = {}

        last_embeddings: Optional[Any] = None
        last_checksums: List[str] = []
        last_metadata: List[Dict[str, Any]] = []

        for idx, (step, spec) in enumerate(
            zip(self.steps, self.workflow.steps)
        ):
            is_last = idx == len(self.steps) - 1
            step_output_bucket = (
                output_bucket
                if is_last
                else path_to_bucket  # staging goes in source bucket
            )
            step_prefix = (
                ""
                if is_last
                else f"_staging/{run_id}/step_{idx}_{spec.name}/"
            )

            logger.info(
                "  step %d/%d [%s]: %d keys → bucket=%s",
                idx + 1, len(self.steps), spec.name,
                len(current_keys), step_output_bucket,
            )

            if self._subprocess_sandbox and self.s3_params:
                # --- Sandboxed execution (subprocess) ---
                result = self._subprocess_sandbox.run_step(
                    entrypoint=spec.entrypoint,
                    step_config=spec.config,
                    script_dir=str(self.workflow.source.path),
                    input_bucket=current_bucket,
                    input_keys=current_keys,
                    output_bucket=step_output_bucket,
                    s3_params=self.s3_params,
                    run_id=run_id,
                    shared_metadata=shared_metadata,
                    env=self.workflow.env,
                )
            else:
                # --- In-process execution (dev / lightweight) ---
                ctx = StepContext(
                    s3=s3,
                    input_bucket=current_bucket,
                    input_keys=current_keys,
                    output_bucket=step_output_bucket,
                    config=spec.config,
                    metadata=shared_metadata,
                    run_id=run_id,
                )
                result = step.run(ctx)

            # Chain outputs → next step's inputs
            if result.output_keys:
                current_keys = result.output_keys
                current_bucket = step_output_bucket

            # Keep the latest non-None embeddings
            if result.embeddings is not None:
                last_embeddings = result.embeddings

            # Keep metadata & checksums from the last step that provides them
            if result.checksums:
                last_checksums = result.checksums
            if result.metadata:
                last_metadata = result.metadata

        # -- Build return tuple matching bentoml_adapter / vision contract --

        # If no step produced checksums, generate them from the original keys
        if not last_checksums:
            last_checksums = self._collect_checksums(s3, path_to_bucket, s3_keys)

        # If no step produced per-key metadata, generate empty dicts
        if not last_metadata:
            last_metadata = self._collect_metadata(s3, path_to_bucket, s3_keys)

        # Embeddings: list of vectors (or None per key)
        if last_embeddings is not None:
            embeddings_list: List[Any] = list(last_embeddings)
        else:
            embeddings_list = [None] * len(s3_keys)

        # Saved keys: the final step's output keys (or None if pass-through)
        saved_keys: List[Optional[str]] = (
            list(current_keys)
            if current_keys != list(s3_keys)
            else [None] * len(s3_keys)
        )

        # Call teardown on each step
        for step in self.steps:
            try:
                step.teardown()
            except Exception:
                logger.warning("teardown failed for step '%s'", step.name,
                               exc_info=True)

        logger.info("Workflow '%s' run=%s complete.", self.workflow.name, run_id)
        return embeddings_list, last_checksums, last_metadata, saved_keys

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_checksums(
        s3: Any, bucket: str, keys: List[str],
    ) -> List[str]:
        checksums = []
        for key in keys:
            try:
                obj = s3.head_object(Bucket=bucket, Key=key)
                checksums.append(obj["ETag"].strip('"'))
            except Exception:
                checksums.append("")
        return checksums

    @staticmethod
    def _collect_metadata(
        s3: Any, bucket: str, keys: List[str],
    ) -> List[Dict[str, Any]]:
        meta = []
        for key in keys:
            try:
                obj = s3.head_object(Bucket=bucket, Key=key)
                meta.append(obj.get("Metadata", {}))
            except Exception:
                meta.append({})
        return meta
