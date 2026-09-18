"""Shared Operation SDK.

Single source for upload/download/run_workflow/cancel operations.
Both UMRS TUI and R4 client consume this layer.

No TUI imports — this is a pure client/runtime module.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OperationResult:
    """Result of an operation call."""

    ok: bool = False
    operation_id: str = ""
    detail: str = ""
    artifacts: list[str] = field(default_factory=list)


class OperationSDK:
    """Shared operation SDK backing both UMRS TUI and R4 client.

    Wraps existing client transport — reuses upload/download/run_workflow
    implementations from batch_upload.py and ops.py; does not reimplement
    transfer logic.
    """

    def __init__(
        self,
        *,
        gateway_url: str | None = None,
        workflow_submitter: Callable[..., dict[str, Any]] | None = None,
        experience_submitter: Callable[..., dict[str, Any]] | None = None,
        job_canceller: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the SDK.

        Args:
            gateway_url: Base URL for the Gateway endpoint.
                Defaults to UMRS_GATEWAY_URL env var or localhost.
            workflow_submitter: Optional callable translating
                (name, params) → a workflow-submission response. When
                absent, `run_workflow` and `run_experience` raise instead
                of faking success (TODO-era behavior fixed 2026-09-18).
            experience_submitter: Optional callable for the experience
                family (same contract).
            job_canceller: Optional callable responsible for applying a
                durable `cancel_job` on `(job_id)`; absent ⇒ `cancel`
                raises the same loud refusal instead of silently returning
                a stale True.
        """
        self.gateway_url = gateway_url or os.environ.get(
            "UMRS_GATEWAY_URL", "http://localhost:8000"
        )
        self._workflow_submitter = workflow_submitter
        self._experience_submitter = experience_submitter
        self._job_canceller = job_canceller

    # -- public operations -----------------------------------------------------

    def upload(
        self,
        source: str,
        bucket: str,
        *,
        limit: int | None = None,
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> OperationResult:
        """Upload files or directory to S3 bucket.

        Args:
            source: Local file or directory path.
            bucket: Target bucket alias.
            limit: Maximum files to upload (optional).
            progress_cb: Callback(pct: int, msg: str) for progress.

        Returns:
            OperationResult with ok=True on success.
        """
        try:
            from ..client.batch_upload import (
                BatchUploadClient,
            )

            client = BatchUploadClient(
                source=source,
                bucket=bucket,
                limit=limit,
            )
            manifest, uploaded = client.run(progress_cb=progress_cb)

            artifact_ids: list[str] = []
            for entry_name, entry in manifest.files.items():
                if entry.state.value == "uploaded":
                    artifact_ids.append(entry_name)

            return OperationResult(
                ok=uploaded > 0,
                operation_id=manifest.manifest_path.name,
                detail=f"Uploaded {uploaded}/{len(manifest.files)} files",
                artifacts=artifact_ids,
            )

        except Exception as exc:
            logger.error("Upload failed: %s", exc)
            return OperationResult(
                ok=False,
                detail=str(exc),
            )

    def download(
        self,
        bucket_or_execution: str,
        dest: str,
        *,
        overwrite: bool = False,
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> OperationResult:
        """Download files from S3 bucket to local path.

        Args:
            bucket_or_execution: Source bucket alias or execution id.
            dest: Destination local path.
            overwrite: Overwrite existing files.
            progress_cb: Callback(pct: int, msg: str) for progress.

        Returns:
            OperationResult with ok=True on success.
        """
        try:
            from ..client.batch_upload import (
                DownloadClient,
            )

            client = DownloadClient(
                source=bucket_or_execution,
                dest=dest,
                overwrite=overwrite,
            )
            downloaded = client.run(progress_cb=progress_cb)

            return OperationResult(
                ok=True,
                operation_id=dest,
                detail=f"Downloaded {downloaded} files",
            )

        except Exception as exc:
            logger.error("Download failed: %s", exc)
            return OperationResult(
                ok=False,
                detail=str(exc),
            )

    def run_workflow(
        self,
        name: str,
        bucket: str,
        params: dict[str, Any] | None = None,
        *,
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> OperationResult:
        """Run a registered workflow.

        Args:
            name: Workflow name.
            bucket: Input bucket.
            params: Runtime parameters.
            progress_cb: Callback(pct: int, msg: str) for progress.

        Returns:
            OperationResult with ok=True on success.
        """
        if self._workflow_submitter is None:
            raise RuntimeError(
                "run_workflow is not wired: no workflow_submitter was "
                "given to OperationSDK (use ControlPlaneClient.run_workflow)"
            )
        try:
            response = self._workflow_submitter(name=name, params=params)
        except Exception as exc:
            logger.error("Run workflow failed: %s", exc)
            return OperationResult(
                ok=False,
                detail=str(exc),
            )
        return OperationResult(
            ok=True,
            operation_id=str(response.get("operation_id") or f"workflow-{name}"),
            detail=str(response.get("detail") or f"Workflow {name} submitted"),
            artifacts=list(response.get("artifacts") or []),
        )

    def run_experience(
        self,
        name: str,
        bucket: str,
        overrides: dict[str, Any] | None = None,
        *,
        progress_cb: Callable[[int, str], None] | None = None,
    ) -> OperationResult:
        """Run an experience (e.g., HMAE reconstruction).

        Args:
            name: Experience name.
            bucket: Input bucket.
            overrides: Experience-specific overrides.
            progress_cb: Callback(pct: int, msg: str) for progress.

        Returns:
            OperationResult with ok=True on success.
        """
        if self._experience_submitter is None:
            raise RuntimeError(
                "run_experience is not wired: no experience_submitter "
                "was given to OperationSDK"
            )
        try:
            response = self._experience_submitter(name=name, bucket=bucket, **(overrides or {}))
        except Exception as exc:
            logger.error("Run experience failed: %s", exc)
            return OperationResult(
                ok=False,
                detail=str(exc),
            )
        return OperationResult(
            ok=True,
            operation_id=str(response.get("operation_id") or f"experience-{name}"),
            detail=str(response.get("detail") or f"Experience {name} submitted"),
            artifacts=list(response.get("artifacts") or []),
        )

    def cancel(self, operation_id: str) -> bool:
        """Cancel a running operation.

        Args:
            operation_id: The operation id to cancel.

        Returns:
            True if cancelled successfully, False otherwise.
        """
        if self._job_canceller is None:
            raise RuntimeError(
                "cancel is not wired: no job_canceller was given to "
                "OperationSDK (use ControlPlaneClient.cancel_job)"
            )
        try:
            self._job_canceller(job_id=operation_id)
            return True

        except Exception as exc:
            logger.error("Cancel failed: %s", exc)
            return False

    def list_artifacts(self, query: dict[str, Any] | None = None) -> list[dict]:
        """List artifacts via the Gateway artifact API.

        Calls the artifact_registry routes exposed by UMRS.

        Args:
            query: Filter dict (artifact_type, owner_id, etc.).

        Returns:
            List of artifact dicts.
        """
        try:
            import requests

            url = f"{self.gateway_url}/umrs/artifacts"
            params = query or {}
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()

        except Exception as exc:
            logger.warning("Failed to list artifacts: %s", exc)
            return []
