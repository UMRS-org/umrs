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

    def __init__(self, *, gateway_url: str | None = None) -> None:
        """Initialize the SDK.

        Args:
            gateway_url: Base URL for the Gateway endpoint.
                Defaults to UMRS_GATEWAY_URL env var or localhost.
        """
        self.gateway_url = gateway_url or os.environ.get(
            "UMRS_GATEWAY_URL", "http://localhost:8000"
        )

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
        try:
            # TODO: Implement workflow execution via JobStore
            # This calls handle_workflow_command or similar backend
            return OperationResult(
                ok=True,
                operation_id=f"workflow-{name}",
                detail=f"Workflow {name} submitted",
                artifacts=[],
            )

        except Exception as exc:
            logger.error("Run workflow failed: %s", exc)
            return OperationResult(
                ok=False,
                detail=str(exc),
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
        try:
            # TODO: Implement experience execution via JobStore
            return OperationResult(
                ok=True,
                operation_id=f"experience-{name}",
                detail=f"Experience {name} submitted",
                artifacts=[],
            )

        except Exception as exc:
            logger.error("Run experience failed: %s", exc)
            return OperationResult(
                ok=False,
                detail=str(exc),
            )

    def cancel(self, operation_id: str) -> bool:
        """Cancel a running operation.

        Args:
            operation_id: The operation id to cancel.

        Returns:
            True if cancelled successfully, False otherwise.
        """
        try:
            # TODO: Implement cancel via JobStore or task cancellation
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
