"""Artifact auto-registration hooks.

Automatically registers artifacts when workflow executions complete,
linking output files to the execution context and input artifacts.

Config flag: ``artifact_registry.auto_register`` (default ``true``),
read via existing site config loader.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from ..artifact_registry.models import Artifact
from ..artifact_registry.store import ArtifactStore, ArtifactStoreError
from ..jobs.models import AttemptStatus

logger = logging.getLogger(__name__)


def _config_auto_register(site_cfg: dict) -> bool:
    """Read ``artifact_registry.auto_register`` from site config.

    Returns True when artifact auto-registration is enabled.
    Falls back to True (on by default) when the flag is absent.
    """
    try:
        return bool(
            site_cfg.get("artifact_registry", {}).get(
                "auto_register", True
            )
        )
    except (TypeError, AttributeError):
        return True


def register_artifacts_on_success(
    job: Any,
    attempt: Any,
    conn_factory: Any,
    site_cfg: dict | None = None,
) -> int:
    """Register output artifacts for a successful job execution.

    Called as the post-success callback on JobStore.

    Args:
        job: The Job object.
        attempt: The JobAttempt object.
        conn_factory: DB connection factory for the artifact store.
        site_cfg: Optional site config dict.

    Returns:
        Number of artifacts registered.
    """
    if site_cfg is None or not _config_auto_register(site_cfg):
        return 0

    # Only register on final success (commit=True)
    if attempt.status != AttemptStatus.SUCCEEDED:
        return 0

    try:
        store = ArtifactStore(conn_factory)
        count = 0

        # Register job outputs as artifacts
        spec = job.spec or {}
        output_artifacts: list[str] = spec.get("output_artifacts", [])

        for artifact_uri in output_artifacts:
            artifact = Artifact(
                artifact_type=_guess_type(artifact_uri),
                name=_base_name(artifact_uri),
                owner_id=job.user_id,
                project_id=job.project_id,
                uri=artifact_uri,
                execution_id=attempt.attempt_id,
                workflow_id=job.spec.get("workflow_name"),
                job_id=job.job_id,
                checksum=_checksum_from_uri(artifact_uri),
                metadata={
                    "registered_by": "auto_registration",
                    "registered_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            created = store.create(artifact)
            count += 1
            logger.debug(
                "Registered artifact %s for job %s attempt %s",
                created.id,
                job.job_id,
                attempt.attempt_id,
            )

        # Register lineage edges from input artifacts
        input_artifacts: list[str] = spec.get("input_artifacts", [])
        for input_uri in input_artifacts:
            # Find the artifact by matching URI
            # TODO: This is a simple lookup — could be optimized with
            # execution_id-based queries in the ArtifactStore
            try:
                # Find parent artifact by URI prefix match
                parent_artifacts = store.list(
                    type("ArtifactQuery", (), {
                        "artifact_type": None,
                        "owner_id": job.user_id,
                        "project_id": job.project_id,
                        "execution_id": None,
                        "name_prefix": None,
                        "limit": 1000,
                        "offset": 0,
                    })()
                )
                for pa in parent_artifacts:
                    if pa.uri == input_uri:
                        store.add_lineage(created.id, pa.id)
                        break
            except Exception as exc:
                logger.warning(
                    "Failed to add lineage for artifact %s: %s",
                    created.id,
                    exc,
                )

        return count

    except ArtifactStoreError as exc:
        logger.warning(
            "Artifact registration failed for job %s: %s",
            job.job_id,
            exc,
        )
        return 0


def _guess_type(uri: str) -> str:
    """Infer artifact type from URI extension."""
    ext = uri.rsplit(".", 1)[-1].lower() if "." in uri else ""
    type_map = {
        "txt": "dataset",
        "csv": "dataset",
        "json": "config",
        "yaml": "config",
        "yml": "config",
        "pt": "checkpoint",
        "pth": "checkpoint",
        "sif": "model",
        "bin": "model",
        "pkl": "result",
        "npy": "result",
        "parquet": "dataset",
        "pdf": "dataset",
        "png": "image",
        "jpg": "image",
        "jpeg": "image",
    }
    return type_map.get(ext, "result")


def _base_name(uri: str) -> str:
    """Extract base filename from URI."""
    return uri.rsplit("/", 1)[-1] if "/" in uri else uri


def _checksum_from_uri(uri: str) -> str | None:
    """Compute sha256 hash of the URI itself (as a simple content identifier)."""
    return f"sha256:{hashlib.sha256(uri.encode()).hexdigest()}"
