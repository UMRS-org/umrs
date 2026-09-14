"""Workflow fork — clone a completed run with modified parameters (MEM-B8).

Creates a new job from an existing completed workflow run, preserving lineage
back to the original via ``forked_from_job_id`` in the new job's spec.

This module lives in ``umrs.jobs`` because it reads from the ``JobStore``
and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Statuses from which forking is allowed
FORKABLE_STATUSES = {"completed", "failed"}


@dataclass
class WorkflowForkRequest:
    """Request to fork a workflow run."""

    source_job_id: str
    user_id: str
    param_overrides: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_job_id": self.source_job_id,
            "user_id": self.user_id,
            "param_overrides": self.param_overrides,
            "description": self.description,
        }


@dataclass
class WorkflowForkResult:
    """Result of a workflow fork operation."""

    new_job_id: str
    source_job_id: str
    forked_spec: dict[str, Any]
    param_overrides: dict[str, Any]
    lineage_chain: list[str]  # [new_job_id, source_job_id, grandparent, ...]
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_job_id": self.new_job_id,
            "source_job_id": self.source_job_id,
            "forked_spec": self.forked_spec,
            "param_overrides": self.param_overrides,
            "lineage_chain": self.lineage_chain,
            "created_at": self.created_at.isoformat()
            if isinstance(self.created_at, datetime)
            else str(self.created_at),
        }


class WorkflowForkError(Exception):
    """Raised when a workflow fork operation fails."""


class WorkflowForker:
    """Forks completed workflow runs as new jobs with modified parameters.

    The forker reads the source job's spec, applies parameter overrides,
    and creates a new job with ``forked_from_job_id`` set in the spec
    for lineage tracking.
    """

    def __init__(self, job_store: Any) -> None:
        self._store = job_store

    def fork(self, request: WorkflowForkRequest) -> WorkflowForkResult:
        """Fork a workflow run, creating a new job with modified params.

        Args:
            request: Fork request with source job ID and overrides.

        Returns:
            WorkflowForkResult with new job ID and lineage.

        Raises:
            WorkflowForkError: If source job not found, not forkable,
                or creation fails.
        """
        source_job = self._get_source_job(request.source_job_id)

        # Validate forkability
        self._validate_forkable(source_job)

        # Build forked spec
        forked_spec = self._build_forked_spec(
            source_job, request.param_overrides, request.description
        )

        # Build lineage chain
        lineage = self._build_lineage_chain(source_job)

        # Create new job
        new_job_id = self._create_forked_job(
            source_job=source_job,
            forked_spec=forked_spec,
            user_id=request.user_id,
        )

        lineage_chain = [new_job_id] + lineage

        return WorkflowForkResult(
            new_job_id=new_job_id,
            source_job_id=request.source_job_id,
            forked_spec=forked_spec,
            param_overrides=request.param_overrides,
            lineage_chain=lineage_chain,
        )

    def get_fork_lineage(self, job_id: str) -> list[str]:
        """Trace the full fork lineage chain for a job.

        Returns a list of job IDs from the given job back to the
        original un-forked job.
        """
        chain = [job_id]
        current_id = job_id
        seen: set[str] = {job_id}

        while True:
            job = self._get_source_job(current_id)
            parent_id = (job.spec or {}).get("forked_from_job_id")
            if not parent_id or parent_id in seen:
                break
            chain.append(parent_id)
            seen.add(parent_id)
            current_id = parent_id

        return chain

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _get_source_job(self, job_id: str) -> Any:
        """Load a job from the store."""
        job = self._store.get_job(job_id)
        if job is None:
            raise WorkflowForkError(f"Source job not found: {job_id}")
        return job

    def _validate_forkable(self, job: Any) -> None:
        """Check that the job can be forked."""
        status_val = (
            job.status.value if hasattr(job.status, "value") else str(job.status)
        )
        if status_val not in FORKABLE_STATUSES:
            raise WorkflowForkError(
                f"Job {job.job_id} has status '{status_val}'; "
                f"only jobs with status in {FORKABLE_STATUSES} can be forked"
            )

    def _build_forked_spec(
        self,
        source_job: Any,
        param_overrides: dict[str, Any],
        description: str,
    ) -> dict[str, Any]:
        """Create the forked spec by deep-copying and applying overrides."""
        spec = copy.deepcopy(source_job.spec or {})

        # Apply parameter overrides
        for key, value in param_overrides.items():
            spec[key] = value

        # Set fork lineage metadata
        spec["forked_from_job_id"] = source_job.job_id
        if description:
            spec["fork_description"] = description

        # Preserve original workflow identity
        if "original_job_id" not in spec:
            # If source was itself a fork, keep pointing to the root
            original = (source_job.spec or {}).get("original_job_id", source_job.job_id)
            spec["original_job_id"] = original

        return spec

    def _build_lineage_chain(self, source_job: Any) -> list[str]:
        """Build lineage chain from the source job backwards."""
        chain = [source_job.job_id]
        current = source_job
        seen: set[str] = {source_job.job_id}

        while True:
            parent_id = (current.spec or {}).get("forked_from_job_id")
            if not parent_id or parent_id in seen:
                break
            chain.append(parent_id)
            seen.add(parent_id)
            current = self._store.get_job(parent_id)
            if current is None:
                break

        return chain

    def _create_forked_job(
        self,
        source_job: Any,
        forked_spec: dict[str, Any],
        user_id: str,
    ) -> str:
        """Create the forked job in the store."""
        job_type = source_job.job_type
        project_id = source_job.project_id

        new_job_id = self._store.create_job(
            project_id=project_id,
            user_id=user_id,
            job_type=job_type,
            spec=forked_spec,
        )
        return new_job_id
