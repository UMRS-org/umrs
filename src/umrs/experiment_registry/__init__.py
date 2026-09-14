"""Experiment registry — experiment definitions, parameters, status, results, artifacts.

Phase 1, Step 2.4 of the RC1 migration plan.

UMRS owns the experiment registry. R4 consumes this API.

This module provides an experiment registry for tracking experiment
lifecycle:

  - Create experiments with hypothesis, protocol, parameters, validation method
  - Track experiment status (pending, designed, submitted, running, completed, failed, cancelled)
  - Add experiment results (metrics, artifacts, logs)
  - List experiments with filters (by project, hypothesis, status)
  - Get experiment history

The registry is append-only for results — new results create new entries.
Experiments can be updated (e.g., status changes), creating new versions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

__all__ = [
    "DuplicateExperimentError",
    "Experiment",
    "ExperimentNotFoundError",
    "ExperimentResult",
    "ExperimentStatus",
    "InMemoryExperimentRegistry",
    "InvalidStatusTransitionError",
    "build_default_store",
]


# ── Experiment Status ─────────────────────────────────────────────────────


class ExperimentStatus(str, Enum):
    """Status of an experiment."""

    PENDING = "pending"  # Created but not yet designed
    DESIGNED = "designed"  # Protocol designed, not yet submitted
    SUBMITTED = "submitted"  # Submitted to execution backend
    RUNNING = "running"  # Currently running
    COMPLETED = "completed"  # Completed successfully
    FAILED = "failed"  # Failed
    CANCELLED = "cancelled"  # Cancelled by user


# ── Experiment ───────────────────────────────────────────────────────────


@dataclass
class Experiment:
    """An experiment definition with lifecycle tracking.

    Attributes:
        experiment_id: Unique identifier for the experiment
        project_id: The project this experiment belongs to
        hypothesis_id: The hypothesis this experiment tests
        protocol: The experimental protocol (JSON-serializable)
        parameters: Experiment parameters (JSON-serializable)
        validation_method: How results are validated
        status: Current status of the experiment
        results: List of experiment results
        created_at: When the experiment was created
        updated_at: When the experiment was last updated
        created_by: Who created the experiment
        submitted_at: When the experiment was submitted
        completed_at: When the experiment completed
        notes: Additional notes
    """

    experiment_id: str = ""
    project_id: str = ""
    hypothesis_id: str = ""
    protocol: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    validation_method: str = "statistical_test"
    status: ExperimentStatus = ExperimentStatus.PENDING
    results: list[ExperimentResult] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    created_by: str = ""
    submitted_at: str | None = None
    completed_at: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id:
            self.experiment_id = f"exp-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "experiment_id": self.experiment_id,
            "project_id": self.project_id,
            "hypothesis_id": self.hypothesis_id,
            "protocol": self.protocol,
            "parameters": self.parameters,
            "validation_method": self.validation_method,
            "status": self.status.value,
            "results": [r.to_dict() for r in self.results],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "submitted_at": self.submitted_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        """Create from dictionary."""
        status_raw = data.get("status", "pending")
        status = (
            ExperimentStatus(status_raw) if isinstance(status_raw, str) else status_raw
        )
        results_data = data.get("results", [])
        results = [ExperimentResult.from_dict(r) for r in results_data]
        return cls(
            experiment_id=data.get("experiment_id", ""),
            project_id=data.get("project_id", ""),
            hypothesis_id=data.get("hypothesis_id", ""),
            protocol=data.get("protocol", {}),
            parameters=data.get("parameters", {}),
            validation_method=data.get("validation_method", "statistical_test"),
            status=status,
            results=results,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            created_by=data.get("created_by", ""),
            submitted_at=data.get("submitted_at"),
            completed_at=data.get("completed_at"),
            notes=data.get("notes", ""),
        )


# ── Experiment Result ────────────────────────────────────────────────────


@dataclass
class ExperimentResult:
    """A result from an experiment run.

    Attributes:
        result_id: Unique identifier for the result
        experiment_id: The experiment this result belongs to
        metrics: Performance metrics (JSON-serializable)
        artifacts: List of artifact URIs
        logs: Execution logs
        status: Status of this result (completed, failed, partial)
        created_at: When the result was created
        created_by: Who created the result
    """

    result_id: str = ""
    experiment_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    logs: str = ""
    status: str = "completed"
    created_at: str = ""
    created_by: str = ""

    def __post_init__(self) -> None:
        if not self.result_id:
            self.result_id = f"res-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "result_id": self.result_id,
            "experiment_id": self.experiment_id,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "logs": self.logs,
            "status": self.status,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        """Create from dictionary."""
        return cls(
            result_id=data.get("result_id", ""),
            experiment_id=data.get("experiment_id", ""),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", []),
            logs=data.get("logs", ""),
            status=data.get("status", "completed"),
            created_at=data.get("created_at", ""),
            created_by=data.get("created_by", ""),
        )


# ── Exceptions ────────────────────────────────────────────────────────────


class ExperimentNotFoundError(Exception):
    """Raised when an experiment is not found."""


class DuplicateExperimentError(Exception):
    """Raised when a duplicate experiment is added."""


class InvalidStatusTransitionError(Exception):
    """Raised when an invalid status transition is attempted."""


# ── In-Memory Experiment Registry ─────────────────────────────────────────


class InMemoryExperimentRegistry:
    """In-memory store for experiments.

    This is the reference implementation. The Postgres-backed implementation
    will follow the same interface.
    """

    # Valid status transitions
    _VALID_TRANSITIONS = {
        ExperimentStatus.PENDING: {
            ExperimentStatus.DESIGNED,
            ExperimentStatus.CANCELLED,
        },
        ExperimentStatus.DESIGNED: {
            ExperimentStatus.SUBMITTED,
            ExperimentStatus.CANCELLED,
        },
        ExperimentStatus.SUBMITTED: {
            ExperimentStatus.RUNNING,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        },
        ExperimentStatus.RUNNING: {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        },
        ExperimentStatus.COMPLETED: set(),  # Terminal
        ExperimentStatus.FAILED: set(),  # Terminal
        ExperimentStatus.CANCELLED: set(),  # Terminal
    }

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}

    def create_experiment(
        self,
        *,
        project_id: str,
        hypothesis_id: str,
        protocol: dict[str, Any] | None = None,
        parameters: dict[str, Any] | None = None,
        validation_method: str = "statistical_test",
        created_by: str = "",
        notes: str = "",
    ) -> str:
        """Create a new experiment.

        Args:
            project_id: The project this experiment belongs to
            hypothesis_id: The hypothesis this experiment tests
            protocol: The experimental protocol
            parameters: Experiment parameters
            validation_method: How results are validated
            created_by: Who created the experiment
            notes: Additional notes

        Returns:
            The experiment_id of the created experiment
        """
        exp = Experiment(
            project_id=project_id,
            hypothesis_id=hypothesis_id,
            protocol=protocol or {},
            parameters=parameters or {},
            validation_method=validation_method,
            created_by=created_by,
            notes=notes,
        )
        self._experiments[exp.experiment_id] = exp
        return exp.experiment_id

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment by ID.

        Args:
            experiment_id: The experiment ID

        Returns:
            The experiment, or None if not found
        """
        return self._experiments.get(experiment_id)

    def update_experiment_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
    ) -> bool:
        """Update the status of an experiment.

        Args:
            experiment_id: The experiment ID
            status: The new status

        Returns:
            True if the experiment was found and updated, False otherwise

        Raises:
            ExperimentNotFoundError: If the experiment is not found
            InvalidStatusTransitionError: If the transition is invalid
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False

        # Validate transition
        current_status = exp.status
        valid_next = self._VALID_TRANSITIONS.get(current_status, set())
        if status not in valid_next and status != current_status:
            raise InvalidStatusTransitionError(
                f"Cannot transition from {current_status.value} to {status.value}"
            )

        exp.status = status
        exp.updated_at = datetime.now(timezone.utc).isoformat()

        # Set timestamps based on status
        if status == ExperimentStatus.SUBMITTED:
            exp.submitted_at = exp.updated_at
        elif status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            exp.completed_at = exp.updated_at

        return True

    def add_experiment_result(
        self,
        experiment_id: str,
        *,
        metrics: dict[str, Any] | None = None,
        artifacts: list[str] | None = None,
        logs: str = "",
        status: str = "completed",
        created_by: str = "",
    ) -> str:
        """Add a result to an experiment.

        Args:
            experiment_id: The experiment ID
            metrics: Performance metrics
            artifacts: List of artifact URIs
            logs: Execution logs
            status: Status of this result
            created_by: Who created the result

        Returns:
            The result_id of the added result

        Raises:
            ExperimentNotFoundError: If the experiment is not found
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(f"Experiment {experiment_id} not found")

        result = ExperimentResult(
            experiment_id=experiment_id,
            metrics=metrics or {},
            artifacts=artifacts or [],
            logs=logs,
            status=status,
            created_by=created_by,
        )
        exp.results.append(result)
        exp.updated_at = datetime.now(timezone.utc).isoformat()
        return result.result_id

    def list_experiments(
        self,
        *,
        project_id: str | None = None,
        hypothesis_id: str | None = None,
        status: ExperimentStatus | None = None,
        limit: int = 100,
    ) -> list[Experiment]:
        """List experiments with optional filters.

        Args:
            project_id: Filter by project ID
            hypothesis_id: Filter by hypothesis ID
            status: Filter by status
            limit: Maximum number of experiments to return

        Returns:
            List of matching experiments, sorted by created_at descending
        """
        results: list[Experiment] = []
        for exp in self._experiments.values():
            if project_id and exp.project_id != project_id:
                continue
            if hypothesis_id and exp.hypothesis_id != hypothesis_id:
                continue
            if status and exp.status != status:
                continue
            results.append(exp)
        results.sort(key=lambda e: e.created_at, reverse=True)
        return results[:limit]

    def get_experiment_history(self, project_id: str) -> list[Experiment]:
        """Get the experiment history for a project, sorted by created_at.

        Args:
            project_id: The project ID

        Returns:
            List of experiments sorted by created_at ascending (oldest first)
        """
        results = [
            exp for exp in self._experiments.values() if exp.project_id == project_id
        ]
        results.sort(key=lambda e: e.created_at)
        return results


# ── Default Store Factory ───────────────────────────────────────────────


def build_default_store(db_pool: Any = None) -> Any:
    """Build the default experiment registry backend.

    Selection via ``UMRS_EXPERIMENT_REGISTRY_STORE_BACKEND`` (default ``auto``):

      - ``pg``     — force Postgres (raises if unavailable)
      - ``memory`` — force the in-memory reference registry
      - ``auto``   — Postgres when a pool is provided or the environment
                     points at a database, otherwise in-memory

    Postgres uses the ``experiments`` / ``experiment_results`` tables
    (migration 0059).
    """
    import os as _os

    from umrs.shared.pg import resolve_satellite_pool

    backend = _os.environ.get("UMRS_EXPERIMENT_REGISTRY_STORE_BACKEND", "auto").lower()
    if backend == "memory":
        return InMemoryExperimentRegistry()
    if backend not in ("auto", "pg"):
        logger.warning("Unknown store backend %r; using auto", backend)
        backend = "auto"

    pool = resolve_satellite_pool(backend, db_pool, "Experiment registry")
    if pool is None:
        return InMemoryExperimentRegistry()

    from .pg_store import PgExperimentRegistry

    return PgExperimentRegistry(pool)
