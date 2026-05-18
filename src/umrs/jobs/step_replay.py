"""Step replay execution engine (MEM-B5).

Retrieves immutable input snapshots from the execution trace store and
creates new trace records with identical inputs for deterministic replay.

This module lives in ``umrs.jobs`` because it reads from the trace store
and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@dataclass
class ReplayRequest:
    """Request to replay a step from a previous trace."""

    source_trace_id: str
    user_id: str
    reason: str = ""
    param_overrides: dict[str, Any] | None = None
    new_job_id: str | None = None


@dataclass
class ReplayResult:
    """Result of a replay operation."""

    replay_id: str
    source_trace_id: str
    new_trace_id: str
    new_job_id: str
    step_name: str
    inputs_hash: str | None = None
    inputs_snapshot: dict[str, Any] | None = None
    inputs_ref: str | None = None
    param_overrides: dict[str, Any] | None = None
    user_id: str = ""
    reason: str = ""
    status: str = "created"
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_id": self.replay_id,
            "source_trace_id": self.source_trace_id,
            "new_trace_id": self.new_trace_id,
            "new_job_id": self.new_job_id,
            "step_name": self.step_name,
            "inputs_hash": self.inputs_hash,
            "inputs_snapshot": self.inputs_snapshot,
            "inputs_ref": self.inputs_ref,
            "param_overrides": self.param_overrides,
            "user_id": self.user_id,
            "reason": self.reason,
            "status": self.status,
            "created_at": _iso(self.created_at),
        }


class StepReplayEngine:
    """Replays workflow steps using stored input snapshots.

    Reads from ExecutionTraceStore to retrieve the original inputs,
    then creates a new trace record with identical (or overridden) inputs
    for deterministic re-execution.
    """

    def __init__(self, trace_store: Any) -> None:
        self._traces = trace_store
        self._replays: dict[str, ReplayResult] = {}
        self._counter = 0

    def replay(self, request: ReplayRequest) -> ReplayResult:
        """Create a replay from a source trace.

        Validates the source trace exists and has input data,
        then creates a new trace with the same inputs.

        Args:
            request: Replay request with source trace ID and options.

        Returns:
            ReplayResult with new trace ID and input snapshot.

        Raises:
            ValueError: If source trace not found or has no input data.
        """
        source = self._traces.get_trace(request.source_trace_id)
        if source is None:
            raise ValueError(
                f"Source trace not found: {request.source_trace_id}"
            )

        # Ensure source has input data for replay
        has_inputs = (
            source.inputs_snapshot is not None
            or source.inputs_ref is not None
            or source.inputs_hash is not None
        )
        if not has_inputs:
            raise ValueError(
                f"Source trace {request.source_trace_id} has no input data for replay"
            )

        # Build replay inputs
        inputs_snapshot = source.inputs_snapshot
        if request.param_overrides and inputs_snapshot:
            # Apply overrides to a copy
            inputs_snapshot = {**inputs_snapshot, **request.param_overrides}

        self._counter += 1
        new_job_id = request.new_job_id or f"replay-{uuid.uuid4().hex[:12]}"
        new_trace_id = f"trc-replay-{uuid.uuid4().hex[:8]}"

        # Create new trace in the store
        new_trace = self._traces.create_trace(
            job_id=new_job_id,
            step_name=source.step_name,
            step_type=source.step_type,
            parent_trace_id=request.source_trace_id,
            inputs=inputs_snapshot,
            model_ref=source.model_ref,
        )

        result = ReplayResult(
            replay_id=f"rpl-{self._counter:04d}",
            source_trace_id=request.source_trace_id,
            new_trace_id=new_trace.trace_id,
            new_job_id=new_job_id,
            step_name=source.step_name,
            inputs_hash=source.inputs_hash if not request.param_overrides else None,
            inputs_snapshot=inputs_snapshot,
            inputs_ref=source.inputs_ref if not request.param_overrides else None,
            param_overrides=request.param_overrides,
            user_id=request.user_id,
            reason=request.reason,
        )

        self._replays[result.replay_id] = result
        logger.info(
            "Replay created: %s (source=%s → new=%s step=%s)",
            result.replay_id, request.source_trace_id,
            new_trace.trace_id, source.step_name,
        )
        return result

    def get_replay(self, replay_id: str) -> ReplayResult | None:
        return self._replays.get(replay_id)

    def list_replays(
        self,
        source_trace_id: str | None = None,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[ReplayResult]:
        """List replay results with optional filters."""
        results = list(self._replays.values())
        if source_trace_id:
            results = [r for r in results if r.source_trace_id == source_trace_id]
        if user_id:
            results = [r for r in results if r.user_id == user_id]
        results.sort(key=lambda r: r.created_at, reverse=True)
        return results[:limit]

    def find_by_inputs_hash(self, inputs_hash: str) -> list[ReplayResult]:
        """Find replays that share the same input hash (dedup)."""
        return [
            r for r in self._replays.values()
            if r.inputs_hash == inputs_hash
        ]
