"""Execution trace store for operational memory (MEM-A1 / MEM-B1 / MEM-B2).

Provides durable persistence for step-level execution traces, linked to
the existing Job/Attempt model.  Replaces the ephemeral in-memory
``ExecutionRecord`` with Postgres-backed storage that survives UMRS
restarts.

This module lives in ``umrs.jobs`` because it reads/writes durable state
tied to jobs and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _generate_trace_id() -> str:
    return f"trc-{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _compute_hash(data: Any) -> str | None:
    """Compute SHA-256 hash of JSON-serializable data."""
    if data is None:
        return None
    try:
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
    except (TypeError, ValueError):
        return None


@dataclass
class TraceRecord:
    """In-memory representation of an execution trace row."""

    trace_id: str
    job_id: str
    step_name: str
    step_type: str = "generic"
    attempt_id: str | None = None
    execution_id: str | None = None
    parent_trace_id: str | None = None
    status: str = "pending"
    progress: float = 0.0
    inputs_hash: str | None = None
    outputs_hash: str | None = None
    inputs_ref: str | None = None
    outputs_ref: str | None = None
    model_ref: dict[str, Any] | None = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieval_refs: list[dict[str, Any]] = field(default_factory=list)
    routing_decision: dict[str, Any] | None = None
    error: str | None = None
    error_detail: dict[str, Any] | None = None
    inputs_snapshot: dict[str, Any] | None = None
    outputs_snapshot: dict[str, Any] | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = field(default_factory=_now)
    ttl_expires_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "job_id": self.job_id,
            "step_name": self.step_name,
            "step_type": self.step_type,
            "attempt_id": self.attempt_id,
            "execution_id": self.execution_id,
            "parent_trace_id": self.parent_trace_id,
            "status": self.status,
            "progress": self.progress,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "inputs_ref": self.inputs_ref,
            "outputs_ref": self.outputs_ref,
            "model_ref": self.model_ref,
            "artifact_refs": self.artifact_refs,
            "tool_calls": self.tool_calls,
            "retrieval_refs": self.retrieval_refs,
            "routing_decision": self.routing_decision,
            "error": self.error,
            "error_detail": self.error_detail,
            "inputs_snapshot": self.inputs_snapshot,
            "outputs_snapshot": self.outputs_snapshot,
            "started_at": _iso(self.started_at),
            "finished_at": _iso(self.finished_at),
            "created_at": _iso(self.created_at),
            "ttl_expires_at": _iso(self.ttl_expires_at),
        }


class ExecutionTraceStore:
    """Durable execution trace store backed by Postgres.

    Falls back to in-memory storage when ``db_url`` is not set (dev mode).
    Thread-safe for concurrent writes.
    """

    def __init__(self, db_url: str | None = None) -> None:
        self._db_url = db_url
        self._psycopg2 = None
        if db_url:
            try:
                import psycopg2  # type: ignore

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning(
                    "db_url set but psycopg2 unavailable; "
                    "execution traces will use in-memory store"
                )

        # In-memory fallback
        self._traces: dict[str, TraceRecord] = {}

    @property
    def db_enabled(self) -> bool:
        return self._psycopg2 is not None and self._db_url is not None

    def _get_conn(self):
        """Get a database connection."""
        from umrs.shared.pg_settings import pg_connect_kwargs

        return self._psycopg2.connect(
            self._db_url, **pg_connect_kwargs("execution-trace")
        )

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def create_trace(
        self,
        job_id: str,
        step_name: str,
        step_type: str = "generic",
        attempt_id: str | None = None,
        execution_id: str | None = None,
        parent_trace_id: str | None = None,
        inputs: Any | None = None,
        model_ref: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
    ) -> TraceRecord:
        """Create a new execution trace record."""
        trace_id = _generate_trace_id()
        now = _now()
        ttl_expires = None
        if ttl_seconds is not None:
            from datetime import timedelta

            ttl_expires = now + timedelta(seconds=ttl_seconds)

        # Store inline snapshot for replay if inputs are small enough
        inputs_snapshot = None
        if inputs is not None:
            try:
                raw = json.dumps(inputs, sort_keys=True, default=str)
                # Only inline if < 64 KB; larger payloads should use inputs_ref (S3)
                if len(raw) < 65536:
                    inputs_snapshot = inputs
            except (TypeError, ValueError):
                pass

        record = TraceRecord(
            trace_id=trace_id,
            job_id=job_id,
            step_name=step_name,
            step_type=step_type,
            attempt_id=attempt_id,
            execution_id=execution_id,
            parent_trace_id=parent_trace_id,
            inputs_hash=_compute_hash(inputs),
            inputs_snapshot=inputs_snapshot,
            model_ref=model_ref,
            created_at=now,
            ttl_expires_at=ttl_expires,
        )

        if self.db_enabled:
            try:
                self._db_insert_trace(record)
                return record
            except Exception as exc:
                logger.warning("DB insert trace failed, using memory: %s", exc)

        self._traces[trace_id] = record
        return record

    def start_trace(self, trace_id: str) -> TraceRecord | None:
        """Mark a trace as running."""
        return self._update_trace(trace_id, status="running", started_at=_now())

    def complete_trace(
        self,
        trace_id: str,
        outputs: Any | None = None,
        outputs_ref: str | None = None,
        artifact_refs: list[dict[str, Any]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        retrieval_refs: list[dict[str, Any]] | None = None,
        routing_decision: dict[str, Any] | None = None,
    ) -> TraceRecord | None:
        """Mark a trace as succeeded with outputs."""
        updates: dict[str, Any] = {
            "status": "succeeded",
            "finished_at": _now(),
            "outputs_hash": _compute_hash(outputs),
        }
        # Store inline output snapshot for replay if small enough
        if outputs is not None:
            try:
                raw = json.dumps(outputs, sort_keys=True, default=str)
                if len(raw) < 65536:
                    updates["outputs_snapshot"] = outputs
            except (TypeError, ValueError):
                pass
        if outputs_ref is not None:
            updates["outputs_ref"] = outputs_ref
        if artifact_refs is not None:
            updates["artifact_refs"] = artifact_refs
        if tool_calls is not None:
            updates["tool_calls"] = tool_calls
        if retrieval_refs is not None:
            updates["retrieval_refs"] = retrieval_refs
        if routing_decision is not None:
            updates["routing_decision"] = routing_decision
        return self._update_trace(trace_id, **updates)

    def fail_trace(
        self,
        trace_id: str,
        error: str,
        error_detail: dict[str, Any] | None = None,
    ) -> TraceRecord | None:
        """Mark a trace as failed."""
        return self._update_trace(
            trace_id,
            status="failed",
            finished_at=_now(),
            error=error,
            error_detail=error_detail,
        )

    def skip_trace(self, trace_id: str) -> TraceRecord | None:
        """Mark a trace as skipped."""
        return self._update_trace(trace_id, status="skipped", finished_at=_now())

    def update_progress(self, trace_id: str, progress: float) -> TraceRecord | None:
        """Update trace progress (0.0 to 1.0)."""
        return self._update_trace(trace_id, progress=min(1.0, max(0.0, progress)))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Get a single trace by ID."""
        if self.db_enabled:
            try:
                return self._db_get_trace(trace_id)
            except Exception as exc:
                logger.warning("DB get_trace failed: %s", exc)
        return self._traces.get(trace_id)

    def get_traces_for_job(self, job_id: str) -> list[TraceRecord]:
        """Get all traces for a job, ordered by created_at."""
        if self.db_enabled:
            try:
                return self._db_get_traces_for_job(job_id)
            except Exception as exc:
                logger.warning("DB get_traces_for_job failed: %s", exc)
        return sorted(
            [t for t in self._traces.values() if t.job_id == job_id],
            key=lambda t: t.created_at,
        )

    def get_traces_for_execution(self, execution_id: str) -> list[TraceRecord]:
        """Get all traces for a workflow execution ID."""
        if self.db_enabled:
            try:
                return self._db_get_traces_for_execution(execution_id)
            except Exception as exc:
                logger.warning("DB get_traces_for_execution failed: %s", exc)
        return sorted(
            [t for t in self._traces.values() if t.execution_id == execution_id],
            key=lambda t: t.created_at,
        )

    def get_lineage(self, trace_id: str) -> list[TraceRecord]:
        """Get the full ancestor chain for a trace (root first)."""
        chain: list[TraceRecord] = []
        current_id: str | None = trace_id
        visited: set[str] = set()
        while current_id and current_id not in visited:
            visited.add(current_id)
            record = self.get_trace(current_id)
            if record is None:
                break
            chain.append(record)
            current_id = record.parent_trace_id
        chain.reverse()
        return chain

    def get_children(self, trace_id: str) -> list[TraceRecord]:
        """Get direct child traces."""
        if self.db_enabled:
            try:
                return self._db_get_children(trace_id)
            except Exception as exc:
                logger.warning("DB get_children failed: %s", exc)
        return [t for t in self._traces.values() if t.parent_trace_id == trace_id]

    def cleanup_expired(self) -> int:
        """Remove traces past their TTL. Returns count removed."""
        now = _now()
        if self.db_enabled:
            try:
                return self._db_cleanup_expired(now)
            except Exception as exc:
                logger.warning("DB cleanup_expired failed: %s", exc)

        expired = [
            tid
            for tid, t in self._traces.items()
            if t.ttl_expires_at is not None and t.ttl_expires_at < now
        ]
        for tid in expired:
            del self._traces[tid]
        return len(expired)

    # ------------------------------------------------------------------
    # Replay support (MEM-B5)
    # ------------------------------------------------------------------

    def get_input_snapshot(self, trace_id: str) -> dict[str, Any] | None:
        """Get the stored input snapshot for a trace (if available)."""
        record = self.get_trace(trace_id)
        if record is None:
            return None
        return record.inputs_snapshot

    def get_output_snapshot(self, trace_id: str) -> dict[str, Any] | None:
        """Get the stored output snapshot for a trace (if available)."""
        record = self.get_trace(trace_id)
        if record is None:
            return None
        return record.outputs_snapshot

    def find_traces_by_step(
        self, step_name: str, status: str | None = None
    ) -> list[TraceRecord]:
        """Find traces for a given step name, optionally filtered by status."""
        if self.db_enabled:
            try:
                return self._db_find_traces_by_step(step_name, status)
            except Exception as exc:
                logger.warning("DB find_traces_by_step failed: %s", exc)

        results = [t for t in self._traces.values() if t.step_name == step_name]
        if status:
            results = [t for t in results if t.status == status]
        return sorted(results, key=lambda t: t.created_at)

    def find_traces_by_inputs_hash(self, inputs_hash: str) -> list[TraceRecord]:
        """Find all traces with identical input hashes (for dedup/comparison)."""
        if self.db_enabled:
            try:
                return self._db_find_by_inputs_hash(inputs_hash)
            except Exception as exc:
                logger.warning("DB find_traces_by_inputs_hash failed: %s", exc)

        return [t for t in self._traces.values() if t.inputs_hash == inputs_hash]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_trace(self, trace_id: str, **fields: Any) -> TraceRecord | None:
        if self.db_enabled:
            try:
                return self._db_update_trace(trace_id, **fields)
            except Exception as exc:
                logger.warning("DB update_trace failed: %s", exc)

        record = self._traces.get(trace_id)
        if record is None:
            return None
        for key, value in fields.items():
            setattr(record, key, value)
        return record

    # ------------------------------------------------------------------
    # Postgres backend
    # ------------------------------------------------------------------

    def _db_insert_trace(self, record: TraceRecord) -> None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO execution_traces (
                        trace_id, job_id, attempt_id, execution_id,
                        parent_trace_id, step_name, step_type,
                        status, progress, inputs_hash, model_ref,
                        created_at, ttl_expires_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    """,
                    (
                        record.trace_id,
                        record.job_id,
                        record.attempt_id,
                        record.execution_id,
                        record.parent_trace_id,
                        record.step_name,
                        record.step_type,
                        record.status,
                        record.progress,
                        record.inputs_hash,
                        json.dumps(record.model_ref) if record.model_ref else None,
                        record.created_at,
                        record.ttl_expires_at,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def _db_update_trace(self, trace_id: str, **fields: Any) -> TraceRecord | None:
        if not fields:
            return self._db_get_trace(trace_id)

        # Build SET clause dynamically
        set_parts: list[str] = []
        values: list[Any] = []
        json_fields = {
            "model_ref",
            "artifact_refs",
            "tool_calls",
            "retrieval_refs",
            "routing_decision",
            "error_detail",
        }
        for key, value in fields.items():
            set_parts.append(f"{key} = %s")
            if key in json_fields and value is not None:
                values.append(json.dumps(value))
            else:
                values.append(value)
        values.append(trace_id)

        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE execution_traces SET {', '.join(set_parts)} WHERE trace_id = %s",
                    values,
                )
            conn.commit()
        finally:
            conn.close()

        return self._db_get_trace(trace_id)

    def _db_get_trace(self, trace_id: str) -> TraceRecord | None:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM execution_traces WHERE trace_id = %s",
                    (trace_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                cols = [desc[0] for desc in cur.description]
                return self._row_to_record(dict(zip(cols, row)))
        finally:
            conn.close()

    def _db_get_traces_for_job(self, job_id: str) -> list[TraceRecord]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM execution_traces WHERE job_id = %s ORDER BY created_at",
                    (job_id,),
                )
                cols = [desc[0] for desc in cur.description]
                return [
                    self._row_to_record(dict(zip(cols, row))) for row in cur.fetchall()
                ]
        finally:
            conn.close()

    def _db_get_traces_for_execution(self, execution_id: str) -> list[TraceRecord]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM execution_traces WHERE execution_id = %s ORDER BY created_at",
                    (execution_id,),
                )
                cols = [desc[0] for desc in cur.description]
                return [
                    self._row_to_record(dict(zip(cols, row))) for row in cur.fetchall()
                ]
        finally:
            conn.close()

    def _db_get_children(self, trace_id: str) -> list[TraceRecord]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM execution_traces WHERE parent_trace_id = %s ORDER BY created_at",
                    (trace_id,),
                )
                cols = [desc[0] for desc in cur.description]
                return [
                    self._row_to_record(dict(zip(cols, row))) for row in cur.fetchall()
                ]
        finally:
            conn.close()

    def _db_cleanup_expired(self, now: datetime) -> int:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM execution_traces WHERE ttl_expires_at IS NOT NULL AND ttl_expires_at < %s",
                    (now,),
                )
                count = cur.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def _db_find_traces_by_step(
        self, step_name: str, status: str | None
    ) -> list[TraceRecord]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                if status:
                    cur.execute(
                        "SELECT * FROM execution_traces "
                        "WHERE step_name = %s AND status = %s "
                        "ORDER BY created_at",
                        (step_name, status),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM execution_traces "
                        "WHERE step_name = %s ORDER BY created_at",
                        (step_name,),
                    )
                cols = [desc[0] for desc in cur.description]
                return [
                    self._row_to_record(dict(zip(cols, row))) for row in cur.fetchall()
                ]
        finally:
            conn.close()

    def _db_find_by_inputs_hash(self, inputs_hash: str) -> list[TraceRecord]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM execution_traces "
                    "WHERE inputs_hash = %s ORDER BY created_at",
                    (inputs_hash,),
                )
                cols = [desc[0] for desc in cur.description]
                return [
                    self._row_to_record(dict(zip(cols, row))) for row in cur.fetchall()
                ]
        finally:
            conn.close()

    @staticmethod
    def _row_to_record(row: dict[str, Any]) -> TraceRecord:
        """Convert a database row dict to a TraceRecord."""
        # Parse JSON fields that come back as strings
        for json_field in (
            "model_ref",
            "artifact_refs",
            "tool_calls",
            "retrieval_refs",
            "routing_decision",
            "error_detail",
            "inputs_snapshot",
            "outputs_snapshot",
        ):
            val = row.get(json_field)
            if isinstance(val, str):
                try:
                    row[json_field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass

        return TraceRecord(
            trace_id=row["trace_id"],
            job_id=row["job_id"],
            step_name=row["step_name"],
            step_type=row.get("step_type", "generic"),
            attempt_id=row.get("attempt_id"),
            execution_id=row.get("execution_id"),
            parent_trace_id=row.get("parent_trace_id"),
            status=row.get("status", "pending"),
            progress=row.get("progress", 0.0),
            inputs_hash=row.get("inputs_hash"),
            outputs_hash=row.get("outputs_hash"),
            inputs_ref=row.get("inputs_ref"),
            outputs_ref=row.get("outputs_ref"),
            model_ref=row.get("model_ref"),
            artifact_refs=row.get("artifact_refs") or [],
            tool_calls=row.get("tool_calls") or [],
            retrieval_refs=row.get("retrieval_refs") or [],
            routing_decision=row.get("routing_decision"),
            error=row.get("error"),
            error_detail=row.get("error_detail"),
            inputs_snapshot=row.get("inputs_snapshot"),
            outputs_snapshot=row.get("outputs_snapshot"),
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
            created_at=row.get("created_at") or _now(),
            ttl_expires_at=row.get("ttl_expires_at"),
        )
