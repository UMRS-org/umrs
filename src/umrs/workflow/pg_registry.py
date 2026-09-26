"""Postgres-backed workflow execution registry (W1).

Durable counterpart to :class:`WorkflowExecutionRegistry` in
``umrs.server.workflow_execution_registry``. Design:

- ``PgWorkflowExecutionRegistry`` **subclasses** the in-memory registry, so
  every existing call site (``create/get/list/cancel/run_in_thread``) keeps
  identical semantics.
- ``DurableExecutionRecord`` fires a persist hook on every mutation
  (``update`` / ``set_step`` / ``add_artifact``) — the runner's in-place
  mutation pattern becomes durable without call-site changes.
- Read-through: on a cold process (restart/failover), ``get()`` falls back
  to Postgres and warms the local map, making existing executions visible
  again (``r4 workflow status/watch`` continuity).
- Factory ``build_execution_registry()`` mirrors the satellite conventions
  (``UMRS_WORKFLOW_EXECUTION_REGISTRY=pg|memory|auto``). Storage is the
  ``workflow_executions`` table (migration 0062).

All DB failures degrade to the in-memory path (log + continue), matching the
current runner's tolerance for the registry being absent.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

from ..server.workflow_execution_registry import (
    ExecutionRecord,
    StepProgress,
    WorkflowExecutionRegistry,
)

logger = logging.getLogger(__name__)

_COLUMNS = (
    "execution_id, workflow_name, user_id, status, submitted_at, record_json, "
    "updated_at"
)


class DurableExecutionRecord(ExecutionRecord):
    """ExecutionRecord that invokes a persist hook after each mutation."""

    _persist_hook: Callable[[DurableExecutionRecord], None] | None = None
    _refresh_hook: Callable[[DurableExecutionRecord], dict[str, Any] | None] | None = (
        None
    )

    def _fire_persist(self) -> None:
        hook = self._persist_hook
        if hook is not None:
            hook(self)

    def refresh_from_store(self) -> None:
        """Merge cross-process fields from the durable snapshot.

        Gate decisions and cancel requests written by another worker (the
        approve/cancel API is not guaranteed to run in the process parking
        on this record's threading events) are union-merged; first decision
        wins, matching ``resolve_gate``. DB failures degrade silently so the
        caller's wait loop keeps its in-process semantics.
        """
        hook = self._refresh_hook
        if hook is None:
            return
        try:
            payload = hook(self)
        except Exception:
            logger.debug("workflow_executions refresh failed", exc_info=True)
            return
        if not payload:
            return
        with self._lock:
            for step_name, decision in (payload.get("gates") or {}).items():
                self.gate_decisions.setdefault(step_name, dict(decision))
            if payload.get("cancel_requested"):
                self.cancel_requested = True

    def update(self, **fields_: Any) -> None:
        super().update(**fields_)
        self._fire_persist()

    def set_step(self, name: str, **fields_: Any) -> None:
        super().set_step(name, **fields_)
        self._fire_persist()

    def update_step_entry(
        self, name: str, key: str, entry: str, facts: dict[str, Any]
    ) -> None:
        super().update_step_entry(name, key, entry, facts)
        self._fire_persist()

    def add_artifact(self, key: str) -> None:
        super().add_artifact(key)
        self._fire_persist()


_BLOB_TAG = "base64"


def _encode_blobs(blobs: Any) -> dict[str, Any]:
    """Encode byte blob values for JSON; non-dict/legacy values pass through."""
    if not isinstance(blobs, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in blobs.items():
        if isinstance(value, (bytes, bytearray)):
            out[key] = {
                "_blob_encoding": _BLOB_TAG,
                "data": base64.b64encode(bytes(value)).decode("ascii"),
            }
        elif isinstance(value, dict) and value.get("_blob_encoding") == _BLOB_TAG:
            out[key] = value  # already encoded (idempotent)
        else:
            out[key] = value
    return out


def _decode_blobs(blobs: Any) -> dict[str, Any]:
    """Decode base64 blob wrappers back to bytes; legacy values pass through."""
    if not isinstance(blobs, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in blobs.items():
        if isinstance(value, dict) and value.get("_blob_encoding") == _BLOB_TAG:
            try:
                out[key] = base64.b64decode(value.get("data", ""))
            except Exception:
                out[key] = b""
        else:
            out[key] = value
    return out


def _record_json(record: ExecutionRecord) -> str:
    """Canonical snapshot for durable storage.

    Includes ``results.blobs`` (base64-wrapped) so output bytes survive the
    update→hydrate round trip: ``to_dict()`` intentionally omits ``results``,
    but the durable workflow_run completion path must see step outputs even
    when the checker resolves the record from Postgres instead of memory.
    """
    payload = record.to_dict()
    results = dict(getattr(record, "results", None) or {})
    if results:
        results = dict(results)
        results["blobs"] = _encode_blobs(results.get("blobs"))
        payload["results"] = results
    return json.dumps(payload, default=str)


def _steps_from_payload(steps: Any) -> dict[str, StepProgress]:
    """Materialize the steps map, tolerating both list and dict payloads."""
    if isinstance(steps, dict):
        steps = list(steps.values())
    result: dict[str, StepProgress] = {}
    for step in steps or []:
        result[str(step["name"])] = StepProgress(
            name=str(step["name"]),
            status=str(step.get("status", "pending")),
            started_at=step.get("started_at"),
            finished_at=step.get("finished_at"),
            error=step.get("error"),
            progress=float(step.get("progress") or 0.0),
            extra=dict(step.get("extra") or {}),
        )
    return result


def _record_from_payload(data: dict[str, Any]) -> DurableExecutionRecord:
    """Materialize a DurableExecutionRecord from a ``to_dict()`` payload."""
    record = DurableExecutionRecord(
        execution_id=str(data.get("execution_id", "")),
        workflow_name=str(data.get("workflow_name", "")),
        user_id=str(data.get("user_id", "")),
        params=dict(data.get("params") or {}),
        status=str(data.get("status", "submitted")),
        submitted_at=float(data.get("submitted_at") or 0),
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
        error=data.get("error"),
        progress=float(data.get("progress") or 0.0),
        durable_job_id=data.get("durable_job_id"),
        cancel_requested=bool(data.get("cancel_requested", False)),
        identity_provenance=str(data.get("identity_provenance", "unverified")),
    )
    record.steps = _steps_from_payload(data.get("steps"))
    results = dict(data.get("results") or {})
    if "blobs" in results:
        results["blobs"] = _decode_blobs(results.get("blobs"))
    record.results = results
    record.artifacts = list(data.get("artifacts") or [])
    return record


class PgWorkflowExecutionRegistry(WorkflowExecutionRegistry):
    """In-memory execution registry with durable Postgres mirroring."""

    def __init__(self, db_pool: Any, max_records: int = 1000) -> None:
        super().__init__(max_records=max_records)
        self._pool = db_pool

    def _install_hooks(self, record: DurableExecutionRecord) -> None:
        """Wire persist + refresh hooks for a durable record."""
        record._persist_hook = self._persist
        record._refresh_hook = lambda rec: self._fetch_payload(rec.execution_id)

    def _fetch_payload(self, execution_id: str) -> dict[str, Any] | None:
        """Return the stored ``record_json`` payload for refresh merges."""
        row = self._fetch(execution_id)
        if row is None:
            return None
        data = row.get("record_json")
        return data if isinstance(data, dict) else None

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            with conn:
                yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _cursor(self):
        from psycopg2.extras import RealDictCursor

        with self._conn() as conn:
            yield conn.cursor(cursor_factory=RealDictCursor)

    # ── overrides with persistence ───────────────────────────────────────

    def create(
        self,
        workflow_name: str,
        user_id: str,
        params: dict[str, Any] | None = None,
        execution_id: str | None = None,
        durable_job_id: str | None = None,
        identity_provenance: str = "unverified",
    ) -> ExecutionRecord:
        """Create a durable execution record (memory + PG mirror)."""
        import uuid

        record = DurableExecutionRecord(
            execution_id=execution_id or f"wf-{uuid.uuid4().hex[:12]}",
            workflow_name=workflow_name,
            user_id=user_id,
            params=dict(params or {}),
            durable_job_id=durable_job_id,
            identity_provenance=identity_provenance,
        )
        self._install_hooks(record)
        with self._lock:
            self._put_locked(record)
        self._persist(record)
        return record

    def get(self, execution_id: str) -> ExecutionRecord | None:
        """Read-through: memory map first, else Postgres (warms cache)."""
        record = self._records.get(execution_id)
        if record is not None:
            return record
        row = self._fetch(execution_id)
        if row is None:
            return None
        record = _record_from_payload(row["record_json"])
        self._install_hooks(record)
        with self._lock:
            self._records[execution_id] = record
        return record

    def list(self, *, user_id: str | None = None) -> list[ExecutionRecord]:
        """Warm from Postgres then list locally (newest first)."""
        self._warm(user_id=user_id)
        return super().list(user_id=user_id)

    # ── internals ─────────────────────────────────────────────────────────

    def _persist(self, record: ExecutionRecord) -> None:
        """Upsert the record snapshot; degrade silently on failure."""
        try:
            with self._cursor() as cur:
                cur.execute(
                    f"INSERT INTO workflow_executions ({_COLUMNS}) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s) "
                    "ON CONFLICT (execution_id) DO UPDATE SET "
                    "workflow_name = excluded.workflow_name, "
                    "user_id = excluded.user_id, "
                    "status = excluded.status, "
                    "submitted_at = excluded.submitted_at, "
                    "record_json = excluded.record_json, "
                    "updated_at = excluded.updated_at",
                    (
                        record.execution_id,
                        record.workflow_name,
                        record.user_id,
                        record.status,
                        float(record.submitted_at or 0),
                        _record_json(record),
                        time.time(),
                    ),
                )
        except Exception:
            logger.debug("workflow_executions persist failed", exc_info=True)

    def _fetch(self, execution_id: str) -> dict[str, Any] | None:
        try:
            with self._cursor() as cur:
                cur.execute(
                    f"SELECT {_COLUMNS} FROM workflow_executions "
                    "WHERE execution_id = %s",
                    (execution_id,),
                )
                return cur.fetchone()
        except Exception:
            logger.debug("workflow_executions fetch failed", exc_info=True)
            return None

    def _warm(self, *, user_id: str | None = None) -> None:
        """Materialize unseen rows into the local map."""
        try:
            sql = f"SELECT {_COLUMNS} FROM workflow_executions"
            params: tuple = ()
            if user_id:
                sql += " WHERE user_id = %s"
                params = (user_id,)
            with self._cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        except Exception:
            logger.debug("workflow_executions warm failed", exc_info=True)
            return
        with self._lock:
            for row in rows:
                eid = str(row["execution_id"])
                if eid in self._records:
                    continue
                record = _record_from_payload(row["record_json"])
                self._install_hooks(record)
                self._records[eid] = record


def build_execution_registry(db_pool: Any = None) -> Any:
    """Factory for the durable workflow execution registry (W1).

    Backend selection via ``UMRS_WORKFLOW_EXECUTION_REGISTRY`` (default
    ``auto``):

      - ``pg``     — force Postgres (raises if unavailable)
      - ``memory`` — force in-memory (test/quota fallback path)
      - ``auto``   — Postgres when a pool is supplied or env resolvable,
                     else the classic in-memory registry
    """
    import os as _os

    from ..shared.pg import resolve_satellite_pool

    backend = _os.environ.get("UMRS_WORKFLOW_EXECUTION_REGISTRY", "auto").lower()
    if backend == "memory":
        return WorkflowExecutionRegistry()
    if backend not in ("auto", "pg"):
        logger.warning("Unknown workflow registry backend %r; using auto", backend)
        backend = "auto"

    pool = resolve_satellite_pool(backend, db_pool, "Workflow execution registry")
    if pool is None:
        if backend == "pg":
            raise RuntimeError(
                "UMRS_WORKFLOW_EXECUTION_REGISTRY=pg but no db_pool resolvable"
            )
        return WorkflowExecutionRegistry()
    return PgWorkflowExecutionRegistry(pool)


__all__ = [
    "DurableExecutionRecord",
    "PgWorkflowExecutionRegistry",
    "build_execution_registry",
]
