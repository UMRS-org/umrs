"""Persistent research-run store (SQLite) for async identity and recovery."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from r4_contracts.research import ResearchRunHandle, ResearchRunStatus

_ALLOWED_TRANSITIONS: dict[ResearchRunStatus, frozenset[ResearchRunStatus]] = {
    ResearchRunStatus.PENDING: frozenset(
        {
            ResearchRunStatus.RUNNING,
            ResearchRunStatus.CANCELLED,
            ResearchRunStatus.FAILED,
        }
    ),
    ResearchRunStatus.RUNNING: frozenset(
        {
            ResearchRunStatus.COMPLETED,
            ResearchRunStatus.FAILED,
            ResearchRunStatus.CANCELLED,
        }
    ),
}

_TERMINAL = frozenset(
    {
        ResearchRunStatus.COMPLETED,
        ResearchRunStatus.FAILED,
        ResearchRunStatus.CANCELLED,
    }
)

_WORKFLOW_ALIVE = frozenset(
    {
        "accepted",
        "queued",
        "pending",
        "running",
        "started",
        "in_progress",
    }
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class ResearchRunStore:
    """SQLite-backed store for research runs and per-iteration workflow mapping."""

    def __init__(self, storage_path: str | Path) -> None:
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self._storage_path / "research_runs.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                status TEXT NOT NULL,
                simulated INTEGER NOT NULL DEFAULT 0,
                loop_id TEXT NOT NULL DEFAULT '',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS iterations (
                iteration_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                iteration_number INTEGER NOT NULL,
                workflow_execution_id TEXT,
                durable_job_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
            CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
            CREATE INDEX IF NOT EXISTS idx_iterations_run ON iterations(run_id);
            """
        )
        self._conn.commit()

    def create_run(
        self,
        project_id: str,
        *,
        simulated: bool = False,
        loop_id: str = "",
        run_id: str | None = None,
    ) -> ResearchRunHandle:
        now = _now()
        handle = ResearchRunHandle(
            run_id=run_id or _generate_id("run"),
            project_id=project_id,
            status=ResearchRunStatus.PENDING,
            simulated=simulated,
            loop_id=loop_id,
            created_at=now,
            updated_at=now,
        )
        self._conn.execute(
            """
            INSERT INTO runs
                (run_id, project_id, status, simulated, loop_id, error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (
                handle.run_id,
                handle.project_id,
                handle.status.value,
                1 if simulated else 0,
                handle.loop_id,
                handle.created_at,
                handle.updated_at,
            ),
        )
        self._conn.commit()
        return handle

    def get_run(self, run_id: str) -> ResearchRunHandle | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_handle(row)

    def update_status(
        self,
        run_id: str,
        new_status: ResearchRunStatus | str,
        *,
        error: str | None = None,
        loop_id: str | None = None,
        allow_same: bool = False,
    ) -> ResearchRunHandle:
        handle = self.get_run(run_id)
        if handle is None:
            raise KeyError(f"research run not found: {run_id}")

        target = ResearchRunStatus(new_status)
        if handle.status == target and allow_same:
            return handle

        allowed = _ALLOWED_TRANSITIONS.get(handle.status, frozenset())
        if target not in allowed:
            raise ValueError(
                f"illegal status transition {handle.status.value} → {target.value} "
                f"for run {run_id}"
            )

        now = _now()
        cur = self._conn.execute(
            """
            UPDATE runs
               SET status = ?, error = COALESCE(?, error),
                   loop_id = COALESCE(?, loop_id), updated_at = ?
             WHERE run_id = ? AND status = ?
            """,
            (
                target.value,
                error,
                loop_id,
                now,
                run_id,
                handle.status.value,
            ),
        )
        self._conn.commit()
        if cur.rowcount != 1:
            raise RuntimeError(
                f"atomic status update failed for run {run_id} "
                f"({handle.status.value} → {target.value})"
            )
        updated = self.get_run(run_id)
        assert updated is not None
        return updated

    def record_iteration(
        self,
        run_id: str,
        *,
        iteration_number: int,
        workflow_execution_id: str | None = None,
        durable_job_id: str | None = None,
        status: str = "completed",
        payload: dict[str, Any] | None = None,
        iteration_id: str | None = None,
    ) -> str:
        """Persist (or upsert) an iteration checkpoint for a run.

        When ``iteration_id`` already exists, the row is updated in place so
        mid-loop checkpoints (e.g. after workflow trigger) and the final
        completed payload share one durable record.
        """
        if self.get_run(run_id) is None:
            raise KeyError(f"research run not found: {run_id}")
        now = _now()
        iid = iteration_id or _generate_id("iter")
        existing = self._conn.execute(
            "SELECT iteration_id FROM iterations WHERE iteration_id = ?",
            (iid,),
        ).fetchone()
        payload_json = json.dumps(payload or {})
        if existing is not None:
            self._conn.execute(
                """
                UPDATE iterations SET
                    iteration_number = ?,
                    workflow_execution_id = ?,
                    durable_job_id = ?,
                    status = ?,
                    payload_json = ?,
                    updated_at = ?
                WHERE iteration_id = ?
                """,
                (
                    iteration_number,
                    workflow_execution_id,
                    durable_job_id,
                    status,
                    payload_json,
                    now,
                    iid,
                ),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO iterations
                    (iteration_id, run_id, iteration_number, workflow_execution_id,
                     durable_job_id, status, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iid,
                    run_id,
                    iteration_number,
                    workflow_execution_id,
                    durable_job_id,
                    status,
                    payload_json,
                    now,
                    now,
                ),
            )
        self._conn.commit()
        return iid

    def list_iterations(self, run_id: str) -> list[dict[str, Any]]:
        """Return iteration rows for *run_id* (oldest first), including payloads."""
        if self.get_run(run_id) is None:
            raise KeyError(f"research run not found: {run_id}")
        rows = self._conn.execute(
            """
            SELECT iteration_id, run_id, iteration_number, workflow_execution_id,
                   durable_job_id, status, payload_json, created_at, updated_at
              FROM iterations
             WHERE run_id = ?
             ORDER BY iteration_number ASC, created_at ASC
            """,
            (run_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload: dict[str, Any] = {}
            raw = row["payload_json"] or "{}"
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    payload = loaded
            except json.JSONDecodeError:
                payload = {}
            out.append(
                {
                    "iteration_id": row["iteration_id"],
                    "run_id": row["run_id"],
                    "iteration_number": row["iteration_number"],
                    "workflow_execution_id": row["workflow_execution_id"],
                    "durable_job_id": row["durable_job_id"],
                    "status": row["status"],
                    "payload": payload,
                    "created_at": row["created_at"] or "",
                    "updated_at": row["updated_at"] or "",
                }
            )
        return out

    def list_running(self, project_id: str | None = None) -> list[ResearchRunHandle]:
        if project_id:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE status = ? AND project_id = ?",
                (ResearchRunStatus.RUNNING.value, project_id),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE status = ?",
                (ResearchRunStatus.RUNNING.value,),
            ).fetchall()
        return [self._row_to_handle(r) for r in rows]

    def list_runs(self, project_id: str) -> list[ResearchRunHandle]:
        rows = self._conn.execute(
            """
            SELECT * FROM runs WHERE project_id = ?
             ORDER BY created_at ASC
            """,
            (project_id,),
        ).fetchall()
        return [self._row_to_handle(r) for r in rows]

    def cancel(self, run_id: str, *, error: str | None = None) -> ResearchRunHandle:
        """Idempotent cancel — second call returns current status."""
        handle = self.get_run(run_id)
        if handle is None:
            raise KeyError(f"research run not found: {run_id}")
        if handle.status in _TERMINAL:
            return handle
        return self.update_status(
            run_id,
            ResearchRunStatus.CANCELLED,
            error=error or "cancelled",
        )

    def resume(self, run_id: str) -> ResearchRunHandle:
        """Resume a run. Completed/failed/cancelled are no-ops returning the handle."""
        handle = self.get_run(run_id)
        if handle is None:
            raise KeyError(f"research run not found: {run_id}")
        if handle.status in _TERMINAL:
            return handle
        if handle.status == ResearchRunStatus.PENDING:
            return self.update_status(run_id, ResearchRunStatus.RUNNING)
        return handle

    def reconcile_on_startup(
        self,
        *,
        status_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> list[ResearchRunHandle]:
        """Reconcile RUNNING runs against the workflow backend.

        If ``status_fn`` is provided, resume waiting when the workflow is still
        alive; otherwise mark the run FAILED. Without ``status_fn``, any
        RUNNING run is marked FAILED (workflow identity unknown after restart).
        """
        reconciled: list[ResearchRunHandle] = []
        for handle in self.list_running():
            iteration = self._latest_iteration(handle.run_id)
            execution_id = (
                iteration["workflow_execution_id"] if iteration is not None else None
            )
            if not execution_id or status_fn is None:
                updated = self.update_status(
                    handle.run_id,
                    ResearchRunStatus.FAILED,
                    error="workflow died or status unavailable on startup",
                )
                reconciled.append(updated)
                continue

            try:
                payload = status_fn(execution_id)
            except Exception as exc:  # noqa: BLE001 — reconcile must not crash startup
                updated = self.update_status(
                    handle.run_id,
                    ResearchRunStatus.FAILED,
                    error=f"workflow status check failed: {exc}",
                )
                reconciled.append(updated)
                continue

            status = str(payload.get("status", "")).lower()
            if status in _WORKFLOW_ALIVE:
                # Still alive — leave RUNNING for callers to resume waiting.
                reconciled.append(handle)
                continue

            if status in {"succeeded", "completed", "success"}:
                updated = self.update_status(
                    handle.run_id,
                    ResearchRunStatus.COMPLETED,
                )
            elif status in {"cancelled", "canceled"}:
                updated = self.update_status(
                    handle.run_id,
                    ResearchRunStatus.CANCELLED,
                    error="workflow cancelled",
                )
            else:
                updated = self.update_status(
                    handle.run_id,
                    ResearchRunStatus.FAILED,
                    error=payload.get("error") or f"workflow ended with status={status}",
                )
            reconciled.append(updated)
        return reconciled

    def _latest_iteration(self, run_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT * FROM iterations
             WHERE run_id = ?
             ORDER BY iteration_number DESC
             LIMIT 1
            """,
            (run_id,),
        ).fetchone()

    @staticmethod
    def _row_to_handle(row: sqlite3.Row) -> ResearchRunHandle:
        return ResearchRunHandle(
            run_id=row["run_id"],
            project_id=row["project_id"],
            status=ResearchRunStatus(row["status"]),
            simulated=bool(row["simulated"]),
            loop_id=row["loop_id"] or "",
            error=row["error"],
            created_at=row["created_at"] or "",
            updated_at=row["updated_at"] or "",
        )
