"""Postgres-backed experiment registry (durable K1 implementation).

Implements the same interface as :class:`InMemoryExperimentRegistry`
against the ``experiments`` / ``experiment_results`` tables (migration
0059). The in-memory registry remains the reference implementation and
test fake.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from . import (
    Experiment,
    ExperimentNotFoundError,
    ExperimentResult,
    ExperimentStatus,
    InMemoryExperimentRegistry,
    InvalidStatusTransitionError,
)

logger = logging.getLogger(__name__)

_EXP_COLUMNS = (
    "experiment_id, project_id, hypothesis_id, protocol, parameters, "
    "validation_method, status, created_at, updated_at, created_by, "
    "submitted_at, completed_at, notes"
)
_RES_COLUMNS = (
    "result_id, experiment_id, metrics, artifacts, logs, status, created_at, created_by"
)


def _row_to_result(row: dict[str, Any]) -> ExperimentResult:
    data = dict(row)
    for key in ("metrics", "artifacts"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    return ExperimentResult.from_dict(data)


def _row_to_experiment(
    row: dict[str, Any], results: list[ExperimentResult]
) -> Experiment:
    data = dict(row)
    for key in ("protocol", "parameters"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    data["results"] = [r.to_dict() for r in results]
    return Experiment.from_dict(data)


class PgExperimentRegistry:
    """Durable experiment registry backed by Postgres."""

    # Same transition table as the in-memory registry.
    _VALID_TRANSITIONS = InMemoryExperimentRegistry._VALID_TRANSITIONS

    def __init__(self, db_pool: Any) -> None:
        self._pool = db_pool

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
        """Create a new experiment and return its experiment_id."""
        if not validation_method:
            raise ValueError("validation_method must be non-empty")
        exp = Experiment(
            project_id=project_id,
            hypothesis_id=hypothesis_id,
            protocol=protocol or {},
            parameters=parameters or {},
            validation_method=validation_method,
            created_by=created_by,
            notes=notes,
        )
        with self._cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO experiments ({_EXP_COLUMNS})
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s,
                        %s, %s, %s)
                """,
                (
                    exp.experiment_id,
                    exp.project_id,
                    exp.hypothesis_id,
                    json.dumps(exp.protocol),
                    json.dumps(exp.parameters),
                    exp.validation_method,
                    exp.status.value,
                    exp.created_at,
                    exp.updated_at,
                    exp.created_by,
                    exp.submitted_at,
                    exp.completed_at,
                    exp.notes,
                ),
            )
        return exp.experiment_id

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get an experiment (with results) by ID, or None."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_EXP_COLUMNS} FROM experiments WHERE experiment_id = %s",
                (experiment_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            results = self._fetch_results(cur, [experiment_id])
        return _row_to_experiment(row, results.get(experiment_id, []))

    def update_experiment_status(
        self, experiment_id: str, status: ExperimentStatus
    ) -> bool:
        """Update status with transition validation; True when found."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT status FROM experiments WHERE experiment_id = %s FOR UPDATE",
                (experiment_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            current = ExperimentStatus(str(row["status"]))
            valid_next = self._VALID_TRANSITIONS.get(current, set())
            if status not in valid_next and status != current:
                raise InvalidStatusTransitionError(
                    f"Cannot transition from {current.value} to {status.value}"
                )
            now = datetime.now(timezone.utc).isoformat()
            submitted, completed = self._timestamps_for(status, now)
            cur.execute(
                "UPDATE experiments SET status = %s, updated_at = %s, "
                "submitted_at = COALESCE(%s, submitted_at), "
                "completed_at = COALESCE(%s, completed_at) "
                "WHERE experiment_id = %s",
                (status.value, now, submitted, completed, experiment_id),
            )
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
        """Append a result; raises ExperimentNotFoundError when missing."""
        result = ExperimentResult(
            experiment_id=experiment_id,
            metrics=metrics or {},
            artifacts=artifacts or [],
            logs=logs,
            status=status,
            created_by=created_by,
        )
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO experiment_results ({_RES_COLUMNS})
                    VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
                    """,
                    (
                        result.result_id,
                        result.experiment_id,
                        json.dumps(result.metrics),
                        json.dumps(result.artifacts),
                        result.logs,
                        result.status,
                        result.created_at,
                        result.created_by,
                    ),
                )
                cur.execute(
                    "UPDATE experiments SET updated_at = %s WHERE experiment_id = %s",
                    (now, experiment_id),
                )
        except Exception as exc:
            if _is_fk_violation(exc):
                raise ExperimentNotFoundError(
                    f"Experiment {experiment_id} not found"
                ) from exc
            raise
        return result.result_id

    def list_experiments(
        self,
        *,
        project_id: str | None = None,
        hypothesis_id: str | None = None,
        status: ExperimentStatus | None = None,
        limit: int = 100,
    ) -> list[Experiment]:
        """List experiments (with results), newest first."""
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if hypothesis_id:
            clauses.append("hypothesis_id = %s")
            params.append(hypothesis_id)
        if status:
            clauses.append("status = %s")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_EXP_COLUMNS} FROM experiments {where} "
                "ORDER BY created_at DESC LIMIT %s",
                params,
            )
            rows = cur.fetchall()
            if not rows:
                return []
            results = self._fetch_results(cur, [r["experiment_id"] for r in rows])
        return [
            _row_to_experiment(row, results.get(row["experiment_id"], []))
            for row in rows
        ]

    def get_experiment_history(self, project_id: str) -> list[Experiment]:
        """Experiments for a project, oldest first."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_EXP_COLUMNS} FROM experiments "
                "WHERE project_id = %s ORDER BY created_at ASC",
                (project_id,),
            )
            rows = cur.fetchall()
            if not rows:
                return []
            results = self._fetch_results(cur, [r["experiment_id"] for r in rows])
        return [
            _row_to_experiment(row, results.get(row["experiment_id"], []))
            for row in rows
        ]

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _fetch_results(
        cur, experiment_ids: list[str]
    ) -> dict[str, list[ExperimentResult]]:
        cur.execute(
            f"SELECT {_RES_COLUMNS} FROM experiment_results "
            "WHERE experiment_id = ANY(%s) ORDER BY created_at ASC",
            (experiment_ids,),
        )
        grouped: dict[str, list[ExperimentResult]] = {}
        for row in cur.fetchall():
            result = _row_to_result(row)
            grouped.setdefault(result.experiment_id, []).append(result)
        return grouped

    @staticmethod
    def _timestamps_for(
        status: ExperimentStatus, now: str
    ) -> tuple[str | None, str | None]:
        submitted = now if status == ExperimentStatus.SUBMITTED else None
        completed = (
            now
            if status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED)
            else None
        )
        return submitted, completed


def _is_fk_violation(exc: Exception) -> bool:
    try:
        from psycopg2 import errors as pg_errors

        return isinstance(exc, pg_errors.ForeignKeyViolation)
    except ImportError:
        return False
