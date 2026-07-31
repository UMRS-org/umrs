"""Postgres-backed hypothesis lifecycle (durable K1-tail implementation).

Implements the same interface as :class:`HypothesisLifecycle` against the
``hypotheses`` / ``hypothesis_history`` tables (migration 0061). The
in-memory lifecycle remains the reference implementation and test fake.

Contract notes:
  - Status transitions are validated with the same ``_VALID_TRANSITIONS``
    map; the current status row is locked ``FOR UPDATE`` during transition
    so concurrent writers serialize instead of silently clobbering.
  - ``hypothesis_history`` is append-only and monotonic via BIGSERIAL seq,
    replayable as (status, at) tuples exactly like the in-memory list.
  - Evidence-list mutations (add_supporting/contradicting) read-modify-write
    inside one transaction and recompute confidence with the same formula.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from .hypothesis_lifecycle import (
    _VALID_TRANSITIONS,
    Hypothesis,
    HypothesisStatus,
)

logger = logging.getLogger(__name__)

_COLUMNS = (
    "hypothesis_id, statement, status, confidence, supporting_evidence, "
    "contradicting_evidence, originating_planner_version, "
    "associated_experiments, parent_hypothesis_id, metadata, "
    "created_at, updated_at"
)


def _row_to_hypothesis(row: dict[str, Any]) -> Hypothesis:
    def _list(value: Any) -> list:
        if isinstance(value, str):
            return json.loads(value)
        return list(value or [])

    return Hypothesis(
        hypothesis_id=str(row["hypothesis_id"]),
        statement=str(row["statement"]),
        status=HypothesisStatus(str(row["status"])),
        confidence=float(row["confidence"]),
        supporting_evidence=_list(row["supporting_evidence"]),
        contradicting_evidence=_list(row["contradicting_evidence"]),
        originating_planner_version=str(row["originating_planner_version"]),
        associated_experiments=_list(row["associated_experiments"]),
        parent_hypothesis_id=row["parent_hypothesis_id"],
        metadata=(
            json.loads(row["metadata"])
            if isinstance(row["metadata"], str)
            else dict(row["metadata"] or {})
        ),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


class PgHypothesisLifecycle:
    """Durable hypothesis lifecycle backed by Postgres."""

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

    def propose(
        self,
        statement: str,
        *,
        planner_version: str = "",
        parent_hypothesis_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Propose a new hypothesis; returns its hypothesis_id."""
        hyp = Hypothesis(
            statement=statement,
            status=HypothesisStatus.PROPOSED,
            originating_planner_version=planner_version,
            parent_hypothesis_id=parent_hypothesis_id,
            metadata=metadata or {},
        )
        with self._cursor() as cur:
            cur.execute(
                f"INSERT INTO hypotheses ({_COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, "
                "%s, %s::jsonb, %s, %s)",
                (
                    hyp.hypothesis_id,
                    hyp.statement,
                    hyp.status.value,
                    hyp.confidence,
                    json.dumps(hyp.supporting_evidence),
                    json.dumps(hyp.contradicting_evidence),
                    hyp.originating_planner_version,
                    json.dumps(hyp.associated_experiments),
                    hyp.parent_hypothesis_id,
                    json.dumps(hyp.metadata),
                    hyp.created_at,
                    hyp.updated_at,
                ),
            )
            self._append_history(cur, hyp.hypothesis_id, hyp.status, hyp.created_at)
        return hyp.hypothesis_id

    def transition(
        self, hypothesis_id: str, new_status: HypothesisStatus
    ) -> HypothesisStatus:
        """Transition status with validation; KeyError/ValueError on misuse."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            cur.execute(
                "SELECT status FROM hypotheses WHERE hypothesis_id = %s FOR UPDATE",
                (hypothesis_id,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Hypothesis {hypothesis_id} not found")
            current = HypothesisStatus(str(row["status"]))
            valid_targets = _VALID_TRANSITIONS.get(current, set())
            if new_status not in valid_targets:
                raise ValueError(
                    f"Invalid transition: {current.value} → {new_status.value}"
                )
            cur.execute(
                "UPDATE hypotheses SET status = %s, updated_at = %s "
                "WHERE hypothesis_id = %s",
                (new_status.value, now, hypothesis_id),
            )
            self._append_history(cur, hypothesis_id, new_status, now)
        return new_status

    def add_evidence(
        self, hypothesis_id: str, evidence_id: str, supports: bool
    ) -> None:
        """Attach evidence id, recomputing confidence."""
        now = datetime.now(timezone.utc).isoformat()
        with self._cursor() as cur:
            row = self._lock_row(cur, hypothesis_id)
            hyp = _row_to_hypothesis(row)
            if supports:
                hyp.add_supporting_evidence(evidence_id)
            else:
                hyp.add_contradicting_evidence(evidence_id)
            self._persist_lists(cur, hyp, now)

    def get(self, hypothesis_id: str) -> Hypothesis | None:
        """Get a hypothesis by ID, or None."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM hypotheses WHERE hypothesis_id = %s",
                (hypothesis_id,),
            )
            row = cur.fetchone()
        return _row_to_hypothesis(row) if row else None

    def get_history(self, hypothesis_id: str) -> list[tuple[HypothesisStatus, str]]:
        """Transition log, oldest first."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT status, at FROM hypothesis_history "
                "WHERE hypothesis_id = %s ORDER BY seq ASC",
                (hypothesis_id,),
            )
            return [
                (HypothesisStatus(str(r["status"])), str(r["at"]))
                for r in cur.fetchall()
            ]

    def list_hypotheses(
        self, *, status: HypothesisStatus | None = None
    ) -> list[Hypothesis]:
        """List hypotheses (optionally status-filtered), oldest first."""
        if status is None:
            sql = f"SELECT {_COLUMNS} FROM hypotheses ORDER BY created_at ASC"
            params: list[Any] = []
        else:
            sql = (
                f"SELECT {_COLUMNS} FROM hypotheses WHERE status = %s "
                "ORDER BY created_at ASC"
            )
            params = [status.value]
        with self._cursor() as cur:
            cur.execute(sql, params)
            return [_row_to_hypothesis(row) for row in cur.fetchall()]

    def refine(
        self,
        parent_hypothesis_id: str,
        new_statement: str,
        *,
        planner_version: str = "",
    ) -> str:
        """Refine: parent → REFINED, new hypothesis links back. Single tx."""
        now = datetime.now(timezone.utc).isoformat()
        child = Hypothesis(
            statement=new_statement,
            status=HypothesisStatus.PROPOSED,
            originating_planner_version=planner_version,
            parent_hypothesis_id=parent_hypothesis_id,
        )
        with self._cursor() as cur:
            cur.execute(
                "SELECT status FROM hypotheses WHERE hypothesis_id = %s FOR UPDATE",
                (parent_hypothesis_id,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"Hypothesis {parent_hypothesis_id} not found")
            current = HypothesisStatus(str(row["status"]))
            target = HypothesisStatus.REFINED
            if target not in _VALID_TRANSITIONS.get(current, set()):
                raise ValueError(
                    f"Invalid transition: {current.value} → {target.value}"
                )
            cur.execute(
                "UPDATE hypotheses SET status = %s, updated_at = %s "
                "WHERE hypothesis_id = %s",
                (target.value, now, parent_hypothesis_id),
            )
            self._append_history(cur, parent_hypothesis_id, target, now)
            cur.execute(
                f"INSERT INTO hypotheses ({_COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb, "
                "%s, %s::jsonb, %s, %s)",
                (
                    child.hypothesis_id,
                    child.statement,
                    child.status.value,
                    child.confidence,
                    json.dumps(child.supporting_evidence),
                    json.dumps(child.contradicting_evidence),
                    child.originating_planner_version,
                    json.dumps(child.associated_experiments),
                    child.parent_hypothesis_id,
                    json.dumps(child.metadata),
                    child.created_at,
                    child.updated_at,
                ),
            )
            self._append_history(
                cur, child.hypothesis_id, child.status, child.created_at
            )
        return child.hypothesis_id

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _append_history(
        cur, hypothesis_id: str, status: HypothesisStatus, at: str
    ) -> None:
        cur.execute(
            "INSERT INTO hypothesis_history (hypothesis_id, status, at) "
            "VALUES (%s, %s, %s)",
            (hypothesis_id, status.value, at),
        )

    def _lock_row(self, cur, hypothesis_id: str) -> dict[str, Any]:
        cur.execute(
            f"SELECT {_COLUMNS} FROM hypotheses WHERE hypothesis_id = %s FOR UPDATE",
            (hypothesis_id,),
        )
        row = cur.fetchone()
        if not row:
            raise KeyError(f"Hypothesis {hypothesis_id} not found")
        return row

    def _persist_lists(self, cur, hyp: Hypothesis, at: str) -> None:
        cur.execute(
            "UPDATE hypotheses SET supporting_evidence = %s::jsonb, "
            "contradicting_evidence = %s::jsonb, confidence = %s, "
            "updated_at = %s WHERE hypothesis_id = %s",
            (
                json.dumps(hyp.supporting_evidence),
                json.dumps(hyp.contradicting_evidence),
                hyp.confidence,
                at,
                hyp.hypothesis_id,
            ),
        )
