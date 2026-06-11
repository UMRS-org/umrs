"""Checkpoint store for long-running HPC step resumption (MEM-B6).

Provides durable persistence for intermediate checkpoints during
long-running Slurm/HPC steps.  A checkpoint captures a serialized
snapshot of step state at a given iteration so that the step can
resume from that point after a crash or preemption.

This module lives in ``umrs.jobs`` because it reads/writes durable state
tied to jobs/attempts and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _generate_checkpoint_id() -> str:
    return f"ckpt-{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


@dataclass
class Checkpoint:
    """Immutable snapshot of intermediate step state."""

    checkpoint_id: str
    job_id: str
    attempt_id: str
    step_name: str
    sequence_no: int = 0
    state_uri: str | None = None  # S3 path to serialized state
    state_size_bytes: int = 0
    state_inline: dict[str, Any] | None = None  # inline for small payloads (<64KB)
    iteration: int = 0
    progress: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "step_name": self.step_name,
            "sequence_no": self.sequence_no,
            "state_uri": self.state_uri,
            "state_size_bytes": self.state_size_bytes,
            "state_inline": self.state_inline,
            "iteration": self.iteration,
            "progress": self.progress,
            "metadata": self.metadata,
            "created_at": _iso(self.created_at),
        }


_MAX_INLINE_BYTES = 65536  # 64KB


class CheckpointStore:
    """Durable checkpoint store backed by Postgres.

    Falls back to in-memory storage when ``db_url`` is not set (dev mode).
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
                    "checkpoints will use in-memory store"
                )
        self._mem: dict[str, Checkpoint] = {}

    @property
    def db_enabled(self) -> bool:
        return self._db_url is not None and self._psycopg2 is not None

    def _conn(self):  # type: ignore[no-untyped-def]
        return self._psycopg2.connect(self._db_url)  # type: ignore[union-attr]

    # ── Create ──────────────────────────────────────────────

    def save_checkpoint(
        self,
        job_id: str,
        attempt_id: str,
        step_name: str,
        *,
        state_uri: str | None = None,
        state_inline: dict[str, Any] | None = None,
        state_size_bytes: int = 0,
        iteration: int = 0,
        progress: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Persist a checkpoint for the given attempt/step."""
        seq = self._next_sequence(job_id, attempt_id, step_name)
        ckpt = Checkpoint(
            checkpoint_id=_generate_checkpoint_id(),
            job_id=job_id,
            attempt_id=attempt_id,
            step_name=step_name,
            sequence_no=seq,
            state_uri=state_uri,
            state_inline=state_inline,
            state_size_bytes=state_size_bytes,
            iteration=iteration,
            progress=progress,
            metadata=metadata or {},
        )
        if self.db_enabled:
            self._db_insert(ckpt)
        else:
            self._mem[ckpt.checkpoint_id] = ckpt
        logger.info(
            "Checkpoint saved: %s (job=%s attempt=%s step=%s seq=%d iter=%d)",
            ckpt.checkpoint_id,
            job_id,
            attempt_id,
            step_name,
            seq,
            iteration,
        )
        return ckpt

    # ── Read ────────────────────────────────────────────────

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Retrieve a single checkpoint by ID."""
        if self.db_enabled:
            return self._db_get(checkpoint_id)
        return self._mem.get(checkpoint_id)

    def latest_checkpoint(
        self,
        job_id: str,
        attempt_id: str,
        step_name: str | None = None,
    ) -> Checkpoint | None:
        """Return the most recent checkpoint for the given attempt (and optionally step)."""
        if self.db_enabled:
            return self._db_latest(job_id, attempt_id, step_name)
        candidates = [
            c
            for c in self._mem.values()
            if c.job_id == job_id
            and c.attempt_id == attempt_id
            and (step_name is None or c.step_name == step_name)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.sequence_no)

    def list_checkpoints(
        self,
        job_id: str,
        attempt_id: str | None = None,
    ) -> list[Checkpoint]:
        """List checkpoints for a job, optionally filtered by attempt."""
        if self.db_enabled:
            return self._db_list(job_id, attempt_id)
        return sorted(
            [
                c
                for c in self._mem.values()
                if c.job_id == job_id
                and (attempt_id is None or c.attempt_id == attempt_id)
            ],
            key=lambda c: c.sequence_no,
        )

    def select_resume_checkpoint(self, job_id: str) -> Checkpoint | None:
        """Select the best checkpoint across all attempts for job resumption.

        Returns the latest checkpoint from any attempt, useful when
        the reconciler needs to decide whether a job can resume.
        """
        all_ckpts = self.list_checkpoints(job_id)
        if not all_ckpts:
            return None
        return max(all_ckpts, key=lambda c: (c.sequence_no, c.created_at))

    # ── Cleanup ─────────────────────────────────────────────

    def delete_checkpoints(self, job_id: str, attempt_id: str | None = None) -> int:
        """Delete checkpoints for a job (all or specific attempt). Returns count deleted."""
        if self.db_enabled:
            return self._db_delete(job_id, attempt_id)
        before = len(self._mem)
        self._mem = {
            k: v
            for k, v in self._mem.items()
            if not (
                v.job_id == job_id
                and (attempt_id is None or v.attempt_id == attempt_id)
            )
        }
        return before - len(self._mem)

    # ── Private: sequence ───────────────────────────────────

    def _next_sequence(self, job_id: str, attempt_id: str, step_name: str) -> int:
        latest = self.latest_checkpoint(job_id, attempt_id, step_name)
        return (latest.sequence_no + 1) if latest else 0

    # ── Private: Postgres ───────────────────────────────────

    def _db_insert(self, ckpt: Checkpoint) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO checkpoints
                       (checkpoint_id, job_id, attempt_id, step_name,
                        sequence_no, state_uri, state_size_bytes,
                        state_inline, iteration, progress, metadata,
                        created_at)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        ckpt.checkpoint_id,
                        ckpt.job_id,
                        ckpt.attempt_id,
                        ckpt.step_name,
                        ckpt.sequence_no,
                        ckpt.state_uri,
                        ckpt.state_size_bytes,
                        json.dumps(ckpt.state_inline) if ckpt.state_inline else None,
                        ckpt.iteration,
                        ckpt.progress,
                        json.dumps(ckpt.metadata),
                        ckpt.created_at,
                    ),
                )

    def _db_get(self, checkpoint_id: str) -> Checkpoint | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM checkpoints WHERE checkpoint_id = %s",
                    (checkpoint_id,),
                )
                row = cur.fetchone()
                return self._row_to_checkpoint(row, cur) if row else None

    def _db_latest(
        self,
        job_id: str,
        attempt_id: str,
        step_name: str | None,
    ) -> Checkpoint | None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                if step_name:
                    cur.execute(
                        """SELECT * FROM checkpoints
                           WHERE job_id = %s AND attempt_id = %s AND step_name = %s
                           ORDER BY sequence_no DESC LIMIT 1""",
                        (job_id, attempt_id, step_name),
                    )
                else:
                    cur.execute(
                        """SELECT * FROM checkpoints
                           WHERE job_id = %s AND attempt_id = %s
                           ORDER BY sequence_no DESC LIMIT 1""",
                        (job_id, attempt_id),
                    )
                row = cur.fetchone()
                return self._row_to_checkpoint(row, cur) if row else None

    def _db_list(self, job_id: str, attempt_id: str | None) -> list[Checkpoint]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                if attempt_id:
                    cur.execute(
                        """SELECT * FROM checkpoints
                           WHERE job_id = %s AND attempt_id = %s
                           ORDER BY sequence_no ASC""",
                        (job_id, attempt_id),
                    )
                else:
                    cur.execute(
                        """SELECT * FROM checkpoints
                           WHERE job_id = %s ORDER BY sequence_no ASC""",
                        (job_id,),
                    )
                return [self._row_to_checkpoint(r, cur) for r in cur.fetchall()]

    def _db_delete(self, job_id: str, attempt_id: str | None) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                if attempt_id:
                    cur.execute(
                        "DELETE FROM checkpoints WHERE job_id = %s AND attempt_id = %s",
                        (job_id, attempt_id),
                    )
                else:
                    cur.execute(
                        "DELETE FROM checkpoints WHERE job_id = %s",
                        (job_id,),
                    )
                return cur.rowcount  # type: ignore[return-value]

    def _row_to_checkpoint(self, row: tuple, cur: Any) -> Checkpoint:
        cols = [desc[0] for desc in cur.description]
        d = dict(zip(cols, row))
        inline = d.get("state_inline")
        if isinstance(inline, str):
            inline = json.loads(inline)
        meta = d.get("metadata", {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        return Checkpoint(
            checkpoint_id=d["checkpoint_id"],
            job_id=d["job_id"],
            attempt_id=d["attempt_id"],
            step_name=d["step_name"],
            sequence_no=d.get("sequence_no", 0),
            state_uri=d.get("state_uri"),
            state_size_bytes=d.get("state_size_bytes", 0),
            state_inline=inline,
            iteration=d.get("iteration", 0),
            progress=d.get("progress", 0.0),
            metadata=meta,
            created_at=d.get("created_at", _now()),
        )
