"""Durable research-loop store backed by Postgres (migration 0058).

Drop-in replacement for the in-memory ``_loop_store`` in ``umrs.api.loop``.
Falls back to a pure in-memory dict when no pool is provided so unit tests and
development setups keep working identically.

API shape (sync-friendly; the engine is sync):
    store = LoopStore(pool=asyncpg_pool)            # pool-backed
    store = LoopStore.memory()                       # in-memory fallback

    store.save(loop)            # upsert loop + iterations
    store.get(loop_id)          # ResearchLoop | None
    store.list_all()            # list[ResearchLoop]
    store.delete(loop_id)
    store.clear()               # only meaningful for memory mode; tests use it
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop


def _row_to_loop(row: Any) -> ResearchLoop:
    loop = ResearchLoop(
        loop_id=row["loop_id"],
        project_id=row["project_id"],
        research_goal=row["research_goal"],
        status=LoopStatus(row["status"]),
        phase=LoopPhase(row["phase"]),
        max_iterations=row["max_iterations"],
        current_iteration=row["current_iteration"],
        total_hypotheses_tested=row["total_hypotheses_tested"],
        total_experiments_run=row["total_experiments_run"],
        total_findings=row["total_findings"],
        success_rate=row["success_rate"],
        created_at=row["created_at"].isoformat()
        if hasattr(row["created_at"], "isoformat")
        else str(row["created_at"] or ""),
        started_at=row["started_at"] or "",
        completed_at=row["completed_at"] or "",
        updated_at=row["updated_at"].isoformat()
        if hasattr(row["updated_at"], "isoformat")
        else str(row["updated_at"] or ""),
        metadata=json.loads(row["metadata"] or "{}"),
    )
    return loop


_F_KEYS = (
    "iteration_id",
    "loop_id",
    "iteration_number",
    "started_at",
    "completed_at",
    "hypothesis",
    "experiment_id",
    "analysis",
    "evaluation",
    "status",
)


def _row_to_iteration(row: Any) -> LoopIteration:
    base = {k: row[k] for k in _F_KEYS}
    for k in (
        "started_at",
        "completed_at",
        "hypothesis",
        "experiment_id",
        "analysis",
        "evaluation",
        "status",
    ):
        base[k] = base[k] or ("" if k != "status" else "pending")
    return LoopIteration(
        **base,
        phase=LoopPhase(row["phase"]),
        results=json.loads(row["results"] or "{}"),
        findings=json.loads(row["findings"] or "[]"),
    )


class LoopStore:
    """Postgres-backed loop store with in-memory fallback mode."""

    def __init__(self, pool: Any | None) -> None:
        self._pool = pool
        self._memory: dict[str, ResearchLoop] = {}

    @property
    def is_memory(self) -> bool:
        return self._pool is None

    # -- async pool ops ---------------------------------------------------- #

    async def _save_pg(self, loop: ResearchLoop) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO research_loops
                   (loop_id, project_id, research_goal, status, phase,
                    max_iterations, current_iteration, plan,
                    total_hypotheses_tested, total_experiments_run,
                    total_findings, success_rate,
                    created_at, started_at, completed_at, updated_at, metadata)
                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,now(),$16)
                   ON CONFLICT (loop_id) DO UPDATE SET
                       status = EXCLUDED.status,
                       phase = EXCLUDED.phase,
                       current_iteration = EXCLUDED.current_iteration,
                       plan = EXCLUDED.plan,
                       total_hypotheses_tested = EXCLUDED.total_hypotheses_tested,
                       total_experiments_run = EXCLUDED.total_experiments_run,
                       total_findings = EXCLUDED.total_findings,
                       success_rate = EXCLUDED.success_rate,
                       completed_at = EXCLUDED.completed_at,
                       updated_at = now(),
                       metadata = EXCLUDED.metadata""",
                loop.loop_id,
                loop.project_id,
                loop.research_goal,
                loop.status.value,
                loop.phase.value,
                loop.max_iterations,
                loop.current_iteration,
                json.dumps(loop.plan.to_dict() if loop.plan else None),
                loop.total_hypotheses_tested,
                loop.total_experiments_run,
                loop.total_findings,
                loop.success_rate,
                loop.created_at,
                loop.started_at or None,
                loop.completed_at or None,
                json.dumps(loop.metadata),
            )
            await conn.execute(
                "DELETE FROM research_loop_iterations WHERE loop_id = $1", loop.loop_id
            )
            for it in loop.iterations:
                await conn.execute(
                    """INSERT INTO research_loop_iterations
                       (iteration_id, loop_id, iteration_number, phase,
                        started_at, completed_at, hypothesis, experiment_id,
                        results, analysis, evaluation, findings, status)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)""",
                    it.iteration_id,
                    it.loop_id,
                    it.iteration_number,
                    it.phase.value,
                    it.started_at or None,
                    it.completed_at or None,
                    it.hypothesis,
                    it.experiment_id,
                    json.dumps(it.results),
                    it.analysis,
                    it.evaluation,
                    json.dumps(it.findings),
                    it.status,
                )

    async def _get_pg(self, loop_id: str) -> ResearchLoop | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM research_loops WHERE loop_id = $1", loop_id
            )
            if row is None:
                return None
            loop = _row_to_loop(row)
            it_rows = await conn.fetch(
                "SELECT * FROM research_loop_iterations WHERE loop_id = $1 ORDER BY iteration_number",
                loop_id,
            )
            loop.iterations = [_row_to_iteration(r) for r in it_rows]
            return loop

    async def _list_all_pg(self) -> list[ResearchLoop]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT loop_id FROM research_loops ORDER BY updated_at DESC"
            )
        out = []
        for r in rows:
            loop = await self._get_pg(r["loop_id"])
            if loop is not None:
                out.append(loop)
        return out

    async def _delete_pg(self, loop_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM research_loop_iterations WHERE loop_id = $1", loop_id
            )
            await conn.execute("DELETE FROM research_loops WHERE loop_id = $1", loop_id)

    # -- sync facade (engine is sync) -------------------------------------- #

    def _run(self, coro: Any) -> Any:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're inside a running event loop (server context), schedule as task
            # and wait for it via loop thread state — sink to dedicated thread.
            raise RuntimeError(
                "LoopStore sync facade cannot be used inside a running event loop. "
                "Use the _pg async methods from async callers instead."
            )
        return loop.run_until_complete(coro)

    # -- public interface --------------------------------------------------- #

    def save(self, loop: ResearchLoop) -> None:
        if self._pool is None:
            self._memory[loop.loop_id] = loop
        else:
            self._run(self._save_pg(loop))

    def get(self, loop_id: str) -> ResearchLoop | None:
        if self._pool is None:
            return self._memory.get(loop_id)
        return self._run(self._get_pg(loop_id))

    def list_all(self) -> list[ResearchLoop]:
        if self._pool is None:
            return list(self._memory.values())
        return self._run(self._list_all_pg())

    def delete(self, loop_id: str) -> None:
        if self._pool is None:
            self._memory.pop(loop_id, None)
        else:
            self._run(self._delete_pg(loop_id))

    def clear(self) -> None:
        """Test utility: memory wipe, or full-table wipe when pool-backed."""
        if self._pool is None:
            self._memory.clear()
        else:
            self._run(self._clear_all_pg())

    async def _clear_all_pg(self) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM research_loop_iterations")
            await conn.execute("DELETE FROM research_loops")

    @classmethod
    def memory(cls) -> LoopStore:
        """In-memory-only store (back-compat with existing dict behavior)."""
        return cls(pool=None)


__all__ = ["LoopStore"]
