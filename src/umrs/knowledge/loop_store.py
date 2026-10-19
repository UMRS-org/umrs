"""Durable research-loop store backed by Postgres (migrations 0058 + 0082).

Drop-in replacement for the in-memory ``_loop_store`` in ``umrs.api.loop``.
Falls back to a pure in-memory dict when no pool is provided so unit tests and
development setups keep working identically.

API shape (sync-friendly; the engine is sync):
    store = LoopStore(pool=asyncpg_pool)            # pool-backed
    store = LoopStore.memory()                       # in-memory fallback

    store.save(loop)            # upsert loop + iterations
    store.save_keeping_control(loop)  # runner save: never undoes a pause/stop
    store.get(loop_id)          # ResearchLoop | None
    store.list_all()            # list[ResearchLoop]
    store.delete(loop_id)
    store.clear()               # only meaningful for memory mode; tests use it
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
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
    # Plan restores the retrieval anchors / hypotheses / experiments a real-mode
    # engine reads every checkpoint cycle; previously dropped on rehydration,
    # which made anchored real-mode loops fail the evidence guard (live
    # ask-probe leg, 2026-09-10).
    raw_plan = row.get("plan")
    if raw_plan:
        from .planner import ResearchPlan

        loop.plan = ResearchPlan.from_dict(
            raw_plan if not isinstance(raw_plan, str) else json.loads(raw_plan)
        )
    return loop


# A pause or stop the REST routes stored while a runner held its own copy.
_CONTROL_STATUSES = (LoopStatus.PAUSED, LoopStatus.STOPPED)


def progress_kept(loop: ResearchLoop, stored: ResearchLoop | None) -> ResearchLoop:
    """The record a control-route save writes, given the stored loop (H-1).

    Route writes (stop/pause/resume) change only ``status`` and
    ``completed_at``; everything else — the phase, the iteration counter,
    the plan, the counters, the metadata a live runner refreshes — comes
    from the stored row, so a concurrent runner's progress is never
    clobbered by the route's stale read (the mirror of ``control_kept``,
    which the runner's lane uses).
    """
    if stored is None:
        return loop
    merged = copy.copy(stored)  # shallow: same iterations and metadata
    merged.status = loop.status
    if loop.completed_at:
        merged.completed_at = loop.completed_at
    return merged


def control_kept(
    loop: ResearchLoop, stored_status: Any, stored_completed_at: Any
) -> ResearchLoop:
    """The record a runner save writes, given the stored status (H-1, 2026-10-18).

    While the stored status is PAUSED or STOPPED and the runner's copy still
    says RUNNING, the copy is written with the stored status (and a stop's
    completed_at); the runner's own object is not changed. Otherwise (no
    record, a resume already stored RUNNING, or a terminal status the runner
    reached itself) the copy is written as is.
    """
    if loop.status != LoopStatus.RUNNING or stored_status not in _CONTROL_STATUSES:
        return loop
    held = copy.copy(loop)  # shallow: same iterations and metadata
    held.status = LoopStatus(stored_status)
    if held.status == LoopStatus.STOPPED and stored_completed_at:
        held.completed_at = stored_completed_at
    return held


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


def _findings_json(findings: list[Any]) -> list[Any]:
    """Findings are ``Finding`` dataclasses (engine) or dicts (reloaded rows)."""
    return [f.to_dict() if hasattr(f, "to_dict") else f for f in findings]


def _row_json(row: Any, key: str, default: str) -> str:
    """JSON column text, tolerant of rows predating the column (0082)."""
    try:
        value = row[key]
    except (KeyError, IndexError):
        return default
    return value or default


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
        artifact_ids=list(json.loads(_row_json(row, "artifact_ids", "[]"))),
    )


logger = logging.getLogger(__name__)


def _is_connection_loss(exc: Exception) -> bool:
    """psycopg2 InterfaceError / OperationalError = the session is gone."""
    try:
        import psycopg2
    except ImportError:  # pragma: no cover - psycopg2 is a server dependency
        return False
    return isinstance(exc, (psycopg2.InterfaceError, psycopg2.OperationalError))


class LoopStore:
    """Postgres-backed loop store with in-memory fallback mode.

    Two pool modes:
    - asyncpg pool (``pool``) — needs an event loop via the sync facade.
    - psycopg2 connection (``sync_pool``) — fully synchronous; safe from any
      thread (runner/driver threads included; asyncpg pools are not
      loop-portable and crash outside their creating event loop).
    """

    def __init__(
        self,
        pool: Any | None,
        sync_pool: Any | None = None,
        reconnect: Any | None = None,
    ) -> None:
        self._pool = pool
        self._sync_pool = sync_pool
        # Self-healing sync connection (live rccs 2026-09-24: reads left the
        # session idle-in-transaction, the server's idle_in_transaction
        # timeout killed it, and every call failed "connection already
        # closed" until a restart). ``reconnect()`` returns a fresh conn.
        self._reconnect = reconnect
        self._sync_lock = threading.RLock()
        self._memory: dict[str, ResearchLoop] = {}

    @property
    def is_memory(self) -> bool:
        return self._pool is None and self._sync_pool is None

    @classmethod
    def sync_pg(cls, conn: Any, reconnect: Any | None = None) -> LoopStore:
        """Sync psycopg2-backed store (thread-safe, no event loop required);
        ``reconnect`` (optional) rebuilds the connection when it is dead."""
        return cls(pool=None, sync_pool=conn, reconnect=reconnect)

    def _sync_call(self, fn: Any, *args: Any) -> Any:
        """Run one sync op: serialized, on a live connection, with its read
        transaction ended afterwards; one retry after a dropped connection."""
        with self._sync_lock:
            self._ensure_live()
            try:
                return self._run_and_end(fn, *args)
            except Exception as exc:
                if self._reconnect is None or not _is_connection_loss(exc):
                    raise
                logger.warning("loop store connection lost (%s); reconnecting", exc)
                self._sync_pool = self._reconnect()
                return self._run_and_end(fn, *args)

    def _ensure_live(self) -> None:
        if self._reconnect is not None and getattr(self._sync_pool, "closed", 0):
            self._sync_pool = self._reconnect()

    def _run_and_end(self, fn: Any, *args: Any) -> Any:
        try:
            return fn(*args)
        finally:
            # A SELECT opens a transaction; never leave it idle (writes have
            # already committed, so this is a no-op for them).
            rollback = getattr(self._sync_pool, "rollback", None)
            if rollback is not None and not getattr(self._sync_pool, "closed", 0):
                rollback()

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
                        results, analysis, evaluation, findings, status,
                        artifact_ids)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)""",
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
                    json.dumps(_findings_json(it.findings)),
                    it.status,
                    json.dumps(list(it.artifact_ids)),
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

    # -- sync psycopg2 ops (runner/driver threads) ------------------------ #

    @staticmethod
    def _locked_control(cur: Any, loop: ResearchLoop) -> ResearchLoop:
        """Lock the loop row for this transaction and apply control_kept."""
        cur.execute(
            "SELECT status, completed_at FROM research_loops WHERE loop_id = %s FOR UPDATE",
            (loop.loop_id,),
        )
        row = cur.fetchone()
        return control_kept(loop, row[0], row[1]) if row else loop

    def _locked_progress(self, cur: Any, loop: ResearchLoop) -> ResearchLoop:
        """Lock the loop row for the merge and merge against the stored record
        (full row mapping via the store's own getter, like control side does:
        a route saves only status/completed_at; progress fields stay stored)."""
        cur.execute(
            "SELECT loop_id FROM research_loops WHERE loop_id = %s FOR UPDATE",
            (loop.loop_id,),
        )
        if not cur.fetchone():
            return loop
        return progress_kept(loop, self._get_sync(loop.loop_id))

    def _save_sync(
        self,
        loop: ResearchLoop,
        keep_control: bool = False,
        keep_progress: bool = False,
    ) -> None:
        from psycopg2.extras import Json

        with self._sync_pool.cursor() as cur:
            if keep_control:
                loop = self._locked_control(cur, loop)
            if keep_progress:
                loop = self._locked_progress(cur, loop)
            cur.execute(
                """INSERT INTO research_loops
                   (loop_id, project_id, research_goal, status, phase,
                    max_iterations, current_iteration, plan,
                    total_hypotheses_tested, total_experiments_run,
                    total_findings, success_rate,
                    created_at, started_at, completed_at, updated_at, metadata)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),%s)
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
                (
                    loop.loop_id,
                    loop.project_id,
                    loop.research_goal,
                    loop.status.value,
                    loop.phase.value,
                    loop.max_iterations,
                    loop.current_iteration,
                    Json(loop.plan.to_dict() if loop.plan else None),
                    loop.total_hypotheses_tested,
                    loop.total_experiments_run,
                    loop.total_findings,
                    loop.success_rate,
                    loop.created_at,
                    loop.started_at or None,
                    loop.completed_at or None,
                    Json(loop.metadata),
                ),
            )
            cur.execute(
                "DELETE FROM research_loop_iterations WHERE loop_id = %s",
                (loop.loop_id,),
            )
            for it in loop.iterations:
                cur.execute(
                    """INSERT INTO research_loop_iterations
                       (iteration_id, loop_id, iteration_number, phase,
                        started_at, completed_at, hypothesis, experiment_id,
                        results, analysis, evaluation, findings, status,
                        artifact_ids)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        it.iteration_id,
                        it.loop_id,
                        it.iteration_number,
                        it.phase.value,
                        it.started_at or None,
                        it.completed_at or None,
                        it.hypothesis,
                        it.experiment_id,
                        Json(it.results),
                        it.analysis,
                        it.evaluation,
                        Json(_findings_json(it.findings)),
                        it.status,
                        Json(list(it.artifact_ids)),
                    ),
                )
        self._sync_pool.commit()

    def _get_sync(self, loop_id: str) -> ResearchLoop | None:
        from psycopg2.extras import RealDictCursor

        with self._sync_pool.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM research_loops WHERE loop_id = %s",
                (loop_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            loop = _row_to_loop(self._jsonify(row))
            cur.execute(
                "SELECT * FROM research_loop_iterations WHERE loop_id = %s"
                " ORDER BY iteration_number",
                (loop_id,),
            )
            loop.iterations = [
                _row_to_iteration(self._jsonify(r)) for r in cur.fetchall()
            ]
            return loop

    @staticmethod
    def _jsonify(row: Any) -> Any:
        """psycopg2 RealDictCursor jsonb fields arrive already parsed."""
        for key in ("plan", "metadata", "results", "findings", "artifact_ids"):
            if key in row and isinstance(row[key], (dict, list)):
                row[key] = json.dumps(row[key])
        return row

    def _list_all_sync(self) -> list[ResearchLoop]:
        from psycopg2.extras import RealDictCursor

        with self._sync_pool.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT loop_id FROM research_loops ORDER BY updated_at DESC")
            ids = [r["loop_id"] for r in cur.fetchall()]
        out = []
        for loop_id in ids:
            loop = self._get_sync(loop_id)
            if loop is not None:
                out.append(loop)
        return out

    def _delete_sync(self, loop_id: str) -> None:
        with self._sync_pool.cursor() as cur:
            cur.execute(
                "DELETE FROM research_loop_iterations WHERE loop_id = %s",
                (loop_id,),
            )
            cur.execute("DELETE FROM research_loops WHERE loop_id = %s", (loop_id,))
        self._sync_pool.commit()

    def _clear_sync(self) -> None:
        with self._sync_pool.cursor() as cur:
            cur.execute("DELETE FROM research_loop_iterations")
            cur.execute("DELETE FROM research_loops")
        self._sync_pool.commit()

    # -- public sync facades ------------------------------------------------ #

    def save(self, loop: ResearchLoop) -> None:
        if self._sync_pool is not None:
            self._sync_call(self._save_sync, loop)
            return
        if self._pool is None:
            self._memory[loop.loop_id] = loop
        else:
            self._run(self._save_pg(loop))

    def save_preserving_progress(self, loop: ResearchLoop) -> None:
        """A control route's save: never clobbers a concurrent runner's progress.

        Mirror of ``save_keeping_control`` (the runner's lane). The route's
        status (and ``completed_at`` when set) lands; the phase, iteration,
        plan, counters and metadata come from the stored record — stop, pause
        and resume land their transition without erasing mid-turn progress on
        the psycopg2 store (merge under SELECT ... FOR UPDATE) or in memory
        (merge under the store lock).
        """
        if self._sync_pool is not None:
            self._sync_call(self._save_sync, loop, False, True)
            return
        with self._sync_lock:
            self.save(progress_kept(loop, self.get(loop.loop_id)))

    def save_keeping_control(self, loop: ResearchLoop) -> None:
        """A runner's save: never undoes a concurrent pause or stop.

        Lost update (ledger 2026-10-18 (8)): a runner saving its RUNNING copy
        after an iteration overwrote a PAUSED or STOPPED status the REST routes
        stored meanwhile. On the psycopg2 store (what runner threads use) the
        stored status is read with SELECT ... FOR UPDATE and the upsert runs in
        the same transaction under the store lock, so this save cannot write
        over a resume stored after its read. The memory and asyncpg modes re-read and
        save under the store lock only (memory hands out the same object; the
        asyncpg facade cannot run in runner threads). REST routes keep save().
        """
        if self._sync_pool is not None:
            self._sync_call(self._save_sync, loop, True)
            return
        with self._sync_lock:
            stored = self.get(loop.loop_id)
            self.save(
                control_kept(
                    loop,
                    getattr(stored, "status", None),
                    getattr(stored, "completed_at", ""),
                )
            )

    def get(self, loop_id: str) -> ResearchLoop | None:
        if self._sync_pool is not None:
            return self._sync_call(self._get_sync, loop_id)
        if self._pool is None:
            return self._memory.get(loop_id)
        return self._run(self._get_pg(loop_id))

    def list_all(self) -> list[ResearchLoop]:
        if self._sync_pool is not None:
            return self._sync_call(self._list_all_sync)
        if self._pool is None:
            return list(self._memory.values())
        return self._run(self._list_all_pg())

    def delete(self, loop_id: str) -> None:
        if self._sync_pool is not None:
            self._sync_call(self._delete_sync, loop_id)
            return
        if self._pool is None:
            self._memory.pop(loop_id, None)
        else:
            self._run(self._delete_pg(loop_id))

    def clear(self) -> None:
        """Test utility: memory wipe, or full-table wipe when pool-backed."""
        if self._sync_pool is not None:
            self._sync_call(self._clear_sync)
            return
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


__all__ = ["LoopStore", "control_kept"]
