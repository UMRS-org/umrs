"""Loop runner lost update (ledger 2026-10-18 (8) finding, fixed 2026-10-18).

With a Postgres-backed loop store every reader gets its own copy. A
thread-hosted runner saved its RUNNING copy after an iteration and so
overwrote a PAUSED or STOPPED status the REST routes stored meanwhile; the
next poll read RUNNING and the loop ran on. The fake store below copies on
every read and write, like the Postgres store, and the "REST" pause/stop
lands in the middle of an iteration.

Review 2026-10-18: a re-read then a save could still write PAUSED over a
resume stored between the two calls. LoopStore.save_keeping_control now
locks the row (SELECT ... FOR UPDATE) and upserts in one transaction under
the store lock; runners use it whenever the store is a LoopStore.
"""

from __future__ import annotations

import copy
import threading
from typing import Any

from umrs.knowledge.agent_loop_runner import AgentLoopRunner
from umrs.knowledge.engine_loop_runner import EngineLoopRunner, save_keeping_control
from umrs.knowledge.loop import LoopIteration, LoopPhase, LoopStatus, ResearchLoop
from umrs.knowledge.loop_store import LoopStore

STOPPED_AT = "2026-10-18T00:00:00+00:00"


def _loop(max_iterations: int = 3) -> ResearchLoop:
    return ResearchLoop(
        loop_id="loop-lu",
        project_id="proj-1",
        research_goal="g",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.PLANNING,
        max_iterations=max_iterations,
    )


class _CopyStore:
    """Copy on read and write (the Postgres store's semantics)."""

    def __init__(self, loop: ResearchLoop | None = None):
        self.rows: dict[str, ResearchLoop] = {}
        self.resume_after_polls: int | None = None
        self.paused_polls = 0
        if loop is not None:
            self.save(loop)

    def save(self, loop: ResearchLoop) -> None:
        self.rows[loop.loop_id] = copy.deepcopy(loop)

    def get(self, loop_id: str) -> ResearchLoop | None:
        row = self.rows.get(loop_id)
        if row is not None and row.status == LoopStatus.PAUSED:
            self._count_paused_poll(row)
        return copy.deepcopy(row)

    def _count_paused_poll(self, row: ResearchLoop) -> None:
        self.paused_polls += 1
        if self.resume_after_polls is not None and self.paused_polls >= self.resume_after_polls:
            row.status = LoopStatus.RUNNING  # the REST resume

    # what the REST routes do: read a copy, set the status, save it
    def rest_pause(self, loop_id: str) -> None:
        self.rows[loop_id].status = LoopStatus.PAUSED

    def rest_stop(self, loop_id: str) -> None:
        self.rows[loop_id].status = LoopStatus.STOPPED
        self.rows[loop_id].completed_at = STOPPED_AT

    def status(self, loop_id: str) -> LoopStatus:
        return self.rows[loop_id].status


def _iteration(n: int) -> LoopIteration:
    return LoopIteration(iteration_id=f"it-{n}", loop_id="loop-lu", iteration_number=n)


# ── the save itself ─────────────────────────────────────────────────────────


def test_checkpoint_keeps_a_concurrent_pause():
    loop = _loop()
    store = _CopyStore(loop)
    runner = EngineLoopRunner(loop, engine_factory=lambda: None, store=store, poll_seconds=0)
    store.rest_pause(loop.loop_id)
    loop.current_iteration = 1
    runner._checkpoint(loop, _iteration(1))
    assert store.status(loop.loop_id) == LoopStatus.PAUSED
    assert store.rows[loop.loop_id].current_iteration == 1  # the progress is saved
    assert loop.status == LoopStatus.RUNNING  # the runner's own copy is untouched


def test_checkpoint_keeps_a_concurrent_stop_and_its_stamp():
    loop = _loop()
    store = _CopyStore(loop)
    runner = EngineLoopRunner(loop, engine_factory=lambda: None, store=store, poll_seconds=0)
    store.rest_stop(loop.loop_id)
    runner._checkpoint(loop, _iteration(1))
    assert store.status(loop.loop_id) == LoopStatus.STOPPED
    assert store.rows[loop.loop_id].completed_at == STOPPED_AT


def test_a_terminal_status_the_runner_reached_is_saved_as_is():
    loop = _loop()
    store = _CopyStore(loop)
    store.rest_pause(loop.loop_id)
    loop.status = LoopStatus.ERROR
    save_keeping_control(store, loop)
    assert store.status(loop.loop_id) == LoopStatus.ERROR


def test_a_resumed_loop_saves_running():
    loop = _loop()
    store = _CopyStore(loop)
    loop.current_iteration = 2
    save_keeping_control(store, loop)
    assert store.status(loop.loop_id) == LoopStatus.RUNNING
    assert store.rows[loop.loop_id].current_iteration == 2


def test_an_unreadable_record_saves_the_copy():
    class _NoRead(_CopyStore):
        def get(self, _loop_id):
            raise RuntimeError("db down")

    loop = _loop()
    store = _NoRead()
    save_keeping_control(store, loop)
    assert store.rows[loop.loop_id].status == LoopStatus.RUNNING


# ── engine runner: a stop during iteration 1 ends the loop ─────────────────


class _Engine:
    """Minimal engine: cancel probe, one iteration, checkpoint hook."""

    def __init__(self, during):
        self._during = during

    def run(self, loop: ResearchLoop) -> ResearchLoop:
        while loop.current_iteration < loop.max_iterations:
            if self._cancel_check():
                loop.status = LoopStatus.STOPPED
                return loop
            loop.current_iteration += 1
            self._during(loop.current_iteration)
            self._iteration_hook(_iteration(loop.current_iteration))
        loop.status = LoopStatus.STOPPED
        return loop


def test_engine_runner_honours_a_stop_stored_during_an_iteration():
    loop = _loop()
    store = _CopyStore(loop)

    def during(n: int) -> None:
        if n == 1:
            store.rest_stop(loop.loop_id)

    runner = EngineLoopRunner(
        loop, engine_factory=lambda: _Engine(during), store=store, poll_seconds=0
    )
    finished = runner.drive(loop)
    assert finished.current_iteration == 1  # old code: the save undid the stop, 3 ran
    assert store.status(loop.loop_id) == LoopStatus.STOPPED


def test_engine_runner_waits_out_a_pause_stored_during_an_iteration():
    loop = _loop(max_iterations=2)
    store = _CopyStore(loop)
    store.resume_after_polls = 3

    def during(n: int) -> None:
        if n == 1:
            store.rest_pause(loop.loop_id)

    runner = EngineLoopRunner(
        loop, engine_factory=lambda: _Engine(during), store=store, poll_seconds=0
    )
    finished = runner.drive(loop)
    assert store.paused_polls >= 3  # old code: the save undid the pause, never waited
    assert finished.current_iteration == 2


# ── agent runner: the same two cases ───────────────────────────────────────


def _executor(store: _CopyStore, action):
    def run(loop_id: str, iteration: int, _goal: str) -> dict:
        if iteration == 1:
            action(loop_id)
        return {"llm": True, "hypothesis": f"h{iteration}"}

    return run


def test_agent_runner_honours_a_stop_stored_during_an_iteration():
    loop = _loop()
    store = _CopyStore(loop)
    runner = AgentLoopRunner(
        loop, executor=_executor(store, store.rest_stop), store=store, poll_seconds=0
    )
    finished = runner.kick(loop)
    assert finished.current_iteration == 1  # old code: 3
    assert store.status(loop.loop_id) == LoopStatus.STOPPED


def test_agent_runner_waits_out_a_pause_stored_during_an_iteration():
    loop = _loop(max_iterations=2)
    store = _CopyStore(loop)
    store.resume_after_polls = 2
    runner = AgentLoopRunner(
        loop, executor=_executor(store, store.rest_pause), store=store, poll_seconds=0
    )
    finished = runner.kick(loop)
    assert store.paused_polls >= 2  # old code: 0
    assert finished.current_iteration == 2


# ── LoopStore: the row lock and the upsert in one transaction ─────────────


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self._last = ""

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self._last = sql
        self._conn.log.append(("execute", sql, params))
        if "FOR UPDATE" in sql and self._conn.during_lock is not None:
            self._conn.during_lock()

    def fetchone(self):
        return self._conn.row if "FOR UPDATE" in self._last else None

    def fetchall(self):
        return []


class _Conn:
    """psycopg2-shaped connection: the stored row is (status, completed_at)."""

    def __init__(self, row: tuple | None) -> None:
        self.row = row
        self.log: list[tuple] = []
        self.during_lock = None

    def cursor(self, cursor_factory=None):  # noqa: ARG002 — psycopg2 signature
        return _Cursor(self)

    def commit(self) -> None:
        self.log.append(("commit",))

    def rollback(self) -> None:
        self.log.append(("rollback",))

    def written_status(self) -> list[tuple[str, Any]]:
        """(status, completed_at) of every research_loops upsert, in order."""
        return [
            (e[2][3], e[2][14])
            for e in self.log
            if e[0] == "execute" and "INSERT INTO research_loops" in e[1]
        ]


def test_sync_store_keeps_a_stored_pause_under_the_row_lock():
    conn = _Conn(("paused", None))
    loop = _loop()
    LoopStore.sync_pg(conn).save_keeping_control(loop)
    kinds = [e[1].split()[0] if e[0] == "execute" else e[0] for e in conn.log]
    assert kinds[:2] == ["SELECT", "INSERT"]  # lock first, same transaction
    assert "FOR UPDATE" in conn.log[0][1]
    assert kinds.count("commit") == 1
    assert conn.written_status() == [("paused", None)]
    assert loop.status == LoopStatus.RUNNING


def test_sync_store_keeps_a_stored_stop_and_its_stamp():
    conn = _Conn(("stopped", STOPPED_AT))
    LoopStore.sync_pg(conn).save_keeping_control(_loop())
    assert conn.written_status() == [("stopped", STOPPED_AT)]


def test_sync_store_writes_running_over_a_stored_resume():
    conn = _Conn(("running", None))
    LoopStore.sync_pg(conn).save_keeping_control(_loop())
    assert conn.written_status() == [("running", None)]


def test_sync_store_inserts_a_missing_record_as_is():
    conn = _Conn(None)
    LoopStore.sync_pg(conn).save_keeping_control(_loop())
    assert conn.written_status() == [("running", None)]


def test_sync_store_writes_the_runners_own_terminal_status():
    conn = _Conn(("paused", None))
    loop = _loop()
    loop.status = LoopStatus.ERROR
    LoopStore.sync_pg(conn).save_keeping_control(loop)
    assert conn.written_status() == [("error", None)]


def test_plain_save_takes_no_lock_so_rest_routes_still_win():
    conn = _Conn(("paused", None))
    LoopStore.sync_pg(conn).save(_loop())  # what the resume route calls
    assert not any("FOR UPDATE" in e[1] for e in conn.log if e[0] == "execute")
    assert conn.written_status() == [("running", None)]


def test_a_resume_waits_for_the_runner_save_and_lands_last():
    """A REST save from another thread cannot slip between the locked read
    and the runner's upsert: it runs after the commit, so the resume stands."""
    conn = _Conn(("paused", None))
    store = LoopStore.sync_pg(conn)
    resumed = _loop()
    resumer = threading.Thread(target=store.save, args=(resumed,))

    def resume_now() -> None:
        conn.during_lock = None
        resumer.start()
        resumer.join(timeout=0.2)
        assert resumer.is_alive()  # blocked on the store lock

    conn.during_lock = resume_now
    store.save_keeping_control(_loop())
    resumer.join(timeout=5)
    assert not resumer.is_alive()
    assert conn.written_status() == [("paused", None), ("running", None)]
    first_commit = conn.log.index(("commit",))
    second_insert = max(
        i
        for i, e in enumerate(conn.log)
        if e[0] == "execute" and "INSERT INTO research_loops" in e[1]
    )
    assert first_commit < second_insert


def test_engine_runner_checkpoint_uses_the_locked_save():
    conn = _Conn(("paused", None))
    store = LoopStore.sync_pg(conn)
    loop = _loop()
    runner = EngineLoopRunner(loop, engine_factory=lambda: None, store=store, poll_seconds=0)
    runner._checkpoint(loop, _iteration(1))
    assert any("FOR UPDATE" in e[1] for e in conn.log if e[0] == "execute")
    assert conn.written_status() == [("paused", None)]


def test_agent_runner_uses_the_locked_save():
    conn = _Conn(("stopped", STOPPED_AT))
    store = LoopStore.sync_pg(conn)
    loop = _loop(max_iterations=1)
    runner = AgentLoopRunner(
        loop,
        executor=lambda _id, _n, _g: {"llm": True, "hypothesis": "h"},
        store=store,
        poll_seconds=0,
    )
    runner.kick(loop)
    assert any("FOR UPDATE" in e[1] for e in conn.log if e[0] == "execute")
    assert conn.written_status()[0] == ("stopped", STOPPED_AT)


def test_memory_store_keeps_a_pause_too():
    store = LoopStore.memory()
    stored = _loop()
    stored.status = LoopStatus.PAUSED
    store.save(stored)
    store.save_keeping_control(_loop())
    assert store.get("loop-lu").status == LoopStatus.PAUSED
