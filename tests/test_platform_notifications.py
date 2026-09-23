"""D-36 leg B pins: durable platform notifications (0109)."""

from __future__ import annotations

import json

from umrs.platform_notifications import NotificationStore, notify, set_emitter


class _Cursor:
    def __init__(self, conn):
        self._conn = conn
        self._result = []

    def execute(self, sql, params=None):
        self._conn._handle(sql, params or (), self)

    def fetchone(self):
        return self._result.pop(0) if self._result else None

    def fetchall(self):
        out, self._result = self._result[:], []
        return out

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Conn:
    def __init__(self):
        self.rows: list[dict] = []
        self._result: list = []
        self._seq = 0

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass

    def _handle(self, sql, params, cur):
        if "INSERT INTO platform_notifications" in sql:
            self._seq += 1
            self.rows.append(
                {
                    "id": self._seq,
                    "kind": params[0],
                    "title": params[1],
                    "detail": json.loads(params[2]),
                    "user_id": params[3],
                    "project_id": params[4],
                    "created_at": "2026-09-23T00:00:00+00:00",
                    "read_at": None,
                }
            )
            cur._result = [self._row_tuple(self.rows[-1])]
        elif "read_at IS NULL" in sql and "UPDATE" in sql:
            for r in self.rows:
                if r["id"] == params[0] and r["read_at"] is None:
                    r["read_at"] = "2026-09-23T01:00:00+00:00"
                    cur._result = [self._row_tuple(r)]
                    return
            cur._result = []
        elif "WHERE" in sql and "user_id = %s" in sql:
            want = params[0]
            cur._result = [self._row_tuple(r) for r in self.rows if r["user_id"] == want]
        elif sql.startswith("SELECT id, kind"):
            cur._result = [self._row_tuple(r) for r in self.rows]

    @staticmethod
    def _row_tuple(r):
        return (
            r["id"],
            r["kind"],
            r["title"],
            json.dumps(r["detail"]),
            r["user_id"],
            r["project_id"],
            r["created_at"],
            r["read_at"],
        )


def test_emit_and_list_round_trip():
    store = NotificationStore(_Conn())
    row = store.emit(
        kind="budget_exhausted", title="loop stopped", detail={"loop_id": "l1"}, project_id="p1"
    )
    assert row["kind"] == "budget_exhausted"
    listed = store.list(unread_only=True)
    assert len(listed) == 1
    assert listed[0]["detail"]["loop_id"] == "l1"


def test_mark_read_idempotent():
    store = NotificationStore(_Conn())
    row = store.emit(kind="k", title="t")
    first = store.mark_read(row["id"])
    assert first["read_at"] is not None
    assert store.mark_read(row["id"]) is None  # second read = no-op, None


def test_notify_unwired_reports_never_silent(caplog):
    set_emitter(None)
    assert notify(kind="k", title="t", project_id="p") is False
    assert any("unwired" in r.getMessage() for r in caplog.records)


def test_notify_wired_lands_row():
    conn = _Conn()

    class _Pool:
        def getconn(self):
            return conn

        def putconn(self, _c):
            pass

    set_emitter(_Pool())
    try:
        assert notify(kind="budget_exhausted", title="x", detail={"a": 1}, project_id="p") is True
        assert conn.rows[0]["kind"] == "budget_exhausted"
        assert conn.rows[0]["detail"] == {"a": 1}
    finally:
        set_emitter(None)


def test_notify_failure_never_blocks(caplog):
    class _BoomPool:
        def getconn(self):
            raise RuntimeError("pool down")

        def putconn(self, _c):
            pass

    set_emitter(_BoomPool())
    try:
        assert notify(kind="k", title="t") is False
        assert any("emit failed" in r.getMessage() for r in caplog.records)
    finally:
        set_emitter(None)


def test_budget_stop_emits_through_the_runner_seam(monkeypatch):
    """The engine runner's budget-stop path calls notify (D-36 leg B)."""
    from umrs.knowledge import engine_loop_runner as runner

    calls = []
    monkeypatch.setattr(
        "umrs.platform_notifications.notify",
        lambda **kw: calls.append(kw) or True,
    )
    loop = type(
        "L",
        (),
        {
            "loop_id": "l-1",
            "project_id": "p1",
            "current_iteration": 3,
            "metadata": {"budget": {"max_wall_clock_hours": 2}},
        },
    )()
    driver = object.__new__(runner.EngineLoopRunner)
    driver._loop = loop
    driver._notify_budget_stop()
    assert calls[0]["kind"] == "budget_exhausted"
    assert calls[0]["detail"]["loop_id"] == "l-1"
    assert calls[0]["detail"]["iterations"] == 3
    assert calls[0]["project_id"] == "p1"
