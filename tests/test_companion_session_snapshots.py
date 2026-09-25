"""Route tests for companion session snapshots (ADR-036 F2 + venue registry 0084)."""

from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.companion_sessions import router


def _venue_state_match(
    row: dict,
    archived_at: str | None,
    venue_filter: list[str] | None,
    state_filter: list[str] | None,
) -> tuple[str, str] | None:
    """Venue/state csv predicate; returns (venue, computed) or None."""
    venue = row.get("venue", "local")
    if venue_filter is not None and venue not in venue_filter:
        return None
    computed = "archived" if archived_at else row.get("state", "")
    if state_filter is not None and computed not in state_filter:
        return None
    return venue, computed


def _row_matches(
    row: dict,
    sid: str,
    *,
    archived_at: str | None,
    include_archived: bool,
    venue_filter: list[str] | None,
    state_filter: list[str] | None,
    project_filter: str | None,
) -> tuple[str, str] | None:
    """One-row list predicate for the fake cursor; returns (venue, computed)."""
    if archived_at and not include_archived:
        return None
    matched = _venue_state_match(row, archived_at, venue_filter, state_filter)
    if matched is None:
        return None
    if project_filter is not None and not _payload_project_matches(row, project_filter):
        return None
    return matched


def _payload_project_matches(row: dict, project_filter: str) -> bool:
    """Fake of the payload::jsonb metadata project lookup."""
    try:
        meta = json.loads(row["payload"]).get("metadata") or {}
    except json.JSONDecodeError:
        return False
    project = (meta.get("config") or {}).get("project_id") or meta.get("project_id")
    return project == project_filter


def _parse_list_params(sql_l: str, rest: list) -> tuple:
    """Split tail params by clause text; both filter params are str lists."""
    venue_filter = state_filter = None
    project_filter = None
    if "venue = any" in sql_l and rest and isinstance(rest[0], list):
        venue_filter = rest.pop(0)
    if "computed_state = any" in sql_l and rest and isinstance(rest[0], list):
        state_filter = rest.pop(0)
    if rest and isinstance(rest[0], str):
        project_filter = rest.pop(0)
    include_archived = "archived_at is null" not in sql_l
    return venue_filter, state_filter, project_filter, include_archived


class _FakeCursor:
    def __init__(self, table, archives, recorded):
        self._table = table
        self._archives = archives
        self._recorded = recorded
        self._result = []
        self._row = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    # -- insert (upsert) ----------------------------------------------------
    def _exec_insert(self, params):
        uid, sid, name, payload, marker, venue, state = params
        venue = venue or "local"
        existing = self._table.get((uid, sid))
        if existing is None or existing["update_marker"] <= marker:
            previous = existing or {}
            self._table[(uid, sid)] = {
                "name": name,
                "payload": payload,
                "update_marker": marker,
                "venue": venue if venue else previous.get("venue", "local"),
                "state": state if state else previous.get("state", ""),
            }
            self.rowcount = 1
        else:
            self.rowcount = 0

    # -- detail select ------------------------------------------------------
    def _exec_get_payload(self, params, *, owner_first: bool = False):
        uid, sid = params if owner_first else params[::-1]
        row = self._table.get((uid, sid))
        self._row = (row["payload"],) if row else None

    def _exec_get(self, params):
        uid, sid = params
        row = self._table.get((uid, sid))
        if row is None:
            self._row = None
        else:
            archived = sid in self._archives
            self._row = (
                sid,
                row["name"],
                row["payload"],
                row["update_marker"],
                row.get("venue", "local"),
                row.get("state", ""),
                "ts" if archived else None,
            )

    # -- listing select -----------------------------------------------------
    def _exec_list(self, sql_l: str, params):
        uid = params[0]
        rest = list(params[1:])
        limit = rest.pop()  # limit is always last
        venue_filter, state_filter, project_filter, include_archived = _parse_list_params(
            sql_l, rest
        )
        rows = []
        for (row_uid, sid), row in self._table.items():
            if row_uid != uid:
                continue
            archived_at = "ts" if sid in self._archives else None
            matched = _row_matches(
                row,
                sid,
                archived_at=archived_at,
                include_archived=include_archived,
                venue_filter=venue_filter,
                state_filter=state_filter,
                project_filter=project_filter,
            )
            if matched is not None:
                venue, computed = matched
                rows.append((sid, row["name"], row["update_marker"], venue, computed, archived_at))
        rows.sort(key=lambda r: r[2], reverse=True)
        self._result = rows[:limit]

    # -- deletes (WP-04 G-P6-1 round 2) --------------------------------------
    def _exec_delete_archive(self, params):
        sid = params[0]
        if sid in self._archives:
            self._archives.discard(sid)
            self.rowcount = 1
        else:
            self.rowcount = 0

    def _exec_delete_snapshot(self, params):
        uid, sid = params
        if (uid, sid) in self._table:
            del self._table[(uid, sid)]
            self.rowcount = 1
        else:
            self.rowcount = 0

    def execute(self, sql, params):
        sql_l = sql.lower().strip()
        self._recorded.append((sql, params))
        if sql_l.startswith("insert"):
            self._exec_insert(params)
            return
        if sql_l.startswith("select s.session_id"):
            self._exec_get(params)
            return
        if sql_l.startswith("select payload from companion_session_snapshots"):
            # the upsert's row lock and the queue mutation bind (user, session)
            self._exec_get_payload(params, owner_first="for update" in sql_l)
            return
        if sql_l.startswith("select session_id, name, update_marker, venue"):
            self._exec_list(sql_l, params)
            return
        if sql_l.startswith("delete from companion_session_archives"):
            self._exec_delete_archive(params)
            return
        if sql_l.startswith("delete from chat_turn_approvals"):
            # F15 audit-block row (2026-09-21): cascade delete including approvals,
            # recorded in run flows; always merge; never raises.
            return
        if sql_l.startswith("delete from companion_session_snapshots"):
            self._exec_delete_snapshot(params)
            return
        raise AssertionError(f"unexpected SQL: {sql_l[:80]}")

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._result


class _FakeConn:
    def __init__(self, table, archives, recorded):
        self._cursor = _FakeCursor(table, archives, recorded)
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.committed = False


class _FakePool:
    def __init__(self):
        self.table: dict = {}
        self.archives: set = set()
        self.recorded: list = []
        self.conn = _FakeConn(self.table, self.archives, self.recorded)

    def getconn(self):
        return self.conn

    def putconn(self, _conn):
        return None


def _client(find_pool: _FakePool) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = find_pool
    return TestClient(app)


def _post(client, pool, uid="alice", sid="s1", **over):
    body = {"name": "demo", "updated_at": "2026-09-09T10:00:00Z", "payload": "{}"}
    body.update(over)
    return client.post(
        f"/umrs/companion/sessions/{sid}/snapshot", json=body, headers={"x-user-id": uid}
    )


class TestSnapshotUpsert:
    def test_insert_then_stale_marker_not_applied(self):
        pool = _FakePool()
        client = _client(pool)
        r1 = _post(client, pool, sid="sess-1")
        assert r1.status_code == 200
        assert r1.json()["applied"] is True
        r2 = _post(client, pool, sid="sess-1", updated_at="2026-09-09T09:00:00Z")
        assert r2.json()["applied"] is False

    def test_missing_identity_header_is_401(self):
        client = _client(_FakePool())
        resp = client.post(
            "/umrs/companion/sessions/s-1/snapshot",
            json={"name": "x", "updated_at": "1", "payload": "{}"},
        )
        assert resp.status_code == 401

    def test_invalid_venue_body_is_400(self):
        pool = _FakePool()
        client = _client(pool)
        resp = _post(client, pool, venue="moonbase")
        assert resp.status_code == 400

    def test_archived_is_not_storable_state(self):
        pool = _FakePool()
        client = _client(pool)
        resp = _post(client, pool, state="archived")
        assert resp.status_code == 400
        assert "archived" in resp.json()["detail"]


class TestSnapshotGet:
    def test_owner_reads_own_row(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="sess-1", payload='{"a": 1}')
        resp = client.get("/umrs/companion/sessions/sess-1", headers={"x-user-id": "alice"})
        assert resp.status_code == 200
        assert resp.json()["payload"] == '{"a": 1}'

    def test_get_reports_venue_and_state(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="sess-9", venue="frontend", state="stopped")
        resp = client.get("/umrs/companion/sessions/sess-9", headers={"x-user-id": "alice"})
        assert resp.json()["venue"] == "frontend"
        assert resp.json()["state"] == "stopped"
        assert resp.json()["archived_at"] == ""

    def test_get_computes_archived_state(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="sess-a", state="stopped")
        pool.archives.add("sess-a")
        resp = client.get("/umrs/companion/sessions/sess-a", headers={"x-user-id": "alice"})
        assert resp.json()["state"] == "archived"
        assert resp.json()["archived_at"]

    def test_other_user_gets_404(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="sess-1")
        resp = client.get("/umrs/companion/sessions/sess-1", headers={"x-user-id": "bob"})
        assert resp.status_code == 404


class TestSnapshotDelete:
    def test_owner_delete_drop_tombstone_and_row(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="sess-d")
        pool.archives.add("sess-d")
        resp = client.delete("/umrs/companion/sessions/sess-d", headers={"x-user-id": "alice"})
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert ("alice", "sess-d") not in pool.table
        assert "sess-d" not in pool.archives
        assert pool.conn.committed is True

    def test_other_user_delete_is_404_and_leaves_row(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="sess-1")
        resp = client.delete("/umrs/companion/sessions/sess-1", headers={"x-user-id": "bob"})
        assert resp.status_code == 404
        assert ("alice", "sess-1") in pool.table

    def test_absent_delete_is_404(self):
        pool = _FakePool()
        client = _client(pool)
        resp = client.delete("/umrs/companion/sessions/sess-ghost", headers={"x-user-id": "alice"})
        assert resp.status_code == 404


class TestSnapshotList:
    def test_lists_only_my_summary_rows(self):
        pool = _FakePool()
        client = _client(pool)
        for uid in ("alice", "bob"):
            _post(client, pool, uid=uid, sid="s1")
        resp = client.get("/umrs/companion/sessions", headers={"x-user-id": "alice"})
        assert resp.status_code == 200
        rows = resp.json()["snapshots"]
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"

    def test_venue_filter(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="local-1", venue="local")
        _post(client, pool, sid="front-1", venue="frontend")
        resp = client.get("/umrs/companion/sessions?venue=frontend", headers={"x-user-id": "alice"})
        rows = resp.json()["snapshots"]
        assert [r["session_id"] for r in rows] == ["front-1"]
        assert rows[0]["venue"] == "frontend"

    def test_state_filter_excludes_archived_by_default(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="open-1", state="ongoing")
        _post(client, pool, sid="dead-1", state="stopped")
        _post(client, pool, sid="gone-1", state="stopped")
        pool.archives.add("gone-1")
        resp = client.get("/umrs/companion/sessions?state=stopped", headers={"x-user-id": "alice"})
        assert [r["session_id"] for r in resp.json()["snapshots"]] == ["dead-1"]

    def test_state_archived_with_include_archived(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="gone-1", state="stopped")
        pool.archives.add("gone-1")
        resp = client.get(
            "/umrs/companion/sessions?state=archived&include_archived=true",
            headers={"x-user-id": "alice"},
        )
        rows = resp.json()["snapshots"]
        assert [r["session_id"] for r in rows] == ["gone-1"]
        assert rows[0]["state"] == "archived"
        assert rows[0]["archived_at"]

    def test_project_filter_reads_payload_metadata(self):
        pool = _FakePool()
        client = _client(pool)
        payload_a = '{"metadata": {"config": {"project_id": "proj-a"}}}'
        _post(client, pool, sid="pa", payload=payload_a)
        _post(client, pool, sid="pb", payload='{"metadata": {"config": {"project_id": "proj-b"}}}')
        resp = client.get("/umrs/companion/sessions?project=proj-a", headers={"x-user-id": "alice"})
        assert [r["session_id"] for r in resp.json()["snapshots"]] == ["pa"]

    def test_invalid_state_filter_is_400(self):
        pool = _FakePool()
        client = _client(pool)
        resp = client.get("/umrs/companion/sessions?state=doomed", headers={"x-user-id": "alice"})
        assert resp.status_code == 400

    def test_invalid_venue_filter_is_400(self):
        pool = _FakePool()
        client = _client(pool)
        resp = client.get("/umrs/companion/sessions?venue=nowhere", headers={"x-user-id": "alice"})
        assert resp.status_code == 400


class TestDegradation:
    def test_no_pool_is_503(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        resp = client.post(
            "/umrs/companion/sessions/s1/snapshot",
            json={"name": "x", "updated_at": "1", "payload": "{}"},
            headers={"x-user-id": "alice"},
        )
        assert resp.status_code == 503


# ── Turn accessor (ADR-057 D-2) ────────────────────────────────────────────

_PAYLOAD_TWO_TURNS = (
    '{"messages": ['
    '{"role": "user", "content": "hi", "timestamp": "t1", "metadata": {"turn_id": "t1"}},'
    '{"role": "assistant", "content": "hello", "timestamp": "t2", "metadata": {"turn_id": "t1"}},'
    '{"role": "user", "content": "next", "timestamp": "t3", "metadata": {"turn_id": "t2"}},'
    '{"role": "assistant", "content": "answer", "timestamp": "t4", "metadata": {"turn_id": "t2"}}'
    "]}"
)


class TestTurnAccessor:
    def test_list_turns_projects_ordered_entries(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="turns-1", payload=_PAYLOAD_TWO_TURNS)
        resp = client.get("/umrs/companion/sessions/turns-1/turns", headers={"x-user-id": "alice"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 4
        assert [e["turn_id"] for e in body["entries"]] == ["t1", "t1", "t2", "t2"]
        assert body["entries"][0]["role"] == "user"

    def test_get_turn_by_id(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="turns-2", payload=_PAYLOAD_TWO_TURNS)
        resp = client.get(
            "/umrs/companion/sessions/turns-2/turns/t2", headers={"x-user-id": "alice"}
        )
        assert resp.status_code == 200
        assert resp.json()["entry"]["turn_id"] == "t2"

    def test_get_unknown_turn_is_404(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="turns-3", payload=_PAYLOAD_TWO_TURNS)
        resp = client.get(
            "/umrs/companion/sessions/turns-3/turns/t99", headers={"x-user-id": "alice"}
        )
        assert resp.status_code == 404

    def test_old_snapshot_without_markers_is_honestly_empty(self):
        pool = _FakePool()
        client = _client(pool)
        # Legacy payloads carry no turn_id markers — no synthesis allowed (ADR-057).
        _post(
            client,
            pool,
            sid="turns-4",
            payload='{"messages": [{"role": "user", "content": "x"}]}',
        )
        resp = client.get("/umrs/companion/sessions/turns-4/turns", headers={"x-user-id": "alice"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 0
        assert resp.json()["entries"] == []

    def test_other_user_turns_is_404(self):
        pool = _FakePool()
        client = _client(pool)
        _post(client, pool, sid="turns-5", payload=_PAYLOAD_TWO_TURNS)
        resp = client.get("/umrs/companion/sessions/turns-5/turns", headers={"x-user-id": "bob"})
        assert resp.status_code == 404
