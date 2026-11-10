"""S5: server-bound remote child snapshots, events and owner inspection."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api._rail_identity import RailPrincipal
from umrs.api.companion_sessions import router as sessions_router
from umrs.api.companion_venue import rail_principal
from umrs.api.companion_venue import router as venue_router

ROOT = "root-session"
CHILD = "child-session"
PARENT_AGENT = "agent-parent"
AGENT = "agent-child"


def _record(
    session_id: str = CHILD,
    *,
    parent_session_id: str = ROOT,
    parent_agent_id: str = PARENT_AGENT,
    agent_id: str = AGENT,
    revision: int = 1,
    status: str = "running",
    venue: str = "frontend",
) -> dict[str, Any]:
    return {
        "id": session_id,
        "name": f"Subagent {agent_id}",
        "created_at": "2026-11-10T00:00:00+00:00",
        "updated_at": f"2026-11-10T00:00:0{revision}+00:00",
        "model": "child-model",
        "messages": [{"role": "user", "content": "actual child task"}],
        "metadata": {
            "session_kind": "subagent",
            "parent_session_id": parent_session_id,
            "parent_agent_id": parent_agent_id,
            "agent_id": agent_id,
            "task": "actual child task",
            "system_prompt": "actual child prompt",
            "declared_tools": ["file_read"],
            "execution_kind": "async",
            "subagent_status": status,
            "subagent_revision": revision,
            "started_at": "2026-11-10T00:00:00+00:00",
            "ended_at": "",
            "terminal_error": "",
            "follow_up_count": 0,
            "venue": venue,
            "event_log_error": "",
        },
    }


def _root_record() -> dict[str, Any]:
    return {
        "id": ROOT,
        "name": "root",
        "created_at": "2026-11-10T00:00:00+00:00",
        "updated_at": "2026-11-10T00:00:00+00:00",
        "model": "parent-model",
        "messages": [],
        "metadata": {"agent_id": PARENT_AGENT, "venue": "frontend"},
    }


def _metadata(payload: str) -> dict[str, Any]:
    raw = json.loads(payload).get("metadata") or {}
    return json.loads(raw) if isinstance(raw, str) else raw


class _Cursor:
    def __init__(self, pool: _Pool) -> None:
        self.pool = pool
        self.row: tuple[Any, ...] | None = None
        self.rows: list[tuple[Any, ...]] = []
        self.rowcount = 0

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_args: Any) -> bool:
        return False

    def execute(self, sql: str, params: tuple[Any, ...] | list[Any] = ()) -> None:
        normalized = " ".join(sql.lower().split())
        self.pool.recorded.append(normalized)
        self.row = None
        self.rows = []
        self.rowcount = 0
        if normalized.startswith("select s.session_id, s.name, s.payload"):
            self._detail(tuple(params))
        elif normalized.startswith("select venue, state, lease_id from companion_venue_sessions"):
            self._venue(tuple(params))
        elif normalized.startswith("select s.name, s.payload, s.venue, s.state"):
            self._bus_lock(tuple(params))
        elif normalized.startswith("select payload from companion_session_snapshots"):
            self._payload(normalized, tuple(params))
        elif normalized.startswith("insert into companion_session_snapshots"):
            self._insert(tuple(params))
        elif normalized.startswith("update companion_session_snapshots"):
            self._update(tuple(params))
        elif normalized.startswith("select session_id, name, update_marker, venue, computed_state"):
            self._list(normalized, tuple(params))
        else:
            raise AssertionError(f"unexpected SQL: {normalized[:120]}")

    def _detail(self, params: tuple[Any, ...]) -> None:
        uid, sid = params
        row = self.pool.table.get((uid, sid))
        if row is not None:
            self.row = (
                sid,
                row["name"],
                row["payload"],
                row["update_marker"],
                row["venue"],
                row["state"],
                None,
            )

    def _venue(self, params: tuple[Any, ...]) -> None:
        uid, sid = params
        row = self.pool.venues.get(sid)
        if row is not None and row["user_id"] == uid:
            self.row = (row["venue"], row["state"], row["lease_id"])

    def _bus_lock(self, params: tuple[Any, ...]) -> None:
        uid, sid = params
        row = self.pool.table.get((uid, sid))
        if row is not None:
            self.row = (row["name"], row["payload"], row["venue"], row["state"], None)

    def _payload(self, sql: str, params: tuple[Any, ...]) -> None:
        uid, sid = (params[1], params[0]) if "session_id = %s and user_id = %s" in sql else params
        row = self.pool.table.get((uid, sid))
        if row is not None:
            self.row = (row["payload"],)

    def _insert(self, params: tuple[Any, ...]) -> None:
        uid, sid, name, payload, marker, venue, state = params
        existing = self.pool.table.get((uid, sid))
        if existing is not None and existing["update_marker"] > marker:
            self.rowcount = 0
            return
        self.pool.table[(uid, sid)] = {
            "name": name,
            "payload": payload,
            "update_marker": marker,
            "venue": venue,
            "state": state,
        }
        self.rowcount = 1

    def _update(self, params: tuple[Any, ...]) -> None:
        if len(params) == 3:
            payload, uid, sid = params
            row = self.pool.table.get((uid, sid))
            if row is not None:
                row["payload"] = payload
                self.rowcount = 1
            return
        name, payload, marker, state, uid, sid = params
        row = self.pool.table.get((uid, sid))
        if row is not None:
            row.update(name=name, payload=payload, update_marker=marker, state=state)
            self.rowcount = 1

    def _list(self, sql: str, params: tuple[Any, ...]) -> None:
        uid = params[0]
        limit = params[-1]
        child_only = "session_kind = 'subagent'" in sql
        rows = []
        for (row_uid, sid), row in self.pool.table.items():
            if row_uid != uid:
                continue
            metadata = _metadata(row["payload"])
            is_child = metadata.get("session_kind") == "subagent"
            if is_child != child_only:
                continue
            if "parent_session_id = %s" in sql and metadata.get("parent_session_id") != params[1]:
                continue
            if "agent_id = %s" in sql and metadata.get("agent_id") != params[1]:
                continue
            rows.append((sid, row["name"], row["update_marker"], row["venue"], row["state"], None))
        rows.sort(key=lambda row: row[2], reverse=True)
        self.rows = rows[:limit]

    def fetchone(self) -> tuple[Any, ...] | None:
        return self.row

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self.rows


class _Connection:
    def __init__(self, pool: _Pool) -> None:
        self.pool = pool
        self.commits = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self.pool)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        return None


class _Pool:
    def __init__(self) -> None:
        self.table: dict[tuple[str, str], dict[str, Any]] = {}
        self.venues: dict[str, dict[str, Any]] = {}
        self.recorded: list[str] = []
        self.connection = _Connection(self)

    def getconn(self) -> _Connection:
        return self.connection

    def putconn(self, _connection: _Connection) -> None:
        return None


def _seed(pool: _Pool) -> None:
    root = _root_record()
    pool.table[("alice", ROOT)] = {
        "name": "root",
        "payload": json.dumps(root),
        "update_marker": root["updated_at"],
        "venue": "frontend",
        "state": "ongoing",
    }
    pool.venues[ROOT] = {
        "user_id": "alice",
        "venue": "frontend",
        "state": "running",
        "lease_id": "lease-1",
    }


def _client(pool: _Pool, *, lease_id: str = "lease-1") -> TestClient:
    app = FastAPI()
    app.include_router(sessions_router)
    app.include_router(venue_router)
    app.state.umrs_db_pool = pool
    principal = RailPrincipal("alice", ROOT, lease_id, object())
    app.dependency_overrides[rail_principal] = lambda: principal
    return TestClient(app)


def _put(client: TestClient, record: dict[str, Any], *, child: str = CHILD) -> Any:
    return client.put(
        f"/umrs/companion/sessions/{ROOT}/rail/subagents/{child}/snapshot",
        json={"payload": json.dumps(record)},
        headers={"Authorization": "Bearer lease-token"},
    )


def _event(source_event_id: int = 1, **changes: Any) -> dict[str, Any]:
    event = {
        "session_id": CHILD,
        "event_type": "tool_result",
        "timestamp": "2026-11-10T00:00:01+00:00",
        "message": "actual tool result",
        "data": {
            "child_session_id": CHILD,
            "parent_session_id": ROOT,
            "parent_agent_id": PARENT_AGENT,
            "agent_id": AGENT,
            "result": "actual output",
        },
        "tool_name": "file_read",
        "tool_status": "success",
        "tool_duration_ms": 2,
        "agent_id": AGENT,
        "parent_agent_id": PARENT_AGENT,
        "task_id": None,
        "progress_pct": None,
        "intent": None,
        "rationale": None,
    }
    event.update(changes)
    return event


def _append(client: TestClient, source_event_id: int = 1, **changes: Any) -> Any:
    return client.post(
        f"/umrs/companion/sessions/{ROOT}/rail/subagents/{CHILD}/events",
        json={"source_event_id": source_event_id, "event": _event(source_event_id, **changes)},
        headers={"Authorization": "Bearer lease-token"},
    )


def _inspect(client: TestClient, *, child: str = CHILD, uid: str = "alice", **params: Any) -> Any:
    return client.get(
        f"/umrs/companion/sessions/{child}/subagent",
        params=params,
        headers={"x-user-id": uid},
    )


def _remote_history() -> tuple[_Pool, TestClient]:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    for source_id in (4, 5, 6):
        assert _append(client, source_id).status_code == 201
    return pool, client


def test_lease_round_trip_and_bounded_owner_read() -> None:
    _, client = _remote_history()
    first = _inspect(client, limit=2)
    second = _inspect(client, after_id=first.json()["next_after_id"], limit=2)

    assert first.status_code == 200, first.text
    assert first.json()["record"]["messages"][0]["content"] == "actual child task"
    assert [row["id"] for row in first.json()["events"]] == [1, 2]
    assert first.json()["has_more"] is True
    assert [row["source_event_id"] for row in second.json()["events"]] == [6]
    assert second.json()["has_more"] is False


def test_remote_child_inventory_has_lineage_excludes_default_and_hides_cross_user() -> None:
    _, client = _remote_history()
    listed = client.get(
        "/umrs/companion/sessions",
        params={"session_kind": "subagent", "parent_session_id": ROOT},
        headers={"x-user-id": "alice"},
    )
    ordinary = client.get("/umrs/companion/sessions", headers={"x-user-id": "alice"})

    (row,) = listed.json()["snapshots"]
    assert (row["parent_session_id"], row["agent_id"], row["subagent_status"]) == (
        ROOT,
        AGENT,
        "running",
    )
    assert [row["session_id"] for row in ordinary.json()["snapshots"]] == [ROOT]
    assert _inspect(client, uid="bob").status_code == 404


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        ({"id": "forged-child"}, "subagent session id does not match path"),
        ({"metadata": {"parent_session_id": "other-parent"}}, "immediate parent"),
        ({"metadata": {"parent_agent_id": "other-agent"}}, "parent agent"),
        ({"metadata": {"agent_id": ""}}, "agent identity"),
        ({"metadata": {"venue": "local"}}, "venue binding"),
    ],
)
def test_client_lineage_and_backend_forgery_is_refused(change: dict[str, Any], detail: str) -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    record = _record()
    record.update((key, value) for key, value in change.items() if key != "metadata")
    if "metadata" in change:
        record["metadata"].update(change["metadata"])

    response = _put(client, record)

    assert response.status_code == 409, response.text
    assert detail in response.json()["detail"]
    assert ("alice", CHILD) not in pool.table


def test_generic_owner_snapshot_cannot_forge_a_child() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)

    response = client.post(
        f"/umrs/companion/sessions/{CHILD}/snapshot",
        json={"payload": json.dumps(_record()), "updated_at": "2026-11-10T00:00:00Z"},
        headers={"x-user-id": "alice"},
    )

    assert response.status_code == 409
    assert "bound venue rail" in response.json()["detail"]
    assert ("alice", CHILD) not in pool.table


def test_generic_owner_queue_cannot_become_a_second_child_writer() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    before = pool.table[("alice", CHILD)]["payload"]

    response = client.post(
        f"/umrs/companion/sessions/{CHILD}/queue",
        json={"action": "append", "content": "forged owner turn", "client_id": "owner"},
        headers={"x-user-id": "alice"},
    )

    assert response.status_code == 409
    assert "read-only" in response.json()["detail"]
    assert pool.table[("alice", CHILD)]["payload"] == before


def test_nested_children_require_the_actual_bound_immediate_parent() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    nested = _record(
        "nested-child",
        parent_session_id=CHILD,
        parent_agent_id=AGENT,
        agent_id="agent-nested",
    )
    response = _put(client, nested, child="nested-child")
    forged = _record(
        "other-nested",
        parent_session_id=CHILD,
        parent_agent_id="not-the-child-agent",
        agent_id="agent-other",
    )
    refusal = _put(client, forged, child="other-nested")

    assert response.status_code == 201, response.text
    assert refusal.status_code == 409
    assert "parent agent" in refusal.json()["detail"]
    stored = json.loads(pool.table[("alice", "nested-child")]["payload"])
    assert stored["subagent_binding"]["parent_session_id"] == CHILD
    assert stored["subagent_binding"]["root_session_id"] == ROOT


def test_offline_writes_fail_without_changing_history_or_using_the_parent() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    before = pool.table[("alice", CHILD)]["payload"]
    pool.venues[ROOT]["state"] = "stopped"

    offline = _put(client, _record(revision=2, status="completed"))
    offline_event = _append(client, 2)

    assert offline.status_code == 409
    assert "not running" in offline.json()["detail"]
    assert offline_event.status_code == 409
    assert pool.table[("alice", CHILD)]["payload"] == before


def test_lease_mismatch_and_unknown_ids_fail_without_parent_fallback() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    before = pool.table[("alice", CHILD)]["payload"]
    other_lease = _client(pool, lease_id="lease-2")
    unknown = client.post(
        f"/umrs/companion/sessions/{ROOT}/rail/subagents/unknown/events",
        json={"source_event_id": 1, "event": _event(1, session_id="unknown")},
        headers={"Authorization": "Bearer lease-token"},
    )

    assert _put(other_lease, _record(revision=2)).status_code == 409
    assert _append(other_lease, 2).status_code == 409
    assert pool.table[("alice", CHILD)]["payload"] == before
    assert unknown.status_code == 404
    assert _inspect(client, child="unknown").status_code == 404


def test_snapshot_replay_stale_order_and_conflict_preserve_the_newer_record() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    newer = _record(revision=3, status="completed")
    newer["messages"].append({"role": "assistant", "content": "complete"})
    assert _put(client, newer).status_code == 200
    before = pool.table[("alice", CHILD)]["payload"]

    stale = _put(client, _record(revision=2))
    replay = _put(client, newer)
    conflict_record = _record(revision=3)
    conflict = _put(client, conflict_record)

    assert stale.status_code == 200
    assert stale.json()["stale"] is True
    assert replay.json()["replay"] is True
    assert conflict.status_code == 409
    assert "replay conflict" in conflict.json()["detail"]
    assert pool.table[("alice", CHILD)]["payload"] == before


def test_event_replay_order_and_lineage_are_replay_safe() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    appended = _append(client, 8)
    duplicate = _append(client, 8)
    conflict = _append(client, 8, message="changed")
    stale = _append(client, 7)

    assert appended.status_code == 201
    assert duplicate.status_code == 200
    assert duplicate.json()["replay"] is True
    assert conflict.status_code == 409
    assert stale.status_code == 409
    assert "order" in stale.json()["detail"]
    mismatched = _event(9)
    mismatched["data"]["parent_session_id"] = "other"
    response = client.post(
        f"/umrs/companion/sessions/{ROOT}/rail/subagents/{CHILD}/events",
        json={"source_event_id": 9, "event": mismatched},
        headers={"Authorization": "Bearer lease-token"},
    )
    assert response.status_code == 409


def test_cancellation_persists_terminal_snapshot_and_actual_event() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    cancelled = _record(revision=2, status="cancelled")
    cancelled["metadata"]["ended_at"] = "2026-11-10T00:01:00+00:00"
    cancelled["metadata"]["terminal_error"] = "cancelled by owner"
    assert _put(client, cancelled).status_code == 201
    terminal = _event(10, event_type="error", message="cancelled by owner")
    terminal["data"]["status"] = "cancelled"
    response = client.post(
        f"/umrs/companion/sessions/{ROOT}/rail/subagents/{CHILD}/events",
        json={"source_event_id": 10, "event": terminal},
        headers={"Authorization": "Bearer lease-token"},
    )

    assert response.status_code == 201
    inspected = _inspect(client)
    assert inspected.json()["record"]["metadata"]["subagent_status"] == "cancelled"
    assert inspected.json()["events"][-1]["data"]["status"] == "cancelled"


def test_event_payload_bound_rejects_oversize_without_blocking_valid_append() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    oversized = _event(1, message="x" * 70_000)
    response = client.post(
        f"/umrs/companion/sessions/{ROOT}/rail/subagents/{CHILD}/events",
        json={"source_event_id": 1, "event": oversized},
        headers={"Authorization": "Bearer lease-token"},
    )

    assert response.status_code == 400
    assert "event payload" in response.json()["detail"]
    assert _append(client, 1).status_code == 201


def test_owner_read_is_paginated_and_ordinary_snapshot_omits_event_vector() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    assert _append(client, 1).status_code == 201
    ordinary = client.get(f"/umrs/companion/sessions/{CHILD}", headers={"x-user-id": "alice"})

    assert ordinary.status_code == 200
    assert "subagent_event_log" not in ordinary.json()["payload"]
    assert _inspect(client, limit=0).status_code == 422
    assert _inspect(client, limit=201).status_code == 422
    assert _inspect(client, after_id=-1).status_code == 422


def test_owner_database_outage_is_named_not_an_empty_or_local_fallback() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    client.app.state.umrs_db_pool = None

    response = _inspect(client)
    listing = client.get(
        "/umrs/companion/sessions?session_kind=subagent", headers={"x-user-id": "alice"}
    )

    assert response.status_code == 503
    assert listing.status_code == 503


def _assert_row_locks(pool: _Pool) -> None:
    locks = [
        sql
        for sql in pool.recorded
        if sql.startswith("select payload from companion_session_snapshots")
    ]
    assert locks
    for sql in locks:
        assert "for update" in sql
        assert "user_id = %s and session_id = %s" in sql


def test_child_lock_is_row_scoped_for_concurrent_append_and_snapshot_writers() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    assert _put(client, _record()).status_code == 201
    assert _append(client, 1).status_code == 201
    assert _put(client, _record(revision=2)).status_code == 200

    _assert_row_locks(pool)
