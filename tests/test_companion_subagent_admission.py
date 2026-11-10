"""S5 reviewer: generic snapshot, bus and venue writes must not own children."""

from __future__ import annotations

import json

from test_companion_session_bus import _bind, _send
from test_companion_session_snapshots import _post
from test_companion_subagents import (
    CHILD,
    ROOT,
    _append,
    _client,
    _inspect,
    _Pool,
    _put,
    _record,
    _seed,
)


def _ordinary_payload() -> str:
    return json.dumps({"id": CHILD, "messages": [], "metadata": {"conversation": "forged"}})


def _bound_history(pool: _Pool, client) -> str:
    assert _put(client, _record()).status_code == 201
    assert _append(client, 1).status_code == 201
    return pool.table[("alice", CHILD)]["payload"]


def test_ordinary_snapshot_cannot_overwrite_a_bound_child_or_its_events() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    before = _bound_history(pool, client)

    response = _post(
        client,
        pool,
        sid=CHILD,
        updated_at="2026-11-11T00:00:00Z",
        payload=_ordinary_payload(),
    )

    assert response.status_code == 409
    assert pool.table[("alice", CHILD)]["payload"] == before
    malformed = _post(
        client,
        pool,
        sid=CHILD,
        updated_at="2026-11-11T00:00:01Z",
        payload="{not-json",
    )
    assert malformed.status_code == 409
    assert pool.table[("alice", CHILD)]["payload"] == before
    inspected = _inspect(client)
    assert inspected.status_code == 200
    assert inspected.json()["record"]["messages"][0]["content"] == "actual child task"
    assert inspected.json()["events"][0]["data"]["result"] == "actual output"


def test_ordinary_conversation_snapshot_still_works() -> None:
    pool = _Pool()
    client = _client(pool)
    sid = "ordinary-session"

    response = _post(
        client,
        pool,
        sid=sid,
        payload=json.dumps({"id": sid, "messages": [], "metadata": {}}),
    )

    assert response.status_code == 200
    assert ("alice", sid) in pool.table


def test_child_cannot_bind_session_bus_presence() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    before = _bound_history(pool, client)

    response = _bind(client, CHILD)

    assert response.status_code == 409
    assert "subagent" in response.json()["detail"]
    assert pool.table[("alice", CHILD)]["payload"] == before


def _mark_child_bound_for_send(pool: _Pool) -> str:
    row = pool.table[("alice", CHILD)]
    payload = json.loads(row["payload"])
    payload["metadata"]["session_bus"] = {
        "bus_session_id": CHILD,
        "name": "child",
        "host": "venue",
        "cwd": "/work",
        "project": "project-a",
        "model": "child-model",
        "mode": "chat",
        "expires_at": "2099-01-01T00:00:00+00:00",
    }
    row["payload"] = json.dumps(payload)
    return row["payload"]


def test_send_to_child_is_rejected_before_recipient_queue_mutation() -> None:
    pool = _Pool()
    _seed(pool)
    client = _client(pool)
    _bound_history(pool, client)
    before = _mark_child_bound_for_send(pool)

    parent_binding = _bind(client, ROOT)
    response = _send(client, ROOT, CHILD)

    assert parent_binding.status_code == 200
    assert response.status_code == 409
    assert "subagent" in response.json()["detail"]
    assert pool.table[("alice", CHILD)]["payload"] == before
    assert "session_bus" in json.loads(pool.table[("alice", ROOT)]["payload"])["metadata"]
