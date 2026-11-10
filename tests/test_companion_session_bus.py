"""ADR-114 S2.1: durable identity + session messages in ADR-046's turn queue."""

from __future__ import annotations

import json

from test_companion_session_snapshots import _client, _FakePool, _post


def _bind(client, sid: str, *, uid: str = "alice", bus_session_id: str | None = None, **labels):
    body = {
        "bus_session_id": sid if bus_session_id is None else bus_session_id,
        "name": labels.pop("name", sid),
        "host": labels.pop("host", "host-a"),
        "cwd": labels.pop("cwd", "/work"),
        "project": labels.pop("project", "project-a"),
        "model": labels.pop("model", "model-a"),
        "mode": labels.pop("mode", "chat"),
        **labels,
    }
    return client.put(
        f"/umrs/companion/sessions/{sid}/session-bus",
        json=body,
        headers={"x-user-id": uid},
    )


def _seed(client, sid: str, *, uid: str = "alice", state: str = "ongoing", venue: str = "local"):
    response = _post(
        client,
        None,
        uid=uid,
        sid=sid,
        state=state,
        venue=venue,
        payload=json.dumps({"id": sid, "metadata": {}}),
    )
    assert response.status_code == 200, response.text


def _send(
    client,
    sender: str,
    recipient: str,
    *,
    uid: str = "alice",
    message_id: str = "msg-aaaaaaaaaaaa",
    body: str = "one",
    reply_to: str = "",
):
    return client.post(
        f"/umrs/companion/sessions/{sender}/session-bus/send",
        json={
            "message_id": message_id,
            "recipient_id": recipient,
            "body": body,
            "sender_name": sender,
            "sender_kind": "session",
            "reply_to": reply_to,
        },
        headers={"x-user-id": uid},
    )


def _inbox(client, sid: str, *, action: str = "peek", message_ids=None, uid: str = "alice"):
    body = {"action": action}
    if message_ids is not None:
        body["message_ids"] = message_ids
    return client.post(
        f"/umrs/companion/sessions/{sid}/session-bus/inbox",
        json=body,
        headers={"x-user-id": uid},
    )


def _assert_refusal(response, status: int, detail: str) -> None:
    assert response.status_code == status
    assert response.json()["detail"] == detail


def _assert_receipt(response, *, duplicate: bool | None = None) -> None:
    assert response.status_code == 200, response.text
    if duplicate is not None:
        assert response.json()["duplicate"] is duplicate


def _assert_bodies(response, bodies: list[str]) -> None:
    assert [message["body"] for message in response.json()["messages"]] == bodies


def _assert_binding(response, session_id: str) -> None:
    _assert_receipt(response)
    assert response.json()["session_id"] == session_id
    assert response.json()["bus_session_id"] == session_id
    assert "\n" not in response.json()["name"]
    assert "<" not in response.json()["name"]


def _assert_single_presence(response, session_id: str) -> None:
    assert response.status_code == 200, response.text
    (row,) = response.json()["sessions"]
    assert row["session_id"] == session_id
    assert row["venue"] == "local"
    assert "payload" not in row
    assert "body" not in row


def _metadata(pool: _FakePool, sid: str, uid: str = "alice") -> dict:
    payload = json.loads(pool.table[(uid, sid)]["payload"])
    metadata = payload["metadata"]
    return json.loads(metadata) if isinstance(metadata, str) else metadata


def _edit_metadata(pool: _FakePool, sid: str, edit) -> None:
    payload = json.loads(pool.table[("alice", sid)]["payload"])
    raw = payload["metadata"]
    metadata = json.loads(raw) if isinstance(raw, str) else raw
    edit(metadata)
    payload["metadata"] = metadata
    pool.table[("alice", sid)]["payload"] = json.dumps(payload)


def _expire_presence(pool: _FakePool, sid: str) -> None:
    def expire(metadata):
        metadata["session_bus"]["expires_at"] = "2000-01-01T00:00:00+00:00"

    _edit_metadata(pool, sid, expire)


def test_binding_proves_durable_identity_before_remote_presence():
    pool = _FakePool()
    client = _client(pool)
    _seed(client, "sess-aaaaaaa1")

    mismatch = _bind(client, "sess-aaaaaaa1", bus_session_id="sess-invented")
    _assert_refusal(mismatch, 409, "session bus identity does not match durable session")
    bound = _bind(client, "sess-aaaaaaa1", name="lab\n<b>")
    _assert_binding(bound, "sess-aaaaaaa1")
    presence = client.get(
        "/umrs/companion/sessions/session-bus/presence", headers={"x-user-id": "alice"}
    )
    _assert_single_presence(presence, "sess-aaaaaaa1")


def test_mismatch_never_bound_stopped_and_cross_user_recipients_are_refused():
    pool = _FakePool()
    client = _client(pool)
    for sid in ("sess-aaaaaaa1", "sess-bbbbbbb2", "sess-ccccccc3", "sess-ddddddd4"):
        _seed(client, sid)
    _seed(client, "sess-bob00001", uid="bob")
    _bind(client, "sess-aaaaaaa1")
    _bind(client, "sess-bbbbbbb2")
    _post(
        client,
        pool,
        sid="sess-ddddddd4",
        state="stopped",
        payload=json.dumps({"id": "sess-ddddddd4", "metadata": {}}),
    )
    _bind(client, "sess-ddddddd4")

    assert _send(client, "sess-aaaaaaa1", "sess-aaaaaaa1").status_code == 409
    never_bound = _send(client, "sess-aaaaaaa1", "sess-ccccccc3")
    _assert_refusal(never_bound, 409, "recipient is not on the session bus")
    stopped = _send(client, "sess-aaaaaaa1", "sess-ddddddd4")
    _assert_refusal(stopped, 409, "recipient session is not accepting messages")
    foreign = _send(client, "sess-aaaaaaa1", "sess-bob00001", message_id="msg-bbbbbbbbbbbb")
    _assert_refusal(foreign, 404, "snapshot not found")
    listing = client.get(
        "/umrs/companion/sessions/session-bus/presence", headers={"x-user-id": "bob"}
    )
    assert listing.status_code == 200
    assert listing.json()["sessions"] == []


def test_offline_delivery_replay_read_receipts_and_fifo_use_one_queue(monkeypatch):
    monkeypatch.setenv("UMRS_SESSION_MESSAGE_TTL_SECONDS", "86400")
    pool = _FakePool()
    client = _client(pool)
    _seed(client, "sess-aaaaaaa1")
    _seed(client, "sess-bbbbbbb2")
    _bind(client, "sess-aaaaaaa1")
    _bind(client, "sess-bbbbbbb2")
    _expire_presence(pool, "sess-bbbbbbb2")
    presence = client.get(
        "/umrs/companion/sessions/session-bus/presence", headers={"x-user-id": "alice"}
    )
    assert [row["session_id"] for row in presence.json()["sessions"]] == ["sess-aaaaaaa1"], (
        presence.text,
        _metadata(pool, "sess-bbbbbbb2"),
    )

    first = _send(client, "sess-aaaaaaa1", "sess-bbbbbbb2")
    _assert_receipt(first, duplicate=False)
    replay = _send(client, "sess-aaaaaaa1", "sess-bbbbbbb2")
    _assert_receipt(replay, duplicate=True)
    conflict = _send(client, "sess-aaaaaaa1", "sess-bbbbbbb2", body="changed")
    _assert_refusal(conflict, 409, "message id replay does not match the original")


def test_fifo_read_receipts_and_replay_after_read(monkeypatch):
    monkeypatch.setenv("UMRS_SESSION_MESSAGE_TTL_SECONDS", "86400")
    pool = _FakePool()
    client = _client(pool)
    _seed(client, "sess-aaaaaaa1")
    _seed(client, "sess-bbbbbbb2")
    _bind(client, "sess-aaaaaaa1")
    _bind(client, "sess-bbbbbbb2")
    first = _send(client, "sess-aaaaaaa1", "sess-bbbbbbb2")
    assert first.status_code == 200
    second = _send(
        client,
        "sess-aaaaaaa1",
        "sess-bbbbbbb2",
        message_id="msg-cccccccccccc",
        body="two",
    )
    assert second.status_code == 200

    peek = _inbox(client, "sess-bbbbbbb2")
    _assert_bodies(peek, ["one", "two"])
    taken = _inbox(
        client,
        "sess-bbbbbbb2",
        action="take",
        message_ids=["msg-aaaaaaaaaaaa"],
    )
    _assert_bodies(taken, ["one"])
    remaining = _inbox(client, "sess-bbbbbbb2")
    _assert_bodies(remaining, ["two"])
    after_read = _send(client, "sess-aaaaaaa1", "sess-bbbbbbb2")
    _assert_receipt(after_read, duplicate=True)
    assert _metadata(pool, "sess-bbbbbbb2")["chat_queue"][0]["read_at"]


def _ordinary_bodies(entries: list[dict]) -> list[str]:
    return [entry["content"] for entry in entries if entry.get("kind") != "session_message"]


def _expire_last_message(metadata):
    metadata["chat_queue"][-1]["expires_at"] = "2000-01-01T00:00:00+00:00"


def test_expiry_order_and_queue_clear_cancellation(monkeypatch):
    monkeypatch.setenv("UMRS_SESSION_MESSAGE_TTL_SECONDS", "86400")
    pool = _FakePool()
    client = _client(pool)
    _seed(client, "sess-aaaaaaa1")
    _seed(client, "sess-bbbbbbb2")
    _bind(client, "sess-aaaaaaa1")
    _bind(client, "sess-bbbbbbb2")
    ordinary = client.post(
        "/umrs/companion/sessions/sess-bbbbbbb2/queue",
        json={"action": "append", "content": "ordinary turn", "client_id": "owner"},
        headers={"x-user-id": "alice"},
    )
    assert ordinary.status_code == 200
    assert _send(client, "sess-aaaaaaa1", "sess-bbbbbbb2").status_code == 200
    monkeypatch.setenv("UMRS_SESSION_MESSAGE_TTL_SECONDS", "1")
    later = _send(
        client,
        "sess-aaaaaaa1",
        "sess-bbbbbbb2",
        message_id="msg-cccccccccccc",
        body="short lived",
    )
    assert later.status_code == 200

    _edit_metadata(pool, "sess-bbbbbbb2", _expire_last_message)
    peeked = _inbox(client, "sess-bbbbbbb2")
    _assert_bodies(peeked, ["one"])
    entries = _metadata(pool, "sess-bbbbbbb2")["chat_queue"]
    assert _ordinary_bodies(entries) == ["ordinary turn"]
    cleared = client.post(
        "/umrs/companion/sessions/sess-bbbbbbb2/queue",
        json={"action": "clear"},
        headers={"x-user-id": "alice"},
    )
    assert cleared.status_code == 200
    assert _inbox(client, "sess-bbbbbbb2").json()["messages"] == []
    assert _metadata(pool, "sess-bbbbbbb2")["chat_queue"] == []


def test_session_bus_database_outage_is_not_turned_into_empty_presence():
    client = _client(_FakePool())
    client.app.state.umrs_db_pool = None
    response = client.get(
        "/umrs/companion/sessions/session-bus/presence", headers={"x-user-id": "alice"}
    )
    assert response.status_code == 503
