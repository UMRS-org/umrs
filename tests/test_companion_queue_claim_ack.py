"""Exactly-once queued turns: claim + ack on the worker rail (ADR-057 D-8, SH-J3).

``pop`` removed a turn before it ran: a worker that died between the pop
and the reply lost the turn. ``claim`` now leaves the entry queued with a
server-minted ``claim_id``; only ``ack`` with that claim removes it, and an
unacked entry is offered again to the next attempt. Entry ids are unique
(the worker's ``metadata.queue_id`` idempotency key); the old ``q-<len+1>``
ids repeated after a pop.

The pure transitions run everywhere; the row tests are env-gated like
``test_companion_snapshot_revision`` (``POSTGRES_*`` or ``UMRS_TEST_DB_URL``)
and build their tables in a scratch schema they drop afterwards.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from umrs_r4.api._companion_queue_claim import (
    QueueAckBody,
    ack_entry,
    claim_front,
    worker_ack_row,
    worker_claim_row,
)

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_TABLES = (
    "0079_companion_trust_session_cascade.sql",
    "0080_companion_session_snapshots.sql",
    "0084_companion_session_registry.sql",
    "0143_companion_snapshot_revision.sql",
)
_USER = "claim-ack-user"
_NOW = "2026-11-27T00:00:00+00:00"


def _turn(entry_id: str, content: str) -> dict[str, Any]:
    return {"id": entry_id, "content": content, "client_id": "tui", "added_at": "t"}


def _message(entry_id: str, *, notified_at: str = "") -> dict[str, Any]:
    return {
        "id": entry_id,
        "kind": "session_message",
        "content": "ping",
        "expires_at": "2099-01-01T00:00:00+00:00",
        "notified_at": notified_at,
        "read_at": "",
    }


# ── pure transitions ───────────────────────────────────────────────────────


def test_claim_leaves_the_turn_queued_and_stamps_the_attempt():
    entries = [_turn("q-a1", "first"), _turn("q-a2", "second")]
    front = claim_front(entries, _NOW)
    assert front is entries[0] and len(entries) == 2
    assert front["claim_id"].startswith("c-")
    assert front["claimed_at"] == _NOW and front["attempts"] == 1


def test_an_unacked_claim_is_offered_again_with_a_new_claim_id():
    entries = [_turn("q-a1", "first")]
    first = dict(claim_front(entries, _NOW))
    again = claim_front(entries, _NOW)
    assert again["id"] == first["id"]
    assert again["claim_id"] != first["claim_id"]
    assert again["attempts"] == 2


def test_ack_by_the_latest_claim_removes_only_that_entry():
    entries = [_turn("q-a1", "first"), _turn("q-a2", "second")]
    claim = claim_front(entries, _NOW)["claim_id"]
    assert ack_entry(entries, queue_id="q-a1", claim_id=claim) == "acked"
    assert [e["id"] for e in entries] == ["q-a2"]
    assert ack_entry(entries, queue_id="q-a1", claim_id=claim) == "absent"


def test_an_ack_from_a_superseded_claim_keeps_the_entry():
    entries = [_turn("q-a1", "first")]
    stale = claim_front(entries, _NOW)["claim_id"]
    claim_front(entries, _NOW)
    assert ack_entry(entries, queue_id="q-a1", claim_id=stale) == "superseded"
    assert len(entries) == 1


def test_messages_are_offered_once_and_never_acked_away():
    entries = [_message("msg-aaaaaaaa1"), _turn("q-a1", "turn")]
    assert claim_front(entries, _NOW)["id"] == "msg-aaaaaaaa1"
    assert entries[0]["notified_at"] == _NOW
    assert claim_front(entries, _NOW)["id"] == "q-a1"
    assert ack_entry(entries, queue_id="msg-aaaaaaaa1", claim_id="c-x") == "not_a_turn"
    assert len(entries) == 2


def test_a_legacy_length_id_is_replaced_before_it_becomes_a_queue_id():
    entries = [_turn("q-2", "old shape"), _turn("q-2", "repeated id")]
    first = claim_front(entries, _NOW)
    assert first["id"] != "q-2" and first["id"].startswith("q-")
    assert ack_entry(entries, queue_id=first["id"], claim_id=first["claim_id"]) == "acked"
    assert claim_front(entries, _NOW)["id"] not in {"q-2", first["id"]}


def test_nothing_to_offer_is_none():
    assert claim_front([_message("msg-aaaaaaaa1", notified_at=_NOW)], _NOW) is None
    assert claim_front([], _NOW) is None


def test_append_ids_never_repeat_after_a_pop():
    from umrs_r4.api.companion_sessions import _queue_apply

    entries: list = []
    kwargs = {"content": "x", "client_id": "c", "uid": "u", "now": _NOW}
    _queue_apply(entries, action="append", **kwargs)
    first = entries[0]["id"]
    _queue_apply(entries, action="pop", **kwargs)
    _queue_apply(entries, action="append", **kwargs)
    assert entries[0]["id"] != first


# ── rows on a real Postgres ────────────────────────────────────────────────


def _db_url() -> str | None:
    explicit = os.environ.get("UMRS_TEST_DB_URL")
    if explicit:
        return explicit
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'test')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'test')}@{host}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/{os.environ.get('POSTGRES_DB', 'umrs_test')}"
    )


_needs_pg = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


@pytest.fixture()
def conn():
    import psycopg2

    schema = f"shj3_{uuid.uuid4().hex[:12]}"
    connection = psycopg2.connect(_db_url())
    with connection.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
        for name in _TABLES:
            cur.execute((_MIGRATIONS / name).read_text())
    connection.commit()
    yield connection
    connection.rollback()
    with connection.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    connection.commit()
    connection.close()


def _seed(conn: Any, sid: str, queue: list[dict[str, Any]]) -> None:
    metadata = json.dumps({"venue": "frontend", "chat_queue": queue})
    payload = json.dumps({"id": sid, "messages": "[]", "metadata": metadata})
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO companion_session_snapshots "
            "(user_id, session_id, name, payload, update_marker) "
            "VALUES (%s, %s, 'n', %s, %s)",
            (_USER, sid, payload, _NOW),
        )
    conn.commit()


def _queue(conn: Any, sid: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT payload FROM companion_session_snapshots "
            "WHERE user_id = %s AND session_id = %s",
            (_USER, sid),
        )
        payload = json.loads(cur.fetchone()[0])
    return json.loads(payload["metadata"])["chat_queue"]


def _ack(conn: Any, sid: str, front: dict[str, Any]) -> dict[str, Any]:
    body = QueueAckBody(queue_id=front["id"], claim_id=front["claim_id"])
    return worker_ack_row(conn, _USER, sid, body)


@_needs_pg
def test_a_dead_attempt_is_reoffered_and_acked_exactly_once(conn):
    """The acceptance: claim, die before the reply, claim again, ack once."""
    _seed(conn, "s1", [_turn("q-a1", "run me")])
    dead = worker_claim_row(conn, _USER, "s1")["front"]
    assert _queue(conn, "s1")[0]["claim_id"] == dead["claim_id"]
    retry = worker_claim_row(conn, _USER, "s1")["front"]
    assert retry["id"] == dead["id"] and retry["attempts"] == 2
    assert _ack(conn, "s1", dead)["outcome"] == "superseded"
    assert _ack(conn, "s1", retry)["outcome"] == "acked"
    assert _queue(conn, "s1") == []
    assert worker_claim_row(conn, _USER, "s1")["front"] is None
    assert _ack(conn, "s1", retry)["outcome"] == "absent"


@_needs_pg
def test_claim_keeps_the_rest_of_the_metadata(conn):
    _seed(conn, "s2", [_turn("q-a1", "one"), _turn("q-a2", "two")])
    front = worker_claim_row(conn, _USER, "s2")["front"]
    _ack(conn, "s2", front)
    assert [e["id"] for e in _queue(conn, "s2")] == ["q-a2"]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT payload FROM companion_session_snapshots WHERE session_id = 's2'"
        )
        metadata = json.loads(json.loads(cur.fetchone()[0])["metadata"])
    assert metadata["venue"] == "frontend"
