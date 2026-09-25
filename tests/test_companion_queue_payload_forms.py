"""The journal's chat queue against r4's real payload shape (review 2026-09-25).

r4 mirrors ``metadata`` as a JSON *string* inside the snapshot payload. The
queue helpers read only a dict, so every queue mutation saw an empty queue and
wrote back ``{"chat_queue": …}`` in place of the whole metadata (venue, state,
turn_seq, config gone) — on every idle pop of a venue worker. And a snapshot
upsert took the incoming payload verbatim, erasing turns queued while a save
was in flight. The route fakes seeded dict metadata, so neither ever showed.
"""

from __future__ import annotations

import json

from umrs.api.companion_sessions import (
    _with_server_queue,
    queue_mutate_row,
    upsert_snapshot_row,
)

_QUEUED = [{"id": "q-1", "content": "next", "client_id": "tui", "added_at": "t"}]


def _r4_payload(queue: list | None = None, **metadata) -> str:
    meta = {"venue": "frontend", "state": "ongoing", "turn_seq": 3, **metadata}
    if queue is not None:
        meta["chat_queue"] = queue
    return json.dumps({"id": "s1", "messages": json.dumps([]), "metadata": json.dumps(meta)})


def _metadata(payload: str) -> dict:
    return json.loads(json.loads(payload)["metadata"])


class _Cursor:
    def __init__(self, conn):
        self._conn = conn
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=None):
        sql_l = " ".join(sql.split()).lower()
        self._conn.sql.append(sql_l)
        if sql_l.startswith("select payload"):
            self._conn.fetched = (self._conn.payload,) if self._conn.payload else None
        elif sql_l.startswith("update"):
            self._conn.payload = params[0]
            self.rowcount = 1
        elif sql_l.startswith("insert"):
            self._conn.payload = params[3]
            self.rowcount = 1

    def fetchone(self):
        return self._conn.fetched


class _Conn:
    def __init__(self, payload: str | None):
        self.payload = payload
        self.sql: list[str] = []
        self.fetched = None
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def test_append_keeps_string_metadata_and_every_key():
    conn = _Conn(_r4_payload())
    result = queue_mutate_row(conn, "u", "s1", action="append", content="hi")
    assert result["applied"] is True
    assert isinstance(json.loads(conn.payload)["metadata"], str)
    meta = _metadata(conn.payload)
    assert meta["venue"] == "frontend" and meta["turn_seq"] == 3
    assert [e["content"] for e in meta["chat_queue"]] == ["hi"]


def test_pop_reads_the_queue_inside_string_metadata():
    conn = _Conn(_r4_payload(queue=list(_QUEUED)))
    result = queue_mutate_row(conn, "u", "s1", action="pop")
    assert result["front"]["content"] == "next"
    meta = _metadata(conn.payload)
    assert meta["chat_queue"] == [] and meta["state"] == "ongoing"


def test_empty_pop_writes_nothing():
    before = _r4_payload()
    conn = _Conn(before)
    result = queue_mutate_row(conn, "u", "s1", action="pop")
    assert result["front"] is None
    assert conn.payload == before
    assert not [s for s in conn.sql if s.startswith("update")]
    assert conn.rollbacks == 1 and conn.commits == 0


def test_upsert_keeps_the_server_queue_over_the_snapshot_copy():
    conn = _Conn(_r4_payload(queue=list(_QUEUED)))
    incoming = _r4_payload(queue=[], turn_seq=4)
    upsert_snapshot_row(
        conn, "u", "s1", name="n", payload=incoming, updated_at="m", venue="", state=""
    )
    meta = _metadata(conn.payload)
    assert meta["chat_queue"] == _QUEUED
    assert meta["turn_seq"] == 4  # everything else is the snapshot's


def test_upsert_drops_a_stale_queue_the_server_already_popped():
    conn = _Conn(_r4_payload(queue=[]))
    upsert_snapshot_row(
        conn,
        "u",
        "s1",
        name="n",
        payload=_r4_payload(queue=list(_QUEUED)),
        updated_at="m",
        venue="",
        state="",
    )
    assert _metadata(conn.payload)["chat_queue"] == []


def test_first_upsert_takes_the_snapshot_as_is():
    conn = _Conn(None)
    incoming = _r4_payload(queue=list(_QUEUED))
    upsert_snapshot_row(
        conn, "u", "s1", name="n", payload=incoming, updated_at="m", venue="", state=""
    )
    assert conn.payload == incoming


def test_unchanged_queue_and_non_object_payloads_pass_through():
    stored = _r4_payload(queue=list(_QUEUED))
    same = _r4_payload(queue=list(_QUEUED), turn_seq=9)
    assert _with_server_queue(same, stored) == same
    assert _with_server_queue("not json", stored) == "not json"
    dict_meta = json.dumps({"metadata": {"chat_queue": []}})
    carried = json.loads(_with_server_queue(dict_meta, stored))
    assert carried["metadata"] == {"chat_queue": _QUEUED}
