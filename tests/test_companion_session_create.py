"""Server-authored companion session create (UI plan §3.4, lane UX-V2).

``POST /umrs/companion/sessions`` makes the first journal row of a session a
browser starts. Before this lane the route did not exist (405: only the
listing answered ``/umrs/companion/sessions``). The fake pool answers the
SQL shapes of the shared row helpers; the ``_pg`` tests run the same writes
on a CI-shaped Postgres (env-gated like ``test_companion_snapshot_lww_pg``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_companion_snapshot_lww_pg import _db_url
from umrs_r4.api.companion_session_create import (
    SessionCreateBody,
    create_session_row,
)
from umrs_r4.api.companion_session_create import router as create_router
from umrs_r4.api.companion_sessions import router as sessions_router
from umrs_r4.api.companion_sessions import snapshot_detail_row

# The columns r4's CompanionSessionStore.load() decodes, with their JSON type
# (r4/src/r4/companion/session_store.py ``_ROW_JSON_TYPES``).
ROW_JSON_TYPES = {"messages": list, "active_skills": list, "memory": dict, "metadata": dict}

ALICE = {"x-user-id": "alice"}
BOB = {"x-user-id": "bob"}


class _Cursor:
    """The create's SQL: the CAS lock + upsert, then the detail read."""

    def __init__(self, table: dict) -> None:
        self.table = table
        self.row = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params=()):
        text = " ".join(sql.split()).lower()
        if text.startswith("select payload, revision"):
            row = self.table.get(tuple(params))
            self.row = (row["payload"], row["revision"]) if row else None
        elif text.startswith("insert into companion_session_snapshots"):
            self._insert(params)
        elif text.startswith("select s.session_id"):
            self._detail(params)
        else:
            raise AssertionError(f"unexpected SQL: {text[:80]}")

    def _insert(self, params):
        uid, sid, name, payload, marker, venue, state, base = params
        existing = self.table.get((uid, sid))
        if existing is not None and existing["revision"] != base:
            self.rowcount, self.row = 0, None
            return
        revision = (existing or {"revision": 0})["revision"] + 1
        self.table[(uid, sid)] = {
            "name": name,
            "payload": payload,
            "marker": marker,
            "venue": venue or "local",
            "state": state,
            "revision": revision,
        }
        self.rowcount, self.row = 1, (revision,)

    def _detail(self, params):
        uid, sid = params
        row = self.table.get((uid, sid))
        self.row = None
        if row is not None:
            self.row = (sid, row["name"], row["payload"], row["marker"], row["venue"],
                        row["state"], None, row["revision"])  # fmt: skip

    def fetchone(self):
        return self.row


class _Conn:
    def __init__(self, table):
        self.table = table

    def cursor(self):
        return _Cursor(self.table)

    def commit(self):
        pass

    def rollback(self):
        pass


class _Pool:
    def __init__(self):
        self.table: dict = {}

    def getconn(self):
        return _Conn(self.table)

    def putconn(self, _conn):
        pass


@pytest.fixture
def api():
    app = FastAPI()
    app.include_router(create_router)
    app.include_router(sessions_router)
    pool = _Pool()
    app.state.umrs_db_pool = pool
    return TestClient(app), pool


def assert_bootstrap_accepts(detail: dict, sid: str) -> dict:
    """What the venue worker's bootstrap checks of a rail snapshot detail row
    (r4 ``venue_worker._snapshot_payload`` + ``adopt_remote_detail`` + ``load``)."""
    assert detail["session_id"] == sid
    payload = json.loads(detail["payload"])
    assert isinstance(payload, dict) and payload["id"] == sid
    for column, kind in ROW_JSON_TYPES.items():
        assert isinstance(payload[column], str), column  # save()'s JSON-string form
        assert isinstance(json.loads(payload[column]), kind), column
    assert isinstance(payload["model"], str)
    return payload


def test_create_then_get_gives_a_row_the_bootstrap_accepts(api):
    client, _ = api
    made = client.post(
        "/umrs/companion/sessions",
        json={"name": "web one", "model": "example-model", "mode": "plan"},
        headers=ALICE,
    )
    assert made.status_code == 201, made.text
    sid = made.json()["session_id"]
    got = client.get(f"/umrs/companion/sessions/{sid}", headers=ALICE)
    assert got.status_code == 200
    detail = got.json()
    assert detail["venue"] == "frontend" and detail["revision"] == 1
    payload = assert_bootstrap_accepts(detail, sid)
    assert payload["model"] == "example-model" and payload["name"] == "web one"
    metadata = json.loads(payload["metadata"])
    assert metadata["venue"] == "frontend" and metadata["interaction_mode"] == "plan"


def test_each_create_mints_a_new_id(api):
    client, pool = api
    ids = {
        client.post("/umrs/companion/sessions", json={}, headers=ALICE).json()["session_id"]
        for _ in range(3)
    }
    assert len(ids) == 3 and len(pool.table) == 3


def test_created_session_is_the_owners_alone(api):
    client, _ = api
    sid = client.post("/umrs/companion/sessions", json={}, headers=ALICE).json()["session_id"]
    assert client.get(f"/umrs/companion/sessions/{sid}", headers=BOB).status_code == 404


def test_create_needs_identity(api):
    client, _ = api
    assert client.post("/umrs/companion/sessions", json={}).status_code == 401


@pytest.mark.parametrize("body", [{"venue": "local"}, {"mode": "yolo"}, {"name": "x" * 300}])
def test_create_refuses_what_it_does_not_offer(api, body):
    client, pool = api
    assert client.post("/umrs/companion/sessions", json=body, headers=ALICE).status_code == 422
    assert pool.table == {}


def test_create_names_an_unnamed_session(api):
    client, _ = api
    detail = client.post("/umrs/companion/sessions", json={}, headers=ALICE).json()
    assert detail["name"].startswith("Session ")


# ── on Postgres ────────────────────────────────────────────────────────────

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_SCHEMA = (
    "0079_companion_trust_session_cascade.sql",
    "0080_companion_session_snapshots.sql",
    "0084_companion_session_registry.sql",
    "0143_companion_snapshot_revision.sql",
)
_PG_USER = "create-pg-test"
needs_pg = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


@pytest.fixture()
def pg_conn():
    import psycopg2

    conn = psycopg2.connect(_db_url())
    with conn.cursor() as cur:
        for name in _SCHEMA:
            cur.execute((_MIGRATIONS / name).read_text())
        cur.execute("DELETE FROM companion_session_snapshots WHERE user_id = %s", (_PG_USER,))
    conn.commit()
    yield conn
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM companion_session_snapshots WHERE user_id = %s", (_PG_USER,))
    conn.commit()
    conn.close()


@needs_pg
def test_pg_create_writes_a_row_the_bootstrap_accepts(pg_conn):
    body = SessionCreateBody(name="pg", model="example-model")
    detail = create_session_row(pg_conn, _PG_USER, body, "example-project")
    sid = detail["session_id"]
    again = snapshot_detail_row(pg_conn, _PG_USER, sid)
    assert again is not None and again["revision"] == 1 and again["venue"] == "frontend"
    payload = assert_bootstrap_accepts(again, sid)
    assert json.loads(payload["metadata"])["project_id"] == "example-project"
    assert snapshot_detail_row(pg_conn, "someone-else", sid) is None
