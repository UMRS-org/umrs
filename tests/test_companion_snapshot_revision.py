"""Journal revision CAS on real Postgres (ADR-057 D-7, slice SH-J1).

Two clients that PUT from one base used to settle by ``update_marker`` LWW,
so the later marker silently dropped the other client's turn. A PUT that
carries ``base_revision`` is now a compare-and-swap on the row's
``revision``: the second PUT from one base gets a typed 409
``companionsessions.revision_conflict`` (current revision on
``x-companion-session-revision``) and the row keeps the first. A PUT without
``base_revision`` keeps the legacy LWW and still bumps the revision.

Env-gated like ``test_companion_snapshot_lww_pg``: runs where ``POSTGRES_*``
(or ``UMRS_TEST_DB_URL``) point at a Postgres. Each test builds its tables
in a scratch schema that it drops afterwards.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_REVISION_MIGRATION = "0142_sh_j1_companion_snapshot_revision.sql"
_BASE_MIGRATIONS = (
    "0079_companion_trust_session_cascade.sql",
    "0080_companion_session_snapshots.sql",
    "0084_companion_session_registry.sql",
)
_USER = "rev-cas-user"


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


pytestmark = pytest.mark.skipif(_db_url() is None, reason="no CI-shape Postgres")


def _run(conn: Any, *names: str) -> None:
    with conn.cursor() as cur:
        for name in names:
            cur.execute((_MIGRATIONS / name).read_text())
    conn.commit()


@pytest.fixture()
def conn():
    import psycopg2

    schema = f"shj1_{uuid.uuid4().hex[:12]}"
    connection = psycopg2.connect(_db_url())
    with connection.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
    connection.commit()
    _run(connection, *_BASE_MIGRATIONS, _REVISION_MIGRATION)
    yield connection
    connection.rollback()
    with connection.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    connection.commit()
    connection.close()


class _Pool:
    """The app-state pool shape the routes borrow one connection from."""

    def __init__(self, connection: Any) -> None:
        self._conn = connection

    def getconn(self) -> Any:
        return self._conn

    def putconn(self, _conn: Any) -> None:
        return None


def _payload(turn: str) -> str:
    meta = json.dumps({"venue": "local", "chat_queue": []})
    return json.dumps({"id": "s", "messages": json.dumps([turn]), "metadata": meta})


@pytest.fixture()
def client(conn, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs_r4.api.companion_sessions import router

    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = _Pool(conn)
    with TestClient(app) as test_client:
        test_client.headers.update({"X-User-ID": _USER})
        yield test_client


def _put(client: Any, sid: str, turn: str, marker: str, base: int | None = None) -> Any:
    body: dict[str, Any] = {"name": "n", "payload": _payload(turn), "updated_at": marker}
    if base is not None:
        body["base_revision"] = base
    return client.post(f"/umrs/companion/sessions/{sid}/snapshot", json=body)


def _messages(client: Any, sid: str) -> list[str]:
    detail = client.get(f"/umrs/companion/sessions/{sid}").json()
    return json.loads(json.loads(detail["payload"])["messages"])


# ── the acceptance pair: two PUTs from one base ────────────────────────────


def test_second_put_from_one_base_gets_409_and_row_keeps_the_first(client):
    created = _put(client, "s1", "base", "2026-11-24T00:00:00+00:00", base=0)
    assert created.status_code == 200
    assert created.json()["revision"] == 1
    first = _put(client, "s1", "turn-a", "2026-11-24T00:00:01+00:00", base=1)
    assert first.json() == {"status": "ok", "applied": True, "revision": 2}
    # The loser's marker is LATER: under the old LWW it overwrote turn-a.
    second = _put(client, "s1", "turn-b", "2026-11-24T00:00:09+00:00", base=1)
    assert second.status_code == 409
    assert second.headers["x-platform-error-code"] == "companionsessions.revision_conflict"
    assert second.headers["x-companion-session-revision"] == "2"
    assert _messages(client, "s1") == ["turn-a"]


def test_rebased_put_after_a_conflict_applies(client):
    _put(client, "s2", "base", "2026-11-24T00:00:00+00:00", base=0)
    _put(client, "s2", "turn-a", "2026-11-24T00:00:01+00:00", base=1)
    retry = _put(client, "s2", "turn-a+b", "2026-11-24T00:00:02+00:00", base=2)
    assert retry.json()["revision"] == 3
    assert _messages(client, "s2") == ["turn-a+b"]


def test_cas_ignores_markers(client):
    """Timestamps are display-only under CAS: an older marker on the right base wins."""
    _put(client, "s3", "base", "2026-11-24T09:00:00+00:00", base=0)
    older = _put(client, "s3", "turn", "2026-11-24T08:00:00+00:00", base=1)
    assert older.json()["applied"] is True
    assert _messages(client, "s3") == ["turn"]


def test_base_on_an_absent_session_is_a_conflict(client):
    stale = _put(client, "gone", "turn", "2026-11-24T00:00:00+00:00", base=4)
    assert stale.status_code == 409
    assert stale.headers["x-companion-session-revision"] == "0"
    assert client.get("/umrs/companion/sessions/gone").status_code == 404


def test_negative_base_revision_is_a_validation_error(client):
    assert _put(client, "neg", "t", "2026-11-24T00:00:00+00:00", base=-1).status_code == 422


# ── legacy PUT (no base_revision): LWW unchanged, revision still bumps ─────


def test_legacy_put_keeps_lww_and_bumps_revision(client):
    assert _put(client, "l1", "a", "2026-11-24T00:00:05+00:00").json()["revision"] == 1
    stale = _put(client, "l1", "old", "2026-11-24T00:00:01+00:00")
    assert stale.status_code == 200
    assert stale.json() == {"status": "ok", "applied": False, "revision": 1}
    newer = _put(client, "l1", "b", "2026-11-24T00:00:09+00:00")
    assert newer.json() == {"status": "ok", "applied": True, "revision": 2}
    assert _messages(client, "l1") == ["b"]


def test_legacy_write_moves_a_cas_clients_base(client):
    """An old client's write is still a write: a CAS client on the old base loses."""
    _put(client, "l2", "base", "2026-11-24T00:00:00+00:00", base=0)
    _put(client, "l2", "legacy", "2026-11-24T00:00:01+00:00")
    assert _put(client, "l2", "cas", "2026-11-24T00:00:02+00:00", base=1).status_code == 409


# ── reads carry the revision (feature detection) ───────────────────────────


def test_get_and_list_answer_the_revision(client):
    _put(client, "r1", "a", "2026-11-24T00:00:00+00:00", base=0)
    _put(client, "r1", "b", "2026-11-24T00:00:01+00:00", base=1)
    assert client.get("/umrs/companion/sessions/r1").json()["revision"] == 2
    listing = client.get("/umrs/companion/sessions").json()["snapshots"]
    assert [(row["session_id"], row["revision"]) for row in listing] == [("r1", 2)]


def test_listing_formats_on_postgres_with_and_without_project(client):
    """The listing's literal '{%' read as a placeholder: IndexError on psycopg2."""
    meta = {"config": {"project_id": "p1"}}
    body = {"payload": json.dumps({"metadata": meta}), "updated_at": "2026-11-24T00:00:00Z"}
    client.post("/umrs/companion/sessions/p/snapshot", json=body)
    for query in ("", "?project=p1"):
        listing = client.get(f"/umrs/companion/sessions{query}")
        assert listing.status_code == 200
        assert [row["session_id"] for row in listing.json()["snapshots"]] == ["p"]


def test_server_owned_queue_writes_do_not_bump_the_revision(client, conn):
    from umrs_r4.api.companion_sessions import queue_mutate_row

    _put(client, "q1", "a", "2026-11-24T00:00:00+00:00", base=0)
    queue_mutate_row(conn, _USER, "q1", action="append", content="next", client_id="c1")
    assert _put(client, "q1", "b", "2026-11-24T00:00:01+00:00", base=1).status_code == 200


# ── the venue rail shares the write ────────────────────────────────────────


@pytest.fixture()
def rail_client(conn, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs_r4.api import companion_venue
    from umrs_r4.api._rail_identity import RailPrincipal

    app = FastAPI()
    app.include_router(companion_venue.router)
    app.state.umrs_db_pool = _Pool(conn)
    app.dependency_overrides[companion_venue.rail_principal] = lambda: RailPrincipal(
        owner=_USER, session_id="v1", lease_id="lease", lease=None
    )
    with TestClient(app) as test_client:
        yield test_client


def _rail_put(rail_client: Any, turn: str, base: int | None) -> Any:
    body: dict[str, Any] = {
        "name": "n",
        "payload": _payload(turn),
        "updated_at": "2026-11-24T00:00:00+00:00",
        "venue": "local",
    }
    if base is not None:
        body["base_revision"] = base
    return rail_client.post("/umrs/companion/sessions/v1/rail/snapshot", json=body)


def test_rail_stale_base_is_a_409_and_venue_stays_pinned(rail_client):
    assert _rail_put(rail_client, "w1", 0).json()["revision"] == 1
    stale = _rail_put(rail_client, "w2", 0)
    assert stale.status_code == 409
    assert stale.headers["x-platform-error-code"] == "companionsessions.revision_conflict"
    detail = rail_client.get("/umrs/companion/sessions/v1/rail/snapshot").json()
    assert detail["revision"] == 1
    assert detail["venue"] == "frontend"  # never body-chosen
    assert json.loads(json.loads(detail["payload"])["messages"]) == ["w1"]


# ── payload-size metric (SH-D1: decides whether a per-turn journal pays) ───


def test_payload_size_metric_counts_by_mode_and_outcome(client):
    registry = pytest.importorskip("prometheus_client").REGISTRY

    def count(mode: str, outcome: str) -> float:
        labels = {"mode": mode, "outcome": outcome}
        value = registry.get_sample_value("umrs_companion_snapshot_payload_bytes_count", labels)
        return value or 0.0

    before = (count("cas", "applied"), count("cas", "conflict"), count("legacy", "stale"))
    _put(client, "m1", "a", "2026-11-24T00:00:05+00:00", base=0)
    _put(client, "m1", "b", "2026-11-24T00:00:06+00:00", base=0)
    _put(client, "m1", "c", "2026-11-24T00:00:01+00:00")
    after = (count("cas", "applied"), count("cas", "conflict"), count("legacy", "stale"))
    assert [b - a for a, b in zip(before, after, strict=True)] == [1.0, 1.0, 1.0]


# ── migration up / down ────────────────────────────────────────────────────


def _has_revision(conn: Any) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_schema = current_schema()"
            " AND table_name = 'companion_session_snapshots' AND column_name = 'revision'"
        )
        return cur.fetchone() is not None


def test_migration_up_down_up_keeps_rows(conn):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO companion_session_snapshots (user_id, session_id, payload,"
            " update_marker) VALUES ('u', 'old', '{}', '')"
        )
    conn.commit()
    _run(conn, _REVISION_MIGRATION)  # idempotent re-run
    with conn.cursor() as cur:
        cur.execute("SELECT revision FROM companion_session_snapshots WHERE session_id = 'old'")
        assert cur.fetchone() == (0,)  # pre-existing rows: never versioned
        cur.execute("ALTER TABLE companion_session_snapshots DROP COLUMN IF EXISTS revision")
    conn.commit()
    assert not _has_revision(conn)
    _run(conn, _REVISION_MIGRATION)
    assert _has_revision(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM companion_session_snapshots")
        assert cur.fetchone() == (1,)
