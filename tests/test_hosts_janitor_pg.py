"""Hosts janitor end to end on Postgres through the r4 extension (SH-H1a).

A host that misses its heartbeats turns ``down``; every live frontend-venue
session on it (this site's, by its durable job's venue) ends ``host_lost``,
its snapshot projects ``error`` and its lease is revoked in
``secret_leases``. Sessions on other hosts, of other sites, or already ended
are untouched.

Env-gated like the other ``*_pg`` suites: runs where ``UMRS_TEST_DB_URL`` or
``POSTGRES_*`` point at a scratch Postgres, in a scratch schema dropped
afterwards. ``jobs`` is reduced to the three columns the provider joins on.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pytest
from umrs.hosts.janitor import HostJanitor
from umrs.hosts.registry import HostRegistry
from umrs.jobs.secret_broker import SecretBroker

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
_BASE = (
    "0079_companion_trust_session_cascade.sql",
    "0080_companion_session_snapshots.sql",
    "0084_companion_session_registry.sql",
    "0085_secret_leases.sql",
    "0099_companion_venue_sessions.sql",
    "0100_companion_venue_sessions_job_id.sql",
    "0124_secret_leases_venue.sql",
)
_JOBS = (
    "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, venue TEXT NOT NULL DEFAULT '',"
    " project_id TEXT NOT NULL DEFAULT '')"
)


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


class _SchemaPool:
    def __init__(self, url: str, schema: str) -> None:
        self.url, self.schema = url, schema

    def getconn(self) -> Any:
        import psycopg2

        return psycopg2.connect(self.url, options=f"-c search_path={self.schema}")

    def putconn(self, conn: Any) -> None:
        conn.close()


def _hosts_migration() -> str:
    found = sorted(_MIGRATIONS.glob("*_hosts.sql"))
    assert len(found) == 1, found
    return found[0].name


@pytest.fixture()
def pool():
    import psycopg2

    url = _db_url()
    schema = f"shh1a_{uuid.uuid4().hex[:12]}"
    admin = psycopg2.connect(url)
    with admin.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
        for name in (*_BASE, _hosts_migration()):
            cur.execute((_MIGRATIONS / name).read_text())
        cur.execute(_JOBS)
    admin.commit()
    yield _SchemaPool(url, schema)
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.commit()
    admin.close()


def _exec(pool: _SchemaPool, sql: str, params: tuple = ()) -> list[Any]:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
        conn.commit()
        return list(rows)
    finally:
        pool.putconn(conn)


def _session(pool: _SchemaPool, sid: str, *, host: str, site: str, state: str) -> None:
    job_id = f"job-{sid}"
    _exec(pool, "INSERT INTO jobs VALUES (%s, %s, 'proj-1')", (job_id, site))
    _exec(
        pool,
        "INSERT INTO companion_venue_sessions (session_id, user_id, state, job_id,"
        " host_id) VALUES (%s, 'alice', %s, %s, %s)",
        (sid, state, job_id, host),
    )
    _exec(
        pool,
        "INSERT INTO companion_session_snapshots (user_id, session_id, payload,"
        " update_marker, state) VALUES ('alice', %s, '{}', '1', 'ongoing')",
        (sid,),
    )


def _state(pool: _SchemaPool, sid: str) -> tuple[str, str]:
    venue_row = _exec(
        pool, "SELECT state FROM companion_venue_sessions WHERE session_id = %s", (sid,)
    )
    snap = _exec(
        pool,
        "SELECT state FROM companion_session_snapshots WHERE session_id = %s",
        (sid,),
    )
    return venue_row[0][0], snap[0][0]


_CASES = (
    ("lost", "cpu-1", "site-a", "running"),
    ("paused", "cpu-1", "site-a", "paused"),
    ("ended", "cpu-1", "site-a", "stopped"),
    ("other-host", "local", "site-a", "running"),
    ("other-site", "cpu-1", "site-b", "running"),
)


def _seed(pool: _SchemaPool, broker: SecretBroker) -> dict[str, str]:
    leases = {}
    for sid, host, site, state in _CASES:
        _session(pool, sid, host=host, site=site, state=state)
        lease, _token = broker.issue_lease(
            f"session-{sid}", "inference-token", "inference", session_id=sid
        )
        leases[sid] = lease.lease_id
    _exec(
        pool,
        "UPDATE hosts SET last_heartbeat_at = now() - interval '1 hour' WHERE host_id = 'cpu-1'",
    )
    return leases


@pytest.fixture()
def swept(pool):
    from umrs_r4.venues.host_sessions import build_host_sessions

    registry = HostRegistry(pool, venue="site-a")
    registry.register("local")
    registry.register("cpu-1", launch_mode="agent")
    broker = SecretBroker(db_pool=pool, venue="site-a")
    leases = _seed(pool, broker)
    janitor = HostJanitor(
        registry, broker=broker, sessions=[build_host_sessions(pool)], down_after_s=90
    )
    return pool, broker, leases, janitor, janitor.sweep()


def test_down_host_ends_only_its_live_sessions_host_lost(swept) -> None:
    pool, _broker, _leases, janitor, report = swept
    assert report.down == ["cpu-1"]
    assert sorted(report.lost) == ["lost", "paused"]
    assert [
        _state(pool, sid) for sid in ("lost", "paused", "other-host", "other-site")
    ] == [
        ("host_lost", "error"),
        ("host_lost", "error"),
        ("running", "ongoing"),
        ("running", "ongoing"),
    ]
    assert _state(pool, "ended")[0] == "stopped"
    assert janitor.sweep().lost == []  # idempotent


def test_down_host_sessions_lose_their_leases(swept) -> None:
    _pool, broker, leases, _janitor, _report = swept
    valid = {sid: broker.get_lease(lid).is_valid for sid, lid in leases.items()}
    assert valid == {
        "lost": False,
        "paused": False,
        "ended": True,  # not live: not the janitor's to end
        "other-host": True,
        "other-site": True,
    }
