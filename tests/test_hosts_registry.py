"""Hosts registry, heartbeat, janitor and admin CLI (ADR-098 D-4, SH-H1a).

Acceptance (decisions-2026-11-24 §5, lane SH-H1a):
- a missed heartbeat turns a host ``down``; its sessions end ``host_lost``
  and their leases are revoked;
- a draining host gets no placement;
- a single node shows one host row (the inline front registers itself).

The in-memory registry runs everywhere; the Postgres cases (migration
``*_hosts.sql``, site scoping on a shared database) run where
``UMRS_TEST_DB_URL`` or ``POSTGRES_*`` point at a scratch Postgres, each in a
scratch schema dropped afterwards.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from umrs.hosts import INLINE_HOST_ID
from umrs.hosts.janitor import (
    HOST_LOST,
    HostJanitor,
    HostMaintenance,
    HostSession,
    down_after_seconds,
    heartbeat_seconds,
)
from umrs.hosts.registry import HostRegistry, HostRegistryError
from umrs.jobs.secret_broker import SecretBroker

_MIGRATIONS = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"


def _age(registry: HostRegistry, host_id: str, seconds: float) -> None:
    """Pretend the host's last heartbeat was ``seconds`` ago (memory backend)."""
    row = registry._backend._rows[(registry.venue, host_id)]
    row.last_heartbeat_at = datetime.now(timezone.utc) - timedelta(seconds=seconds)


class _Sessions:
    """A fake extension: live sessions per host; records host_lost marks."""

    def __init__(self, on_host: dict[str, list[str]]) -> None:
        self.on_host = {h: list(s) for h, s in on_host.items()}
        self.lost: list[tuple[str, str]] = []

    def sessions_on_host(self, venue: str, host_id: str) -> list[HostSession]:
        del venue  # one site per fake
        return [
            HostSession(sid, owner="alice") for sid in self.on_host.get(host_id, [])
        ]

    def mark_host_lost(self, session_id: str, reason: str) -> bool:
        for sessions in self.on_host.values():
            if session_id in sessions:
                sessions.remove(session_id)
                self.lost.append((session_id, reason))
                return True
        return False


def _lease(broker: SecretBroker, session_id: str) -> str:
    lease, _token = broker.issue_lease(
        f"session-{session_id}", "inference-token", "inference", session_id=session_id
    )
    return lease.lease_id


# ── registry ────────────────────────────────────────────────────────────────


def test_single_node_shows_one_host_row() -> None:
    registry = HostRegistry(venue="site-a")
    registry.register(INLINE_HOST_ID)
    registry.register(INLINE_HOST_ID)  # every restart re-registers
    hosts = registry.hosts()
    assert [(h.host_id, h.state, h.launch_mode) for h in hosts] == [
        ("local", "active", "inline")
    ]
    assert registry.placeable()[0].host_id == "local"


def test_draining_host_gets_no_placement() -> None:
    registry = HostRegistry(venue="site-a")
    registry.register("local")
    registry.register("cpu-1", launch_mode="agent")
    drained = registry.drain("cpu-1", "maintenance")
    assert (drained.state, drained.reason) == ("draining", "maintenance")
    assert [h.host_id for h in registry.placeable()] == ["local"]
    registry.register("cpu-1", launch_mode="agent")  # a heartbeat keeps the drain
    assert registry.get("cpu-1").state == "draining"
    assert registry.undrain("cpu-1").state == "active"
    assert [h.host_id for h in registry.placeable()] == ["cpu-1", "local"]


def test_admin_transitions_refuse_by_name() -> None:
    registry = HostRegistry(venue="site-a")
    registry.register("local")
    cases = (
        (registry.remove, "local", "hosts.host_active"),
        (registry.undrain, "local", "hosts.not_draining"),
        (registry.drain, "ghost", "hosts.not_found"),
    )
    for verb, host_id, code in cases:
        with pytest.raises(HostRegistryError) as err:
            verb(host_id)
        assert err.value.code == code
    with pytest.raises(HostRegistryError) as err:
        registry.register("x", launch_mode="ssh")
    assert err.value.code == "hosts.invalid"


def test_remove_forgets_a_drained_host() -> None:
    registry = HostRegistry(venue="site-a")
    registry.register("local")
    registry.drain("local")
    assert registry.remove("local").host_id == "local"
    assert registry.hosts() == []
    assert registry.heartbeat("local") is None  # removed hosts get no beat


def test_heartbeat_brings_a_down_host_back() -> None:
    registry = HostRegistry(venue="site-a")
    registry.register("cpu-1", launch_mode="agent")
    _age(registry, "cpu-1", 600)
    assert registry.mark_down("cpu-1", 90, "no heartbeat")
    with pytest.raises(HostRegistryError) as err:
        registry.drain("cpu-1")
    assert err.value.code == "hosts.host_down"
    back = registry.heartbeat("cpu-1")
    assert (back.state, back.reason) == ("active", "")


def test_mark_down_loses_to_a_fresh_heartbeat() -> None:
    registry = HostRegistry(venue="site-a")
    registry.register("cpu-1", launch_mode="agent")
    assert registry.stale(90) == []
    assert not registry.mark_down("cpu-1", 90, "no heartbeat")
    assert registry.get("cpu-1").state == "active"


# ── janitor ─────────────────────────────────────────────────────────────────


def test_missed_heartbeat_downs_host_ends_sessions_revokes_leases() -> None:
    registry = HostRegistry(venue="site-a")
    registry.register("local")
    registry.register("cpu-1", launch_mode="agent")
    broker = SecretBroker(venue="site-a")
    lost_lease = _lease(broker, "s-on-cpu")
    kept_lease = _lease(broker, "s-on-local")
    sessions = _Sessions({"cpu-1": ["s-on-cpu"], "local": ["s-on-local"]})
    janitor = HostJanitor(registry, broker=broker, sessions=[sessions], down_after_s=90)
    _age(registry, "cpu-1", 600)

    report = janitor.sweep()

    assert (report.down, report.lost, report.revoked) == (["cpu-1"], ["s-on-cpu"], 1)
    assert registry.get("cpu-1").state == "down"
    assert registry.get("local").state == "active"
    assert sessions.lost[0][0] == "s-on-cpu"
    assert sessions.lost[0][1].startswith(HOST_LOST)
    assert not broker.get_lease(lost_lease).is_valid
    assert broker.get_lease(kept_lease).is_valid
    assert registry.placeable()[0].host_id == "local"


def test_sweep_finishes_sessions_of_an_already_down_host() -> None:
    """A crash between marking down and ending sessions loses nothing."""
    registry = HostRegistry(venue="site-a")
    registry.register("cpu-1", launch_mode="agent")
    _age(registry, "cpu-1", 600)
    registry.mark_down("cpu-1", 90, "no heartbeat")
    broker = SecretBroker(venue="site-a")
    lease_id = _lease(broker, "s1")
    sessions = _Sessions({"cpu-1": ["s1"]})
    report = HostJanitor(
        registry, broker=broker, sessions=[sessions], down_after_s=90
    ).sweep()
    assert (report.down, report.lost) == ([], ["s1"])
    assert not broker.get_lease(lease_id).is_valid
    again = HostJanitor(registry, broker=broker, sessions=[sessions], down_after_s=90)
    assert again.sweep().lost == []  # idempotent


def test_own_tick_never_downs_the_inline_host() -> None:
    """Heartbeat before sweep: a restart longer than the timeout is no loss."""
    registry = HostRegistry(venue="site-a")
    registry.register(INLINE_HOST_ID)
    _age(registry, INLINE_HOST_ID, 3600)
    sessions = _Sessions({INLINE_HOST_ID: ["s1"]})
    janitor = HostJanitor(
        registry,
        broker=SecretBroker(venue="site-a"),
        sessions=[sessions],
        down_after_s=90,
    )
    report = HostMaintenance(registry, janitor, interval=30).tick()
    assert (report.down, report.lost) == ([], [])
    assert registry.get(INLINE_HOST_ID).state == "active"


def test_timing_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UMRS_HOST_HEARTBEAT_SEC", raising=False)
    monkeypatch.delenv("UMRS_HOST_DOWN_AFTER_SEC", raising=False)
    assert (heartbeat_seconds(), down_after_seconds(30)) == (30.0, 90.0)
    monkeypatch.setenv("UMRS_HOST_HEARTBEAT_SEC", "1")
    monkeypatch.setenv("UMRS_HOST_DOWN_AFTER_SEC", "10")
    assert (heartbeat_seconds(), down_after_seconds(heartbeat_seconds())) == (5.0, 10.0)
    monkeypatch.setenv("UMRS_HOST_DOWN_AFTER_SEC", "nonsense")
    assert down_after_seconds(30) == 90.0


# ── admin CLI ───────────────────────────────────────────────────────────────


@pytest.fixture()
def cli_registry(monkeypatch: pytest.MonkeyPatch) -> HostRegistry:
    from umrs.cli.admin import hosts as cli_hosts

    registry = HostRegistry(venue="site-a")
    registry.register(INLINE_HOST_ID)
    monkeypatch.setattr(cli_hosts, "_context", lambda *_: (registry, None))
    monkeypatch.setattr(cli_hosts, "live_sessions", lambda *_: 0)
    return registry


def _cli(argv: list[str]) -> int:
    from umrs.cli.admin.main import build_parser

    args = build_parser().parse_args(["host", *argv])
    return int(args.func(args, None))


def test_cli_list_shows_the_single_node_row(
    cli_registry: HostRegistry, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _cli(["list"]) == 0
    assert "local" in capsys.readouterr().out
    assert _cli(["list", "--json"]) == 0
    assert '"host_id": "local"' in capsys.readouterr().out


def test_cli_drain_undrain(cli_registry: HostRegistry) -> None:
    assert _cli(["drain", "local", "--reason", "kernel update"]) == 0
    assert cli_registry.placeable() == []
    assert _cli(["undrain", "local"]) == 0
    assert cli_registry.get("local").state == "active"


def test_cli_remove_needs_a_drain_first(
    cli_registry: HostRegistry, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _cli(["remove", "local"]) == 1
    assert "hosts.host_active" in capsys.readouterr().err
    assert _cli(["drain", "local"]) == 0
    assert _cli(["remove", "local"]) == 0
    assert cli_registry.hosts() == []


def test_cli_remove_refuses_a_draining_host_with_sessions(
    cli_registry: HostRegistry,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from umrs.cli.admin import hosts as cli_hosts

    monkeypatch.setattr(cli_hosts, "live_sessions", lambda *_: 2)
    cli_registry.drain("local")
    assert _cli(["remove", "local"]) == 1
    assert "hosts.host_busy" in capsys.readouterr().err
    assert cli_registry.get("local").state == "draining"


def test_cli_without_database_refuses(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import umrs.cli.admin.cascade as cascade

    monkeypatch.setattr(cascade, "cli_db_pool", lambda: None)
    assert _cli(["list"]) == 2
    assert "no site database" in capsys.readouterr().err


# ── Postgres ────────────────────────────────────────────────────────────────


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


def _hosts_migration() -> str:
    found = sorted(_MIGRATIONS.glob("*_hosts.sql"))
    assert len(found) == 1, found
    return found[0].name


class _SchemaPool:
    """psycopg2 pool shape: a fresh connection per lease, on the test schema."""

    def __init__(self, url: str, schema: str) -> None:
        self.url, self.schema = url, schema

    def getconn(self) -> Any:
        import psycopg2

        conn = psycopg2.connect(self.url, options=f"-c search_path={self.schema}")
        return conn

    def putconn(self, conn: Any) -> None:
        conn.close()


@pytest.fixture()
def pg_pool():
    import psycopg2

    url = _db_url()
    schema = f"shh1a_{uuid.uuid4().hex[:12]}"
    admin = psycopg2.connect(url)
    with admin.cursor() as cur:
        cur.execute(f"CREATE SCHEMA {schema}")
        cur.execute(f"SET search_path TO {schema}")
        for name in ("0099_companion_venue_sessions.sql", _hosts_migration()):
            cur.execute((_MIGRATIONS / name).read_text())
    admin.commit()
    yield _SchemaPool(url, schema)
    with admin.cursor() as cur:
        cur.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.commit()
    admin.close()


@_needs_pg
def test_pg_migration_is_idempotent_and_backfills_local(pg_pool: _SchemaPool) -> None:
    conn = pg_pool.getconn()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO companion_venue_sessions (session_id, user_id, state)"
            " VALUES ('s1', 'alice', 'running')"
        )
        cur.execute((_MIGRATIONS / _hosts_migration()).read_text())
        cur.execute("SELECT host_id FROM companion_venue_sessions")
        assert cur.fetchall() == [("local",)]
    conn.commit()
    pg_pool.putconn(conn)


@_needs_pg
def test_pg_registry_is_site_scoped(pg_pool: _SchemaPool) -> None:
    site_a = HostRegistry(pg_pool, venue="site-a")
    site_b = HostRegistry(pg_pool, venue="site-b")
    site_a.register(INLINE_HOST_ID, labels={"tier": "front"})
    site_a.register(INLINE_HOST_ID)
    site_b.register(INLINE_HOST_ID)
    site_b.register("cpu-1", launch_mode="agent")
    assert [h.host_id for h in site_a.hosts()] == ["local"]
    site_b.drain("cpu-1", "maintenance")
    assert [h.host_id for h in site_b.placeable()] == ["local"]
    assert site_a.get(INLINE_HOST_ID).state == "active"
    with pytest.raises(HostRegistryError):
        site_a.drain("cpu-1")  # another site's host is not ours to drain


@_needs_pg
def test_pg_janitor_downs_only_its_own_sites_hosts(pg_pool: _SchemaPool) -> None:
    site_a = HostRegistry(pg_pool, venue="site-a")
    site_b = HostRegistry(pg_pool, venue="site-b")
    for registry in (site_a, site_b):
        registry.register("cpu-1", launch_mode="agent")
    conn = pg_pool.getconn()
    with conn.cursor() as cur:
        cur.execute("UPDATE hosts SET last_heartbeat_at = now() - interval '1 hour'")
    conn.commit()
    pg_pool.putconn(conn)
    sessions = _Sessions({"cpu-1": ["s1"]})
    report = HostJanitor(
        site_a,
        broker=SecretBroker(venue="site-a"),
        sessions=[sessions],
        down_after_s=90,
    ).sweep()
    assert (report.down, report.lost) == (["cpu-1"], ["s1"])
    assert site_a.get("cpu-1").state == "down"
    assert site_b.get("cpu-1").state == "active"
    assert site_a.heartbeat("cpu-1").state == "active"
    assert site_a.get("cpu-1").to_dict()["launch_mode"] == "agent"


# ── boot wiring ─────────────────────────────────────────────────────────────


def test_start_inline_host_registers_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    from umrs.hosts import janitor as janitor_mod

    monkeypatch.setattr(janitor_mod, "extension_host_sessions", lambda *_: [])
    maintenance = janitor_mod.start_inline_host(None, venue="site-a")
    try:
        hosts = maintenance.registry.hosts()
        assert [(h.host_id, h.launch_mode, h.roles) for h in hosts] == [
            ("local", "inline", ("session",))
        ]
    finally:
        maintenance.stop()


def test_sidecar_hosts_switch_and_missing_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from umrs.api import _asyncpg_pool
    from umrs.server import http_sidecar

    monkeypatch.setenv("UMRS_HOSTS_ENABLED", "false")
    assert http_sidecar._start_hosts() is None
    monkeypatch.setenv("UMRS_HOSTS_ENABLED", "true")
    monkeypatch.setattr(_asyncpg_pool, "_resolve_knowledge_dsn", lambda: ("", None))
    assert http_sidecar._start_hosts() is None


# ── review (SH-H1a adversarial pass) ────────────────────────────────────────


def test_a_drain_survives_a_missed_heartbeat_and_return() -> None:
    """An admin's drain is not undone by a host blip (down, then a beat)."""
    registry = HostRegistry(venue="site-a")
    registry.register("cpu-1", launch_mode="agent")
    registry.drain("cpu-1", "kernel update")
    _age(registry, "cpu-1", 600)
    assert registry.mark_down("cpu-1", 90, "no heartbeat")
    back = registry.heartbeat("cpu-1")
    assert (back.state, back.reason) == ("draining", "kernel update")
    assert registry.placeable() == []
    _age(registry, "cpu-1", 600)
    registry.mark_down("cpu-1", 90, "no heartbeat")
    assert registry.register("cpu-1", launch_mode="agent").state == "draining"
    assert registry.undrain("cpu-1").state == "active"


def _stale_read(registry: HostRegistry, state: str) -> None:
    """The next ``get`` answers ``state`` (a read that lost a race)."""
    real_get = registry.get
    calls = {"n": 0}

    def get(host_id: str):
        calls["n"] += 1
        record = real_get(host_id)
        if calls["n"] == 1 and record is not None:
            record.state = state
        return record

    registry.get = get  # type: ignore[method-assign]


def test_drain_does_not_overwrite_a_host_the_janitor_just_downed() -> None:
    registry = HostRegistry(venue="site-a")
    registry.register("cpu-1", launch_mode="agent")
    _age(registry, "cpu-1", 600)
    registry.mark_down("cpu-1", 90, "no heartbeat")
    _stale_read(registry, "active")
    with pytest.raises(HostRegistryError) as err:
        registry.drain("cpu-1")
    assert err.value.code == "hosts.host_down"
    assert registry.get("cpu-1").state == "down"


def test_remove_does_not_delete_a_host_undrained_meanwhile() -> None:
    registry = HostRegistry(venue="site-a")
    registry.register("cpu-1", launch_mode="agent")
    _stale_read(registry, "draining")
    with pytest.raises(HostRegistryError) as err:
        registry.remove("cpu-1")
    assert err.value.code == "hosts.host_active"
    assert registry.get("cpu-1").state == "active"


def _age_pg(pool: _SchemaPool) -> None:
    conn = pool.getconn()
    with conn.cursor() as cur:
        cur.execute("UPDATE hosts SET last_heartbeat_at = now() - interval '1 hour'")
    conn.commit()
    pool.putconn(conn)


@_needs_pg
def test_pg_a_drain_survives_a_missed_heartbeat_and_return(
    pg_pool: _SchemaPool,
) -> None:
    registry = HostRegistry(pg_pool, venue="site-a")
    registry.register("cpu-1", launch_mode="agent")
    registry.drain("cpu-1", "kernel update")
    _age_pg(pg_pool)
    assert registry.mark_down("cpu-1", 90, "no heartbeat")
    assert registry.get("cpu-1").state == "down"
    back = registry.heartbeat("cpu-1")
    assert (back.state, back.reason) == ("draining", "kernel update")
    _age_pg(pg_pool)
    registry.mark_down("cpu-1", 90, "no heartbeat")
    assert registry.register("cpu-1", launch_mode="agent").state == "draining"
    assert registry.undrain("cpu-1").state == "active"
    assert registry.heartbeat("cpu-1").state == "active"


@_needs_pg
def test_pg_janitor_reads_staleness_on_the_database_clock(
    pg_pool: _SchemaPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process clock ahead of the database never downs a fresh host (nor
    its own inline host right after its beat, ending every session)."""
    from umrs.hosts import registry as registry_mod

    registry = HostRegistry(pg_pool, venue="site-a")
    registry.register(INLINE_HOST_ID)
    ahead = datetime.now(timezone.utc) + timedelta(hours=1)
    monkeypatch.setattr(registry_mod, "_now", lambda: ahead)
    sessions = _Sessions({INLINE_HOST_ID: ["s1"]})
    janitor = HostJanitor(
        registry,
        broker=SecretBroker(venue="site-a"),
        sessions=[sessions],
        down_after_s=90,
    )
    report = HostMaintenance(registry, janitor, interval=30).tick()
    assert (report.down, report.lost) == ([], [])
    assert registry.get(INLINE_HOST_ID).state == "active"
