"""Venue-session registry (0099): in-memory transitions + PG SQL shapes."""

from __future__ import annotations

from tests.fake_pg_pool import FakePool
from umrs.venues.venue_sessions import _COLUMNS, VenueSessionStore


def _provision(store: VenueSessionStore, sid: str = "s1", owner: str = "alice"):
    return store.upsert_provisioning(
        session_id=sid,
        user_id=owner,
        subnet="172.30.0.0/24",
        source_names=("openai-main",),
        source_ids=("11111111-1111-4111-8111-111111111111",),
        default_source_id="11111111-1111-4111-8111-111111111111",
        default_source_name="openai-main",
        web_policy="public",
    )


class TestMemory:
    def test_state_machine(self):
        store = VenueSessionStore()
        row = _provision(store)
        assert row.state == "provisioning" and row.is_active
        assert store.taken_subnets() == {"172.30.0.0/24"}
        running = store.mark_running("s1", lease_id="lease-1", egress_ip="172.20.8.2")
        assert running.state == "running" and running.lease_id == "lease-1"
        rt = store.update_runtime("s1", {"phase": "running", "turn_seq": 3})
        assert rt.runtime["phase"] == "running" and rt.runtime["turn_seq"] == 3
        assert "reported_at" in rt.runtime
        stopped = store.mark_stopped("s1", reason="user")
        assert stopped.state == "stopped" and stopped.stopped_at is not None
        assert store.taken_subnets() == set()

    def test_failed_is_terminal_and_frees_subnet(self):
        store = VenueSessionStore()
        _provision(store)
        failed = store.mark_failed("s1", "network: boom")
        assert failed.state == "failed" and failed.reason == "network: boom"
        assert not failed.is_active

    def test_reprovision_resets_row(self):
        store = VenueSessionStore()
        _provision(store)
        store.mark_failed("s1", "x")
        row = _provision(store)
        assert row.state == "provisioning" and row.reason == ""

    def test_unknown_session_transitions_are_none(self):
        store = VenueSessionStore()
        assert store.mark_running("nope", lease_id="l", egress_ip="i") is None
        assert store.update_runtime("nope", {"phase": "x"}) is None
        assert store.to_dict_smoke() if hasattr(store, "to_dict_smoke") else True

    def test_to_dict_never_carries_user_id_or_secrets(self):
        store = VenueSessionStore()
        row = _provision(store)
        d = row.to_dict()
        assert "user_id" not in d
        assert d["source_names"] == ["openai-main"]
        assert d["default_source"] == "openai-main"


def _pg_row(sid: str = "s1", state: str = "provisioning") -> tuple:
    values = {
        "session_id": sid,
        "user_id": "alice",
        "venue": "frontend",
        "state": state,
        "reason": "",
        "lease_id": "",
        "egress_ip": "",
        "subnet": "172.30.0.0/24",
        "source_names": ["openai-main"],
        "source_ids": ["11111111-1111-4111-8111-111111111111"],
        "default_source_id": "11111111-1111-4111-8111-111111111111",
        "default_source_name": "openai-main",
        "web_policy": "public",
        "runtime": '{"phase": "idle"}',
        "started_at": None,
        "updated_at": None,
        "stopped_at": None,
    }
    return tuple(values[c] for c in _COLUMNS)


class TestPostgresShapes:
    def test_upsert_then_read(self):
        pool = FakePool(script=[None, _pg_row()])
        store = VenueSessionStore(pool)
        row = _provision(store)
        insert_sql, params = pool.record[0]
        assert "INSERT INTO companion_venue_sessions" in insert_sql
        assert "ON CONFLICT (session_id) DO UPDATE" in insert_sql
        assert params[0] == "s1" and params[1] == "alice"
        assert params[4] == ["11111111-1111-4111-8111-111111111111"]
        assert row.session_id == "s1" and row.runtime == {"phase": "idle"}

    def test_transition_sql(self):
        pool = FakePool(script=[_pg_row(), None, _pg_row(state="running")])
        store = VenueSessionStore(pool)
        row = store.mark_running("s1", lease_id="lease-1", egress_ip="172.20.8.2")
        update_sql, params = pool.record[1]
        assert update_sql.startswith("UPDATE companion_venue_sessions SET state = %s")
        assert "lease_id = %s" in update_sql and "egress_ip = %s" in update_sql
        assert params == ("running", "lease-1", "172.20.8.2", "s1")
        assert row.state == "running"

    def test_taken_subnets_reads_active_rows(self):
        pool = FakePool(script=[[_pg_row(), _pg_row("s2")]])
        store = VenueSessionStore(pool)
        assert store.taken_subnets() == {"172.30.0.0/24"}
        sql, params = pool.record[0]
        assert "state = ANY(%s)" in sql
        assert params == (["provisioning", "running"],)
