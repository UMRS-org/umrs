"""R0-2: durable secret-lease broker (memory + Postgres backends).

Pins: issue/validate/revoke/renew/book semantics identical across backends;
pool mode survives a simulated restart (fresh broker over the same pool);
inference-scope bindings (session/sources/egress_ip/project/max_lifetime);
revoke cascades by session and by source.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.jobs.secret_broker import SecretBroker

_MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "src/umrs/migrations"
# 0085 creates the tables; 0124 adds the issuing site (RS-0 S-2).
_MIGRATIONS = ("0085_secret_leases.sql", "0124_secret_leases_venue.sql")

_TABLES = "secret_audit, secret_leases"


def _pg_pool_or_skip():
    pytest.importorskip("psycopg2")
    from umrs.shared.pg import build_sync_pool_from_env

    try:
        pool = build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:
        pytest.skip(f"Postgres unavailable: {exc}")
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            for name in _MIGRATIONS:
                cur.execute((_MIGRATIONS_DIR / name).read_text())
            cur.execute(f"TRUNCATE {_TABLES}")
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture(params=["memory", "pg"])
def broker(request):
    if request.param == "memory":
        return SecretBroker()
    pool = _pg_pool_or_skip()
    request.addfinalizer(pool.closeall)
    return SecretBroker(db_pool=pool)


class TestIssueValidateCycle:
    def test_issue_and_validate(self, broker):
        lease, secret = broker.issue_lease("job-1", "key", "general", owner_id="alice")
        assert lease.owner_id == "alice"
        found = broker.validate_secret(secret)
        assert found is not None
        assert found.lease_id == lease.lease_id
        assert found.access_count == 1

    def test_validate_unknown_returns_none(self, broker):
        assert broker.validate_secret("nope") is None

    def test_revoked_lease_no_longer_validates(self, broker):
        lease, secret = broker.issue_lease("job-1", "key", "general")
        assert broker.revoke_lease(lease.lease_id)
        assert broker.validate_secret(secret) is None

    def test_renew_caps_at_max_lifetime(self, broker):
        lease, _ = broker.issue_lease(
            "job-1", "key", "general", lease_seconds=60, max_lifetime_seconds=3600
        )
        assert broker.renew_lease(lease.lease_id, 120)
        assert not broker.renew_lease(lease.lease_id, 10_000)

    def test_plaintext_never_in_storage(self, broker):
        _, secret = broker.issue_lease("job-1", "key", "general")
        serialized = repr(broker.get_audit_trail(limit=50)) + repr(
            [lease.to_dict() for lease in broker.list_leases(limit=50)]
        )
        assert secret not in serialized


class TestInferenceBindings:
    def test_binding_fields_round_trip(self, broker):
        sid = "3c9f5a8e-1111-4222-8333-444455556666"
        lease, _ = broker.issue_lease(
            "job-1",
            "inference-token",
            "inference",
            owner_id="alice",
            session_id="sess-1",
            source_ids=(sid,),
            default_source_id=sid,
            egress_ip="10.99.0.4",
            project="proj-x",
            max_lifetime_seconds=86_400,
        )
        reloaded = broker.get_lease(lease.lease_id)
        assert reloaded is not None
        assert reloaded.session_id == "sess-1"
        assert reloaded.source_ids == (sid,)
        assert reloaded.default_source_id == sid
        assert reloaded.egress_ip == "10.99.0.4"
        assert reloaded.project == "proj-x"
        assert reloaded.max_lifetime_at is not None

    def test_bind_egress_ip(self, broker):
        lease, _ = broker.issue_lease("job-1", "k", "inference", session_id="s")
        assert broker.bind_egress_ip(lease.lease_id, "10.88.0.9", actor="provisioner")
        assert broker.get_lease(lease.lease_id).egress_ip == "10.88.0.9"


class TestRevokeCascades:
    def test_revoke_by_session(self, broker):
        kept, _ = broker.issue_lease("job-1", "k", "inference", session_id="sess-a")
        dropped, _ = broker.issue_lease("job-1", "k", "inference", session_id="sess-b")
        assert broker.revoke_by_session("sess-b") == 1
        assert broker.get_lease(kept.lease_id).is_valid
        assert not broker.get_lease(dropped.lease_id).is_valid

    def test_revoke_by_source(self, broker):
        src_a = "11111111-1111-4111-8111-111111111111"
        src_b = "22222222-2222-4222-8222-222222222222"
        a, _ = broker.issue_lease("job-1", "k", "inference", source_ids=(src_a,))
        b, _ = broker.issue_lease("job-1", "k", "inference", source_ids=(src_b,))
        assert broker.revoke_by_source(src_a) == 1
        assert not broker.get_lease(a.lease_id).is_valid
        assert broker.get_lease(b.lease_id).is_valid


class TestAuditTrail:
    def test_audit_rows_persist_operations(self, broker):
        lease, secret = broker.issue_lease("job-1", "k", "inference", session_id="s")
        broker.validate_secret(secret)
        broker.bind_egress_ip(lease.lease_id, "10.0.0.2")
        ops = {e.operation for e in broker.get_audit_trail(lease_id=lease.lease_id)}
        assert {"issue", "access", "bind"} <= ops


class TestRestartSemantics:
    def test_fresh_broker_sees_persisted_state(self, request):
        pool = _pg_pool_or_skip()
        request.addfinalizer(pool.closeall)
        first = SecretBroker(db_pool=pool)
        lease, secret = first.issue_lease(
            "job-1",
            "k",
            "inference",
            owner_id="alice",
            session_id="sess-1",
            lease_seconds=3600,
        )
        # Simulate a restart: a brand-new broker over the same pool.
        second = SecretBroker(db_pool=pool)
        assert second.validate_secret(secret) is not None
        assert second.get_lease(lease.lease_id) is not None
        assert second.revoke_lease(lease.lease_id)
        # And a third view confirms the revocation.
        third = SecretBroker(db_pool=pool)
        assert third.get_lease(lease.lease_id).is_revoked
        assert third.validate_secret(secret) is None

    def test_memory_broker_clears_state_fixture_verification(self):
        b = SecretBroker()
        _, secret = b.issue_lease("job-1", "k", "general")
        fresh = SecretBroker()
        assert fresh.validate_secret(secret) is None  # memory mode restart semantics
