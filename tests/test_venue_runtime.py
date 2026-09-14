"""R1-4: venue runtime executor (heartbeat/idle/expiry sweep semantics)."""

from __future__ import annotations

from unittest.mock import MagicMock

from umrs.jobs.secret_broker import SecretBroker
from umrs.jobs.session_venue_runtime import VenueRuntimeExecutor


def _executor(idle_seconds: int = 1800):
    broker = SecretBroker()
    provisioner = MagicMock()
    executor = VenueRuntimeExecutor(broker, provisioner, idle_seconds=idle_seconds)
    return broker, provisioner, executor


def _lease_for(broker: SecretBroker, session_id: str, **kw):
    args = {
        "job_id": f"session-{session_id}",
        "secret_name": "inference-token",
        "scope": "inference",
        "owner_id": "alice",
        "session_id": session_id,
        "source_ids": ("11111111-1111-4111-8111-111111111111",),
    }
    args.update(kw)
    return broker.issue_lease(**args)[0]


class TestHeartbeat:
    def test_renew_extends_and_marks_recency(self):
        broker, _, executor = _executor()
        lease = _lease_for(broker, "sess-h1", lease_seconds=900)
        assert executor.heartbeat(lease.lease_id, extend_seconds=600)
        fresh = broker.get_lease(lease.lease_id)
        assert fresh.last_accessed_at is not None

    def test_renew_denied_after_revoke(self):
        broker, _, executor = _executor()
        lease = _lease_for(broker, "sess-h2")
        broker.revoke_lease(lease.lease_id)
        assert not executor.heartbeat(lease.lease_id, extend_seconds=60)


class TestSweep:
    def test_fresh_session_untouched(self):
        broker, provisioner, executor = _executor(idle_seconds=1800)
        _lease_for(broker, "sess-keep")
        result = executor.sweep_once()
        assert result.expired == 0 and result.idled == 0 and not result.torn_down
        provisioner.teardown.assert_not_called()

    def test_expired_lease_swept_and_torn_down(self):
        broker, provisioner, executor = _executor()
        lease = _lease_for(broker, "sess-dead", lease_seconds=1)
        # Force expiry: move expires_at into the past.
        lease.expires_at = lease.issued_at
        broker._leases[lease.lease_id] = lease
        result = executor.sweep_once()
        assert result.expired == 1
        assert result.torn_down == ["sess-dead"]
        provisioner.teardown.assert_called_once_with("sess-dead")
        assert not broker.get_lease(lease.lease_id).is_valid

    def test_idle_session_revoked_and_torn_down(self):
        broker, provisioner, executor = _executor(idle_seconds=600)
        lease = _lease_for(broker, "sess-idle1", lease_seconds=3600)
        from datetime import timedelta

        lease.last_accessed_at = lease.issued_at - timedelta(hours=2)
        broker._leases[lease.lease_id] = lease
        result = executor.sweep_once()
        assert result.idled == 1
        provisioner.teardown.assert_called_once_with("sess-idle1")
        assert not broker.get_lease(lease.lease_id).is_valid

    def test_recent_access_survives_idle_window(self):
        broker, provisioner, executor = _executor(idle_seconds=600)
        lease = _lease_for(broker, "sess-live", lease_seconds=3600)
        lease.access_count = 5
        broker._leases[lease.lease_id] = lease
        result = executor.sweep_once()
        assert result.idled == 0
        provisioner.teardown.assert_not_called()

    def test_revoked_only_session_gets_teardown(self):
        broker, provisioner, executor = _executor()
        lease = _lease_for(broker, "sess-revoke-then-sweep")
        broker.revoke_lease(lease.lease_id)
        result = executor.sweep_once()
        assert result.torn_down == ["sess-revoke-then-sweep"]
        provisioner.teardown.assert_called_once_with("sess-revoke-then-sweep")

    def test_revoked_session_with_valid_sibling_is_skipped(self):
        broker, provisioner, executor = _executor()
        revoked, _secret = broker.issue_lease(
            job_id="session-sx",
            secret_name="t",
            scope="inference",
            owner_id="alice",
            session_id="sess-sx",
        )
        live, _ = broker.issue_lease(
            job_id="session-sx",
            secret_name="t2",
            scope="inference",
            owner_id="alice",
            session_id="sess-sx",
            lease_seconds=3600,
        )
        broker.revoke_lease(revoked.lease_id)
        result = executor.sweep_once()
        assert "sess-sx" not in result.torn_down
        provisioner.teardown.assert_not_called()

    def test_teardown_error_isolated_sweep_continues(self):
        broker, provisioner, executor = _executor()
        a = _lease_for(broker, "sess-a", lease_seconds=1)
        b = _lease_for(broker, "sess-b", lease_seconds=1)
        a.expires_at = a.issued_at
        b.expires_at = b.issued_at
        broker._leases[a.lease_id] = a
        broker._leases[b.lease_id] = b
        provisioner.teardown.side_effect = [RuntimeError("podman down"), None]
        result = executor.sweep_once()
        assert result.expired == 2
        assert len(result.torn_down) == 1  # newest lease swept first; error isolated
        assert (result.torn_down == ["sess-a"]) != (result.torn_down == ["sess-b"])
        assert len(result.errors) == 1
        assert "podman down" in result.errors[0]

    def test_revoked_lease_gets_teardown_via_sweeper(self):
        """F-5: a revoked inference lease + no valid siblings → sweeper tears down."""
        broker, provisioner, executor = _executor()
        lease = _lease_for(broker, "sess-end", lease_seconds=1)
        broker.revoke_lease(lease.lease_id)
        result = executor.sweep_once()
        assert result.expired == 0 and result.idled == 0
        assert result.torn_down == ["sess-end"]
        provisioner.teardown.assert_called_once_with("sess-end")
