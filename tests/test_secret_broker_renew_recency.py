"""Lease renewal records holder activity without weakening lifetime caps."""

from datetime import timedelta

from umrs.jobs.secret_broker import SecretBroker
from umrs_r4.jobs.frontend_session_backend import FrontendSessionStatusChecker


def test_cap_refused_renew_still_records_recency():
    broker = SecretBroker()
    lease, _ = broker.issue_lease(
        "job-1",
        "venue",
        "inference",
        lease_seconds=3600,
        owner_id="alice",
        session_id="s1",
        max_lifetime_seconds=3600,
    )
    stale = lease.issued_at - timedelta(hours=1)
    lease.last_accessed_at = stale
    before = lease.expires_at

    assert broker.renew_lease(lease.lease_id, 900) is False
    persisted = broker.get_lease(lease.lease_id)
    assert persisted.expires_at == before
    assert persisted.last_accessed_at > stale + timedelta(minutes=30)


def test_cap_refused_renew_is_not_idle_for_the_session_checker():
    broker = SecretBroker()
    lease, _ = broker.issue_lease(
        "job-1",
        "venue",
        "inference",
        lease_seconds=3600,
        owner_id="alice",
        session_id="s1",
        max_lifetime_seconds=3600,
    )
    lease.last_accessed_at = lease.issued_at - timedelta(hours=1)
    assert FrontendSessionStatusChecker._idle(lease, 60) is True

    assert broker.renew_lease(lease.lease_id, 900) is False
    persisted = broker.get_lease(lease.lease_id)
    assert FrontendSessionStatusChecker._idle(persisted, 60) is False


def test_revoked_lease_renew_records_nothing():
    broker = SecretBroker()
    lease, _ = broker.issue_lease(
        "job-1",
        "venue",
        "inference",
        lease_seconds=3600,
        owner_id="alice",
        session_id="s1",
        max_lifetime_seconds=3600,
    )
    stale = lease.issued_at - timedelta(hours=1)
    lease.last_accessed_at = stale
    before = lease.expires_at
    assert broker.revoke_lease(lease.lease_id)

    assert broker.renew_lease(lease.lease_id, 900) is False
    persisted = broker.get_lease(lease.lease_id)
    assert persisted.last_accessed_at == stale
    assert persisted.expires_at == before


def test_successful_renew_still_extends_and_records():
    broker = SecretBroker()
    lease, _ = broker.issue_lease(
        "job-1",
        "venue",
        "inference",
        lease_seconds=60,
        owner_id="alice",
        session_id="s1",
        max_lifetime_seconds=3600,
    )
    before = lease.expires_at

    assert broker.renew_lease(lease.lease_id, 120) is True
    persisted = broker.get_lease(lease.lease_id)
    assert persisted.expires_at == before + timedelta(seconds=120)
    assert persisted.last_accessed_at is not None
