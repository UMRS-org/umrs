"""ADR-099 D-7 (RS-1 lane Q): wall clock and idle from the session policy.

The status checker ends a session idle beyond ``spec.policy.lifetime.idle_s``
(before: only ``UMRS_VENUE_IDLE_SECONDS``); the lease's hard cap is the
smaller of ``lifetime.wall_clock_s`` and ``UMRS_VENUE_MAX_LIFETIME_SECONDS``;
the job spec's ``policy`` passes validation unchanged and reaches the
registry row. Defaults equal today's values, so nothing changes without a rule.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from types import SimpleNamespace

from test_frontend_session_backend import (
    NOW,
    FakeBroker,
    FakeProvisioner,
    _job,
    _spec,
    _submit,
)
from tests.fake_pg_pool import FakePool
from umrs.jobs.frontend_session_backend import (
    FrontendSessionBackend,
    FrontendSessionStatusChecker,
    policy_seconds,
    validate_frontend_session_spec,
)
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.venue_sessions import VenueSessionStore

POLICY = {
    "schema": 1,
    "lifetime": {"idle_s": 120, "wall_clock_s": 3600},
    "sessions": {"concurrent_user": 12},
}


class _PolicyStore(VenueSessionStore):
    """Lane P's store takes ``policy=``; this records what the backend sent."""

    def __init__(self):
        super().__init__()
        self.policies: list = []

    def upsert_provisioning(self, *, policy=None, **kw):
        self.policies.append(policy)
        return super().upsert_provisioning(**kw)


def _backend(*, idle=1800, store=None):
    config = SimpleNamespace(
        subnet_pool="172.30.0.0/16",
        idle_seconds=idle,
        provision_config=FrontendVenueConfig(max_lifetime_seconds=7 * 86_400).provision_config,
    )
    prov = FakeProvisioner()
    broker = FakeBroker()
    backend = FrontendSessionBackend(
        config=config,
        provisioner=prov,
        store=store or _PolicyStore(),
        broker=broker,
        pool=FakePool(script=[None] * 20),
    )
    return backend, prov, broker


def _check(checker, ref, job):
    return asyncio.run(checker.check_status(SimpleNamespace(backend_ref=ref), job))


class TestSpec:
    def test_policy_passes_through_unchanged(self):
        assert validate_frontend_session_spec(_spec(policy=POLICY))["policy"] == POLICY

    def test_no_policy_is_empty(self):
        assert validate_frontend_session_spec(_spec())["policy"] == {}
        assert validate_frontend_session_spec(_spec(policy="bad"))["policy"] == {}

    def test_policy_seconds(self):
        assert policy_seconds(POLICY, "idle_s") == 120
        assert policy_seconds({}, "idle_s") is None
        assert policy_seconds({"lifetime": {"idle_s": None}}, "idle_s") is None
        assert policy_seconds({"lifetime": {"idle_s": True}}, "idle_s") is None


class TestIdleFromPolicy:
    def test_policy_idle_ends_a_session_the_site_value_would_keep(self):
        backend, prov, broker = _backend(idle=1800)
        job = _job(spec=_spec(policy=POLICY))
        ref = _submit(backend, job)
        broker.add("lease-1", last=NOW - timedelta(seconds=300))
        res = _check(FrontendSessionStatusChecker(backend), ref, job)
        assert res.status == "succeeded" and res.metrics["end"] == "idle"
        assert backend.store.get("sess-1").reason == "idle beyond 120s"
        assert broker.revoked == ["lease-1"] and prov.torn == ["sess-1"]

    def test_without_policy_the_site_value_holds(self):
        backend, prov, broker = _backend(idle=1800)
        job = _job()
        ref = _submit(backend, job)
        broker.add("lease-1", last=NOW - timedelta(seconds=300))
        res = _check(FrontendSessionStatusChecker(backend), ref, job)
        assert res.status == "running" and prov.torn == []


class TestWallClock:
    def test_lease_cap_is_the_smaller_of_policy_and_site(self):
        backend, prov, _broker = _backend()
        _submit(backend, _job(spec=_spec(policy=POLICY)))
        assert prov.provisioned[0].max_lifetime_seconds == 3600

    def test_site_cap_wins_over_a_longer_policy(self):
        config = FrontendVenueConfig(max_lifetime_seconds=600)
        cfg = config.provision_config(
            session_id="s",
            owner="o",
            source_ids=("a",),
            default_source_id="a",
            subnet="172.30.0.0/24",
            web_policy="none",
            max_lifetime_seconds=3600,
        )
        assert cfg.max_lifetime_seconds == 600

    def test_no_policy_keeps_the_site_cap(self):
        backend, prov, _broker = _backend()
        _submit(backend, _job())
        assert prov.provisioned[0].max_lifetime_seconds == 7 * 86_400


class TestRegistryRow:
    def test_backend_passes_the_policy_to_the_row(self):
        store = _PolicyStore()
        backend, _prov, _broker = _backend(store=store)
        _submit(backend, _job(spec=_spec(policy=POLICY)))
        assert store.policies == [POLICY]

    def test_no_policy_sends_no_policy_keyword(self):
        store = _PolicyStore()
        backend, _prov, _broker = _backend(store=store)
        _submit(backend, _job())
        assert store.policies == [None]
