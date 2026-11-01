"""Replica controller for service-tier models (ADR-106 D-3/D-4, plan S2/S9).

Each case names one rule the controller must keep so that a restart never
drops the service. See ``umrs.orchestrator.service_controller`` for the rules.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from umrs.orchestrator.endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    RegisteredEndpoint,
)
from umrs.orchestrator.endpoint_relay import (
    KeepAlivePolicy,
    entry_digest,
    keep_alive_policies,
    plan_relay,
)
from umrs.orchestrator.service_controller import (
    DIGEST_KEY,
    DRAINING_SINCE_KEY,
    ROLL_KEY,
)

NOW = datetime(2026, 11, 1, 12, 0, 0)
WALLTIME = 23 * 3600
R, D, P, F = (
    EndpointStatus.READY,
    EndpointStatus.DRAINING,
    EndpointStatus.PROVISIONING,
    EndpointStatus.FAILED,
)
SVC = KeepAlivePolicy(
    model="glm-5.3-flash",
    project_id="default",
    service=True,
    replicas=2,
    min_available=1,
    max_surge=1,
    digest="new",
)


def _ep(job, status=R, *, age_h=1.0, ready_min=50, active=0, **meta):
    metadata = {DIGEST_KEY: "new", **meta}
    return RegisteredEndpoint(
        project_id="default",
        model="glm-5.3-flash",
        job_id=job,
        host="h",
        port=1,
        status=status,
        created_at=NOW - timedelta(hours=age_h),
        ready_at=NOW - timedelta(minutes=ready_min) if ready_min is not None else None,
        active_requests=active,
        metadata=metadata,
    )


def _plan(*eps, policy=SVC, walltime=WALLTIME):
    return [(a.action, a.job_id) for a in plan_relay(policy, eps, walltime_sec=walltime, now=NOW)]


def test_respawns_up_to_replicas_one_job_at_a_time() -> None:
    plans = [_plan(), _plan(_ep("a")), _plan(_ep("a"), _ep("p", P, ready_min=None))]
    assert plans == [[("provision", None)], [("provision", None)], []]


def test_healthy_fleet_needs_nothing() -> None:
    assert _plan(_ep("a", age_h=3), _ep("b", age_h=9)) == []


def test_walltime_due_replica_gets_a_surge_successor() -> None:
    assert _plan(_ep("a", age_h=21.5), _ep("b", age_h=9)) == [("provision", None)]


def test_no_second_surge_while_one_is_pending() -> None:
    eps = (_ep("a", age_h=21.5), _ep("b", age_h=9), _ep("s", P, ready_min=None))
    assert _plan(*eps) == []


def test_due_replica_drains_once_the_successor_is_handed_over() -> None:
    eps = (_ep("a", age_h=21.5), _ep("b", age_h=9), _ep("s", age_h=0.3, ready_min=10))
    assert _plan(*eps) == [("drain", "a")]


def test_no_drain_before_the_successor_passes_its_grace() -> None:
    # replicas=1: the only other READY replica is the 2-minute-old successor
    policy = replace(SVC, replicas=1)
    eps = (_ep("a", age_h=21.5), _ep("s", age_h=0.1, ready_min=2))
    assert _plan(*eps, policy=policy) == []


def test_no_surge_roll_drains_first_when_another_replica_serves() -> None:
    # bdrt: no walltime, no spare pair (max_surge 0), weekly max_age
    policy = replace(SVC, max_surge=0, max_age_sec=7 * 86400)
    eps = (_ep("a", age_h=24 * 8, ready_min=60 * 24 * 8), _ep("b", age_h=24 * 2))
    assert _plan(*eps, policy=policy, walltime=None) == [("drain", "a")]


def test_min_available_blocks_a_drain_that_would_empty_the_service() -> None:
    policy = replace(SVC, replicas=1, max_surge=0, max_age_sec=86400)
    eps = (_ep("a", age_h=48, ready_min=60 * 48),)
    assert _plan(*eps, policy=policy, walltime=None) == []


@pytest.mark.parametrize(
    ("active", "draining_min", "expected"),
    [(0, 1, [("retire", "a")]), (2, 1, []), (2, 20, [("retire", "a")])],
)
def test_draining_replica_retires_when_drained_or_timed_out(active, draining_min, expected) -> None:
    since = (NOW - timedelta(minutes=draining_min)).isoformat()
    eps = (_ep("a", D, active=active, **{DRAINING_SINCE_KEY: since}), _ep("b"), _ep("c"))
    assert _plan(*eps) == expected


def test_draining_replica_retires_at_the_walltime_floor() -> None:
    since = NOW.isoformat()
    eps = (_ep("a", D, age_h=22.95, active=3, **{DRAINING_SINCE_KEY: since}), _ep("b"), _ep("c"))
    assert _plan(*eps) == [("retire", "a")]


def test_one_replica_drains_at_a_time() -> None:
    since = NOW.isoformat()
    eps = (
        _ep("a", D, active=1, **{DRAINING_SINCE_KEY: since}),
        _ep("b", age_h=21.6),
        _ep("c", age_h=21.5),
    )
    assert _plan(*eps) == []


def test_changed_recipe_rolls_as_an_upgrade() -> None:
    eps = (_ep("a", **{DIGEST_KEY: "old"}), _ep("b", **{DIGEST_KEY: "old"}))
    assert _plan(*eps) == [("provision", None)]


def test_failed_canary_holds_the_upgrade() -> None:
    eps = (
        _ep("a", **{DIGEST_KEY: "old"}),
        _ep("b", **{DIGEST_KEY: "old"}),
        _ep("x", F, ready_min=None),
    )
    assert _plan(*eps) == []


def test_rows_without_a_digest_are_not_upgraded() -> None:
    eps = (_ep("a", **{DIGEST_KEY: ""}), _ep("b", **{DIGEST_KEY: ""}))
    assert _plan(*eps) == []


def test_operator_roll_request_makes_a_replica_due() -> None:
    eps = (_ep("a", **{ROLL_KEY: True}), _ep("b"))
    assert _plan(*eps) == [("provision", None)]


def test_aligned_deadlines_roll_the_oldest_early_once() -> None:
    aligned = (_ep("a", age_h=12.1), _ep("b", age_h=12.0))
    spread = (_ep("a", age_h=12.1), _ep("b", age_h=1.0))
    assert (_plan(*aligned), _plan(*spread)) == ([("provision", None)], [])


def test_surplus_replica_is_trimmed_when_nothing_is_due() -> None:
    eps = (_ep("a", age_h=3, ready_min=170), _ep("b", age_h=2), _ep("c", age_h=1))
    assert _plan(*eps) == [("drain", "a")]


def test_policy_from_a_service_block() -> None:
    entry = {
        "model": "glm-5.3-flash",
        "aliases": ["svc-chat"],
        "env_vars": {"VLLM_TP": "2"},
        "service": {"project_id": "default", "replicas": 2, "max_surge": 0, "max_age": "7-00:00:00"},
    }
    (policy,) = keep_alive_policies([entry])
    facts = (policy.service, policy.replicas, policy.max_surge, policy.min_available, policy.max_age_sec)
    assert facts == (True, 2, 0, 1, 7 * 86400)


def test_digest_ignores_routing_and_lifecycle_but_not_the_recipe() -> None:
    base = {"model": "m", "env_vars": {"A": "1"}, "aliases": ["svc-chat"], "service": {"replicas": 1}}
    same = {**base, "aliases": ["svc-fast"], "service": {"replicas": 3}, "name": "renamed"}
    changed = {**base, "env_vars": {"A": "2"}}
    digests = (entry_digest(base) == entry_digest(same), entry_digest(base) == entry_digest(changed))
    assert digests == (True, False)


def test_keep_alive_entries_keep_the_original_relay() -> None:
    (policy,) = keep_alive_policies([{"model": "m", "keep_alive": {"project_id": "default"}}])
    assert (policy.service, policy.replicas) == (False, 1)


def test_executor_drains_a_handed_over_replica(monkeypatch) -> None:
    from umrs.server import endpoint_relay as runner

    registry = EndpointRegistry()
    for job, age in (("a", 21.5), ("b", 9.0), ("s", 0.3)):
        registry.register(project_id="default", model="glm-5.3-flash", job_id=job, host="h", port=1, backend="slurm")
        row = registry.get_by_job_id(job)
        row.status, row.created_at = R, NOW - timedelta(hours=age)
        row.ready_at = NOW - timedelta(minutes=10 if job == "s" else 50)
        row.metadata[DIGEST_KEY] = "x"
    entry = {"model": "glm-5.3-flash", "service": {"project_id": "default", "replicas": 2}}
    digest = keep_alive_policies([entry])[0].digest
    for job in ("a", "b", "s"):
        registry.get_by_job_id(job).metadata[DIGEST_KEY] = digest

    monkeypatch.setattr(runner, "roster_entries", lambda: [entry])
    monkeypatch.setattr(runner, "get_registry", lambda: registry)
    monkeypatch.setattr(runner, "_site_walltime_sec", lambda: WALLTIME)
    monkeypatch.setattr(runner, "_relay_backend", lambda: None)
    actions = asyncio.run(runner.run_relay_once(now=NOW))
    row = registry.get_by_job_id("a")
    facts = ([(a.action, a.job_id) for a in actions], row.status, row.metadata.get(DRAINING_SINCE_KEY))
    assert facts == ([("drain", "a")], D, NOW.isoformat())


def test_executor_stamps_the_digest_on_a_new_replica(monkeypatch) -> None:
    from umrs.api import endpoints as api
    from umrs.server import endpoint_relay as runner

    registry = EndpointRegistry()
    registry.register(project_id="default", model="glm-5.3-flash", job_id="j-new", host="h", port=1, backend="slurm")

    async def fake_provision_endpoint(**kwargs):
        return api.ProvisionResponse(status="provisioning", job_id="j-new")

    entry = {"model": "glm-5.3-flash", "service": {"project_id": "default", "replicas": 1}}
    monkeypatch.setattr(api, "provision_endpoint", fake_provision_endpoint)
    monkeypatch.setattr(runner, "roster_entries", lambda: [entry])
    monkeypatch.setattr(runner, "get_registry", lambda: _OnlyNew(registry))
    monkeypatch.setattr(runner, "_site_walltime_sec", lambda: WALLTIME)
    monkeypatch.setattr(runner, "_relay_backend", lambda: None)
    asyncio.run(runner.run_relay_once(now=NOW))
    digest = keep_alive_policies([entry])[0].digest
    assert registry.get_by_job_id("j-new").metadata.get(DIGEST_KEY) == digest


class _OnlyNew:
    """The registry, but listing nothing (so the planner sees an empty fleet)."""

    def __init__(self, registry: EndpointRegistry) -> None:
        self._registry = registry

    def list_all(self):
        return []

    def __getattr__(self, name):
        return getattr(self._registry, name)


def _roll_client(registry: EndpointRegistry, admin: bool):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import endpoints as endpoints_api
    from umrs.orchestrator.endpoint_registry import get_registry

    app = FastAPI()
    app.include_router(endpoints_api.router)
    app.dependency_overrides[get_registry] = lambda: registry
    app.dependency_overrides[endpoints_api._caller_is_admin] = lambda: admin
    return TestClient(app)


@pytest.mark.parametrize(("admin", "code", "flagged"), [(True, 200, True), (False, 403, None)])
def test_roll_route_marks_a_replica_for_admins_only(admin, code, flagged) -> None:
    registry = EndpointRegistry()
    registry.register(project_id="default", model="glm-5.3-flash", job_id="a", host="h", port=1, backend="slurm")
    resp = _roll_client(registry, admin).post("/umrs/endpoints/a/roll")
    assert (resp.status_code, registry.get_by_job_id("a").metadata.get(ROLL_KEY)) == (code, flagged)


def test_service_readiness_probe_reaches_the_provision_request(monkeypatch) -> None:
    from umrs.api import endpoints as api
    from umrs.server import endpoint_relay as runner

    seen = []

    async def fake_provision_endpoint(**kwargs):
        seen.append(kwargs["request"].readiness_probe)
        return api.ProvisionResponse(status="provisioning", job_id="j-e")

    entry = {"model": "Qwen3-Embedding-0.6B", "service": {"project_id": "default", "readiness_probe": "models"}}
    registry = EndpointRegistry()
    registry.register(project_id="default", model="Qwen3-Embedding-0.6B", job_id="j-e", host="h", port=1, backend="slurm")
    monkeypatch.setattr(api, "provision_endpoint", fake_provision_endpoint)
    monkeypatch.setattr(runner, "roster_entries", lambda: [entry])
    monkeypatch.setattr(runner, "get_registry", lambda: _OnlyNew(registry))
    monkeypatch.setattr(runner, "_site_walltime_sec", lambda: None)
    monkeypatch.setattr(runner, "_relay_backend", lambda: None)
    asyncio.run(runner.run_relay_once(now=NOW))
    assert seen == ["models"]
