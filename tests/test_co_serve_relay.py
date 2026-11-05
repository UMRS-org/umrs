"""ADR-109 D-2 (V3): the replica controller and co-served rows.

A co-served alias counts as served only while its own row is READY. A
replica whose co-serving never becomes READY is rolled like any failed
replica; a retry whose co-serving fails too stops the rolling (the primary
keeps serving).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from umrs.orchestrator.co_serve import CO_SERVED_BY_KEY
from umrs.orchestrator.endpoint_registry import EndpointStatus, RegisteredEndpoint
from umrs.orchestrator.endpoint_relay import KeepAlivePolicy, RelayAction, plan_relay
from umrs.orchestrator.service_controller import CO_RETRY_KEY, DIGEST_KEY
from umrs.server.endpoint_relay import replica_metadata

NOW = datetime(2026, 11, 4, 12, 0, 0)
WALLTIME = 23 * 3600
R, P, F, D = (
    EndpointStatus.READY,
    EndpointStatus.PROVISIONING,
    EndpointStatus.FAILED,
    EndpointStatus.DRAINING,
)
VLM = "Qwen3.6-35B-A3B-NVFP4"
SVC = KeepAlivePolicy(
    model=VLM,
    project_id="default",
    service=True,
    replicas=1,
    min_available=1,
    max_surge=1,
    digest="d",
    co_serve=("SAM3.1",),
    co_serve_grace_sec=900,
)


def _vlm(job, status=R, *, ready_min=50, **meta):
    return RegisteredEndpoint(
        project_id="default",
        model=VLM,
        job_id=job,
        host="h",
        port=11434,
        status=status,
        created_at=NOW - timedelta(hours=1),
        ready_at=NOW - timedelta(minutes=ready_min) if ready_min is not None else None,
        metadata={DIGEST_KEY: "d", **meta},
    )


def _sam(job, status=R, error=None):
    return RegisteredEndpoint(
        project_id="default",
        model="SAM3.1",
        job_id=f"{job}::sam3.1",
        host="h",
        port=12434,
        status=status,
        created_at=NOW - timedelta(hours=1),
        error_message=error,
        metadata={CO_SERVED_BY_KEY: job, "job_group": job},
    )


def _plan(*rows, policy=SVC):
    return plan_relay(policy, list(rows), walltime_sec=WALLTIME, now=NOW)


def _kinds(actions):
    return [(a.action, a.job_id, a.co_retry) for a in actions]


def test_a_replica_with_a_ready_co_serving_needs_nothing():
    assert _plan(_vlm("a"), _sam("a")) == []


def test_a_failed_co_serving_rolls_the_replica_with_a_retry_surge():
    actions = _plan(_vlm("a"), _sam("a", F, "no mask"))
    assert _kinds(actions) == [("provision", None, True)]
    assert "co-served SAM3.1 failed: no mask" in actions[0].reason


def test_a_loading_co_serving_is_waited_for_within_the_grace():
    assert _plan(_vlm("a", ready_min=5), _sam("a", P)) == []
    assert _plan(_vlm("a", ready_min=5)) == []


def test_a_co_serving_not_ready_after_the_grace_rolls_the_replica():
    for rows in ([_vlm("a", ready_min=20), _sam("a", P)], [_vlm("a", ready_min=20)]):
        actions = _plan(*rows)
        assert _kinds(actions) == [("provision", None, True)]
        assert "not READY" in actions[0].reason


def test_the_old_replica_drains_once_a_healthy_successor_serves_both():
    rows = [
        _vlm("a", ready_min=50),
        _sam("a", F),
        _vlm("b", ready_min=10, **{CO_RETRY_KEY: True}),
        _sam("b"),
    ]
    assert _kinds(_plan(*rows)) == [("drain", "a", False)]


def test_a_retry_whose_co_serving_fails_too_holds_the_roll():
    rows = [
        _vlm("a", ready_min=50),
        _sam("a", F),
        _vlm("b", ready_min=10, **{CO_RETRY_KEY: True}),
        _sam("b", F),
    ]
    # Co-serving no longer rolls anything; the surplus replica is trimmed.
    assert _kinds(_plan(*rows)) == [("drain", "a", False)]
    assert _plan(_vlm("b", ready_min=10, **{CO_RETRY_KEY: True}), _sam("b", F)) == []


def test_co_served_rows_of_other_jobs_and_drained_jobs_do_not_count():
    rows = [_vlm("a"), _sam("zzz", F), _sam("a")]
    assert _plan(*rows) == []
    # A co-served row the job's drain took along is no reason for anything.
    assert _plan(_vlm("a", D), _sam("a", D), _vlm("b"), _sam("b")) == [
        RelayAction("retire", VLM, "default", "drained", "a")
    ]


def test_without_co_serve_the_controller_ignores_sidecar_rows():
    plain = KeepAlivePolicy(
        model=VLM, project_id="default", service=True, replicas=1, digest="d"
    )
    assert _plan(_vlm("a"), _sam("a", F), policy=plain) == []


def test_the_executor_marks_a_retry_surge():
    retry = RelayAction("provision", VLM, "default", "surge", co_retry=True)
    assert replica_metadata(SVC, retry) == {DIGEST_KEY: "d", CO_RETRY_KEY: True}
    plain = RelayAction("provision", VLM, "default", "respawn")
    assert replica_metadata(SVC, plain) == {DIGEST_KEY: "d"}


# ── ADR-109 D-4 trigger-time ensure meets D-2 co-serving ────────────────

CO_ENTRY = {
    "name": "qwen3.6-35b-a3b-nvfp4",
    "model": VLM,
    "aliases": ["svc-vision"],
    "service": {"project_id": "default", "user": "mnagaso", "replicas": 1},
    "co_serve": [
        {
            "name": "sam3.1",
            "model": "SAM3.1",
            "aliases": ["svc-vision2"],
            "model_type": "vision",
            "adapter": "sam3",
        }
    ],
}


def _co_roster(monkeypatch):
    from umrs.orchestrator.endpoint_registry import EndpointRegistry
    from umrs.orchestrator.provision_roster import _alias_models
    from umrs.server import endpoint_relay as relay

    registry = EndpointRegistry()

    def alias_models(alias: str) -> tuple[str, ...]:
        return tuple(_alias_models(CO_ENTRY, alias))

    monkeypatch.setattr(relay, "roster_entries", lambda: [CO_ENTRY])
    monkeypatch.setattr(relay, "role_alias_models", alias_models)
    monkeypatch.setattr(relay, "get_registry", lambda: registry)
    return relay, registry


def test_a_co_served_alias_is_served_by_its_primarys_policy(monkeypatch):
    relay, _registry = _co_roster(monkeypatch)
    (policy,) = relay.service_policies_for("svc-vision2")
    assert (policy.model, policy.co_serve) == (VLM, ("SAM3.1",))


async def test_a_trigger_for_svc_vision2_counts_the_running_primary(monkeypatch):
    relay, registry = _co_roster(monkeypatch)
    registry.register("default", VLM, "a", "h", 11434, ttl_sec=0)
    registry.mark_ready("a")
    answer = await relay.ensure_service_replica("svc-vision2", reason="t")
    assert (answer["outcome"], answer["model"]) == ("present", VLM)
