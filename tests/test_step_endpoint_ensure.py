"""ADR-109 D-4: a trigger starts the endpoints its steps need; steps wait.

Covers the trigger-time ensure (user tier vs the relay for service models,
dedupe on a starting row, steps that are left alone), the router's ADR-078
sharing and its "starting = wait" answer, the runner's bounded wait, and the
relay's on-demand (``replicas: 0``) services.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime
from typing import Any

import pytest
from fastapi.responses import JSONResponse
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus
from umrs.orchestrator.endpoint_relay import KeepAlivePolicy, keep_alive_policies
from umrs.orchestrator.service_controller import DIGEST_KEY, plan_service
from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.pipeline_sdk.workflow import StepSpec
from umrs.server import endpoint_relay as relay
from umrs.server import step_endpoint_ensure as ensure
from umrs.server import step_router as sr
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_runner import run_workflow_record

RUN = "wf-abc123"
VISION = {
    "name": "hmae",
    "scope": "system",
    "model_type": "vision",
    "capabilities": {"supports_vision": True},
}
SAM3 = {
    "name": "SAM3.1",
    "scope": "system",
    "model_type": "vision",
    "capabilities": {"supports_vision": True},
}
SVC_ENTRY = {
    "name": "sam3.1",
    "model": "SAM3.1",
    "aliases": ["svc-vision2"],
    "service": {"project_id": "platform", "user": "svcuser", "replicas": 0},
    "access": ["system"],
}
NOW = datetime(2026, 11, 4, 12, 0, 0)


class _Store:
    def __init__(self, entries: list[dict[str, Any]]):
        self._entries = entries

    def list_models(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return [types.SimpleNamespace(to_dict=lambda e=e: dict(e)) for e in self._entries]


@pytest.fixture
def registry(monkeypatch) -> EndpointRegistry:
    """One in-memory registry behind every lookup the ensure path makes.

    ``umrs.api.endpoints`` is imported first: its routes bind
    ``Depends(get_registry)`` at import, and a first import under this patch
    would leave the fake behind for every later test.
    """
    import umrs.api.endpoints  # noqa: F401

    reg = EndpointRegistry()
    monkeypatch.setattr("umrs.orchestrator.endpoint_registry.get_registry", lambda: reg)
    monkeypatch.setattr(relay, "get_registry", lambda: reg)
    return reg


@pytest.fixture
def catalog(monkeypatch):
    """The model registry the router reads (as admin, as in production)."""

    def install(*entries: dict[str, Any]) -> None:
        monkeypatch.setattr("umrs.api.models.get_model_store", lambda: _Store(list(entries)))

    install(VISION)
    return install


@pytest.fixture
def roster(monkeypatch):
    """The provision roster the relay and the alias lookup read."""

    def install(*entries: dict[str, Any]) -> None:
        def alias_models(alias: str) -> tuple[str, ...]:
            return tuple(e["model"] for e in entries if alias in (e.get("aliases") or ()))

        monkeypatch.setattr(relay, "roster_entries", lambda: list(entries))
        monkeypatch.setattr(relay, "role_alias_models", alias_models)
        monkeypatch.setattr(sr, "_role_alias_models", alias_models)

    install()
    return install


@pytest.fixture
def provisions(monkeypatch, registry):
    """Fake ``provision_endpoint``: registers a PROVISIONING row and answers
    the 202 JSONResponse a ``wait=False`` submit answers in production."""
    from umrs.api import endpoints as api

    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> Any:
        calls.append(kwargs)
        request = kwargs["request"]
        job = f"job-{len(calls)}"
        registry.register(
            project_id=kwargs["project_id"],
            model=request.model,
            job_id=job,
            host="node",
            port=8000,
            backend="slurm",
            metadata={"access": list(request.access or [])},
        )
        body = {"status": "provisioning", "job_id": job, "retry_after": 30}
        return JSONResponse(content=body, status_code=202)

    monkeypatch.setattr(api, "provision_endpoint", fake)
    return calls


def _router(project: str = "spring-8", user: str = "alice") -> sr.StepRouter:
    return sr.build_registry_backed_router(project_id=project, user_id=user)


def _ensure(steps: list[dict[str, Any]], project: str = "spring-8") -> list[dict[str, Any]]:
    return ensure.ensure_step_endpoints(
        steps, router=_router(project), project_id=project, user_id="alice", execution_id=RUN
    )


def _vision_step(name: str = "seg", **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "entrypoint": "m:C",
        "requires": ["vision"],
        "model_type": "vision",
        **extra,
    }


def _ready(reg: EndpointRegistry, job: str, *, project: str, model: str, access=None) -> None:
    reg.register(
        project_id=project,
        model=model,
        job_id=job,
        host="node",
        port=4000,
        backend="slurm",
        ttl_sec=0,
        metadata={"access": access} if access else None,
    )
    reg.get_by_job_id(job).status = EndpointStatus.READY


# ── (a) a step whose model has no endpoint: one user-tier provision ────────


def test_missing_endpoint_is_provisioned_once_as_the_user(roster, catalog, provisions, registry):
    results = _ensure([_vision_step("a"), _vision_step("b")])

    (call,) = provisions
    request = call["request"]
    assert (request.model, request.tier, request.ttl_sec, request.wait, request.access) == (
        "hmae",
        "user",
        0,
        False,
        ["project:spring-8"],
    )
    assert (call["project_id"], call["requesting_user"], call["is_admin"]) == (
        "spring-8",
        "alice",
        False,
    )
    assert [(r["via"], r["outcome"], r["job_id"]) for r in results] == [
        ("user", "provisioned", "job-1")
    ]
    assert registry.get_by_job_id("job-1").metadata[ensure.STARTED_BY_RUN_KEY] == RUN


def test_a_refused_provision_is_recorded_not_raised(roster, catalog, monkeypatch, registry):
    from umrs.api import endpoints as api

    async def refuse(**_kwargs: Any) -> Any:
        raise PermissionError("Model 'hmae' is user-scope and not provisionable by this caller")

    monkeypatch.setattr(api, "provision_endpoint", refuse)
    (row,) = _ensure([_vision_step()])
    assert (row["outcome"], "not provisionable" in row["error"]) == ("refused", True)


# ── (b) a service alias goes to the relay, never to a user-tier provision ─


def test_service_alias_asks_the_relay_for_a_replica(roster, catalog, provisions, registry):
    roster(SVC_ENTRY)
    catalog(VISION, SAM3)

    (row,) = _ensure([_vision_step(model="svc-vision2")])

    (call,) = provisions
    request = call["request"]
    assert (request.model, request.tier, request.wait) == ("SAM3.1", "system", False)
    assert (call["project_id"], call["requesting_user"], call["is_admin"]) == (
        "platform",
        "svcuser",
        True,
    )
    assert (row["via"], row["outcome"], row["job_id"]) == ("service", "provisioned", "job-1")
    metadata = registry.get_by_job_id("job-1").metadata
    assert metadata[ensure.STARTED_BY_RUN_KEY] == RUN
    assert metadata[DIGEST_KEY] == keep_alive_policies([SVC_ENTRY])[0].digest


def test_alias_without_a_service_entry_starts_nothing(roster, catalog, provisions):
    roster({"model": "SAM3.1", "aliases": ["svc-vision2"]})  # no service block
    catalog(VISION, SAM3)

    (row,) = _ensure([_vision_step(model="svc-vision2")])

    assert (provisions, row["via"], row["outcome"]) == ([], "none", "no_service")


def test_a_service_the_user_may_not_use_is_not_started(roster, catalog, provisions):
    roster({**SVC_ENTRY, "access": ["project:platform"]})
    catalog(VISION, SAM3)

    (row,) = _ensure([_vision_step(model="svc-vision2")])

    assert (provisions, row["via"], row["outcome"]) == ([], "service", "forbidden")


# ── (c) dedupe on an existing starting row ─────────────────────────────────


def test_existing_provisioning_row_is_reused(roster, catalog, provisions, registry):
    registry.register(project_id="spring-8", model="hmae", job_id="j0", host="n", port=1)

    assert (_ensure([_vision_step()]), provisions) == ([], [])


def test_a_second_trigger_sees_the_first_triggers_row(roster, catalog, provisions):
    _ensure([_vision_step()])
    second = _ensure([_vision_step()])

    assert (len(provisions), second) == (1, [])


def test_a_shared_service_replica_starting_elsewhere_counts(roster, catalog, provisions, registry):
    roster(SVC_ENTRY)
    catalog(VISION, SAM3)
    registry.register(
        project_id="platform",
        model="SAM3.1",
        job_id="s1",
        host="n",
        port=1,
        metadata={"access": ["system"]},
    )

    assert (_ensure([_vision_step(model="svc-vision2")]), provisions) == ([], [])


# ── (d) the router honours ADR-078 sharing ─────────────────────────────────


def test_router_routes_to_an_endpoint_shared_by_another_project(catalog, registry, monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    _ready(registry, "sys-1", project="platform", model="hmae", access=["system"])

    decision = _router().resolve(StepSpec.from_dict(_vision_step()), project_id="spring-8")

    assert (decision.resolved_via, decision.endpoint_id) == ("registry_capability", "sys-1")


def test_router_does_not_route_to_an_unshared_endpoint(catalog, registry, monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    _ready(registry, "priv-1", project="other", model="hmae")

    with pytest.raises(sr.StepRoutingError, match="no matching active endpoint"):
        _router().resolve(StepSpec.from_dict(_vision_step()), project_id="spring-8")


def test_router_answers_pending_for_a_starting_endpoint(catalog, registry, monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    registry.register(project_id="spring-8", model="hmae", job_id="p1", host="n", port=1)

    with pytest.raises(sr.StepEndpointPending) as raised:
        _router().resolve(StepSpec.from_dict(_vision_step()), project_id="spring-8")
    assert (raised.value.model, raised.value.status) == ("hmae", "provisioning")


# ── (e)/(f) the runner waits, then proceeds or fails by name ───────────────


class _Flip:
    """Router callbacks whose endpoint turns READY after *after* polls."""

    def __init__(self, after: int):
        self.after = after
        self.polls = 0

    def find_models(self, *_args: Any) -> list[dict[str, Any]]:
        return [VISION]

    def find_endpoint(self, *_args: Any) -> dict[str, Any] | None:
        self.polls += 1
        if self.polls > self.after:
            return {"litellm_url": "http://vision", "job_id": "ep-1"}
        return None

    def find_pending(self, *_args: Any) -> dict[str, Any] | None:
        return {"status": "starting"}

    def router(self) -> sr.StepRouter:
        return sr.StepRouter(
            find_models=self.find_models,
            find_endpoint=self.find_endpoint,
            find_pending=self.find_pending,
        )


class _Seg(Step):
    name = "seg"

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(extra={"api_base": ctx.config.get("api_base")})


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ROUTER_MOCK", raising=False)
    monkeypatch.setenv("UMRS_ROUTER_UCB_WEIGHT", "0")
    monkeypatch.setenv("UMRS_STEP_ENDPOINT_POLL_SEC", "0.01")
    mod = types.ModuleType("_ensure_probe")
    mod._Seg = _Seg  # type: ignore[attr-defined]
    sys.modules["_ensure_probe"] = mod
    reset_execution_registry_for_tests()
    yield
    sys.modules.pop("_ensure_probe", None)
    reset_execution_registry_for_tests()


def _run(router: sr.StepRouter, tmp_path, prelaunch=None):
    record = get_execution_registry().create(workflow_name="w", user_id="alice")
    step = _vision_step(entrypoint="_ensure_probe:_Seg")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={"name": "w", "steps": [step]},
        router=router,
        prelaunch=prelaunch,
    )
    return record


def test_runner_waits_for_a_starting_endpoint_then_runs(probe, tmp_path):
    flip = _Flip(after=3)
    seen: list[str] = []

    record = _run(flip.router(), tmp_path, prelaunch=lambda rec: seen.append(rec.execution_id))

    assert record.status == "succeeded", record.error
    assert (record.results["_routing"]["seg"]["api_base"], flip.polls, seen) == (
        "http://vision",
        4,
        [record.execution_id],
    )


def test_runner_fails_by_name_when_the_endpoint_never_turns_ready(probe, tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_STEP_ENDPOINT_WAIT_SEC", "0.05")

    record = _run(_Flip(after=10**9).router(), tmp_path)

    assert record.status == "failed"
    assert "endpoint for hmae not READY within 0 s (last status: starting)" in str(record.error)


def test_wait_is_bounded_by_the_clock_not_the_poll_count(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_ENDPOINT_WAIT_SEC", "2700")
    monkeypatch.setenv("UMRS_STEP_ENDPOINT_POLL_SEC", "15")
    now = [0.0]
    sleeps: list[float] = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        now[0] += sec

    with pytest.raises(sr.StepRoutingError, match="not READY within 2700 s"):
        sr.resolve_waiting(
            _Flip(after=10**9).router(),
            StepSpec.from_dict(_vision_step()),
            clock=lambda: now[0],
            sleep=sleep,
        )
    assert (len(sleeps), sum(sleeps)) == (180, 2700)


def test_a_cancelled_run_stops_waiting(monkeypatch):
    with pytest.raises(sr.StepRoutingError, match="cancelled while waiting"):
        sr.resolve_waiting(
            _Flip(after=10**9).router(), StepSpec.from_dict(_vision_step()), cancelled=lambda: True
        )


def test_a_prelaunch_error_does_not_fail_the_run(probe, tmp_path):
    def boom(_record: Any) -> None:
        raise RuntimeError("relay unreachable")

    assert _run(_Flip(after=0).router(), tmp_path, prelaunch=boom).status == "succeeded"


# ── (g) steps the ensure leaves alone ──────────────────────────────────────


@pytest.mark.parametrize(
    "step",
    [
        _vision_step(config={"api_base": "http://pinned"}),  # explicit upstream
        {"name": "plain", "entrypoint": "m:C"},  # no model requirement
        {"name": "gate", "gate": True, "requires": ["vision"]},  # manual gate
        {"name": "ext", "entrypoint": "m:C", "model": "gpt-x"},  # model not in the registry
    ],
)
def test_steps_without_a_registry_route_are_untouched(step, roster, catalog, provisions):
    assert (_ensure([step]), provisions) == ([], [])


def test_ambiguous_unpinned_step_starts_nothing(roster, catalog, provisions):
    catalog(VISION, SAM3)
    assert (_ensure([_vision_step()]), provisions) == ([], [])


def test_live_own_endpoint_needs_nothing(roster, catalog, provisions, registry):
    _ready(registry, "own-1", project="spring-8", model="hmae")
    assert (_ensure([_vision_step()]), provisions) == ([], [])


# ── the handler gates the ensure on a verified project ─────────────────────


@pytest.mark.parametrize(
    ("project", "user", "env", "armed"),
    [
        ("spring-8", "alice", None, True),
        (None, "alice", None, False),  # unverified claim / default bucket
        ("spring-8", "", None, False),
        ("spring-8", "alice", "false", False),
    ],
)
def test_prelaunch_needs_a_verified_project(project, user, env, armed, monkeypatch):
    from umrs.server.workflow_registry_handler import _endpoint_prelaunch

    if env is not None:
        monkeypatch.setenv("UMRS_STEP_ENDPOINT_ENSURE", env)
    hook = _endpoint_prelaunch({"steps": []}, _router(), user, project)
    assert (hook is not None) == armed


def test_prelaunch_records_what_it_started(roster, catalog, provisions):
    from umrs.server.workflow_registry_handler import (
        ENDPOINT_ENSURE_RESULTS_KEY,
        _endpoint_prelaunch,
    )

    record = types.SimpleNamespace(execution_id=RUN, results={})
    _endpoint_prelaunch({"steps": [_vision_step()]}, _router(), "alice", "spring-8")(record)

    assert [r["job_id"] for r in record.results[ENDPOINT_ENSURE_RESULTS_KEY]] == ["job-1"]


# ── relay: on-demand services and the 202 job id ───────────────────────────


def test_replicas_zero_is_an_on_demand_service():
    (policy,) = keep_alive_policies([SVC_ENTRY])
    assert (policy.service, policy.replicas, policy.min_available) == (True, 0, 0)


def test_controller_never_respawns_rolls_or_trims_an_on_demand_service(registry):
    (policy,) = keep_alive_policies([SVC_ENTRY])
    assert plan_service(policy, [], walltime_sec=23 * 3600, now=NOW) == []
    _ready(registry, "r1", project="platform", model="SAM3.1")
    row = registry.get_by_job_id("r1")
    row.created_at = row.ready_at = datetime(2026, 11, 3, 13, 30)  # 30 min to walltime
    assert plan_service(policy, [row], walltime_sec=23 * 3600, now=NOW) == []


def test_ensure_service_replica_is_a_no_op_while_one_runs(roster, registry, provisions):
    roster(SVC_ENTRY)
    _ready(registry, "r1", project="platform", model="SAM3.1")

    answer = asyncio.run(relay.ensure_service_replica("svc-vision2", reason="t"))

    assert (answer["outcome"], provisions) == ("present", [])


def test_relay_reads_the_job_id_of_a_202_answer(roster, registry, provisions, monkeypatch):
    """Since #720 (wait=False) the handler answers a JSONResponse: the relay
    read ``result.job_id`` (None) and never recorded the replica's digest."""
    policy = KeepAlivePolicy(model="glm", project_id="default", service=True, digest="d1")
    action = relay.RelayAction("provision", "glm", "default", "respawn")

    job = asyncio.run(relay._provision(policy, action, NOW))

    assert (job, registry.get_by_job_id("job-1").metadata.get(DIGEST_KEY)) == ("job-1", "d1")


def test_provision_result_reads_both_answer_shapes():
    from umrs.api.endpoints import ProvisionResponse

    accepted = JSONResponse(content={"status": "provisioning", "job_id": "j1"}, status_code=202)
    ready = ProvisionResponse(status="ready", job_id="j2")
    assert [relay.provision_result(r)["job_id"] for r in (accepted, ready)] == ["j1", "j2"]
    assert json.loads(accepted.body)["status"] == "provisioning"
