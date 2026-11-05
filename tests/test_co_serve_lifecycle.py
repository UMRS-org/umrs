"""ADR-109 D-2 (V3): one Slurm job, two endpoint rows, one lifecycle.

A co-served job registers its primary (vLLM, LiteLLM port) and one row per
co-serving (BentoML, its own port), both with ``job_group`` = the primary's
job id. Lifecycle acts on the job: drain, failure, teardown and renewal of the
primary move the co-served rows; a co-served row's own probe never touches the
primary; ``r4 endpoint down`` on a co-served row is refused with the primary's
name.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from umrs.orchestrator import co_serve
from umrs.orchestrator.base import EndpointInfo, JobInfo, JobStatus
from umrs.orchestrator.co_serve_lifecycle import (
    probe_co_served,
    register_co_served,
    resume_co_served,
)
from umrs.orchestrator.detection import BackendType
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus
from umrs.orchestrator.janitor import Janitor, JanitorConfig
from umrs.orchestrator.readiness import ReadinessConfig, ReadinessPoller
from umrs.orchestrator.service import OrchestrationService, ProvisionRequest

SAM3 = {
    "name": "sam3.1",
    "model": "SAM3.1",
    "aliases": ["svc-vision2"],
    "model_type": "vision",
    "adapter": "sam3",
}
VLM = "Qwen3.6-35B-A3B-NVFP4"
CO = "4242::sam3.1"


def _primary_metadata(**extra) -> dict:
    return {
        co_serve.CO_SERVE_KEY: [SAM3],
        "allocated_port": 11434,
        "tier": "system",
        "owner_type": "system",
        "access": ["system"],
        **extra,
    }


def _registry_with_job(host: str = "gb10-01", ttl_sec: int | None = 0) -> EndpointRegistry:
    registry = EndpointRegistry()
    primary = registry.register(
        project_id="default",
        model=VLM,
        job_id="4242",
        host=host,
        port=11434,
        backend="slurm-remote",
        ttl_sec=ttl_sec,
        metadata=_primary_metadata(),
    )
    register_co_served(registry, primary, "4242")
    return registry


# ── registration ────────────────────────────────────────────────────────


def test_a_co_served_job_registers_two_rows_with_one_job_group():
    registry = _registry_with_job()
    primary, row = registry.get_by_job_id("4242"), registry.get_by_job_id(CO)
    groups = (
        primary.metadata[co_serve.JOB_GROUP_KEY],
        row.metadata[co_serve.JOB_GROUP_KEY],
        row.metadata[co_serve.CO_SERVED_BY_KEY],
        row.backend_job_id,
    )
    assert groups == ("4242",) * 4
    assert (row.model, row.host, row.port, row.status) == (
        "SAM3.1",
        "gb10-01",
        12434,
        EndpointStatus.PROVISIONING,
    )
    assert registry.co_rows("4242") == [row]


def test_the_co_served_row_is_a_vision_service_shared_like_its_primary():
    row = _registry_with_job().get_by_job_id(CO)
    wanted = {
        "service_kind": "bentoml_vision",
        "model_type": "vision",
        "capabilities": {"supports_vision": True},
        "service_url": "http://gb10-01:12434",
        "access": ["system"],
        "vision_adapter": "sam3",
        "role_probe": "segment",
    }
    assert {key: row.metadata.get(key) for key in wanted} == wanted


def test_a_plain_job_registers_one_row():
    registry = EndpointRegistry()
    primary = registry.register("p", "m", "7", "h", 1, metadata={})
    assert register_co_served(registry, primary, "7") == []
    assert [ep.job_id for ep in registry.list_all()] == ["7"]


def test_the_co_served_row_follows_the_primary_to_its_node():
    registry = _registry_with_job(host="pending")
    assert registry.get_by_job_id(CO).metadata["service_url"] == ""
    registry.update_location("4242", host="gb10-07", port=11434)
    row = registry.get_by_job_id(CO)
    assert (row.host, row.port) == ("gb10-07", 12434)
    assert row.metadata["service_url"] == "http://gb10-07:12434"


# ── the way down follows the job ────────────────────────────────────────


@pytest.mark.parametrize(
    "status",
    [EndpointStatus.DRAINING, EndpointStatus.FAILED, EndpointStatus.TERMINATED],
)
def test_drain_and_failure_of_the_primary_move_the_co_served_row(status):
    registry = _registry_with_job()
    registry.mark_ready(CO)
    registry.update_status("4242", status, "job ended" if status.value == "failed" else None)
    row = registry.get_by_job_id(CO)
    assert row.status == status
    assert "4242" in (row.error_message or "")


def test_a_drained_co_served_row_is_not_brought_back_by_its_probe():
    registry = _registry_with_job()
    registry.mark_draining("4242")
    registry.mark_ready(CO)
    assert registry.get_by_job_id(CO).status == EndpointStatus.DRAINING


def test_a_failed_co_serving_never_touches_the_primary():
    registry = _registry_with_job()
    registry.mark_ready("4242")
    registry.mark_failed(CO, "sidecar never came up")
    assert registry.get_by_job_id("4242").status == EndpointStatus.READY


def test_removing_the_primary_removes_its_co_served_rows():
    registry = _registry_with_job()
    registry.remove("4242")
    assert registry.list_all() == []


def test_renewing_the_primary_renews_the_group():
    registry = _registry_with_job(ttl_sec=600)
    row = registry.get_by_job_id(CO)
    assert row.expires_at == registry.get_by_job_id("4242").expires_at
    later = datetime.utcnow() + timedelta(hours=5)
    registry.extend_ttl("4242", later)
    assert registry.get_by_job_id(CO).expires_at == later


# ── the janitor acts on the job ─────────────────────────────────────────


class _JanitorBackend:
    def __init__(self, status: JobStatus):
        self.status = status
        self.asked: list[str] = []
        self.cleaned: list[str] = []

    async def get_job_status(self, job_id):
        self.asked.append(job_id)
        return SimpleNamespace(status=self.status, state_reason=None)

    async def cleanup(self, job_id, **_kwargs):
        self.cleaned.append(job_id)

    async def list_jobs(self):
        return []


def _janitor(registry, status: JobStatus) -> tuple[Janitor, _JanitorBackend]:
    backend = _JanitorBackend(status)
    return Janitor(config=JanitorConfig(), registry=registry, backend=backend), backend


async def test_janitor_liveness_checks_only_the_job_and_prunes_the_group():
    registry = _registry_with_job()
    registry.mark_ready("4242")
    registry.mark_ready(CO)
    janitor, backend = _janitor(registry, JobStatus.STOPPED)
    assert await janitor._cleanup_ghost_ready() == 1
    assert backend.asked == ["4242"]
    assert registry.list_all() == []


async def test_janitor_renews_the_co_served_row_with_its_job():
    registry = _registry_with_job(ttl_sec=60)
    registry.mark_ready("4242")
    registry.mark_ready(CO)
    janitor, backend = _janitor(registry, JobStatus.RUNNING)
    assert await janitor._auto_extend_ttl() == 1
    assert backend.asked == ["4242"]
    row, primary = registry.get_by_job_id(CO), registry.get_by_job_id("4242")
    assert row.expires_at == primary.expires_at


async def test_janitor_never_reaps_a_co_served_row_as_a_stuck_job():
    registry = _registry_with_job()
    janitor, backend = _janitor(registry, JobStatus.PENDING)
    janitor.config.stuck_timeout_sec = 0
    await janitor._cleanup_stuck()
    assert CO not in backend.asked and CO not in backend.cleaned


# ── provision: one job, two rows, the co-serving's role probe ──────────


class _Backend:
    def __init__(self):
        self.config = SimpleNamespace(startup_timeout_sec=0)
        self.env: dict = {}

    async def submit_job(self, *, env, **_kwargs):
        self.env = dict(env)
        return JobInfo(
            job_id="4242",
            status=JobStatus.RUNNING,
            endpoint=EndpointInfo(node="gb10-01", litellm_port=int(env["LITELLM_PORT"])),
        )

    async def get_job_status(self, _job_id):
        return SimpleNamespace(status=JobStatus.RUNNING, endpoint=None)

    async def get_job_status_before_ready(self, _job_id):
        return SimpleNamespace(status=JobStatus.RUNNING, endpoint=None)

    async def cleanup(self, _job_id, **_kwargs):
        pass


class _Ports:
    def allocate(self, _job_id, _host):
        return 11434

    def release(self, _port, _host):
        pass


def _service(registry=None) -> tuple[OrchestrationService, _Backend]:
    backend = _Backend()
    service = OrchestrationService(
        registry=registry or EndpointRegistry(),
        port_manager=_Ports(),
        readiness_poller=ReadinessPoller(
            ReadinessConfig(timeout_sec=5, liveness_interval_sec=0.0)
        ),
        backend=backend,
    )
    service._backend_type = BackendType.SLURM
    return service, backend


async def test_provision_submits_one_job_and_registers_both_rows(monkeypatch):
    service, backend = _service()
    monkeypatch.setattr(service, "_determine_target_host", lambda: "gb10-01")

    async def finalize_later(**_kwargs):
        return None

    monkeypatch.setattr(service, "_finalize_provision", finalize_later)
    result = await service.provision(
        ProvisionRequest(
            project_id="default",
            model=VLM,
            ttl_sec=0,
            prefer_warm=False,
            wait_for_ready=False,
            metadata={co_serve.CO_SERVE_KEY: [SAM3]},
        )
    )
    assert result.success and result.job_id == "4242"
    assert backend.env["BENTOML_VISION_DOCKER_PORT"] == "12434"
    assert backend.env["LITELLM_PORT"] == "11434"
    rows = {ep.job_id: ep for ep in service.registry.list_all()}
    assert set(rows) == {"4242", CO}
    assert rows[CO].port == 12434 and rows[CO].expires_at is None
    await asyncio.sleep(0)  # let the (stubbed) background finalize run


def _probe_answers(monkeypatch, *, segment_ok: bool):
    seen: list[tuple[int, str]] = []

    async def check_once(self, host, port, *, probe_model=None, probe_kind="models", probe_path=None):
        seen.append((port, probe_kind))
        if probe_kind == "segment" and not segment_ok:
            return False, True, [], "Segment probe returned no mask"
        return True, False, [str(probe_model)], ""

    async def no_sleep(self, *_a, **_k):
        return None

    monkeypatch.setattr(ReadinessPoller, "_check_once", check_once)
    monkeypatch.setattr(ReadinessPoller, "_sleep_before_next_attempt", no_sleep)
    return seen


async def test_a_ready_job_runs_its_co_servings_role_probe(monkeypatch):
    seen = _probe_answers(monkeypatch, segment_ok=True)
    service, _ = _service(_registry_with_job())
    service.registry.mark_ready("4242")
    assert await probe_co_served(service, "4242") == 1
    assert seen == [(12434, "segment")]
    assert service.registry.get_by_job_id(CO).status == EndpointStatus.READY


async def test_the_primary_turning_ready_starts_the_role_probe(monkeypatch):
    seen = _probe_answers(monkeypatch, segment_ok=True)
    service, _ = _service(_registry_with_job())
    loop = asyncio.get_running_loop()
    result = await service._finalize_provision(
        request=ProvisionRequest("default", VLM, metadata=_primary_metadata()),
        job_id="4242",
        target_host="gb10-01",
        actual_host="gb10-01",
        actual_port=11434,
        deadline=loop.time() + 60,
        progress=None,
        start_time=loop.time(),
    )
    assert result.success
    await asyncio.gather(*service._resume_tasks)
    assert seen == [(11434, "chat"), (12434, "segment")]
    assert service.registry.get_by_job_id(CO).status == EndpointStatus.READY


async def test_a_failed_role_probe_fails_only_the_co_served_row(monkeypatch):
    _probe_answers(monkeypatch, segment_ok=False)
    monkeypatch.setenv("UMRS_CO_SERVE_READY_TIMEOUT_SEC", "1")
    service, _ = _service(_registry_with_job())
    service.registry.mark_ready("4242")
    assert await asyncio.wait_for(probe_co_served(service, "4242"), 10) == 0
    assert service.registry.get_by_job_id(CO).status == EndpointStatus.FAILED
    assert service.registry.get_by_job_id("4242").status == EndpointStatus.READY


async def test_co_served_rows_wait_for_their_primary(monkeypatch):
    seen = _probe_answers(monkeypatch, segment_ok=True)
    service, _ = _service(_registry_with_job())
    assert await probe_co_served(service, "4242") == 0
    assert seen == []
    assert service._unfinished_rows() == [service.registry.get_by_job_id("4242")]


async def test_a_restart_resumes_the_probe_of_a_ready_job(monkeypatch):
    _probe_answers(monkeypatch, segment_ok=True)
    service, _ = _service(_registry_with_job())
    service.registry.mark_ready("4242")
    assert resume_co_served(service) == ["4242"]
    await asyncio.gather(*service._resume_tasks)
    assert service.registry.get_by_job_id(CO).status == EndpointStatus.READY


async def test_backend_liveness_of_a_co_served_row_asks_for_its_job():
    from umrs.api.endpoints import _backend_job_liveness, _is_backend_job_alive

    service, backend = _service()
    asked: list[str] = []

    async def status(job_id):
        asked.append(job_id)
        return SimpleNamespace(status=JobStatus.RUNNING)

    backend.get_job_status = status
    assert await _backend_job_liveness(service, CO) is True
    assert await _is_backend_job_alive(service, CO) is True
    assert await service._verify_backend_job_alive(CO) is True
    assert asked == ["4242", "4242", "4242"]


# ── r4 endpoint down / roll on a co-served row ──────────────────────────


def test_stopping_a_co_served_row_is_refused_with_the_primary_named():
    from fastapi import HTTPException
    from umrs.api.endpoints import _refuse_co_served

    registry = _registry_with_job()
    with pytest.raises(HTTPException) as caught:
        _refuse_co_served(registry.get_by_job_id(CO), "stop")
    assert caught.value.status_code == 409
    assert "4242" in str(caught.value.detail)
    _refuse_co_served(registry.get_by_job_id("4242"), "stop")  # the primary may


async def test_deprovision_route_refuses_a_co_served_row(monkeypatch):
    from fastapi import HTTPException
    from umrs.api import endpoints

    registry = _registry_with_job()
    monkeypatch.setattr(endpoints, "_require_identity_for_mutation", lambda *_a: None)
    with pytest.raises(HTTPException) as caught:
        await endpoints.deprovision_endpoint(
            job_id=CO, force=False, project_id="default", registry=registry
        )
    assert caught.value.status_code == 409
    assert registry.get_by_job_id("4242") is not None
    with pytest.raises(HTTPException):
        await endpoints.roll_endpoint(job_id=CO, is_admin=True, registry=registry)


def test_an_endpoint_refresh_holds_a_co_served_row_to_its_role_probe():
    """GET /endpoints/{job} refreshes a starting row: a co-served one turns
    READY only on its segmentation, never on /readyz alone."""
    from umrs.api.endpoints import _refresh_probe

    registry = _registry_with_job()
    assert _refresh_probe(registry.get_by_job_id(CO)) == ("segment", "sam3", "/readyz")
    assert _refresh_probe(registry.get_by_job_id("4242"))[:2] == ("chat", VLM)
