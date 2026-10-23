"""Failing re-provision cap (2026-10-23).

bdrt 2026-10-20..22: job 8218 held dgx-spark-0's GPU behind a dead LiteLLM
and every new request for the same model submitted a fresh job onto that node;
vLLM could not start, the job failed, and the next request did the same: 49
FAILED jobs. Pinned here: each finished attempt is recorded in the audit
trail; after three failures of one (project, model, variant, venue) inside 30
min, POST /provision answers a typed 429 naming the failures and the retry
time and submits nothing; a success ends the run; an admin may override.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import audit_events
from umrs.api import endpoints as endpoints_api
from umrs.api.endpoints import router
from umrs.orchestrator import provision_failure_cap as cap
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry
from umrs.orchestrator.readiness import ReadinessResult, ReadinessState
from umrs.orchestrator.service import (
    OrchestrationService,
    ProvisionRequest,
    ProvisionResult,
)

VENUE = "bdrt"
KEY = cap.AttemptKey("default", "Qwen3.8-27B-NVFP4", None, VENUE)


@pytest.fixture(autouse=True)
def _ring_buffer_only(monkeypatch):
    """The in-process audit buffer, empty, and no persistent store."""
    monkeypatch.setattr(audit_events, "_audit_store", None)
    monkeypatch.setattr(audit_events, "_audit_buffer", [])
    monkeypatch.setenv("UMRS_CLUSTER", VENUE)
    for name in ("UMRS_PROVISION_FAILURE_CAP", "UMRS_PROVISION_FAILURE_WINDOW_SEC"):
        monkeypatch.delenv(name, raising=False)


def _row(job_id: str, ts: float, ok: bool = False, key: cap.AttemptKey = KEY) -> dict:
    details = key.details()
    details.update({"job_id": job_id, "error": "" if ok else "Backend job ended (FAILED)"})
    return {"ts": ts, "outcome": "success" if ok else "error", "details": details}


def _record_failures(*job_ids: str, key: cap.AttemptKey = KEY) -> None:
    for job_id in job_ids:
        cap.record_attempt(key, job_id=job_id, ok=False, error="Backend job ended (FAILED)")


# ── the policy, on attempt rows ─────────────────────────────────────────────


NOW = 10_000.0
THREE_FAILURES = [_row("8290", NOW - 1500), _row("8291", NOW - 1000), _row("8292", NOW - 500)]


def test_three_failures_inside_the_window_refuse_until_the_oldest_ages_out() -> None:
    refusal = cap.evaluate(THREE_FAILURES, KEY, cap.CapPolicy(), NOW)
    assert refusal is not None
    assert [f.job_id for f in refusal.failures] == ["8292", "8291", "8290"]
    assert refusal.retry_at == NOW - 1500 + 1800
    assert refusal.retry_after_sec(NOW) == 300


def test_the_refusal_message_names_the_jobs_the_wait_and_the_override() -> None:
    refusal = cap.evaluate(THREE_FAILURES, KEY, cap.CapPolicy(), NOW)
    assert refusal is not None
    message = refusal.message(NOW)
    for part in ("job 8290", "job 8291", "job 8292", "in 300 s", "override_failure_cap"):
        assert part in message


def test_a_success_between_failures_ends_the_run() -> None:
    now = 10_000.0
    rows = [
        _row("1", now - 900),
        _row("2", now - 800, ok=True),
        _row("3", now - 700),
        _row("4", now - 600),
    ]
    assert cap.evaluate(rows, KEY, cap.CapPolicy(), now) is None


def test_failures_older_than_the_window_do_not_count() -> None:
    now = 10_000.0
    rows = [_row("1", now - 4000), _row("2", now - 500), _row("3", now - 400)]
    assert cap.evaluate(rows, KEY, cap.CapPolicy(), now) is None


def test_a_job_recorded_twice_counts_once() -> None:
    now = 10_000.0
    rows = [_row("1", now - 300), _row("1", now - 299), _row("2", now - 200)]
    assert cap.evaluate(rows, KEY, cap.CapPolicy(), now) is None


def test_the_env_tunes_and_zero_disables(monkeypatch) -> None:
    monkeypatch.setenv("UMRS_PROVISION_FAILURE_CAP", "0")
    _record_failures("1", "2", "3")
    assert cap.check(KEY) is None
    monkeypatch.setenv("UMRS_PROVISION_FAILURE_CAP", "2")
    assert cap.check(KEY) is not None


# ── the audit trail as the store ────────────────────────────────────────────


def test_recorded_failures_of_this_key_trip_the_cap() -> None:
    _record_failures("8290", "8291", "8292")
    refusal = cap.check(KEY)
    assert refusal is not None
    assert [f.job_id for f in refusal.failures] == ["8292", "8291", "8290"]


@pytest.mark.parametrize(
    "other",
    [
        cap.AttemptKey("default", "Qwen3.8-27B-NVFP4", None, "rccs"),
        cap.AttemptKey("default", "glm-5.3-flash", None, VENUE),
        cap.AttemptKey("default", "Qwen3.8-27B-NVFP4", "128k", VENUE),
        cap.AttemptKey("other-project", "Qwen3.8-27B-NVFP4", None, VENUE),
    ],
)
def test_failures_of_another_venue_model_variant_or_project_do_not_count(other) -> None:
    _record_failures("1", "2", "3", key=other)
    assert cap.check(KEY) is None


def test_the_api_layer_hands_the_audit_trail_in() -> None:
    assert cap._TRAIL["emit"] is audit_events.emit_audit
    assert cap._TRAIL["query"] is audit_events.query_audit_events


def test_without_a_trail_nothing_is_recorded_or_refused(monkeypatch) -> None:
    monkeypatch.setattr(cap, "_TRAIL", {})
    _record_failures("1", "2", "3")
    assert audit_events.query_audit_events(event=cap.ATTEMPT_EVENT) == []
    assert cap.check(KEY) is None


# ── the service records every finished attempt ─────────────────────────────


class _Backend:
    config = SimpleNamespace(readiness_poll_interval_sec=1)

    async def cleanup(self, *_args, **_kwargs):
        return None


def _service(state: ReadinessState) -> OrchestrationService:
    registry = EndpointRegistry()
    registry.register(
        project_id="default",
        model=KEY.model,
        job_id="8300",
        host="n",
        port=1,
        backend="slurm-remote",
    )
    service = OrchestrationService(
        registry=registry,
        port_manager=SimpleNamespace(release=lambda *_args: None),
        backend=_Backend(),
    )

    async def readiness(**_kw):
        return ReadinessResult(
            state=state, job_id="8300", host="n", port=1, error="Backend job ended (FAILED)"
        )

    service._wait_for_ready_with_budget = readiness
    return service


def _finalize(service: OrchestrationService) -> ProvisionResult:
    async def run():
        now = asyncio.get_running_loop().time()
        return await service._finalize_provision(
            request=ProvisionRequest("default", KEY.model, requesting_user="mnagaso"),
            job_id="8300",
            target_host="n",
            actual_host="n",
            actual_port=1,
            deadline=now + 60,
            progress=None,
            start_time=now,
        )

    return asyncio.run(run())


def _attempt_events() -> list[dict]:
    return audit_events.query_audit_events(event=cap.ATTEMPT_EVENT)


def test_a_failed_attempt_is_recorded_with_its_venue_and_job() -> None:
    assert _finalize(_service(ReadinessState.FAILED)).success is False
    (event,) = _attempt_events()
    assert event["outcome"] == "error"
    assert event["actor"] == "mnagaso"
    assert event["details"]["job_id"] == "8300"
    assert KEY.matches(event["details"])


def test_a_ready_attempt_is_recorded_as_a_success() -> None:
    assert _finalize(_service(ReadinessState.READY)).success is True
    (event,) = _attempt_events()
    assert event["outcome"] == "success"


# ── POST /provision ─────────────────────────────────────────────────────────


class _Service:
    def __init__(self):
        self.requests = []

    async def provision(self, request):
        self.requests.append(request)
        return ProvisionResult(success=True, job_id="8400")


def _post(monkeypatch, *, admin: bool, override: bool = False):
    service = _Service()
    monkeypatch.setattr(
        endpoints_api,
        "_resolve_registered_model_entry",
        lambda _name: {"storage_ref": "/models/qwen", "model_type": "llm"},
    )
    monkeypatch.setattr("umrs.orchestrator.service.get_orchestration_service", lambda: service)
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = EndpointRegistry
    app.dependency_overrides[endpoints_api._caller_is_admin] = lambda: admin
    body = {"project_id": "default", "model": KEY.model, "override_failure_cap": override}
    with TestClient(app) as client:
        response = client.post("/umrs/endpoints/provision", json=body)
    return response, service


def test_a_capped_provision_gets_a_typed_429_and_submits_nothing(monkeypatch) -> None:
    _record_failures("8290", "8291", "8292")
    response, service = _post(monkeypatch, admin=False)
    assert response.status_code == 429
    assert response.headers["x-platform-error-code"] == "endpoints.provision_capped"
    assert 1 <= int(response.headers["Retry-After"]) <= 1800
    assert "job 8290" in response.json()["detail"]
    assert service.requests == []


def test_a_non_admin_override_is_ignored(monkeypatch) -> None:
    _record_failures("8290", "8291", "8292")
    response, service = _post(monkeypatch, admin=False, override=True)
    assert response.status_code == 429
    assert service.requests == []


def test_an_admin_override_provisions_and_leaves_an_audit_trail(monkeypatch) -> None:
    _record_failures("8290", "8291", "8292")
    response, service = _post(monkeypatch, admin=True, override=True)
    assert response.status_code == 202
    assert len(service.requests) == 1
    (event,) = audit_events.query_audit_events(event=cap.OVERRIDE_EVENT)
    assert event["details"]["failed_jobs"] == ["8292", "8291", "8290"]


def test_an_admin_without_the_override_is_capped_too(monkeypatch) -> None:
    _record_failures("8290", "8291", "8292")
    response, _ = _post(monkeypatch, admin=True)
    assert response.status_code == 429


def test_an_uncapped_provision_goes_ahead(monkeypatch) -> None:
    _record_failures("8290", "8291")
    response, service = _post(monkeypatch, admin=False)
    assert response.status_code == 202
    assert len(service.requests) == 1
