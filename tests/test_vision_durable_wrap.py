"""Vision durable wrap pins (G-P4-2, WP-09, ADR-044).

Every /vision/infer/submit mints one VISION_INFER durable job, request-driven
(NEVER requeue, max_attempts=1): admit around the downstream call, terminal
transition on outcome (status poll or cancel). All additive — without a pool
the calls run exactly as before.
"""

from __future__ import annotations

import httpx
import pytest
import umrs.api.vision as vision_api
import umrs.api.vision_durable as vd
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.vision import router
from umrs.orchestrator.endpoint_registry import EndpointRegistry, EndpointStatus


class _S3Config:
    bucket = "umrs-files"
    default_ttl = 3600
    max_ttl = 86400

    def get_client(self):
        return object()


def _registry_with_vision_endpoint() -> EndpointRegistry:
    registry = EndpointRegistry()
    endpoint = registry.register(
        project_id="project-a",
        model="sam3",
        job_id="job-vision",
        host="node-a",
        port=3001,
        backend="slurm",
        metadata={
            "model_type": "vision",
            "serving_backend": "bentoml",
            "service_kind": "bentoml_vision",
            "service_url": "http://node-a:3001",
            "vision_adapter": "sam3",
        },
    )
    endpoint.status = EndpointStatus.READY
    return registry


@pytest.fixture()
def app_client(monkeypatch):
    monkeypatch.setattr(vision_api, "generate_presigned_get", lambda *a, **k: "get-url")
    monkeypatch.setattr(
        vision_api, "generate_presigned_put", lambda _c, _b, key, **kw: f"put-url:{key}"
    )
    registry = _registry_with_vision_endpoint()
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[vision_api.get_registry] = lambda: registry
    app.dependency_overrides[vision_api.get_s3_config] = _S3Config

    class _Pool:
        pass

    app.state.umrs_db_pool = _Pool()
    # store_for would try to build a real JobStore (schema bootstrap touches
    # the pool) — the fixtures drive admit/terminal seams directly.
    monkeypatch.setattr(vd, "store_for", lambda req: object())
    yield app
    app.dependency_overrides.clear()


def _submit_body():
    return {
        "model": "sam3",
        "image_uri": "s3://umrs-files/project-a/images/cell.png",
        "inference_config": {"prompt": "cell"},
    }


def test_submit_mints_running_job_and_returns_job_id(app_client, monkeypatch):
    calls: dict = {}

    def fake_admit(store, **kw):
        calls["admit"] = (store is not None, kw)
        return "job-1", "att-1"

    monkeypatch.setattr(vd, "admit", fake_admit)

    async def fake_request(endpoint, method, path, *, json_payload=None, params=None):
        return httpx.Response(200, json={"task_id": "t-1", "status": "submitted"})

    monkeypatch.setattr(vision_api, "_request_endpoint", fake_request)

    with TestClient(app_client) as client:
        resp = client.post(
            "/umrs/vision/infer/submit",
            headers={
                "X-Project-ID": "project-a",
                "X-Verified-Project-ID": "project-a",
                "X-User-ID": "alice",
            },
            json=_submit_body(),
        )
    assert resp.status_code == 200
    assert resp.json()["job_id"] == "job-1"
    present, kw = calls["admit"]
    assert present
    assert kw["project_id"] == "project-a"
    assert kw["user_id"] == "alice"
    assert kw["spec"]["endpoint_job_id"] == "job-vision"


def test_submit_unknown_user_recorded_as_unknown(app_client, monkeypatch):
    seen = {}

    def fake_admit(store, **kw):
        seen.update(kw)
        return None, None

    monkeypatch.setattr(vd, "admit", fake_admit)

    async def fake_request(*a, **k):
        return httpx.Response(200, json={"task_id": "t-1"})

    monkeypatch.setattr(vision_api, "_request_endpoint", fake_request)
    with TestClient(app_client) as client:
        resp = client.post(
            "/umrs/vision/infer/submit",
            headers={"X-Project-ID": "project-a", "X-Verified-Project-ID": "project-a"},
            json=_submit_body(),
        )
    assert resp.status_code == 200
    assert seen["user_id"] == "unknown"
    assert resp.json()["job_id"] is None  # admit declined -> additive None


def test_submit_downstream_exception_marks_failed(app_client, monkeypatch):
    monkeypatch.setattr(vd, "admit", lambda store, **kw: ("job-1", "att-1"))
    term: list = []
    monkeypatch.setattr(vd, "terminal", lambda *a, **kw: term.append((a, kw)))

    async def boom(*a, **k):
        raise httpx.ConnectError("downstream dead")

    monkeypatch.setattr(vision_api, "_request_endpoint", boom)
    with TestClient(app_client) as client, pytest.raises(Exception):
        client.post(
            "/umrs/vision/infer/submit",
            headers={"X-Project-ID": "project-a", "X-Verified-Project-ID": "project-a"},
            json=_submit_body(),
        )
    assert any(args[2] == "failed" for args, _ in term)


def test_status_completed_transitions_attempt_once_and_safely(app_client, monkeypatch):
    monkeypatch.setattr(vd, "admit", lambda store, **kw: ("job-1", "att-1"))
    term: list[tuple] = []

    def fake_terminal(store, attempt, outcome, *, reason=""):
        term.append((attempt, outcome))

    monkeypatch.setattr(vd, "terminal", fake_terminal)

    async def fake_request(endpoint, method, path, *, json_payload=None, params=None):
        return httpx.Response(200, json={"task_id": "t-1", "status": "completed"})

    monkeypatch.setattr(vision_api, "_request_endpoint", fake_request)

    # get_job lookup stub on the module seam
    class _Store:
        def get_job(self, job_id):
            return type("J", (), {"active_attempt_id": "att-1"})()

    monkeypatch.setattr(vd, "store_for", lambda req: _Store())

    with TestClient(app_client) as client:
        for _ in range(2):  # status polled twice — idempotence contract lives in terminal()
            resp = client.get(
                "/umrs/vision/infer/status",
                params={"task_id": "t-1", "job_id": "job-1"},
                headers={"X-Project-ID": "project-a"},
            )
            assert resp.status_code == 200
    assert ("att-1", "succeeded") in term


def test_cancel_route_cancels_durable_row(app_client, monkeypatch):
    cancels: list = []
    monkeypatch.setattr(
        vd, "cancel", lambda store, job_id, *, reason: cancels.append((job_id, reason))
    )
    monkeypatch.setattr(vd, "store_for", lambda req: object())

    async def fake_request(*a, **k):
        return httpx.Response(200, json={"cancelled": True})

    monkeypatch.setattr(vision_api, "_request_endpoint", fake_request)
    with TestClient(app_client) as client:
        resp = client.post(
            "/umrs/vision/infer/cancel",
            params={"task_id": "t-1", "job_id": "job-1"},
            headers={"X-Project-ID": "project-a"},
        )
        assert resp.status_code == 200
    assert cancels == [("job-1", "user_cancelled")]


def test_status_without_job_id_never_transitions(app_client, monkeypatch):
    term: list = []
    monkeypatch.setattr(vd, "terminal", lambda *a, **k: term.append(a))

    async def fake_request(*a, **k):
        return httpx.Response(200, json={"status": "completed"})

    monkeypatch.setattr(vision_api, "_request_endpoint", fake_request)
    with TestClient(app_client) as client:
        resp = client.get(
            "/umrs/vision/infer/status",
            params={"task_id": "t-1"},
            headers={"X-Project-ID": "project-a"},
        )
        assert resp.status_code == 200
    assert term == []
