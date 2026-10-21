"""Cycle-2 umrs-api lane (2026-10-21): a lease request the broker refuses is
a 400, not a 500.

Found live: POST /umrs/secrets/lease with a valid identity and
``"scope": "x1"`` (or any value outside the five scopes) answered 500
Internal Server Error; the broker's ``ValueError("Invalid scope ...")`` /
``ValueError("... exceeds maximum")`` escaped the route.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import secret_broker as api
from umrs.api._scope_identity import get_caller_scope
from umrs.jobs.secret_broker import SecretBroker
from umrs.knowledge.access_scope import CallerScope


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(api, "_broker", SecretBroker())
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[get_caller_scope] = lambda: CallerScope(
        user_id="alice", project_id="pa"
    )
    return TestClient(app)


@pytest.mark.parametrize(
    "extra",
    [{"scope": "x1"}, {"scope": "inference", "lease_seconds": 10**9}],
)
def test_refused_lease_is_400(client: TestClient, extra: dict) -> None:
    body = {"job_id": "job-1", "secret_name": "hf", **extra}
    r = client.post("/umrs/secrets/lease", json=body)
    assert r.status_code == 400, r.text
    assert r.headers.get("x-platform-error-code") == "secretbroker.bad_request"


def test_valid_lease_still_issues(client: TestClient) -> None:
    r = client.post(
        "/umrs/secrets/lease", json={"job_id": "job-1", "secret_name": "hf", "scope": "general"}
    )
    assert r.status_code in (200, 201), r.text
    assert r.json()["secret_value"]
