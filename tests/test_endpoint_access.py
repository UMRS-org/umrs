"""Endpoint access scopes (ADR-078): who may route to an endpoint.

user:<id> | project:<id> | group:<id> | system, the owning project always
included; no access key = owning project only (the historical rule).
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api import endpoints as endpoints_api
from umrs.api import llm as llm_api
from umrs.orchestrator.endpoint_access import (
    AccessError,
    EndpointCaller,
    can_use,
    endpoint_scopes,
    validate_access,
)
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry


def _registry(*rows):
    registry = EndpointRegistry()
    for project, job, access, active in rows:
        metadata = {"access": access} if access is not None else {}
        registry.register(
            project_id=project,
            model="Bespoke-Nimble-9B",
            job_id=job,
            host="node-a",
            port=4000,
            backend="slurm-remote",
            metadata=metadata,
        )
        registry.mark_ready(job)
        registry.get_by_job_id(job).active_requests = active
    return registry


def _caller(user="alice", project="opossum", groups=(), admin=False, calls=None):
    def resolve():
        if calls is not None:
            calls.append(user)
        return groups

    return EndpointCaller(user, project, is_admin=admin, groups_resolver=resolve)


def _ep(project="default", access=None):
    (row,) = _registry((project, "j1", access, 0)).list_all()
    return row


# ── pure policy ──────────────────────────────────────────────────────────


def test_no_access_key_keeps_the_owning_project_only():
    ep = _ep()
    assert [str(s) for s in endpoint_scopes(ep)] == ["project:default"]
    assert can_use(ep, _caller(project="default"))
    assert not can_use(ep, _caller(project="opossum"))


@pytest.mark.parametrize(
    ("access", "caller", "allowed"),
    [
        (["system"], _caller(), True),
        (["project:opossum"], _caller(project="opossum"), True),
        (["project:opossum"], _caller(project="tj-047"), False),
        (["user:alice"], _caller(user="alice"), True),
        (["user:alice"], _caller(user="bob"), False),
        (["group:lab"], _caller(groups=("lab",)), True),
        (["group:lab"], _caller(groups=("other",)), False),
        (["garbage", "session:s1"], _caller(), False),  # malformed entries grant nothing
    ],
)
def test_each_scope_kind(access, caller, allowed):
    assert can_use(_ep(access=access), caller) is allowed


def test_admin_flag_does_not_widen_routing():
    assert not can_use(_ep(), _caller(project="opossum", admin=True))


def test_groups_are_resolved_only_when_they_decide():
    calls: list[str] = []
    caller = _caller(project="default", groups=("lab",), calls=calls)
    assert can_use(_ep(access=["group:lab"]), caller)  # own project decides first
    assert calls == []
    assert can_use(_ep(access=["group:lab"]), _caller(groups=("lab",), calls=calls))
    assert calls == ["alice"]


@pytest.mark.parametrize(
    ("access", "caller"),
    [
        (["system"], _caller()),  # system is admin-only
        (["session:s1"], _caller()),  # not an endpoint scope
        (["global"], _caller()),  # legacy alias refused on writes
        ([f"user:u{i}" for i in range(17)], _caller(admin=True)),  # too many
    ],
)
def test_refused_grants(access, caller):
    with pytest.raises(AccessError):
        validate_access(access, caller)


def test_grantable_lists_normalize():
    # The owner shares its own GPU: any user, project or group may be named.
    caller = _caller(groups=())
    assert validate_access(["user:bob", "project:tj-047", "group:lab", "user:bob"], caller) == [
        "user:bob",
        "project:tj-047",
        "group:lab",
    ]
    assert validate_access(["system"], _caller(admin=True)) == ["system"]
    assert validate_access(None, caller) == []


# ── registry selection ──────────────────────────────────────────────────


def test_shared_selection_skips_own_and_private_rows_and_balances_load():
    registry = _registry(
        ("opossum", "own", None, 0),
        ("default", "private", None, 0),
        ("default", "busy", ["system"], 5),
        ("tj-047", "idle", ["system"], 1),
    )
    pick = registry.select_shared_endpoint(_caller(), "Bespoke-Nimble-9B")
    assert pick.job_id == "idle"
    assert registry.select_shared_endpoint(_caller(), "other-model") is None


# ── routes ──────────────────────────────────────────────────────────────


def _app(registry, router):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def _headers(user="alice", project="opossum"):
    return {"x-user-id": user, "x-project-id": project}


def _proxy_target(client, project="opossum"):
    seen = {}

    async def fake_request(self, method, url, **kwargs):
        seen["url"] = str(url)
        return httpx.Response(200, json={"choices": [{"text": "A"}]})

    with patch.object(httpx.AsyncClient, "request", fake_request):
        response = client.post(
            "/umrs/llm/v1/completions",
            json={"model": "Bespoke-Nimble-9B", "prompt": "p", "max_tokens": 1},
            headers=_headers(project=project),
        )
    return response, seen


def test_a_system_endpoint_serves_every_project():
    client = _app(_registry(("default", "nimble", ["system"], 0)), llm_api.router)
    response, seen = _proxy_target(client, project="opossum")
    assert response.status_code == 200, response.text
    assert "node-a:4000" in seen["url"]


def test_a_private_endpoint_stays_private():
    client = _app(_registry(("default", "nimble", None, 0)), llm_api.router)
    response, _ = _proxy_target(client, project="opossum")
    assert response.status_code == 503 or "No active endpoint" in response.text


def test_own_project_endpoint_wins_over_a_shared_one():
    registry = _registry(("default", "shared", ["system"], 0))
    registry.register(
        project_id="opossum",
        model="Bespoke-Nimble-9B",
        job_id="mine",
        host="node-mine",
        port=4001,
        backend="slurm-remote",
    )
    registry.mark_ready("mine")
    response, seen = _proxy_target(_app(registry, llm_api.router))
    assert response.status_code == 200
    assert "node-mine:4001" in seen["url"]


def test_shared_models_are_listed(monkeypatch):
    async def fake_models(ep):
        return [{"id": ep.model, "object": "model"}]

    monkeypatch.setattr(llm_api, "_fetch_endpoint_models", fake_models)
    client = _app(_registry(("default", "nimble", ["system"], 0)), llm_api.router)
    response = client.get("/umrs/llm/v1/models", headers=_headers())
    assert response.status_code == 200
    assert "Bespoke-Nimble-9B" in json.dumps(response.json())


def test_active_listing_includes_shared_endpoints(monkeypatch):
    async def all_live(registry, endpoints):
        return endpoints

    monkeypatch.setattr(endpoints_api, "_filter_live_ready_endpoints", all_live)
    registry = _registry(("default", "nimble", ["system"], 0), ("tj-047", "private", None, 0))
    client = _app(registry, endpoints_api.router)
    response = client.get("/umrs/endpoints/active", headers=_headers())
    assert response.status_code == 200, response.text
    rows = response.json()["endpoints"]
    assert [(r["job_id"], r["access"]) for r in rows] == [("nimble", ["system"])]


def _access_client(registry, *, admin=False):
    client = _app(registry, endpoints_api.router)
    client.app.dependency_overrides[endpoints_api._caller_is_admin] = lambda: admin
    return client


def test_owner_project_shares_with_a_user_but_not_with_everyone():
    registry = _registry(("opossum", "e1", None, 0))
    client = _access_client(registry)
    ok = client.put("/umrs/endpoints/e1/access", json={"access": ["user:bob"]}, headers=_headers())
    assert ok.status_code == 200, ok.text
    assert registry.get_by_job_id("e1").metadata["access"] == ["user:bob"]
    refused = client.put(
        "/umrs/endpoints/e1/access", json={"access": ["system"]}, headers=_headers()
    )
    assert refused.status_code == 403
    assert registry.get_by_job_id("e1").metadata["access"] == ["user:bob"]


def test_another_project_cannot_rescope_an_endpoint():
    registry = _registry(("default", "e1", None, 0))
    client = _access_client(registry)
    response = client.put(
        "/umrs/endpoints/e1/access", json={"access": ["project:opossum"]}, headers=_headers()
    )
    assert response.status_code == 404
    assert "access" not in registry.get_by_job_id("e1").metadata


def test_admin_opens_an_endpoint_to_everyone():
    registry = _registry(("default", "e1", None, 0))
    client = _access_client(registry, admin=True)
    response = client.put(
        "/umrs/endpoints/e1/access", json={"access": ["system"]}, headers=_headers()
    )
    assert response.status_code == 200, response.text
    assert response.json()["access"] == ["system"]


# ── provision: explicit access and roster defaults ──────────────────────


def _request(**fields):
    return endpoints_api.ProvisionRequest(model="Bespoke-Nimble-9B", **fields)


def test_roster_access_applies_only_when_the_caller_may_grant_it(monkeypatch):
    from umrs.api import _endpoint_access as glue

    monkeypatch.setattr(glue, "_roster_access", lambda *_: ["system"])
    assert glue.provision_access(_request(), _caller(admin=True)) == ["system"]
    assert glue.provision_access(_request(), _caller()) == []  # silently private


def test_explicit_access_is_refused_loudly():
    from fastapi import HTTPException
    from umrs.api import _endpoint_access as glue

    with pytest.raises(HTTPException) as exc:
        glue.provision_access(_request(access=["system"]), _caller())
    assert exc.value.status_code == 403
    assert glue.provision_access(_request(access=["user:alice"]), _caller()) == ["user:alice"]


def test_rccs_roster_access_reaches_the_provision_unmocked(monkeypatch):
    """Live regression 2026-10-01: the relay's Nimble endpoint came up with
    no access list — the roster lookup read a key the defaults dict strips."""
    from pathlib import Path

    from umrs.api import _endpoint_access as glue
    from umrs.orchestrator import provision_roster as pr

    pool = Path(__file__).resolve().parents[1] / "deploy/rccs/pool-rccs.yaml"
    monkeypatch.setenv(pr.ROSTER_ENV_VAR, str(pool))
    pr._cache.clear()
    try:
        assert pr.resolve_model_recipe("Bespoke-Nimble-9B")["access"] == ["system"]
        assert glue.provision_access(_request(), _caller(admin=True)) == ["system"]
    finally:
        pr._cache.clear()
