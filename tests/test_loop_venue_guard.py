"""S1a (Y-1): the loop start route admits only this site's cluster venue.

POST /umrs/loops accepted any well-formed ``cluster:<site>``; the loop's
``research_loop_job`` then landed in the shared jobs table bound to that
site, whose server claimed and ran it without its own gateway or user gate.
A foreign cluster venue is now refused (403 ``cross_site_venue_refused``)
before the loop record exists; this site's own cluster venue and the
default ``frontend`` stay.
"""

from __future__ import annotations

import pytest

OWN = "rccs-front"


@pytest.fixture
def client(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import loop as loop_api
    from umrs.jobs import store as job_store
    from umrs.knowledge.loop import AutonomousResearchLoop
    from umrs.knowledge.loop_store import LoopStore
    from umrs.server import durable_job_runtime as djr

    monkeypatch.setattr(loop_api, "_loop_store", LoopStore.memory())
    monkeypatch.setattr(loop_api, "_loop_engine", AutonomousResearchLoop(simulated=True))
    loop_api.clear_loop_store()
    monkeypatch.setattr(job_store, "configured_site_name", lambda: OWN)
    calls: list[dict] = []

    class _FakeRuntime:
        async def submit_job_sync(self, **kw):
            calls.append(kw)
            return "job-loop-1"

    monkeypatch.setattr(djr, "get_runtime", lambda: _FakeRuntime())
    app = FastAPI()
    app.include_router(loop_api.router)
    yield TestClient(app), calls
    loop_api.clear_loop_store()


def _start(client, venue: str, headers: dict[str, str] | None = None):
    body = {"project_id": "proj-v", "research_goal": "venue guard", "max_iterations": 1}
    if venue:
        body["venue"] = venue
    return client.post("/umrs/loops", json=body, headers=headers or {})


def _owner(user: str = "alice", project: str = "proj-v", admin: bool = False):
    """Verified identity headers (mvp mode: presence is enough)."""
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
    }


def _declared(monkeypatch, allowed, admins=("ops",)):
    """This site declares its personal service account (ADR-112); *admins*
    are administrators by their user record (the one admin source)."""
    from types import SimpleNamespace

    from umrs.orchestrator.site_config import HpcAccessConfig, ServiceAccountAccess
    from umrs.shared import hpc_user_work_access as hpc_user_work

    hpc = HpcAccessConfig(
        modes_allowed=["service_account"],
        default_mode="service_account",
        service_account=ServiceAccountAccess(
            login="operator", personal=True, allowed_projects=list(allowed)
        ),
    )
    site = SimpleNamespace(
        name=OWN, extra={"ssh_user": "operator"}, scheduler=SimpleNamespace(hpc_access=hpc)
    )
    monkeypatch.setattr(hpc_user_work, "current_site", lambda: site)
    monkeypatch.setattr(hpc_user_work, "user_is_admin", lambda user: user in admins)


def test_foreign_cluster_venue_is_refused_before_anything_exists(client):
    from umrs.api import loop as loop_api

    test_client, calls = client
    resp = _start(test_client, "cluster:bdrt-front")
    assert resp.status_code == 403, resp.text
    assert "cross_site_venue_refused" in resp.text
    assert calls == []
    assert loop_api._loop_store.list_all() == []


@pytest.mark.parametrize("venue", ["", "frontend", f"cluster:{OWN}"])
def test_own_site_venues_still_start(client, monkeypatch, venue):
    test_client, calls = client
    # ADR-112: the cluster venue needs a verified owner in a listed project.
    _declared(monkeypatch, ["proj-v"])
    resp = _start(test_client, venue, _owner())
    assert resp.status_code == 201, resp.text
    assert len(calls) == 1
    if venue.startswith("cluster:"):
        assert calls[0]["spec"]["venue"] == venue


# ── ADR-112 A.3.3: a cluster loop is user work on the site's login ─────────


def test_cluster_loop_of_an_unlisted_project_is_refused_typed(client, monkeypatch):
    from umrs.api import loop as loop_api

    test_client, calls = client
    _declared(monkeypatch, ["other-project"])
    resp = _start(test_client, f"cluster:{OWN}", _owner())
    assert resp.status_code == 403, resp.text
    assert resp.headers["x-platform-error-code"] == "hpcaccess.project_not_allowed"
    assert calls == []
    assert loop_api._loop_store.list_all() == []


def test_cluster_loop_on_an_admins_only_site(client, monkeypatch):
    test_client, calls = client
    _declared(monkeypatch, [])
    refused = _start(test_client, f"cluster:{OWN}", _owner())
    assert refused.status_code == 403, refused.text
    admitted = _start(test_client, f"cluster:{OWN}", _owner("ops", admin=True))
    assert admitted.status_code == 201, admitted.text
    assert len(calls) == 1


def test_a_token_admin_without_an_admin_record_is_refused(client, monkeypatch):
    """Review 2026-11-08: the start read the token's admin flag, the durable
    submit the user record; a token-only admin was accepted, then refused at
    submit. Both read the record now."""
    test_client, calls = client
    _declared(monkeypatch, [], admins=())
    resp = _start(test_client, f"cluster:{OWN}", _owner("ops", admin=True))
    assert resp.status_code == 403, resp.text
    assert resp.headers["x-platform-error-code"] == "hpcaccess.project_not_allowed"
    assert calls == []


def test_ownerless_cluster_loop_is_refused(client, monkeypatch):
    test_client, calls = client
    _declared(monkeypatch, ["proj-v"])
    resp = _start(test_client, f"cluster:{OWN}")
    assert resp.status_code == 403, resp.text
    # An admin flag without an owner admits nothing (review 2026-11-08).
    resp = _start(test_client, f"cluster:{OWN}", {"X-Verified-Is-Admin": "true"})
    assert resp.status_code == 403, resp.text
    assert resp.headers["x-platform-error-code"].startswith("hpcaccess.")
    assert calls == []


def test_frontend_loop_is_not_checked(client, monkeypatch):
    test_client, calls = client
    _declared(monkeypatch, [])
    resp = _start(test_client, "frontend", _owner())
    assert resp.status_code == 201, resp.text
