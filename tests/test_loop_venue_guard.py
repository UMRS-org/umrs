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


def _start(client, venue: str):
    body = {"project_id": "proj-v", "research_goal": "venue guard", "max_iterations": 1}
    if venue:
        body["venue"] = venue
    return client.post("/umrs/loops", json=body)


def test_foreign_cluster_venue_is_refused_before_anything_exists(client):
    from umrs.api import loop as loop_api

    test_client, calls = client
    resp = _start(test_client, "cluster:bdrt-front")
    assert resp.status_code == 403, resp.text
    assert "cross_site_venue_refused" in resp.text
    assert calls == []
    assert loop_api._loop_store.list_all() == []


@pytest.mark.parametrize("venue", ["", "frontend", f"cluster:{OWN}"])
def test_own_site_venues_still_start(client, venue):
    test_client, calls = client
    resp = _start(test_client, venue)
    assert resp.status_code == 201, resp.text
    assert len(calls) == 1
    if venue.startswith("cluster:"):
        assert calls[0]["spec"]["venue"] == venue
