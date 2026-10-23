"""S1a (Y-1): a site's job door admits only work for that site.

``JobSpec`` keeps extra fields, so POST /umrs/jobs (and the ZMQ JOB_SUBMIT
seam) took ``spec.venue = "cluster:bdrt-front"`` at another site's gateway.
All fronts share one jobs table and claims are venue-scoped, so bdrt's
server would run that job without its own gateway or user gate ever being
consulted. Both seams now refuse a foreign venue with
``cross_site_venue_refused`` — the parent spec and every fan-out child.
Allowed: empty, this site's name, ``cluster:<this site>``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.jobs.models import JobType

from r4_contracts.handshake_hmac import sign_claims

SECRET = "s1a-venue-guard-secret"
OWN = "rccs-front"
FOREIGN = ["cluster:bdrt-front", "bdrt-front", "bdrt"]
ALLOWED = ["", OWN, f"cluster:{OWN}"]


def _signed(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


ALICE = _signed("alice", "proj-a")
ADMIN = _signed("ops", "proj-a", admin=True)


class _JobStore:
    _get_conn = None  # no catalog autopin

    def __init__(self, site_name: str):
        self.site_name = site_name
        self.created: list[dict] = []

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return "job-1"

    create_fan_out_graph = create_job

    def enqueue_job(self, _job_id):
        pass


def _client(monkeypatch, site_name: str = OWN):
    from umrs.api.jobs import get_job_store
    from umrs.api.jobs import router as jobs_router

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    store = _JobStore(site_name)
    app = FastAPI()
    app.include_router(jobs_router)
    app.dependency_overrides[get_job_store] = lambda: store
    return TestClient(app), store


def _body(venue: str, *, child_venue: str | None = None) -> dict:
    spec = {"venue": venue} if venue else {}
    body = {
        "project_id": "proj-a",
        "user_id": "alice",
        "job_type": "cleanup",
        "spec": spec,
        "enqueue_immediately": False,
    }
    if child_venue is not None:
        body["fan_out_jobs"] = [
            {"name": "c", "job_type": "cleanup", "spec": {"venue": child_venue}}
        ]
    return body


@pytest.mark.parametrize("venue", FOREIGN)
@pytest.mark.parametrize("headers", [ALICE, ADMIN], ids=["user", "admin"])
def test_rest_refuses_a_foreign_venue(monkeypatch, venue, headers):
    client, store = _client(monkeypatch)
    r = client.post("/umrs/jobs", json=_body(venue), headers=headers)
    assert r.status_code == 403, r.text
    assert "cross_site_venue_refused" in r.text
    assert store.created == []


@pytest.mark.parametrize("venue", ALLOWED)
def test_rest_accepts_own_site_and_empty_venue(monkeypatch, venue):
    client, store = _client(monkeypatch)
    r = client.post("/umrs/jobs", json=_body(venue), headers=ALICE)
    assert r.status_code == 201, r.text
    assert len(store.created) == 1


def test_rest_refuses_a_fan_out_child_with_a_foreign_venue(monkeypatch):
    client, store = _client(monkeypatch)
    body = _body("", child_venue="cluster:bdrt-front")
    r = client.post("/umrs/jobs", json=body, headers=ALICE)
    assert r.status_code == 403, r.text
    assert "cross_site_venue_refused" in r.text
    assert store.created == []


def test_rest_accepts_a_fan_out_child_on_its_own_site(monkeypatch):
    client, store = _client(monkeypatch)
    body = _body("", child_venue=f"cluster:{OWN}")
    r = client.post("/umrs/jobs", json=body, headers=ALICE)
    assert r.status_code == 201, r.text
    assert len(store.created) == 1


def test_rest_without_a_site_name_is_unchanged(monkeypatch):
    """No site.yaml (dev/mvp, the pre-venue shared lane): nothing to guard."""
    client, store = _client(monkeypatch, site_name="")
    r = client.post("/umrs/jobs", json=_body("cluster:bdrt-front"), headers=ALICE)
    assert r.status_code == 201, r.text


def _zmq_backend(site_name: str = OWN):
    from umrs.server.job_tracking_handler import DurableJobCommandBackend

    store = MagicMock()
    store.site_name = site_name
    store.create_job.return_value = "job-1"
    store.get_job.return_value = SimpleNamespace(
        job_id="job-1",
        user_id="alice",
        job_type=JobType.CLEANUP,
        to_dict=lambda: {"job_id": "job-1"},
    )
    store.get_attempts_for_job.return_value = []
    return DurableJobCommandBackend(store), store


def _zmq_payload(venue: str) -> str:
    spec = {"target": "endpoint", "job_id": "job-0", "venue": venue}
    return json.dumps({"project_id": "p", "job_type": "cleanup", "spec": spec})


@pytest.mark.parametrize("venue", FOREIGN)
def test_zmq_refuses_a_foreign_venue(venue):
    backend, store = _zmq_backend()
    out = json.loads(backend.submit("alice", _zmq_payload(venue)))
    assert out["status"] == "error"
    assert "cross_site_venue_refused" in out["message"]
    store.create_job.assert_not_called()


@pytest.mark.parametrize("venue", ALLOWED)
def test_zmq_accepts_own_site_and_empty_venue(venue):
    backend, store = _zmq_backend()
    out = json.loads(backend.submit("alice", _zmq_payload(venue)))
    assert out["status"] == "ok", out
    store.create_job.assert_called_once()
