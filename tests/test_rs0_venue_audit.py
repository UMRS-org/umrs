"""RS-0 S-7: every venue lifecycle transition writes one ``audit_events`` row.

Rows are captured at the persistent-store seam of the existing emitter
(``umrs.api.audit_events._audit_store``), so what is asserted is what the
sidecar's ``AuditStore`` would insert.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.test_companion_venue_routes import (
    READY_ENV,
    FakeJobStore,
    FakePodman,
    FakeSealed,
    _meta,
    _Pool,
    _run_lane,
)
from tests.test_frontend_session_backend import NOW, _check, _job, _lane, _submit
from umrs.api import audit_events
from umrs.api.companion_venue import router
from umrs.jobs.frontend_session_backend import FrontendSessionStatusChecker
from umrs.jobs.secret_broker import SecretBroker
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.frontend_podman import FrontendPodmanProvisioner
from umrs.venues.frontend_service import FrontendVenueService
from umrs.venues.venue_sessions import VenueSessionStore

H = {"x-user-id": "alice"}
BASE = "/umrs/companion/sessions/sess-1/venue"


class _Sink:
    def __init__(self):
        self.rows: list[dict] = []

    def write(self, record):
        self.rows.append(record)

    def venue(self) -> list[dict]:
        return [r for r in self.rows if r["event"].startswith("venue.")]

    def named(self, event: str) -> list[dict]:
        return [r for r in self.rows if r["event"] == event]


@pytest.fixture
def sink(monkeypatch):
    s = _Sink()
    monkeypatch.setattr(audit_events, "_audit_store", s)
    return s


@pytest.fixture
def venue(sink):
    pool = _Pool()
    broker = SecretBroker()
    podman = FakePodman()
    service = FrontendVenueService(
        config=FrontendVenueConfig.from_env(READY_ENV),
        broker=broker,
        store=VenueSessionStore(),
        provisioner=FrontendPodmanProvisioner(
            broker, podman=podman, session_root=Path(tempfile.mkdtemp(prefix="audit-venue-"))
        ),
        pool=pool,
        sealed_store=FakeSealed({("alice", "openai-main"): _meta("openai-main")}),
        job_store=FakeJobStore(),
        site_name="rccs",
    )
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = pool
    app.state.companion_venue_service = service
    h = SimpleNamespace(
        client=TestClient(app), pool=pool, broker=broker, podman=podman, service=service
    )
    h.jobs = service.job_store
    pool.seed("alice", "sess-1")
    return h


def _start(h, **body):
    r = h.client.post(f"{BASE}/start", json={"source_refs": ["openai-main"], **body}, headers=H)
    assert r.status_code == 200, r.text
    return r


class TestOwnerLifecycle:
    def test_start_writes_one_start_row(self, venue, sink):
        _start(venue, web_policy="allowlist")
        (row,) = sink.venue()
        assert (row["event"], row["actor"], row["outcome"]) == (
            "venue.session.start",
            "alice",
            "success",
        )
        d = row["details"]
        assert (d["session_id"], d["owner"], d["job_id"], d["web_policy"]) == (
            "sess-1",
            "alice",
            "job-1",
            "allowlist",
        )
        assert d["sources"] == ["openai-main"]

    def test_provisioning_issues_one_lease_row(self, venue, sink):
        _start(venue)
        _run_lane(venue)
        (row,) = sink.named("venue.lease.issue")
        lease = venue.broker.list_leases(scope="inference")[0]
        assert row["actor"] == "provisioner"
        assert (row["details"]["lease_id"], row["details"]["owner"]) == (lease.lease_id, "alice")
        assert row["details"]["expires_at"]

    def test_owner_stop_writes_stop_and_revoke(self, venue, sink):
        _start(venue)
        _run_lane(venue)
        venue.client.post(f"{BASE}/stop", headers=H)
        (stop,) = sink.named("venue.session.stop")
        assert (stop["actor"], stop["details"]["cause"]) == ("alice", "owner")
        (revoke,) = sink.named("venue.lease.revoke")
        assert (revoke["details"]["via"], revoke["details"]["revoked"]) == ("teardown", 1)
        # an idempotent second stop is no transition: no second row
        venue.client.post(f"{BASE}/stop", headers=H)
        assert len(sink.named("venue.session.stop")) == 1

    def test_pause_and_resume_rows(self, venue, sink):
        _start(venue)
        _run_lane(venue)
        venue.client.post(f"{BASE}/pause", headers=H)
        venue.client.post(f"{BASE}/pause", headers=H)  # idempotent: no row
        venue.client.post(f"{BASE}/resume", headers=H)
        assert [r["event"] for r in sink.venue()][-2:] == [
            "venue.session.pause",
            "venue.session.resume",
        ]
        assert len(sink.named("venue.session.pause")) == 1

    def test_no_token_or_secret_in_any_row(self, venue, sink):
        _start(venue)
        _run_lane(venue)
        venue.client.post(f"{BASE}/stop", headers=H)
        blob = repr(sink.venue()).lower()
        assert "token" not in blob and "secret" not in blob


class TestLaneLifecycle:
    def _running(self, idle=600):
        backend, prov, broker, pool = _lane(idle=idle)
        ref = _submit(backend, _job())
        broker.add("lease-1", last=NOW)
        return backend, broker, FrontendSessionStatusChecker(backend), ref

    @staticmethod
    def _stops(sink) -> list[tuple[str, str, str]]:
        return [
            (r["actor"], r["details"]["cause"], r["outcome"])
            for r in sink.named("venue.session.stop")
        ]

    def test_idle_writes_revoke_and_one_stop(self, sink):
        _backend, broker, checker, ref = self._running(idle=60)
        broker.leases["lease-1"].last_accessed_at = NOW - timedelta(seconds=120)
        _check(checker, ref)
        (revoke,) = sink.named("venue.lease.revoke")
        assert (revoke["actor"], revoke["details"]["via"]) == ("monitor", "idle")
        assert self._stops(sink) == [("monitor", "idle", "success")]

    def test_lease_end_writes_one_stop(self, sink):
        _backend, broker, checker, ref = self._running()
        broker.leases["lease-1"].is_valid = False
        _check(checker, ref)
        assert self._stops(sink) == [("monitor", "lease_end", "success")]
        assert sink.named("venue.lease.revoke") == []  # it ended; nobody revoked it here

    def test_worker_exit_writes_one_error_stop(self, sink):
        backend, _broker, checker, ref = self._running()
        backend.store.update_runtime("sess-1", {"phase": "exited", "detail": "lease refused"})
        _check(checker, ref)
        assert self._stops(sink) == [("monitor", "exit", "error")]
        assert sink.named("venue.session.stop")[0]["details"]["detail"] == "lease refused"

    def test_plan_run_done_writes_one_stop(self, sink):
        backend, _broker, checker, ref = self._running()
        backend.store.update_runtime("sess-1", {"phase": "done"})
        _check(checker, ref)
        assert self._stops(sink) == [("monitor", "done", "success")]

    def test_durable_cancel_writes_one_stop(self, sink):
        backend, *_ = self._running()
        backend.on_cancel("job-1")
        backend.on_cancel("job-1")  # already stopped: no second row
        assert self._stops(sink) == [("monitor", "cancel", "success")]

    def test_provision_failure_writes_one_error_stop(self, sink):
        backend, *_ = _lane(fail="egress: boom")
        with pytest.raises(RuntimeError):
            _submit(backend, _job())
        assert self._stops(sink) == [("provisioner", "provision_failed", "error")]

    def test_a_broken_audit_store_never_breaks_the_transition(self, monkeypatch):
        class _Boom:
            def write(self, record):
                del record
                raise RuntimeError("audit db down")

        monkeypatch.setattr(audit_events, "_audit_store", _Boom())
        backend, broker, checker, ref = self._running()
        broker.leases["lease-1"].is_valid = False
        assert asyncio.run(checker.check_status(SimpleNamespace(backend_ref=ref), None)).status == (
            "succeeded"
        )
        assert backend.store.get("sess-1").state == "stopped"
