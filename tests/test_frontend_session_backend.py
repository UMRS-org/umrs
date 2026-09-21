"""``frontend_session`` job lane (ADR-046 A2 P10): submit → provision, checker
→ lease/idle/exit rules, cancel bus → teardown, spec validation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from tests.fake_pg_pool import FakePool
from umrs.jobs import download_safety
from umrs.jobs.frontend_session_backend import (
    REF_PREFIX,
    FrontendSessionBackend,
    FrontendSessionStatusChecker,
    validate_frontend_session_spec,
)
from umrs.jobs.models import JobType
from umrs.venues.venue_sessions import VenueSessionStore

SRC = "11111111-1111-4111-8111-111111111111"
NOW = datetime.now(timezone.utc)


def _spec(**over):
    base = {
        "session_id": "sess-1",
        "owner": "alice",
        "source_ids": [SRC],
        "source_names": ["openai-main"],
        "default_source_id": SRC,
        "default_source_name": "openai-main",
        "web_policy": "public",
        "venue": "rccs",
    }
    base.update(over)
    return base


class FakeProvisioner:
    def __init__(self, *, fail: str = ""):
        self.fail = fail
        self.provisioned: list = []
        self.torn: list[str] = []

    def provision(self, cfg):
        self.provisioned.append(cfg)
        if self.fail:
            return SimpleNamespace(state="failed", reason=self.fail, lease_id="", egress_ip="")
        return SimpleNamespace(
            state="running", reason="", lease_id="lease-1", egress_ip="172.20.8.2"
        )

    def teardown(self, sid):
        self.torn.append(sid)


class FakeBroker:
    def __init__(self):
        self.leases: dict[str, SimpleNamespace] = {}
        self.revoked: list[str] = []

    def add(self, lease_id, *, valid=True, last=None):
        self.leases[lease_id] = SimpleNamespace(
            lease_id=lease_id, is_valid=valid, issued_at=NOW, last_accessed_at=last
        )

    def get_lease(self, lease_id):
        return self.leases.get(lease_id)

    def revoke_lease(self, lease_id, actor=""):
        self.revoked.append(lease_id)
        self.leases[lease_id].is_valid = False
        return True


def _config(idle=600):
    return SimpleNamespace(
        subnet_pool="172.30.0.0/16",
        idle_seconds=idle,
        provision_config=lambda **kw: SimpleNamespace(**kw),
    )


def _lane(*, fail="", idle=600):
    prov = FakeProvisioner(fail=fail)
    broker = FakeBroker()
    pool = FakePool(script=[None] * 20)
    backend = FrontendSessionBackend(
        config=_config(idle), provisioner=prov, store=VenueSessionStore(), broker=broker, pool=pool
    )
    return backend, prov, broker, pool


def _job(job_id="job-1", spec=None):
    return SimpleNamespace(job_id=job_id, spec=spec or _spec())


def _submit(backend, job):
    return asyncio.run(backend.submit_job(job, SimpleNamespace(attempt_id="a1")))


def _check(checker, ref):
    return asyncio.run(checker.check_status(SimpleNamespace(backend_ref=ref), SimpleNamespace()))


class TestSpec:
    def test_valid_spec_normalizes(self):
        out = validate_frontend_session_spec(_spec(web_allow_domains=["a.example", ""]))
        assert out["web_allow_domains"] == ["a.example"]
        assert out["default_source_id"] == SRC

    @pytest.mark.parametrize(
        "bad",
        [
            {"session_id": ""},
            {"source_ids": []},
            {"source_names": ["x", "y"]},
            {"default_source_id": "22222222-2222-4222-8222-222222222222"},
            {"web_policy": "wide"},
        ],
    )
    def test_invalid_specs_refuse(self, bad):
        with pytest.raises(ValueError):
            validate_frontend_session_spec(_spec(**bad))

    def test_non_object_spec(self):
        with pytest.raises(TypeError):
            validate_frontend_session_spec("nope")


class TestSubmit:
    def test_submit_provisions_and_projects_running(self):
        backend, prov, _broker, pool = _lane()
        ref = _submit(backend, _job())
        assert ref == f"{REF_PREFIX}session_id=sess-1|lease_id=lease-1"
        row = backend.store.get("sess-1")
        assert row.state == "running" and row.job_id == "job-1" and row.egress_ip == "172.20.8.2"
        cfg = prov.provisioned[0]
        assert cfg.session_id == "sess-1" and cfg.subnet.startswith("172.30.")
        sql, params = pool.record[-1]
        assert "UPDATE companion_session_snapshots" in sql and params[1] == "ongoing"

    def test_submit_reuses_the_row_the_start_route_reserved(self):
        backend, prov, _b, _p = _lane()
        backend.store.upsert_provisioning(
            session_id="sess-1",
            user_id="alice",
            subnet="172.30.7.0/24",
            source_names=("openai-main",),
            source_ids=(SRC,),
            default_source_id=SRC,
            default_source_name="openai-main",
            web_policy="public",
            job_id="job-1",
        )
        _submit(backend, _job())
        assert prov.provisioned[0].subnet == "172.30.7.0/24"
        assert backend.store.taken_subnets() == {"172.30.7.0/24"}

    def test_provision_failure_marks_row_and_raises(self):
        backend, _prov, _b, pool = _lane(fail="egress-sess-1")
        with pytest.raises(RuntimeError, match="egress-sess-1"):
            _submit(backend, _job())
        row = backend.store.get("sess-1")
        assert row.state == "failed" and "egress" in row.reason
        assert pool.record[-1][1][1] == "error"

    def test_bad_spec_fails_before_touching_podman(self):
        backend, prov, _b, _p = _lane()
        with pytest.raises(ValueError):
            _submit(backend, _job(spec=_spec(source_ids=[])))
        assert prov.provisioned == []


class TestChecker:
    def _running(self, idle=600):
        backend, prov, broker, pool = _lane(idle=idle)
        ref = _submit(backend, _job())
        broker.add("lease-1", last=NOW)
        return backend, prov, broker, pool, FrontendSessionStatusChecker(backend), ref

    def test_registers_for_its_job_type(self):
        backend, *_ = _lane()
        assert FrontendSessionStatusChecker(backend).job_type is JobType.FRONTEND_SESSION

    def test_live_lease_reports_running_detail(self):
        backend, _p, _b, _pool, checker, ref = self._running()
        backend.store.update_runtime("sess-1", {"phase": "running", "turn_seq": 2})
        res = _check(checker, ref)
        assert res.status == "running"
        assert "phase=running" in res.status_detail and "turn_seq=2" in res.status_detail

    def test_revoked_lease_tears_down_and_succeeds(self):
        backend, prov, broker, _pool, checker, ref = self._running()
        broker.leases["lease-1"].is_valid = False
        res = _check(checker, ref)
        assert res.status == "succeeded" and res.metrics["end"] == "lease_ended"
        assert prov.torn == ["sess-1"]
        assert backend.store.get("sess-1").state == "stopped"

    def test_idle_lease_is_revoked_then_torn_down(self):
        backend, prov, broker, _pool, checker, ref = self._running(idle=60)
        broker.leases["lease-1"].last_accessed_at = NOW - timedelta(seconds=120)
        res = _check(checker, ref)
        assert res.status == "succeeded" and res.metrics["end"] == "idle"
        assert broker.revoked == ["lease-1"] and prov.torn == ["sess-1"]

    def test_worker_exit_fails_the_attempt(self):
        backend, prov, _b, pool, checker, ref = self._running()
        backend.store.update_runtime("sess-1", {"phase": "exited", "detail": "lease refused"})
        res = _check(checker, ref)
        assert res.status == "failed" and "lease refused" in res.exit_reason
        assert backend.store.get("sess-1").state == "failed" and prov.torn == ["sess-1"]
        assert pool.record[-1][1][1] == "error"

    def test_row_already_stopped_by_owner_succeeds_without_teardown(self):
        backend, prov, _b, _pool, checker, ref = self._running()
        backend.store.mark_stopped("sess-1", reason="stopped by user")
        prov.torn.clear()
        res = _check(checker, ref)
        assert res.status == "succeeded" and res.metrics["end"] == "stopped by user"
        assert prov.torn == []

    def test_missing_row_or_foreign_ref_fails(self):
        backend, *_ = _lane()
        checker = FrontendSessionStatusChecker(backend)
        assert _check(checker, "slurm|123").status == "failed"
        assert _check(checker, f"{REF_PREFIX}session_id=ghost").status == "failed"


class TestCancelBus:
    def test_cancel_listener_tears_the_sandbox_down(self):
        backend, prov, _b, _pool = _lane()
        _submit(backend, _job())
        download_safety.register_cancel_listener(backend.on_cancel)
        try:
            download_safety.broadcast_cancel("job-1")
        finally:
            download_safety._listeners.remove(backend.on_cancel)  # noqa: SLF001
        assert prov.torn == ["sess-1"]
        assert backend.store.get("sess-1").state == "stopped"

    def test_cancel_of_unlinked_job_is_a_noop(self):
        backend, prov, _b, _pool = _lane()
        backend.on_cancel("job-unknown")
        assert prov.torn == []
