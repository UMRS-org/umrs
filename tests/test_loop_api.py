"""Tests for UMRS Loop REST API (TASK-036).

Covers:
  - POST /umrs/loops — start a loop with/without approval
  - GET /umrs/loops — list loops
  - GET /umrs/loops/{id} — loop status
  - GET /umrs/loops/{id}/iterations — list iterations
  - POST /umrs/loops/{id}/stop — stop a loop
  - POST /umrs/loops/{id}/pause — pause a loop
  - POST /umrs/loops/{id}/resume — resume a loop
  - Unapproved project rejection (403)
"""

from __future__ import annotations

import pytest


def _try_fastapi():
    try:
        pytest.importorskip("fastapi")
    except Exception:
        pytest.skip("fastapi not available")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    return FastAPI, TestClient


@pytest.fixture(autouse=True)
def _clean_loop_store(monkeypatch):
    from umrs.api import loop as loop_api
    from umrs.api.loop import clear_loop_store
    from umrs.knowledge.loop import AutonomousResearchLoop
    from umrs.knowledge.loop_store import LoopStore

    # A memory store of its own (2026-10-18): in a full run an earlier test
    # can leave the module store bound to Postgres, where each request loads
    # its own copy and the simulated runner's save overwrote a PAUSED status
    # (test_resume answered 400 intermittently).
    monkeypatch.setattr(loop_api, "_loop_store", LoopStore.memory())
    clear_loop_store()
    # Server-side driver (engine runner) advances REST-started loops. Use a
    # simulated engine so test loops complete deterministically; real mode
    # needs a workflow port + per-experiment workflow_name (covered by the
    # engine tests, not this API-surface suite).
    previous_engine = loop_api._loop_engine
    loop_api._loop_engine = AutonomousResearchLoop(simulated=True)
    yield
    loop_api._loop_engine = previous_engine
    clear_loop_store()


def _make_app():
    FastAPI, TestClient = _try_fastapi()
    from umrs.api.loop import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestStartLoop:
    def test_unapproved_project_rejected(self):
        """Core TASK-036 acceptance: unapproved project refuses to start.

        research-workflow spec §12 — Loop criteria.
        """
        from unittest.mock import patch

        from umrs.knowledge.readiness import (
            ConfidenceLevel,
            IssueSeverity,
            ReadinessIssue,
            ReadinessLevel,
            ResearchReadinessReport,
        )

        unapproved_report = ResearchReadinessReport(
            project_id="unapproved-proj",
            readiness_level=ReadinessLevel.PARTIALLY_READY,
            confidence_level=ConfidenceLevel.MEDIUM,
            issues=[
                ReadinessIssue(
                    severity=IssueSeverity.WARNING,
                    category="missing_dataset",
                    description="Missing dataset",
                    recommendation="Add dataset",
                )
            ],
            recommendation="Improve coverage",
            next_steps=["Add data"],
        )

        client = _make_app()

        # The handler does: from umrs.knowledge.project import ResearchProject
        # which is now imported at module level. Patching umrs.api.loop.ResearchProject
        # affects the name in the module namespace that the handler uses.
        with patch("umrs.api.loop.ResearchProject") as mock_cls:
            mock_inst = mock_cls.return_value
            mock_inst.check_readiness.return_value = unapproved_report
            mock_inst.knowledge_store.list_artifacts.return_value = ["artifact1"]

            resp = client.post(
                "/umrs/loops",
                json={
                    "project_id": "unapproved-proj",
                    "research_goal": "Test goal",
                    "max_iterations": 5,
                    "approved": True,
                },
            )
            assert resp.status_code == 403
            detail = resp.json().get("detail", "")
            assert "not approved" in detail.lower()

    def test_trust_mode_no_approval_check(self):
        """approved=False (default) = trust the client."""
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-trust",
                "research_goal": "Trust test",
                "max_iterations": 3,
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["project_id"] == "proj-trust"
        assert body["status"] == "running"
        assert body["loop_id"].startswith("loop-")

    def test_validation_rejects_empty_goal(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "proj", "research_goal": "", "max_iterations": 5},
        )
        assert resp.status_code == 422

    def test_validation_rejects_max_iterations_0(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "proj", "research_goal": "G", "max_iterations": 0},
        )
        assert resp.status_code == 422

    def test_validation_rejects_max_iterations_101(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "proj", "research_goal": "G", "max_iterations": 101},
        )
        assert resp.status_code == 422

    def test_approved_flag_skips_check_on_fresh_project(self):
        """Fresh project (readiness not approved, zero artifacts) — trust the client."""
        from unittest.mock import patch

        from umrs.knowledge.readiness import (
            ConfidenceLevel,
            ReadinessLevel,
            ResearchReadinessReport,
        )

        fresh_report = ResearchReadinessReport(
            project_id="fresh",
            readiness_level=ReadinessLevel.PARTIALLY_READY,
            confidence_level=ConfidenceLevel.MEDIUM,
            issues=[],
        )

        client = _make_app()
        with patch("umrs.api.loop.ResearchProject") as mock_cls:
            mock_inst = mock_cls.return_value
            mock_inst.check_readiness.return_value = fresh_report
            mock_inst.knowledge_store.list_artifacts.return_value = []

            resp = client.post(
                "/umrs/loops",
                json={
                    "project_id": "fresh",
                    "research_goal": "Fresh",
                    "max_iterations": 2,
                    "approved": True,
                },
            )
            assert resp.status_code == 201


class TestReadinessGateFailClosed:
    """#208: gate-unavailable must deny (503), never pass the gate silently."""

    _payload = {
        "project_id": "proj-x",
        "research_goal": "Gate test",
        "max_iterations": 2,
        "approved": True,
    }

    def test_project_construction_error_denies_with_503(self):
        from unittest.mock import patch

        client = _make_app()
        with patch("umrs.api.loop.ResearchProject") as mock_cls:
            mock_cls.side_effect = RuntimeError("store offline")
            resp = client.post("/umrs/loops", json=self._payload)
            assert resp.status_code == 503
            assert "unavailable" in resp.json().get("detail", "").lower()

    def test_readiness_check_error_denies_with_503(self):
        from unittest.mock import patch

        client = _make_app()
        with patch("umrs.api.loop.ResearchProject") as mock_cls:
            mock_inst = mock_cls.return_value
            mock_inst.check_readiness.side_effect = RuntimeError("db timeout")
            resp = client.post("/umrs/loops", json=self._payload)
            assert resp.status_code == 503


class TestListLoops:
    def test_empty(self):
        client = _make_app()
        resp = client.get("/umrs/loops")
        assert resp.status_code == 200
        assert resp.json()["loops"] == []

    def test_with_loops(self):
        client = _make_app()
        client.post("/umrs/loops", json={"project_id": "p1", "research_goal": "G1"})
        resp = client.get("/umrs/loops")
        assert resp.status_code == 200
        assert len(resp.json()["loops"]) == 1


class TestLoopStatus:
    def test_existing_loop(self):
        client = _make_app()
        start = client.post(
            "/umrs/loops",
            json={"project_id": "p-status", "research_goal": "Status test"},
        )
        lid = start.json()["loop_id"]

        resp = client.get(f"/umrs/loops/{lid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["loop_id"] == lid
        # Server drives loops now: a fast fake port may already have finished.
        assert body["status"] in ("running", "completed", "stopped")

    def test_not_found(self):
        client = _make_app()
        resp = client.get("/umrs/loops/nonexistent")
        assert resp.status_code == 404


class TestIterations:
    def test_empty(self):
        client = _make_app()
        start = client.post(
            "/umrs/loops",
            json={"project_id": "p-iter", "research_goal": "Iter"},
        )
        lid = start.json()["loop_id"]
        resp = client.get(f"/umrs/loops/{lid}/iterations")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_after_run(self):
        client = _make_app()
        start = client.post(
            "/umrs/loops",
            json={"project_id": "p-iter-run", "research_goal": "Iter run", "max_iterations": 1},
        )
        lid = start.json()["loop_id"]
        resp = client.get(f"/umrs/loops/{lid}/iterations")
        assert resp.status_code == 200
        for it in resp.json():
            assert "iteration_id" in it
            assert "phase" in it


class TestLoopControl:
    def test_stop(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "p-stop", "research_goal": "S"})
        lid = start.json()["loop_id"]
        resp = client.post(f"/umrs/loops/{lid}/stop")
        assert resp.status_code == 200
        assert resp.json()["status"] == "stopped"

    def test_pause(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "p-pause", "research_goal": "S"})
        lid = start.json()["loop_id"]
        resp = client.post(f"/umrs/loops/{lid}/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

    def test_resume(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "p-resume", "research_goal": "S"})
        lid = start.json()["loop_id"]
        client.post(f"/umrs/loops/{lid}/pause")
        resp = client.post(f"/umrs/loops/{lid}/resume")
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"

    def test_resume_not_paused_fails(self):
        client = _make_app()
        start = client.post("/umrs/loops", json={"project_id": "p-rn", "research_goal": "S"})
        lid = start.json()["loop_id"]
        resp = client.post(f"/umrs/loops/{lid}/resume")
        assert resp.status_code == 400


class TestHosting:
    """ADR-036 D1 (WP-05): job hosting when a durable runtime exists, thread otherwise."""

    def test_job_hosting_records_job_and_spawns_no_thread(self):
        from umrs.knowledge import engine_loop_runner

        client = _make_app()
        seen: list[dict] = []

        async def _submitter(loop):
            seen.append({"loop_id": loop.loop_id, "mode": loop.metadata.get("mode")})
            return "job-hosted-1"

        client.app.state.loop_job_submitter = _submitter
        resp = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-h",
                "research_goal": "host me",
                "max_iterations": 2,
                "mode": "ask",
            },
        )
        assert resp.status_code == 201
        loop_id = resp.json()["loop_id"]
        assert seen == [{"loop_id": loop_id, "mode": "ask"}]
        meta = client.get(f"/umrs/loops/{loop_id}").json()["metadata"]
        assert meta["hosting"] == "job"
        assert meta["durable_job_id"] == "job-hosted-1"
        assert engine_loop_runner.get(loop_id) is None  # the job backend drives it, not the route

    def test_no_runtime_falls_back_to_thread_hosting(self):
        # Premise is "NO durable runtime" — park any process-global runtime a
        # prior suite genuinely started (durable orchestration suites boot
        # one successfully after the SESSION_GC import repair).
        from umrs.server.durable_job_runtime import _set_runtime, get_runtime

        prior_runtime = get_runtime()
        _set_runtime(None)
        try:
            client = _make_app()
            resp = client.post(
                "/umrs/loops", json={"project_id": "proj-t", "research_goal": "thread me"}
            )
            assert resp.status_code == 201
            meta = client.get(f"/umrs/loops/{resp.json()['loop_id']}").json()["metadata"]
            assert meta["hosting"] == "thread"
            assert meta["driver"] == "engine"
        finally:
            _set_runtime(prior_runtime)

    def test_submission_failure_falls_back_and_is_recorded(self):
        client = _make_app()

        async def _boom(loop):
            raise RuntimeError("runtime exploded")

        client.app.state.loop_job_submitter = _boom
        resp = client.post(
            "/umrs/loops", json={"project_id": "proj-f", "research_goal": "fallback"}
        )
        meta = client.get(f"/umrs/loops/{resp.json()['loop_id']}").json()["metadata"]
        assert meta["hosting"] == "thread"
        assert "runtime exploded" in meta["hosting_note"]


class TestOwnerIdentity:
    """ADR-036 (a): the loop owner is pinned from the gateway-verified identity."""

    def test_owner_pinned_from_verified_headers(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "proj-o", "research_goal": "own me"},
            headers={"X-User-ID": "alice", "X-Verified-Project-ID": "proj-o"},
        )
        assert resp.status_code == 201
        meta = client.get(f"/umrs/loops/{resp.json()['loop_id']}").json()["metadata"]
        assert meta["owner_user_id"] == "alice"

    def test_no_identity_keeps_ownerless_loop(self):
        client = _make_app()
        resp = client.post("/umrs/loops", json={"project_id": "proj-o", "research_goal": "anon"})
        meta = client.get(f"/umrs/loops/{resp.json()['loop_id']}").json()["metadata"]
        assert "owner_user_id" not in meta

    def test_strict_mode_rejects_forged_identity(self, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", "s3cret")
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "proj-o", "research_goal": "forged"},
            headers={
                "X-User-ID": "mallory",
                "X-Verified-Project-ID": "proj-o",
                "X-Verified-HMAC": "not-a-valid-signature",
            },
        )
        assert resp.status_code == 401

    def test_strict_mode_rejects_ownerless_start(self, monkeypatch):
        """WP-08: an ownerless durable approval inbox would let any caller
        decide the loop's tool grants — strict mode fails the start closed."""
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        client = _make_app()
        resp = client.post("/umrs/loops", json={"project_id": "proj-o", "research_goal": "anon"})
        assert resp.status_code == 401
        assert "gateway-verified identity" in resp.json()["detail"]


class TestWorkflowResolution:
    """G-P1-3 (WP-07): real-mode loops resolve their workflow at start.

    The API fixture's engine is simulated (no workflow needed); these tests
    swap in a real-mode engine, host the loop through the injected job
    submitter (no driver thread), and stub the registry read.
    """

    @pytest.fixture(autouse=True)
    def _real_mode_engine(self):
        from umrs.api import loop as loop_api
        from umrs.knowledge.execution_port import FakeWorkflowExecutionPort
        from umrs.knowledge.loop import AutonomousResearchLoop

        previous = loop_api._loop_engine
        loop_api._loop_engine = AutonomousResearchLoop(
            simulated=False, workflow_port=FakeWorkflowExecutionPort()
        )
        yield
        loop_api._loop_engine = previous

    @staticmethod
    def _client(monkeypatch, registry):
        from umrs.api import loop as loop_api

        client = _make_app()

        async def _submitter(loop):
            return "job-wf"

        client.app.state.loop_job_submitter = _submitter
        if isinstance(registry, Exception):

            def _boom(owner):
                raise registry

            monkeypatch.setattr(loop_api, "_registered_workflow_names", _boom)
        else:
            monkeypatch.setattr(
                loop_api, "_registered_workflow_names", lambda _owner: list(registry)
            )
        return client

    @staticmethod
    def _start(client, **extra):
        body = {"project_id": "proj-wf", "research_goal": "yield", "max_iterations": 2}
        body.update(extra)
        return client.post("/umrs/loops", json=body)

    def test_single_registered_workflow_is_the_default(self, monkeypatch):
        from umrs.api import loop as loop_api

        client = self._client(monkeypatch, ["demo08_yield_eval"])
        resp = self._start(client)
        assert resp.status_code == 201, resp.text
        assert resp.json()["workflow_name"] == "demo08_yield_eval"
        loop = loop_api._loop_store.get(resp.json()["loop_id"])
        assert loop.metadata["workflow_name"] == "demo08_yield_eval"
        assert loop.metadata["workflow_resolution"] == {
            "workflow_name": "demo08_yield_eval",
            "source": "single-registered-default",
            "candidates": ["demo08_yield_eval"],
        }
        assert loop.plan is not None and loop.plan.experiments
        assert all(
            e.metadata.get("workflow_name") == "demo08_yield_eval" for e in loop.plan.experiments
        )

    def test_no_registered_workflow_fails_closed_with_guidance(self, monkeypatch):
        client = self._client(monkeypatch, [])
        resp = self._start(client)
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "none is registered" in detail
        assert "r4 workflow list" in detail and "r4 workflow upload" in detail
        assert client.get("/umrs/loops").json()["loops"] == []  # nothing half-started

    def test_several_registered_workflows_require_a_choice(self, monkeypatch):
        client = self._client(monkeypatch, ["wf-a", "wf-b"])
        resp = self._start(client)
        assert resp.status_code == 400
        assert "wf-a" in resp.json()["detail"] and "wf-b" in resp.json()["detail"]
        assert "workflow_name" in resp.json()["detail"]

    def test_explicit_workflow_is_validated_and_recorded(self, monkeypatch):
        from umrs.api import loop as loop_api

        client = self._client(monkeypatch, ["wf-a", "wf-b"])
        bad = self._start(client, workflow_name="wf-zzz")
        assert bad.status_code == 400
        assert "not a registered UMRS workflow" in bad.json()["detail"]
        ok = self._start(client, workflow_name="wf-b")
        assert ok.status_code == 201
        assert ok.json()["workflow_name"] == "wf-b"
        meta = loop_api._loop_store.get(ok.json()["loop_id"]).metadata
        assert meta["workflow_resolution"]["source"] == "request"

    def test_catalog_unavailable_is_503_unless_a_name_is_given(self, monkeypatch):
        from umrs.api import loop as loop_api

        client = self._client(monkeypatch, RuntimeError("catalog down"))
        assert self._start(client).status_code == 503
        ok = self._start(client, workflow_name="wf-a")
        assert ok.status_code == 201
        meta = loop_api._loop_store.get(ok.json()["loop_id"]).metadata
        assert meta["workflow_resolution"]["source"] == "request-unverified"

    def test_agentic_loops_skip_resolution(self, monkeypatch):
        from umrs.api import loop as loop_api

        client = self._client(monkeypatch, [])  # would 400 for a phased loop
        resp = self._start(client, agentic=True)
        assert resp.status_code == 201
        assert resp.json()["workflow_name"] == ""
        assert (
            "workflow_resolution" not in loop_api._loop_store.get(resp.json()["loop_id"]).metadata
        )


def test_simulated_engine_needs_no_workflow():
    """The default (simulated) fixture engine never consults the registry."""
    from umrs.api import loop as loop_api

    client = _make_app()
    resp = client.post("/umrs/loops", json={"project_id": "p", "research_goal": "g"})
    assert resp.status_code == 201
    assert resp.json()["workflow_name"] == ""
    assert "workflow_resolution" not in loop_api._loop_store.get(resp.json()["loop_id"]).metadata


def test_engine_uses_loop_level_workflow_default():
    """G-P1-3: the engine falls back to metadata.workflow_name when an experiment has none."""
    from umrs.knowledge.execution_port import FakeWorkflowExecutionPort
    from umrs.knowledge.loop import (
        AutonomousResearchLoop,
        LoopIteration,
        LoopPhase,
        LoopStatus,
        ResearchLoop,
    )
    from umrs.knowledge.planner import ResearchExperiment

    port = FakeWorkflowExecutionPort()
    engine = AutonomousResearchLoop(simulated=False, workflow_port=port)
    loop = ResearchLoop(
        loop_id="loop-wf",
        project_id="p",
        research_goal="g",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.EXPERIMENT_EXECUTION,
        max_iterations=1,
        current_iteration=1,
        metadata={"workflow_name": "wf-default"},
    )
    iteration = LoopIteration(iteration_id="it", loop_id="loop-wf", iteration_number=1)
    engine._execute_experiment(loop, iteration, ResearchExperiment(experiment_id="e1"))
    assert port.calls[0][1][0] == "wf-default"
    assert iteration.durable_job_id == "job-fake-durable"


class TestLoopVenueStartLane:
    """WP-05 r2 start leg: venue vocabulary at loop start (cluster wiring)."""

    def test_cluster_venue_rides_loop_metadata_and_job_spec(self, monkeypatch):
        from umrs.server import durable_job_runtime as djr

        client = _make_app()
        calls: list[dict] = []

        class _FakeRuntime:
            async def submit_job_sync(self, **kw):
                calls.append(kw)
                return "job-cluster-1"

        monkeypatch.setattr(djr, "get_runtime", lambda: _FakeRuntime())
        resp = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-v",
                "research_goal": "cluster venue probe",
                "max_iterations": 1,
                "mode": "auto",
                "venue": "cluster:rccs-front",
            },
        )
        assert resp.status_code == 201, resp.text
        loop_id = resp.json()["loop_id"]
        assert calls and calls[0]["spec"]["venue"] == "cluster:rccs-front"
        meta = client.get(f"/umrs/loops/{loop_id}").json()["metadata"]
        assert meta["venue"] == "cluster:rccs-front"
        assert meta["hosting"] == "job"

    def test_default_frontend_stays_byte_compatible_no_venue_key(self, monkeypatch):
        from umrs.server import durable_job_runtime as djr

        client = _make_app()
        calls: list[dict] = []

        class _FakeRuntime:
            async def submit_job_sync(self, **kw):
                calls.append(kw)
                return "job-front-1"

        monkeypatch.setattr(djr, "get_runtime", lambda: _FakeRuntime())
        resp = client.post(
            "/umrs/loops",
            json={"project_id": "proj-f2", "research_goal": "front default"},
        )
        assert resp.status_code == 201, resp.text
        assert "venue" not in calls[0]["spec"]
        meta = client.get(f"/umrs/loops/{resp.json()['loop_id']}").json()["metadata"]
        assert meta["venue"] == "frontend"

    def test_bad_venue_rejected_closed_at_start(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-b",
                "research_goal": "garbage venue",
                "venue": "cluster:",
            },
        )
        assert resp.status_code == 400

        agentic_cluster = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-b",
                "research_goal": "agentic on cluster",
                "agentic": True,
                "venue": "cluster:rccs-front",
            },
        )
        assert agentic_cluster.status_code == 400


class TestDriverNoteADR064:
    """ADR-064: agentic starts are stamped agent-deprecated (engine runner is
    the one server driver of record); engine starts carry no marker."""

    def test_agentic_start_marks_driver_note(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-agentic",
                "research_goal": "agentic loop",
                "max_iterations": 2,
                "agentic": True,
            },
        )
        assert resp.status_code == 201
        loop_id = resp.json()["loop_id"]
        from umrs.api import loop as loop_mod

        stored = loop_mod._loop_store.get(loop_id)
        assert stored.metadata.get("driver_note") == "agent-deprecated"

    def test_engine_start_has_no_driver_note(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-engine",
                "research_goal": "engine loop",
                "max_iterations": 2,
            },
        )
        assert resp.status_code == 201
        loop_id = resp.json()["loop_id"]
        from umrs.api import loop as loop_mod

        stored = loop_mod._loop_store.get(loop_id)
        assert "driver_note" not in stored.metadata


class TestTraceIdD52:
    """D-52: loops carry one trace id (= loop id) from start; findings mint
    with it (finding_ingest metadata)."""

    def test_start_stamps_trace_id(self):
        client = _make_app()
        resp = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-trace",
                "research_goal": "trace me",
                "max_iterations": 2,
            },
        )
        assert resp.status_code == 201
        loop_id = resp.json()["loop_id"]
        from umrs.api import loop as loop_mod

        stored = loop_mod._loop_store.get(loop_id)
        assert stored.metadata.get("trace_id") == loop_id


# ── D-43/D-52: findings pin onto the single active catalog (api layer) ──


class _FakePool:
    def __init__(self, conn):
        self._conn = conn
        self.leased = 0

    def getconn(self):
        self.leased += 1
        return self._conn

    def putconn(self, _conn):
        self.leased -= 1


def _catalog_row_tuple(c):
    """The 19-column scientific_catalogs row shape the store reads."""
    import json as _json

    return (
        c["catalog_id"],
        c["project_id"],
        "n",
        "0.1.0",
        "",
        "",
        "[]",
        "[]",
        "[]",
        "[]",
        "[]",
        "",
        _json.dumps(c.get("provenance") or {}),
        "",
        "[]",
        "o",
        "active",
        None,
        None,
    )


def _catalog_rows(catalogs, key, value):
    """Row tuples whose ``key`` equals ``value`` (None = every row)."""
    return [_catalog_row_tuple(c) for c in catalogs if value is None or c[key] == value]


def _append_pins(catalogs, pin_arr, cid):
    import json as _json

    for c in catalogs:
        if c["catalog_id"] == cid:
            prov = c.setdefault("provenance", {"pins": []})
            prov["pins"] = prov["pins"] + _json.loads(pin_arr)


def _pin_catalog_conn(catalogs):
    class _Cur:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=None):
            if "jsonb_set" in sql:
                _append_pins(self._conn.catalogs, *params)
            elif "WHERE catalog_id = %s" in sql:
                self._result = _catalog_rows(self._conn.catalogs, "catalog_id", params[0])
            elif "FROM scientific_catalogs" in sql:
                self._result = _catalog_rows(catalogs, "project_id", params[0] if params else None)

        def fetchone(self):
            return (
                getattr(self, "_result", [None]).pop(0) if getattr(self, "_result", None) else None
            )

        def fetchall(self):
            out = getattr(self, "_result", [])
            self._result = []
            return out

    class _Conn:
        def __init__(self, catalogs):
            self.catalogs = catalogs

        def cursor(self):
            return _Cur(self)

        def commit(self):
            pass

    return _Conn(catalogs)


def _pin_findings_via_pool(catalogs, project_id, pins):
    from umrs.api.loop import _pin_findings_onto_catalog

    conn = _pin_catalog_conn(catalogs)
    pool = _FakePool(conn)
    _pin_findings_onto_catalog(pool, project_id, pins)
    assert pool.leased == 0  # conn always returned
    return conn.catalogs


def test_findings_pin_onto_single_active_catalog():
    catalogs = [{"catalog_id": "sc-1", "project_id": "p1", "provenance": {"pins": []}}]
    rows = _pin_findings_via_pool(catalogs, "p1", [("f-1", "h-1 wf=wf-1"), ("f-2", "h-2 wf=wf-1")])
    pins = rows[0]["provenance"]["pins"]
    assert [(p["kind"], p["ref"], p["note"]) for p in pins] == [
        ("finding", "f-1", "h-1 wf=wf-1"),
        ("finding", "f-2", "h-2 wf=wf-1"),
    ]


def test_findings_pin_never_guesses_ambiguous_catalogs():
    catalogs = [
        {"catalog_id": "sc-1", "project_id": "p1"},
        {"catalog_id": "sc-2", "project_id": "p1"},
    ]
    rows = _pin_findings_via_pool(catalogs, "p1", [("f-1", "n")])
    assert all(not (c.get("provenance") or {}).get("pins") for c in rows)


def test_findings_pin_dedupes_existing_refs():
    catalogs = [
        {
            "catalog_id": "sc-1",
            "project_id": "p1",
            "provenance": {"pins": [{"kind": "finding", "ref": "f-1", "note": "old"}]},
        }
    ]
    rows = _pin_findings_via_pool(catalogs, "p1", [("f-1", "new"), ("f-2", "n")])
    pins = rows[0]["provenance"]["pins"]
    assert [p["ref"] for p in pins] == ["f-1", "f-2"]
    assert pins[0]["note"] == "old"  # append-only, never rewritten


# ── D-33 tail: loops inherit the project's persisted ask-policy default ──


def _policy_request(mode="", triggers=None):
    from umrs.api.loop import StartLoopRequest

    return StartLoopRequest(
        project_id="proj-ask",
        research_goal="goal",
        mode=mode,
        pause_triggers=triggers or [],
    )


def _policy_http(monkeypatch, row=None):
    from types import SimpleNamespace

    pool = SimpleNamespace()

    def getconn():
        return SimpleNamespace(cursor=lambda: _FakeAskConn(row).cursor())

    pool.getconn = getconn
    pool.putconn = lambda _c: None
    http = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(umrs_db_pool=pool)))
    return http


class _FakeAskConn:
    def __init__(self, row):
        self._row = row

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        del sql, params

    def fetchone(self):
        return self._row


def test_unset_mode_inherits_persisted_default(monkeypatch):
    from umrs.api.loop import _apply_project_ask_policy

    row = ("proj-ask", "p", "n", "active", None, None, None, "sensitive", "profile", ["cost"])
    http = _policy_http(monkeypatch, row)
    req = _policy_request(mode="")
    applied = _apply_project_ask_policy(http, req)
    assert applied.mode == "profile"
    assert list(applied.pause_triggers) == ["cost"]


def test_explicit_mode_wins(monkeypatch):
    from umrs.api.loop import _apply_project_ask_policy

    row = ("proj-ask", "p", "n", "active", None, None, None, "sensitive", "ask", [])
    http = _policy_http(monkeypatch, row)
    req = _policy_request(mode="auto")
    applied = _apply_project_ask_policy(http, req)
    assert applied.mode == "auto"  # explicit wins, never overridden


def test_absent_project_row_is_unchanged(monkeypatch):
    from umrs.api.loop import _apply_project_ask_policy

    http = _policy_http(monkeypatch, row=None)
    req = _policy_request(mode="")
    assert _apply_project_ask_policy(http, req).mode == ""  # never invented


def test_unreadable_pool_is_unchanged(monkeypatch):
    from types import SimpleNamespace

    from umrs.api.loop import _apply_project_ask_policy

    http = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(umrs_db_pool=None)))
    req = _policy_request(mode="")
    assert _apply_project_ask_policy(http, req).mode == ""
