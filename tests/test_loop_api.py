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
def _clean_loop_store():
    from umrs.api import loop as loop_api
    from umrs.api.loop import clear_loop_store
    from umrs.knowledge.loop import AutonomousResearchLoop

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
            json={"project_id": "proj-h", "research_goal": "host me", "max_iterations": 2, "mode": "ask"},
        )
        assert resp.status_code == 201
        loop_id = resp.json()["loop_id"]
        assert seen == [{"loop_id": loop_id, "mode": "ask"}]
        meta = client.get(f"/umrs/loops/{loop_id}").json()["metadata"]
        assert meta["hosting"] == "job"
        assert meta["durable_job_id"] == "job-hosted-1"
        assert engine_loop_runner.get(loop_id) is None  # the job backend drives it, not the route

    def test_no_runtime_falls_back_to_thread_hosting(self):
        client = _make_app()
        resp = client.post("/umrs/loops", json={"project_id": "proj-t", "research_goal": "thread me"})
        assert resp.status_code == 201
        meta = client.get(f"/umrs/loops/{resp.json()['loop_id']}").json()["metadata"]
        assert meta["hosting"] == "thread"
        assert meta["driver"] == "engine"

    def test_submission_failure_falls_back_and_is_recorded(self):
        client = _make_app()

        async def _boom(loop):  # noqa: ARG001
            raise RuntimeError("runtime exploded")

        client.app.state.loop_job_submitter = _boom
        resp = client.post("/umrs/loops", json={"project_id": "proj-f", "research_goal": "fallback"})
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
