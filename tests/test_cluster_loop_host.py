"""WP-05 round 2 unit pins: HttpLoopStore adapter shape + cluster entry."""

from __future__ import annotations

import pytest
from umrs.knowledge.http_loop_store import HttpLoopStore, _loop_from_status
from umrs.knowledge.loop import LoopPhase, LoopStatus


class _FakeTransport:
    """Requests-style transport seam for unit tests (no live gateway)."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, path: str) -> dict:
        self.calls.append(("get", {"path": path}))
        return self.responses[("GET", path)]

    def put(self, path: str, body: dict) -> dict:
        self.calls.append(("put", {"path": path, "body": body}))
        return self.responses.get(("PUT", path), {})

    def post(self, path: str, body: dict) -> dict:
        self.calls.append(("post", {"path": path, "body": body}))
        return self.responses.get(("POST", path), {})


def _load_payload(**over) -> dict:
    return {
        "project_id": "proj-x",
        "research_goal": "run workflows",
        "status": "running",
        "phase": "experiment_execution",
        "current_iteration": 2,
        "max_iterations": 9,
        "total_hypotheses_tested": 6,
        "total_experiments_run": 3,
        "total_findings": 1,
        "success_rate": 0.5,
        "started_at": "2026-09-10T01:00:00",
        "completed_at": "",
        "metadata": {"agent": False},
        **over,
    }


class TestHttpLoopStore:
    def test_load_maps_wire_to_loop_record(self):
        t = _FakeTransport({("GET", "/abc"): _load_payload()})
        store = HttpLoopStore(t)
        loop = store.load("abc")
        assert loop.loop_id == "abc"
        assert loop.project_id == "proj-x"
        assert loop.status is LoopStatus.RUNNING
        assert loop.phase is LoopPhase.EXPERIMENT_EXECUTION
        assert loop.current_iteration == 2
        assert loop.max_iterations == 9
        assert loop.total_hypotheses_tested == 6
        assert loop.total_experiments_run == 3
        assert loop.total_findings == 1
        assert loop.success_rate == 0.5
        assert loop.metadata["agent"] is False

    def test_load_rejects_error_payload_no_fabrication(self):
        t = _FakeTransport({("GET", "/ghost"): {"detail": "Loop ghost not found"}})
        with pytest.raises(KeyError, match="not found"):
            HttpLoopStore(t).load("ghost")

    def test_save_writes_owner_fields_only(self):
        t = _FakeTransport({})
        store = HttpLoopStore(t)
        loop = _loop_from_status("abc", _load_payload())
        loop.status = LoopStatus.PAUSED
        loop.phase = LoopPhase.ANALYSIS
        loop.current_iteration = 4
        loop.total_hypotheses_tested = 7
        loop.total_experiments_run = 5
        loop.total_findings = 2
        loop.success_rate = 0.4
        loop.metadata["paused_by"] = "worker-1"
        store.save(loop)
        assert len(t.calls) == 1
        verb, kw = t.calls[0]
        assert verb == "put"
        body = kw["body"]
        assert body["status"] == "paused"
        assert body["phase"] == "analysis"
        assert body["current_iteration"] == 4
        assert body["total_hypotheses_tested"] == 7
        assert body["total_experiments_run"] == 5
        assert body["total_findings"] == 2
        assert body["success_rate"] == 0.4
        assert body["metadata"] == {"paused_by": "worker-1", "agent": False}
        # Driver never ships plan/iterations/server timestamps: they stay server-side.
        assert "plan" not in body
        assert "iterations" not in body
        assert "created_at" not in body
        assert "updated_at" not in body


class TestClusterEntry:
    def test_engine_only_gate_for_agent_loops(self, monkeypatch):
        from umrs.knowledge import cluster_loop_entry

        class _Store:
            def load(self, loop_id):
                return type(
                    "L",
                    (),
                    {"loop_id": loop_id, "metadata": {"agent": True}},
                )()

        monkeypatch.setattr(cluster_loop_entry, "build_cluster_store", lambda **_kw: _Store())
        rc = cluster_loop_entry.main(["--loop-id", "ghost"])
        assert rc == 78

    def test_engine_gate_missing_store_config(self, monkeypatch):
        from umrs.knowledge import cluster_loop_entry

        def _boom(**_kw):
            raise RuntimeError("R4_GATEWAY_BASE is required")

        monkeypatch.setattr(cluster_loop_entry, "build_cluster_store", _boom)
        rc = cluster_loop_entry.main(["--loop-id", "ghost"])
        assert rc == 2
