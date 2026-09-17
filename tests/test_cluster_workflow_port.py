"""Cluster venue workflow port pins (WP-05 r2 venue leg).

Cluster-side real-mode executions have to travel the gateway lane: these
lock the port's transport contract (trigger id extraction, terminal-shaped
wait, honest missing-linkage raising) with an in-memory transport so no
HTTP actually leaves test-land.
"""

from __future__ import annotations

from umrs.knowledge.cluster_workflow_port import ClusterHttpWorkflowExecutionPort


class _FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.execution_status = "pending"
        self.results_status = "succeeded"
        self.durable_job_id = ""

    def post(self, path: str, body: dict):
        self.calls.append(("POST", path, body))
        if path.endswith("/trigger"):
            return {"execution_id": "exe-123"}
        if path.endswith("/cancel"):
            return {"status": "cancelled"}
        raise AssertionError(f"unexpected POST {path}")

    def get(self, path: str):
        self.calls.append(("GET", path))
        base = {"status": self.execution_status}
        if self.durable_job_id:
            base["durable_job_id"] = self.durable_job_id
        if path.endswith("/results"):
            # results payload uses results_status (base is status-route only)
            res = dict(base)
            res["status"] = self.results_status
            return res
        return {"execution": base}

    def put(self, path: str, body: dict):
        self.calls.append(("PUT", path, body))
        return {}


def test_trigger_returns_extracted_execution_id():
    t = _FakeTransport()
    port = ClusterHttpWorkflowExecutionPort(transport=t)
    assert port.trigger("probe-wf", {"x": 1}) == "exe-123"
    assert t.calls[0][1] == "/probe-wf/trigger"


def test_trigger_missing_id_raises_honestly():
    class T(_FakeTransport):
        def post(self, path: str, body: dict):
            return {"weird": "shape"}

    import pytest

    with pytest.raises(RuntimeError, match="no execution id"):
        ClusterHttpWorkflowExecutionPort(transport=T()).trigger("probe-wf", {})


def test_wait_blocks_until_terminal_status():
    t = _FakeTransport()
    seq = iter(["pending", "running", "succeeded"])

    def get(path: str):
        return {"execution": {"status": next(seq)}}

    t.get = get  # type: ignore[assignment]
    port = ClusterHttpWorkflowExecutionPort(transport=t, poll_interval=0.0)
    out = port.wait("exe-123", timeout=5)
    assert out["status"] == "succeeded"


def test_results_terminal_without_durable_id_raises():
    t = _FakeTransport()
    t.results_status = "succeeded"
    port = ClusterHttpWorkflowExecutionPort(transport=t)
    import pytest

    with pytest.raises(RuntimeError, match="durable_job_id"):
        port.results("exe-123")


def test_results_terminal_with_durable_id_passes():
    t = _FakeTransport()
    t.results_status = "succeeded"
    t.durable_job_id = "job-x"
    out = ClusterHttpWorkflowExecutionPort(transport=t).results("exe-123")
    assert out["durable_job_id"] == "job-x"


def test_cancel_posts_to_cancel_route():
    t = _FakeTransport()
    out = ClusterHttpWorkflowExecutionPort(transport=t).cancel("exe-123")
    assert out["status"] == "cancelled"
    assert t.calls[-1][1] == "/executions/exe-123/cancel"
