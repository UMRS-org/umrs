"""Tests for lineage graph assembly and API (MEM-B4)."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys

import pytest
from datetime import datetime, timezone, timedelta

# Direct module imports to avoid circular import through umrs.jobs.__init__
_src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, _src)

_ets_spec = importlib.util.spec_from_file_location(
    "umrs.jobs.execution_trace_store",
    os.path.join(_src, "umrs", "jobs", "execution_trace_store.py"),
)
_ets_mod = importlib.util.module_from_spec(_ets_spec)
sys.modules["umrs.jobs.execution_trace_store"] = _ets_mod
_ets_spec.loader.exec_module(_ets_mod)
TraceRecord = _ets_mod.TraceRecord

_lg_spec = importlib.util.spec_from_file_location(
    "umrs.jobs.lineage_graph",
    os.path.join(_src, "umrs", "jobs", "lineage_graph.py"),
)
_lg_mod = importlib.util.module_from_spec(_lg_spec)
sys.modules["umrs.jobs.lineage_graph"] = _lg_mod
_lg_spec.loader.exec_module(_lg_mod)

LineageGraph = _lg_mod.LineageGraph
LineageGraphMetrics = _lg_mod.LineageGraphMetrics
LineageNode = _lg_mod.LineageNode
build_lineage_graph = _lg_mod.build_lineage_graph


def _ts(offset_seconds: int = 0) -> datetime:
    """Create a timestamp with offset from a base time."""
    base = datetime(2026, 5, 18, 10, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


def _trace(
    trace_id: str,
    job_id: str = "job-123",
    step_name: str = "step",
    step_type: str = "generic",
    parent_trace_id: str | None = None,
    execution_id: str | None = None,
    status: str = "succeeded",
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    model_ref: dict | None = None,
    artifact_refs: list | None = None,
    error: str | None = None,
) -> TraceRecord:
    """Helper to create a TraceRecord for testing."""
    return TraceRecord(
        trace_id=trace_id,
        job_id=job_id,
        step_name=step_name,
        step_type=step_type,
        parent_trace_id=parent_trace_id,
        execution_id=execution_id,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        model_ref=model_ref,
        artifact_refs=artifact_refs or [],
        error=error,
        created_at=started_at or _ts(),
    )


class TestBuildLineageGraphEmpty:
    """Tests for empty or minimal graphs."""

    def test_empty_traces(self):
        graph = build_lineage_graph("job-123", [])
        assert graph.job_id == "job-123"
        assert graph.nodes == []
        assert graph.edges == []
        assert graph.metrics.total_nodes == 0

    def test_single_trace(self):
        traces = [_trace("t1", step_name="embed", started_at=_ts(0), finished_at=_ts(5))]
        graph = build_lineage_graph("job-123", traces)

        assert len(graph.nodes) == 1
        assert graph.nodes[0].trace_id == "t1"
        assert graph.nodes[0].step_name == "embed"
        assert graph.edges == []
        assert graph.root_ids == ["t1"]
        assert graph.leaf_ids == ["t1"]
        assert graph.metrics.total_nodes == 1
        assert graph.metrics.depth == 1
        assert graph.metrics.root_count == 1
        assert graph.metrics.leaf_count == 1


class TestBuildLineageGraphLinear:
    """Tests for linear (sequential) DAGs."""

    def test_two_step_chain(self):
        traces = [
            _trace("t1", step_name="fetch", started_at=_ts(0), finished_at=_ts(3)),
            _trace("t2", step_name="process", parent_trace_id="t1",
                   started_at=_ts(3), finished_at=_ts(8)),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert graph.edges[0].source == "t1"
        assert graph.edges[0].target == "t2"
        assert graph.root_ids == ["t1"]
        assert graph.leaf_ids == ["t2"]
        assert graph.metrics.depth == 2
        assert graph.metrics.total_duration_seconds == 8.0

    def test_three_step_chain(self):
        traces = [
            _trace("t1", step_name="load", started_at=_ts(0), finished_at=_ts(2)),
            _trace("t2", step_name="transform", parent_trace_id="t1",
                   started_at=_ts(2), finished_at=_ts(5)),
            _trace("t3", step_name="store", parent_trace_id="t2",
                   started_at=_ts(5), finished_at=_ts(7)),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert graph.metrics.depth == 3
        assert graph.metrics.total_nodes == 3
        assert graph.metrics.total_edges == 2
        assert graph.metrics.critical_path_seconds == 7.0


class TestBuildLineageGraphBranching:
    """Tests for DAGs with parallel branches."""

    def test_fan_out(self):
        """One root with two children."""
        traces = [
            _trace("t1", step_name="split", started_at=_ts(0), finished_at=_ts(2)),
            _trace("t2", step_name="branch_a", parent_trace_id="t1",
                   started_at=_ts(2), finished_at=_ts(5)),
            _trace("t3", step_name="branch_b", parent_trace_id="t1",
                   started_at=_ts(2), finished_at=_ts(7)),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.root_ids == ["t1"]
        assert set(graph.leaf_ids) == {"t2", "t3"}
        assert graph.metrics.depth == 2
        assert graph.metrics.leaf_count == 2
        # Critical path: t1(2s) + t3(5s) = 7s
        assert graph.metrics.critical_path_seconds == 7.0

    def test_diamond_dag(self):
        """Diamond shape: root -> A,B -> merge."""
        traces = [
            _trace("t1", step_name="start", started_at=_ts(0), finished_at=_ts(1)),
            _trace("t2", step_name="left", parent_trace_id="t1",
                   started_at=_ts(1), finished_at=_ts(4)),
            _trace("t3", step_name="right", parent_trace_id="t1",
                   started_at=_ts(1), finished_at=_ts(3)),
            _trace("t4", step_name="merge", parent_trace_id="t2",
                   started_at=_ts(4), finished_at=_ts(6)),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert graph.metrics.depth == 3
        assert graph.root_ids == ["t1"]
        # t3 is a leaf (no children), t4 is a leaf
        assert set(graph.leaf_ids) == {"t3", "t4"}
        # Critical path: t1(1s) + t2(3s) + t4(2s) = 6
        assert graph.metrics.critical_path_seconds == 6.0


class TestBuildLineageGraphStatus:
    """Tests for status counting in metrics."""

    def test_mixed_statuses(self):
        traces = [
            _trace("t1", step_name="s1", status="succeeded"),
            _trace("t2", step_name="s2", status="failed", error="timeout"),
            _trace("t3", step_name="s3", status="running"),
            _trace("t4", step_name="s4", status="pending"),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert graph.metrics.succeeded_count == 1
        assert graph.metrics.failed_count == 1
        assert graph.metrics.running_count == 1
        assert graph.metrics.total_nodes == 4


class TestBuildLineageGraphMetadata:
    """Tests for model refs, artifacts, execution_id propagation."""

    def test_model_ref_in_node(self):
        model = {"name": "e5-base-v2", "path": "/models/e5", "version": "1.0"}
        traces = [
            _trace("t1", step_name="embed", model_ref=model,
                   started_at=_ts(0), finished_at=_ts(3)),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert graph.nodes[0].model_ref == model

    def test_artifact_refs_in_node(self):
        artifacts = [
            {"uri": "s3://bucket/out.csv", "media_type": "text/csv", "size_bytes": 1024}
        ]
        traces = [
            _trace("t1", step_name="export", artifact_refs=artifacts),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert graph.nodes[0].artifact_refs == artifacts

    def test_execution_id_propagation(self):
        traces = [
            _trace("t1", step_name="s1", execution_id="exec-abc"),
            _trace("t2", step_name="s2", execution_id="exec-abc", parent_trace_id="t1"),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert graph.execution_id == "exec-abc"


class TestBuildLineageGraphEdgeCases:
    """Edge cases and robustness tests."""

    def test_orphan_parent_reference(self):
        """Parent trace_id references a trace NOT in the provided list."""
        traces = [
            _trace("t2", step_name="child", parent_trace_id="t-missing"),
        ]
        graph = build_lineage_graph("job-123", traces)

        # Should not create an edge for the missing parent
        assert graph.edges == []
        assert graph.root_ids == ["t2"]
        assert graph.leaf_ids == ["t2"]

    def test_no_timing_data(self):
        """Traces without timing should still produce a valid graph."""
        traces = [
            _trace("t1", step_name="s1"),
            _trace("t2", step_name="s2", parent_trace_id="t1"),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert graph.metrics.total_duration_seconds is None
        assert graph.metrics.critical_path_seconds == 0.0
        assert graph.metrics.depth == 2

    def test_multiple_roots(self):
        """Parallel roots (no shared parent)."""
        traces = [
            _trace("t1", step_name="pipeline_a", started_at=_ts(0), finished_at=_ts(3)),
            _trace("t2", step_name="pipeline_b", started_at=_ts(0), finished_at=_ts(5)),
        ]
        graph = build_lineage_graph("job-123", traces)

        assert graph.metrics.root_count == 2
        assert graph.metrics.leaf_count == 2
        assert graph.metrics.depth == 1


class TestLineageGraphSerialization:
    """Tests for to_dict() serialization."""

    def test_full_graph_to_dict(self):
        traces = [
            _trace("t1", step_name="a", started_at=_ts(0), finished_at=_ts(2)),
            _trace("t2", step_name="b", parent_trace_id="t1",
                   started_at=_ts(2), finished_at=_ts(5)),
        ]
        graph = build_lineage_graph("job-123", traces)
        d = graph.to_dict()

        assert d["job_id"] == "job-123"
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["edges"][0] == {"source": "t1", "target": "t2"}
        assert d["root_ids"] == ["t1"]
        assert d["leaf_ids"] == ["t2"]
        assert d["metrics"]["total_nodes"] == 2
        assert d["metrics"]["depth"] == 2

    def test_empty_graph_to_dict(self):
        graph = build_lineage_graph("job-x", [])
        d = graph.to_dict()

        assert d["job_id"] == "job-x"
        assert d["nodes"] == []
        assert d["edges"] == []
        assert d["metrics"]["total_nodes"] == 0
