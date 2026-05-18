"""Lineage graph assembly from execution traces (MEM-B4).

Builds a structured DAG (directed acyclic graph) representation from
execution traces for a given job.  The graph shows step-level execution
flow with computed metrics for operator and research introspection.

This module lives in ``umrs.jobs`` because it reads from the execution
trace store and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .execution_trace_store import ExecutionTraceStore, TraceRecord

logger = logging.getLogger(__name__)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _duration_seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start and end:
        return round((end - start).total_seconds(), 3)
    return None


@dataclass
class LineageNode:
    """A single node in the lineage graph (one execution trace step)."""

    trace_id: str
    step_name: str
    step_type: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    model_ref: dict[str, Any] | None = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    retrieval_refs: list[dict[str, Any]] = field(default_factory=list)
    routing_decision: dict[str, Any] | None = None
    inputs_hash: str | None = None
    outputs_hash: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "step_name": self.step_name,
            "step_type": self.step_type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "model_ref": self.model_ref,
            "artifact_refs": self.artifact_refs,
            "tool_calls": self.tool_calls,
            "retrieval_refs": self.retrieval_refs,
            "routing_decision": self.routing_decision,
            "inputs_hash": self.inputs_hash,
            "outputs_hash": self.outputs_hash,
            "error": self.error,
        }


@dataclass
class LineageEdge:
    """A directed edge in the lineage graph (parent -> child)."""

    source: str  # parent trace_id
    target: str  # child trace_id

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "target": self.target}


@dataclass
class LineageGraphMetrics:
    """Computed metrics for the lineage graph."""

    total_nodes: int = 0
    total_edges: int = 0
    depth: int = 0
    root_count: int = 0
    leaf_count: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    running_count: int = 0
    total_duration_seconds: float | None = None
    critical_path_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "depth": self.depth,
            "root_count": self.root_count,
            "leaf_count": self.leaf_count,
            "succeeded_count": self.succeeded_count,
            "failed_count": self.failed_count,
            "running_count": self.running_count,
            "total_duration_seconds": self.total_duration_seconds,
            "critical_path_seconds": self.critical_path_seconds,
        }


@dataclass
class LineageGraph:
    """Full lineage graph for a job."""

    job_id: str
    execution_id: str | None = None
    nodes: list[LineageNode] = field(default_factory=list)
    edges: list[LineageEdge] = field(default_factory=list)
    root_ids: list[str] = field(default_factory=list)
    leaf_ids: list[str] = field(default_factory=list)
    metrics: LineageGraphMetrics = field(default_factory=LineageGraphMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "root_ids": self.root_ids,
            "leaf_ids": self.leaf_ids,
            "metrics": self.metrics.to_dict(),
        }


def _trace_to_node(trace: TraceRecord) -> LineageNode:
    """Convert a TraceRecord to a LineageNode."""
    return LineageNode(
        trace_id=trace.trace_id,
        step_name=trace.step_name,
        step_type=trace.step_type,
        status=trace.status,
        started_at=_iso(trace.started_at),
        finished_at=_iso(trace.finished_at),
        duration_seconds=_duration_seconds(trace.started_at, trace.finished_at),
        model_ref=trace.model_ref,
        artifact_refs=trace.artifact_refs,
        tool_calls=trace.tool_calls,
        retrieval_refs=trace.retrieval_refs,
        routing_decision=trace.routing_decision,
        inputs_hash=trace.inputs_hash,
        outputs_hash=trace.outputs_hash,
        error=trace.error,
    )


def _compute_depth(
    node_id: str,
    children_map: dict[str, list[str]],
    cache: dict[str, int],
) -> int:
    """Compute the depth of the DAG below a given node (inclusive)."""
    if node_id in cache:
        return cache[node_id]
    kids = children_map.get(node_id, [])
    if not kids:
        cache[node_id] = 1
        return 1
    max_child = max(_compute_depth(c, children_map, cache) for c in kids)
    cache[node_id] = 1 + max_child
    return cache[node_id]


def _compute_critical_path(
    roots: list[str],
    children_map: dict[str, list[str]],
    durations: dict[str, float],
) -> float | None:
    """Compute the longest (critical) path through the DAG by duration."""
    if not roots:
        return None

    longest: dict[str, float] = {}

    def _longest_path(node_id: str) -> float:
        if node_id in longest:
            return longest[node_id]
        dur = durations.get(node_id, 0.0)
        kids = children_map.get(node_id, [])
        if not kids:
            longest[node_id] = dur
            return dur
        max_child = max(_longest_path(c) for c in kids)
        longest[node_id] = dur + max_child
        return longest[node_id]

    return max(_longest_path(r) for r in roots)


def build_lineage_graph(
    job_id: str,
    traces: list[TraceRecord],
) -> LineageGraph:
    """Build a lineage graph from a list of traces for a job.

    Pure assembly function — no database access. The caller provides
    the traces (already filtered for the job).
    """
    if not traces:
        return LineageGraph(job_id=job_id, metrics=LineageGraphMetrics())

    # Build nodes and edge maps
    nodes: list[LineageNode] = []
    edges: list[LineageEdge] = []
    trace_ids: set[str] = {t.trace_id for t in traces}
    children_map: dict[str, list[str]] = {}
    parent_set: set[str] = set()
    durations: dict[str, float] = {}

    execution_id: str | None = None

    for trace in traces:
        nodes.append(_trace_to_node(trace))

        if trace.execution_id and not execution_id:
            execution_id = trace.execution_id

        dur = _duration_seconds(trace.started_at, trace.finished_at)
        if dur is not None:
            durations[trace.trace_id] = dur

        if trace.parent_trace_id and trace.parent_trace_id in trace_ids:
            edges.append(LineageEdge(
                source=trace.parent_trace_id,
                target=trace.trace_id,
            ))
            parent_set.add(trace.trace_id)
            children_map.setdefault(trace.parent_trace_id, []).append(trace.trace_id)

    # Root nodes have no parent within this job's traces
    root_ids = [t.trace_id for t in traces if t.trace_id not in parent_set]
    # Leaf nodes have no children within this job's traces
    leaf_ids = [t.trace_id for t in traces if t.trace_id not in children_map]

    # Compute metrics
    depth_cache: dict[str, int] = {}
    max_depth = 0
    for root_id in root_ids:
        d = _compute_depth(root_id, children_map, depth_cache)
        max_depth = max(max_depth, d)

    # Status counts
    succeeded = sum(1 for t in traces if t.status == "succeeded")
    failed = sum(1 for t in traces if t.status == "failed")
    running = sum(1 for t in traces if t.status == "running")

    # Total duration: earliest start to latest finish
    starts = [t.started_at for t in traces if t.started_at]
    ends = [t.finished_at for t in traces if t.finished_at]
    total_dur = None
    if starts and ends:
        total_dur = _duration_seconds(min(starts), max(ends))

    critical_path = _compute_critical_path(root_ids, children_map, durations)

    metrics = LineageGraphMetrics(
        total_nodes=len(nodes),
        total_edges=len(edges),
        depth=max_depth,
        root_count=len(root_ids),
        leaf_count=len(leaf_ids),
        succeeded_count=succeeded,
        failed_count=failed,
        running_count=running,
        total_duration_seconds=total_dur,
        critical_path_seconds=critical_path,
    )

    return LineageGraph(
        job_id=job_id,
        execution_id=execution_id,
        nodes=nodes,
        edges=edges,
        root_ids=root_ids,
        leaf_ids=leaf_ids,
        metrics=metrics,
    )
