"""Lineage graph assembly from execution traces (MEM-B4).

Builds a structured DAG (directed acyclic graph) representation from
execution traces for a given job.  The graph shows step-level execution
flow with computed metrics for operator and research introspection.

This module lives in ``umrs.jobs`` because it reads from the execution
trace store and does not import from ``umrs.api`` or ``umrs.server``.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .execution_trace_store import TraceRecord

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


@dataclass
class _LineageParts:
    nodes: list[LineageNode] = field(default_factory=list)
    edges: list[LineageEdge] = field(default_factory=list)
    children_map: dict[str, list[str]] = field(default_factory=dict)
    parent_set: set[str] = field(default_factory=set)
    durations: dict[str, float] = field(default_factory=dict)
    execution_id: str | None = None


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

    parts = _collect_lineage_parts(traces)
    root_ids = _root_ids(traces, parts.parent_set)
    leaf_ids = _leaf_ids(traces, parts.children_map)

    return LineageGraph(
        job_id=job_id,
        execution_id=parts.execution_id,
        nodes=parts.nodes,
        edges=parts.edges,
        root_ids=root_ids,
        leaf_ids=leaf_ids,
        metrics=_build_metrics(traces, parts, root_ids, leaf_ids),
    )


def _collect_lineage_parts(traces: list[TraceRecord]) -> _LineageParts:
    parts = _LineageParts()
    trace_ids: set[str] = {t.trace_id for t in traces}
    for trace in traces:
        parts.nodes.append(_trace_to_node(trace))
        parts.execution_id = parts.execution_id or trace.execution_id
        _record_trace_duration(parts, trace)
        _record_trace_edge(parts, trace, trace_ids)
    return parts


def _record_trace_duration(parts: _LineageParts, trace: TraceRecord) -> None:
    duration = _duration_seconds(trace.started_at, trace.finished_at)
    if duration is not None:
        parts.durations[trace.trace_id] = duration


def _record_trace_edge(parts: _LineageParts, trace: TraceRecord, trace_ids: set[str]) -> None:
    if not trace.parent_trace_id or trace.parent_trace_id not in trace_ids:
        return
    parts.edges.append(LineageEdge(source=trace.parent_trace_id, target=trace.trace_id))
    parts.parent_set.add(trace.trace_id)
    parts.children_map.setdefault(trace.parent_trace_id, []).append(trace.trace_id)


def _root_ids(traces: list[TraceRecord], parent_set: set[str]) -> list[str]:
    return [trace.trace_id for trace in traces if trace.trace_id not in parent_set]


def _leaf_ids(traces: list[TraceRecord], children_map: dict[str, list[str]]) -> list[str]:
    return [trace.trace_id for trace in traces if trace.trace_id not in children_map]


def _build_metrics(
    traces: list[TraceRecord],
    parts: _LineageParts,
    root_ids: list[str],
    leaf_ids: list[str],
) -> LineageGraphMetrics:
    depth_cache: dict[str, int] = {}
    status_counts = Counter(trace.status for trace in traces)
    return LineageGraphMetrics(
        total_nodes=len(parts.nodes),
        total_edges=len(parts.edges),
        depth=_max_depth(root_ids, parts.children_map, depth_cache),
        root_count=len(root_ids),
        leaf_count=len(leaf_ids),
        succeeded_count=status_counts["succeeded"],
        failed_count=status_counts["failed"],
        running_count=status_counts["running"],
        total_duration_seconds=_total_duration_seconds(traces),
        critical_path_seconds=_compute_critical_path(root_ids, parts.children_map, parts.durations),
    )


def _max_depth(
    root_ids: list[str],
    children_map: dict[str, list[str]],
    depth_cache: dict[str, int],
) -> int:
    return max((_compute_depth(root_id, children_map, depth_cache) for root_id in root_ids), default=0)


def _total_duration_seconds(traces: list[TraceRecord]) -> float | None:
    starts = [trace.started_at for trace in traces if trace.started_at]
    ends = [trace.finished_at for trace in traces if trace.finished_at]
    if starts and ends:
        return _duration_seconds(min(starts), max(ends))
    return None
