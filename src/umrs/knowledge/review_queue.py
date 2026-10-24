"""Review queue: the proposed (D-35 tier) rows of one knowledge graph.

Knowledge plan K1.5 (2026-10-24, minimal review): ``r4 knowledge review``
lists what waits for a person to establish. No route exposed the tier: the
graph content read and ``list_nodes`` select no ``status``, so a proposed
loop finding looked like every other row. This read returns only rows with
``status = 'proposed'`` in one graph (legacy ``graph_id = ''`` rows count
as members of their project's default graph, as everywhere else), newest
first, bounded. It writes nothing; promotion stays on the human-only
establish routes.

Each row carries a ``run_id`` so a client can promote one run's rows in one
action (ADR-084 D-2 "establish run"): the loop id for research-loop rows,
else the writing job or the source artifact when the row records one.

Review of K1 part A (2026-10-24): the listing takes a ``run_id`` filter, so
"establish run" reaches every row of a run however many newer rows other
runs wrote (the newest-200 window used to hide them); and every row names
its origin: ``agent``, ``loop``, ``extracted`` (entities and relations a
model extracted from a person's upload, K-1) or the recorded actor.
"""

from __future__ import annotations

from typing import Any

from .graph_registry import (
    KnowledgeGraphMeta,
    _iso,
    _SyncPoolStore,
    graph_id_is_default,
)

# Content keys naming the run that wrote a row, most specific first.
_RUN_KEYS = ("source_loop", "loop_id", "job_id", "artifact_id")
_LOOP_KEYS = ("source_loop", "loop_id")
_SUMMARY_MAX = 200

_GRAPH_SCOPE = "(graph_id = %s OR (%s AND graph_id = '' AND project_id = %s))"
# The run a row belongs to, exactly as :func:`row_run_id` reads it.
_RUN_EXPR = "COALESCE(" + ", ".join(f"NULLIF(content->>'{k}', '')" for k in _RUN_KEYS) + ")"
_RUN_FILTER = f"(%s::text IS NULL OR {_RUN_EXPR} = %s)"

_NODES_SQL = (
    "SELECT node_id, label, node_type, confidence, content, created_at, graph_id "
    "FROM knowledge_nodes WHERE status = 'proposed' AND " + _GRAPH_SCOPE + " "
    "AND " + _RUN_FILTER + " ORDER BY created_at DESC LIMIT %s"
)
_EDGES_SQL = (
    "SELECT edge_id, source_id, target_id, relation_type, confidence, content, "
    "created_at, graph_id FROM knowledge_edges "
    "WHERE status = 'proposed' AND " + _GRAPH_SCOPE + " "
    "AND " + _RUN_FILTER + " ORDER BY created_at DESC LIMIT %s"
)


def _content(row: Any) -> dict[str, Any]:
    value = row.get("content")
    return value if isinstance(value, dict) else {}


def row_run_id(content: dict[str, Any]) -> str:
    """The run that wrote the row, or '' when the row names none."""
    return next((str(content[k]) for k in _RUN_KEYS if content.get(k)), "")


def row_origin(content: dict[str, Any]) -> str:
    """Who wrote the row: the recorded ``origin_actor``, 'loop' for loop
    rows, 'extracted' for a model's extraction written before rows recorded
    their origin (semantic rows of an upload), else ''."""
    actor = str(content.get("origin_actor") or "").strip()
    if actor:
        return actor
    if any(content.get(k) for k in _LOOP_KEYS):
        return "loop"
    return "extracted" if content.get("semantic") or content.get("semantic_type") else ""


def _common(row: Any, content: dict[str, Any]) -> dict[str, Any]:
    confidence = row.get("confidence")
    return {
        "graph_id": str(row.get("graph_id") or ""),
        "confidence": float(confidence) if confidence is not None else 0.0,
        "created_at": _iso(row.get("created_at")),
        "origin": row_origin(content),
        "run_id": row_run_id(content),
    }


def _node_dict(row: Any) -> dict[str, Any]:
    content = _content(row)
    label = str(row.get("label") or row["node_id"])
    summary = str(content.get("statement") or label)[:_SUMMARY_MAX]
    return {
        "kind": "node",
        "id": str(row["node_id"]),
        "label": label,
        "node_type": str(row.get("node_type") or ""),
        "summary": summary,
        **_common(row, content),
    }


def _edge_dict(row: Any) -> dict[str, Any]:
    content = _content(row)
    relation = str(row.get("relation_type") or "related")
    return {
        "kind": "edge",
        "id": str(row["edge_id"]),
        "source_id": str(row["source_id"]),
        "target_id": str(row["target_id"]),
        "relation_type": relation,
        "summary": f"{row['source_id']} -{relation}-> {row['target_id']}"[
            :_SUMMARY_MAX
        ],
        **_common(row, content),
    }


class ReviewQueueStore(_SyncPoolStore):
    """Read-only: proposed nodes and edges of one registered graph."""

    def proposed_rows(
        self, graph: KnowledgeGraphMeta, *, limit: int = 200, run_id: str | None = None
    ) -> dict[str, Any]:
        project_id = graph.project_id or ""
        legacy_default = bool(
            project_id and graph_id_is_default(graph.graph_id, project_id)
        )
        run = run_id or None
        params = (graph.graph_id, legacy_default, project_id, run, run, limit)
        with self._cursor() as cur:
            cur.execute(_NODES_SQL, params)
            nodes = [_node_dict(row) for row in cur.fetchall()]
            cur.execute(_EDGES_SQL, params)
            edges = [_edge_dict(row) for row in cur.fetchall()]
        return {
            "nodes": nodes,
            "edges": edges,
            "truncated": limit in (len(nodes), len(edges)),
        }
