"""Repair knowledge nodes that hang on no artifact (ADR-093 D-11 / MD-17).

Usage::

    python -m umrs.knowledge.orphan_repair --project P [--graph G] [--apply]

A dry run is the default: it counts each class and prints one JSON object,
writing nothing. ``--apply`` deletes in ONE transaction (all classes or
none) and is re-runnable: a second run finds nothing.

Only nodes that carry their artifact in ``content.artifact_id`` are ever
considered. Chunks minted by rag.upload and the legacy canonizer key their
rows by ``document_id`` and are never touched; nor is a file node of those
writers (``content.origin`` set), whose chunks hang on it.

Classes:

- ``rowless_chunks``: chunk nodes whose artifact has no row in the node's
  project. Deleted; their part_of edges and vectors go with them
  (``ON DELETE CASCADE``). The 3 orphans S7's first live run found on the
  stand-alone stack (2026-10-11) are this class: an embed write that failed
  half-way, before the writer became one transaction.
- ``rowless_files``: file nodes written by understand (no ``origin``) whose
  artifact has no row in the node's project. Deleted, with their edges.
- ``edgeless_chunks``: chunk nodes the embed job wrote whose artifact row
  exists but which have no part_of edge. Deleted, and their artifacts are
  marked ``embedding_status = 'stale'`` so the next KNOWLEDGE_REINDEX
  rewrites them.

Run it on a site through Ansible (MD-17); agents never edit hosts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from umrs.knowledge.access_scope import EMBED_CHUNK_ORIGIN
from umrs.knowledge.graph_registry import default_graph_id

logger = logging.getLogger(__name__)

#: The artifact id a node carries; migration 0122 indexes
#: ``(project_id, (content->>'artifact_id'))`` on exactly this expression.
ARTIFACT_ID_EXPR = "content->>'artifact_id'"

_NO_ROW = (
    f"n.content ? 'artifact_id' AND NOT EXISTS (SELECT 1 FROM knowledge_artifacts a "
    f"WHERE a.artifact_id = n.{ARTIFACT_ID_EXPR} AND a.project_id = n.project_id)"
)
_HAS_ROW = (
    f"n.content ? 'artifact_id' AND EXISTS (SELECT 1 FROM knowledge_artifacts a "
    f"WHERE a.artifact_id = n.{ARTIFACT_ID_EXPR} AND a.project_id = n.project_id)"
)
_NO_PART_OF = (
    "NOT EXISTS (SELECT 1 FROM knowledge_edges e "
    "WHERE e.source_id = n.node_id AND e.relation_type = 'part_of')"
)

#: class -> the rows it holds (over alias ``n``, before the scope).
CLASSES: dict[str, str] = {
    "rowless_chunks": f"n.content->>'kind' = 'chunk' AND {_NO_ROW}",
    "rowless_files": f"n.node_type = 'file' AND (n.content->>'origin') IS NULL AND {_NO_ROW}",
    "edgeless_chunks": (
        f"n.content->>'kind' = 'chunk' AND n.content->>'origin' = '{EMBED_CHUNK_ORIGIN}' "
        f"AND {_HAS_ROW} AND {_NO_PART_OF}"
    ),
}

_GRAPH_SQL = (
    "SELECT graph_id FROM knowledge_graphs "
    "WHERE graph_id = %s OR (name = %s AND coalesce(project_id, '') IN (%s, '')) "
    "ORDER BY (graph_id = %s) DESC, graph_id LIMIT 1"
)
_STALE_SQL = (
    "UPDATE knowledge_artifacts SET embedding_status = 'stale', updated_at = %s "
    "WHERE project_id = %s AND artifact_id = ANY(%s)"
)
#: Node ids listed per class in the report (the counts are complete).
SAMPLE = 20


@dataclass
class RepairReport:
    project_id: str
    graph_id: str | None = None
    applied: bool = False
    found: dict[str, int] = field(default_factory=dict)
    deleted: dict[str, int] = field(default_factory=dict)
    samples: dict[str, list[str]] = field(default_factory=dict)
    artifacts_marked_stale: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "applied": self.applied,
            "found": dict(self.found),
            "deleted": dict(self.deleted),
            "samples": {k: list(v) for k, v in self.samples.items()},
            "artifacts_marked_stale": self.artifacts_marked_stale,
        }


def scope(project_id: str, graph_id: str | None) -> tuple[str, list[Any]]:
    """The project's nodes; with a graph, legacy ``''`` rows belong to the
    project's default graph (``KnowledgeStore.list_nodes``' rule)."""
    sql = "n.project_id = %s"
    params: list[Any] = [project_id]
    if graph_id is not None:
        sql += " AND (n.graph_id = %s OR (n.graph_id = '' AND %s))"
        params += [graph_id, graph_id == default_graph_id(project_id)]
    return sql, params


def _resolve_graph(cur: Any, project_id: str, graph: str | None) -> str | None:
    """A graph id from an id or a name in the project; the token when unknown."""
    if not graph:
        return None
    cur.execute(_GRAPH_SQL, (graph, graph, project_id, graph))
    row = cur.fetchone()
    return str(row[0]) if row else graph


def _select(
    cur: Any, name: str, where: str, params: list[Any]
) -> list[tuple[str, str]]:
    cur.execute(
        f"/* orphan_repair:{name} */ SELECT n.node_id, n.{ARTIFACT_ID_EXPR} "
        f"FROM knowledge_nodes n WHERE {CLASSES[name]} AND {where} ORDER BY n.node_id",
        params,
    )
    return [(str(row[0]), str(row[1] or "")) for row in cur.fetchall()]


def _delete(cur: Any, node_ids: list[str], project_id: str) -> int:
    if not node_ids:
        return 0
    cur.execute(
        "/* orphan_repair:delete */ DELETE FROM knowledge_nodes "
        "WHERE project_id = %s AND node_id = ANY(%s)",
        (project_id, node_ids),
    )
    return max(int(getattr(cur, "rowcount", 0) or 0), 0)


def _mark_stale(cur: Any, project_id: str, artifact_ids: set[str]) -> int:
    if not artifact_ids:
        return 0
    stamp = datetime.now(timezone.utc).isoformat()
    cur.execute(_STALE_SQL, (stamp, project_id, sorted(artifact_ids)))
    return max(int(getattr(cur, "rowcount", 0) or 0), 0)


def _repair(cur: Any, report: RepairReport, graph: str | None, apply: bool) -> None:
    report.graph_id = _resolve_graph(cur, report.project_id, graph)
    where, params = scope(report.project_id, report.graph_id)
    found = {name: _select(cur, name, where, params) for name in CLASSES}
    for name, rows in found.items():
        report.found[name] = len(rows)
        report.samples[name] = [node_id for node_id, _a in rows[:SAMPLE]]
    if not apply:
        return
    for name, rows in found.items():
        report.deleted[name] = _delete(
            cur, [nid for nid, _a in rows], report.project_id
        )
    stale = {aid for _nid, aid in found["edgeless_chunks"] if aid}
    report.artifacts_marked_stale = _mark_stale(cur, report.project_id, stale)
    report.applied = True


def run_repair(
    pool: Any, project_id: str, *, graph: str | None = None, apply: bool = False
) -> RepairReport:
    """Count (and with ``apply``, delete) the orphan classes in one
    transaction; a failure rolls every class back."""
    report = RepairReport(project_id=project_id)
    conn = pool.getconn()
    try:
        with conn:  # one transaction: commit on success, roll back on error
            with conn.cursor() as cur:
                _repair(cur, report, graph, apply)
    finally:
        pool.putconn(conn)
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────


def _build_pool() -> Any:
    from psycopg2.pool import ThreadedConnectionPool

    from umrs.jobs.knowledge_adapters import _resolve_knowledge_dsn
    from umrs.shared.pg_settings import pg_connect_kwargs

    dsn = _resolve_knowledge_dsn()
    if not dsn:
        raise SystemExit(
            "no knowledge database configured (DATABASE_URL or DB_* variables)"
        )
    return ThreadedConnectionPool(0, 1, dsn, **pg_connect_kwargs("orphan-repair"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m umrs.knowledge.orphan_repair",
        description=(
            "Count (default) or delete (--apply) knowledge nodes that hang on no "
            "artifact. Only nodes carrying content.artifact_id are considered."
        ),
    )
    parser.add_argument("--project", required=True, help="project id")
    parser.add_argument(
        "--graph", default=None, help="graph id or name (default: whole project)"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete in one transaction (default: dry run)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    pool = _build_pool()
    try:
        report = run_repair(pool, args.project, graph=args.graph, apply=args.apply)
    finally:
        pool.closeall()
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover — CLI path
    sys.exit(main())


__all__ = ["ARTIFACT_ID_EXPR", "CLASSES", "RepairReport", "main", "run_repair", "scope"]
