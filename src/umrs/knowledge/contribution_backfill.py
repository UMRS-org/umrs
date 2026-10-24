"""Backfill ``knowledge_artifact_contributions`` (migration 0129, K1.2).

Usage (dry run by default; ``--apply`` writes; safe to repeat)::

    python -m umrs.knowledge.contribution_backfill --db-url postgresql://...
    python -m umrs.knowledge.contribution_backfill --db-url ... --apply
    python -m umrs.knowledge.contribution_backfill --db-url ... --project opossum

Each artifact gets the contributions its history names:

* **ingest jobs** — every ``knowledge_ingest`` attempt's ``metrics.artifacts``,
  joined to the job's owner (``jobs.user_id``), its target graph
  (``spec.graph_id``, else the project default graph) and its actor
  (``spec.origin_actor``). A personal-graph upload is recorded in that
  personal graph, so it stays hidden from other members;
* **artifact metadata** — a rag upload's ``owner_id``, a workflow output's
  producing job's owner, a loop finding (origin ``loop``);
* **untraced** — an artifact none of the above names gets one placeholder in
  its project's default graph (``user_id ''``, origin ``unknown``), and the
  report counts them. Without a row an artifact would be listed nowhere.

Rows already present are never rewritten (``ON CONFLICT DO NOTHING``); a
second run plans nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from .contributions import (
    PLACEHOLDER_ORIGIN,
    Contribution,
    origin_for,
    record_contribution_sync,
    title_for,
)
from .graph_registry import default_graph_id

_ARTIFACTS_SQL = """SELECT artifact_id, project_id, metadata, source_ref
  FROM knowledge_artifacts
 WHERE %s::text[] IS NULL OR project_id = ANY(%s::text[])"""
_EXISTING_SQL = (
    "SELECT artifact_id, graph_id, user_id FROM knowledge_artifact_contributions"
)
_INGEST_ATTEMPTS_SQL = """SELECT j.project_id, j.user_id, j.spec, t.metrics
  FROM jobs j JOIN job_attempts t ON t.job_id = j.job_id
 WHERE j.job_type = 'knowledge_ingest'"""
_JOB_OWNERS_SQL = "SELECT job_id, user_id FROM jobs WHERE job_id = ANY(%s)"


@dataclass
class _Artifact:
    artifact_id: str
    project_id: str
    metadata: dict[str, Any]
    source_ref: str

    @property
    def title(self) -> str:
        return title_for(self.metadata, self.source_ref)


@dataclass
class BackfillPlan:
    """What a run would write; ``apply_backfill`` writes exactly this."""

    traced: list[Contribution] = field(default_factory=list)
    untraced: list[Contribution] = field(default_factory=list)
    already: int = 0

    def summary(self) -> dict[str, Any]:
        sources: dict[str, int] = {}
        for row in self.traced:
            sources[row.origin_actor] = sources.get(row.origin_actor, 0) + 1
        return {
            "traced": len(self.traced),
            "traced_by_origin": dict(sorted(sources.items())),
            "untraced_placeholders": len(self.untraced),
            "artifacts_already_attributed": self.already,
        }


def _json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    return value if value is not None else {}


def _fetch(conn: Any, sql: str, params: tuple[Any, ...] = ()) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def _artifacts(conn: Any, projects: list[str] | None) -> dict[str, _Artifact]:
    """Every artifact (of *projects*, when given)."""
    rows = _fetch(conn, _ARTIFACTS_SQL, (projects, projects))
    return {
        str(r[0]): _Artifact(str(r[0]), str(r[1] or ""), dict(_json(r[2]) or {}), str(r[3] or ""))
        for r in rows
    }


def _spec_target(project_id: str, user_id: Any, spec_raw: Any) -> tuple[str, str, str]:
    """``(graph_id, user_id, origin)`` an ingest job wrote for."""
    spec = dict(_json(spec_raw) or {})
    user = str(user_id or "")
    graph = str(spec.get("graph_id") or "") or default_graph_id(project_id)
    return graph, user, origin_for(spec.get("origin_actor"), user)


def _recorded_ids(metrics: Any) -> list[str]:
    """The artifact ids an attempt's ``metrics.artifacts`` lists."""
    items = _json(metrics).get("artifacts") or []
    return [str((item or {}).get("artifact_id") or "") for item in items]


def _attempt_rows(row: tuple, artifacts: dict[str, _Artifact]) -> list[Contribution]:
    """The contributions one ingest attempt names (artifacts of its project)."""
    project_id, user_id, spec_raw, metrics = row
    graph, user, origin = _spec_target(str(project_id), user_id, spec_raw)
    held = [artifacts.get(i) for i in _recorded_ids(metrics)]
    return [
        Contribution(a.artifact_id, graph, a.project_id, user, origin, a.title)
        for a in held
        if a is not None and a.project_id == project_id
    ]


def _ingest_rows(conn: Any, artifacts: dict[str, _Artifact]) -> list[Contribution]:
    """One contribution per (artifact, job) an ingest attempt recorded."""
    rows: list[Contribution] = []
    for row in _fetch(conn, _INGEST_ATTEMPTS_SQL):
        rows.extend(_attempt_rows(row, artifacts))
    return rows


def _metadata_origin(art: _Artifact, owners: dict[str, str]) -> tuple[str, str] | None:
    """``(user_id, origin)`` the artifact's own metadata names, or None."""
    meta = art.metadata
    if meta.get("asset_kind") == "finding":
        return "", "loop"
    producer = str(meta.get("produced_by_job_id") or "")
    if producer:
        return owners.get(producer, ""), "workflow"
    owner = str(meta.get("owner_id") or "")
    if owner:
        return owner, "user"
    return None


def _metadata_rows(conn: Any, artifacts: dict[str, _Artifact]) -> list[Contribution]:
    producers = sorted(
        {str(a.metadata.get("produced_by_job_id")) for a in artifacts.values()}
        - {"None", ""}
    )
    owners = {str(r[0]): str(r[1] or "") for r in _fetch(conn, _JOB_OWNERS_SQL, (producers,))}
    rows: list[Contribution] = []
    for art in artifacts.values():
        named = _metadata_origin(art, owners)
        if named is not None:
            rows.append(
                Contribution(
                    art.artifact_id,
                    default_graph_id(art.project_id),
                    art.project_id,
                    named[0],
                    named[1],
                    art.title,
                )
            )
    return rows


def plan_backfill(conn: Any, projects: list[str] | None = None) -> BackfillPlan:
    """Read everything (or *projects* only); decide what to write (no writes)."""
    artifacts = _artifacts(conn, projects)
    existing = {
        (str(a), str(g), str(u))
        for a, g, u in _fetch(conn, _EXISTING_SQL)
        if str(a) in artifacts
    }
    attributed = {key[0] for key in existing}
    plan = BackfillPlan(already=len(attributed & set(artifacts)))
    seen = set(existing)
    for row in [*_ingest_rows(conn, artifacts), *_metadata_rows(conn, artifacts)]:
        key = (row.artifact_id, row.graph_id, row.user_id)
        if key not in seen:
            seen.add(key)
            plan.traced.append(row)
    named = attributed | {row.artifact_id for row in plan.traced}
    plan.untraced = [
        Contribution(a.artifact_id, default_graph_id(a.project_id), a.project_id, "", PLACEHOLDER_ORIGIN, a.title)
        for a in artifacts.values()
        if a.artifact_id not in named
    ]
    return plan


def apply_backfill(conn: Any, plan: BackfillPlan) -> int:
    """Write the plan in one transaction; returns the rows attempted."""
    rows = [*plan.traced, *plan.untraced]
    with conn.cursor() as cur:
        for row in rows:
            record_contribution_sync(cur, row)
    conn.commit()
    return len(rows)


def _db_url(arg: str | None) -> str:
    return arg or os.environ.get("UMRS_DATA_CATALOG_DB") or os.environ.get("DATABASE_URL", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--db-url", default=None, help="Postgres URL (else UMRS_DATA_CATALOG_DB / DATABASE_URL)")
    parser.add_argument("--apply", action="store_true", help="write the rows (default: dry run)")
    parser.add_argument(
        "--project", action="append", default=None, help="limit to this project (repeatable)"
    )
    args = parser.parse_args(argv)
    url = _db_url(args.db_url)
    if not url:
        print("no database URL: pass --db-url", file=sys.stderr)
        return 2
    import psycopg2

    from umrs.shared.pg_settings import pg_connect_kwargs

    conn = psycopg2.connect(url, **pg_connect_kwargs("contribution-backfill"))
    try:
        plan = plan_backfill(conn, args.project)
        report = {"mode": "apply" if args.apply else "dry-run", **plan.summary()}
        if args.apply:
            report["written"] = apply_backfill(conn, plan)
    finally:
        conn.close()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
