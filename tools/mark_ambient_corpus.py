#!/usr/bin/env python3
"""Visibility-tier marker for the knowledge planes (ADR-050 operator tool).

Audits rows per plane / project / tier and marks the chosen projects'
rows into one tier. Ambient rows are read-gated everywhere and never
deleted (user decision). Nothing is ever marked without an explicit
project list — the default is audit-only.

Planes (2026-09-13, R-02 / ADR-050 wave 2 — catalog plane added):

    knowledge_nodes        canonical graph nodes           (0092)
    sat_knowledge_objects  legacy sat catalog metadata     (0092)
    catalog_objects        data-catalog metadata           (0093)

Audit only:

    python tools/mark_ambient_corpus.py --dsn postgresql://... --audit

Mark the listed projects' rows ambient (hidden from every read lane):

    python tools/mark_ambient_corpus.py --dsn postgresql://... \\
        --mark-ambient my-old-demo,seed-corpus

Mark system-wide (visible to every caller):

    python tools/mark_ambient_corpus.py --dsn postgresql://... \\
        --mark-system platform-handbook

Restore / unmark (back to the owner-only default tier):

    python tools/mark_ambient_corpus.py --dsn postgresql://... \\
        --mark-project my-old-demo
"""

from __future__ import annotations

import argparse
import sys

# (plane label, table) — every table carries the 0092/0093 visibility column.
PLANES: tuple[tuple[str, str], ...] = (
    ("knowledge_nodes", "knowledge_nodes"),
    ("sat_catalog", "sat_knowledge_objects"),
    ("catalog_objects", "catalog_objects"),
)

TIERS: tuple[str, ...] = ("project", "system", "ambient")


def _connect(dsn: str):
    import psycopg2
    from umrs.shared.pg_settings import pg_connect_kwargs

    return psycopg2.connect(dsn, **pg_connect_kwargs("mark-ambient-corpus"))


def _audit_plane(cur, plane: str, table: str) -> list[tuple[str, str, str, int]]:
    try:
        cur.execute(
            f"SELECT project_id, visibility, COUNT(*) FROM {table} "
            "GROUP BY project_id, visibility ORDER BY project_id, visibility"
        )
    except Exception as exc:  # noqa: BLE001 — audit must tolerate missing tier col
        if "visibility" in str(exc):
            print(
                f"WARN: {table}.visibility absent — apply migrations 0092/0093 first",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc
        raise
    return [(plane, project, visibility, count) for project, visibility, count in cur.fetchall()]


def audit(conn) -> list[tuple[str, str, str, int]]:
    """Return [(plane, project_id, visibility, row_count)] ordered by plane/project."""
    out: list[tuple[str, str, str, int]] = []
    with conn.cursor() as cur:
        for plane, table in PLANES:
            out.extend(_audit_plane(cur, plane, table))
    return out


def plane_counts(rows: list[tuple[str, str, str, int]]) -> dict[str, dict[str, int]]:
    """Per-plane {tier: rows} totals (every tier present, zero when absent)."""
    counts: dict[str, dict[str, int]] = {plane: dict.fromkeys(TIERS, 0) for plane, _ in PLANES}
    for plane, _project, visibility, count in rows:
        counts.setdefault(plane, dict.fromkeys(TIERS, 0))
        counts[plane][visibility] = counts[plane].get(visibility, 0) + count
    return counts


def _mark(conn, table: str, projects: list[str], tier: str) -> int:
    if not projects:
        return 0
    if tier not in TIERS:
        raise ValueError(f"unknown visibility tier {tier!r}")
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET visibility = %s, updated_at = now() "
            "WHERE project_id = ANY(%s) AND visibility <> %s RETURNING 1",
            (tier, projects, tier),
        )
        count = len(cur.fetchall())
    return count


def mark_projects(conn, projects: list[str], tier: str) -> dict[str, int]:
    """Mark the projects' rows ``tier`` on every plane; returns {plane: changed}."""
    return {plane: _mark(conn, table, projects, tier) for plane, table in PLANES}


def _csv(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def _print_audit(rows: list[tuple[str, str, str, int]]) -> None:
    if rows:
        print(f"{'plane':<18} {'project':<28} {'visibility':<10} rows")
        for plane, project, visibility, count in rows:
            print(f"{plane:<18} {project:<28} {visibility:<10} {count}")
    print("\nper-plane totals:")
    for plane, tiers in plane_counts(rows).items():
        summary = "  ".join(f"{tier}={tiers.get(tier, 0)}" for tier in TIERS)
        print(f"  {plane:<18} {summary}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dsn", required=True, help="postgresql:// DSN")
    parser.add_argument("--audit", action="store_true", help="audit only")
    parser.add_argument("--mark-ambient", default="", help="comma-separated project ids")
    parser.add_argument("--mark-system", default="", help="comma-separated project ids")
    parser.add_argument(
        "--mark-project",
        default="",
        help="comma-separated project ids to restore to the owner-only 'project' tier",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    actions = [
        ("ambient", _csv(args.mark_ambient)),
        ("system", _csv(args.mark_system)),
        ("project", _csv(args.mark_project)),
    ]
    if not args.audit and not any(projects for _tier, projects in actions):
        print("ERROR: no action requested (use --audit or --mark-*)", file=sys.stderr)
        return 2

    conn = _connect(args.dsn)
    try:
        _print_audit(audit(conn))
        changed: dict[str, int] = dict.fromkeys(dict(PLANES), 0)
        for tier, projects in actions:
            for plane, count in mark_projects(conn, projects, tier).items():
                changed[plane] += count
        total = sum(changed.values())
        if total:
            conn.commit()
            print("\nmarked: " + ", ".join(f"{plane}={n}" for plane, n in changed.items()))
            print(f"marked total: {total} row(s)")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
