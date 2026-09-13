#!/usr/bin/env python3
"""Ambient-corpus marker (ADR-050 operator tool).

Audits knowledge/corpus rows by project and marks the chosen ones
`visibility = 'ambient'` — read-gated everywhere, never deleted
(user decision). Spec is explicit: pass project ids; nothing is marked
without that list, and `--system-projects` marks rows belonging to the
named projects as 'system' instead.

Default behavior is audit-only:

    python tools/mark_ambient_corpus.py --dsn postgresql://... --audit

Apply (marks the listed projects' rows ambient):

    python tools/mark_ambient_corpus.py --dsn postgresql://... \
        --mark-ambient my-old-demo,seed-corpus

System-wide marking (visible to every caller):

    python tools/mark_ambient_corpus.py --dsn postgresql://... \
        --mark-system platform-handbook
"""

from __future__ import annotations

import argparse
import sys


def _connect(dsn: str):
    import psycopg2

    return psycopg2.connect(dsn)


def audit(conn) -> list[tuple[str, str, int]]:
    """Return [(plane, project_id, row_count)] counts, ordered by plane/project."""
    out: list[tuple[str, str, int]] = []
    with conn.cursor() as cur:
        for plane, table in (
            ("knowledge_nodes", "knowledge_nodes"),
            ("catalog_meta", "sat_knowledge_objects"),
        ):
            try:
                cur.execute(
                    f"SELECT project_id, visibility, COUNT(*) FROM {table} "
                    "GROUP BY project_id, visibility ORDER BY project_id, visibility"
                )
            except Exception as exc:  # noqa: BLE001 — audit must tolerate missing tier col
                if "visibility" in str(exc):
                    print(
                        f"WARN: {table}.visibility absent — apply migration 0092 first",
                        file=sys.stderr,
                    )
                    raise SystemExit(2)
                raise
            for project, visibility, count in cur.fetchall():
                out.append((plane, project, visibility, count))
    return out


def _mark(conn, table: str, projects: list[str], tier: str) -> int:
    if not projects:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET visibility = %s, updated_at = now() "
            "WHERE project_id = ANY(%s) RETURNING 1",
            (tier, projects),
        )
        count = len(cur.fetchall())
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True, help="postgresql:// DSN")
    parser.add_argument("--audit", action="store_true", help="audit only")
    parser.add_argument("--mark-ambient", default="", help="comma-separated project ids")
    parser.add_argument("--mark-system", default="", help="comma-separated project ids")
    args = parser.parse_args()

    ambients = [t.strip() for t in args.mark_ambient.split(",") if t.strip()]
    systems = [t.strip() for t in args.mark_system.split(",") if t.strip()]
    if not args.audit and not ambients and not systems:
        print("ERROR: no action requested (use --audit or --mark-*)", file=sys.stderr)
        raise SystemExit(2)

    conn = _connect(args.dsn)
    try:
        rows = audit(conn)
        if rows:
            print(f"{'plane':<18} {'project':<28} {'visibility':<10} rows")
            for plane, project, visibility, count in rows:
                print(f"{plane:<18} {project:<28} {visibility:<10} {count}")
        changed = 0
        if ambients:
            changed += _mark(conn, "knowledge_nodes", ambients, "ambient")
            changed += _mark(conn, "sat_knowledge_objects", ambients, "ambient")
        if systems:
            changed += _mark(conn, "knowledge_nodes", systems, "system")
            changed += _mark(conn, "sat_knowledge_objects", systems, "system")
        if changed:
            conn.commit()
            print(f"\nmarked: {changed} row(s)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
