#!/usr/bin/env python3
"""Upgrade check from the pre-merge ``main`` schema (U4, plan §7.2 and §4).

Databases created on UMRS ``main`` before the one-branch merge (tag
``archive/main-pre-merge-2026-11``) recorded 0001…0054 without ``0016``,
``0018`` and ``0019``, which the merged line adds *below* their last
migration. This check builds that database and upgrades it:

1. apply the archived ``.sql`` files (``OLD_DIR``) with the current runner,
   so ``schema_migrations`` holds exactly the old versions;
2. apply the current migrations: the gap files run out of order;
3. a second run applies nothing, and every current file is recorded once.

The runner's boot-time jobs schema (``umrs.jobs.schema``) is the current
one in step 1 as well: the servers create it at boot on either line.

Usage: python scripts/migration_upgrade_check.py DB_URL OLD_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import umrs.migrations as runner


def _apply(db_url: str, migrations_dir: Path) -> int:
    saved = runner.MIGRATIONS_DIR
    runner.MIGRATIONS_DIR = migrations_dir
    conn = psycopg2.connect(db_url)
    try:
        return runner.apply_migrations(conn)  # takes a pool or a connection
    finally:
        conn.close()
        runner.MIGRATIONS_DIR = saved


def _recorded(db_url: str) -> list[str]:
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT version FROM schema_migrations ORDER BY version")
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def _upgrade_problems(db_url: str, old: list[str]) -> list[str]:
    current = [v for v, _ in runner.get_migration_files()]
    gaps = [v for v in current if v not in old and v < old[-1]]
    upgraded = _apply(db_url, runner.MIGRATIONS_DIR)
    print(f"pre-merge {len(old)}, upgrade applied {upgraded}, out-of-order gaps {gaps}")
    expected = len(current) - len(old)
    problems = [] if upgraded == expected else [f"upgrade applied {upgraded}, expected {expected}"]
    if _apply(db_url, runner.MIGRATIONS_DIR) != 0:
        problems.append("a second upgrade run was not a no-op")
    if _recorded(db_url) != sorted(current):
        problems.append("schema_migrations does not match the current files")
    return problems


def check(db_url: str, old_dir: Path) -> list[str]:
    """Problems found; empty when the upgrade is complete and idempotent."""
    old = sorted(p.stem.split("_")[0] for p in old_dir.glob("*.sql"))
    first = _apply(db_url, old_dir)
    if first != len(old) or _recorded(db_url) != old:
        return [f"pre-merge schema not built: applied {first} of {len(old)}"]
    return _upgrade_problems(db_url, old)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    problems = check(argv[1], Path(argv[2]))
    for problem in problems:
        print(f"::error::migration upgrade: {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
