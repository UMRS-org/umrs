"""Fresh-DB migration chain invariants (WP: clean bootstrap green belt).

Live found 2026-09-10: a fresh ``python -m umrs.migrations`` bootstrap failed
twice on invariants the migrated-in-place deployed DBs never exercised:

1. 0018/0019 reference ``knowledge_artifacts(id)`` before any migration
   created the table (0055 is artifact_id-PK and later; no `id` column at all).
2. 0051's ``CREATE INDEX CONCURRENTLY`` cannot run inside the runner's
   per-migration transaction.

These tests pin the static form of both: reference targets must be created
by an *earlier* migration, and CONCURRENTLY files must be runnable by the
 runner's autocommit branch.
"""

from __future__ import annotations

import re
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"

CREATE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+([A-Za-z_][A-Za-z0-9_]*)", re.IGNORECASE)
REF_RE = re.compile(r"REFERENCES\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE)

# Tables referenced by migrations but (legitimately) created outside the
# migration chain — e.g. boot-time DDL in app code. MUST NOT grow silently:
# adding an entry here is a code-level decision, reviewed.
EXTERNAL_TABLES: set[str] = {
    # Durable-jobs schema is boot-time DDL: src/umrs/jobs/schema.py
    # (ensure_schema, SCHEMA_VERSION migrations). Referenced by
    # 0040_execution_traces.sql and 0071_companion_turns.sql.
    "jobs",
}


def _migration_files() -> list[Path]:
    return sorted(MIGRATIONS_DIR.glob("[0-9]*_*.sql"))


def _created_before(idx: int, files: list[Path]) -> set[str]:
    created: set[str] = set()
    for path in files[:idx]:
        created.update(CREATE_RE.findall(path.read_text()))
    return created


def test_referenced_tables_are_created_earlier():
    files = _migration_files()
    violations: list[str] = []
    for idx, path in enumerate(files):
        text = path.read_text()
        targets = {t.lower() for t in REF_RE.findall(text)}
        created = {t.lower() for t in _created_before(idx, files)} | {
            t.lower() for t in EXTERNAL_TABLES
        }
        # Self-referencing FK inside the same file is legal.
        created.update(t.lower() for t in CREATE_RE.findall(text))
        missing = sorted(targets - created)
        for table in missing:
            violations.append(f"{path.name}: REFERENCES {table} before create")
    assert not violations, "fresh-DB migration order drifted:\n" + "\n".join(violations)


def test_knowledge_artifacts_precedes_a2l_bridges():
    files = [p.name for p in _migration_files()]
    base = next(n for n in files if n.startswith("0016_knowledge_artifacts_base"))
    assert files.index(base) < files.index("0018_a2l_artifact_refs_bridge.sql")
    assert files.index(base) < files.index("0019_knowledge_a2l_links.sql")


def test_concurrently_files_are_runner_supported():
    files = _migration_files()
    concurrent_files = [p.name for p in files if "CONCURRENTLY" in p.read_text().upper()]
    if not concurrent_files:
        return
    runner = (MIGRATIONS_DIR / "__init__.py").read_text()
    assert "conn.autocommit = True" in runner, (
        f"runner must handle CONCURRENTLY files out-of-transaction; flagged: {concurrent_files}"
    )
    assert "autocommit = False" in runner, "autocommit must be restored"
