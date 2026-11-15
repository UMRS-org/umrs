"""ADR-116 D-4: migration 0139 adds the ontology spec and source columns."""

from __future__ import annotations

import re

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0139_knowledge_ontology_spec.sql"


def test_0139_follows_0138() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0139") == 1
    assert versions.index("0139") == versions.index("0138") + 1


def test_0139_adds_each_column_idempotently() -> None:
    sql = MIGRATION.read_text()
    added = re.findall(r"ADD COLUMN IF NOT EXISTS (\w+)", sql)
    assert added == ["spec", "source_format", "source_text", "source_sha256"]
    assert "spec JSONB NOT NULL DEFAULT '{}'::jsonb" in sql
    assert not re.search(r"\bDROP\s+(COLUMN|TABLE)", sql, re.I)


def test_0139_scopes_the_live_coordinate_per_project() -> None:
    """The 0091 index made id@version one namespace across projects: project
    B got an unexplained 409 for a coordinate only project A could see."""
    sql = " ".join(MIGRATION.read_text().split())
    create = (
        "CREATE UNIQUE INDEX IF NOT EXISTS knowledge_ontologies_live_scoped_coordinate "
        "ON knowledge_ontologies (ontology_id, version, coalesce(project_id, '')) "
        "WHERE retired_at IS NULL"
    )
    drop = "DROP INDEX IF EXISTS knowledge_ontologies_live_coordinate"
    assert create in sql
    assert sql.index(create) < sql.index(drop)  # never a window with no index
    assert re.findall(r"\bDROP\s+INDEX\s+(?:IF\s+EXISTS\s+)?(\w+)", sql, re.I) == [
        "knowledge_ontologies_live_coordinate"
    ]
