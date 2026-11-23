"""Hunt #26: migration 0140 makes MCP server names unique per project."""

from __future__ import annotations

import re

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0140_mcp_servers_project_name.sql"


def test_0140_follows_0139() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0140") == 1
    assert versions.index("0140") == versions.index("0139") + 1


def test_0140_keys_names_by_project_before_dropping_the_global_key() -> None:
    sql = " ".join(MIGRATION.read_text().split())
    create = (
        "CREATE UNIQUE INDEX IF NOT EXISTS mcp_servers_project_name_key "
        "ON mcp_servers ((COALESCE(TRIM(project_id), '')), name)"
    )
    drop = "ALTER TABLE mcp_servers DROP CONSTRAINT IF EXISTS mcp_servers_name_key"
    assert create in sql
    assert sql.index(create) < sql.index(drop)  # never a window with no key
    assert not re.search(r"\bDROP\s+(COLUMN|TABLE)", sql, re.I)
