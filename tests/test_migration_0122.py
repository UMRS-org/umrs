"""ADR-093 D-11 (wave F): migration 0122, honest knowledge writes.

Pins the three statements to the code that relies on them: the hashless
predicate of the per-project dedup index matches ``insert_artifact`` (which
never matches ``''``), ``knowledge_events.artifact_id`` becomes nullable
(its FK is ON DELETE SET NULL), and the node index carries the expression
the orphan repair and K3's live-artifact check read. The integration test
(``--db-url`` / ``UMRS_TEST_DB_URL``) applies the file in a rolled-back
transaction and shows two hashless active rows now coexist.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pytest

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0122_knowledge_honest_writes.sql"


def _code() -> str:
    lines = MIGRATION.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("--"))


def test_0122_is_the_only_migration_with_its_number_and_follows_0121() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0122") == 1
    assert dict(migrations.get_migration_files())["0122"] == MIGRATION
    assert versions.index("0122") == versions.index("0121") + 1


def test_0122_rebuilds_the_dedup_index_for_hashed_rows_only() -> None:
    code = _code()
    assert "DROP INDEX IF EXISTS idx_artifacts_project_hash_active;" in code
    statement = re.search(
        r"CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_project_hash_active\s+"
        r"ON knowledge_artifacts \(project_id, content_hash\)\s+WHERE (.+);",
        code,
    )
    assert statement, "0122 must recreate the per-project dedup index"
    assert statement.group(1) == "status = 'active' AND content_hash <> ''"


def test_0122_keeps_an_artifacts_event_trail() -> None:
    code = _code()
    assert "ALTER TABLE knowledge_events ALTER COLUMN artifact_id DROP NOT NULL;" in code
    assert "CASCADE" not in code.upper()


#: The expression the orphan repair and K3 read; tests/test_orphan_repair.py
#: pins the repair's own SQL to it.
ARTIFACT_ID_EXPR = "content->>'artifact_id'"


def test_0122_indexes_the_artifact_id_nodes_carry() -> None:
    statement = re.search(
        r"CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_content_artifact\s+"
        r"ON knowledge_nodes \(project_id, \((.+)\)\);",
        _code(),
    )
    assert statement and statement.group(1) == ARTIFACT_ID_EXPR


def test_0122_runs_inside_the_runners_transaction() -> None:
    assert "CONCURRENTLY" not in _code().upper()


# ── integration: a real Postgres (skipped without a database URL) ────────────


@pytest.fixture()
def pg_conn(request: pytest.FixtureRequest) -> Any:
    psycopg2 = pytest.importorskip("psycopg2")
    url = request.config.getoption("--db-url") or os.environ.get("UMRS_TEST_DB_URL")
    if not url:
        pytest.skip("no database URL (--db-url / UMRS_TEST_DB_URL)")
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('knowledge_artifacts') IS NOT NULL")
            if not cur.fetchone()[0]:
                pytest.skip("knowledge_artifacts is absent: migrate the test database first")
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.integration
def test_0122_lets_two_hashless_active_artifacts_coexist(pg_conn: Any) -> None:
    with pg_conn.cursor() as cur:
        cur.execute(MIGRATION.read_text())
        for artifact_id in ("art-122-a", "art-122-b"):
            cur.execute(
                "INSERT INTO knowledge_artifacts (artifact_id, content_hash, kind, project_id, "
                "status) VALUES (%s, '', 'document', 'p-122', 'active')",
                (artifact_id,),
            )
        cur.execute(
            "SELECT count(*) FROM knowledge_artifacts WHERE project_id = 'p-122' "
            "AND content_hash = ''"
        )
        assert cur.fetchone()[0] == 2
