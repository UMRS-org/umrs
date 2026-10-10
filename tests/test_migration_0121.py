"""ADR-093 D-14: migration 0121, the graph bridge lane's label index.

The lane (``graph_lane``) is index-ready, and above
``LABEL_SCAN_MAX_ROWS`` semantic nodes it reports ``unavailable`` unless
``LABEL_INDEX`` exists (review profiles-graph-4, 2026-10-10). These tests pin
the index to the lane's own expression and predicate, so the planner can use
it. The integration test (``--db-url`` / ``UMRS_TEST_DB_URL``) applies the file
inside a rolled-back transaction and checks that the link statement reaches
the index.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pytest
from umrs.knowledge.retrieval import graph_lane
from umrs.knowledge.retrieval.types import RetrieveSpec

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0121_knowledge_nodes_semantic_label_fts.sql"


def _code() -> str:
    """The file without ``--`` comment lines."""
    lines = MIGRATION.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("--"))


def test_0121_is_the_only_migration_with_its_number_and_follows_0120() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0121") == 1
    assert dict(migrations.get_migration_files())["0121"] == MIGRATION
    assert versions.index("0121") == versions.index("0120") + 1


def test_0121_builds_the_index_the_lane_names() -> None:
    statement = re.search(
        r"CREATE INDEX IF NOT EXISTS (\w+)\s+ON knowledge_nodes USING gin \((.+)\)\s+"
        r"WHERE node_id LIKE '([^']+)';",
        _code(),
    )
    assert statement, "0121 must build one partial GIN index on knowledge_nodes"
    name, expression, prefix = statement.groups()
    assert name == graph_lane.LABEL_INDEX
    # The lane matches labels on this expression; the planner needs the same.
    lane_expression = graph_lane._LABEL_TSV.replace("n.label", "label")
    assert expression.replace("'english'::regconfig", "'english'") == lane_expression
    # The lane's scope binds the same prefix the partial predicate holds.
    _where, params = graph_lane._semantic_where(RetrieveSpec(queries=("q",), project_id="p"))
    assert prefix == f"{graph_lane.SEMANTIC_PREFIX}%" and prefix in params


def test_0121_runs_inside_the_runners_transaction() -> None:
    # The runner switches a file mentioning CONCURRENTLY to autocommit.
    assert "CONCURRENTLY" not in _code().upper()
    assert [s.split()[0].upper() for s in migrations._split_statements(_code()) if s.strip()] == [
        "CREATE"
    ]


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
            cur.execute("SELECT to_regclass('knowledge_nodes') IS NOT NULL")
            if not cur.fetchone()[0]:
                pytest.skip("knowledge_nodes is absent: migrate the test database first")
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.integration
def test_0121_applies_and_the_link_statement_reaches_it(pg_conn: Any) -> None:
    spec = RetrieveSpec(queries=("q",), project_id="g121-it")
    where, params = graph_lane._semantic_where(spec)
    with pg_conn.cursor() as cur:
        cur.execute(MIGRATION.read_text())
        cur.execute("SELECT to_regclass(%s) IS NOT NULL", (graph_lane.LABEL_INDEX,))
        assert cur.fetchone()[0]
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute("SET LOCAL enable_indexscan = off")
        sql = graph_lane._LINK_SQL.format(where=where)
        cur.execute("EXPLAIN " + sql, ["zeolite adsorption", *params, *params, 8])
        plan = "\n".join(row[0] for row in cur.fetchall())
    assert f"Bitmap Index Scan on {graph_lane.LABEL_INDEX}" in plan, plan
