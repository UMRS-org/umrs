"""ADR-093 C6: migration 0120, ``knowledge_nodes.search_text`` and its indexes.

The static tests pin the shape the design asks for (CONSTITUTION §5.2 review
input): a nullable column with no default (a catalog-only change, no table
rewrite), a partial expression GIN index whose expression is the one the lex
lane queries, and an optional ``pg_trgm`` whose absence is a NOTICE, never a
failed migration. The integration test (``--db-url`` / ``UMRS_TEST_DB_URL``)
applies the file inside a rolled-back transaction, checks that the planner
reaches the expression index, and runs both lanes against real rows.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pytest
from umrs.knowledge.retrieval import lexical
from umrs.knowledge.retrieval.types import RetrieveSpec

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0120_knowledge_chunk_search_text.sql"


def _sql() -> str:
    return MIGRATION.read_text()


def _code(sql: str) -> str:
    """The file without ``--`` comment lines."""
    return "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))


def test_0120_is_the_only_migration_with_its_number_and_sorts_last_so_far() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0120") == 1
    assert dict(migrations.get_migration_files())["0120"] == MIGRATION
    assert versions.index("0120") == versions.index("0119") + 1


def test_0120_adds_a_nullable_column_without_default() -> None:
    code = _code(_sql())
    match = re.search(r"ADD COLUMN IF NOT EXISTS search_text TEXT\s*;", code)
    assert match, "search_text must be a plain nullable TEXT column"
    statement = code[code.rfind("ALTER TABLE", 0, match.end()) : match.end()]
    assert "DEFAULT" not in statement.upper() and "NOT NULL" not in statement.upper()


def test_0120_fts_index_is_partial_on_the_lex_lanes_expression() -> None:
    code = _code(_sql())
    fts = re.search(
        rf"CREATE INDEX IF NOT EXISTS {lexical.FTS_INDEX}\s+ON knowledge_nodes USING gin "
        r"\(to_tsvector\('english'::regconfig, search_text\)\)\s+WHERE search_text IS NOT NULL;",
        code,
    )
    assert fts, "the FTS index must be a partial GIN on to_tsvector('english', search_text)"
    # The lane queries the same expression, so the planner can use the index.
    lane_sql = lexical._search_sql("TRUE") + lexical._MATCH_ONE
    assert lane_sql.count("to_tsvector('english', n.search_text) @@") == 2
    assert "n.search_text IS NOT NULL" in lexical._search_sql("TRUE")


def test_0120_pg_trgm_is_guarded_and_its_index_conditional() -> None:
    code = _code(_sql())
    guarded = re.search(
        r"DO \$\$\s*BEGIN\s*CREATE EXTENSION IF NOT EXISTS pg_trgm;\s*"
        r"EXCEPTION WHEN OTHERS THEN\s*RAISE NOTICE",
        code,
    )
    assert guarded, "pg_trgm must be created in a DO block that turns failure into a NOTICE"
    assert f"CREATE INDEX IF NOT EXISTS {lexical.TRGM_INDEX} " in code
    assert "%I.gin_trgm_ops" in code  # opclass qualified by the extension's schema
    assert "WHERE e.extname = 'pg_trgm'" in code and "IF trgm_schema IS NULL THEN" in code
    trgm_stmt = code[code.index(f"{lexical.TRGM_INDEX} ") :]
    assert "WHERE search_text IS NOT NULL" in trgm_stmt


def test_0120_runs_inside_the_runners_transaction() -> None:
    # The runner switches a file mentioning CONCURRENTLY to autocommit,
    # statement by statement; 0120 must stay one transactional unit.
    assert "CONCURRENTLY" not in _sql().upper()

    class _Cursor:
        def __init__(self) -> None:
            self.executed: list[tuple[str, Any]] = []
            self.connection = self

        def execute(self, sql: str, params: Any = None) -> None:
            self.executed.append((sql, params))

    cursor = _Cursor()
    assert migrations.apply_migration(cursor, "0120", MIGRATION) is True
    assert cursor.executed[0] == (_sql(), None)
    assert cursor.executed[1][1] == ("0120",)


def test_0120_dollar_quotes_balance() -> None:
    statements = migrations._split_statements(_sql())
    kinds = [s.split()[0].upper() for s in (_code(s).strip() for s in statements) if s]
    assert kinds == ["ALTER", "COMMENT", "CREATE", "DO", "DO"]


# ── integration: a real Postgres (skipped without a database URL) ────────────


class _NoCommit:
    """One connection whose transaction the test rolls back at the end."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def __enter__(self) -> _NoCommit:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def cursor(self) -> Any:
        return self._conn.cursor()

    def getconn(self) -> _NoCommit:
        return self

    def putconn(self, _conn: Any) -> None:
        return None


_ROWS = [
    ("c6-it-brca", "Basal tumours in BRCA1 mutation carriers arise from luminal progenitors."),
    ("c6-it-il6", "Exercise raises plasma IL-6 and TNF-alpha in untrained adults."),
    ("c6-it-other", "Perovskite films degrade under humidity and light."),
]


def _seed(cur: Any) -> None:
    for node_id, text in _ROWS:
        cur.execute(
            "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, content, "
            "content_hash, graph_id, search_text) VALUES (%s, 'c6-it', %s, 'entity', "
            "jsonb_build_object('kind', 'chunk', 'artifact_id', %s, 'chunk_index', 0), "
            "md5(%s), '', %s)",
            (node_id, node_id, node_id, text, text),
        )


@pytest.fixture()
def pg_conn(request: pytest.FixtureRequest) -> Any:
    psycopg2 = pytest.importorskip("psycopg2")
    url = request.config.getoption("--db-url") or os.environ.get("UMRS_TEST_DB_URL")
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
def test_0120_applies_and_both_lanes_read_it(pg_conn: Any) -> None:
    lexical.clear_caches()
    with pg_conn.cursor() as cur:
        cur.execute(_sql())
        _seed(cur)
        # Three rows: take the scans a tiny table prefers off the table, so
        # the plan shows whether the lane's expression reaches the index (a
        # GIN index is only ever read by a Bitmap Index Scan).
        cur.execute("SET LOCAL enable_seqscan = off")
        cur.execute("SET LOCAL enable_indexscan = off")
        spec = RetrieveSpec(queries=("q",), project_id="c6-it")
        scope, params = lexical.scope_predicate(spec)
        cur.execute(
            "EXPLAIN " + lexical._search_sql(scope),
            [["brca1"], [2.0], *params, 50, 0.1, 100, 0.1, 32, 50],
        )
        plan = "\n".join(row[0] for row in cur.fetchall())
    assert f"Bitmap Index Scan on {lexical.FTS_INDEX}" in plan, plan
    pool = _NoCommit(pg_conn)
    lex = lexical.lexical_lane(pool, "BRCA1 mutation carriers", spec)
    assert lex.status == "ok", lex.reason
    assert lex.candidates[0].node_id == "c6-it-brca"
    ident = lexical.ident_lane(pool, "Is IL-6 raised?", spec)
    assert ident.status == "ok", ident.reason
    assert [c.node_id for c in ident.candidates] == ["c6-it-il6"]
    lexical.clear_caches()


def test_migration_file_lives_where_the_runner_looks() -> None:
    assert MIGRATION.parent == Path(migrations.__file__).parent and MIGRATION.is_file()
