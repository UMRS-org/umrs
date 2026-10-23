"""ADR-099 D-4 (RS-1 lane Q): migration 0126, the proxy's usage columns.

Pins the statements the code relies on: the three columns the proxy's
INSERT names (``api.llm_usage._INSERT_SQL_VENUE``), the two partial indexes
the quota SUM reads (``origin = 'infer_proxy'``, venue + subject + time),
and the conditional ``umrs_infer`` grant (the 0086 role pattern). The
integration leg (``--db-url`` / ``UMRS_TEST_DB_URL``) applies 0050, 0053 and
0126 in a rolled-back transaction and writes one proxy row.
"""

from __future__ import annotations

import os
import re

import pytest
from umrs.api.llm_usage import _INSERT_SQL_VENUE

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0126_llm_usage_log_venue.sql"


def _code() -> str:
    lines = MIGRATION.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("--"))


def test_0126_is_the_only_migration_with_its_number():
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0126") == 1
    assert dict(migrations.get_migration_files())["0126"] == MIGRATION


def test_0126_adds_the_columns_the_proxy_insert_names():
    code = _code()
    for column in ("venue", "session_id", "origin"):
        assert re.search(rf"ADD COLUMN IF NOT EXISTS {column}\s+TEXT NOT NULL DEFAULT ''", code), (
            column
        )
        assert column in _INSERT_SQL_VENUE


def test_0126_indexes_the_quota_sum():
    code = _code()
    for subject in ("user_id", "project_id"):
        assert re.search(
            rf"ON llm_usage_log \(venue, {subject}, event_time DESC\) "
            r"WHERE origin = 'infer_proxy';",
            code,
        ), subject


def test_0126_grants_umrs_infer_only_where_the_role_exists():
    code = _code()
    assert "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'umrs_infer')" in code
    assert "GRANT SELECT, INSERT ON llm_usage_log TO umrs_infer" in code
    assert "DROP" not in code.upper()


def _db_url(request) -> str:
    return request.config.getoption("--db-url", default=None) or os.environ.get(
        "UMRS_TEST_DB_URL", ""
    )


class _NoCommit:
    """One connection as a pool whose commits never land (rolled back after)."""

    def __init__(self, conn):
        self.conn = conn

    def getconn(self):
        return self

    def putconn(self, _conn):
        pass

    def cursor(self):
        return self.conn.cursor()

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _apply(cur) -> None:
    cur.execute("CREATE ROLE umrs_infer NOLOGIN")  # the grant leg runs too
    for name in ("0050_llm_usage_log.sql", "0053_llm_usage_log_reasoning_tokens.sql"):
        cur.execute((migrations.MIGRATIONS_DIR / name).read_text())
    cur.execute(MIGRATION.read_text())


def test_0126_applies_and_the_quota_queries_read_it(request):
    url = _db_url(request)
    if not url:
        pytest.skip("no --db-url / UMRS_TEST_DB_URL")
    psycopg2 = pytest.importorskip("psycopg2")
    from datetime import datetime, timedelta, timezone

    from umrs.api.admin_usage import usage_rows
    from umrs.api.llm_usage import UsageRecord, _do_record_sync
    from umrs.quotas.enforcer import sum_tokens

    conn = psycopg2.connect(url)
    pool = _NoCommit(conn)
    try:
        with conn.cursor() as cur:
            _apply(cur)
        for user, tokens in (("u", 7), ("u", 5), ("v", 100)):
            _do_record_sync(
                pool,
                UsageRecord(
                    user_id=user,
                    project_id="p",
                    model="m",
                    total_tokens=tokens,
                    venue="rccs",
                    session_id="s",
                    origin="infer_proxy",
                    emit_metrics=False,
                ),
            )
        _do_record_sync(
            pool,
            UsageRecord(
                user_id="u", project_id="p", model="m", total_tokens=1000, emit_metrics=False
            ),
        )
        since = datetime.now(timezone.utc) - timedelta(hours=1)
        assert sum_tokens(pool, venue="rccs", scope="user", subject="u", since=since) == 12
        assert sum_tokens(pool, venue="rccs", scope="project", subject="p", since=since) == 112
        scope, rows = usage_rows(pool, venue="rccs", days=1)
        assert scope == "user"
        assert {(r["subject"], r["total_tokens"], r["requests"]) for r in rows} == {
            ("u", 12, 2),
            ("v", 100, 1),
        }
    finally:
        conn.rollback()
        conn.close()
