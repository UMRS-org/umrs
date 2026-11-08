"""ADR-112 wave B1: migration 0135, project links to a user's site allocation.

Pins the table to ``umrs.hpc_accounts.store``: the (venue, project_id,
username) key (invariant I-11), the active/suspended vocabulary, and the
link's dependence on its site account (0134). The integration test applies
0134 and 0135 in a rolled-back transaction.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pytest
from umrs.hpc_accounts.store import LINK_COLUMNS, LINK_STATUSES

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0135_project_site_allocations.sql"
ACCOUNTS = migrations.MIGRATIONS_DIR / "0134_site_accounts.sql"


def _code() -> str:
    lines = MIGRATION.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("--"))


def test_0135_is_the_only_migration_with_its_number_and_follows_0134() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0135") == 1
    assert dict(migrations.get_migration_files())["0135"] == MIGRATION
    assert versions.index("0135") == versions.index("0134") + 1


def test_0135_keys_on_venue_project_and_username() -> None:
    assert "PRIMARY KEY (venue, project_id, username)" in _code()


def test_0135_vocabulary_matches_the_store() -> None:
    match = re.search(r"CHECK \(status IN \(([^)]*)\)\)", _code())
    assert match
    assert {v.strip().strip("'") for v in match.group(1).split(",")} == set(LINK_STATUSES)


def test_0135_has_every_column_the_store_reads() -> None:
    columns = set(re.findall(r"^\s{4}(\w+)\s+(?:TEXT|VARCHAR|TIMESTAMPTZ)", _code(), re.M))
    assert columns == set(LINK_COLUMNS) | {"suspended_reason"}


def test_0135_links_need_their_site_account() -> None:
    assert re.search(
        r"FOREIGN KEY \(venue, username\)\s+REFERENCES site_accounts \(venue, username\)",
        _code(),
    )


# ── integration: a real Postgres (skipped without a database URL) ────────────


@pytest.fixture()
def pg_conn(request: pytest.FixtureRequest) -> Any:
    psycopg2 = pytest.importorskip("psycopg2")
    url = request.config.getoption("--db-url") or os.environ.get("UMRS_TEST_DB_URL")
    if not url:
        pytest.skip("no database URL (--db-url / UMRS_TEST_DB_URL)")
    conn = psycopg2.connect(url)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.integration
def test_0135_refuses_a_link_without_an_account_or_a_known_status(pg_conn: Any) -> None:
    psycopg2 = pytest.importorskip("psycopg2")
    with pg_conn.cursor() as cur:
        cur.execute(ACCOUNTS.read_text())
        cur.execute(MIGRATION.read_text())
        cur.execute("SAVEPOINT s")
        with pytest.raises(psycopg2.errors.ForeignKeyViolation):
            cur.execute(
                "INSERT INTO project_site_allocations (venue, project_id, username,"
                " slurm_account) VALUES ('v-134', 'p1', 'nobody', 'acct')"
            )
        cur.execute("ROLLBACK TO SAVEPOINT s")
        cur.execute("INSERT INTO site_accounts (venue, username) VALUES ('v-134', 'u1')")
        cur.execute(
            "INSERT INTO project_site_allocations (venue, project_id, username,"
            " slurm_account) VALUES ('v-134', 'p1', 'u1', 'acct')"
        )
        cur.execute("SAVEPOINT t")
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute("UPDATE project_site_allocations SET status = 'gone'")
        cur.execute("ROLLBACK TO SAVEPOINT t")
        cur.execute("SELECT status FROM project_site_allocations WHERE venue = 'v-134'")
        assert cur.fetchone() == ("active",)
