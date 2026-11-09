"""ADR-112 wave B1: migration 0134, user-registered HPC site accounts.

Pins the table to the store that relies on it (``umrs.hpc_accounts.store``):
the (venue, username) key every front scopes by (invariant I-11), the status,
method and probe vocabularies, and that no column can hold a private key.
The integration test (``--db-url`` / ``UMRS_TEST_DB_URL``) applies the file in
a rolled-back transaction and shows the checks refuse values outside them.
"""

from __future__ import annotations

import os
import re
from typing import Any

import pytest
from umrs.hpc_accounts.store import ACCOUNT_COLUMNS, ACCOUNT_STATUSES, GATE_PROBES

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0134_site_accounts.sql"


def _code() -> str:
    lines = MIGRATION.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("--"))


def _check_values(name: str) -> set[str]:
    match = re.search(rf"CONSTRAINT {name}\s+CHECK \(\w+ IN \(([^)]*)\)\)", _code())
    assert match, name
    return {v.strip().strip("'") for v in match.group(1).split(",")}


def test_0134_is_the_only_migration_with_its_number_and_follows_0133() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0134") == 1
    assert dict(migrations.get_migration_files())["0134"] == MIGRATION
    assert versions.index("0134") == versions.index("0133") + 1


def test_0134_keys_on_venue_and_username() -> None:
    assert "PRIMARY KEY (venue, username)" in _code()
    assert re.search(r"venue\s+VARCHAR\(32\) NOT NULL", _code())


def test_0134_vocabularies_match_the_store() -> None:
    assert _check_values("site_accounts_status_check") == set(ACCOUNT_STATUSES)
    assert _check_values("site_accounts_method_check") == {"gated_key"}
    assert _check_values("site_accounts_gate_probe_check") == set(GATE_PROBES)


def test_0134_has_every_column_the_store_reads_and_no_key_column() -> None:
    code = _code()
    columns = set(
        re.findall(r"^\s{4}(\w+)\s+(?:TEXT|VARCHAR|JSONB|TIMESTAMPTZ|BOOLEAN)", code, re.M)
    )
    assert set(ACCOUNT_COLUMNS) == columns
    assert not [c for c in columns if "private" in c or "secret" in c]


def test_0134_runs_inside_the_runners_transaction() -> None:
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
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.integration
def test_0134_refuses_values_outside_its_vocabularies(pg_conn: Any) -> None:
    psycopg2 = pytest.importorskip("psycopg2")
    with pg_conn.cursor() as cur:
        cur.execute(MIGRATION.read_text())
        cur.execute("INSERT INTO site_accounts (venue, username) VALUES ('v-133', 'u1')")
        cur.execute(
            "SELECT status, method, gate_probe, verified_assocs FROM site_accounts"
            " WHERE venue = 'v-133'"
        )
        assert cur.fetchone() == ("pending", "gated_key", "unknown", [])
        cur.execute("SAVEPOINT s")
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute("UPDATE site_accounts SET status = 'enabled' WHERE venue = 'v-133'")
        cur.execute("ROLLBACK TO SAVEPOINT s")
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute("INSERT INTO site_accounts (venue, username) VALUES ('v-133', 'u1')")
