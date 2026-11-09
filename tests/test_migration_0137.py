"""ADR-112 wave C: migration 0137, the grant mirror (plan §6 C/D.0 C4).

Pins the table to the store that relies on it (``umrs.hpc_accounts.grants``):
the (venue, grant_id) key (I-11), the vocabularies, the cascade with the
site account, and the four grant-run columns of ``workflow_executions``.
The integration test applies the file on a migrated database and shows the
checks refuse values outside them and the account delete cascades.
"""

from __future__ import annotations

import re
import uuid

import pytest
from hpc_ledger_pg import execute, ledger_pool_or_skip
from umrs.hpc_accounts.grants import COLUMNS, ENDED_REASONS, GATE_CANCELS, INPUTS_MODES, STATUSES

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0137_hpc_grants.sql"


def _code() -> str:
    lines = MIGRATION.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("--"))


def _check_values(name: str) -> set[str]:
    match = re.search(rf"CONSTRAINT {name}\s+CHECK \(\w+ IN \(([^)]*)\)\)", _code())
    assert match, name
    return {v.strip().strip("'") for v in match.group(1).split(",")}


def test_0137_follows_0136_and_is_alone_with_its_number() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0137") == 1
    assert versions.index("0137") == versions.index("0136") + 1


def test_0137_keys_on_venue_and_grant_and_cascades_with_the_account() -> None:
    code = _code()
    assert "PRIMARY KEY (venue, grant_id)" in code
    assert re.search(r"venue\s+VARCHAR\(32\) NOT NULL", code)
    assert "REFERENCES site_accounts (venue, username) ON DELETE CASCADE" in code


def test_0137_vocabularies_match_the_store() -> None:
    assert _check_values("hpc_grants_status_check") == set(STATUSES)
    assert _check_values("hpc_grants_ended_reason_check") == set(ENDED_REASONS)
    assert _check_values("hpc_grants_gate_cancel_check") == set(GATE_CANCELS)
    assert _check_values("hpc_grants_inputs_mode_check") == set(INPUTS_MODES)


def test_0137_has_every_column_the_store_reads() -> None:
    columns = set(re.findall(r"^\s{4}(\w+)\s+(?:TEXT|VARCHAR|CHAR|JSONB|TIMESTAMPTZ)", _code(), re.M))
    assert columns == set(COLUMNS)


def test_0137_adds_the_grant_run_columns_to_executions() -> None:
    code = _code()
    for column in ("grant_id CHAR(32)", "run_spec_b64 TEXT", "run_spec_id CHAR(64)", "workflow_steps_b64 TEXT"):
        assert f"ALTER TABLE workflow_executions ADD COLUMN IF NOT EXISTS {column}" in code
    assert "WHERE grant_id IS NOT NULL" in code
    assert "CONCURRENTLY" not in code.upper()


@pytest.mark.integration
def test_0137_refuses_outside_values_and_cascades_on_postgres() -> None:
    psycopg2 = pytest.importorskip("psycopg2")
    pool = ledger_pool_or_skip()
    venue = f"g-{uuid.uuid4().hex[:8]}"
    execute(pool, "INSERT INTO site_accounts (venue, username) VALUES (%s, 'u1')", (venue,))
    insert = (
        "INSERT INTO hpc_grants (venue, grant_id, username, site_login, payload_b64, signature,"
        " grant_sha256, expires_at, status) VALUES (%s, %s, 'u1', 'l1', 'cA==', 's', %s, now(), %s)"
    )
    execute(pool, insert, (venue, "a" * 32, "b" * 64, "active"))
    with pytest.raises(psycopg2.errors.CheckViolation):
        execute(pool, insert, (venue, "c" * 32, "b" * 64, "revoked"))
    assert execute(pool, "SELECT gate_cancel, ended_reason, usage FROM hpc_grants WHERE venue = %s", (venue,)) == [("", "", {})]
    execute(pool, "DELETE FROM site_accounts WHERE venue = %s", (venue,))
    assert execute(pool, "SELECT count(*) FROM hpc_grants WHERE venue = %s", (venue,)) == [(0,)]
