"""ADR-112 wave D: migration 0138, the HPC usage ledger (plan §6 C/D.0 D1).

The table is append-only (a trigger refuses UPDATE and DELETE), keyed by
venue on every row (I-11), with no foreign key on ``person`` (the account
cascade never deletes a row); ``hpc_usage_current`` gives each attempt's
latest row. The integration tests run on a migrated pg15.
"""

from __future__ import annotations

import re
import uuid

import pytest
from hpc_ledger_pg import delete_ledger_rows, execute, ledger_pool_or_skip
from umrs.hpc_accounts.usage_ledger import ACCESS_MODES, COLUMNS, LANES, METHODS, SOURCES

from umrs import migrations

MIGRATION = migrations.MIGRATIONS_DIR / "0138_hpc_usage_ledger.sql"


def _code() -> str:
    lines = MIGRATION.read_text().splitlines()
    return "\n".join(line for line in lines if not line.lstrip().startswith("--"))


def _check_values(name: str) -> set[str]:
    match = re.search(rf"CONSTRAINT {name}\s+CHECK \(\w+ IN \(([^)]*)\)\)", _code())
    assert match, name
    return {v.strip().strip("'") for v in match.group(1).split(",")}


def test_0138_follows_0137() -> None:
    versions = [version for version, _ in migrations.get_migration_files()]
    assert versions.count("0138") == 1
    assert versions.index("0138") == versions.index("0137") + 1


def test_0138_vocabularies_match_the_ledger() -> None:
    assert _check_values("hpc_usage_ledger_access_mode_check") == set(ACCESS_MODES)
    assert _check_values("hpc_usage_ledger_method_check") == set(METHODS)
    assert _check_values("hpc_usage_ledger_lane_check") == set(LANES)
    assert _check_values("hpc_usage_ledger_source_check") == set(SOURCES)


def test_0138_has_every_ledger_column_and_no_person_foreign_key() -> None:
    code = _code()
    found = set(re.findall(r"^\s{4}(\w+)\s+(?:TEXT|VARCHAR|BIGINT|INT|DOUBLE|TIMESTAMPTZ|BIGSERIAL)", code, re.M))
    assert found == {"ledger_id", "recorded_at", *COLUMNS}
    assert "REFERENCES" not in code
    assert "BEFORE UPDATE OR DELETE ON hpc_usage_ledger" in code
    assert "SELECT DISTINCT ON (venue, run_id, attempt_id, step) *" in code


def test_0138_indexes_scheduler_job_recovery() -> None:
    assert re.search(
        r"ON hpc_usage_ledger \(venue, scheduler_job_id, ledger_id\)", _code()
    )


def _insert(pool, venue: str, state: str) -> None:
    execute(
        pool,
        "INSERT INTO hpc_usage_ledger (venue, attempt_id, tag, person, access_mode, method, lane,"
        " state, source) VALUES (%s, 'r/s/1', 't', 'alice', 'service_account', 'platform',"
        " 'staged_step', %s, 'estimate')",
        (venue, state),
    )


@pytest.mark.integration
def test_0138_refuses_update_and_delete_and_the_view_shows_the_latest_row() -> None:
    psycopg2 = pytest.importorskip("psycopg2")
    pool = ledger_pool_or_skip()
    venue = f"l-{uuid.uuid4().hex[:8]}"
    try:
        _insert(pool, venue, "pending")
        _insert(pool, venue, "submitted")
        with pytest.raises(psycopg2.errors.RaiseException):
            execute(pool, "UPDATE hpc_usage_ledger SET state = 'x' WHERE venue = %s", (venue,))
        with pytest.raises(psycopg2.errors.RaiseException):
            execute(pool, "DELETE FROM hpc_usage_ledger WHERE venue = %s", (venue,))
        rows = execute(pool, "SELECT state FROM hpc_usage_current WHERE venue = %s", (venue,))
        assert rows == [("submitted",)]
        with pytest.raises(psycopg2.errors.CheckViolation):
            execute(
                pool,
                "INSERT INTO hpc_usage_ledger (venue, attempt_id, tag, person, access_mode, method,"
                " lane, state, source) VALUES (%s, 'x', 't', 'a', 'service_account', 'platform',"
                " 'endpoint', 'pending', 'estimate')",
                (venue,),
            )
    finally:
        delete_ledger_rows(pool, [venue])
