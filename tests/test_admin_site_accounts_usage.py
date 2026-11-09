"""``umrs-server admin site-accounts usage`` (ADR-112 wave D; plan §6
C/D.0 D4): the ledger's current rows at one venue (every column: the admin
view) and their totals, over the site database; a clear refusal without
one. The Postgres case runs on pg15 (skipped without ``DATABASE_URL``).
"""

from __future__ import annotations

import json
import uuid

import pytest
from hpc_ledger_pg import delete_ledger_rows, ledger_pool_or_skip
from umrs.cli.admin.main import main as admin_main
from umrs.hpc_accounts.usage_ledger import InMemoryUsageLedger, LedgerUnit, PgUsageLedger, Resources


def _run(tmp_path, *argv: str) -> int:
    return admin_main(["--user-files-dir", str(tmp_path), "site-accounts", "usage", *argv])


def _fill(ledger, venue: str) -> None:
    for person, attempt in (("alice", "r1/s/1"), ("bob", "r2/s/1")):
        unit = LedgerUnit(venue=venue, attempt_id=attempt, person=person, lane="hpc_script", tag="t",
                          project_id="p1", site_login=f"{person}-login")  # fmt: skip
        ledger.pending(unit, Resources(2, 1, 600))
        ledger.submitted(unit, f"{len(person)}00", None)


def test_without_a_database_the_command_refuses_clearly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("umrs.cli.admin.cascade.cli_db_pool", lambda: None)
    assert _run(tmp_path, "--venue", "front-a") == 2
    assert "usage ledger unavailable: no site database is configured" in capsys.readouterr().err


@pytest.fixture
def memory(monkeypatch):
    ledger = InMemoryUsageLedger()
    _fill(ledger, "front-a")
    monkeypatch.setattr("umrs.cli.admin.cascade.cli_db_pool", lambda: object())
    monkeypatch.setattr("umrs.hpc_accounts.usage_ledger.PgUsageLedger", lambda _pool: ledger)
    monkeypatch.setattr("umrs.admin.user_store.site_venue", lambda: "front-a")
    return ledger


def test_the_table_shows_every_member_with_their_login(memory, tmp_path, capsys):
    assert _run(tmp_path) == 0
    out = capsys.readouterr().out
    assert all(word in out for word in ("alice", "bob", "alice-login"))


def test_the_person_filter_and_json(memory, tmp_path, capsys):
    assert _run(tmp_path, "--person", "bob", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["person"] for r in data["rows"]] == ["bob"]
    assert "recorded_at" in data["rows"][0]
    assert data["totals"]["reserved_gpu_hours"] == pytest.approx(600 / 3600)


def test_another_venue_without_rows(memory, tmp_path, capsys):
    assert _run(tmp_path, "--venue", "front-b") == 0
    assert "no HPC usage rows at front-b" in capsys.readouterr().out


@pytest.mark.integration
def test_the_command_reads_postgres(tmp_path, monkeypatch, capsys):
    pool = ledger_pool_or_skip()
    venue = f"u-{uuid.uuid4().hex[:8]}"
    _fill(PgUsageLedger(pool), venue)
    monkeypatch.setattr("umrs.cli.admin.cascade.cli_db_pool", lambda: pool)
    try:
        assert _run(tmp_path, "--venue", venue, "--project", "p1", "--json") == 0
        data = json.loads(capsys.readouterr().out)
        assert sorted(r["person"] for r in data["rows"]) == ["alice", "bob"]
        assert {r["state"] for r in data["rows"]} == {"submitted"}
        assert data["totals"]["jobs"] == 2
    finally:
        delete_ledger_rows(pool, [venue])
