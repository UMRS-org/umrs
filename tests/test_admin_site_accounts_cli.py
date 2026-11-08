"""``umrs-server admin site-accounts list|show`` (ADR-112 wave B1).

Reads one venue's site accounts from the site database; refuses clearly
without one; prints key ids and public fingerprints, never a key.
"""

from __future__ import annotations

import json

import pytest
from umrs.cli.admin.main import build_parser
from umrs.cli.admin.main import main as admin_main
from umrs.hpc_accounts.store import InMemorySiteAccountStore

_VERIFIED = {
    "site_login": "alice1",
    "verified_login": "alice1",
    "verified_assocs": [{"cluster": "c", "account": "grp-a", "partition": "", "qos": ""}],
    "gate_version": "1",
    "gate_sha256": "ab" * 32,
    "signer_fingerprints": ["SHA256:sig"],
    "uv_required": False,
}


@pytest.fixture
def db(monkeypatch):
    tables = InMemorySiteAccountStore("x").tables
    here = InMemorySiteAccountStore("front-a", tables)
    here.issue_key("alice", "tk-000000000001", "SHA256:fp-a")
    here.record_verified("alice", "tk-000000000001", _VERIFIED)
    here.upsert_link("alice", "p1", "grp-a")
    here.issue_key("bob", "tk-000000000002", "SHA256:fp-b")
    InMemorySiteAccountStore("front-b", tables).issue_key("carol", "tk-000000000003", "SHA256:c")
    monkeypatch.setattr("umrs.cli.admin.cascade.cli_db_pool", lambda: object())
    monkeypatch.setattr(
        "umrs.hpc_accounts.store.SiteAccountStore",
        lambda _pool, venue: InMemorySiteAccountStore(venue, tables),
    )
    monkeypatch.setattr("umrs.admin.user_store.site_venue", lambda: "front-a")
    return tables


def _run(tmp_path, *argv: str) -> int:
    return admin_main(["--user-files-dir", str(tmp_path), "site-accounts", *argv])


def test_list_shows_this_venues_accounts_only(db, tmp_path, capsys) -> None:
    assert _run(tmp_path, "list") == 0
    out = capsys.readouterr().out
    assert "alice" in out and "bob" in out and "carol" not in out
    assert "grp-a" in out and "tk-000000000001" in out


def test_list_json_and_another_venue(db, tmp_path, capsys) -> None:
    assert _run(tmp_path, "list", "--venue", "front-b", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["venue"] == "front-b"
    assert [a["username"] for a in data["accounts"]] == ["carol"] and data["links"] == []


def test_show_one_account_with_its_links(db, tmp_path, capsys) -> None:
    assert _run(tmp_path, "show", "alice") == 0
    data = json.loads(capsys.readouterr().out)
    assert data["account"]["status"] == "active"
    assert [x["project_id"] for x in data["links"]] == ["p1"]
    assert _run(tmp_path, "show", "nobody") == 1


def test_an_empty_venue_says_so(db, tmp_path, capsys) -> None:
    assert _run(tmp_path, "list", "--venue", "front-c") == 0
    assert "no HPC site accounts at front-c" in capsys.readouterr().out


def test_no_database_is_a_clear_refusal(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("umrs.cli.admin.cascade.cli_db_pool", lambda: None)
    assert _run(tmp_path, "list", "--venue", "front-a") == 2
    assert "no site database is configured" in capsys.readouterr().err


def test_no_venue_is_a_clear_refusal(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("umrs.cli.admin.cascade.cli_db_pool", lambda: object())
    monkeypatch.setattr("umrs.admin.user_store.site_venue", lambda: "")
    assert _run(tmp_path, "list") == 2
    assert "pass --venue" in capsys.readouterr().err


def test_nothing_printed_is_a_key(db, tmp_path, capsys) -> None:
    _run(tmp_path, "list")
    _run(tmp_path, "list", "--json")
    _run(tmp_path, "show", "alice")
    out = capsys.readouterr().out
    assert "PRIVATE KEY" not in out and "ssh-ed25519" not in out


def test_the_group_is_registered() -> None:
    args = build_parser().parse_args(["site-accounts", "list", "--venue", "v", "--json"])
    assert (args.group, args.action, args.venue, args.json) == ("site-accounts", "list", "v", True)
