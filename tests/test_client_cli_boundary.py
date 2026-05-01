"""CLI ownership boundary tests."""

from __future__ import annotations

import json
import sys

import pytest

from umrs.client import cli


def test_umrs_client_admin_is_disabled_by_default(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["umrs-client", "admin", "token", "list"])

    exit_code = cli.main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "umrs-client admin is disabled" in captured.err
    assert "umrs-server admin" in captured.err


def test_umrs_client_admin_denial_respects_json_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["umrs-client", "--output", "json", "admin", "token", "list"],
    )

    exit_code = cli.main()

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["status"] == "error"
    assert "umrs-server admin" in payload["message"]


def test_umrs_client_admin_escape_hatch_is_removed(monkeypatch, capsys) -> None:
    monkeypatch.setenv("UMRS_CLIENT_ENABLE_LEGACY_ADMIN", "1")
    monkeypatch.setattr(sys, "argv", ["umrs-client", "admin", "token", "list"])

    exit_code = cli.main()

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "umrs-client admin is disabled" in captured.err


def test_requested_action_skips_global_options() -> None:
    argv = [
        "umrs-client",
        "--profile",
        "rccs",
        "--output=json",
        "admin",
        "token",
        "list",
    ]

    assert cli._requested_action(argv) == "admin"


def test_umrs_client_help_points_to_r4_and_admin_cli(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["umrs-client", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "Use r4 for Gateway-mediated research workflows" in captured.out
    assert "Use umrs-server admin" in captured.out
