"""``umrs-server admin migrate [--check]`` pins (R-10 support item, 2026-09-13).

The verb is a thin dispatch in ``umrs.server.main`` to ``umrs.migrations``:
``--check`` exits 0 when nothing is pending and 1 otherwise (pending list
printed); without ``--check`` it applies. The migrations entry points are
monkeypatched — no database is touched here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.server import main as server_main


class _Conn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.fixture
def fake_migrations(monkeypatch):
    """Patch the connection opener + umrs.migrations verbs; return the call log."""
    calls: dict = {"pending": [], "applied": 0, "apply_calls": []}
    conn = _Conn()
    monkeypatch.setattr(server_main, "_open_migrate_connection", lambda _url: conn)

    from umrs import migrations

    monkeypatch.setattr(migrations, "get_pending_migrations", lambda _c: list(calls["pending"]))
    monkeypatch.setattr(migrations, "get_current_version", lambda _c: "0092")

    def _apply(_c, target_version=None):
        calls["apply_calls"].append(target_version)
        return calls["applied"]

    monkeypatch.setattr(migrations, "apply_migrations", _apply)
    calls["conn"] = conn
    return calls


def _run(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(server_main.sys, "argv", ["umrs-server", *argv])
    return server_main.main()


def test_check_exits_zero_when_current(monkeypatch, capsys, fake_migrations):
    rc = _run(monkeypatch, ["admin", "migrate", "--check", "--db-url", "postgresql://x"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Pending migrations: 0" in out
    assert fake_migrations["apply_calls"] == []  # --check never applies
    assert fake_migrations["conn"].closed is True


def test_check_exits_one_and_lists_pending(monkeypatch, capsys, fake_migrations):
    fake_migrations["pending"] = [
        ("0093", Path("0093_catalog_objects_visibility.sql")),
        ("0094", Path("0094_next.sql")),
    ]
    rc = _run(monkeypatch, ["admin", "migrate", "--check", "--db-url", "postgresql://x"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "Pending migrations: 2" in out
    assert "0093: 0093_catalog_objects_visibility.sql" in out
    assert fake_migrations["apply_calls"] == []


def test_apply_without_check(monkeypatch, capsys, fake_migrations):
    fake_migrations["applied"] = 3
    rc = _run(monkeypatch, ["admin", "migrate", "--db-url", "postgresql://x", "--target", "0093"])
    assert rc == 0
    assert fake_migrations["apply_calls"] == ["0093"]
    assert "Applied 3 migration(s)" in capsys.readouterr().out


def test_database_url_env_is_the_default(monkeypatch, fake_migrations):
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env")
    seen: list[str] = []
    monkeypatch.setattr(
        server_main, "_open_migrate_connection", lambda url: seen.append(url) or _Conn()
    )
    assert _run(monkeypatch, ["admin", "migrate", "--check"]) == 0
    assert seen == ["postgresql://from-env"]


def test_missing_dsn_exits_two(monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    rc = _run(monkeypatch, ["admin", "migrate", "--check"])
    assert rc == 2
    assert "--db-url or DATABASE_URL required" in capsys.readouterr().err


def test_connection_failure_exits_two(monkeypatch, capsys):
    def _boom(_url):
        raise RuntimeError("no route to host")

    monkeypatch.setattr(server_main, "_open_migrate_connection", _boom)
    rc = _run(monkeypatch, ["admin", "migrate", "--check", "--db-url", "postgresql://x"])
    assert rc == 2
    assert "cannot connect: no route to host" in capsys.readouterr().err


def test_other_admin_verbs_still_reach_admin_cli(monkeypatch):
    """The migrate fast-path must not swallow the regular admin dispatcher."""
    seen: list[list[str]] = []
    import umrs.cli.admin as admin_pkg

    monkeypatch.setattr(admin_pkg, "main", lambda argv: seen.append(list(argv)) or 0)
    assert _run(monkeypatch, ["admin", "user", "list"]) == 0
    assert seen == [["user", "list"]]


def test_toplevel_help_mentions_migrate(monkeypatch, capsys):
    assert _run(monkeypatch, ["--help"]) == 0
    assert "admin migrate [--check]" in capsys.readouterr().err
