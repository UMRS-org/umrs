"""``umrs-server admin embedding-space index`` (slice KN-S6, task kn-s6-03).

The store and index modules are patched (their own tests cover them); the
connection is a sentinel from a patched ``index_conn``. The plumbing
(``index_conn``: autocommit on, restored after) is tested here too.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
from umrs.cli.admin import embedding_spaces as cli
from umrs.cli.admin.main import build_parser
from umrs.knowledge import embedding_space_index, embedding_space_store
from umrs.knowledge.embedding_space_index import IndexWork
from umrs.knowledge.embedding_spaces import (
    CODE_INVALID,
    EmbeddingSpace,
    EmbeddingSpaceError,
    IndexState,
)

CONN = object()
ME5B = EmbeddingSpace(space_id="me5b", model_name="/models/me5b", dimension=768)
OLD = EmbeddingSpace(space_id="old", model_name="/models/old", dimension=8, status="retired")
WORK = [
    IndexWork("me5b", "hnsw_me5b", ("CREATE me5b", "COMMENT me5b")),
    IndexWork("old", "hnsw_old", ("DROP old",)),
]


@pytest.fixture()
def calls(monkeypatch) -> list[tuple[str, tuple, dict]]:
    """Patch the connection and every module call; record (name, args, kwargs)."""
    seen: list[tuple[str, tuple, dict]] = []

    @contextlib.contextmanager
    def fake_conn():
        seen.append(("conn", (), {}))
        yield CONN

    monkeypatch.setattr(cli, "index_conn", fake_conn)
    _patch(monkeypatch, seen, embedding_space_store, "list_spaces", [ME5B, OLD])
    states = {"me5b": IndexState(), "old": IndexState(exists=True)}
    _patch(monkeypatch, seen, embedding_space_index, "read_index_states", states)
    _patch(monkeypatch, seen, embedding_space_index, "plan_index_work", WORK)
    _patch(monkeypatch, seen, embedding_space_index, "stored_model_names", ["/x/me5b"])
    _patch(monkeypatch, seen, embedding_space_index, "coverage_warnings", [])
    _patch(
        monkeypatch, seen, embedding_space_index, "apply_index_work", ["CREATE me5b", "COMMENT me5b", "DROP old"]
    )
    return seen


def _patch(monkeypatch, seen, module, name: str, result: Any) -> None:
    def fake(*args, **kwargs):
        seen.append((name, args, kwargs))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(module, name, fake)


def _run(argv: list[str]) -> int:
    args = build_parser().parse_args(["embedding-space", *argv])
    return args.func(args, None)


def _names(calls) -> list[str]:
    return [name for name, _a, _k in calls]


# ── the parser (wired by the designer) ──────────────────────────────────────


def test_the_index_verb_parses_spaces_and_dry_run():
    args = build_parser().parse_args(
        ["embedding-space", "index", "--space", "me5b", "--space", "old", "--dry-run"]
    )
    assert args.func is cli.cmd_index
    assert (args.space, args.dry_run) == (["me5b", "old"], True)
    plain = build_parser().parse_args(["embedding-space", "index"])
    assert (plain.space, plain.dry_run) == ([], False)


# ── the handler ─────────────────────────────────────────────────────────────


def test_index_builds_every_planned_statement(calls, capsys):
    assert _run(["index"]) == cli.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out == ["ran: CREATE me5b", "ran: COMMENT me5b", "ran: DROP old"]


def test_index_calls_the_modules_in_order(calls):
    assert _run(["index"]) == cli.EXIT_OK
    assert _names(calls) == [
        "conn",
        "list_spaces",
        "read_index_states",
        "plan_index_work",
        "stored_model_names",
        "coverage_warnings",
        "apply_index_work",
    ]


def test_index_passes_the_connection_and_results_along(calls):
    assert _run(["index"]) == cli.EXIT_OK
    assert calls[1] == ("list_spaces", (CONN,), {"include_retired": True})
    assert calls[2] == ("read_index_states", (CONN, [ME5B, OLD]), {})
    _name, args, kwargs = calls[3]
    assert args == ([ME5B, OLD], {"me5b": IndexState(), "old": IndexState(exists=True)})
    assert kwargs == {"only": None}
    assert calls[4] == ("stored_model_names", (CONN,), {})
    assert calls[5] == ("coverage_warnings", ([ME5B, OLD], ["/x/me5b"]), {})
    assert calls[6] == ("apply_index_work", (CONN, WORK), {})


def test_index_passes_the_selected_spaces(calls):
    assert _run(["index", "--space", "me5b"]) == cli.EXIT_OK
    assert calls[3][2] == {"only": ["me5b"]}


def test_dry_run_prints_and_runs_nothing(calls, capsys):
    assert _run(["index", "--dry-run"]) == cli.EXIT_OK
    out = capsys.readouterr().out.splitlines()
    assert out == ["would run: CREATE me5b", "would run: COMMENT me5b", "would run: DROP old"]
    assert "apply_index_work" not in _names(calls)


def test_nothing_to_do_says_so(monkeypatch, calls, capsys):
    _patch(monkeypatch, calls, embedding_space_index, "plan_index_work", [])
    assert _run(["index"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == "all embedding-space indexes are current"
    assert "apply_index_work" not in _names(calls)


def test_coverage_warnings_go_to_stderr(monkeypatch, calls, capsys):
    _patch(monkeypatch, calls, embedding_space_index, "coverage_warnings", ["me5b: a", "x: b"])
    assert _run(["index", "--dry-run"]) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert captured.err.splitlines() == ["warning: me5b: a", "warning: x: b"]
    assert "warning" not in captured.out


def test_an_unknown_space_is_refused(monkeypatch, calls, capsys):
    refusal = EmbeddingSpaceError(CODE_INVALID, "no space 'nope'")
    _patch(monkeypatch, calls, embedding_space_index, "plan_index_work", refusal)
    assert _run(["index", "--space", "nope"]) == cli.EXIT_REFUSED
    assert capsys.readouterr().err.startswith(f"error: {CODE_INVALID}")


def test_no_database_exits_with_its_code(monkeypatch, capsys):
    @contextlib.contextmanager
    def missing():
        raise cli.RegistryUnavailable("no site database configured")
        yield  # pragma: no cover

    monkeypatch.setattr(cli, "index_conn", missing)
    assert _run(["index"]) == cli.EXIT_NO_DB
    assert "no site database" in capsys.readouterr().err


# ── index_conn (designer plumbing) ──────────────────────────────────────────


class _Conn:
    def __init__(self) -> None:
        self.autocommit = False
        self.seen: list[Any] = []

    def rollback(self) -> None:
        self.seen.append("rollback")


class _Pool:
    def __init__(self) -> None:
        self.conn = _Conn()
        self.returned: list[Any] = []

    def getconn(self) -> _Conn:
        return self.conn

    def putconn(self, conn: _Conn) -> None:
        self.returned.append((conn, conn.autocommit))


def test_index_conn_is_autocommit_and_restored(monkeypatch):
    from umrs.cli.admin import cascade

    pool = _Pool()
    monkeypatch.setattr(cascade, "cli_db_pool", lambda: pool)
    with cli.index_conn() as conn:
        assert conn.autocommit is True and conn.seen == ["rollback"]
    assert pool.returned == [(pool.conn, False)]


def test_index_conn_without_a_database(monkeypatch):
    from umrs.cli.admin import cascade

    monkeypatch.setattr(cascade, "cli_db_pool", lambda: None)
    with pytest.raises(cli.RegistryUnavailable), cli.index_conn():
        pass
