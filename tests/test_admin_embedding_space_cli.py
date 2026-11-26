"""``umrs-server admin embedding-space`` handlers (slice KN-S5, task kn-s5-03).

The store and site modules are patched (their own tests cover them); the
connection is a sentinel from a patched ``registry_conn``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from typing import Any

import pytest
from umrs.cli.admin import embedding_spaces as cli
from umrs.cli.admin.main import build_parser
from umrs.knowledge import embedding_space_site, embedding_space_store
from umrs.knowledge.embedding_spaces import (
    BUNDLED_SPACE,
    CODE_EXISTS,
    CODE_INVALID,
    COLUMNS,
    EmbeddingSpace,
    EmbeddingSpaceError,
)

CONN = object()
ME5B = EmbeddingSpace(
    space_id="me5b", model_name="/models/me5b", dimension=768, legacy_names=("me5b", "x/me5b")
)


@pytest.fixture()
def calls(monkeypatch) -> list[tuple[str, tuple, dict]]:
    """Patch the connection; record every store call as (name, args, kwargs)."""
    seen: list[tuple[str, tuple, dict]] = []

    @contextlib.contextmanager
    def fake_conn():
        seen.append(("conn", (), {}))
        yield CONN

    monkeypatch.setattr(cli, "registry_conn", fake_conn)
    return seen


def _patch_store(monkeypatch, calls, name: str, result: Any) -> None:
    def fake(*args, **kwargs):
        calls.append((name, args, kwargs))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(embedding_space_store, name, fake)


def _run(argv: list[str]) -> int:
    args = build_parser().parse_args(["embedding-space", *argv])
    return args.func(args, None)


def test_space_as_dict_has_every_column_and_the_index_name():
    data = cli.space_as_dict(BUNDLED_SPACE)
    assert set(data) == {*COLUMNS, "index_name"}
    assert data["index_name"] == "hnsw_e5_base_v2"
    assert data["legacy_names"] == list(BUNDLED_SPACE.legacy_names)
    assert data["modalities"] == ["text"]
    json.dumps(data)


def test_list_prints_one_tab_separated_line_per_space(monkeypatch, calls, capsys):
    _patch_store(monkeypatch, calls, "list_spaces", [BUNDLED_SPACE, ME5B])
    assert _run(["list"]) == cli.EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "me5b\tactive\t768\t/models/me5b\tme5b,x/me5b"
    assert lines[0].startswith("e5_base_v2\tactive\t768\t/opt/umrs/models/e5-base-v2\t")
    assert calls[-1] == ("list_spaces", (CONN,), {"include_retired": False})


def test_list_all_json(monkeypatch, calls, capsys):
    _patch_store(monkeypatch, calls, "list_spaces", [ME5B])
    assert _run(["list", "--all", "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out) == [cli.space_as_dict(ME5B)]
    assert calls[-1][2] == {"include_retired": True}


def test_list_says_when_empty(monkeypatch, calls, capsys):
    _patch_store(monkeypatch, calls, "list_spaces", [])
    assert _run(["list"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == "(no embedding spaces)"


def test_space_from_args_builds_an_admin_space():
    args = build_parser().parse_args(
        ["embedding-space", "register", "--space-id", "me5b", "--model-name", "/models/me5b",
         "--dimension", "768", "--legacy-name", "me5b", "--legacy-name", "x/me5b",
         "--query-prefix", "query: "]
    )
    space = cli.space_from_args(args)
    assert space == dataclasses.replace(ME5B, query_prefix="query: ", source="admin")
    assert space.modalities == ("text",) and space.status == "active"


def test_space_from_args_keeps_given_modalities():
    args = build_parser().parse_args(
        ["embedding-space", "register", "--space-id", "c", "--model-name", "/m/c",
         "--dimension", "512", "--modality", "image", "--modality", "text"]
    )
    assert cli.space_from_args(args).modalities == ("image", "text")


def test_register_prints_registered(monkeypatch, calls, capsys):
    _patch_store(monkeypatch, calls, "register_space", True)
    argv = ["register", "--space-id", "me5b", "--model-name", "/models/me5b", "--dimension", "768"]
    assert _run(argv) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == "registered me5b"
    name, args, kwargs = calls[-1]
    assert name == "register_space" and args[0] is CONN and kwargs == {"replace": False}
    assert args[1].space_id == "me5b"


def test_register_replace_prints_replaced(monkeypatch, calls, capsys):
    _patch_store(monkeypatch, calls, "register_space", False)
    argv = ["register", "--space-id", "me5b", "--model-name", "/m", "--dimension", "8", "--replace"]
    assert _run(argv) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == "replaced me5b"
    assert calls[-1][2] == {"replace": True}


def test_a_refused_register_is_exit_one_on_stderr(monkeypatch, calls, capsys):
    _patch_store(
        monkeypatch, calls, "register_space", EmbeddingSpaceError(CODE_EXISTS, "me5b exists")
    )
    argv = ["register", "--space-id", "me5b", "--model-name", "/m", "--dimension", "8"]
    assert _run(argv) == cli.EXIT_REFUSED
    out = capsys.readouterr()
    assert out.out == "" and out.err.strip() == f"error: {CODE_EXISTS}: me5b exists"


def test_no_database_is_exit_two(monkeypatch, capsys):
    @contextlib.contextmanager
    def no_db():
        raise cli.RegistryUnavailable("no site database configured")
        yield  # pragma: no cover

    monkeypatch.setattr(cli, "registry_conn", no_db)
    assert _run(["list"]) == cli.EXIT_NO_DB
    assert capsys.readouterr().err.strip() == "error: no site database configured"


@pytest.mark.parametrize(("verb", "status", "word"), [("retire", "retired", "retired"), ("activate", "active", "activated")])
def test_status_verbs(monkeypatch, calls, capsys, verb, status, word):
    _patch_store(monkeypatch, calls, "set_status", True)
    assert _run([verb, "me5b"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == f"{word} me5b"
    assert calls[-1] == ("set_status", (CONN, "me5b", status), {})


@pytest.mark.parametrize("verb", ["retire", "activate"])
def test_status_verb_on_a_missing_space(monkeypatch, calls, capsys, verb):
    _patch_store(monkeypatch, calls, "set_status", False)
    assert _run([verb, "nope"]) == cli.EXIT_REFUSED
    assert capsys.readouterr().err.strip() == "error: no embedding space nope"


def test_seed_site_without_declarations_opens_no_connection(monkeypatch, calls, capsys):
    monkeypatch.setattr(embedding_space_site, "site_embedding_spaces", lambda: [])
    assert _run(["seed-site"]) == cli.EXIT_OK
    assert capsys.readouterr().out.strip() == "no knowledge.embedding_spaces in the site file"
    assert calls == []


def test_seed_site_reports_each_declared_space(monkeypatch, calls, capsys):
    clip = EmbeddingSpace(space_id="clip", model_name="/m/clip", dimension=512)
    monkeypatch.setattr(embedding_space_site, "site_embedding_spaces", lambda: [ME5B, clip])
    _patch_store(monkeypatch, calls, "seed_spaces", ["clip"])
    assert _run(["seed-site"]) == cli.EXIT_OK
    assert capsys.readouterr().out.splitlines() == ["already registered me5b", "seeded clip"]
    assert calls[-1] == ("seed_spaces", (CONN, [ME5B, clip]), {})


def test_seed_site_dry_run_writes_nothing(monkeypatch, calls, capsys):
    clip = EmbeddingSpace(space_id="clip", model_name="/m/clip", dimension=512)
    monkeypatch.setattr(embedding_space_site, "site_embedding_spaces", lambda: [ME5B, clip])

    def get_space(conn, space_id):
        calls.append(("get_space", (conn, space_id), {}))
        return ME5B if space_id == "me5b" else None

    monkeypatch.setattr(embedding_space_store, "get_space", get_space)
    _patch_store(monkeypatch, calls, "seed_spaces", AssertionError("dry run seeded"))
    assert _run(["seed-site", "--dry-run"]) == cli.EXIT_OK
    assert capsys.readouterr().out.splitlines() == ["already registered me5b", "would seed clip"]


def test_a_bad_site_declaration_is_exit_one(monkeypatch, calls, capsys):
    def bad():
        raise EmbeddingSpaceError(CODE_INVALID, "site.yaml knowledge.embedding_spaces[0]: bad")

    monkeypatch.setattr(embedding_space_site, "site_embedding_spaces", bad)
    assert _run(["seed-site"]) == cli.EXIT_REFUSED
    assert "embedding_spaces[0]" in capsys.readouterr().err
    assert calls == []
