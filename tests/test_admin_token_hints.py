"""Login hints of the two JWT-minting admin commands (2026-10-03).

A token minted where UMRS runs carries the server's public gateway URL as the
``gateway_url`` claim (--gateway-url, else $UMRS_PUBLIC_GATEWAY_URL, which the
umrs-front converge sets per server container), so the user logs in with
``r4 auth login --api-key TOKEN`` alone. The printed hint is rendered from the
token's own claims by one helper for ``token generate-jwt`` and
``user add --issue-token``, and every printed command is shell-safe: bash read
the old ``<GATEWAY_URL>`` placeholder as an input redirect.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import shlex
from pathlib import Path

import pytest
from jose import jwt as jose_jwt
from umrs.cli.admin.main import (
    PUBLIC_GATEWAY_URL_ENV,
    _format_login_hint,
    _normalize_gateway_url,
    build_parser,
)
from umrs.cli.admin.main import main as admin_main

# The package re-exports main(); the module itself, for monkeypatching.
admin_cli = importlib.import_module("umrs.cli.admin.main")

SECRET = "hint-test-secret"
RCCS_URL = "https://data0.ai.r-ccs.riken.jp/rccs"
EXAMPLE = "https://HOST/SITE"
Q_EXAMPLE = shlex.quote(EXAMPLE)
# A command line (umrs-server / umrs / r4 / export) must never hold <WORD>.
_COMMAND = re.compile(r"(^|[\s`'\"(])(umrs-server|umrs|r4|export)\s")
_ANGLE = re.compile(r"<[A-Za-z_][\w-]*>")


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv(PUBLIC_GATEWAY_URL_ENV, raising=False)
    monkeypatch.delenv("UMRS_SITE_CONFIG", raising=False)


def _run(tmp_path: Path, *argv: str) -> int:
    return admin_main(["--user-files-dir", str(tmp_path / "ufiles"), *argv])


def _claims(token: str) -> dict:
    return jose_jwt.decode(token, SECRET, algorithms=["HS256"])


def _generate_json(tmp_path: Path, capsys: pytest.CaptureFixture[str], *flags: str) -> dict:
    assert (
        _run(
            tmp_path,
            "token",
            "generate-jwt",
            "--user",
            "alice",
            "--project",
            "p1",
            "--json",
            *flags,
        )
        == 0
    )
    return json.loads(capsys.readouterr().out)


def _token_line(out: str, prefix: str) -> str:
    return next(line.split(prefix, 1)[1].strip() for line in out.splitlines() if prefix in line)


def _assert_shell_safe(out: str) -> None:
    # No <PLACEHOLDER> anywhere a user could copy a command from.
    assert "<" not in out and ">" not in out, out


# --- gateway URL resolution --------------------------------------------------


def test_env_default_embeds_gateway_url(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL + "/")
    out = _generate_json(tmp_path, capsys)
    assert out["claims"]["gateway_url"] == RCCS_URL  # trailing slash stripped
    assert _claims(out["token"])["gateway_url"] == RCCS_URL
    # --json shape unchanged: only the claims gained the defaulted URL.
    assert set(out) == {"token", "claims", "expires_at_utc"}


def test_explicit_flag_overrides_env(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    out = _generate_json(tmp_path, capsys, "--gateway-url", "https://other.example/ai4s")
    assert _claims(out["token"])["gateway_url"] == "https://other.example/ai4s"


def test_no_gateway_url_suppresses_env(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    out = _generate_json(tmp_path, capsys, "--no-gateway-url")
    assert "gateway_url" not in out["claims"]
    assert "gateway_url" not in _claims(out["token"])


def test_unset_env_embeds_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = _generate_json(tmp_path, capsys)
    assert "gateway_url" not in out["claims"]


@pytest.mark.parametrize(
    "bad", ["data0.ai.r-ccs.riken.jp/rccs", "ftp://host/x", "https://", "https://a b/c"]
)
def test_invalid_env_is_ignored_with_a_warning(
    bad: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, bad)
    assert (
        _run(tmp_path, "token", "generate-jwt", "--user", "alice", "--project", "p1", "--json") == 0
    )
    captured = capsys.readouterr()
    assert "gateway_url" not in json.loads(captured.out)["claims"]
    assert f"ignoring ${PUBLIC_GATEWAY_URL_ENV}" in captured.err


def test_gateway_url_flags_are_mutually_exclusive() -> None:
    for argv in (
        ["token", "generate-jwt", "--gateway-url", RCCS_URL, "--no-gateway-url"],
        ["user", "add", "bob", "--gateway-url", RCCS_URL, "--no-gateway-url"],
    ):
        with pytest.raises(SystemExit) as exc:
            build_parser().parse_args(argv)
        assert exc.value.code == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://h.example/rccs/", "https://h.example/rccs"),
        ("  http://127.0.0.1:50080  ", "http://127.0.0.1:50080"),
        ("https://h.example", "https://h.example"),
        ("h.example/rccs", None),
        ("https://", None),
        ("", None),
        ("http://[::1", None),
        # shell-active, credential-bearing or odd parts never reach a claim
        ("https://h.example/$(id)", None),
        ("https://h.example/`id`", None),
        ("https://h.example/a;b", None),
        ('https://h.example/"x"', None),
        ("https://user:pw@h.example/rccs", None),
        ("https://h.example/rccs?x=1", None),
        ("https://h.example/rccs#frag", None),
        ("https://h.example:0/rccs", None),
        ("https://h.example:99999/rccs", None),
        ("https://h.example:8443/a_b~c-d.e", "https://h.example:8443/a_b~c-d.e"),
    ],
)
def test_normalize_gateway_url(raw: str, expected: str | None) -> None:
    assert _normalize_gateway_url(raw) == expected


# --- token generate-jwt hint -------------------------------------------------


def test_generate_jwt_hint_with_embedded_url(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    assert _run(tmp_path, "token", "generate-jwt", "--user", "alice", "--project", "p1") == 0
    out = capsys.readouterr().out
    token = _token_line(out, "token:")
    assert f"r4 auth login --api-key {token}" in out
    assert f"carries the gateway URL {RCCS_URL}" in out
    assert "r4 auth login --gateway-url" not in out
    _assert_shell_safe(out)


def test_generate_jwt_hint_without_url_is_shell_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, "token", "generate-jwt", "--user", "alice", "--project", "p1") == 0
    out = capsys.readouterr().out
    token = _token_line(out, "token:")
    assert f"r4 auth login --gateway-url {Q_EXAMPLE} --api-key {token}" in out
    assert PUBLIC_GATEWAY_URL_ENV in out  # says how to embed it next time
    _assert_shell_safe(out)


def test_generate_jwt_no_gateway_url_drops_the_embed_advice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The admin chose no URL while the env is set: no "set $UMRS_..." advice.
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    argv = ("token", "generate-jwt", "--user", "alice", "--project", "p1", "--no-gateway-url")
    assert _run(tmp_path, *argv) == 0
    out = capsys.readouterr().out
    token = _token_line(out, "token:")
    assert f"r4 auth login --gateway-url {Q_EXAMPLE} --api-key {token}" in out
    assert PUBLIC_GATEWAY_URL_ENV not in out
    _assert_shell_safe(out)


def test_explicit_flag_is_normalized(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = _generate_json(tmp_path, capsys, "--gateway-url", RCCS_URL + "/")
    assert _claims(out["token"])["gateway_url"] == RCCS_URL


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "garbage/",
        "h.example/rccs",
        "https://h/$(id)",
        "https://u:p@h/x",
        "https://h/x?y=1",
        EXAMPLE,
    ],
)
def test_invalid_explicit_flag_fails_loudly(
    bad: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An explicit flag exits 2 at parse time: no token, and no user record.
    for argv in (
        ("token", "generate-jwt", "--user", "alice", "--project", "p1"),
        ("--no-reload", "user", "add", "frank", "--project", "p1", "--issue-token"),
    ):
        with pytest.raises(SystemExit) as exc:
            _run(tmp_path, *argv, "--gateway-url", bad)
        assert exc.value.code == 2
        captured = capsys.readouterr()
        assert "--gateway-url" in captured.err
        assert "eyJ" not in captured.out
    assert not (tmp_path / "ufiles" / "frank.json").exists()


# --- user add --issue-token hint ---------------------------------------------


def _issue(tmp_path: Path, capsys: pytest.CaptureFixture[str], *flags: str) -> tuple[str, str]:
    assert (
        _run(
            tmp_path,
            "--no-reload",
            "user",
            "add",
            "bob",
            "--project",
            "p2",
            "--issue-token",
            *flags,
        )
        == 0
    )
    out = capsys.readouterr().out
    token = next(line.strip() for line in out.splitlines() if line.strip().startswith("eyJ"))
    return out, token


def test_issue_token_embeds_env_url_and_hints_api_key_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    out, token = _issue(tmp_path, capsys)
    assert _claims(token)["gateway_url"] == RCCS_URL
    assert f"r4 auth login --api-key {token}" in out
    assert "r4 auth login --gateway-url" not in out
    assert f"umrs login --host HOST --api-key {token}" in out
    _assert_shell_safe(out)


def test_issue_token_without_url_is_shell_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, token = _issue(tmp_path, capsys)
    assert "gateway_url" not in _claims(token)
    assert f"r4 auth login --gateway-url {Q_EXAMPLE} --api-key {token}" in out
    _assert_shell_safe(out)


def test_issue_token_no_gateway_url_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    out, token = _issue(tmp_path, capsys, "--no-gateway-url")
    assert "gateway_url" not in _claims(token)
    assert f"--gateway-url {Q_EXAMPLE} --api-key {token}" in out
    assert PUBLIC_GATEWAY_URL_ENV not in out  # the admin chose no URL
    _assert_shell_safe(out)


def test_issue_token_explicit_flag_wins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    _out, token = _issue(tmp_path, capsys, "--gateway-url", "https://other.example/bdrt")
    assert _claims(token)["gateway_url"] == "https://other.example/bdrt"


def test_both_minting_paths_render_the_same_hint() -> None:
    for payload in ({"gateway_url": RCCS_URL}, {}):
        a = _format_login_hint(payload, "TOK", lead="  Log in with ")
        b = _format_login_hint(payload, "TOK", lead="  1. (recommended) ")
        assert a.splitlines()[1:] == b.splitlines()[1:]
        _assert_shell_safe(a)


# --- hints that mint nothing -------------------------------------------------


def test_user_add_without_token_names_a_runnable_mint_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(tmp_path, "--no-reload", "user", "add", "carol", "--project", "p3") == 0
    out = capsys.readouterr().out
    assert f"generate-jwt --user carol --project p3 --gateway-url {Q_EXAMPLE}" in out
    assert "r4 auth login --api-key TOKEN" in out
    _assert_shell_safe(out)

    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    assert _run(tmp_path, "--no-reload", "user", "add", "dave", "--project", "p3") == 0
    out = capsys.readouterr().out
    assert "generate-jwt --user dave --project p3\n" in out  # the env embeds the URL
    assert f"${PUBLIC_GATEWAY_URL_ENV}={RCCS_URL}" in out
    _assert_shell_safe(out)


def test_generate_template_hint_is_shell_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out_file = tmp_path / "eve.json"
    assert _run(tmp_path, "user", "generate-template", "eve", "-o", str(out_file)) == 0
    out = capsys.readouterr().out
    assert f"r4 config init --gateway-url {Q_EXAMPLE} --umrs-username eve" in out
    _assert_shell_safe(out)

    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    assert _run(tmp_path, "user", "generate-template", "eve", "-o", str(out_file), "--force") == 0
    assert f"r4 config init --gateway-url {RCCS_URL} --umrs-username eve" in capsys.readouterr().out


def test_epilog_onboarding_flow_is_shell_safe() -> None:
    epilog = build_parser().epilog or ""
    assert "r4 auth login --api-key TOKEN" in epilog
    assert PUBLIC_GATEWAY_URL_ENV in epilog
    _assert_shell_safe(epilog)


_GINA_FLAGS = (
    "--email",
    "gina@example.org",
    "--project",
    "p1",
    "--project",
    "p2",
    "--profile",
    "lab",
)


def _claims_from_printed_mint_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict:
    """Run ``user add`` without a token, then the generate-jwt line it prints."""
    assert _run(tmp_path, "--no-reload", "user", "add", "gina", *_GINA_FLAGS) == 0
    out = capsys.readouterr().out
    line = next(ln.strip() for ln in out.splitlines() if "generate-jwt" in ln)
    assert "--projects p1 p2" in line
    assert "--profile lab" in line
    argv = shlex.split(line)
    assert argv[:2] == ["umrs-server", "admin"]
    assert _run(tmp_path, *argv[2:], "--json") == 0
    return json.loads(capsys.readouterr().out)["claims"]


def _claims_from_issue_token(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> dict:
    """The same user in a second store, this time with --issue-token."""
    store = ["--user-files-dir", str(tmp_path / "ufiles2"), "--no-reload"]
    assert admin_main([*store, "user", "add", "gina", *_GINA_FLAGS, "--issue-token"]) == 0
    out = capsys.readouterr().out
    token = next(ln.strip() for ln in out.splitlines() if ln.strip().startswith("eyJ"))
    return _claims(token)


def test_user_add_mint_command_matches_issue_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The printed mint command and --issue-token mint the same claims:
    # several projects travel as --projects (project_memberships), --profile too.
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    minted = _claims_from_printed_mint_command(tmp_path, capsys)
    issued = _claims_from_issue_token(tmp_path, capsys)
    for claims in (minted, issued):
        claims.pop("iat")
        claims.pop("exp")
    assert minted == issued
    assert issued["project_memberships"] == ["p1", "p2"]
    assert issued["profile"] == "lab"
    assert issued["gateway_url"] == RCCS_URL


def test_user_add_no_gateway_url_without_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    argv = ("--no-reload", "user", "add", "hank", "--project", "p3", "--no-gateway-url")
    assert _run(tmp_path, *argv) == 0
    out = capsys.readouterr().out
    assert "generate-jwt --user hank --project p3 --no-gateway-url" in out
    assert f"r4 auth login --gateway-url {Q_EXAMPLE} --api-key TOKEN" in out
    assert f"set ${PUBLIC_GATEWAY_URL_ENV}" not in out
    _assert_shell_safe(out)


def test_invalid_env_warning_says_what_happens(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # No token is minted without --issue-token, so the warning must not claim one.
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, "not-a-url")
    assert _run(tmp_path, "--no-reload", "user", "add", "ivy", "--project", "p3") == 0
    captured = capsys.readouterr()
    assert f"ignoring ${PUBLIC_GATEWAY_URL_ENV}" in captured.err
    assert "the token carries" not in captured.err
    assert "mint command" in captured.err
    assert f"--gateway-url {Q_EXAMPLE}" in captured.out


# --- every printed or documented command is shell-safe ----------------------


def _parsers(parser: argparse.ArgumentParser):
    seen: set[int] = set()
    stack = [parser]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                stack.extend(action.choices.values())


def test_no_angle_placeholder_on_any_command_line_of_the_parser_tree() -> None:
    offenders = []
    for parser in _parsers(build_parser()):
        texts = [parser.description, parser.epilog] + [a.help for a in parser._actions]
        for text in filter(None, texts):
            for line in str(text).splitlines():
                if _COMMAND.search(line) and _ANGLE.search(line):
                    offenders.append(f"{parser.prog}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


class _FakeTokenManager:
    def list_tokens(self) -> list[dict]:
        return []

    def generate_token(self, *_args: object, **_kwargs: object) -> str:
        return "umrs-admin-AbC_d-123"


def test_admin_token_generate_hint_is_shell_safe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(admin_cli, "_token_manager", _FakeTokenManager)
    assert _run(tmp_path, "token", "generate", "ops") == 0
    out = capsys.readouterr().out
    assert "umrs --admin-token umrs-admin-AbC_d-123 COMMAND" in out
    assert "export UMRS_ADMIN_TOKEN=umrs-admin-AbC_d-123" in out
    _assert_shell_safe(out)
