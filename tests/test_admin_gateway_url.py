"""Per-venue gateway URL when minting tokens (S-D47c, DEC-D47, 2026-11-24).

A hub that mints for several venues must not put one venue's URL into every
user's token (an octopus user would get the rccs URL). ``--venue VENUE`` on
``token generate-jwt`` and ``user add`` reads ``UMRS_<VENUE>_PUBLIC_GATEWAY_URL``
first; ``UMRS_PUBLIC_GATEWAY_URL`` stays the fallback where one URL serves all,
and a stand-alone that passes no ``--venue`` mints exactly as before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jose import jwt as jose_jwt
from umrs.cli.admin.main import PUBLIC_GATEWAY_URL_ENV, venue_gateway_url_env
from umrs.cli.admin.main import main as admin_main

SECRET = "venue-url-test-secret"
RCCS_URL = "https://gw.example.org/rccs"
OCTOPUS_URL = "https://gw.example.org/octopus"
RCCS_ENV = "UMRS_RCCS_PUBLIC_GATEWAY_URL"
OCTOPUS_ENV = "UMRS_OCTOPUS_PUBLIC_GATEWAY_URL"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("UMRS_SITE_CONFIG", raising=False)
    for name in (PUBLIC_GATEWAY_URL_ENV, RCCS_ENV, OCTOPUS_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def two_venues(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hub's .env: both venue lines, plus a single URL left from one site."""
    monkeypatch.setenv(RCCS_ENV, RCCS_URL)
    monkeypatch.setenv(OCTOPUS_ENV, OCTOPUS_URL + "/")
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)


def _run(tmp_path: Path, *argv: str) -> int:
    return admin_main(["--user-files-dir", str(tmp_path / "ufiles"), *argv])


def _mint(tmp_path: Path, capsys: pytest.CaptureFixture[str], *flags: str) -> dict:
    argv = ["token", "generate-jwt", "--user", "alice", "--project", "p1", "--json"]
    assert _run(tmp_path, *argv, *flags) == 0
    token = json.loads(capsys.readouterr().out)["token"]
    return jose_jwt.decode(token, SECRET, algorithms=["HS256"])


@pytest.mark.parametrize(
    ("venue", "name"),
    [
        ("octopus", OCTOPUS_ENV),
        ("rccs-front", "UMRS_RCCS_FRONT_PUBLIC_GATEWAY_URL"),
        ("ai4s.b", "UMRS_AI4S_B_PUBLIC_GATEWAY_URL"),
    ],
)
def test_venue_key_is_derived(venue: str, name: str) -> None:
    assert venue_gateway_url_env(venue) == name


@pytest.mark.usefixtures("two_venues")
@pytest.mark.parametrize(("venue", "url"), [("octopus", OCTOPUS_URL), ("rccs", RCCS_URL)])
def test_each_venue_user_gets_its_venue_url(
    venue: str, url: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _mint(tmp_path, capsys, "--venue", venue)["gateway_url"] == url


@pytest.mark.usefixtures("two_venues")
def test_issue_token_uses_the_venue_url(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc = _run(
        tmp_path, "user", "add", "bob", "--project", "p1", "--issue-token", "--venue", "octopus"
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert f"carries the gateway URL {OCTOPUS_URL}," in out
    assert RCCS_URL not in out


def test_single_url_stand_alone_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    monkeypatch.setenv(OCTOPUS_ENV, OCTOPUS_URL)  # ignored without --venue
    assert _mint(tmp_path, capsys)["gateway_url"] == RCCS_URL


@pytest.mark.usefixtures("two_venues")
def test_explicit_flag_beats_the_venue(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    claims = _mint(tmp_path, capsys, "--venue", "octopus", "--gateway-url", "https://x.example/y")
    assert claims["gateway_url"] == "https://x.example/y"


@pytest.mark.usefixtures("two_venues")
def test_no_gateway_url_beats_the_venue(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert "gateway_url" not in _mint(tmp_path, capsys, "--venue", "octopus", "--no-gateway-url")


def _mint_venue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], venue: str
) -> tuple[dict, str]:
    argv = ["token", "generate-jwt", "--user", "alice", "--project", "p1", "--json"]
    assert _run(tmp_path, *argv, "--venue", venue) == 0
    captured = capsys.readouterr()
    return json.loads(captured.out)["claims"], captured.err


def test_unset_venue_line_on_a_single_url_server_falls_back_with_a_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)  # the only URL key
    claims, err = _mint_venue(tmp_path, capsys, "octopus")
    assert claims["gateway_url"] == RCCS_URL
    assert f"${OCTOPUS_ENV} is unset" in err
    assert f"${PUBLIC_GATEWAY_URL_ENV}={RCCS_URL} (no other venue URL is set)" in err


# Review fix (2026-11-24): on the fleet each container's single URL is its own
# site's and every venue key arrives through env_file: .env; the converge
# drops an invalid line, so an unset venue key must not borrow that URL.
@pytest.mark.parametrize("venue", ["octopus", "octopus-front", "Octopus", "octopus.front"])
def test_unset_venue_line_never_takes_another_venues_url(
    venue: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    monkeypatch.setenv(RCCS_ENV, RCCS_URL)
    claims, err = _mint_venue(tmp_path, capsys, venue)
    assert "gateway_url" not in claims
    assert f"${RCCS_ENV}" in err
    assert "--gateway-url https://HOST/SITE" in err


def test_user_add_without_venue_line_prints_the_example(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    monkeypatch.setenv(RCCS_ENV, RCCS_URL)
    assert _run(tmp_path, "user", "add", "dave", "--project", "p1", "--venue", "octopus") == 0
    out = capsys.readouterr().out
    assert RCCS_URL not in out
    mint = next(line.strip() for line in out.splitlines() if "generate-jwt" in line)
    assert mint.endswith("--gateway-url https://HOST/SITE")


def test_invalid_venue_line_never_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(PUBLIC_GATEWAY_URL_ENV, RCCS_URL)
    monkeypatch.setenv(OCTOPUS_ENV, "ftp://bad/x")
    argv = ["token", "generate-jwt", "--user", "alice", "--project", "p1", "--json"]
    assert _run(tmp_path, *argv, "--venue", "octopus") == 0
    captured = capsys.readouterr()
    assert "gateway_url" not in json.loads(captured.out)["claims"]
    assert f"warning: ignoring ${OCTOPUS_ENV}" in captured.err


@pytest.mark.parametrize("bad", ["", "-x", "a b", "oct;rm", "a/b"])
def test_malformed_venue_fails_loudly(bad: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, "token", "generate-jwt", "--user", "alice", "--venue", bad)
    assert exc.value.code == 2


@pytest.mark.usefixtures("two_venues")
def test_user_add_mint_command_carries_the_venue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path, "user", "add", "carol", "--project", "p1", "--venue", "octopus") == 0
    out = capsys.readouterr().out
    assert f"embeds ${OCTOPUS_ENV}={OCTOPUS_URL}" in out
    mint = next(line.strip() for line in out.splitlines() if "generate-jwt" in line)
    assert mint.endswith("--venue octopus")
