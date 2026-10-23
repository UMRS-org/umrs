"""ADR-099: ``umrs-server admin policy …`` and the WEB column of
``admin session list``. ``urlopen`` is faked (the RS-0 CLI test fixture)."""

from __future__ import annotations

import json
import urllib.request
from urllib.parse import parse_qs, urlparse

import pytest
from tests.test_rs0_admin_http_cli import FAKE_JWT, _Resp, _run

BASE = "http://umrs.test:8000/umrs/admin/session-policy"


@pytest.fixture
def calls(monkeypatch):
    """(requests sent, replies to answer them with, in order)."""
    seen: list[urllib.request.Request] = []
    replies: list[object] = []

    def fake_urlopen(req, timeout=None):
        del timeout
        seen.append(req)
        return _Resp(json.dumps(replies.pop(0) if replies else {}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("UMRS_ADMIN_JWT", raising=False)
    monkeypatch.setenv("UMRS_BASE_URL", "http://umrs.test:8000")
    return seen, replies


@pytest.fixture
def jwt(monkeypatch):
    monkeypatch.setenv("UMRS_ADMIN_JWT", FAKE_JWT)


def _stored(doc, *, venue="rccs", scope="user", subject="alice", layer="admin"):
    rule = {"venue": venue, "scope": scope, "subject": subject, "layer": layer, "doc": doc}
    return {"rules": [rule], "site": "rccs"}


def test_set_merges_key_value_pairs_into_the_stored_rule(tmp_path, calls, jwt, capsys):
    seen, replies = calls
    replies.extend([_stored({"sessions": {"concurrent_user": 20}}), {"ok": True}])
    argv = (
        "policy",
        "set",
        "--scope",
        "user",
        "--subject",
        "alice",
        "budget.tokens_per_day_user=2000000",
        "web.allow=none,allowlist",
        "budget.rpm_user=",
        "--note",
        "pilot",
    )
    assert _run(tmp_path, *argv) == 0
    lookup, put = seen
    assert lookup.get_method() == "GET"
    assert parse_qs(urlparse(lookup.full_url).query) == {"scope": ["user"]}
    assert put.get_method() == "PUT" and put.full_url == f"{BASE}/user/alice"
    assert put.headers["Authorization"] == f"Bearer {FAKE_JWT}"
    assert json.loads(put.data) == {
        "doc": {
            "sessions": {"concurrent_user": 20},
            "budget": {"tokens_per_day_user": 2000000, "rpm_user": None},
            "web": {"allow": ["none", "allowlist"]},
        },
        "note": "pilot",
    }


def test_replace_and_all_sites_and_the_site_subject(tmp_path, calls, jwt):
    seen, _ = calls
    argv = ("policy", "set", "--scope", "site", "--all-sites", "--replace")
    assert _run(tmp_path, *argv, "sessions.concurrent_user=null") == 0
    (put,) = seen
    assert put.full_url == f"{BASE}/site/_"
    assert json.loads(put.data) == {
        "doc": {"sessions": {"concurrent_user": None}},
        "note": "",
        "venue": "",
    }


def test_set_from_a_file_replaces(tmp_path, calls, jwt):
    seen, _ = calls
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps({"lifetime": {"idle_s": 600}}))
    argv = ("policy", "set", "--scope", "project", "--subject", "opossum", "--file", str(doc))
    assert _run(tmp_path, *argv) == 0
    (put,) = seen
    assert json.loads(put.data)["doc"] == {"lifetime": {"idle_s": 600}}


@pytest.mark.parametrize(
    ("pair", "why"),
    [
        ("budget.usd=3", "unknown policy field 'budget.usd'"),
        ("budget.rpm_user=sixty", "budget.rpm_user takes an integer"),
        ("no-equals-sign", "expected KEY=VALUE, got 'no-equals-sign'"),
        ("web=public", "unknown policy field 'web'"),
    ],
)
def test_bad_pairs_exit_2_before_any_write(tmp_path, calls, jwt, capsys, pair, why):
    # The reason is asserted: argparse also exits 2 (for an unknown command),
    # so the exit code alone passes without the policy CLI.
    seen, _ = calls
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, "policy", "set", "--scope", "user", "--subject", "a", "--replace", pair)
    assert exc.value.code == 2
    assert why in capsys.readouterr().err
    assert [r for r in seen if r.get_method() == "PUT"] == []


def test_a_user_rule_needs_its_subject(tmp_path, calls, jwt, capsys):
    with pytest.raises(SystemExit):
        _run(tmp_path, "policy", "unset", "--scope", "user")
    assert "--subject" in capsys.readouterr().err


def test_unset_deletes_the_admin_rule(tmp_path, calls, jwt, capsys):
    seen, replies = calls
    replies.append({"deleted": True})
    assert _run(tmp_path, "policy", "unset", "--scope", "user", "--subject", "alice") == 0
    (delete,) = seen
    assert delete.get_method() == "DELETE"
    assert delete.full_url == f"{BASE}/user/alice?layer=admin"
    assert "removed the admin rule user/alice" in capsys.readouterr().out


def test_effective_and_list(tmp_path, calls, jwt, capsys):
    seen, replies = calls
    replies.extend([{"budget": {}}, _stored({"budget": {"rpm_user": 6}})])
    assert _run(tmp_path, "policy", "effective", "--user", "alice", "--project", "p") == 0
    assert _run(tmp_path, "policy", "list") == 0
    effective, listing = seen
    assert parse_qs(urlparse(effective.full_url).query) == {"user": ["alice"], "project": ["p"]}
    assert listing.full_url == BASE
    out = capsys.readouterr().out
    assert "rpm_user" in out and "alice" in out


def test_policy_commands_refuse_without_a_credential(tmp_path, calls, capsys):
    seen, _ = calls
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, "policy", "list")
    assert exc.value.code == 2
    assert "no admin credential: set $UMRS_ADMIN_JWT" in capsys.readouterr().err
    assert seen == []


def test_session_list_shows_the_web_column(tmp_path, calls, jwt, capsys):
    _, replies = calls
    row = {"session_id": "s-new", "user_id": "alice", "state": "running", "host": "local"}
    replies.append(
        {
            "sessions": [
                {**row, "policy": {"web": {"mode": "allowlist"}}, "web_policy": "allowlist"},
                {**row, "session_id": "s-old", "policy": {}, "web_policy": "public"},
            ]
        }
    )
    assert _run(tmp_path, "session", "list") == 0
    lines = capsys.readouterr().out.splitlines()
    assert "WEB" in lines[0].split()
    assert "allowlist" in next(line for line in lines if "s-new" in line)
    assert "public" in next(line for line in lines if "s-old" in line)
