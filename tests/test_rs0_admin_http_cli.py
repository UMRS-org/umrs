"""RS-0: ``umrs-server admin endpoint|session`` send the admin credential.

Before RS-0 the endpoint subcommands sent no credential at all. They now
send the admin JWT from ``$UMRS_ADMIN_JWT`` / ``--admin-jwt`` as
``Authorization: Bearer`` and refuse, before any request, when none (or a
non-JWT) is configured. ``urlopen`` is faked; the token is a fake string.
"""

from __future__ import annotations

import io
import json
import urllib.request
from pathlib import Path

import pytest
from umrs.cli.admin.main import main as admin_main

FAKE_JWT = "aaaa.bbbb.cccc"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def calls(monkeypatch):
    seen: list[urllib.request.Request] = []
    replies: list[object] = []

    def fake_urlopen(req, timeout=None):
        del timeout
        seen.append(req)
        body = replies.pop(0) if replies else {}
        return _Resp(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.delenv("UMRS_ADMIN_JWT", raising=False)
    monkeypatch.setenv("UMRS_BASE_URL", "http://umrs.test:8000")
    seen_replies = (seen, replies)
    return seen_replies


def _run(tmp_path: Path, *argv: str) -> int:
    return admin_main(["--user-files-dir", str(tmp_path), *argv])


ENDPOINT_COMMANDS = [
    ("endpoint", "list"),
    ("endpoint", "show", "job-1"),
    ("endpoint", "stop", "job-1", "-y"),
    ("endpoint", "stats"),
    ("endpoint", "refresh", "job-1"),
    ("session", "list"),
    ("session", "show", "sess-1"),
    ("session", "stop", "sess-1", "-y"),
]


@pytest.mark.parametrize("argv", ENDPOINT_COMMANDS)
def test_refuses_without_a_credential_before_any_request(tmp_path, calls, capsys, argv):
    seen, _ = calls
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, *argv)
    assert exc.value.code == 2
    assert seen == []
    err = capsys.readouterr().err
    assert "UMRS_ADMIN_JWT" in err and "generate-jwt" in err


def test_refuses_an_opaque_zmq_token(tmp_path, calls, capsys, monkeypatch):
    seen, _ = calls
    monkeypatch.setenv("UMRS_ADMIN_JWT", "opaque-zmq-admin-token")
    with pytest.raises(SystemExit):
        _run(tmp_path, "endpoint", "stats")
    assert seen == []
    assert "not a JWT" in capsys.readouterr().err


@pytest.mark.parametrize("argv", ENDPOINT_COMMANDS)
def test_every_http_admin_command_sends_the_admin_jwt(tmp_path, calls, capsys, monkeypatch, argv):
    seen, _ = calls
    monkeypatch.setenv("UMRS_ADMIN_JWT", FAKE_JWT)
    assert _run(tmp_path, *argv) == 0
    assert len(seen) == 1
    assert seen[0].get_header("Authorization") == f"Bearer {FAKE_JWT}"
    assert seen[0].full_url.startswith("http://umrs.test:8000/umrs/")
    assert FAKE_JWT not in capsys.readouterr().out  # never echoed


def test_flag_wins_over_env(tmp_path, calls, monkeypatch):
    seen, _ = calls
    monkeypatch.setenv("UMRS_ADMIN_JWT", FAKE_JWT)
    assert _run(tmp_path, "endpoint", "--admin-jwt", "x.y.z", "stats") == 0
    assert seen[0].get_header("Authorization") == "Bearer x.y.z"


def test_endpoint_list_asks_for_every_project(tmp_path, calls, monkeypatch):
    seen, replies = calls
    monkeypatch.setenv("UMRS_ADMIN_JWT", FAKE_JWT)
    replies.append({"endpoints": [], "total": 0})
    assert _run(tmp_path, "endpoint", "list") == 0
    assert seen[0].full_url.endswith("/umrs/endpoints?all_projects=true")


def test_session_list_builds_the_filtered_query(tmp_path, calls, capsys, monkeypatch):
    seen, replies = calls
    monkeypatch.setenv("UMRS_ADMIN_JWT", FAKE_JWT)
    replies.append(
        {
            "sessions": [
                {
                    "session_id": "sess-1",
                    "user_id": "alice",
                    "project": "p1",
                    "state": "running",
                    "host": "local",
                    "started_at": "2026-10-22T01:02:03+00:00",
                    "reason": "",
                }
            ],
            "total": 1,
        }
    )
    argv = ("session", "list", "--user", "alice", "--state", "running", "--limit", "5")
    assert _run(tmp_path, *argv) == 0
    url = seen[0].full_url
    assert url.startswith("http://umrs.test:8000/umrs/admin/sessions?")
    assert "user=alice" in url and "state=running" in url and "limit=5" in url
    out = capsys.readouterr().out
    assert "sess-1" in out and "alice" in out and "Total: 1 session(s)" in out


def test_session_stop_posts_the_note(tmp_path, calls, capsys, monkeypatch):
    seen, replies = calls
    monkeypatch.setenv("UMRS_ADMIN_JWT", FAKE_JWT)
    replies.append({"state": "stopped"})
    assert _run(tmp_path, "session", "stop", "a/b", "--note", "breach", "-y") == 0
    req = seen[0]
    assert req.get_method() == "POST"
    assert req.full_url == "http://umrs.test:8000/umrs/admin/sessions/a%2Fb/stop"
    assert json.loads(req.data) == {"note": "breach"}
    assert "session a/b: stopped" in capsys.readouterr().out
