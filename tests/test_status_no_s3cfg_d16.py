"""DEC-016 (2026-11-24): STATUS / reload replies never carry s3cfg to a client.

Two legs survived S-D16-1/2: ``UMRSThreads.record_status`` copied a payload's
``s3cfg_content`` into the status store that STATUS replies return, and the
``umrs`` CLI wrote any ``s3cfg_content`` it found in a STATUS or reload reply
to the local ``s3_configuration_file``. These pins fail on that code.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import client_umrs
import pytest
from umrs.core.UMRSThreads import UMRSThreads

_SECRET = "[default]\naccess_key = alice\nsecret_key = s3cr3t\nhost_base = 127.0.0.1:9000\n"
_REPLY = {"status": "ok", "summary": {"s3cfg_content": _SECRET}}


class _FakeControlAPI:
    def __init__(self, *_args, **_kwargs):
        pass

    def request_status(self, _target):
        return dict(_REPLY)

    def maintenance(self, _action):
        return dict(_REPLY)


def _ctx(tmp_path, action="status"):
    args = SimpleNamespace(
        action=action, output="json", username="alice", target="all", status_all=False
    )
    return SimpleNamespace(
        args=args,
        user={"username": "alice", "s3_configuration_file": str(tmp_path / "s3cfg_alice")},
        configure_file=None,
        connection=SimpleNamespace(minio_url="http://minio.example:9000"),
        control_mq=MagicMock(),
        config_state=SimpleNamespace(user=SimpleNamespace(valid=False, path=None)),
        admin_token=None,
    )


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setattr(client_umrs, "UMRSControlAPI", _FakeControlAPI)


def test_record_status_never_stores_s3cfg_content():
    fake_self = SimpleNamespace(status_store={}, _now_iso=lambda: "2026-11-24T00:00:00Z")
    UMRSThreads.record_status(
        fake_self, "alice", "reload", {"summary": "ok", "s3cfg_content": _SECRET}
    )
    entry = fake_self.status_store["alice"]["reload"]
    assert entry["summary"] == "ok"
    assert "s3cfg_content" not in entry


def test_record_status_keeps_the_other_fields():
    fake_self = SimpleNamespace(status_store={}, _now_iso=lambda: "t0")
    payload = {
        "summary": "s",
        "last_updated_at": "t1",
        "requires_update": True,
        "error": "e",
        "data": {"model": SimpleNamespace(model_name="m"), "n": [1, (2, 3)], "x": object},
    }
    UMRSThreads.record_status(fake_self, None, "k", payload)
    entry = fake_self.status_store["system"]["k"]
    assert entry["last_updated_at"] == "t1"
    assert entry["requires_update"] is True
    assert entry["error"] == "e"
    assert entry["data"]["model"] == "m"
    assert entry["data"]["n"] == [1, [2, 3]]
    assert entry["data"]["x"] == str(object)


def test_status_reply_with_s3cfg_writes_no_file(tmp_path, fake_api):
    client_umrs._action_status(_ctx(tmp_path))
    assert not (tmp_path / "s3cfg_alice").exists()


def test_zmq_reload_reply_with_s3cfg_writes_no_file(tmp_path, fake_api, monkeypatch):
    monkeypatch.setattr(client_umrs, "_get_gateway_url_from_ctx", lambda _ctx: None)
    client_umrs._action_maintenance(_ctx(tmp_path, action="reload"))
    assert not (tmp_path / "s3cfg_alice").exists()


def test_http_reload_does_not_fetch_s3cfg_over_status(tmp_path, fake_api, monkeypatch):
    monkeypatch.setattr(client_umrs, "_get_gateway_url_from_ctx", lambda _ctx: "http://gw")
    monkeypatch.setattr(client_umrs, "_get_api_key_from_profile", lambda _ctx: "k")
    monkeypatch.setattr(client_umrs, "_http_reload", lambda *_a, **_k: {"status": "ok"})
    client_umrs._action_maintenance(_ctx(tmp_path, action="reload"))
    assert not (tmp_path / "s3cfg_alice").exists()


def test_cli_has_no_status_s3cfg_writer():
    assert not hasattr(client_umrs, "_maybe_write_s3cfg_from_status")


def test_maintenance_reject_messages_unchanged(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(client_umrs, "_get_gateway_url_from_ctx", lambda _ctx: "http://gw")
    monkeypatch.setattr(client_umrs, "_get_api_key_from_profile", lambda _ctx: "k")
    monkeypatch.setattr(client_umrs, "_http_reload", lambda *_a, **_k: {"error": "busy"})
    ctx = _ctx(tmp_path, action="reload")
    ctx.args.output = "text"
    assert client_umrs._action_maintenance(ctx) == 0
    assert "rejected: busy" in capsys.readouterr().out
