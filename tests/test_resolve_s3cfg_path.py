"""Regression tests for `_resolve_s3cfg_path`.

These guard the umrs-front stability property: a per-user MinIO secret must be
created once and stay identical across reloads, restarts and both cluster
servers that share one MinIO. The resolver must therefore converge on the
durable, shared `~/.config/umrs/s3cfg_<username>` path whenever the configured
client path is not reachable on the server — and must never create the remote
client's (ephemeral) path on the server, which previously caused the secret to
be regenerated on every reload (SignatureDoesNotMatch).
"""

import os

from umrs.server.provisioning import _resolve_s3cfg_path


def _user_cfg(s3_path, config_file_path=""):
    return {
        "username": "alice",
        "s3_configuration_file": s3_path,
        "config_file_path": config_file_path,
    }


def test_fast_path_returns_configured_path_when_file_exists(tmp_path):
    cfg = tmp_path / "alice.s3cfg"
    cfg.write_text("[default]\n", encoding="utf-8")
    resolved = _resolve_s3cfg_path(_user_cfg(str(cfg)))
    assert resolved == str(cfg)


def test_remote_client_path_resolves_to_durable_shared_path(tmp_path, monkeypatch):
    # HOME points into tmp so `~/.config/umrs` is sandboxed.
    monkeypatch.setenv("HOME", str(tmp_path))
    # A Mac-style client path that does not (and cannot) exist on this server.
    remote_path = "/Users/alice/.config/umrs/s3cfg_alice"

    resolved = _resolve_s3cfg_path(_user_cfg(remote_path))

    expected = str(tmp_path / ".config" / "umrs" / "s3cfg_alice")
    assert resolved == expected
    # The resolver must NOT have created the ephemeral remote client path.
    assert not os.path.exists("/Users/alice/.config/umrs/s3cfg_alice")


def test_existing_shared_path_is_preferred(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    shared = tmp_path / ".config" / "umrs" / "s3cfg_alice"
    shared.parent.mkdir(parents=True)
    shared.write_text("[default]\nsecret_key = stable\n", encoding="utf-8")

    resolved = _resolve_s3cfg_path(_user_cfg("/Users/alice/.config/umrs/s3cfg_alice"))
    assert resolved == str(shared)


def test_existing_legacy_fallback_is_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    server_dir = tmp_path / "server_conf"
    server_dir.mkdir()
    legacy = server_dir / "s3cfg_alice"
    legacy.write_text("[default]\nsecret_key = legacy\n", encoding="utf-8")
    config_file_path = str(server_dir / "conf_umrs.json")

    resolved = _resolve_s3cfg_path(
        _user_cfg("/Users/alice/.config/umrs/s3cfg_alice", config_file_path)
    )
    assert resolved == str(legacy)


def test_local_server_as_client_first_run_uses_configured_path(tmp_path):
    # Configured path does not yet exist but its parent does (server is its own
    # client, e.g. single-node rccs-cloud) — keep writing to the configured path.
    parent = tmp_path / ".config" / "umrs"
    parent.mkdir(parents=True)
    cfg = parent / "s3cfg_alice"
    resolved = _resolve_s3cfg_path(_user_cfg(str(cfg)))
    assert resolved == str(cfg)
