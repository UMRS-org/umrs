"""Startup logging (plan U1 A7; port of upstream 0033b525 and 68077fec).

* ``resolve_site_config_path`` returns the site.yaml ``load_site_config``
  would read, in the same order: explicit path, ``UMRS_SITE_CONFIG``,
  ``./site.yaml``, then the resolved UMRS config dir.
* The server logs the ZMQ bind host, port, data-port range and auth type at
  startup, and the site config path it resolved, never key material.
"""

from __future__ import annotations

import logging
import types

import pytest
from umrs.orchestrator.site_config import (
    load_site_config,
    reset_site_config,
    resolve_site_config_path,
)

MINIMAL_YAML = "name: {name}\n"


@pytest.fixture(autouse=True)
def _isolated_search_paths(tmp_path, monkeypatch):
    """No env path, an empty cwd and an empty config dir by default."""
    monkeypatch.delenv("UMRS_SITE_CONFIG", raising=False)
    monkeypatch.delenv("UMRS_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    reset_site_config()
    yield
    reset_site_config()


def _site(path, name):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MINIMAL_YAML.format(name=name))
    return path


class TestResolveSiteConfigPath:
    def test_none_when_nothing_found(self):
        assert resolve_site_config_path() is None

    def test_missing_explicit_path_is_none(self, tmp_path):
        _site(tmp_path / "cwd" / "site.yaml", "cwd-site")
        assert resolve_site_config_path(str(tmp_path / "nope.yaml")) is None

    def test_explicit_path_wins(self, tmp_path, monkeypatch):
        explicit = _site(tmp_path / "explicit.yaml", "explicit")
        env = _site(tmp_path / "env.yaml", "env")
        monkeypatch.setenv("UMRS_SITE_CONFIG", str(env))
        assert resolve_site_config_path(str(explicit)) == str(explicit)
        assert load_site_config(str(explicit)).name == "explicit"

    def test_env_before_cwd(self, tmp_path, monkeypatch):
        env = _site(tmp_path / "env.yaml", "env")
        _site(tmp_path / "cwd" / "site.yaml", "cwd-site")
        monkeypatch.setenv("UMRS_SITE_CONFIG", str(env))
        assert resolve_site_config_path() == str(env)
        assert load_site_config().name == "env"

    def test_cwd_before_config_dir(self, tmp_path):
        cwd_site = _site(tmp_path / "cwd" / "site.yaml", "cwd-site")
        _site(tmp_path / "xdg" / "umrs" / "site.yaml", "dir-site")
        assert resolve_site_config_path() == str(cwd_site)
        assert load_site_config().name == "cwd-site"

    def test_config_dir_last(self, tmp_path):
        dir_site = _site(tmp_path / "xdg" / "umrs" / "site.yaml", "dir-site")
        assert resolve_site_config_path() == str(dir_site.resolve())
        assert load_site_config().name == "dir-site"

    def test_env_pointing_at_missing_file_falls_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UMRS_SITE_CONFIG", str(tmp_path / "gone.yaml"))
        cwd_site = _site(tmp_path / "cwd" / "site.yaml", "cwd-site")
        assert resolve_site_config_path() == str(cwd_site)

    def test_load_raises_when_none_found(self):
        with pytest.raises(FileNotFoundError):
            load_site_config()


SECRET = "S3CR3T-curve-key-material-must-not-leak"


@pytest.fixture
def server_umrs():
    return pytest.importorskip("server_umrs")


@pytest.fixture
def curve_env(monkeypatch):
    monkeypatch.setenv("UMRS_ZMQ_AUTH_TYPE", "Curve")
    for name in (
        "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY",
        "UMRS_ZMQ_CURVE_SERVER_SECRET_KEY",
        "UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY",
    ):
        monkeypatch.setenv(name, SECRET)


class TestStartupLogLines:
    @pytest.mark.usefixtures("curve_env")
    def test_bind_line_has_host_port_range_and_auth_type(self, server_umrs, caplog):
        mq = types.SimpleNamespace(tunnel_mode=False, port=5555)
        with caplog.at_level(logging.INFO, logger="server_umrs"):
            server_umrs._log_zmq_bind(mq)
        text = caplog.text
        assert "*:5555" in text
        assert f"data ports {server_umrs.DATA_PORT_RANGE.label}" in text
        assert "auth=curve" in text
        assert SECRET not in text

    def test_bind_line_tunnel_mode_is_loopback_and_default_auth(
        self, server_umrs, monkeypatch, caplog
    ):
        monkeypatch.delenv("UMRS_ZMQ_AUTH_TYPE", raising=False)
        mq = types.SimpleNamespace(tunnel_mode=True, port=6000)
        with caplog.at_level(logging.INFO, logger="server_umrs"):
            server_umrs._log_zmq_bind(mq)
        assert "127.0.0.1:6000" in caplog.text
        assert "auth=none" in caplog.text

    def test_site_config_line_names_resolved_file(
        self, server_umrs, tmp_path, monkeypatch, caplog
    ):
        site = _site(tmp_path / "env.yaml", "env")
        monkeypatch.setenv("UMRS_SITE_CONFIG", str(site))
        with caplog.at_level(logging.INFO, logger="server_umrs"):
            server_umrs._log_site_config_path()
        assert f"Site config file: {site}" in caplog.text

    def test_site_config_line_when_none_found(self, server_umrs, caplog):
        with caplog.at_level(logging.INFO, logger="server_umrs"):
            server_umrs._log_site_config_path()
        assert "none found (environment variables only)" in caplog.text
