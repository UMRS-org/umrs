from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

# splice removed (ADR-053): umrs installed editable in .venv
import client_umrs
import pytest


class _FakeConnection:
    mode = "ssh_tunnel"
    server_host = "127.0.0.1"
    control_port = 5555
    tunnel_mode = True
    health_check_timeout_ms = 2500

    def to_dict(self):
        return _expected_connection_info()


def _config_layer(path: str | None, *, valid: bool, source: str = ""):
    return SimpleNamespace(
        path=path,
        valid=valid,
        source=source,
        error=None,
        exists=bool(path),
        details={},
    )


class _FakeAuthContext:
    username = "configured-user"
    is_admin = True

    def can_execute_command(self, command, target_user=None, **kwargs):
        allowed = {
            client_umrs.TaskCmd.RECOMPUTE: True,
            client_umrs.TaskCmd.UPDATE: False,
            client_umrs.TaskCmd.CHECK: True,
            client_umrs.TaskCmd.RELOAD: True,
        }
        return allowed.get(command, False)


def _monitor_user_cfg():
    return {
        "username": "configured-user",
        "experiences": [
            {
                "modalities": [
                    {"s3": {"bucket_alias": "zeta-bucket"}},
                    {"s3": {"bucket_alias": "alpha-bucket"}},
                ]
            }
        ],
    }


def _expected_connection_info():
    return {
        "mode": "ssh_tunnel",
        "server_host": "127.0.0.1",
        "control_port": 5555,
        "data_port_range_label": "5556-5560",
        "tunnel_mode": True,
    }


def _expected_capabilities():
    return {
        "username": "configured-user",
        "is_admin": True,
        "has_user_config": True,
        "bucket_aliases": ["alpha-bucket", "zeta-bucket"],
        "allowed_actions": {
            "recompute": True,
            "update": False,
            "check": True,
            "reload": True,
        },
    }


def _install_fake_dashboard(monkeypatch, dashboard_calls):
    class FakeDashboard:
        def __init__(self, **kwargs):
            dashboard_calls["kwargs"] = kwargs

        def render_once(self):
            dashboard_calls["render_once"] = True

        def run(self):
            dashboard_calls["run"] = True

    fake_dashboard_module = types.ModuleType("umrs.tui.dashboard")
    fake_dashboard_module.Dashboard = FakeDashboard
    monkeypatch.setitem(sys.modules, "umrs.tui.dashboard", fake_dashboard_module)


def _configure_monitor_main_dependencies(monkeypatch, fake_control_mq, user_cfg):
    monkeypatch.setattr(client_umrs, "MessageQueue", MagicMock(return_value=fake_control_mq))
    monkeypatch.setattr(client_umrs, "resolve_connection", lambda **kwargs: _FakeConnection())
    monkeypatch.setattr(
        client_umrs,
        "resolve_loaded_config",
        lambda **kwargs: SimpleNamespace(
            server=_config_layer(None, valid=False),
            user=_config_layer("/tmp/monitor-user.json", valid=True, source="current:user"),
            s3=_config_layer(None, valid=False),
            warnings=[],
        ),
    )
    monkeypatch.setattr(client_umrs, "load_client_cfg_file", lambda path: user_cfg)
    monkeypatch.setattr(
        client_umrs,
        "create_auth_context_from_request",
        lambda client_id, token=None: _FakeAuthContext(),
    )


def _monitor_argv(tab="status"):
    return [
        "umrs",
        "--output",
        "text",
        "--admin-token",
        "secret-token",
        "monitor",
        "--once",
        "--tab",
        tab,
        "--interval",
        "1.5",
        "--status",
        "running",
        "--project",
        "demo-project",
    ]


def _assert_dashboard_bootstrap(kwargs, user_cfg, *, initial_tab="status", job_type_filter=None):
    assert kwargs["client_id"] == "configured-user"
    assert kwargs["interval"] == 1.5
    assert kwargs["status_filter"] == "running"
    assert kwargs["project_filter"] == "demo-project"
    assert kwargs["initial_tab"] == initial_tab
    assert kwargs["job_type_filter"] == job_type_filter
    assert kwargs["config_path"] == "/tmp/monitor-user.json"
    assert kwargs["user_cfg"] is user_cfg
    assert kwargs["connection_info"] == _expected_connection_info()
    assert kwargs["capabilities"] == _expected_capabilities()


def test_main_monitor_bootstraps_dashboard_with_capabilities(monkeypatch):
    dashboard_calls = {}
    _install_fake_dashboard(monkeypatch, dashboard_calls)

    fake_control_mq = MagicMock()
    fake_control_mq.is_server_ready.return_value = True
    user_cfg = _monitor_user_cfg()

    _configure_monitor_main_dependencies(monkeypatch, fake_control_mq, user_cfg)
    monkeypatch.setattr(sys, "argv", _monitor_argv())

    with pytest.raises(SystemExit) as excinfo:
        client_umrs.main()

    assert excinfo.value.code == 0
    assert dashboard_calls["render_once"] is True
    assert "run" not in dashboard_calls
    _assert_dashboard_bootstrap(dashboard_calls["kwargs"], user_cfg)
    fake_control_mq.close.assert_called_once()


def test_main_monitor_endpoints_tab_filters_endpoint_jobs(monkeypatch):
    dashboard_calls = {}
    _install_fake_dashboard(monkeypatch, dashboard_calls)

    fake_control_mq = MagicMock()
    fake_control_mq.is_server_ready.return_value = True
    user_cfg = _monitor_user_cfg()

    _configure_monitor_main_dependencies(monkeypatch, fake_control_mq, user_cfg)
    monkeypatch.setattr(sys, "argv", _monitor_argv(tab="endpoints"))

    with pytest.raises(SystemExit) as excinfo:
        client_umrs.main()

    assert excinfo.value.code == 0
    assert dashboard_calls["render_once"] is True
    _assert_dashboard_bootstrap(
        dashboard_calls["kwargs"],
        user_cfg,
        initial_tab="endpoints",
    )
    fake_control_mq.close.assert_called_once()
