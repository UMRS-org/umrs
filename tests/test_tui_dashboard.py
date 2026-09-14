"""Tests for the Dashboard shim (non-interactive render_once).

The old ~3200-line Rich Live Dashboard has been replaced by a Textual
MonitorApp.  Dashboard is now a thin shim that delegates run() to
MonitorApp and provides render_once() for snapshot output.  These
tests verify the non-interactive snapshot path used by --once mode
and CI validation.
"""

from __future__ import annotations

import json
import sys
import types
from io import StringIO
from unittest.mock import MagicMock

from rich.console import Console

SAMPLE_JOBS = [
    {
        "job_id": "job-abc123def456",
        "project_id": "default",
        "user_id": "testuser",
        "job_type": "endpoint_provision",
        "spec": {"model": "llama-3"},
        "status": "running",
        "priority": 0,
        "desired_max_attempts": 3,
        "restart_policy": "on_crash",
        "keep_alive": False,
        "attempts": [],
        "created_at": "2026-04-05T09:58:00",
        "updated_at": "2026-04-05T10:00:00",
    },
    {
        "job_id": "job-fffeeedddbbb",
        "project_id": "default",
        "user_id": "testuser",
        "job_type": "index_build",
        "spec": {},
        "status": "completed",
        "priority": 0,
        "desired_max_attempts": 5,
        "restart_policy": "on_failure",
        "keep_alive": False,
        "attempts": [],
        "created_at": "2026-04-04T11:58:00",
        "updated_at": "2026-04-04T12:05:00",
    },
]


def _make_mock_mq() -> MagicMock:
    mq = MagicMock()
    mq.request_job_list.return_value = {
        "status": "ok",
        "jobs": SAMPLE_JOBS,
        "total": len(SAMPLE_JOBS),
    }
    mq.request_status.return_value = json.dumps({"status": "ok"})
    mq.request_model_command.return_value = json.dumps(
        {"status": "ok", "models": [], "workflows": []}
    )
    mq.is_server_ready.return_value = True
    mq.notify_server.return_value = True
    return mq


class TestRenderOnce:
    """Test the non-interactive render_once snapshot path."""

    def test_render_once_shows_jobs_table(self):
        from umrs.tui.dashboard import Dashboard

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120, record=True)
        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser", console=console)

        dash.render_once()

        plain = console.export_text(clear=False)
        assert "UMRS Jobs" in plain
        assert "endpoint_provision" in plain or "Job" in plain

    def test_render_once_with_empty_jobs(self):
        from umrs.tui.dashboard import Dashboard

        mq = _make_mock_mq()
        mq.request_job_list.return_value = {"status": "ok", "jobs": [], "total": 0}
        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120, record=True)
        dash = Dashboard(control_mq=mq, client_id="testuser", console=console)

        dash.render_once()

        plain = console.export_text(clear=False)
        assert "UMRS Jobs" in plain

    def test_render_once_uses_initial_tab_jobs(self):
        from umrs.tui.dashboard import Dashboard

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120, record=True)
        dash = Dashboard(
            control_mq=_make_mock_mq(), client_id="testuser", console=console, initial_tab="jobs"
        )

        dash.render_once()

        plain = console.export_text(clear=False)
        assert "Jobs" in plain

    def test_constructor_accepts_all_kwargs(self):
        """Dashboard __init__ must accept all configuration kwargs without error."""
        from umrs.tui.dashboard import Dashboard

        mq = _make_mock_mq()
        dash = Dashboard(
            control_mq=mq,
            client_id="testuser",
            interval=5.0,
            status_filter="running",
            project_filter="default",
            initial_tab="jobs",
            job_type_filter="endpoint_provision",
            config_path="/tmp/test.yaml",
            user_cfg={"key": "val"},
            connection_info={"gateway": "http://localhost"},
            capabilities={"is_admin": True},
            workflow_id="wf-1",
        )

        assert dash._status_filter == "running"
        assert dash._project_filter == "default"
        assert dash._initial_tab == "jobs"
        assert dash._capabilities == {"is_admin": True}

    def test_constructor_signature_matches_old_api(self):
        """Backward-compat: old calling convention still works."""
        from umrs.tui.dashboard import Dashboard

        mq = _make_mock_mq()
        dash = Dashboard(mq, "testuser", 2.0, None)
        assert dash._interval == 2.0

    def test_run_delegates_to_monitor_app(self):
        """run() must create MonitorApp and call its run()."""
        from umrs.tui.dashboard import Dashboard

        mq = _make_mock_mq()
        dash = Dashboard(control_mq=mq, client_id="testuser")

        mock_app_instance = MagicMock()
        with MagicMock() as mock_app_cls:
            mock_app_cls.return_value = mock_app_instance
            import umrs.tui.app

            original = umrs.tui.app.MonitorApp
            try:
                umrs.tui.app.MonitorApp = mock_app_cls  # type: ignore[assignment]
                dash.run()
            finally:
                umrs.tui.app.MonitorApp = original

        mock_app_cls.assert_called_once()
        mock_app_instance.run.assert_called_once()

    def test_render_once_handles_missing_control_api(self):
        """render_once must handle ImportError of UMRSControlAPI gracefully."""
        from umrs.tui.dashboard import Dashboard

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120, record=True)

        mq = _make_mock_mq()
        dash = Dashboard(control_mq=mq, client_id="testuser", console=console)

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "umrs.client.control_api" or name.startswith("umrs.client.control_api."):
                raise ImportError("No module named 'umrs.client.control_api'")
            return real_import(name, *args, **kwargs)

        try:
            builtins.__import__ = fake_import  # type: ignore[attr-defined]
            dash.render_once()
        finally:
            builtins.__import__ = real_import

        plain = console.export_text(clear=False)
        assert "Cannot import UMRSControlAPI" in plain

    def test_render_once_handles_api_error(self):
        """render_once must show error message when API call fails."""
        from umrs.tui.dashboard import Dashboard

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120, record=True)

        mq = _make_mock_mq()
        dash = Dashboard(control_mq=mq, client_id="testuser", console=console)

        class _FakeAPI:
            """Fake API that succeeds in init but raises on list_jobs."""

            def __init__(self, *args, **kwargs):
                pass

            def list_jobs(self):
                raise RuntimeError("connection lost")

        # Patch the module that render_once imports from
        capi_mod = sys.modules.get("umrs.client.control_api")
        if capi_mod is None:
            capi_mod = types.ModuleType("umrs.client.control_api")
            sys.modules["umrs.client.control_api"] = capi_mod

        orig = getattr(capi_mod, "UMRSControlAPI", None)
        try:
            capi_mod.UMRSControlAPI = _FakeAPI
            dash.render_once()
        except RuntimeError:
            pass
        finally:
            if orig is not None:
                capi_mod.UMRSControlAPI = orig

        plain = console.export_text(clear=False)
        assert "Error fetching jobs" in plain
