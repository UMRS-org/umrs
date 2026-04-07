"""
Tests for the TUI dashboard — renders from mock data without a live terminal.

Uses Console(file=StringIO()) so no actual TTY is needed.
"""

from __future__ import annotations
from io import StringIO
from unittest.mock import MagicMock

import pytest

from rich.console import Console


# ---------------------------------------------------------------------------
# Mock MessageQueue that returns canned job data
# ---------------------------------------------------------------------------
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
        "attempts": [
            {
                "attempt_id": "att-111222333444",
                "attempt_no": 1,
                "status": "running",
                "started_at": "2026-04-05T10:00:00",
                "ended_at": None,
                "exit_reason": None,
            }
        ],
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
        "attempts": [
            {
                "attempt_id": "att-aaa000bbb111",
                "attempt_no": 1,
                "status": "succeeded",
                "started_at": "2026-04-04T12:00:00",
                "ended_at": "2026-04-04T12:05:00",
                "exit_reason": None,
            }
        ],
        "created_at": "2026-04-04T11:58:00",
        "updated_at": "2026-04-04T12:05:00",
    },
    {
        "job_id": "job-000111222333",
        "project_id": "myproj",
        "user_id": "admin",
        "job_type": "cleanup",
        "spec": {},
        "status": "failed",
        "priority": 0,
        "desired_max_attempts": 1,
        "restart_policy": "never",
        "keep_alive": False,
        "attempts": [
            {
                "attempt_id": "att-xxx999yyy888",
                "attempt_no": 1,
                "status": "failed",
                "started_at": "2026-04-05T08:00:00",
                "ended_at": "2026-04-05T08:01:00",
                "exit_reason": "disk full",
            }
        ],
        "created_at": "2026-04-05T07:59:00",
        "updated_at": "2026-04-05T08:01:00",
    },
]


def _make_mock_mq(jobs=None):
    """Create a mock control_mq that returns canned job list."""
    mq = MagicMock()
    mq.request_job_list.return_value = {
        "status": "ok",
        "jobs": jobs if jobs is not None else SAMPLE_JOBS,
        "total": len(jobs if jobs is not None else SAMPLE_JOBS),
    }
    return mq


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDashboardRendering:
    """Verify the dashboard renders to a string buffer without crashing."""

    def test_render_once_with_jobs(self):
        from umrs.tui.dashboard import Dashboard

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        mq = _make_mock_mq()

        dash = Dashboard(
            control_mq=mq,
            client_id="testuser",
            console=console,
        )
        dash.render_once()

        output = buf.getvalue()
        assert "UMRS Jobs Monitor" in output
        assert "endpoint_provision" in output
        assert "running" in output

    def test_render_once_empty(self):
        from umrs.tui.dashboard import Dashboard

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120, record=True)
        mq = _make_mock_mq(jobs=[])

        dash = Dashboard(control_mq=mq, client_id="testuser", console=console)
        dash.render_once()

        plain = console.export_text(clear=False)
        assert "UMRS Jobs Monitor" in plain
        assert "jobs=0" in plain

    def test_render_with_error(self):
        from umrs.tui.dashboard import Dashboard

        mq = MagicMock()
        mq.request_job_list.side_effect = RuntimeError("connection lost")

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)

        dash = Dashboard(control_mq=mq, client_id="testuser", console=console)
        dash.render_once()

        output = buf.getvalue()
        assert "connection lost" in output

    def test_detail_view_toggle(self):
        from umrs.tui.dashboard import Dashboard

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        mq = _make_mock_mq()

        dash = Dashboard(control_mq=mq, client_id="testuser", console=console)
        dash._fetch_jobs()

        # Simulate Enter on first job
        dash._on_key("\r")
        assert dash._detail_job_id == SAMPLE_JOBS[0]["job_id"]

        # Toggle off
        dash._on_key("\r")
        assert dash._detail_job_id is None

    def test_filter_cycle(self):
        from umrs.tui.dashboard import Dashboard, FILTER_CYCLE

        mq = _make_mock_mq()
        dash = Dashboard(control_mq=mq, client_id="testuser")

        assert dash._status_filter is None
        dash._on_key("f")
        assert dash._status_filter == "running"
        dash._on_key("f")
        assert dash._status_filter == "failed"

    def test_cursor_navigation(self):
        from umrs.tui.dashboard import Dashboard

        mq = _make_mock_mq()
        dash = Dashboard(control_mq=mq, client_id="testuser")
        dash._fetch_jobs()

        assert dash._cursor == 0
        dash._on_key("j")
        assert dash._cursor == 1
        dash._on_key("j")
        assert dash._cursor == 2
        dash._on_key("j")  # should clamp at max
        assert dash._cursor == 2
        dash._on_key("k")
        assert dash._cursor == 1
        dash._on_key("k")
        assert dash._cursor == 0
        dash._on_key("k")  # should clamp at 0
        assert dash._cursor == 0

    def test_compact_toggle(self):
        from umrs.tui.dashboard import Dashboard

        mq = _make_mock_mq()
        dash = Dashboard(control_mq=mq, client_id="testuser")

        assert dash._compact is True
        dash._on_key("/")
        assert dash._compact is False
        dash._on_key("/")
        assert dash._compact is True

    def test_quit_key(self):
        from umrs.tui.dashboard import Dashboard

        mq = _make_mock_mq()
        dash = Dashboard(control_mq=mq, client_id="testuser")

        dash._on_key("q")
        assert dash._quit.is_set()


    def test_fetch_jobs_passes_filters_to_message_queue(self):
        from umrs.tui.dashboard import Dashboard

        mq = _make_mock_mq()
        dash = Dashboard(
            control_mq=mq,
            client_id="testuser",
            status_filter="running",
            project_filter="default",
        )

        dash._fetch_jobs()

        mq.request_job_list.assert_called_once_with(
            "testuser",
            {"status": "running", "project_id": "default"},
        )
