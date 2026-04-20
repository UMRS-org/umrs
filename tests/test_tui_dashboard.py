from __future__ import annotations

import json
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

SAMPLE_STATUS = {
    "status": "ok",
    "client": "testuser",
    "summaries": {
        "process_objects": {
            "generated_at": "2026-04-18T10:20:00",
            "last_updated_at": "2026-04-18T10:19:30",
            "data": [
                {
                    "thread_id": "snapshot",
                    "status": "Done",
                    "updated_at": "2026-04-18T10:19:30",
                    "work": [
                        {
                            "work_type": "process_objects",
                            "user": "testuser",
                            "nb_objects": 12,
                            "bucket": "user1-textfiles",
                            "data_base": "pgvector_testuser",
                            "table_name": "embeddings",
                        }
                    ],
                }
            ],
            "summary": "RAW_PRETTY_PRINT_BLOCK\nProcessed object queue with 3 items still pending and 12 completed.",
        },
        "PROCESS_OBJECTS": {
            "generated_at": "2026-04-18T10:18:00",
            "summary": "LEGACY_UPPERCASE_DUPLICATE",
        },
        "recompute": {
            "generated_at": "2026-04-18T10:22:00",
            "last_updated_at": "2026-04-18T10:21:55",
            "data": [],
            "summary": "Recompute finished successfully for all tracked embeddings.",
        },
    },
    "uncompleted_key_list": ["process_objects"],
}


def _make_mock_mq() -> MagicMock:
    mq = MagicMock()
    mq.request_job_list.return_value = {"status": "ok", "jobs": SAMPLE_JOBS, "total": len(SAMPLE_JOBS)}
    mq.request_status.return_value = json.dumps(SAMPLE_STATUS)
    mq.request_model_command.return_value = json.dumps({"status": "ok", "models": [], "workflows": []})
    mq.is_server_ready.return_value = True
    mq.notify_server.return_value = True
    return mq


class TestDashboardRendering:
    def test_render_once_with_jobs(self):
        from umrs.tui.dashboard import Dashboard

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120, record=True)
        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser", console=console)

        dash.render_once()

        plain = console.export_text(clear=False)
        assert "endpoint_provision" in plain
        assert "Jobs:" in plain

    def test_render_status_tab_uses_list_detail_shape(self):
        from umrs.tui.dashboard import Dashboard, TAB_STATUS

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120, record=True)
        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser", console=console, initial_tab=TAB_STATUS)

        dash.render_once()

        plain = console.export_text(clear=False)
        assert "Status Tasks" in plain
        assert "Details: process_objects" in plain
        assert "pending" in plain

    def test_render_status_tab_shows_per_process_last_update(self):
        from umrs.tui.dashboard import Dashboard, TAB_STATUS

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=180, record=True)
        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser", console=console, initial_tab=TAB_STATUS)

        dash.render_once()

        plain = console.export_text(clear=False)
        assert "Last Update" in plain
        assert "2026-04-18 10:19:30" in plain
        assert "snapshot" in plain
        assert "user1-textfiles" in plain

    def test_render_status_tab_deduplicates_case_variants_and_omits_raw_pretty_summary(self):
        from umrs.tui.dashboard import Dashboard, TAB_STATUS

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=180, record=True)
        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser", console=console, initial_tab=TAB_STATUS)

        dash.render_once()

        plain = console.export_text(clear=False)
        assert "PROCESS_OBJECTS" not in plain
        assert "RAW_PRETTY_PRINT_BLOCK" not in plain
        assert "1 process" in plain

    def test_status_footer_shows_actions_only(self):
        from umrs.tui.dashboard import Dashboard, TAB_STATUS

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser", initial_tab=TAB_STATUS)

        footer = dash._build_footer().plain
        assert ":" not in footer
        assert "cmd" not in footer
        assert "actions" in footer

    def test_status_navigation_moves_status_cursor(self):
        from umrs.tui.dashboard import Dashboard, TAB_STATUS

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser", initial_tab=TAB_STATUS)
        dash._fetch_status()

        assert dash._status_cursor == 0
        dash._on_key("j")
        assert dash._status_cursor == 1
        dash._on_key("k")
        assert dash._status_cursor == 0

    def test_colon_key_no_longer_opens_command_prompt(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser")
        footer_before = dash._build_footer().plain
        dash._on_key(":")

        assert dash._show_actions is False
        assert dash._form_active is False
        assert dash._search_active is False
        assert dash._build_footer().plain == footer_before

    def test_actions_hide_admin_commands_for_regular_user(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser")
        actions = dash._get_actions()
        action_names = {item["action"] for item in actions}

        assert "submit_job_form" in action_names
        assert "command_prompt" not in action_names
        assert "recompute" not in action_names
        assert "update" not in action_names
        assert "check" not in action_names

    def test_actions_include_admin_commands_for_admin_capabilities(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(
            control_mq=_make_mock_mq(),
            client_id="testuser",
            config_path="conf_umrs.json",
            capabilities={
                "is_admin": True,
                "has_user_config": False,
                "bucket_aliases": [],
                "allowed_actions": {
                    "recompute": True,
                    "update": True,
                    "check": True,
                    "reload": True,
                },
            },
        )

        actions = dash._get_actions()
        action_names = {item["action"] for item in actions}

        assert "recompute" in action_names
        assert "update" in action_names
        assert "check" in action_names
        assert "reload" in action_names

    def test_run_workflow_form_uses_bucket_choices_from_capabilities(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(
            control_mq=_make_mock_mq(),
            client_id="testuser",
            capabilities={
                "has_user_config": True,
                "bucket_aliases": ["user2-textfiles", "user1-textfiles"],
            },
        )
        dash._workflows = [{"name": "demo-workflow"}]

        dash._open_run_workflow_form()

        bucket_field = next(field for field in dash._form_fields if field["name"] == "bucket")
        assert bucket_field["choices"] == ["", "user1-textfiles", "user2-textfiles"]

    def test_submit_job_form_executes_api(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser")
        dash._api = MagicMock()
        dash._api.submit_job.return_value = {"status": "ok", "message": "submitted"}

        dash._open_submit_job_form()
        dash._form_fields[0]["value"] = "cleanup"
        dash._form_fields[1]["value"] = '{"dry_run": true}'
        dash._form_fields[2]["value"] = "demo"
        dash._form_fields[3]["value"] = "2"
        dash._form_fields[4]["value"] = "true"
        dash._form_fields[5]["value"] = "debug"
        dash._form_fields[6]["value"] = "node-a"
        dash._form_cursor = len(dash._form_fields) - 1

        dash._on_key("\r")

        dash._api.submit_job.assert_called_once_with(
            job_type="cleanup",
            spec={"dry_run": True},
            project="demo",
            max_attempts=2,
            keep_alive=True,
            partition="debug",
            node="node-a",
        )
        assert dash._form_active is False

    def test_form_choice_cycle_updates_selected_field(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser")
        dash._open_submit_job_form()

        assert dash._form_fields[0]["value"] == "endpoint_provision"
        dash._on_key("l")
        assert dash._form_fields[0]["value"] == "index_build"
        dash._on_key("h")
        assert dash._form_fields[0]["value"] == "endpoint_provision"

    def test_submit_job_form_rejects_invalid_choice(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser")
        dash._api = MagicMock()

        dash._open_submit_job_form()
        dash._form_fields[0]["value"] = "not-a-job-type"
        dash._form_cursor = len(dash._form_fields) - 1

        dash._on_key("\r")

        dash._api.submit_job.assert_not_called()
        assert dash._form_active is True
        assert "must be one of" in (dash._flash_message or "")

    def test_model_filter_form_updates_dashboard_filter(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser")

        dash._open_model_filter_form()
        dash._form_fields[0]["value"] = "project"
        dash._form_cursor = len(dash._form_fields) - 1

        dash._on_key("\r")

        assert dash._model_scope_filter == "project"
        assert dash._form_active is False
        assert dash._force_refresh.is_set()

    def test_workflow_filter_form_all_clears_scope(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser")
        dash._workflow_scope_filter = "user"

        dash._open_workflow_filter_form()
        dash._form_fields[0]["value"] = "all"
        dash._form_cursor = len(dash._form_fields) - 1

        dash._on_key("\r")

        assert dash._workflow_scope_filter is None
        assert dash._form_active is False

    def test_render_form_overlay(self):
        from umrs.tui.dashboard import Dashboard

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120, record=True)
        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser", console=console)
        dash._open_model_download_form()

        dash._console.print(dash._build_display())

        plain = console.export_text(clear=False)
        assert "Download Model" in plain
        assert "Repo ID" in plain

    def test_jobs_navigation_uses_jobs_cursor(self):
        from umrs.tui.dashboard import Dashboard

        dash = Dashboard(control_mq=_make_mock_mq(), client_id="testuser")
        dash._fetch_jobs()

        assert dash._jobs_cursor == 0
        dash._on_key("j")
        assert dash._jobs_cursor == 1
        dash._on_key("k")
        assert dash._jobs_cursor == 0
