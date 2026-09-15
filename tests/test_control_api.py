from __future__ import annotations

import json
from unittest.mock import MagicMock

from umrs.client.control_api import UMRSControlAPI


def test_ping_returns_ready_flag():
    mq = MagicMock()
    mq.is_server_ready.return_value = True

    api = UMRSControlAPI(mq, "tester")

    assert api.ping(timeout_ms=1234) == {"status": "ok", "server_ready": True}
    mq.is_server_ready.assert_called_once_with(timeout_ms=1234)


def test_cancel_job_uses_job_cancel_command():
    mq = MagicMock()
    mq.request_model_command.return_value = json.dumps({"status": "ok", "message": "cancelled"})

    api = UMRSControlAPI(mq, "tester")
    result = api.cancel_job("job-123")

    assert result["status"] == "ok"
    mq.request_model_command.assert_called_once()
    _, task_cmd, payload = mq.request_model_command.call_args.args
    assert task_cmd == "JOB_CANCEL"
    assert json.loads(payload) == {"job_id": "job-123"}


def test_list_jobs_uses_job_list_filters():
    mq = MagicMock()
    mq.request_job_list.return_value = {"status": "ok", "jobs": []}

    api = UMRSControlAPI(mq, "tester")
    result = api.list_jobs(
        project="demo", status="running", job_type="index_build", limit=5, offset=2
    )

    assert result == {"status": "ok", "jobs": []}
    mq.request_job_list.assert_called_once_with(
        "tester",
        {
            "limit": 5,
            "offset": 2,
            "project_id": "demo",
            "status": "running",
            "job_type": "index_build",
        },
    )


def test_get_job_returns_first_job_from_job_list():
    mq = MagicMock()
    mq.request_job_list.return_value = {
        "status": "ok",
        "jobs": [{"job_id": "job-123", "status": "running"}],
    }

    api = UMRSControlAPI(mq, "tester")
    result = api.get_job("job-123")

    assert result == {"status": "ok", "job": {"job_id": "job-123", "status": "running"}}
    mq.request_job_list.assert_called_once_with(
        "tester",
        {"limit": 1, "offset": 0, "job_id": "job-123"},
    )


def test_execute_command_supports_workflow_run():
    mq = MagicMock()
    mq.request_model_command.return_value = json.dumps(
        {"status": "ok", "message": "workflow started"}
    )

    api = UMRSControlAPI(mq, "tester")
    result = api.execute_command('workflow run demo-flow --params {"alpha": 1} --bucket outbox')

    assert result["status"] == "ok"
    _, task_cmd, payload = mq.request_model_command.call_args.args
    assert task_cmd == "RUN_WORKFLOW"
    parsed_payload = json.loads(payload)
    assert parsed_payload["workflow_name"] == "demo-flow"
    assert parsed_payload["params"] == {"alpha": 1, "bucket": "outbox"}


def test_list_workflows_uses_fast_fail_timeout():
    """Control-plane queries must fail fast, not block for the model-download
    window, when the server is unresponsive or too old to know the command."""
    mq = MagicMock()
    mq.request_model_command.return_value = json.dumps(
        {"status": "ok", "count": 0, "workflows": []}
    )

    api = UMRSControlAPI(mq, "tester")
    result = api.list_workflows()

    assert result["status"] == "ok"
    _, task_cmd, _payload = mq.request_model_command.call_args.args
    assert task_cmd == "LIST_WORKFLOWS"
    assert mq.request_model_command.call_args.kwargs.get("timeout_ms") == 60_000


def test_maintenance_reload_notifies_server():
    mq = MagicMock()
    mq.notify_server.return_value = True

    api = UMRSControlAPI(mq, "tester")
    result = api.maintenance("reload")

    assert result == {"status": "ok", "action": "reload", "accepted": True}
    mq.notify_server.assert_called_once_with("tester", "RELOAD", 0, admin_token=None)


def test_maintenance_variants_notify_server():
    api = UMRSControlAPI(MagicMock(), "tester")

    for action, expected_cmd in (
        ("recompute", "RECOMPUTE"),
        ("update", "UPDATE"),
        ("check", "CHECK"),
    ):
        api._mq.reset_mock()
        api._mq.notify_server.return_value = True

        result = api.maintenance(action)

        assert result == {"status": "ok", "action": action, "accepted": True}
        api._mq.notify_server.assert_called_once_with("tester", expected_cmd, 0, admin_token=None)


def test_register_model_includes_project_id_and_admin_token():
    mq = MagicMock()
    mq.request_model_command.return_value = json.dumps({"status": "ok"})

    api = UMRSControlAPI(mq, "tester", admin_token="secret-token")
    result = api.register_model(
        name="demo-model",
        path="/tmp/model",
        model_type="embedding",
        scope="project",
        project_id="proj-123",
    )

    assert result == {"status": "ok"}
    _, task_cmd, payload = mq.request_model_command.call_args.args
    assert task_cmd == "MODEL_REGISTER"
    parsed_payload = json.loads(payload)
    assert parsed_payload["scope"] == "project"
    assert parsed_payload["project_id"] == "proj-123"
    assert parsed_payload["admin_token"] == "secret-token"


def test_download_model_server_side_includes_project_id_and_staging_dir():
    mq = MagicMock()
    mq.request_model_command.return_value = json.dumps({"status": "ok"})

    api = UMRSControlAPI(mq, "tester", admin_token="secret-token")
    result = api.download_model_server_side(
        repo_id="org/demo-model",
        scope="project",
        name="demo-model",
        project_id="proj-123",
        staging_dir="/shared/projects/proj-123/models/experimental",
    )

    assert result == {"status": "ok"}
    _, task_cmd, payload = mq.request_model_command.call_args.args
    assert task_cmd == "MODEL_DOWNLOAD"
    parsed_payload = json.loads(payload)
    assert parsed_payload["project_id"] == "proj-123"
    assert parsed_payload["staging_dir"] == "/shared/projects/proj-123/models/experimental"
    assert parsed_payload["admin_token"] == "secret-token"


def test_execute_command_supports_job_submit_with_nested_spec_and_hints():
    mq = MagicMock()
    mq.request_model_command.return_value = json.dumps({"status": "ok", "message": "submitted"})

    api = UMRSControlAPI(mq, "tester")
    result = api.execute_command(
        'job submit endpoint_provision --spec {"nested": {"alpha": 1}} --partition gpu --node node-1'
    )

    assert result["status"] == "ok"
    _, task_cmd, payload = mq.request_model_command.call_args.args
    assert task_cmd == "JOB_SUBMIT"
    parsed_payload = json.loads(payload)
    assert parsed_payload["project_id"] == "default"
    assert parsed_payload["max_attempts"] == 3
    assert parsed_payload["spec"] == {
        "nested": {"alpha": 1},
        "partition_override": "gpu",
        "nodelist_hint": "node-1",
    }


def test_execute_command_supports_job_status():
    mq = MagicMock()
    mq.request_job_list.return_value = {
        "status": "ok",
        "jobs": [{"job_id": "job-123", "status": "completed"}],
    }

    api = UMRSControlAPI(mq, "tester")
    result = api.execute_command("job status job-123")

    assert result == {"status": "ok", "job": {"job_id": "job-123", "status": "completed"}}
    mq.request_job_list.assert_called_once_with(
        "tester",
        {"limit": 1, "offset": 0, "job_id": "job-123"},
    )


def test_execute_command_supports_workflow_list_filters():
    mq = MagicMock()
    mq.request_model_command.return_value = json.dumps({"status": "ok", "workflows": []})

    api = UMRSControlAPI(mq, "tester")
    result = api.execute_command("workflow list --scope user --user alice --name demo")

    assert result == {"status": "ok", "workflows": []}
    _, task_cmd, payload = mq.request_model_command.call_args.args
    assert task_cmd == "LIST_WORKFLOWS"
    assert json.loads(payload) == {"scope": "user", "user_id": "alice", "name": "demo"}


def test_execute_command_rejects_non_object_json_params():
    api = UMRSControlAPI(MagicMock(), "tester")

    try:
        api.execute_command("workflow run demo --params []")
    except ValueError as exc:
        assert "params must decode to a JSON object" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-object workflow params")


def test_execute_command_rejects_unsupported_commands():
    api = UMRSControlAPI(MagicMock(), "tester")

    try:
        api.execute_command("profile list")
    except SystemExit:
        pass
    else:
        raise AssertionError("Expected parser rejection for unsupported command")
