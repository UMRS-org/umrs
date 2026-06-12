from __future__ import annotations

from umrs.client.control_commands import build_control_command_parser, parse_json_object_arg


def test_shared_control_parser_uses_user_model_scope_defaults():
    parser = build_control_command_parser()

    register_args = parser.parse_args(["model", "register", "demo-model", "/tmp/model"])
    download_args = parser.parse_args(["model", "download", "demo/repo"])

    assert register_args.scope == "user"
    assert download_args.scope == "user"


def test_shared_control_parser_accepts_project_id_for_model_commands():
    parser = build_control_command_parser()

    register_args = parser.parse_args(
        ["model", "register", "demo-model", "/tmp/model", "--project-id", "proj-123"]
    )
    download_args = parser.parse_args(
        ["model", "download", "demo/repo", "--project-id", "proj-123"]
    )

    assert register_args.project_id == "proj-123"
    assert download_args.project_id == "proj-123"


def test_shared_control_parser_uses_job_and_workflow_defaults():
    parser = build_control_command_parser()

    job_args = parser.parse_args(["job", "submit", "cleanup"])
    workflow_args = parser.parse_args(["workflow", "run", "demo-flow"])

    assert job_args.spec == "{}"
    assert job_args.project == "default"
    assert job_args.max_attempts == 3
    assert workflow_args.params == "{}"


def test_parse_json_object_arg_rejects_non_object_payloads():
    try:
        parse_json_object_arg("[]", field_name="params")
    except ValueError as exc:
        assert "must decode to a JSON object" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for non-object JSON")


def test_parse_json_object_arg_rejects_invalid_json():
    try:
        parse_json_object_arg('{"broken":', field_name="spec")
    except ValueError as exc:
        assert "Invalid JSON for spec" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for malformed JSON")
