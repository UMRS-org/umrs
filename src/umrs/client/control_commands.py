from __future__ import annotations

import argparse
import json
import re

from umrs.core.TaskCmdStatus import TaskCmd

PING_TIMEOUT_MS_DEFAULT = 2000
MODEL_TYPE_CHOICES = ("llm", "embedding", "vision")
MODEL_SCOPE_CHOICES = ("system", "project", "user")
WORKFLOW_SCOPE_CHOICES = ("system", "user")
JOB_TYPE_CHOICES = ("endpoint_provision", "index_build", "cleanup")
JOB_STATUS_FILTER_CHOICES = (
    "planned",
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
)
STATUS_CHOICES = tuple(TaskCmd.status_cmd())

MODEL_REGISTER_SCOPE_DEFAULT = "user"
MODEL_DOWNLOAD_SCOPE_DEFAULT = "user"
JOB_SPEC_DEFAULT = "{}"
JOB_PROJECT_DEFAULT = "default"
JOB_MAX_ATTEMPTS_DEFAULT = 3
ENDPOINT_IDLE_TTL_DEFAULT = "30m"
WORKFLOW_PARAMS_DEFAULT = "{}"

MAINTENANCE_ACTIONS = {
    "recompute": TaskCmd.RECOMPUTE,
    "update": TaskCmd.UPDATE,
    "check": TaskCmd.CHECK,
    "reload": TaskCmd.RELOAD,
}


def configure_ping_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout-ms", type=int, default=PING_TIMEOUT_MS_DEFAULT)


def configure_status_parser(
    parser: argparse.ArgumentParser,
    *,
    include_username: bool = False,
) -> None:
    parser.add_argument(
        "--all",
        action="store_true",
        dest="status_all",
        help="Alias for 'status all'",
    )
    parser.add_argument(
        "target",
        choices=STATUS_CHOICES,
        metavar="key",
        nargs="?",
        default="all",
        help=(
            f"Summary to fetch (options: {', '.join(STATUS_CHOICES)}; defaults to all)"
        ),
    )
    if include_username:
        parser.add_argument(
            "--username",
            help="User whose summary should be retrieved (defaults to config if omitted)",
        )


def configure_maintenance_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "maintenance_action",
        choices=sorted(MAINTENANCE_ACTIONS.keys()),
    )


def configure_model_register_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("name", help="Model name (e.g., e5-base-v2)")
    parser.add_argument("path", help="Filesystem path to model weights")
    parser.add_argument(
        "--type",
        choices=MODEL_TYPE_CHOICES,
        default="llm",
        dest="model_type",
        help="Model type",
    )
    parser.add_argument(
        "--scope",
        choices=MODEL_SCOPE_CHOICES,
        default=MODEL_REGISTER_SCOPE_DEFAULT,
        help="Model scope",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Project identifier (required for project scope)",
    )
    parser.add_argument("--description", default=None, help="Model description")
    parser.add_argument(
        "--huggingface-id",
        default=None,
        help="HuggingFace repo ID (for reference)",
    )


def configure_model_list_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        default=None,
        choices=MODEL_SCOPE_CHOICES,
        help="Filter by scope",
    )


def configure_model_download_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "repo_id",
        help="HuggingFace repository ID (e.g., intfloat/e5-base-v2)",
    )
    parser.add_argument(
        "--name", default=None, help="Local name (defaults to repo basename)"
    )
    parser.add_argument(
        "--scope",
        choices=MODEL_SCOPE_CHOICES,
        default=MODEL_DOWNLOAD_SCOPE_DEFAULT,
        help="Model scope",
    )
    parser.add_argument(
        "--project-id",
        default=None,
        help="Project identifier (required for project scope)",
    )


def configure_job_submit_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job_type", choices=JOB_TYPE_CHOICES, help="Job type")
    parser.add_argument(
        "--spec", default=JOB_SPEC_DEFAULT, help="Job spec as JSON string"
    )
    parser.add_argument("--project", default=JOB_PROJECT_DEFAULT, help="Project ID")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=JOB_MAX_ATTEMPTS_DEFAULT,
        help="Max retry attempts",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        default=False,
        help=(
            "Keep backend Slurm/K8s endpoint running after task completes "
            "(for endpoint_provision; default: auto-release)"
        ),
    )
    parser.add_argument(
        "--partition",
        default=None,
        help="Override Slurm partition for this job (must be in site-configured candidates)",
    )
    parser.add_argument(
        "--node",
        default=None,
        help='Slurm nodelist hint (e.g. "ai-h200-brc-00"); passed as --nodelist to sbatch',
    )
    parser.add_argument(
        "--reservation",
        default=None,
        help='Slurm reservation ID (e.g. "demo_day_2026"); passed as --reservation to sbatch',
    )


def configure_job_cancel_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job_id", help="Job ID to cancel")


def configure_job_status_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job_id", help="Job ID to inspect")


def configure_job_list_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=None, help="Filter by project ID")
    parser.add_argument(
        "--status",
        default=None,
        choices=JOB_STATUS_FILTER_CHOICES,
        help="Filter by job status",
    )
    parser.add_argument(
        "--type",
        dest="job_type",
        default=None,
        choices=JOB_TYPE_CHOICES,
        help="Filter by job type",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum jobs to return")
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset into the result set"
    )


def configure_endpoint_up_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Model name to provision")
    parser.add_argument(
        "--model-path", default=None, help="Explicit server-side model path"
    )
    parser.add_argument("--project", default=JOB_PROJECT_DEFAULT, help="Project ID")
    parser.add_argument(
        "--experience", default=None, help="Experience name used for endpoint defaults"
    )
    parser.add_argument(
        "--idle-ttl",
        default=None,
        help=(
            "Idle timeout before UMRS cleans up the endpoint "
            f"(default: {ENDPOINT_IDLE_TTL_DEFAULT}, overridable in experience JSON)"
        ),
    )
    parser.add_argument(
        "--tier",
        default="user",
        choices=("user", "project", "system"),
        help="Endpoint tier",
    )
    parser.add_argument("--gpu-count", type=int, default=1, help="GPU count")
    parser.add_argument(
        "--node-count",
        type=int,
        default=1,
        help="Number of Slurm nodes for multi-node pipeline parallelism (default: 1)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096, help="vLLM max model length"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=JOB_MAX_ATTEMPTS_DEFAULT,
        help="Max retry attempts",
    )
    parser.add_argument(
        "--spec",
        default=JOB_SPEC_DEFAULT,
        help="Additional endpoint spec as JSON string",
    )
    parser.add_argument(
        "--partition", default=None, help="Override Slurm partition for this endpoint"
    )
    parser.add_argument(
        "--node", default=None, help='Slurm nodelist hint (e.g. "ai-h200-brc-00")'
    )
    parser.add_argument(
        "--keep-endpoint",
        action="store_true",
        default=False,
        help="Keep the endpoint alive indefinitely (overrides idle-ttl, 999999s TTL)",
    )


def configure_endpoint_list_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", default=None, help="Filter by project ID")
    parser.add_argument(
        "--status",
        default=None,
        choices=JOB_STATUS_FILTER_CHOICES,
        help="Filter by durable job status",
    )
    parser.add_argument(
        "--limit", type=int, default=200, help="Maximum endpoints to return"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset into the result set"
    )


def configure_endpoint_status_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job_id", help="Endpoint provision job ID to inspect")


def configure_endpoint_stop_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("job_id", help="Endpoint provision job ID to stop")


def parse_duration_seconds(raw: object, *, field_name: str = "duration") -> int:
    if isinstance(raw, bool):
        raise ValueError(f"{field_name} must be a positive duration")
    if isinstance(raw, int):
        seconds = raw
    else:
        text = str(raw).strip().lower()
        if not text:
            raise ValueError(f"{field_name} must not be empty")
        if text.isdigit():
            seconds = int(text)
        else:
            match = re.fullmatch(
                r"(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)",
                text,
            )
            if not match:
                raise ValueError(
                    f"Invalid {field_name}: {raw!r}. Use seconds or a suffix like 30m, 1h, or 2d."
                )
            amount = int(match.group(1))
            unit = match.group(2)
            if unit.startswith("s"):
                multiplier = 1
            elif unit.startswith("m"):
                multiplier = 60
            elif unit.startswith("h"):
                multiplier = 3600
            else:
                multiplier = 86400
            seconds = amount * multiplier
    if seconds <= 0:
        raise ValueError(f"{field_name} must be positive")
    return seconds


def configure_workflow_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "workflow_name", help="Name of the registered workflow to execute"
    )
    parser.add_argument(
        "--bucket", default=None, help="Bucket alias to use as input data source"
    )
    parser.add_argument(
        "--params",
        default=WORKFLOW_PARAMS_DEFAULT,
        help="JSON string of additional parameters",
    )


def configure_workflow_list_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        default=None,
        choices=WORKFLOW_SCOPE_CHOICES,
        help="Filter by scope",
    )
    parser.add_argument("--user", default=None, help="Filter by owner user ID")
    parser.add_argument(
        "--name", default=None, help="Filter by workflow name (substring match)"
    )


def configure_workflow_upload_parser(parser: argparse.ArgumentParser) -> None:
    """Parser for ``workflow upload`` — upload a workflow directory."""
    parser.add_argument(
        "path", help="Local directory containing workflow.json and step scripts"
    )
    parser.add_argument(
        "--name", default=None, help="Workflow name (defaults to directory basename)"
    )
    parser.add_argument(
        "--scope",
        default="user",
        choices=WORKFLOW_SCOPE_CHOICES,
        help="Workflow visibility scope (default: user)",
    )


def configure_workflow_status_parser(parser: argparse.ArgumentParser) -> None:
    """Parser for ``workflow status`` — show execution status."""
    parser.add_argument("execution_id", help="Execution ID to inspect")


def configure_workflow_results_parser(parser: argparse.ArgumentParser) -> None:
    """Parser for ``workflow results`` — fetch results."""
    parser.add_argument("execution_id", help="Execution ID to fetch results for")


def configure_workflow_cancel_parser(parser: argparse.ArgumentParser) -> None:
    """Parser for ``workflow cancel`` — cancel execution."""
    parser.add_argument("execution_id", help="Execution ID to cancel")


def configure_workflow_errors_parser(parser: argparse.ArgumentParser) -> None:
    """Parser for ``workflow errors`` — show error logs for failed Attempts."""
    parser.add_argument("execution_id", help="Execution ID to fetch error logs for")
    parser.add_argument(
        "--tail",
        type=int,
        default=50,
        help="Maximum number of error log lines to show (default: 50)",
    )


def configure_workflow_logs_parser(parser: argparse.ArgumentParser) -> None:
    """Parser for ``workflow logs`` — stream execution logs for a workflow run."""
    parser.add_argument("execution_id", help="Execution ID to stream logs for")
    parser.add_argument(
        "--tail",
        type=int,
        default=200,
        help="Maximum number of log lines to show (default: 200)",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        default=False,
        help="Follow log output until workflow reaches a terminal state",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds for --follow (default: 2.0)",
    )


def configure_workflow_visualize_parser(parser: argparse.ArgumentParser) -> None:
    """Parser for ``workflow visualize`` — render the workflow DAG."""
    parser.add_argument("name", help="Registered workflow name to visualize")
    parser.add_argument(
        "--format",
        default="ascii",
        choices=["ascii", "dot"],
        help="Output format: ascii (default) or dot (Graphviz)",
    )


def configure_workflow_validate_sandbox_parser(parser: argparse.ArgumentParser) -> None:
    """Parser for ``workflow validate-sandbox`` — validate Dockerfile sandbox."""
    parser.add_argument(
        "path", help="Local directory containing Dockerfile + workflow.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate syntax without building the image",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        default=False,
        help="Attempt a real container build (podman/docker) to verify the Dockerfile",
    )


def build_control_command_parser(
    *,
    prog: str = "umrs-control",
    add_help: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, add_help=add_help)
    subparsers = parser.add_subparsers(dest="action", required=True)

    ping = subparsers.add_parser("ping", add_help=add_help)
    configure_ping_parser(ping)

    status = subparsers.add_parser("status", add_help=add_help)
    configure_status_parser(status)

    maintenance = subparsers.add_parser("maintenance", add_help=add_help)
    configure_maintenance_parser(maintenance)

    model = subparsers.add_parser("model", add_help=add_help)
    model_subparsers = model.add_subparsers(dest="model_action", required=True)
    model_list = model_subparsers.add_parser("list", add_help=add_help)
    configure_model_list_parser(model_list)
    model_register = model_subparsers.add_parser("register", add_help=add_help)
    configure_model_register_parser(model_register)
    model_download = model_subparsers.add_parser("download", add_help=add_help)
    configure_model_download_parser(model_download)

    job = subparsers.add_parser("job", add_help=add_help)
    job_subparsers = job.add_subparsers(dest="job_action", required=True)
    job_submit = job_subparsers.add_parser("submit", add_help=add_help)
    configure_job_submit_parser(job_submit)
    job_cancel = job_subparsers.add_parser("cancel", add_help=add_help)
    configure_job_cancel_parser(job_cancel)
    job_status = job_subparsers.add_parser("status", add_help=add_help)
    configure_job_status_parser(job_status)
    job_list = job_subparsers.add_parser("list", add_help=add_help)
    configure_job_list_parser(job_list)

    endpoint = subparsers.add_parser("endpoint", add_help=add_help)
    endpoint_subparsers = endpoint.add_subparsers(dest="endpoint_action", required=True)
    endpoint_up = endpoint_subparsers.add_parser("up", add_help=add_help)
    configure_endpoint_up_parser(endpoint_up)
    endpoint_list = endpoint_subparsers.add_parser("list", add_help=add_help)
    configure_endpoint_list_parser(endpoint_list)
    endpoint_status = endpoint_subparsers.add_parser("status", add_help=add_help)
    configure_endpoint_status_parser(endpoint_status)
    endpoint_stop = endpoint_subparsers.add_parser("stop", add_help=add_help)
    configure_endpoint_stop_parser(endpoint_stop)

    workflow = subparsers.add_parser("workflow", add_help=add_help)
    workflow_subparsers = workflow.add_subparsers(dest="workflow_action", required=True)

    # workflow upload
    workflow_upload = workflow_subparsers.add_parser("upload", add_help=add_help)
    configure_workflow_upload_parser(workflow_upload)

    # workflow list
    workflow_list = workflow_subparsers.add_parser("list", add_help=add_help)
    configure_workflow_list_parser(workflow_list)

    # workflow run
    workflow_run = workflow_subparsers.add_parser("run", add_help=add_help)
    configure_workflow_run_parser(workflow_run)

    # workflow status
    workflow_status = workflow_subparsers.add_parser("status", add_help=add_help)
    configure_workflow_status_parser(workflow_status)

    # workflow results
    workflow_results = workflow_subparsers.add_parser("results", add_help=add_help)
    configure_workflow_results_parser(workflow_results)

    # workflow cancel
    workflow_cancel = workflow_subparsers.add_parser("cancel", add_help=add_help)
    configure_workflow_cancel_parser(workflow_cancel)

    # workflow errors
    workflow_errors = workflow_subparsers.add_parser("errors", add_help=add_help)
    configure_workflow_errors_parser(workflow_errors)

    # workflow logs
    workflow_logs = workflow_subparsers.add_parser("logs", add_help=add_help)
    configure_workflow_logs_parser(workflow_logs)

    # workflow visualize
    workflow_visualize = workflow_subparsers.add_parser("visualize", add_help=add_help)
    configure_workflow_visualize_parser(workflow_visualize)

    # workflow validate-sandbox
    workflow_validate = workflow_subparsers.add_parser(
        "validate-sandbox", add_help=add_help
    )
    configure_workflow_validate_sandbox_parser(workflow_validate)

    return parser


def parse_json_object_arg(raw: str, *, field_name: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for {field_name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must decode to a JSON object")
    return parsed


def prepare_control_command_line(command_line: str) -> str:
    prepared = command_line
    for option_name in ("--params", "--spec"):
        prepared = _quote_braced_option(prepared, option_name)
    return prepared


def _quote_braced_option(command_line: str, option_name: str) -> str:
    start = 0
    while True:
        option_idx = command_line.find(option_name, start)
        if option_idx == -1:
            return command_line

        value_idx = option_idx + len(option_name)
        while value_idx < len(command_line) and command_line[value_idx].isspace():
            value_idx += 1
        if value_idx >= len(command_line):
            return command_line
        if command_line[value_idx] in ('"', "'"):
            start = value_idx + 1
            continue
        if command_line[value_idx] != "{":
            start = value_idx + 1
            continue

        depth = 0
        end_idx = value_idx
        while end_idx < len(command_line):
            char = command_line[end_idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx += 1
                    break
            end_idx += 1

        if depth != 0:
            return command_line

        json_blob = command_line[value_idx:end_idx]
        quoted = json.dumps(json_blob)
        command_line = command_line[:value_idx] + quoted + command_line[end_idx:]
        start = value_idx + len(quoted)


def configure_admin_token_parser(
    parser: argparse.ArgumentParser,
    *,
    add_help: bool = False,
) -> None:
    """Add ``admin token {generate,list,revoke}`` subparsers."""
    token_sub = parser.add_subparsers(dest="token_action", required=True)

    gen_p = token_sub.add_parser(
        "generate", add_help=add_help, help="Generate a new admin token"
    )
    gen_p.add_argument("name", help="Token name (for identification)")
    gen_p.add_argument("--description", default="", help="Optional description")

    token_sub.add_parser("list", add_help=add_help, help="List existing admin tokens")

    rev_p = token_sub.add_parser(
        "revoke", add_help=add_help, help="Revoke an admin token by name"
    )
    rev_p.add_argument("name", help="Token name to revoke")


def configure_admin_user_parser(
    parser: argparse.ArgumentParser,
    *,
    add_help: bool = False,
) -> None:
    """Add ``admin user {add,list,remove}`` subparsers."""
    user_sub = parser.add_subparsers(dest="user_action", required=True)

    add_p = user_sub.add_parser(
        "add", add_help=add_help, help="Enroll a new user on the server"
    )
    add_p.add_argument("config_path", help="Path to the user config JSON file")

    user_sub.add_parser(
        "list", add_help=add_help, help="List enrolled users on the server"
    )

    rm_p = user_sub.add_parser(
        "remove",
        add_help=add_help,
        help="Remove an enrolled user (not yet implemented)",
    )
    rm_p.add_argument("username", help="Username to remove")


__all__ = [
    "JOB_MAX_ATTEMPTS_DEFAULT",
    "JOB_PROJECT_DEFAULT",
    "JOB_SPEC_DEFAULT",
    "JOB_TYPE_CHOICES",
    "ENDPOINT_IDLE_TTL_DEFAULT",
    "MAINTENANCE_ACTIONS",
    "MODEL_DOWNLOAD_SCOPE_DEFAULT",
    "MODEL_REGISTER_SCOPE_DEFAULT",
    "MODEL_SCOPE_CHOICES",
    "MODEL_TYPE_CHOICES",
    "PING_TIMEOUT_MS_DEFAULT",
    "STATUS_CHOICES",
    "WORKFLOW_PARAMS_DEFAULT",
    "WORKFLOW_SCOPE_CHOICES",
    "build_control_command_parser",
    "configure_admin_token_parser",
    "configure_admin_user_parser",
    "configure_endpoint_list_parser",
    "configure_endpoint_status_parser",
    "configure_endpoint_stop_parser",
    "configure_endpoint_up_parser",
    "configure_job_cancel_parser",
    "configure_job_submit_parser",
    "configure_maintenance_parser",
    "configure_model_download_parser",
    "configure_model_list_parser",
    "configure_model_register_parser",
    "configure_ping_parser",
    "configure_status_parser",
    "configure_workflow_list_parser",
    "configure_workflow_run_parser",
    "configure_workflow_upload_parser",
    "configure_workflow_status_parser",
    "configure_workflow_results_parser",
    "configure_workflow_cancel_parser",
    "configure_workflow_errors_parser",
    "configure_workflow_logs_parser",
    "configure_workflow_visualize_parser",
    "configure_workflow_validate_sandbox_parser",
    "parse_duration_seconds",
    "parse_json_object_arg",
    "prepare_control_command_line",
]
