from __future__ import annotations

import contextlib
import json
import os
import shlex
from typing import Any

from umrs.client.control_commands import (
    MAINTENANCE_ACTIONS,
    build_control_command_parser,
    parse_duration_seconds,
    parse_json_object_arg,
    prepare_control_command_line,
)
from umrs.core.TaskCmdStatus import TaskCmd
from umrs.core.zmq_correlation import dumps_with_correlation_id


class UMRSControlAPI:
    """Reusable control-plane client operations backed by MessageQueue."""

    def __init__(
        self,
        control_mq: Any,
        client_id: str,
        config_path: str | None = None,
        admin_token: str | None = None,
    ):
        self._mq = control_mq
        self._client_id = client_id
        self._config_path = config_path
        self._admin_token = admin_token

    @property
    def client_id(self) -> str:
        return self._client_id

    def ping(self, timeout_ms: int = 2000) -> dict[str, Any]:
        ready = self._mq.is_server_ready(timeout_ms=timeout_ms)
        return {"status": "ok", "server_ready": ready}

    def request_status(self, target: str = "all") -> dict[str, Any]:
        raw = self._mq.request_status(self._client_id, target)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Invalid status response")
        return parsed

    def request_status_all(self) -> dict[str, Any]:
        raw = self._mq.request_status_all()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Invalid status-all response")
        return parsed

    def list_models(self, scope: str | None = None) -> dict[str, Any]:
        payload = dumps_with_correlation_id({"scope": scope}) if scope else ""
        raw = self._mq.request_model_command(
            self._client_id, TaskCmd.MODEL_LIST, payload
        )
        if raw.startswith("UNAUTHORIZED:"):
            return {
                "status": "error",
                "message": "Authorization denied for MODEL_LIST",
                "code": "UNAUTHORIZED",
            }
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return {"status": "ok", "models": parsed}
        if isinstance(parsed, dict):
            return parsed
        raise RuntimeError("Invalid model list response")

    def register_model(
        self,
        *,
        name: str,
        path: str,
        model_type: str,
        scope: str = "user",
        project_id: str | None = None,
        description: str | None = None,
        huggingface_id: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "name": name,
            "path": os.path.abspath(path),
            "model_type": model_type,
            "scope": scope,
            "project_id": project_id,
            "description": description,
            "huggingface_id": huggingface_id,
        }
        if self._admin_token:
            body["admin_token"] = self._admin_token
        payload = dumps_with_correlation_id(body)

    def download_model_server_side(
        self,
        *,
        repo_id: str,
        scope: str = "user",
        name: str | None = None,
        project_id: str | None = None,
        staging_dir: str | None = None,
    ) -> dict[str, Any]:
        body = {
            "repo_id": repo_id,
            "name": name or repo_id.split("/")[-1],
            "scope": scope,
            "project_id": project_id,
            "staging_dir": staging_dir,
        }
        if self._admin_token:
            body["admin_token"] = self._admin_token
        payload = dumps_with_correlation_id(body)
        return self._request_json(TaskCmd.MODEL_DOWNLOAD, payload)

    def submit_job(
        self,
        *,
        job_type: str,
        spec: dict[str, Any] | None = None,
        project: str | None = None,
        max_attempts: int = 3,
        keep_alive: bool = False,
        partition: str | None = None,
        node: str | None = None,
    ) -> dict[str, Any]:
        job_spec = dict(spec or {})
        if partition:
            job_spec["partition_override"] = partition
        if node:
            job_spec["nodelist_hint"] = node
        payload = dumps_with_correlation_id(
            {
                "job_type": job_type,
                "spec": job_spec,
                "project_id": project,
                "max_attempts": max_attempts,
                "keep_alive": keep_alive,
            }
        )
        return self._request_json(TaskCmd.JOB_SUBMIT, payload)

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        return self._request_json(TaskCmd.JOB_CANCEL, dumps_with_correlation_id({"job_id": job_id}))

    def purge_buckets(self, bucket_names: list[str]) -> dict[str, Any]:
        """Request server-side deletion of user-owned S3 buckets."""
        return self._request_json(
            TaskCmd.PURGE_BUCKETS,
            dumps_with_correlation_id({"buckets": bucket_names}),
        )

    def list_jobs(
        self,
        *,
        job_id: str | None = None,
        project: str | None = None,
        status: str | None = None,
        job_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        filters: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if job_id:
            filters["job_id"] = job_id
        if project:
            filters["project_id"] = project
        if status:
            filters["status"] = status
        if job_type:
            filters["job_type"] = job_type
        return self._mq.request_job_list(self._client_id, filters)

    def get_job(self, job_id: str) -> dict[str, Any]:
        parsed = self.list_jobs(job_id=job_id, limit=1)
        if parsed.get("status") != "ok":
            return parsed
        jobs = parsed.get("jobs", [])
        if not jobs:
            return {"status": "error", "message": f"Job not found: {job_id}"}
        return {"status": "ok", "job": jobs[0]}

    def endpoint_up(
        self,
        *,
        model: str,
        model_path: str | None = None,
        project: str | None = None,
        idle_ttl_sec: int = 1800,
        tier: str = "user",
        gpu_count: int = 1,
        node_count: int = 1,
        max_model_len: int = 4096,
        max_attempts: int = 3,
        spec: dict[str, Any] | None = None,
        partition: str | None = None,
        node: str | None = None,
    ) -> dict[str, Any]:
        endpoint_spec = dict(spec or {})
        endpoint_spec.update(
            {
                "model": model,
                "ttl_sec": idle_ttl_sec,
                "idle_ttl_sec": idle_ttl_sec,
                "tier": tier,
                "gpu_count": gpu_count,
                "node_count": node_count,
                "max_model_len": max_model_len,
            }
        )
        if model_path:
            endpoint_spec["model_path"] = model_path

        parsed = self.submit_job(
            job_type="endpoint_provision",
            spec=endpoint_spec,
            project=project,
            max_attempts=max_attempts,
            keep_alive=True,
            partition=partition,
            node=node,
        )
        if parsed.get("status") == "ok":
            parsed["endpoint"] = {
                "job_id": (parsed.get("job") or {}).get("job_id"),
                "model": model,
                "project_id": project,
                "idle_ttl_sec": idle_ttl_sec,
            }
        return parsed

    def list_endpoints(
        self,
        *,
        project: str | None = None,
        status: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        parsed = self.list_jobs(
            project=project,
            status=status,
            job_type="endpoint_provision",
            limit=limit,
            offset=offset,
        )
        if parsed.get("status") == "ok":
            parsed["endpoints"] = parsed.get("jobs", [])
        return parsed

    def get_endpoint(self, job_id: str) -> dict[str, Any]:
        parsed = self.get_job(job_id)
        if parsed.get("status") == "ok":
            parsed["endpoint"] = parsed.get("job")
        return parsed

    def stop_endpoint(self, job_id: str) -> dict[str, Any]:
        return self.cancel_job(job_id)

    def list_workflows(
        self,
        *,
        scope: str | None = None,
        user: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        filter_payload: dict[str, Any] = {}
        if scope:
            filter_payload["scope"] = scope
        if user:
            filter_payload["user_id"] = user
        if name:
            filter_payload["name"] = name
        payload = json.dumps(filter_payload) if filter_payload else ""
        return self._request_json(TaskCmd.LIST_WORKFLOWS, payload)

    def run_workflow(
        self,
        *,
        workflow_name: str,
        params: dict[str, Any] | None = None,
        bucket: str | None = None,
    ) -> dict[str, Any]:
        workflow_params = dict(params or {})
        if bucket:
            workflow_params["bucket"] = bucket
        payload = json.dumps(
            {"workflow_name": workflow_name, "params": workflow_params}
        )
        return self._request_json(TaskCmd.RUN_WORKFLOW, payload)

    def workflow_status(self, execution_id: str) -> dict[str, Any]:
        """Show execution status for a workflow run."""
        payload = json.dumps({"execution_id": execution_id})
        return self._request_json(TaskCmd.WORKFLOW_STATUS, payload)

    def workflow_results(self, execution_id: str) -> dict[str, Any]:
        """Fetch results for a completed workflow run."""
        payload = json.dumps({"execution_id": execution_id})
        return self._request_json(TaskCmd.WORKFLOW_RESULTS, payload)

    def workflow_cancel(self, execution_id: str) -> dict[str, Any]:
        """Cancel a running workflow execution."""
        payload = json.dumps({"execution_id": execution_id})
        return self._request_json(TaskCmd.WORKFLOW_CANCEL, payload)

    def workflow_errors(
        self, execution_id: str, *, tail: int = 50
    ) -> dict[str, Any]:
        """Show error logs for a failed workflow execution (Attempts)."""
        payload = json.dumps(
            {"execution_id": execution_id, "tail": tail}
        )
        return self._request_json(TaskCmd.WORKFLOW_ERRORS, payload)

    def workflow_logs(
        self, execution_id: str, *, tail: int = 200
    ) -> dict[str, Any]:
        """Stream execution logs for a workflow run."""
        payload = json.dumps(
            {"execution_id": execution_id, "tail": tail}
        )
        return self._request_json(TaskCmd.WORKFLOW_LOGS, payload)

    def maintenance(
        self, action: str, config_path: str | None = None
    ) -> dict[str, Any]:
        action = action.lower()
        if action not in MAINTENANCE_ACTIONS:
            raise ValueError(f"Unsupported maintenance action: {action}")

        if action == "reload":
            self._upload_reload_config(config_path or self._config_path)

        ok = self._mq.notify_server(
            self._client_id,
            MAINTENANCE_ACTIONS[action],
            0,
            admin_token=self._admin_token,
        )
        if ok:
            return {"status": "ok", "action": action, "accepted": True}
        return {
            "status": "error",
            "action": action,
            "accepted": False,
            "message": "Authorization denied (admin required)",
        }

    def enroll_user(self, config_path: str) -> dict[str, Any]:
        """Enroll a new user on the server (admin-only).

        Uploads the user config JSON via the data queue and sends the
        ``ENROLL_USER`` command.  The server will persist the config,
        update ``conf_umrs.json``, and auto-reload to provision the user.
        """
        if not os.path.isfile(config_path):
            return {"status": "error", "message": f"File not found: {config_path}"}

        try:
            with open(config_path, encoding="utf-8") as fh:
                cfg_json = json.load(fh)
            upload_user = cfg_json.get("username")
            if not upload_user:
                return {
                    "status": "error",
                    "message": "Config file missing 'username' field",
                }
        except Exception as e:
            return {"status": "error", "message": f"Cannot read config file: {e}"}

        data_mq = self._mq.get_data_queue(self._client_id)
        try:
            data_mq.send_file(config_path)
            ok = self._mq.notify_server(
                self._client_id,
                TaskCmd.ENROLL_USER,
                1,
                admin_token=self._admin_token,
            )
        finally:
            with contextlib.suppress(Exception):
                data_mq.close()

        if ok:
            return {"status": "ok", "username": upload_user, "accepted": True}
        return {
            "status": "error",
            "username": upload_user,
            "accepted": False,
            "message": "Server rejected ENROLL_USER (check admin token)",
        }

    def execute_command(self, command_line: str) -> dict[str, Any]:
        parser = build_control_command_parser()
        args = parser.parse_args(
            shlex.split(prepare_control_command_line(command_line))
        )

        if args.action == "ping":
            return self.ping(timeout_ms=args.timeout_ms)
        if args.action == "status":
            return self.request_status(args.target)
        if args.action == "maintenance":
            return self.maintenance(args.maintenance_action)
        if args.action == "model":
            if args.model_action == "list":
                return self.list_models(scope=args.scope)
            if args.model_action == "register":
                return self.register_model(
                    name=args.name,
                    path=args.path,
                    model_type=args.model_type,
                    scope=args.scope,
                    project_id=getattr(args, "project_id", None),
                    description=args.description,
                    huggingface_id=args.huggingface_id,
                )
            if args.model_action == "download":
                return self.download_model_server_side(
                    repo_id=args.repo_id,
                    scope=args.scope,
                    name=args.name,
                    project_id=getattr(args, "project_id", None),
                    staging_dir=getattr(args, "staging_dir", None),
                )
        if args.action == "job":
            if args.job_action == "cancel":
                return self.cancel_job(args.job_id)
            if args.job_action == "status":
                return self.get_job(args.job_id)
            if args.job_action == "list":
                return self.list_jobs(
                    project=getattr(args, "project", None),
                    status=getattr(args, "status", None),
                    job_type=getattr(args, "job_type", None),
                    limit=getattr(args, "limit", 200),
                    offset=getattr(args, "offset", 0),
                )
            if args.job_action == "submit":
                spec = (
                    parse_json_object_arg(args.spec, field_name="spec")
                    if args.spec
                    else {}
                )
                return self.submit_job(
                    job_type=args.job_type,
                    spec=spec,
                    project=args.project,
                    max_attempts=args.max_attempts,
                    keep_alive=args.keep_alive,
                    partition=args.partition,
                    node=args.node,
                )
        if args.action == "endpoint":
            if args.endpoint_action == "up":
                spec = (
                    parse_json_object_arg(args.spec, field_name="spec")
                    if args.spec
                    else {}
                )
                idle_ttl_sec = parse_duration_seconds(
                    args.idle_ttl or "30m",
                    field_name="idle_ttl",
                )
                return self.endpoint_up(
                    model=args.model,
                    model_path=args.model_path,
                    project=args.project,
                    idle_ttl_sec=idle_ttl_sec,
                    tier=args.tier,
                    gpu_count=args.gpu_count,
                    node_count=getattr(args, "node_count", 1),
                    max_model_len=args.max_model_len,
                    max_attempts=args.max_attempts,
                    spec=spec,
                    partition=args.partition,
                    node=args.node,
                )
            if args.endpoint_action == "list":
                return self.list_endpoints(
                    project=getattr(args, "project", None),
                    status=getattr(args, "status", None),
                    limit=getattr(args, "limit", 200),
                    offset=getattr(args, "offset", 0),
                )
            if args.endpoint_action == "status":
                return self.get_endpoint(args.job_id)
            if args.endpoint_action == "stop":
                return self.stop_endpoint(args.job_id)
        if args.action == "workflow":
            if args.workflow_action == "list":
                return self.list_workflows(
                    scope=args.scope, user=args.user, name=args.name
                )
            if args.workflow_action == "run":
                params = (
                    parse_json_object_arg(args.params, field_name="params")
                    if args.params
                    else {}
                )
                return self.run_workflow(
                    workflow_name=args.workflow_name, params=params, bucket=args.bucket
                )

        raise ValueError(f"Unsupported command: {command_line}")

    def _request_json(self, task_cmd: str, payload: str) -> dict[str, Any]:
        # Control-plane queries (workflow list/status/results/...) are quick
        # registry lookups.  Fail fast instead of blocking for the full
        # model-download window when the server is unresponsive or too old to
        # know the command.
        raw = self._mq.request_model_command(
            self._client_id, task_cmd, payload, timeout_ms=60_000
        )
        if raw.startswith("UNAUTHORIZED:"):
            return {
                "status": "error",
                "message": f"Authorization denied for {task_cmd}",
                "code": "UNAUTHORIZED",
            }
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return {"status": "ok", "items": parsed}
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Invalid response for {task_cmd}")
        return parsed

    def _upload_reload_config(self, config_path: str | None) -> None:
        if not config_path or not os.path.isfile(config_path):
            return

        try:
            with open(config_path, encoding="utf-8") as handle:
                cfg_json = json.load(handle)
            upload_user = cfg_json.get("username") or self._client_id
        except Exception:
            upload_user = self._client_id

        data_mq = self._mq.get_data_queue(self._client_id)
        try:
            data_mq.send_file(config_path)
            self._mq.notify_server(upload_user, TaskCmd.FILE, 1)
        finally:
            with contextlib.suppress(Exception):
                data_mq.close()
