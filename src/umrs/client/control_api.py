from __future__ import annotations

import json
import os
import shlex
from typing import Any, Dict, Optional

from umrs.core.TaskCmdStatus import TaskCmd

from umrs.client.control_commands import (
    MAINTENANCE_ACTIONS,
    build_control_command_parser,
    parse_json_object_arg,
    prepare_control_command_line,
)


class UMRSControlAPI:
    """Reusable control-plane client operations backed by MessageQueue."""

    def __init__(
        self,
        control_mq: Any,
        client_id: str,
        config_path: Optional[str] = None,
        admin_token: Optional[str] = None,
    ):
        self._mq = control_mq
        self._client_id = client_id
        self._config_path = config_path
        self._admin_token = admin_token

    @property
    def client_id(self) -> str:
        return self._client_id

    def ping(self, timeout_ms: int = 2000) -> Dict[str, Any]:
        ready = self._mq.is_server_ready(timeout_ms=timeout_ms)
        return {"status": "ok", "server_ready": ready}

    def request_status(self, target: str = "all") -> Dict[str, Any]:
        raw = self._mq.request_status(self._client_id, target)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Invalid status response")
        return parsed

    def list_models(self, scope: Optional[str] = None) -> Dict[str, Any]:
        payload = json.dumps({"scope": scope}) if scope else ""
        raw = self._mq.request_model_command(self._client_id, TaskCmd.MODEL_LIST, payload)
        if raw.startswith("UNAUTHORIZED:"):
            return {"status": "error", "message": "Authorization denied for MODEL_LIST", "code": "UNAUTHORIZED"}
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
        project_id: Optional[str] = None,
        description: Optional[str] = None,
        huggingface_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        payload = json.dumps(body)
        return self._request_json(TaskCmd.MODEL_REGISTER, payload)

    def download_model_server_side(
        self,
        *,
        repo_id: str,
        scope: str = "user",
        name: Optional[str] = None,
        project_id: Optional[str] = None,
        staging_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        body = {
            "repo_id": repo_id,
            "name": name or repo_id.split("/")[-1],
            "scope": scope,
            "project_id": project_id,
            "staging_dir": staging_dir,
        }
        if self._admin_token:
            body["admin_token"] = self._admin_token
        payload = json.dumps(body)
        return self._request_json(TaskCmd.MODEL_DOWNLOAD, payload)

    def submit_job(
        self,
        *,
        job_type: str,
        spec: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        max_attempts: int = 3,
        keep_alive: bool = False,
        partition: Optional[str] = None,
        node: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_spec = dict(spec or {})
        if partition:
            job_spec["partition_override"] = partition
        if node:
            job_spec["nodelist_hint"] = node
        payload = json.dumps(
            {
                "job_type": job_type,
                "spec": job_spec,
                "project_id": project,
                "max_attempts": max_attempts,
                "keep_alive": keep_alive,
            }
        )
        return self._request_json(TaskCmd.JOB_SUBMIT, payload)

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return self._request_json(TaskCmd.JOB_CANCEL, json.dumps({"job_id": job_id}))

    def list_jobs(
        self,
        *,
        job_id: Optional[str] = None,
        project: Optional[str] = None,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 200,
        offset: int = 0,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {
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

    def get_job(self, job_id: str) -> Dict[str, Any]:
        parsed = self.list_jobs(job_id=job_id, limit=1)
        if parsed.get("status") != "ok":
            return parsed
        jobs = parsed.get("jobs", [])
        if not jobs:
            return {"status": "error", "message": f"Job not found: {job_id}"}
        return {"status": "ok", "job": jobs[0]}

    def list_workflows(
        self,
        *,
        scope: Optional[str] = None,
        user: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        filter_payload: Dict[str, Any] = {}
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
        params: Optional[Dict[str, Any]] = None,
        bucket: Optional[str] = None,
    ) -> Dict[str, Any]:
        workflow_params = dict(params or {})
        if bucket:
            workflow_params["bucket"] = bucket
        payload = json.dumps({"workflow_name": workflow_name, "params": workflow_params})
        return self._request_json(TaskCmd.RUN_WORKFLOW, payload)

    def maintenance(self, action: str, config_path: Optional[str] = None) -> Dict[str, Any]:
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
        return {"status": "error", "action": action, "accepted": False, "message": "Command rejected by server"}

    def enroll_user(self, config_path: str) -> Dict[str, Any]:
        """Enroll a new user on the server (admin-only).

        Uploads the user config JSON via the data queue and sends the
        ``ENROLL_USER`` command.  The server will persist the config,
        update ``conf_umrs.json``, and auto-reload to provision the user.
        """
        if not os.path.isfile(config_path):
            return {"status": "error", "message": f"File not found: {config_path}"}

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg_json = json.load(fh)
            upload_user = cfg_json.get("username")
            if not upload_user:
                return {"status": "error", "message": "Config file missing 'username' field"}
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
            try:
                data_mq.close()
            except Exception:
                pass

        if ok:
            return {"status": "ok", "username": upload_user, "accepted": True}
        return {"status": "error", "username": upload_user, "accepted": False,
                "message": "Server rejected ENROLL_USER (check admin token)"}

    def execute_command(self, command_line: str) -> Dict[str, Any]:
        parser = build_control_command_parser()
        args = parser.parse_args(shlex.split(prepare_control_command_line(command_line)))

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
                spec = parse_json_object_arg(args.spec, field_name="spec") if args.spec else {}
                return self.submit_job(
                    job_type=args.job_type,
                    spec=spec,
                    project=args.project,
                    max_attempts=args.max_attempts,
                    keep_alive=args.keep_alive,
                    partition=args.partition,
                    node=args.node,
                )
        if args.action == "workflow":
            if args.workflow_action == "list":
                return self.list_workflows(scope=args.scope, user=args.user, name=args.name)
            if args.workflow_action == "run":
                params = parse_json_object_arg(args.params, field_name="params") if args.params else {}
                return self.run_workflow(workflow_name=args.workflow_name, params=params, bucket=args.bucket)

        raise ValueError(f"Unsupported command: {command_line}")

    def _request_json(self, task_cmd: str, payload: str) -> Dict[str, Any]:
        raw = self._mq.request_model_command(self._client_id, task_cmd, payload)
        if raw.startswith("UNAUTHORIZED:"):
            return {"status": "error", "message": f"Authorization denied for {task_cmd}", "code": "UNAUTHORIZED"}
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return {"status": "ok", "items": parsed}
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Invalid response for {task_cmd}")
        return parsed

    def _upload_reload_config(self, config_path: Optional[str]) -> None:
        if not config_path or not os.path.isfile(config_path):
            return

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                cfg_json = json.load(handle)
            upload_user = cfg_json.get("username") or self._client_id
        except Exception:
            upload_user = self._client_id

        data_mq = self._mq.get_data_queue(self._client_id)
        try:
            data_mq.send_file(config_path)
            self._mq.notify_server(upload_user, TaskCmd.FILE, 1)
        finally:
            try:
                data_mq.close()
            except Exception:
                pass