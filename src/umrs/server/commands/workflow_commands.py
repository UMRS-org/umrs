"""Handlers for LIST_WORKFLOWS, RUN_WORKFLOW and related workflow commands."""

from __future__ import annotations

import json
import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.server.workflow_registry_handler import handle_workflow_command

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


def _send_reply(
    ctx: HandlerContext, zmq_sender_id: Any, response_json: str, label: str
) -> None:
    """Send *response_json* back to the ZeroMQ caller. Log but never raise."""
    try:
        ctx.control_mq.send_string(response_json, zmq_sender_id)
    except RuntimeError as exc:
        logger.error("Error sending %s response: %s", label, exc)



def _scoped_list_payload(payload_str: str, client_id: str, is_admin: bool) -> str:
    """LIST_WORKFLOWS over ZMQ scoped like the HTTP route: a non-admin sees
    their own workflows plus those whose ADR-080 access list admits them.
    The payload's owner filter and share subject become the bound caller,
    and its project claim is dropped (ZMQ carries no gateway-verified
    project). Before 2026-10-04 an empty payload listed every user's
    workflows, source paths and access lists included."""
    if is_admin:
        return payload_str
    try:
        payload = json.loads(payload_str) if payload_str else {}
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload["user_id"] = client_id
    payload["include_shared_for"] = client_id
    payload["include_system"] = True
    payload.pop("verified_project", None)
    return json.dumps(payload)


class ListWorkflowsHandler(CommandHandler):
    """Handle LIST_WORKFLOWS — list registered workflows for a user."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.LIST_WORKFLOWS, payload)
        client_id, payload_str, zmq_sender_id = payload
        auth = ctx.control_mq.get_auth_context(client_id)
        response_json = handle_workflow_command(
            "LIST_WORKFLOWS",
            client_id,
            _scoped_list_payload(payload_str, client_id, getattr(auth, "is_admin", False) is True),
        )
        _send_reply(ctx, zmq_sender_id, response_json, "LIST_WORKFLOWS")
        update_status(TaskStatus.COMPLETED)


class RunWorkflowHandler(CommandHandler):
    """Handle RUN_WORKFLOW — trigger execution of a registered workflow."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.RUN_WORKFLOW, payload)
        client_id, payload_str, zmq_sender_id = payload
        job_backend = ctx.job_command_backend if ctx.workflow_scheduler_active else None
        # ADR-074 D-8: another owner's workflow only for an admin — the role
        # of the server's own auth session, never a payload claim.
        auth = ctx.control_mq.get_auth_context(client_id)
        response_json = handle_workflow_command(
            "RUN_WORKFLOW",
            client_id,
            payload_str,
            job_backend=job_backend,
            caller_is_admin=getattr(auth, "is_admin", False) is True,
        )
        _send_reply(ctx, zmq_sender_id, response_json, "RUN_WORKFLOW")
        update_status(TaskStatus.COMPLETED)


class WorkflowStatusHandler(CommandHandler):
    """Handle WORKFLOW_STATUS — show execution status for a workflow run."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.WORKFLOW_STATUS, payload)
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_workflow_command("WORKFLOW_STATUS", client_id, payload_str)
        _send_reply(ctx, zmq_sender_id, response_json, "WORKFLOW_STATUS")
        update_status(TaskStatus.COMPLETED)


class WorkflowExecutionsHandler(CommandHandler):
    """Handle WORKFLOW_EXECUTIONS — list workflow execution ids for the caller."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(
            ctx, TaskCmd.WORKFLOW_EXECUTIONS, payload
        )
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_workflow_command(
            "WORKFLOW_EXECUTIONS",
            client_id,
            payload_str,
        )
        _send_reply(ctx, zmq_sender_id, response_json, "WORKFLOW_EXECUTIONS")
        update_status(TaskStatus.COMPLETED)


class WorkflowResultsHandler(CommandHandler):
    """Handle WORKFLOW_RESULTS — fetch results for a completed workflow run."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.WORKFLOW_RESULTS, payload)
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_workflow_command("WORKFLOW_RESULTS", client_id, payload_str)
        _send_reply(ctx, zmq_sender_id, response_json, "WORKFLOW_RESULTS")
        update_status(TaskStatus.COMPLETED)


class WorkflowCancelHandler(CommandHandler):
    """Handle WORKFLOW_CANCEL — cancel a running workflow execution."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.WORKFLOW_CANCEL, payload)
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_workflow_command("WORKFLOW_CANCEL", client_id, payload_str)
        _send_reply(ctx, zmq_sender_id, response_json, "WORKFLOW_CANCEL")
        update_status(TaskStatus.COMPLETED)


class WorkflowErrorsHandler(CommandHandler):
    """Handle WORKFLOW_ERRORS — show error logs for a failed workflow execution."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.WORKFLOW_ERRORS, payload)
        client_id, payload_str, zmq_sender_id = payload
        response_json = handle_workflow_command("WORKFLOW_ERRORS", client_id, payload_str)
        _send_reply(ctx, zmq_sender_id, response_json, "WORKFLOW_ERRORS")
        update_status(TaskStatus.COMPLETED)
