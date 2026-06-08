"""Handler for the UPLOAD_WORKFLOW command.

The client sends a zip archive containing a ``workflow.json`` and its step
scripts.  The handler:

  1. Extracts the archive to the server's workflow storage area via
     ``handle_upload_workflow``.
  2. Parses the embedded ``workflow.json`` and auto-registers the workflow
     in the server's workflow catalog via ``register_workflow``.  A failure
     at this step is non-fatal: the zip is kept and a warning is logged.

The task is marked COMPLETED only when the zip extraction succeeds;
missing files or extraction errors result in ERROR.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.server.file_handling import handle_upload_workflow as _handle_upload_workflow
from umrs.server.workflow_registry_handler import register_workflow

from .base import CommandHandler, HandlerContext, StatusCallback

logger = logging.getLogger(__name__)


class UploadWorkflowHandler(CommandHandler):
    """Handle UPLOAD_WORKFLOW — extract zip and register workflow in catalog."""

    def execute(
        self,
        payload: Any,
        update_status: StatusCallback,
        ctx: HandlerContext,
        task_id: int,
    ) -> None:
        client_id, files, workflow_name = payload

        if not files:
            logger.warning(
                "UPLOAD_WORKFLOW received with no files (task_id=%s)", task_id
            )
            update_status(TaskStatus.ERROR)
            return

        zip_path = files[0]
        ok, dest, err = _handle_upload_workflow(client_id, zip_path, workflow_name)

        if not ok:
            logger.error("UPLOAD_WORKFLOW failed for '%s': %s", workflow_name, err)
            update_status(TaskStatus.ERROR)
            return

        logger.info(
            "Workflow '%s' uploaded for user '%s' at %s", workflow_name, client_id, dest
        )

        # Auto-register in the workflow catalog.  Non-fatal: if the
        # workflow.json is missing or malformed the zip is already stored
        # and the user can register manually later.
        try:
            wf_json_path = os.path.join(dest, "workflow.json")
            with open(wf_json_path, encoding="utf-8") as fh:
                wf_data = json.load(fh)
            register_workflow(
                workflow_name=workflow_name,
                version=str(wf_data.get("version", "0.1")),
                user_id=client_id,
                source_path=dest,
                workflow_json=wf_data,
                scope="user",
                description=wf_data.get("description", ""),
            )
        except Exception as exc:  # noqa: BLE001 - non-fatal
            logger.warning(
                "Auto-registration of workflow '%s' failed: %s", workflow_name, exc
            )

        update_status(TaskStatus.COMPLETED)
