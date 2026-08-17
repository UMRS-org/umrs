"""Handler for the deprecated ZMQ UPLOAD_WORKFLOW command.

Executable workflow uploads must go through Gateway HTTP so UMRS can
enforce HMAC-covered workflow_author/admin authorization. This handler
rejects the command for every caller.
"""

from __future__ import annotations

import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskStatus

from .base import CommandHandler, HandlerContext, StatusCallback

logger = logging.getLogger(__name__)

ZMQ_UPLOAD_DEPRECATED = (
    "UPLOAD_WORKFLOW via ZMQ is deprecated; upload through Gateway HTTP "
    "(/gw/workflows/upload) so UMRS can authorize workflow_author/admin"
)


class UploadWorkflowHandler(CommandHandler):
    """Reject ZMQ workflow uploads; Gateway HTTP is the only upload path."""

    def execute(
        self,
        payload: Any,
        update_status: StatusCallback,
        ctx: HandlerContext,
        task_id: int,
    ) -> None:
        logger.warning("%s (task_id=%s)", ZMQ_UPLOAD_DEPRECATED, task_id)
        update_status(TaskStatus.ERROR)
