"""Handler for the UPLOAD_WORKFLOW command — DEPRECATED (ADR-027 P1).

The ZMQ upload path trusted only self-asserted ``client_id`` (no verified
identity at all) while still executing uploaded code server-side. All
uploads route through Gateway HTTP (``POST /gw/workflows/upload``) — the
single enforcement point with verified identity + workflow_author role
gating. The handler now fails fast with guidance in the task error.
"""

from __future__ import annotations

import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskStatus

from .base import CommandHandler, HandlerContext, StatusCallback

logger = logging.getLogger(__name__)


class UploadWorkflowHandler(CommandHandler):
    """Handle UPLOAD_WORKFLOW — DEPRECATED (ADR-027 P1).

    Uploads route exclusively through Gateway HTTP
    (``POST /gw/workflows/upload``) — the single enforcement point where
    verified identity + workflow_author role gating apply. This handler
    trusted only self-asserted ``client_id``; it now fails fast with
    guidance instead of accepting unauthenticated code uploads.
    """

    def execute(
        self,
        payload: Any,
        update_status: StatusCallback,
        ctx: HandlerContext,
        task_id: int,
    ) -> None:
        _client_id, _files, workflow_name = payload
        logger.warning(
            "UPLOAD_WORKFLOW rejected (deprecated, ADR-027) for '%s' — "
            "use Gateway HTTP POST /gw/workflows/upload (task_id=%s)",
            workflow_name,
            task_id,
        )
        update_status(TaskStatus.ERROR)
