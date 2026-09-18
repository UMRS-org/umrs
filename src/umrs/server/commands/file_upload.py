"""Handler for the FILE command.

The client sends one or more config files (e.g. user*.json, s3cfg) that
the server should persist on behalf of the user.  Each file is validated
and saved via ``_persist_uploaded_user_config``; unrecognised files are
left in place with a log note rather than raising an error.

An empty payload is accepted (logged as a warning) and still marked
COMPLETED — matching the original behaviour where the status update was
unconditional and outside the ``if payload`` branch.
"""

from __future__ import annotations

import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.server.file_handling import (
    persist_uploaded_user_config as _persist_uploaded_user_config,
)

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


class FileHandler(CommandHandler):
    """Handle FILE — persist uploaded user config files on the server."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.FILE, payload)
        if not payload:
            logger.warning(
                "FILE command received with no payload (task_id=%s)", task_id
            )
            update_status(TaskStatus.COMPLETED)
            return

        client_id, files = payload if isinstance(payload, tuple) else (None, payload)

        for file in files:
            logger.info("Received file: %s", file)
            ok, uname, dest = _persist_uploaded_user_config(
                file,
                ctx.configure_file,
                ctx.user_config_map,
                expected_username=client_id,
            )
            if ok:
                logger.info("User config persisted for '%s' -> %s", uname, dest)
            else:
                logger.info(
                    "File '%s' not recognised as valid user config or identity mismatch; left in place.",
                    file,
                )

        update_status(TaskStatus.COMPLETED)
