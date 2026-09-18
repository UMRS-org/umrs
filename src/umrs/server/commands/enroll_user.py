"""Handler for the ENROLL_USER command (admin only).

An admin uploads one or more user JSON files to register new users.
For each file ``_persist_uploaded_user_config`` is called with
``allow_create=True`` and ``expected_username=None`` (the admin acts on
behalf of any user).

If at least one user was successfully enrolled, a hot-reload is triggered
so the new user is immediately provisioned in MinIO and PostgreSQL without
restarting the server.  The task is marked COMPLETED only when at least
one enrolment succeeded; it is marked ERROR otherwise.
"""

from __future__ import annotations

import logging
from typing import Any

from umrs.core.TaskCmdStatus import TaskCmd, TaskStatus
from umrs.server.config import ENABLE_PRETTY_PRINT
from umrs.server.file_handling import (
    persist_uploaded_user_config as _persist_uploaded_user_config,
)
from umrs.server.maintenance import filter_outcome, reload_configuration

from .base import CommandHandler, HandlerContext

logger = logging.getLogger(__name__)


class EnrollUserHandler(CommandHandler):
    """Handle ENROLL_USER — admin registers one or more new users."""

    def execute(self, payload: Any, ctx: HandlerContext) -> None:
        task_id, update_status = self._make_task(ctx, TaskCmd.ENROLL_USER, payload)
        if not payload:
            logger.warning(
                "ENROLL_USER command received with no payload (task_id=%s)", task_id
            )
            update_status(TaskStatus.ERROR)
            return

        client_id, files = payload if isinstance(payload, tuple) else (None, payload)

        enrolled_any = False
        for file in files:
            logger.info("Enrolling user from file: %s", file)
            ok, uname, dest = _persist_uploaded_user_config(
                file,
                ctx.configure_file,
                ctx.user_config_map,
                expected_username=None,  # admin acts on behalf of any user
                allow_create=True,
            )
            if ok:
                logger.info("User '%s' enrolled -> %s", uname, dest)
                enrolled_any = True
            else:
                logger.warning("Failed to enroll user from '%s'", file)

        if enrolled_any:
            # Hot-reload so the new user is provisioned in MinIO + PostgreSQL
            # without restarting the server.
            outcome = reload_configuration(
                ctx.configure_file,
                ctx.buckets_cfg,
                ctx.user_config_map,
                log_output=ENABLE_PRETTY_PRINT,
            )
            filter_outcome(
                ctx.task_manager.active_threads,
                outcome,
                ctx.buckets_cfg,
                "enroll_user",
            )
            if "error" in outcome:
                logger.warning(
                    "Reload after enrollment encountered errors: %s", outcome["error"]
                )

        update_status(TaskStatus.COMPLETED if enrolled_any else TaskStatus.ERROR)
