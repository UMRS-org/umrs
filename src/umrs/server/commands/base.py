"""Base classes for the UMRS command-handler pattern.

Each TaskCmd maps to one CommandHandler subclass.  The handler receives
the raw ZeroMQ payload and a HandlerContext that carries every server-level
dependency (task_manager, control_mq, server_cfg, …).

Each handler creates its own task_id(s) via ``CommandHandler._make_task``,
which registers the task and returns the (task_id, update_status) pair.
This allows batch handlers to create N tasks without special-casing in the
event loop.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HandlerContext:
    """All server-level dependencies passed to every CommandHandler."""

    task_manager: Any
    control_mq: Any
    server_cfg: Any
    device: str
    embedding_backend: Any
    configure_file: str
    job_command_backend: Any = field(default=None)
    model_command_backend: Any = field(default=None)
    # True only when a durable job *scheduler* is actively running to execute
    # queued jobs. When False (e.g. front-end deployments with no in-container
    # orchestrator backend), workflow runs must execute in-process instead of
    # being queued as durable jobs that nothing would ever pick up.
    workflow_scheduler_active: bool = field(default=False)

    # Convenience accessors ------------------------------------------------
    @property
    def buckets_cfg(self):
        return self.server_cfg.buckets_cfg

    @property
    def user_config_map(self):
        return self.server_cfg.user_config_map


# Type alias for the status-update callback.
StatusCallback = Callable[[str], None]


def make_status_callback(task_manager: Any, task_id: int) -> StatusCallback:
    """Return a callback that updates *task_id*'s status in *task_manager*."""
    def callback(status: str) -> None:
        task_manager.tasks_status.update_status(task_id, status)
    return callback


class CommandHandler(ABC):
    """Abstract base for a single TaskCmd handler.

    Subclasses implement ``execute(payload, ctx)``.  Authorization has already
    been checked by the CommandRegistry before ``execute()`` is called.
    Each handler creates its own task_id(s) via ``_make_task``.
    """

    @staticmethod
    def _make_task(
        ctx: HandlerContext,
        cmd: str,
        payload: Any,
    ) -> tuple[int, StatusCallback]:
        """Register a new task and return ``(task_id, update_status)``."""
        task_id = ctx.task_manager.tasks_status.create_task(cmd, payload)
        return task_id, make_status_callback(ctx.task_manager, task_id)

    @abstractmethod
    def execute(
        self,
        payload: Any,
        ctx: HandlerContext,
    ) -> None:
        """Execute the command.

        Parameters
        ----------
        payload:
            Raw payload extracted from the ZeroMQ message.
        ctx:
            Server-level dependencies, including ``task_manager``.
            Call ``self._make_task(ctx, TaskCmd.XXX, payload)`` to create the
            task and obtain an ``update_status`` callback.
        """
