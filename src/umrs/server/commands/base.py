"""Base classes for the UMRS command-handler pattern.

Each TaskCmd maps to one CommandHandler subclass.  The handler receives
the raw ZeroMQ payload, a status-update callback, a HandlerContext that
carries every server-level dependency (task_manager, control_mq,
server_cfg, …), and the ``task_id`` created by the event loop.

``task_id`` is passed to every handler — even those that do not launch a
background thread — so that all commands can be tracked uniformly.  This
makes it straightforward to add per-command observability or audit in the
future without changing the handler contract.
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


# Type alias for the status-update callback injected by the event loop.
StatusCallback = Callable[[str], None]


class CommandHandler(ABC):
    """Abstract base for a single TaskCmd handler.

    Subclasses implement ``execute()``.  Authorization has already been
    checked by the CommandRegistry before ``execute()`` is called.
    """

    @abstractmethod
    def execute(
        self,
        payload: Any,
        update_status: StatusCallback,
        ctx: HandlerContext,
        task_id: int,
    ) -> None:
        """Execute the command.

        Parameters
        ----------
        payload:
            Raw payload extracted from the ZeroMQ message.
        update_status:
            Callback to update the task status in the TaskManager.
            Call with a ``TaskStatus.*`` constant.
        ctx:
            Server-level dependencies.
        task_id:
            Identifier created by the event loop for this command
            invocation.  Always provided — use it to associate a
            background thread (``ctx.task_manager.associate_thread_with_task``)
            or for logging/observability even when no thread is launched.
        """
