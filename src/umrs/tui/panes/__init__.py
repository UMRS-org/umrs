"""Per-tab panes for the UMRS monitor TUI."""

from .jobs import JobsPane
from .endpoints import EndpointsPane
from .status import StatusPane
from .models import ModelsPane
from .workflows import WorkflowsPane
from .connection import ConnectionPane
from .logs import LogsPane
from .errors import ErrorsPane
from .workflow_logs import WorkflowLogsPane

__all__ = [
    "JobsPane",
    "EndpointsPane",
    "StatusPane",
    "ModelsPane",
    "WorkflowsPane",
    "ConnectionPane",
    "LogsPane",
    "ErrorsPane",
    "WorkflowLogsPane",
]
