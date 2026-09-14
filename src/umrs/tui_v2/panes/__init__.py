"""Panels for monitor v2."""

from .artifacts import ArtifactsPane
from .config_editor import ConfigEditorPane
from .connection import ConnectionPane
from .data_transfer import DataTransferPane
from .endpoints import EndpointsPane
from .errors import ErrorsPane
from .jobs import JobsPane
from .jobs_submit import JobsSubmitPane
from .logs import LogsPane
from .models import ModelsPane
from .status import StatusPane
from .workflow_logs import WorkflowLogsPane
from .workflows import WorkflowsPaneV2 as WorkflowsPane

__all__ = [
    "ArtifactsPane",
    "ConfigEditorPane",
    "ConnectionPane",
    "DataTransferPane",
    "EndpointsPane",
    "ErrorsPane",
    "JobsPane",
    "JobsSubmitPane",
    "LogsPane",
    "ModelsPane",
    "StatusPane",
    "WorkflowLogsPane",
    "WorkflowsPane",
]
