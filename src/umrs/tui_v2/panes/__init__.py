"""Panels for monitor v2."""

from .artifacts import ArtifactsPane
from .jobs import JobsPane
from .endpoints import EndpointsPane
from .status import StatusPane
from .models import ModelsPane
from .workflows import WorkflowsPaneV2 as WorkflowsPane
from .connection import ConnectionPane
from .logs import LogsPane
from .errors import ErrorsPane
from .workflow_logs import WorkflowLogsPane
from .data_transfer import DataTransferPane
from .jobs_submit import JobsSubmitPane
from .config_editor import ConfigEditorPane

__all__ = [
    "ArtifactsPane",
    "JobsPane",
    "EndpointsPane",
    "StatusPane",
    "ModelsPane",
    "WorkflowsPane",
    "ConnectionPane",
    "LogsPane",
    "ErrorsPane",
    "WorkflowLogsPane",
    "DataTransferPane",
    "JobsSubmitPane",
    "ConfigEditorPane",
]
