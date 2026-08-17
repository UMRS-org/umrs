"""ADR-027 P1: ZMQ UPLOAD_WORKFLOW deprecated — fails fast with guidance."""

from __future__ import annotations

from unittest.mock import MagicMock

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.server.commands.upload_workflow import UploadWorkflowHandler


def test_upload_workflow_fails_fast_with_deprecation_guidance(caplog):
    statuses: list[TaskStatus] = []
    handler = UploadWorkflowHandler()
    payload = ("alice", ["/tmp/wf.zip"], "demo-wf")
    with caplog.at_level("WARNING", logger="umrs.server.commands.upload_workflow"):
        handler.execute(
            payload,
            statuses.append,
            MagicMock(name="ctx"),
            task_id=42,
        )
    assert statuses == [TaskStatus.ERROR]
    assert any(
        "ADR-027" in r.getMessage() and "/gw/workflows/upload" in r.getMessage()
        for r in caplog.records
    )


def test_handler_module_documents_single_enforcement_point():
    from umrs.server.commands import upload_workflow

    assert "single enforcement point" in upload_workflow.__doc__
