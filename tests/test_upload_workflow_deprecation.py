"""ADR-027 P1: ZMQ UPLOAD_WORKFLOW deprecated — fails fast with guidance."""

from __future__ import annotations

from types import SimpleNamespace

from umrs.core.TaskCmdStatus import TaskStatus
from umrs.server.commands.upload_workflow import UploadWorkflowHandler


class _TasksStatus:
    """Minimal tasks_status stand-in for the _make_task contract."""

    def __init__(self) -> None:
        self.updated: list[tuple[int, TaskStatus]] = []

    def create_task(self, cmd, payload):
        return 42

    def update_status(self, task_id, status):
        self.updated.append((task_id, status))


def test_upload_workflow_fails_fast_with_deprecation_guidance(caplog):
    tasks_status = _TasksStatus()
    ctx = SimpleNamespace(task_manager=SimpleNamespace(tasks_status=tasks_status))
    handler = UploadWorkflowHandler()
    payload = ("alice", ["/tmp/wf.zip"], "demo-wf")
    with caplog.at_level("WARNING", logger="umrs.server.commands.upload_workflow"):
        # Handler contract since the ejeannot batch sync (upstream 211790ab):
        # execute(payload, ctx) and the handler owns its task lifecycle.
        handler.execute(payload, ctx)
    assert tasks_status.updated == [(42, TaskStatus.ERROR)]
    assert any(
        "ADR-027" in r.getMessage() and "/gw/workflows/upload" in r.getMessage()
        for r in caplog.records
    )


def test_handler_module_documents_single_enforcement_point():
    from umrs.server.commands import upload_workflow

    assert "single enforcement point" in upload_workflow.__doc__
