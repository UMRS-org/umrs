"""Tests for pane rendering with mock data."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from umrs.tui.panes.jobs import JobsPane
from umrs.tui.panes.endpoints import EndpointsPane
from umrs.tui.panes.models import ModelsPane
from umrs.tui.panes.workflows import WorkflowsPane


def _make_pane(pane_cls):
    from umrs.tui.app import MonitorApp
    app = MonitorApp(control_mq=MagicMock(), client_id="test", interval=100.0)
    return app, pane_cls()


@pytest.mark.asyncio
async def test_jobs_pane_renders():
    app, pane = _make_pane(JobsPane)
    async with app.run_test():
        app.mount(pane)
        data = {"jobs": [{"job_id": "j1234567890", "status": "running", "job_type": "index_build", "user_id": "u1", "attempts": [], "created_at": "2026-06-27T10:00:00", "updated_at": "2026-06-27T10:01:00"}]}
        pane.update_data(data)
        assert pane._jobs[0]["job_id"] == "j1234567890"


@pytest.mark.asyncio
async def test_endpoints_pane_renders():
    app, pane = _make_pane(EndpointsPane)
    async with app.run_test():
        app.mount(pane)
        data = {"endpoints": [{"job_id": "e1", "status": "ready", "spec": {"model": "qwen"}, "created_at": "2026-06-27T10:00:00"}]}
        pane.update_data(data)
        assert len(pane._endpoints) == 1


@pytest.mark.asyncio
async def test_models_pane_renders():
    app, pane = _make_pane(ModelsPane)
    async with app.run_test():
        app.mount(pane)
        data = {"models": [{"name": "qwen", "model_type": "llm", "scope": "user"}]}
        pane.update_data(data)
        assert pane._models[0]["name"] == "qwen"


@pytest.mark.asyncio
async def test_workflows_pane_renders():
    app, pane = _make_pane(WorkflowsPane)
    async with app.run_test():
        app.mount(pane)
        data = {"workflows": [{"name": "rag", "version": "1", "scope": "system", "steps": []}]}
        pane.update_data(data)
        assert pane._workflows[0]["name"] == "rag"
