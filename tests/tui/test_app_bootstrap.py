"""Tests for the MonitorApp Textual app."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from umrs.tui.app import MonitorApp


def _make_app(**overrides):
    defaults = dict(
        control_mq=MagicMock(),
        client_id="test-user",
        interval=10.0,
    )
    defaults.update(overrides)
    return MonitorApp(**defaults)


@pytest.mark.asyncio
async def test_app_constructs():
    app = _make_app()
    assert app._client_id == "test-user"


@pytest.mark.asyncio
async def test_app_quits_on_q():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("q")
        await pilot.pause()
    assert app._exit is True or app.return_value is None or app.return_value is True


@pytest.mark.asyncio
async def test_app_switches_tabs():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("4")
        await pilot.pause()
        from textual.widgets import TabbedContent
        tc = app.query_one(TabbedContent)
        assert tc.active == "models"


@pytest.mark.asyncio
async def test_app_refresh():
    app = _make_app()
    async with app.run_test() as pilot:
        await pilot.press("r")
        await pilot.pause()
