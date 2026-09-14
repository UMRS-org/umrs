"""Tests for the async data fetchers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from umrs.tui.data.fetchers import Fetchers
from umrs.tui.data.store import MonitorStore


def _make_store():
    return MonitorStore()


def _make_fetchers(store, mq=None):
    mq = mq or MagicMock()
    api = MagicMock()
    return Fetchers(store=store, control_api=api, control_mq=mq, client_id="test")


@pytest.mark.asyncio
async def test_fetch_jobs_success():
    store = _make_store()
    fetchers = _make_fetchers(store)
    fetchers._api.list_jobs.return_value = {"status": "ok", "jobs": [{"job_id": "j1"}]}
    await fetchers.fetch_jobs()
    snapshot = store.tab("jobs")
    assert snapshot.fetched
    assert snapshot.data["jobs"][0]["job_id"] == "j1"
    assert snapshot.last_error is None


@pytest.mark.asyncio
async def test_fetch_jobs_error_surfaces():
    store = _make_store()
    fetchers = _make_fetchers(store)
    fetchers._api.list_jobs.side_effect = RuntimeError("boom")
    await fetchers.fetch_jobs()
    snapshot = store.tab("jobs")
    assert snapshot.last_error == "boom"
    assert snapshot.data is None


@pytest.mark.asyncio
async def test_fetch_connection_uses_mq():
    store = _make_store()
    mq = MagicMock()
    mq.is_server_ready.return_value = True
    fetchers = _make_fetchers(store, mq=mq)
    await fetchers.fetch_connection()
    snapshot = store.tab("connection")
    assert snapshot.data["ping_ok"] is True
    assert snapshot.data["ping_ms"] >= 0


@pytest.mark.asyncio
async def test_fetch_status_propagates():
    store = _make_store()
    fetchers = _make_fetchers(store)
    fetchers._api.request_status.return_value = {"client": "c1", "tasks": []}
    await fetchers.fetch_status()
    assert store.tab("status").data["client"] == "c1"


@pytest.mark.asyncio
async def test_fetch_models_propagates():
    store = _make_store()
    fetchers = _make_fetchers(store)
    fetchers._api.list_models.return_value = {"models": [{"name": "m1"}]}
    await fetchers.fetch_models()
    assert store.tab("models").data["models"][0]["name"] == "m1"


@pytest.mark.asyncio
async def test_fetch_workflows_propagates():
    store = _make_store()
    fetchers = _make_fetchers(store)
    fetchers._api.list_workflows.return_value = {"workflows": [{"name": "w1"}]}
    await fetchers.fetch_workflows()
    assert store.tab("workflows").data["workflows"][0]["name"] == "w1"


@pytest.mark.asyncio
async def test_fetch_endpoints_propagates():
    store = _make_store()
    fetchers = _make_fetchers(store)
    fetchers._api.list_endpoints.return_value = {"endpoints": [{"job_id": "e1"}]}
    await fetchers.fetch_endpoints()
    assert store.tab("endpoints").data["endpoints"][0]["job_id"] == "e1"
