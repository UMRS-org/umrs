"""Phase 2D regression tests: ad-hoc asyncpg pool lifecycle.

Live failure that motivated this: async DB operations used pools built by the
synchronous ``_get_pool()``. Those pools are never initialized — every fetch
raises ``InterfaceError: pool is not initialized`` which the collectors
silently swallowed. The fix creates/awaits/closes the pool inside ONE
coroutine via ``ka._open_pool``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from umrs.jobs import knowledge_adapters as ka

_DESCRIPTOR = {
    "artifact_id": "wfart-life",
    "content_hash": "aa11bb22cc33dd44",
    "kind": "s3_uri",
    "source_ref": "s3://r4-data/jobs/j-life/attempt_1/out.json",
}


class _AwaitablePool:
    """Models asyncpg.create_pool()'s awaitable: awaiting initializes."""

    def __init__(self, events: list[str], inner: object) -> None:
        self._events = events
        self._inner = inner

    def __await__(self):  # type: ignore[no-untyped-def]
        async def _init() -> object:
            self._events.append("create_awaited")
            return self._inner

        return _init().__await__()


class _RecordingPool:
    """Records fetch/close activity like an initialized asyncpg pool."""

    def __init__(self, events: list[str], row: object = None, boom: bool = False) -> None:
        self._events = events
        self._row = row
        self._boom = boom

    async def fetch(self, _query: str, *_args: object) -> list[dict]:
        self._events.append("fetch")
        return [{"attempt_id": "att-1"}]

    async def fetchval(self, _query: str, *_args: object) -> object:
        self._events.append("fetchval")
        if self._boom:
            raise RuntimeError("boom")
        return self._row

    async def close(self) -> None:
        self._events.append("close")

    async def wait_closed(self) -> None:
        self._events.append("wait_closed")


def _attach_fake_asyncpg(monkeypatch: pytest.MonkeyPatch, events: list[str], inner: object) -> None:
    import asyncpg

    def _factory(*, dsn: str = "", **_kw: object) -> _AwaitablePool:
        events.append("create_called")
        return _AwaitablePool(events, inner)

    monkeypatch.setattr(asyncpg, "create_pool", _factory)


def _job() -> SimpleNamespace:
    return SimpleNamespace(job_id="j-life", project_id="proj-life", commit_attempt_id="att-1")


# ── Test A: pool is created AND awaited inside the coroutine ─────────────────


def test_pool_created_inside_coroutine_and_awaited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    events: list[str] = []
    _attach_fake_asyncpg(monkeypatch, events, _RecordingPool(events, {}))

    ka._collect_artifact_ids(_job())

    assert "create_called" in events
    assert "create_awaited" in events
    # Regression window: the pool must be initialized before any DB op.
    assert events.index("create_awaited") < events.index("fetchval")


# ── Test B: initialized pool actually performs the fetch ─────────────────────


def test_initialized_pool_reaches_fetchval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    events: list[str] = []
    _attach_fake_asyncpg(
        monkeypatch, events, _RecordingPool(events, {"artifacts": [dict(_DESCRIPTOR)]})
    )

    result = ka._collect_artifact_ids(_job())

    assert "fetchval" in events, "collector never reached the DB fetch"
    assert result and result[0]["artifact_id"] == "wfart-life"


# ── Test C: pool closes on success AND on exception ──────────────────────────


def test_pool_closes_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    events: list[str] = []
    _attach_fake_asyncpg(monkeypatch, events, _RecordingPool(events, {}))

    ka._collect_artifact_ids(_job())
    assert "close" in events
    assert "wait_closed" in events


def test_pool_closes_on_fetch_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    events: list[str] = []
    _attach_fake_asyncpg(monkeypatch, events, _RecordingPool(events, boom=True))

    ka._collect_artifact_ids(_job())  # failure must not leak the pool either
    assert "close" in events
    assert "wait_closed" in events


# ── Test D: live JSONB shape through the initialized pool ────────────────────


def test_jsonb_string_through_initialized_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DB_HOST", "127.0.0.1")
    events: list[str] = []
    row = json.dumps({"artifacts": [dict(_DESCRIPTOR)]})
    _attach_fake_asyncpg(monkeypatch, events, _RecordingPool(events, row))

    result = ka._collect_artifact_ids(_job())

    assert result and result[0]["content_hash"] == _DESCRIPTOR["content_hash"]
    assert result[0]["kind"] == "s3_uri"
