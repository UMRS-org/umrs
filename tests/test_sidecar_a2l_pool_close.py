"""The sidecar lifespan closes the a2l recorder's asyncpg pool on shutdown.

Nothing closed it before 2026-11-24: every app lifespan (each
``TestClient(app)`` in the suite) left its connections open for the life of
the process, part of what ran the UMRS-org CI Postgres out of slots
(``too many clients``, run 45171800447).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from umrs.a2l import recorder as a2l_recorder
from umrs.server.http_sidecar import _close_a2l_recorder


class _Pool:
    def __init__(self, *, close_fails: bool = False) -> None:
        self.closed = self.terminated = False
        self._close_fails = close_fails

    async def close(self) -> None:
        if self._close_fails:
            raise RuntimeError("connection busy")
        self.closed = True

    def terminate(self) -> None:
        self.terminated = True


@pytest.fixture
def restore_recorder():
    saved = a2l_recorder.get_recorder()
    yield
    a2l_recorder.configure_recorder(saved)


def _app(pool):
    return SimpleNamespace(state=SimpleNamespace(umrs_knowledge_asyncpg_pool=pool))


def test_shutdown_closes_the_pool_and_resets_the_recorder(restore_recorder):
    pool = _Pool()
    app = _app(pool)
    a2l_recorder.configure_recorder(object())  # stands in for StoreA2lRecorder

    asyncio.run(_close_a2l_recorder(app))

    assert pool.closed and not pool.terminated
    assert app.state.umrs_knowledge_asyncpg_pool is None
    assert isinstance(a2l_recorder.get_recorder(), a2l_recorder.NullA2lRecorder)


def test_a_failing_close_terminates_instead(restore_recorder):
    pool = _Pool(close_fails=True)

    asyncio.run(_close_a2l_recorder(_app(pool)))

    assert pool.terminated


def test_no_pool_is_a_no_op(restore_recorder):
    marker = object()
    a2l_recorder.configure_recorder(marker)

    asyncio.run(_close_a2l_recorder(_app(None)))

    assert a2l_recorder.get_recorder() is marker
