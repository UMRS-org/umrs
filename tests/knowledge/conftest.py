"""Shared fixtures for knowledge tests.

Several knowledge test modules are synchronous and drive coroutines via
``asyncio.get_event_loop().run_until_complete``. pytest-asyncio (auto mode)
closes/unsets the thread's event loop after async tests in other modules
run, which breaks ``asyncio.get_event_loop()`` when the full suite runs
together. Provide a fresh, current event loop per test.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _fresh_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)
