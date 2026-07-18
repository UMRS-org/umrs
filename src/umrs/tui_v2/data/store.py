"""Snapshot store for the UMRS monitor v2.

Stores per-tab data fetched asynchronously. Thread-safe via a simple
lock so concurrent fetches don't corrupt state.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TabData:
    """Data for a single tab."""
    name: str
    data: Any = None
    fetched_at: float = 0.0
    error: str | None = None


class MonitorStore:
    """Per-tab data store for the monitor."""

    def __init__(self) -> None:
        self._tabs: dict[str, TabData] = {}
        self._lock = asyncio.Lock()

    def tab(self, name: str) -> TabData:
        """Get (or create) tab data by name."""
        if name not in self._tabs:
            self._tabs[name] = TabData(name=name)
        return self._tabs[name]

    async def set(
        self,
        name: str,
        data: Any,
        error: str | None = None,
    ) -> None:
        """Set tab data (thread-safe)."""
        async with self._lock:
            td = self.tab(name)
            td.data = data
            td.error = error
            import time as _time
            td.fetched_at = _time.monotonic()

    async def get(self, name: str) -> Any:
        """Get tab data (thread-safe)."""
        async with self._lock:
            td = self.tab(name)
            return td.data

    async def clear(self, name: str) -> None:
        """Clear tab data."""
        async with self._lock:
            if name in self._tabs:
                self._tabs[name].data = None

    def list_tabs(self) -> list[str]:
        return list(self._tabs.keys())

    def has_data(self, name: str) -> bool:
        return name in self._tabs and self._tabs[name].data is not None
