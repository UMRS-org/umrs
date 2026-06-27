"""Snapshot store for the UMRS monitor.

Each tab has its own cached snapshot plus metadata (last refresh, last
error, dirty flag).  The store is a plain data container; refresh logic
lives in :mod:`umrs.tui.data.fetchers`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TabSnapshot:
    """Cached data + metadata for one tab."""

    data: Any = None
    last_refresh: float = 0.0
    last_error: str | None = None
    fetched: bool = False

    def mark(self, data: Any, *, error: str | None = None) -> None:
        self.data = data
        self.last_refresh = time.monotonic()
        self.last_error = error
        self.fetched = True


@dataclass
class MonitorStore:
    """Per-tab snapshot store.

    The store does not perform any I/O — it only holds the most recent
    snapshot for each tab so the UI can render without blocking on a
    fetch.  Refresh workers call :meth:`update` when new data arrives.
    """

    jobs: TabSnapshot = field(default_factory=TabSnapshot)
    endpoints: TabSnapshot = field(default_factory=TabSnapshot)
    status: TabSnapshot = field(default_factory=TabSnapshot)
    models: TabSnapshot = field(default_factory=TabSnapshot)
    workflows: TabSnapshot = field(default_factory=TabSnapshot)
    connection: TabSnapshot = field(default_factory=TabSnapshot)
    logs: TabSnapshot = field(default_factory=TabSnapshot)

    def tab(self, name: str) -> TabSnapshot:
        return getattr(self, name)

    def update(self, tab: str, data: Any, *, error: str | None = None) -> None:
        self.tab(tab).mark(data, error=error)

    def is_fetched(self, tab: str) -> bool:
        return self.tab(tab).fetched

    def last_error(self, tab: str) -> str | None:
        return self.tab(tab).last_error

    def last_refresh(self, tab: str) -> float:
        return self.tab(tab).last_refresh
