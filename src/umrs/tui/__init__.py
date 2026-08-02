"""Textual-based TUI dashboard for UMRS (v1, RETIRED).

Deprecated since 2026-08-02 (U5 — umrs-backbone-proposals): all unique
widgets and the tab-export helper now live in :mod:`umrs.tui_v2`.

The package remains importable for its test coverage and the
:class:`Dashboard` compatibility shim (``client_umrs`` bootstrap path)
until the cleanup cycle removes it. New TUI features must land only in
:mod:`umrs.tui_v2`, never here.

The monitor is a decomposed Textual widget app replacing the legacy
monolithic Rich ``Live`` dashboard.  See :mod:`umrs.tui.app` for the main
:class:`MonitorApp` entry point and :mod:`umrs.tui.dashboard` for the
:class:`Dashboard` compatibility shim that preserves the constructor
signature expected by :mod:`client_umrs` and
:mod:`umrs.tests.test_client_monitor_bootstrap`.
"""

# ruff: noqa: E402
import warnings

warnings.warn(
    "umrs.tui is retired (U5): widgets and export helpers now live in "
    "umrs.tui_v2. This package remains only as the compatibility shim "
    "until the cleanup cycle removes it.",
    DeprecationWarning,
    stacklevel=2,
)

from .app import MonitorApp
from .dashboard import Dashboard

__all__ = ["MonitorApp", "Dashboard"]
