"""Textual-based TUI dashboard for UMRS.

The monitor is a decomposed Textual widget app replacing the legacy
monolithic Rich ``Live`` dashboard.  See :mod:`umrs.tui.app` for the main
:class:`MonitorApp` entry point and :mod:`umrs.tui.dashboard` for the
:class:`Dashboard` compatibility shim that preserves the constructor
signature expected by :mod:`client_umrs` and
:mod:`umrs.tests.test_client_monitor_bootstrap`.
"""

from .app import MonitorApp
from .dashboard import Dashboard

__all__ = ["MonitorApp", "Dashboard"]
