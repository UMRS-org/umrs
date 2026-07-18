"""UMRS Monitor TUI v2 — interactive monitoring and actions.

Replaces the read-only v1 with a full-featured interactive monitor that
supports:
- All v1 read tabs (jobs, endpoints, status, models, workflows, connection, logs, errors, wf-logs)
- New action tabs (data-transfer, jobs-submit, config-editor)
- Inline actions on tables (run, stop, cancel, upload, download, submit)
- Confirmation screens for write operations
- Progress indicators for long-running operations
- Multi-tab refresh (all or active)
- Saved actions history

Entry point: ``MonitorAppV2`` (use via DashboardV2.run())
"""

from __future__ import annotations

from .app import MonitorAppV2
from .dashboard import DashboardV2

__all__ = ["MonitorAppV2", "DashboardV2"]
