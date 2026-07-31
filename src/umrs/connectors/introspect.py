"""Connector introspection — static inventory of intake connectors (U3).

Single source of truth for "which connectors does this UMRS build ship,
and are they importable right now". Consumed by the dashboard overview
endpoint and by ``r4 data intake``.

Rich sync state (cursors, per-source health, DLQ depth) lands with P2
intake hardening; this module deliberately reports *presence and
loadability* only, so dashboards always have honest data to render.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConnectorInfo:
    """Introspected view of one connector module."""

    name: str
    module: str
    class_name: str
    available: bool
    error: str | None = None

    def to_dict(self) -> dict:
        out = {
            "name": self.name,
            "class": self.class_name,
            "available": self.available,
        }
        if self.error:
            out["error"] = self.error
        return out


# name → (module, expected class)
_REGISTRY: dict[str, tuple[str, str]] = {
    "confluence": ("umrs.connectors.confluence", "ConfluenceConnector"),
    "federated": ("umrs.connectors.federated", "FederatedConnector"),
    "filesystem": ("umrs.connectors.filesystem", "FilesystemConnector"),
    "gdrive": ("umrs.connectors.gdrive", "GoogleDriveConnector"),
    "github": ("umrs.connectors.github", "GitHubConnector"),
    "notion": ("umrs.connectors.notion", "NotionConnector"),
    "slack": ("umrs.connectors.slack", "SlackConnector"),
    "web_crawl": ("umrs.connectors.web_crawl", "WebCrawlConnector"),
}

CONNECTOR_NAMES: tuple[str, ...] = tuple(sorted(_REGISTRY))


def probe_connector(name: str) -> ConnectorInfo:
    """Import one connector module and verify its class resolves."""
    module_name, class_name = _REGISTRY[name]
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 — availability probe, not failure
        return ConnectorInfo(
            name=name,
            module=module_name,
            class_name=class_name,
            available=False,
            error=f"import: {exc.__class__.__name__}: {exc}",
        )
    if not hasattr(module, class_name):
        return ConnectorInfo(
            name=name,
            module=module_name,
            class_name=class_name,
            available=False,
            error=f"class {class_name} missing in {module_name}",
        )
    return ConnectorInfo(
        name=name, module=module_name, class_name=class_name, available=True
    )


@dataclass
class ConnectorReport:
    """Inventory result across all registered connectors."""

    connectors: list[ConnectorInfo] = field(default_factory=list)

    @property
    def available(self) -> list[str]:
        return sorted(c.name for c in self.connectors if c.available)

    @property
    def unavailable(self) -> list[str]:
        return sorted(c.name for c in self.connectors if not c.available)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "unavailable": self.unavailable,
            "connectors": [c.to_dict() for c in self.connectors],
        }


def inventory() -> ConnectorReport:
    """Probe every registered connector."""
    return ConnectorReport(
        connectors=[probe_connector(name) for name in CONNECTOR_NAMES]
    )
